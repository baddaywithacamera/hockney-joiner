"""
placement.py — Image placement engine.

Stage 1 (grid):      Fallback when LightGlue unavailable or overlap insufficient.
Stage 2 (LightGlue): DISK feature extraction + LightGlue matching → canvas positions.

The interface is stable — PlacementWorker always emits a PlacementResult
with a list of ImagePlacement objects. TrayView doesn't care how they were computed.

LightGlue strategy:
  1. Extract DISK features from every thumbnail (300px — fast, consistent).
  2. Run LightGlue matcher on all pairs. Skip pairs with <MIN_MATCHES inliers.
  3. From matched pairs, build a graph of pairwise similarity transforms
     (translation + rotation only — no scaling, geometry must be honest).
  4. Anchor one image at (0,0). Chain transforms outward via BFS to place all others.
  5. Pairs with insufficient overlap fall back to grid offset from their nearest placed neighbour.

The misalignments that remain after placement are not failures.
They are the Hockney in the machine.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass

from PyQt6.QtCore import QThread, pyqtSignal

from hockney.core.image_store import ImageStore
from hockney.core.models import ImagePlacement, ProjectConfig

log = logging.getLogger(__name__)

THUMB_LONG_EDGE = 300
PREVIEW_LONG_EDGE = 1500
GRID_PADDING = 24
MIN_MATCHES = 8         # minimum RANSAC inlier pairs to trust a placement
MIN_MATCHES_RELAXED = 5 # second-pass threshold for stragglers
MAX_KEYPOINTS = 2048    # per image, for DISK
RANSAC_REPROJ = 5.0     # RANSAC reprojection threshold (px) for reference matching
OVERLAP_REPEL = 0.25    # fraction of tile size to nudge overlapping tiles apart
                        # (lower = tighter shingling, more Hockney-like)
OVERLAP_THRESHOLD = 0.6   # centres closer than this fraction of tile_size → overlapping

# Subject-type tuning overrides
SUBJECT_TUNING = {
    "landscape":  {"max_keypoints": 2048, "min_matches": 8},
    "skylife":    {"max_keypoints": 1024, "min_matches": 6},
    "urban":      {"max_keypoints": 2048, "min_matches": 10},
    "indoor":     {"max_keypoints": 2048, "min_matches": 6},
    "people":     {"max_keypoints": 2048, "min_matches": 8},
    "object":     {"max_keypoints": 2048, "min_matches": 6},
    "vehicle":    {"max_keypoints": 2048, "min_matches": 8},
    "building":   {"max_keypoints": 2048, "min_matches": 10},
    "person":     {"max_keypoints": 2048, "min_matches": 8},
}


def _homography_center(inlier_det, inlier_ref, dw, dh, ref_w, ref_h,
                       ransac_reproj=5.0, margin=0.2):
    """
    Map detail image center through homography to find where it lands on the reference.
    Returns (cx, cy) or None if the homography is degenerate or lands way outside bounds.

    `margin` allows placement up to margin*ref_size beyond the reference edge
    (photos near edges legitimately hang over).
    """
    import cv2
    import numpy as np

    if len(inlier_det) < 4:
        return None

    H, _ = cv2.findHomography(
        inlier_det.reshape(-1, 1, 2).astype(np.float32),
        inlier_ref.reshape(-1, 1, 2).astype(np.float32),
        cv2.RANSAC, ransac_reproj,
    )
    if H is not None:
        det_center = np.array([[[dw / 2.0, dh / 2.0]]], dtype=np.float32)
        ref_point = cv2.perspectiveTransform(det_center, H)
        cx = float(ref_point[0, 0, 0])
        cy = float(ref_point[0, 0, 1])

        # Sanity check — reject if way outside reference bounds
        margin_x = ref_w * margin
        margin_y = ref_h * margin
        if (cx < -margin_x or cx > ref_w + margin_x or
                cy < -margin_y or cy > ref_h + margin_y):
            log.debug("Homography projected to (%.0f, %.0f) — outside ref %dx%d, rejecting",
                      cx, cy, ref_w, ref_h)
            # Fall back to inlier centroid
            cx = float(inlier_ref[:, 0].mean())
            cy = float(inlier_ref[:, 1].mean())
        return (cx, cy)
    else:
        # Fallback to inlier centroid
        cx = float(inlier_ref[:, 0].mean())
        cy = float(inlier_ref[:, 1].mean())
        return (cx, cy)


@dataclass
class PlacementResult:
    placements: list[ImagePlacement]
    used_lightglue: bool
    fallback_count: int
    message: str


class PlacementWorker(QThread):
    finished = pyqtSignal(object)   # PlacementResult
    progress = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, store: ImageStore, model_ready: bool, models_dir=None,
                 config: ProjectConfig | None = None):
        super().__init__()
        self.store = store
        self.model_ready = model_ready
        self.models_dir = models_dir
        self.config = config
        self._cancelled = False

        self._thumb_edge = store.thumb_long_edge  # tile size for grid layout

        # Apply subject-type tuning
        if config and config.subject_type in SUBJECT_TUNING:
            t = SUBJECT_TUNING[config.subject_type]
            self._max_kp = t["max_keypoints"]
            self._min_matches = t["min_matches"]
        else:
            self._max_kp = MAX_KEYPOINTS
            self._min_matches = MIN_MATCHES

    def cancel(self):
        self._cancelled = True

    def run(self):
        records = self.store.all_records()
        if not records:
            self.finished.emit(PlacementResult([], False, 0, "No images to place."))
            return

        # Determine which engine to use
        engine = "auto"
        if self.config:
            engine = getattr(self.config, "matching_engine", "auto") or "auto"

        has_refs = self.config and self.config.has_references()

        # Set up placement logger
        from hockney.core.placement_log import PlacementLog
        plog = PlacementLog(
            self.store.session.session_dir,
            project_name=self.config.project_name if self.config else "",
        )

        if has_refs:
            result = self._place_with_references_dispatched(records, engine, plog)
        elif self.model_ready:
            result = self._place_lightglue(records)
        else:
            result = self._place_grid(records)

        if not self._cancelled:
            self.finished.emit(result)

    # ── Engine dispatch ──────────────────────────────────────────────────────

    def _place_with_references_dispatched(self, records, engine: str,
                                           plog) -> PlacementResult:
        """
        Dispatch to the right engine for reference-based placement.
        "auto" tries DISK+LightGlue first; if placement rate < 60%,
        re-runs with SIFT.  Best result wins.
        """
        from hockney.core.placement_log import PlacementLog

        if engine == "sift":
            return self._place_with_sift(records, plog)
        elif engine == "disk_lightglue":
            if self.model_ready:
                return self._place_with_references(records, plog)
            else:
                log.warning("DISK+LightGlue requested but model not ready, using SIFT")
                return self._place_with_sift(records, plog)
        else:
            # "auto" — try DISK+LightGlue and SIFT, keep the better result
            disk_result = None

            if self.model_ready:
                disk_result = self._place_with_references(records, plog=None)
                disk_placed = len(records) - disk_result.fallback_count
                disk_rate = disk_placed / max(len(records), 1)
                log.info("DISK+LightGlue: %.0f%% placed (%d/%d)",
                         disk_rate * 100, disk_placed, len(records))

                if disk_rate >= 0.7:
                    plog.set_engine("disk_lightglue")
                    plog.set_counts(len(records), disk_placed,
                                    disk_result.fallback_count)
                    plog.set_notes(f"DISK accepted at {disk_rate*100:.0f}%")
                    plog.flush()
                    return disk_result

            # Try SIFT
            self.progress.emit(0)
            sift_result = self._place_with_sift(records, plog=None)
            sift_placed = len(records) - sift_result.fallback_count
            sift_rate = sift_placed / max(len(records), 1)
            log.info("SIFT: %.0f%% placed (%d/%d)",
                     sift_rate * 100, sift_placed, len(records))

            # Pick the winner
            if disk_result:
                disk_placed = len(records) - disk_result.fallback_count
                if disk_placed >= sift_placed:
                    plog.set_engine("disk_lightglue (auto-winner)")
                    plog.set_counts(len(records), disk_placed,
                                    disk_result.fallback_count)
                    plog.set_notes(f"DISK won: {disk_placed}/{len(records)} vs SIFT {sift_placed}/{len(records)}")
                    plog.flush()
                    return disk_result

            # SIFT won (or DISK wasn't available)
            plog.set_engine("sift (auto-winner)")
            plog.set_counts(len(records), sift_placed,
                            sift_result.fallback_count)
            plog.set_notes(f"SIFT won: {sift_placed}/{len(records)}")
            plog.flush()
            return sift_result

    def _place_with_sift(self, records, plog) -> PlacementResult:
        """Run the SIFT engine and wrap results into a PlacementResult."""
        from hockney.core.placement_sift import place_with_sift

        placements, odds_and_ends, placed_count, log_data = place_with_sift(
            store=self.store,
            config=self.config,
            records=records,
            thumb_edge=self._thumb_edge,
            min_matches=self._min_matches,
            min_matches_relaxed=MIN_MATCHES_RELAXED,
            progress_cb=lambda pct: self.progress.emit(pct),
            cancel_check=lambda: self._cancelled,
        )

        if self._cancelled:
            return self._place_grid(records)

        # Template matching fallback for remaining unplaced
        if odds_and_ends:
            placements, odds_and_ends, placed_count = self._template_fallback(
                records, placements, odds_and_ends, placed_count,
            )

        # Spread overlaps
        _spread_overlaps(placements, self._thumb_edge, OVERLAP_REPEL, OVERLAP_THRESHOLD)

        # Odds & Ends tray
        fallback_count = len(odds_and_ends)
        if odds_and_ends:
            placed_ys = [p.y for p in placements.values()] or [0.0]
            tray_y = max(placed_ys) + self._thumb_edge + GRID_PADDING * 4
            for i, (uid, gx, gy, gr) in enumerate(odds_and_ends):
                p = ImagePlacement(
                    image_id=uid,
                    x=float(i * (self._thumb_edge + GRID_PADDING)),
                    y=tray_y, rotation=0.0,
                    z_order=placed_count + i,
                    auto_x=gx, auto_y=gy, auto_rotation=gr,
                )
                placements[uid] = p

        self.progress.emit(100)

        result_list = list(placements.values())
        total = len(records)
        msg = f"SIFT placed {placed_count}/{total} images."
        if fallback_count:
            msg += f" {fallback_count} in Odds & Ends tray."

        # Log results
        if plog:
            plog.set_engine("sift")
            plog.set_counts(total, placed_count, fallback_count)
            plog.merge_image_stats(log_data)
            plog.flush()

        log.info(msg)
        return PlacementResult(
            placements=result_list,
            used_lightglue=False,
            fallback_count=fallback_count,
            message=msg,
        )

    def _place_with_loftr(self, records, plog) -> PlacementResult:
        """Run the LoFTR dense matching engine and wrap results into a PlacementResult."""
        try:
            from hockney.core.placement_loftr import place_with_loftr
        except ImportError as e:
            log.warning("LoFTR import failed (%s), falling back to SIFT", e)
            return self._place_with_sift(records, plog)

        placements, odds_and_ends, placed_count, log_data = place_with_loftr(
            store=self.store,
            config=self.config,
            records=records,
            thumb_edge=self._thumb_edge,
            min_matches=self._min_matches,
            min_matches_relaxed=MIN_MATCHES_RELAXED,
            progress_cb=lambda pct: self.progress.emit(pct),
            cancel_check=lambda: self._cancelled,
        )

        if self._cancelled:
            return self._place_grid(records)

        # Template matching fallback for remaining unplaced
        if odds_and_ends:
            placements, odds_and_ends, placed_count = self._template_fallback(
                records, placements, odds_and_ends, placed_count,
            )

        # Spread overlaps
        _spread_overlaps(placements, self._thumb_edge, OVERLAP_REPEL, OVERLAP_THRESHOLD)

        # Odds & Ends tray
        fallback_count = len(odds_and_ends)
        if odds_and_ends:
            placed_ys = [p.y for p in placements.values()] or [0.0]
            tray_y = max(placed_ys) + self._thumb_edge + GRID_PADDING * 4
            for i, (uid, gx, gy, gr) in enumerate(odds_and_ends):
                p = ImagePlacement(
                    image_id=uid,
                    x=float(i * (self._thumb_edge + GRID_PADDING)),
                    y=tray_y, rotation=0.0,
                    z_order=placed_count + i,
                    auto_x=gx, auto_y=gy, auto_rotation=gr,
                )
                placements[uid] = p

        self.progress.emit(100)

        result_list = list(placements.values())
        total = len(records)
        msg = f"LoFTR placed {placed_count}/{total} images."
        if fallback_count:
            msg += f" {fallback_count} in Odds & Ends tray."

        # Log results
        if plog:
            plog.set_engine("loftr")
            plog.set_counts(total, placed_count, fallback_count)
            plog.merge_image_stats(log_data)
            plog.flush()

        log.info(msg)
        return PlacementResult(
            placements=result_list,
            used_lightglue=False,
            fallback_count=fallback_count,
            message=msg,
        )

    # ── Grid placement ─────────────────────────────────────────────────────────

    def _place_grid(self, records) -> PlacementResult:
        n = len(records)
        cols = max(1, math.isqrt(n))
        placements = []

        for i, record in enumerate(records):
            col = i % cols
            row = i // cols
            aspect = record.width / max(record.height, 1)
            th = int(self._thumb_edge / aspect) if aspect >= 1 else self._thumb_edge
            x = float(col * (self._thumb_edge + GRID_PADDING))
            y = float(row * (th + GRID_PADDING))
            placements.append(ImagePlacement(
                image_id=record.id,
                x=x, y=y, rotation=0.0, z_order=i,
                auto_x=x, auto_y=y, auto_rotation=0.0,
            ))
            self.progress.emit(int((i + 1) / n * 100))

        log.info("Grid fallback: %d images, %d columns", n, cols)
        return PlacementResult(
            placements=placements,
            used_lightglue=False,
            fallback_count=n,
            message=f"Grid layout ({n} images). LightGlue model not available.",
        )

    # ── LightGlue DISK placement ───────────────────────────────────────────────

    def _place_lightglue(self, records) -> PlacementResult:
        try:
            import torch
            from lightglue import LightGlue, DISK
            from lightglue.utils import rbd
            import numpy as np
            import cv2
        except ImportError as e:
            log.warning("LightGlue import failed (%s) — falling back to grid", e)
            r = self._place_grid(records)
            r.message = f"LightGlue import failed: {e}. Grid layout applied."
            return r

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("LightGlue placement on device: %s", device)

        # ── 1. Extract DISK features from all thumbnails ───────────────────────
        extractor = DISK(max_num_keypoints=self._max_kp).eval().to(device)
        matcher = LightGlue(features="disk").eval().to(device)

        n = len(records)
        features = {}   # record.id → lightglue feature dict

        for i, record in enumerate(records):
            if self._cancelled:
                return self._place_grid(records)
            arr = self.store.get_thumbnail(record.id)   # 300px numpy HxWx3
            if arr is None:
                continue
            tensor = _arr_to_tensor(arr, device)
            with torch.no_grad():
                feats = extractor.extract(tensor)
            features[record.id] = feats
            self.progress.emit(int((i + 1) / n * 40))

        # ── 2. Match all pairs, collect pairwise transforms ────────────────────
        ids = [r.id for r in records if r.id in features]
        n_ids = len(ids)
        transforms = {}   # (id_a, id_b) → (tx, ty, rot_deg) or None

        pair_count = n_ids * (n_ids - 1) // 2
        pair_idx = 0

        for i in range(n_ids):
            if self._cancelled:
                return self._place_grid(records)
            for j in range(i + 1, n_ids):
                id_a, id_b = ids[i], ids[j]
                try:
                    result = matcher({
                        "image0": features[id_a],
                        "image1": features[id_b],
                    })
                    matches = result["matches"][0]   # [N, 2] indices
                    kp_a = features[id_a]["keypoints"][0].cpu().numpy()
                    kp_b = features[id_b]["keypoints"][0].cpu().numpy()
                    m_kp_a = kp_a[matches[:, 0].cpu().numpy()]
                    m_kp_b = kp_b[matches[:, 1].cpu().numpy()]

                    if len(m_kp_a) >= self._min_matches:
                        transform = _estimate_transform(m_kp_a, m_kp_b)
                        if transform is not None:
                            transforms[(id_a, id_b)] = transform
                            transforms[(id_b, id_a)] = _invert_transform(transform)
                except Exception as e:
                    log.debug("Match failed %s↔%s: %s", id_a[:6], id_b[:6], e)

                pair_idx += 1
                pct = 40 + int(pair_idx / max(pair_count, 1) * 45)
                self.progress.emit(pct)

        # ── 3. BFS from anchor to build global placements ─────────────────────
        placements = {}   # id → ImagePlacement
        fallback_count = 0

        if not ids:
            return self._place_grid(records)

        # Anchor: first image at origin
        anchor = ids[0]
        placements[anchor] = ImagePlacement(
            image_id=anchor, x=0.0, y=0.0, rotation=0.0, z_order=0,
            auto_x=0.0, auto_y=0.0, auto_rotation=0.0,
        )

        queue = deque([anchor])
        placed = {anchor}
        unplaced = set(ids[1:])

        while queue and unplaced:
            current = queue.popleft()
            current_p = placements[current]

            for other in list(unplaced):
                t = transforms.get((current, other))
                if t is None:
                    continue

                tx, ty, rot = t
                # Apply transform relative to current image's placement
                new_x = current_p.x + tx
                new_y = current_p.y + ty
                new_rot = current_p.rotation + rot

                p = ImagePlacement(
                    image_id=other,
                    x=new_x, y=new_y, rotation=new_rot,
                    z_order=len(placed),
                    auto_x=new_x, auto_y=new_y, auto_rotation=new_rot,
                )
                placements[other] = p
                placed.add(other)
                unplaced.discard(other)
                queue.append(other)

        # ── 4. Grid fallback for unplaced images ──────────────────────────────
        if unplaced:
            fallback_count = len(unplaced)
            log.info("%d images had insufficient overlap — using grid offset", fallback_count)

            # Place them below the main composition
            placed_ys = [placements[pid].y for pid in placed]
            grid_y_start = max(placed_ys) + self._thumb_edge + GRID_PADDING * 4

            for i, uid in enumerate(sorted(unplaced)):
                x = float(i * (self._thumb_edge + GRID_PADDING))
                y = grid_y_start
                p = ImagePlacement(
                    image_id=uid,
                    x=x, y=y, rotation=0.0, z_order=len(placed) + i,
                    auto_x=x, auto_y=y, auto_rotation=0.0,
                )
                placements[uid] = p

        self.progress.emit(100)

        result_list = [placements[pid] for pid in ids if pid in placements]
        total = len(ids)
        matched = total - fallback_count

        msg = (
            f"LightGlue placed {matched}/{total} images via keypoint matching."
        )
        if fallback_count:
            msg += f" {fallback_count} had insufficient overlap — placed separately below."

        log.info(msg)
        return PlacementResult(
            placements=result_list,
            used_lightglue=True,
            fallback_count=fallback_count,
            message=msg,
        )


    # ── Reference-based placement ───────────────────────────────────────────

    def _place_with_references(self, records, plog=None) -> PlacementResult:
        """
        Match each detail photo against reference images (the puzzle box lid).
        Each detail gets an absolute position — no BFS chain, no error accumulation.
        """
        try:
            import torch
            from lightglue import LightGlue, DISK
            import numpy as np
            import cv2
        except ImportError as e:
            log.warning("LightGlue import failed (%s) — falling back to grid", e)
            return self._place_grid(records)

        from PIL import Image as PILImage

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Reference-based placement on device: %s", device)

        extractor = DISK(max_num_keypoints=self._max_kp).eval().to(device)
        matcher = LightGlue(features="disk").eval().to(device)

        refs = self.config.references
        n_refs = len(refs)
        n_detail = len(records)
        total_work = n_refs + n_detail + n_detail * n_refs
        done = 0

        # ── 1. Extract features from reference images at preview resolution ──
        ref_features = {}   # slot → feature dict
        ref_sizes = {}      # slot → (w, h) at preview resolution

        for ref in refs:
            if self._cancelled:
                return self._place_grid(records)
            try:
                pil = PILImage.open(ref.source_path).convert("RGB")
                # Resize to preview resolution for better matching
                w, h = pil.size
                scale = PREVIEW_LONG_EDGE / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
                ref_sizes[ref.slot] = (new_w, new_h)

                arr = np.array(pil)
                # No CLAHE for DISK references — DISK works better on
                # natural tones; CLAHE distorts the feature space
                tensor = _arr_to_tensor(arr, device)
                with torch.no_grad():
                    feats = extractor.extract(tensor)
                ref_features[ref.slot] = feats
            except Exception as e:
                log.warning("Failed to extract features from reference %s: %s", ref.slot, e)

            done += 1
            self.progress.emit(int(done / total_work * 100))

        if not ref_features:
            log.warning("No reference features extracted — falling back to pairwise")
            return self._place_lightglue(records)

        # ── 2. Multi-scale detail feature extraction ─────────────────────────
        # Detail photos and references may come from very different cameras
        # (e.g. 1.6 MP detail vs 12 MP reference).  When both are resized to
        # 1500px, the same physical feature (e.g. a knob on the espresso maker)
        # is ~80px in the reference but ~600px in the detail — DISK features
        # won't match across that scale gap.
        #
        # Solution: extract each detail at MULTIPLE scales relative to the
        # reference.  The scale that produces the most inlier matches wins.
        # This is analogous to how humans squint or step back to see how a
        # puzzle piece fits the box lid.

        # Estimate how many detail photos tile across the reference.
        # For a typical joiner: 6-20 detail shots cover one reference.
        # We try scales that assume the detail covers 1/2, 1/4, 1/6, 1/9
        # of the reference long edge.
        DETAIL_SCALES = [1.0, 0.5, 0.33, 0.25, 0.17]

        detail_multiscale = {}   # image_id → [(scale, features, (w,h)), ...]

        for record in records:
            if self._cancelled:
                return self._place_grid(records)
            try:
                preview_arr = self.store.get_preview(record.id)
                if preview_arr is None:
                    preview_arr = self.store.get_thumbnail(record.id)
                if preview_arr is None:
                    continue

                # Light denoise only for DISK path — CLAHE hurts DISK
                # by flattening the tonal range that DISK uses for descriptors
                import cv2 as cv2_pre
                preview_arr = cv2_pre.fastNlMeansDenoisingColored(
                    preview_arr, None, 3, 3, 7, 21,
                )

                h_full, w_full = preview_arr.shape[:2]
                scales_feats = []

                for ds in DETAIL_SCALES:
                    new_w = max(64, int(w_full * ds))
                    new_h = max(64, int(h_full * ds))

                    if ds < 1.0:
                        import cv2
                        resized = cv2.resize(preview_arr, (new_w, new_h),
                                             interpolation=cv2.INTER_AREA)
                    else:
                        resized = preview_arr
                        new_w, new_h = w_full, h_full

                    tensor = _arr_to_tensor(resized, device)
                    with torch.no_grad():
                        feats = extractor.extract(tensor)
                    scales_feats.append((ds, feats, (new_w, new_h)))

                detail_multiscale[record.id] = scales_feats
            except Exception as e:
                log.warning("Failed to extract features from %s: %s",
                            record.id[:6], e)
            done += 1
            self.progress.emit(int(done / total_work * 100))

        # ── 3. Match each detail (at each scale) against all references ──────
        # Pick the (slot, scale) combination with the most inlier matches.
        # The matched ref-side keypoints tell us WHERE in the reference the
        # detail belongs.  Centroid → canvas position.

        placements = {}
        odds_and_ends = []
        placed_count = 0

        # NO canvas_scale — place at reference pixel coordinates.
        # Tiles are self._thumb_edge pixels wide but the reference is
        # PREVIEW_LONG_EDGE pixels wide.  This means tiles will tile
        # across the reference naturally.  fit_all() zooms the view.

        for record in records:
            if self._cancelled:
                return self._place_grid(records)

            if record.id not in detail_multiscale:
                odds_and_ends.append((record.id, 0.0, 0.0, 0.0))
                done += n_refs
                self.progress.emit(int(done / total_work * 100))
                continue

            best_slot = None
            best_inliers = 0
            best_ref_centroid = None
            best_rot = 0.0
            best_detail_size = (0, 0)

            for slot, ref_feat in ref_features.items():
                for ds, det_feat, (dw, dh) in detail_multiscale[record.id]:
                    try:
                        result = matcher({
                            "image0": ref_feat,
                            "image1": det_feat,
                        })
                        matches = result["matches"][0]
                        if len(matches) == 0:
                            continue

                        kp_ref = ref_feat["keypoints"][0].cpu().numpy()
                        kp_det = det_feat["keypoints"][0].cpu().numpy()
                        m_kp_ref = kp_ref[matches[:, 0].cpu().numpy()]
                        m_kp_det = kp_det[matches[:, 1].cpu().numpy()]

                        if len(m_kp_ref) < 4:
                            continue

                        # RANSAC filter — only trust geometrically consistent
                        # matches.  Raw LightGlue matches include outliers that
                        # pull the centroid off-target.
                        import cv2 as cv2_ransac
                        M, inlier_mask = cv2_ransac.estimateAffinePartial2D(
                            m_kp_ref.reshape(-1, 1, 2).astype(np.float32),
                            m_kp_det.reshape(-1, 1, 2).astype(np.float32),
                            method=cv2_ransac.RANSAC,
                            ransacReprojThreshold=RANSAC_REPROJ,
                        )
                        if M is None or inlier_mask is None:
                            continue

                        inlier_idx = inlier_mask.ravel().astype(bool)
                        n_inliers = int(inlier_idx.sum())

                        if n_inliers > best_inliers and n_inliers >= self._min_matches:
                            inlier_det = m_kp_det[inlier_idx]
                            inlier_ref = m_kp_ref[inlier_idx]

                            rw, rh = ref_sizes.get(slot, (PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE))
                            center = _homography_center(
                                inlier_det, inlier_ref, dw, dh, rw, rh)
                            if center is None:
                                continue
                            cx, cy = center

                            rot = math.degrees(math.atan2(M[1, 0], M[0, 0]))

                            best_slot = slot
                            best_inliers = n_inliers
                            best_ref_centroid = (cx, cy)
                            best_rot = rot
                            best_detail_size = (dw, dh)
                    except Exception as e:
                        log.debug("Match %s↔%s@%.0f%% failed: %s",
                                  slot, record.id[:6], ds * 100, e)

                done += 1
                self.progress.emit(int(done / total_work * 100))

            if best_slot and best_ref_centroid:
                cx, cy = best_ref_centroid

                # For perspective projects with multiple refs, offset by
                # the reference anchor position
                if self.config.project_type == "perspective":
                    offsets = {
                        "standing": (0, 0),
                        "left": (-1, 0),
                        "right": (1, 0),
                        "down_low": (0, 1),
                        "up_high": (0, -1),
                    }
                    ox, oy = offsets.get(best_slot, (0, 0))
                    rw, rh = ref_sizes.get(best_slot,
                                           (PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE))
                    cx += ox * rw
                    cy += oy * rh

                # Place at reference pixel coordinates directly
                final_x = cx
                final_y = cy

                # Centre the tile on the match point (tile is at thumb size)
                aspect = record.width / max(record.height, 1)
                if aspect >= 1:
                    tile_w = self._thumb_edge
                    tile_h = self._thumb_edge / aspect
                else:
                    tile_w = self._thumb_edge * aspect
                    tile_h = self._thumb_edge
                final_x -= tile_w / 2
                final_y -= tile_h / 2

                p = ImagePlacement(
                    image_id=record.id,
                    x=final_x, y=final_y, rotation=best_rot,
                    z_order=placed_count,
                    auto_x=final_x, auto_y=final_y, auto_rotation=best_rot,
                )
                placements[record.id] = p
                placed_count += 1
            else:
                # Save best partial match info for second-pass attempt
                odds_and_ends.append((record.id, best_inliers,
                                      best_ref_centroid, best_rot))

        # ── 3b. Second pass — place near-misses with relaxed threshold ────
        # Images that got SOME inliers but not enough for the strict threshold
        # often have useful position info.  Accept them at a lower bar.
        still_unmatched = []
        for uid, n_inliers, ref_centroid, rot in odds_and_ends:
            if n_inliers >= MIN_MATCHES_RELAXED and ref_centroid:
                cx, cy = ref_centroid
                record = self.store.get_record(uid)
                if record:
                    aspect = record.width / max(record.height, 1)
                    tile_w = self._thumb_edge if aspect >= 1 else self._thumb_edge * aspect
                    tile_h = self._thumb_edge / aspect if aspect >= 1 else self._thumb_edge
                else:
                    tile_w = tile_h = self._thumb_edge
                final_x = cx - tile_w / 2
                final_y = cy - tile_h / 2
                p = ImagePlacement(
                    image_id=uid,
                    x=final_x, y=final_y, rotation=rot,
                    z_order=placed_count,
                    auto_x=final_x, auto_y=final_y, auto_rotation=rot,
                )
                placements[uid] = p
                placed_count += 1
                log.info("Second-pass placed %s with %d inliers (relaxed threshold)",
                         uid[:6], n_inliers)
            else:
                gx, gy, gr = 0.0, 0.0, 0.0
                if ref_centroid:
                    gx, gy = ref_centroid
                    gr = rot
                still_unmatched.append((uid, gx, gy, gr))
        odds_and_ends = [(uid, gx, gy, gr) for uid, gx, gy, gr in still_unmatched]

        # ── 4. Template matching fallback for low-quality images ─────────────
        if odds_and_ends:
            placements, odds_and_ends, placed_count = self._template_fallback(
                records, placements, odds_and_ends, placed_count,
            )

        # ── 5. Spread overlapping tiles ─────────────────────────────────────
        # Multiple detail shots covering the same area stack on top of each
        # other.  Nudge them apart so the joiner looks assembled, not piled.
        _spread_overlaps(placements, self._thumb_edge, OVERLAP_REPEL, OVERLAP_THRESHOLD)

        # ── 6. Odds & Ends tray — place unmatched below the composition ──────
        fallback_count = len(odds_and_ends)
        if odds_and_ends:
            placed_ys = [p.y for p in placements.values()] or [0.0]
            tray_y = max(placed_ys) + self._thumb_edge + GRID_PADDING * 4

            for i, (uid, gx, gy, gr) in enumerate(odds_and_ends):
                x = float(i * (self._thumb_edge + GRID_PADDING))
                p = ImagePlacement(
                    image_id=uid,
                    x=x, y=tray_y, rotation=0.0,
                    z_order=placed_count + i,
                    auto_x=x, auto_y=tray_y, auto_rotation=0.0,
                )
                # Store best guess in auto_ fields for ghost preview
                # (actual position is in the tray; auto_ holds the guess)
                p.auto_x = gx
                p.auto_y = gy
                p.auto_rotation = gr
                placements[uid] = p

        self.progress.emit(100)

        result_list = list(placements.values())
        matched = placed_count
        total = len(records)

        msg = f"Reference placed {matched}/{total} images."
        if fallback_count:
            msg += f" {fallback_count} in Odds & Ends tray."

        # Log results
        if plog:
            plog.set_engine("disk_lightglue")
            plog.set_counts(total, matched, fallback_count)
            plog.flush()

        log.info(msg)
        return PlacementResult(
            placements=result_list,
            used_lightglue=True,
            fallback_count=fallback_count,
            message=msg,
        )

    # ── Template matching fallback (shared by DISK and SIFT paths) ────────

    def _template_fallback(self, records, placements, odds_and_ends, placed_count):
        """
        Try template matching (cv2.matchTemplate) for images that failed
        keypoint matching.  Works on overall appearance — handles blurry
        and noisy images better than keypoint methods.
        Returns updated (placements, odds_and_ends, placed_count).
        """
        import cv2 as cv2_tmpl
        import numpy as np
        from PIL import Image as PILImage

        refs = self.config.references
        ref_sizes = {}

        # Build greyscale reference images
        ref_grey = {}
        for ref in refs:
            try:
                pil = PILImage.open(ref.source_path).convert("RGB")
                w, h = pil.size
                scale = PREVIEW_LONG_EDGE / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
                ref_sizes[ref.slot] = (new_w, new_h)
                # CLAHE before greyscale conversion for better contrast
                from hockney.core.placement_sift import clahe_preprocess
                arr = clahe_preprocess(np.array(pil), clip_limit=3.0)
                ref_grey[ref.slot] = cv2_tmpl.cvtColor(arr, cv2_tmpl.COLOR_RGB2GRAY)
            except Exception:
                pass

        if not ref_grey:
            return placements, odds_and_ends, placed_count

        still_unplaced = []
        for uid, gx, gy, gr in odds_and_ends:
            try:
                preview_arr = self.store.get_preview(uid)
                if preview_arr is None:
                    preview_arr = self.store.get_thumbnail(uid)
                if preview_arr is None:
                    still_unplaced.append((uid, gx, gy, gr))
                    continue

                # CLAHE + greyscale
                from hockney.core.placement_sift import clahe_preprocess
                preview_arr = clahe_preprocess(preview_arr, clip_limit=3.0)
                detail_grey = cv2_tmpl.cvtColor(preview_arr, cv2_tmpl.COLOR_RGB2GRAY)
                detail_grey = cv2_tmpl.GaussianBlur(detail_grey, (3, 3), 0)

                best_val = -1.0
                best_loc = None
                best_slot = None
                best_tmpl_scale = 1.0

                for slot, rg in ref_grey.items():
                    dh, dw = detail_grey.shape[:2]
                    rh, rw = rg.shape[:2]

                    for tmpl_scale in [1.0, 0.75, 0.5, 0.35, 0.25]:
                        sw = int(dw * tmpl_scale)
                        sh = int(dh * tmpl_scale)
                        if sw < 16 or sh < 16 or sw >= rw or sh >= rh:
                            continue

                        if tmpl_scale < 1.0:
                            tmpl = cv2_tmpl.resize(detail_grey, (sw, sh),
                                                   interpolation=cv2_tmpl.INTER_AREA)
                        else:
                            tmpl = detail_grey
                            if dw >= rw or dh >= rh:
                                continue

                        result = cv2_tmpl.matchTemplate(
                            rg, tmpl, cv2_tmpl.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2_tmpl.minMaxLoc(result)

                        if max_val > best_val:
                            best_val = max_val
                            best_loc = max_loc
                            best_slot = slot
                            best_tmpl_scale = tmpl_scale

                if best_val >= 0.35 and best_loc and best_slot:
                    mx, my = best_loc
                    sw = int(detail_grey.shape[1] * best_tmpl_scale)
                    sh = int(detail_grey.shape[0] * best_tmpl_scale)
                    cx = mx + sw / 2
                    cy = my + sh / 2

                    if self.config.project_type == "perspective":
                        offsets = {
                            "standing": (0, 0), "left": (-1, 0),
                            "right": (1, 0), "down_low": (0, 1),
                            "up_high": (0, -1),
                        }
                        ox, oy = offsets.get(best_slot, (0, 0))
                        rw_s, rh_s = ref_sizes.get(
                            best_slot, (PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE))
                        cx += ox * rw_s
                        cy += oy * rh_s

                    final_x = cx
                    final_y = cy

                    record = self.store.get_record(uid)
                    if record:
                        aspect = record.width / max(record.height, 1)
                        tile_w = self._thumb_edge if aspect >= 1 else self._thumb_edge * aspect
                        tile_h = self._thumb_edge / aspect if aspect >= 1 else self._thumb_edge
                    else:
                        tile_w = tile_h = self._thumb_edge
                    final_x -= tile_w / 2
                    final_y -= tile_h / 2

                    p = ImagePlacement(
                        image_id=uid,
                        x=final_x, y=final_y, rotation=0.0,
                        z_order=placed_count,
                        auto_x=final_x, auto_y=final_y, auto_rotation=0.0,
                    )
                    placements[uid] = p
                    placed_count += 1
                    log.info("Template matched %s in %s (corr=%.2f, scale=%.0f%%)",
                             uid[:6], best_slot, best_val, best_tmpl_scale * 100)
                else:
                    still_unplaced.append((uid, gx, gy, gr))
            except Exception as e:
                log.debug("Template match failed for %s: %s", uid[:6], e)
                still_unplaced.append((uid, gx, gy, gr))

        rescued = len(odds_and_ends) - len(still_unplaced)
        if rescued > 0:
            log.info("Template matching rescued %d images", rescued)

        return placements, still_unplaced, placed_count


# ── Overlap spreading ─────────────────────────────────────────────────────────

def _spread_overlaps(placements: dict[str, ImagePlacement],
                     tile_size: float, repel_frac: float,
                     overlap_threshold: float = 0.6,
                     iterations: int = 12):
    """
    Nudge tiles that overlap so they spread into a 2D mosaic instead of piling up.

    The key insight: homography-based placement often puts many tiles in the
    same general area (they all match the same high-contrast region).  A simple
    3-iteration push isn't enough — we need many iterations with perpendicular
    jitter to break diagonal cascade patterns.

    Each iteration:
      1. Finds overlapping pairs (centres closer than overlap_threshold * tile_size)
      2. Pushes them apart along the line between their centres
      3. Adds a small perpendicular jitter to prevent stable diagonal lines
    """
    if len(placements) < 2:
        return

    ids = list(placements.keys())
    n = len(ids)
    push = tile_size * repel_frac
    min_dist = tile_size * overlap_threshold

    import random
    rng = random.Random(42)  # deterministic jitter

    for iteration in range(iterations):
        moved = False
        # Decay push over iterations — start aggressive, end gentle
        decay = 1.0 - (iteration / iterations) * 0.5  # 1.0 → 0.5
        this_push = push * decay

        for i in range(n):
            for j in range(i + 1, n):
                p_a = placements[ids[i]]
                p_b = placements[ids[j]]
                dx = p_b.x - p_a.x
                dy = p_b.y - p_a.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < min_dist:
                    moved = True
                    if dist < 1.0:
                        # Near-identical position — random direction
                        angle = rng.uniform(0, 2 * math.pi)
                        dx = math.cos(angle)
                        dy = math.sin(angle)
                        dist = 1.0
                    nx = dx / dist
                    ny = dy / dist

                    # Add perpendicular jitter (breaks diagonal cascades)
                    jitter = rng.uniform(-0.3, 0.3) * this_push
                    perp_x = -ny * jitter
                    perp_y = nx * jitter

                    half_push = this_push * 0.5
                    p_a.x -= nx * half_push - perp_x
                    p_a.y -= ny * half_push - perp_y
                    p_a.auto_x = p_a.x
                    p_a.auto_y = p_a.y
                    p_b.x += nx * half_push + perp_x
                    p_b.y += ny * half_push + perp_y
                    p_b.auto_x = p_b.x
                    p_b.auto_y = p_b.y

        if not moved:
            break  # converged early


# ── Transform helpers ──────────────────────────────────────────────────────────

def _arr_to_tensor(arr, device):
    """Convert HxWx3 uint8 numpy array to 1xCxHxW float tensor on device."""
    import torch
    import numpy as np
    t = torch.from_numpy(arr.astype(np.float32) / 255.0)  # HxWx3
    t = t.permute(2, 0, 1).unsqueeze(0)                   # 1x3xHxW
    return t.to(device)


def _estimate_transform(kp_a, kp_b) -> tuple[float, float, float] | None:
    """
    Estimate similarity transform (translation + rotation) from matched keypoints.
    Returns (tx, ty, rotation_degrees) or None if estimation fails.
    Uses RANSAC for robustness.
    """
    import cv2
    import numpy as np

    if len(kp_a) < 4:
        return None

    # estimateAffinePartial2D: similarity transform (scale+rotation+translation)
    # We discard scale — geometry must be honest
    M, inliers = cv2.estimateAffinePartial2D(
        kp_a.reshape(-1, 1, 2).astype(np.float32),
        kp_b.reshape(-1, 1, 2).astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )

    if M is None or inliers is None or inliers.sum() < MIN_MATCHES:
        return None

    # Extract translation and rotation, discard scale
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    rot_rad = math.atan2(M[1, 0], M[0, 0])
    rot_deg = math.degrees(rot_rad)

    return tx, ty, rot_deg


def _invert_transform(t: tuple[float, float, float]) -> tuple[float, float, float]:
    """Invert a (tx, ty, rot_deg) similarity transform."""
    tx, ty, rot_deg = t
    rot_rad = math.radians(rot_deg)
    cos_r, sin_r = math.cos(-rot_rad), math.sin(-rot_rad)
    inv_tx = -(tx * cos_r - ty * sin_r)
    inv_ty = -(tx * sin_r + ty * cos_r)
    return inv_tx, inv_ty, -rot_deg
