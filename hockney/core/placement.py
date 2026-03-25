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
MIN_MATCHES = 12        # minimum inlier keypoint pairs to trust a transform
MAX_KEYPOINTS = 1024    # per image, for DISK

# Subject-type tuning overrides
SUBJECT_TUNING = {
    "landscape":  {"max_keypoints": 1024, "min_matches": 12},
    "skylife":    {"max_keypoints": 512,  "min_matches": 8},
    "urban":      {"max_keypoints": 2048, "min_matches": 15},
    "indoor":     {"max_keypoints": 1024, "min_matches": 10},
    "people":     {"max_keypoints": 1024, "min_matches": 12},
}


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

        # Use reference-based placement if references are loaded
        if self.config and self.config.has_references() and self.model_ready:
            result = self._place_with_references(records)
        elif self.model_ready:
            result = self._place_lightglue(records)
        else:
            result = self._place_grid(records)

        if not self._cancelled:
            self.finished.emit(result)

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

    def _place_with_references(self, records) -> PlacementResult:
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

        # ── 2. Extract features from detail photos (thumbnails) ──────────────
        detail_features = {}

        for record in records:
            if self._cancelled:
                return self._place_grid(records)
            arr = self.store.get_thumbnail(record.id)
            if arr is None:
                continue
            tensor = _arr_to_tensor(arr, device)
            with torch.no_grad():
                feats = extractor.extract(tensor)
            detail_features[record.id] = feats
            done += 1
            self.progress.emit(int(done / total_work * 100))

        # ── 3. Compute reference anchor positions on canvas ──────────────────
        # References tile based on project type
        ref_anchors = {}  # slot → (anchor_x, anchor_y)
        slot_list = list(ref_features.keys())

        if self.config.project_type == "perspective":
            # Standing at origin; Left offset left, Right offset right
            # Down Low below, Up High above
            offsets = {
                "standing": (0, 0),
                "left": (-1, 0),
                "right": (1, 0),
                "down_low": (0, 1),
                "up_high": (0, -1),
            }
            for slot in slot_list:
                ox, oy = offsets.get(slot, (0, 0))
                rw, rh = ref_sizes.get(slot, (PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE))
                ref_anchors[slot] = (ox * rw * 0.5, oy * rh * 0.5)
        else:
            # Time of Day / Seasonal: tile left to right
            x_offset = 0.0
            for slot in slot_list:
                rw, rh = ref_sizes.get(slot, (PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE))
                ref_anchors[slot] = (x_offset, 0.0)
                x_offset += rw + GRID_PADDING

        # ── 4. Match each detail against all references, pick best ───────────
        placements = {}
        odds_and_ends = []    # (image_id, best_guess_x, best_guess_y, best_guess_rot)
        placed_count = 0

        for record in records:
            if self._cancelled:
                return self._place_grid(records)

            if record.id not in detail_features:
                odds_and_ends.append((record.id, 0.0, 0.0, 0.0))
                done += n_refs
                self.progress.emit(int(done / total_work * 100))
                continue

            best_slot = None
            best_inliers = 0
            best_transform = None

            for slot, ref_feat in ref_features.items():
                try:
                    result = matcher({
                        "image0": ref_feat,
                        "image1": detail_features[record.id],
                    })
                    matches = result["matches"][0]
                    kp_ref = ref_feat["keypoints"][0].cpu().numpy()
                    kp_det = detail_features[record.id]["keypoints"][0].cpu().numpy()
                    m_kp_ref = kp_ref[matches[:, 0].cpu().numpy()]
                    m_kp_det = kp_det[matches[:, 1].cpu().numpy()]

                    n_inliers = len(m_kp_ref)
                    if n_inliers > best_inliers:
                        t = _estimate_transform(m_kp_ref, m_kp_det)
                        if t is not None:
                            best_slot = slot
                            best_inliers = n_inliers
                            best_transform = t
                except Exception as e:
                    log.debug("Match %s↔%s failed: %s", slot, record.id[:6], e)

                done += 1
                self.progress.emit(int(done / total_work * 100))

            if best_slot and best_transform and best_inliers >= self._min_matches:
                tx, ty, rot = best_transform
                ax, ay = ref_anchors.get(best_slot, (0.0, 0.0))

                # Scale transform from reference resolution to thumbnail
                # resolution since detail features are at 300px but ref is 1500px
                ref_w, ref_h = ref_sizes.get(best_slot, (PREVIEW_LONG_EDGE, PREVIEW_LONG_EDGE))
                # The transform tx,ty is in reference pixel coords — scale to thumb coords
                # Detail is at THUMB scale, reference is at PREVIEW scale
                # We place on canvas using reference coordinate system
                final_x = ax + tx
                final_y = ay + ty

                p = ImagePlacement(
                    image_id=record.id,
                    x=final_x, y=final_y, rotation=rot,
                    z_order=placed_count,
                    auto_x=final_x, auto_y=final_y, auto_rotation=rot,
                )
                placements[record.id] = p
                placed_count += 1
            else:
                # Save best guess even if below threshold — for ghost preview
                guess_x, guess_y, guess_rot = best_transform if best_transform else (0.0, 0.0, 0.0)
                if best_slot:
                    ax, ay = ref_anchors.get(best_slot, (0.0, 0.0))
                    guess_x += ax
                    guess_y += ay
                odds_and_ends.append((record.id, guess_x, guess_y, guess_rot))

        # ── 5. Odds & Ends tray — place unmatched below the composition ──────
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

        log.info(msg)
        return PlacementResult(
            placements=result_list,
            used_lightglue=True,
            fallback_count=fallback_count,
            message=msg,
        )


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
