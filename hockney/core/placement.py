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
from hockney.core.models import ImagePlacement

log = logging.getLogger(__name__)

THUMB_LONG_EDGE = 300
GRID_PADDING = 24
MIN_MATCHES = 12        # minimum inlier keypoint pairs to trust a transform
MAX_KEYPOINTS = 1024    # per image, for DISK


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

    def __init__(self, store: ImageStore, model_ready: bool, models_dir=None):
        super().__init__()
        self.store = store
        self.model_ready = model_ready
        self.models_dir = models_dir

    def run(self):
        records = self.store.all_records()
        if not records:
            self.finished.emit(PlacementResult([], False, 0, "No images to place."))
            return

        if self.model_ready:
            result = self._place_lightglue(records)
        else:
            result = self._place_grid(records)

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
            th = int(THUMB_LONG_EDGE / aspect) if aspect >= 1 else THUMB_LONG_EDGE
            x = float(col * (THUMB_LONG_EDGE + GRID_PADDING))
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
        extractor = DISK(max_num_keypoints=MAX_KEYPOINTS).eval().to(device)
        matcher = LightGlue(features="disk").eval().to(device)

        n = len(records)
        features = {}   # record.id → lightglue feature dict

        for i, record in enumerate(records):
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

                    if len(m_kp_a) >= MIN_MATCHES:
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
            grid_y_start = max(placed_ys) + THUMB_LONG_EDGE + GRID_PADDING * 4

            for i, uid in enumerate(sorted(unplaced)):
                x = float(i * (THUMB_LONG_EDGE + GRID_PADDING))
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
