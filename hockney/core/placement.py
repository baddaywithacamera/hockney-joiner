"""
placement.py — Image placement engine.

Stage 1 (now):   Grid fallback. Images arranged in a tidy grid.
Stage 2 (next):  LightGlue keypoint matching for overlap detection.

The interface is the same either way — PlacementWorker emits a list of
ImagePlacement objects. The TrayView doesn't care how they were computed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from PyQt6.QtCore import QThread, pyqtSignal

from hockney.core.image_store import ImageStore
from hockney.core.models import ImagePlacement

log = logging.getLogger(__name__)

THUMB_LONG_EDGE = 300    # matches image_store.THUMB_LONG_EDGE
GRID_PADDING = 24        # px between images in grid fallback


@dataclass
class PlacementResult:
    placements: list[ImagePlacement]
    used_lightglue: bool
    fallback_count: int   # how many pairs fell back to grid (0 = all LightGlue)
    message: str


class PlacementWorker(QThread):
    """
    Background thread that computes image placements.
    Emits finished(PlacementResult) or error(str).
    """

    finished = pyqtSignal(object)   # PlacementResult
    progress = pyqtSignal(int)      # 0-100
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
        """
        Arrange images in a grid. Used when LightGlue isn't available,
        or when pairs have insufficient overlap.
        """
        n = len(records)
        cols = max(1, math.isqrt(n))

        placements = []
        for i, record in enumerate(records):
            col = i % cols
            row = i // cols

            # Use actual thumbnail aspect ratio for spacing
            aspect = record.width / max(record.height, 1)
            if aspect >= 1:
                tw, th = THUMB_LONG_EDGE, int(THUMB_LONG_EDGE / aspect)
            else:
                tw, th = int(THUMB_LONG_EDGE * aspect), THUMB_LONG_EDGE

            x = col * (THUMB_LONG_EDGE + GRID_PADDING)
            y = row * (THUMB_LONG_EDGE + GRID_PADDING)

            p = ImagePlacement(
                image_id=record.id,
                x=float(x), y=float(y),
                rotation=0.0, z_order=i,
                auto_x=float(x), auto_y=float(y), auto_rotation=0.0,
            )
            placements.append(p)
            self.progress.emit(int((i + 1) / n * 100))

        log.info("Grid placement: %d images, %d columns", n, cols)
        return PlacementResult(
            placements=placements,
            used_lightglue=False,
            fallback_count=n,
            message=f"Grid layout applied ({n} images). Auto-place requires LightGlue model.",
        )

    # ── LightGlue placement ────────────────────────────────────────────────────

    def _place_lightglue(self, records) -> PlacementResult:
        """
        Use LightGlue to match keypoints between overlapping image pairs,
        then compute canvas positions from matched transforms.

        Falls back to grid for pairs with insufficient overlap (<30%).
        """
        try:
            import torch
            import lightglue  # noqa: F401
        except ImportError as e:
            log.warning("LightGlue import failed: %s — falling back to grid", e)
            return self._place_grid(records)

        # TODO: implement LightGlue pipeline
        # Outline:
        #   1. Extract SuperPoint features for each thumbnail (store.get_process_array)
        #   2. For each image pair with likely spatial proximity, run LightGlue matcher
        #   3. From matched keypoints, compute affine/similarity transform (cv2.estimateAffinePartial2D)
        #   4. Build a global placement by chaining transforms from a fixed anchor image
        #   5. For pairs with <30% inlier ratio, flag as fallback and use grid offset instead
        #   6. Emit progress as pairs are processed

        log.info("LightGlue pipeline not yet implemented — falling back to grid")
        result = self._place_grid(records)
        result.message = (
            "LightGlue pipeline coming soon. Grid layout applied for now."
        )
        return result
