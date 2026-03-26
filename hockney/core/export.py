"""
export.py — Full-resolution composite renderer.

Takes all visible ImagePlacement objects and composites the original
full-resolution source images into a single output file.

Scale modes:
  "screen"    — 1× thumbnail size (300px per image slot). Fast preview.
  "medium"    — 5× thumbnail size (~1500px per image slot). Good for web.
  "full"      — Use full original resolution. Largest output, print quality.
  float       — Explicit scale multiplier (e.g. 3.0).

The composition geometry is preserved exactly as seen in the Tray View.
Rotation is applied around each image's centre, same as Qt does it.

Runs in a background thread — never blocks the UI.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Union

from PyQt6.QtCore import QThread, pyqtSignal

from hockney.core.image_store import ImageStore, THUMB_LONG_EDGE
from hockney.core.models import ImagePlacement

log = logging.getLogger(__name__)

ScaleMode = Union[str, float]   # "screen" | "medium" | "full" | float


# ── Geometry helpers ───────────────────────────────────────────────────────────

def _rotated_corners(x: float, y: float, w: float, h: float, angle_deg: float):
    """
    Return the four corners of a rectangle after rotation around its centre.
    Angle is in degrees, clockwise (Qt convention).
    """
    cx = x + w / 2
    cy = y + h / 2
    r = math.radians(-angle_deg)   # PIL rotates CCW, Qt CW
    cos_r, sin_r = math.cos(r), math.sin(r)

    corners = [(-w / 2, -h / 2), (w / 2, -h / 2),
               (w / 2,  h / 2), (-w / 2,  h / 2)]
    return [
        (cx + dx * cos_r - dy * sin_r, cy + dx * sin_r + dy * cos_r)
        for dx, dy in corners
    ]


def _canvas_size(placements: list[ImagePlacement], store: ImageStore,
                 scale: float) -> tuple[int, int, float, float]:
    """
    Compute the canvas dimensions and top-left origin needed to contain
    all placed images at the given scale. Returns (width, height, origin_x, origin_y).
    """
    all_x, all_y = [], []

    for p in placements:
        record = store.get_record(p.image_id)
        if not record:
            continue
        # Thumbnail dimensions determine the slot size
        aspect = record.width / max(record.height, 1)
        if aspect >= 1:
            tw, th = THUMB_LONG_EDGE, THUMB_LONG_EDGE / aspect
        else:
            tw, th = THUMB_LONG_EDGE * aspect, THUMB_LONG_EDGE

        corners = _rotated_corners(p.x * scale, p.y * scale,
                                   tw * scale, th * scale, p.rotation)
        for cx, cy in corners:
            all_x.append(cx)
            all_y.append(cy)

    if not all_x:
        return 100, 100, 0.0, 0.0

    padding = 40 * scale
    min_x = min(all_x) - padding
    min_y = min(all_y) - padding
    max_x = max(all_x) + padding
    max_y = max(all_y) + padding

    return int(max_x - min_x), int(max_y - min_y), min_x, min_y


def _resolve_scale(mode: ScaleMode, placements: list[ImagePlacement],
                   store: ImageStore) -> float:
    if isinstance(mode, float):
        return mode
    if mode == "screen":
        return 1.0
    if mode == "medium":
        return 5.0
    if mode == "full":
        # Scale so the largest image renders at its original resolution
        max_scale = 1.0
        for p in placements:
            record = store.get_record(p.image_id)
            if record:
                s = max(record.width, record.height) / THUMB_LONG_EDGE
                max_scale = max(max_scale, s)
        return max_scale
    return 5.0   # fallback


# ── Compositor ─────────────────────────────────────────────────────────────────

def render_composite(
    placements: list[ImagePlacement],
    store: ImageStore,
    scale: float,
    processing: dict | None = None,
    progress_cb=None,
) -> "Image":
    """
    Render a composite PIL Image from the given placements at the given scale.
    Images are drawn in z_order (lowest first = furthest back).
    processing: optional dict with histogram_eq, filter, surface_effect keys.
    progress_cb(pct: int) called periodically if provided.
    """
    from PIL import Image, ImageOps

    processing = processing or {}
    do_eq = processing.get("histogram_eq", False)
    filter_name = processing.get("filter", "None")
    surface_effect = processing.get("surface_effect", "None")
    surface_intensity = processing.get("surface_intensity", 0) / 100.0

    if filter_name and filter_name != "None":
        from hockney.effects.filters import apply_filter as _apply_filter
    else:
        _apply_filter = None

    if surface_effect in ("Random Curves", "Combined") and surface_intensity > 0:
        from hockney.effects.surface import apply_per_tile_warp
    else:
        apply_per_tile_warp = None

    sorted_placements = sorted(placements, key=lambda p: p.z_order)
    canvas_w, canvas_h, origin_x, origin_y = _canvas_size(placements, store, scale)

    log.info("Canvas: %dx%d at scale %.1f×", canvas_w, canvas_h, scale)

    transparent = processing.get("transparent_bg", False)
    bg_color = processing.get("bg_color", (30, 30, 30))

    if transparent:
        canvas = Image.new("RGBA", (canvas_w, canvas_h), color=(0, 0, 0, 0))
    else:
        canvas = Image.new("RGB", (canvas_w, canvas_h), color=bg_color)
    n = len(sorted_placements)

    for i, p in enumerate(sorted_placements):
        record = store.get_record(p.image_id)
        if not record:
            continue

        # Load full-res original
        full = store.get_full_res(p.image_id)
        if full is None:
            log.warning("Could not load full-res for %s", p.image_id)
            continue

        full = full.convert("RGB")

        # ── Batch processing ───────────────────────────────────────────────────
        if do_eq:
            full = ImageOps.equalize(full)
        if _apply_filter:
            full = _apply_filter(full, filter_name)
        if apply_per_tile_warp:
            full = apply_per_tile_warp(full, p.image_id, surface_intensity)

        full = full.convert("RGBA")

        # Determine slot size at this scale
        aspect = record.width / max(record.height, 1)
        if aspect >= 1:
            tw = int(THUMB_LONG_EDGE * scale)
            th = int((THUMB_LONG_EDGE / aspect) * scale)
        else:
            tw = int((THUMB_LONG_EDGE * aspect) * scale)
            th = int(THUMB_LONG_EDGE * scale)

        # Resize original to slot size
        resized = full.resize((tw, th), Image.LANCZOS)

        # Add white border if specified
        border_width = processing.get("border_width", 0)
        if border_width > 0:
            border_width_scaled = int(border_width * scale)
            if border_width_scaled > 0:
                # Determine if we need alpha channel for transparent backgrounds
                if transparent and resized.mode != "RGBA":
                    resized = resized.convert("RGBA")
                resized = ImageOps.expand(resized, border=border_width_scaled, fill=(255, 255, 255, 255) if resized.mode == "RGBA" else (255, 255, 255))
                # Update dimensions for paste position calculation
                tw = resized.width
                th = resized.height

        # Apply crop if set (crop values are in thumbnail space, scale them)
        if p.crop_top or p.crop_right or p.crop_bottom or p.crop_left:
            ct = int(p.crop_top * scale)
            cr = int(p.crop_right * scale)
            cb = int(p.crop_bottom * scale)
            cl = int(p.crop_left * scale)
            resized = resized.crop((cl, ct, resized.width - cr, resized.height - cb))
            tw = resized.width
            th = resized.height

        # Rotate around image centre (PIL rotates CCW; Qt CW — negate angle)
        if abs(p.rotation) > 0.001:
            rotated = resized.rotate(-p.rotation, expand=True, resample=Image.BICUBIC)
        else:
            rotated = resized

        # Paste position: centre of rotated image at (cx, cy) on canvas
        cx = p.x * scale - origin_x + tw / 2
        cy = p.y * scale - origin_y + th / 2
        paste_x = int(cx - rotated.width / 2)
        paste_y = int(cy - rotated.height / 2)

        # Drop shadow rendering (if enabled)
        if processing.get("shadow", False):
            from PIL import ImageDraw, ImageFilter
            shadow_offset = int(4 * scale)
            shadow = Image.new("RGBA", rotated.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            shadow_draw.rectangle([0, 0, shadow.width-1, shadow.height-1], fill=(0, 0, 0, 80))
            shadow = shadow.filter(ImageFilter.GaussianBlur(radius=int(6*scale)))
            shadow_alpha = shadow.split()[3]
            if transparent:
                canvas.paste(shadow, (paste_x + shadow_offset, paste_y + shadow_offset), shadow_alpha)
            else:
                canvas.paste(shadow.convert("RGB"), (paste_x + shadow_offset, paste_y + shadow_offset), shadow_alpha)

        # Use alpha channel as mask so rotated corners don't leave rectangles
        alpha = rotated.split()[3] if rotated.mode == "RGBA" else None
        if transparent:
            canvas.paste(rotated, (paste_x, paste_y), alpha)
        else:
            canvas.paste(rotated.convert("RGB"), (paste_x, paste_y), alpha)

        if progress_cb:
            progress_cb(int((i + 1) / n * 95))

    return canvas


# ── Export worker ──────────────────────────────────────────────────────────────

class ExportWorker(QThread):
    """Background thread for composite rendering + file save."""

    progress = pyqtSignal(int)
    finished = pyqtSignal(str)    # path of saved file
    error = pyqtSignal(str)

    def __init__(
        self,
        placements: list[ImagePlacement],
        store: ImageStore,
        output_path: Path,
        scale_mode: ScaleMode = "medium",
        processing: dict | None = None,
    ):
        super().__init__()
        self.placements = [p for p in placements if p.image_id]
        self.store = store
        self.output_path = output_path
        self.scale_mode = scale_mode
        self.processing = processing or {}

    def run(self):
        try:
            scale = _resolve_scale(self.scale_mode, self.placements, self.store)
            log.info("Exporting at scale %.1f× to %s", scale, self.output_path)

            composite = render_composite(
                self.placements,
                self.store,
                scale,
                processing=self.processing,
                progress_cb=self.progress.emit,
            )

            # Apply surface texture effects to finished composite
            surface_effect = self.processing.get("surface_effect", "None")
            surface_intensity = self.processing.get("surface_intensity", 0) / 100.0
            if surface_effect != "None" and surface_intensity > 0:
                from hockney.effects.surface import apply_surface
                log.info("Applying surface effect: %s at %.0f%%", surface_effect, surface_intensity * 100)
                composite = apply_surface(composite, surface_effect, surface_intensity)

            self.progress.emit(97)
            self._save(composite)
            self.progress.emit(100)
            self.finished.emit(str(self.output_path))

        except Exception as e:
            log.exception("Export failed")
            self.error.emit(str(e))

    def _save(self, img):
        suffix = self.output_path.suffix.lower()
        # JPEG doesn't support alpha — flatten to RGB
        if suffix in (".jpg", ".jpeg") and img.mode == "RGBA":
            bg = img.copy()
            bg_color = self.processing.get("bg_color", (30, 30, 30))
            flat = __import__("PIL").Image.new("RGB", img.size, bg_color)
            flat.paste(img, mask=img.split()[3])
            img = flat

        if suffix in (".tif", ".tiff"):
            img.save(self.output_path, format="TIFF", compression="lzw")
        elif suffix == ".png":
            img.save(self.output_path, format="PNG", optimize=True)
        elif suffix in (".jpg", ".jpeg"):
            img.save(self.output_path, format="JPEG", quality=95, optimize=True)
        else:
            img.save(self.output_path, format="TIFF", compression="lzw")
        log.info("Saved: %s (%s)", self.output_path, self.output_path.stat())


# ── Contact sheet renderer (used by moondream) ─────────────────────────────────

def render_contact_sheet(
    placements: list[ImagePlacement],
    store: ImageStore,
    thumb_size: int = 200,
    cols: int = 8,
) -> "Image":
    """
    Render a numbered contact sheet of all images for moondream analysis.
    Numbers are burned into the top-left corner of each thumbnail.
    Returns a PIL Image.
    """
    from PIL import Image, ImageDraw, ImageFont

    sorted_p = sorted(placements, key=lambda p: p.z_order)
    n = len(sorted_p)
    rows = math.ceil(n / cols)

    sheet_w = cols * thumb_size
    sheet_h = rows * thumb_size
    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(sheet)

    for i, p in enumerate(sorted_p):
        arr = store.get_thumbnail(p.image_id)
        if arr is None:
            continue

        from PIL import Image as PILImage
        import numpy as np
        thumb = PILImage.fromarray(arr)
        thumb = thumb.resize((thumb_size, thumb_size), PILImage.LANCZOS)

        col = i % cols
        row = i // cols
        x = col * thumb_size
        y = row * thumb_size
        sheet.paste(thumb, (x, y))

        # Burn index number into corner
        label = str(i + 1)
        draw.rectangle([x, y, x + 28, y + 20], fill=(0, 0, 0, 180))
        draw.text((x + 3, y + 2), label, fill=(255, 220, 0))

    return sheet
