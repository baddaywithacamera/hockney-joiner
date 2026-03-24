"""
surface.py — Surface texture effects for the final composite.

Simulates physical photographs laid on a surface. Subtle is correct.
If it calls attention to itself, dial it back.

Effects:
  None           — Flat composite. No treatment. Default.
  Random Curves  — Subtle per-image warping as if pasted and slightly wrinkled.
                   Randomised per image using image_id as seed (reproducible).
                   Intensity 0-100 maps to displacement 0-8px at thumbnail scale.
  Surface Lighting — Directional light as if composite sits on a surface.
                   Shadow/highlight variation simulates physical depth.
  Combined       — Both. Maximum physical appearance.

All effects applied to the final composite PIL Image, not to individual tiles.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────

def apply_surface(
    composite: Image.Image,
    effect: str,
    intensity: float,   # 0.0 – 1.0
) -> Image.Image:
    """
    Apply surface texture effect to a composite PIL Image.
    Returns a new Image. Original untouched.
    effect: "None" | "Random Curves" | "Surface Lighting" | "Combined"
    intensity: 0.0 (none) to 1.0 (full)
    """
    if effect == "None" or intensity <= 0:
        return composite

    result = composite.convert("RGB")

    if effect in ("Random Curves", "Combined"):
        result = _apply_random_curves(result, intensity)
    if effect in ("Surface Lighting", "Combined"):
        result = _apply_surface_lighting(result, intensity)

    return result


def apply_per_tile_warp(
    tile: Image.Image,
    image_id: str,
    intensity: float,
) -> Image.Image:
    """
    Apply random curve warping to an individual tile before compositing.
    Uses image_id as RNG seed so the warp is reproducible across sessions.
    """
    if intensity <= 0:
        return tile
    seed = int(hashlib.sha1(image_id.encode()).hexdigest()[:8], 16)
    return _warp_image(tile, seed, intensity)


# ── Random curves warping ──────────────────────────────────────────────────────

def _warp_image(img: Image.Image, seed: int, intensity: float) -> Image.Image:
    """
    Warp an image using a smooth random displacement field.
    Produces the slight wrinkling of a photograph pasted on a surface.
    """
    rng = np.random.default_rng(seed)
    w, h = img.size
    arr = np.array(img, dtype=np.uint8)

    # Maximum displacement in pixels — scales with image size and intensity
    max_disp = max(2, min(w, h) * 0.015 * intensity)

    # Generate low-res smooth displacement, upscale
    grid_w, grid_h = max(4, w // 20), max(4, h // 20)
    dx_small = rng.uniform(-max_disp, max_disp, (grid_h, grid_w)).astype(np.float32)
    dy_small = rng.uniform(-max_disp, max_disp, (grid_h, grid_w)).astype(np.float32)

    # Upscale displacement maps to image size using PIL (avoids scipy dependency)
    dx_img = Image.fromarray(
        ((dx_small + max_disp) / (2 * max_disp) * 255).clip(0, 255).astype(np.uint8)
    ).resize((w, h), Image.BILINEAR)
    dy_img = Image.fromarray(
        ((dy_small + max_disp) / (2 * max_disp) * 255).clip(0, 255).astype(np.uint8)
    ).resize((w, h), Image.BILINEAR)

    dx = (np.array(dx_img, dtype=np.float32) / 255.0 * 2 - 1) * max_disp
    dy = (np.array(dy_img, dtype=np.float32) / 255.0 * 2 - 1) * max_disp

    # Build remap coordinates
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    src_x = np.clip(x_coords + dx, 0, w - 1).astype(np.float32)
    src_y = np.clip(y_coords + dy, 0, h - 1).astype(np.float32)

    # Apply displacement using bilinear sampling
    warped = _bilinear_sample(arr, src_x, src_y)
    return Image.fromarray(warped)


def _apply_random_curves(composite: Image.Image, intensity: float) -> Image.Image:
    """
    Apply subtle global warping to the whole composite — different from
    per-tile warping, this simulates the entire surface being slightly uneven.
    """
    seed = 42   # global composite warp is always the same seed
    return _warp_image(composite, seed, intensity * 0.4)   # gentler on the composite


def _bilinear_sample(arr: np.ndarray, src_x: np.ndarray, src_y: np.ndarray) -> np.ndarray:
    """Bilinear interpolation sampling of arr at (src_x, src_y) coordinates."""
    h, w = arr.shape[:2]
    x0 = np.floor(src_x).astype(np.int32).clip(0, w - 2)
    y0 = np.floor(src_y).astype(np.int32).clip(0, h - 2)
    x1 = x0 + 1
    y1 = y0 + 1

    fx = (src_x - x0)[..., np.newaxis]
    fy = (src_y - y0)[..., np.newaxis]

    c00 = arr[y0, x0].astype(np.float32)
    c10 = arr[y0, x1].astype(np.float32)
    c01 = arr[y1, x0].astype(np.float32)
    c11 = arr[y1, x1].astype(np.float32)

    result = (c00 * (1 - fx) * (1 - fy) +
              c10 * fx * (1 - fy) +
              c01 * (1 - fx) * fy +
              c11 * fx * fy)

    return result.clip(0, 255).astype(np.uint8)


# ── Surface lighting ───────────────────────────────────────────────────────────

def _apply_surface_lighting(composite: Image.Image, intensity: float) -> Image.Image:
    """
    Simulate angled directional light hitting a surface on which the
    photographs are laid. Creates subtle shadow/highlight variation that
    reads as physical depth without screaming 'photoshop effect'.

    Light comes from upper-left (classic studio / window light angle).
    Shadows fall toward lower-right.
    """
    w, h = composite.size
    arr = np.array(composite, dtype=np.float32)

    # Build directional gradient — light from upper-left
    y_idx, x_idx = np.mgrid[0:h, 0:w]

    # Normalise 0-1
    norm_x = x_idx / max(w - 1, 1)
    norm_y = y_idx / max(h - 1, 1)

    # Light gradient: bright upper-left, darker lower-right
    gradient = 1.0 - (norm_x * 0.5 + norm_y * 0.5)  # 1.0 top-left → 0.5 bottom-right

    # Scale to subtle range — never pure black, never blown out
    light_strength = intensity * 0.18   # max ±18% brightness shift at full intensity
    lighting = 1.0 + (gradient - 0.75) * light_strength   # centred around 0.75

    # Add very subtle ambient occlusion at corners (darkens all 4 corners)
    corner_dist = np.sqrt((norm_x - 0.5) ** 2 + (norm_y - 0.5) ** 2)
    ao = 1.0 - corner_dist * intensity * 0.12

    final_mult = (lighting * ao)[..., np.newaxis]
    result = (arr * final_mult).clip(0, 255).astype(np.uint8)

    return Image.fromarray(result)
