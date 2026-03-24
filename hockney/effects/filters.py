"""
filters.py — Algorithmic Instagram-style filters.

All 13 filters from the spec: Lo-Fi, Clarendon, Juno, Lark, Ludwig,
Perpetua, Rise, Slumber, Toaster, Valencia, Walden, Willow, Xpro2.

Implemented as pure PIL pipelines — no LUT files, no extra dependencies.
Each filter is a composition of: brightness, contrast, saturation,
colour temperature, curve adjustments, and optional vignette.

Applied uniformly across all images — not per-image.
Applied to 300px previews for placement, then re-applied to full-res at export.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter

# ── Public API ─────────────────────────────────────────────────────────────────

def apply_filter(img: Image.Image, name: str) -> Image.Image:
    """Apply a named filter. Returns a new Image. Original untouched."""
    if name == "None" or name not in FILTERS:
        return img
    return FILTERS[name](img.convert("RGB"))


def filter_names() -> list[str]:
    return ["None"] + sorted(FILTERS.keys())


# ── Curve helpers ──────────────────────────────────────────────────────────────

def _curve(img: Image.Image, r_pts, g_pts, b_pts) -> Image.Image:
    """
    Apply per-channel tone curves. Points are (input, output) pairs 0-255.
    Interpolates a lookup table from the control points.
    """
    def _lut(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        lut = np.interp(np.arange(256), xs, ys).clip(0, 255).astype(np.uint8)
        return lut.tolist()

    r_lut = _lut(r_pts)
    g_lut = _lut(g_pts)
    b_lut = _lut(b_pts)

    lut = r_lut + g_lut + b_lut
    return img.point(lut)


def _vignette(img: Image.Image, strength: float = 0.4) -> Image.Image:
    """Darken edges with a radial gradient. Strength 0-1."""
    w, h = img.size
    arr = np.array(img, dtype=np.float32)

    cx, cy = w / 2, h / 2
    y_idx, x_idx = np.mgrid[0:h, 0:w]
    dist = np.sqrt(((x_idx - cx) / cx) ** 2 + ((y_idx - cy) / cy) ** 2)
    vignette = 1.0 - dist * strength
    vignette = np.clip(vignette, 0.0, 1.0)[..., np.newaxis]

    arr = (arr * vignette).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _warm(img: Image.Image, amount: float = 0.15) -> Image.Image:
    """Shift colour temperature warmer. Amount 0-1."""
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 + amount), 0, 255)   # R up
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 - amount), 0, 255)   # B down
    return Image.fromarray(arr.astype(np.uint8))


def _cool(img: Image.Image, amount: float = 0.15) -> Image.Image:
    """Shift colour temperature cooler."""
    arr = np.array(img, dtype=np.float32)
    arr[:, :, 0] = np.clip(arr[:, :, 0] * (1 - amount), 0, 255)
    arr[:, :, 2] = np.clip(arr[:, :, 2] * (1 + amount), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _fade(img: Image.Image, amount: float = 0.15) -> Image.Image:
    """Lift shadows / reduce overall contrast for a faded film look."""
    arr = np.array(img, dtype=np.float32)
    arr = arr * (1 - amount) + 255 * amount * 0.3
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _desaturate(img: Image.Image, amount: float = 0.4) -> Image.Image:
    """Partially desaturate toward greyscale."""
    grey = img.convert("L").convert("RGB")
    return Image.blend(img, grey, amount)


# ── Filter definitions ─────────────────────────────────────────────────────────

def _lofi(img):
    img = ImageEnhance.Color(img).enhance(1.9)
    img = ImageEnhance.Contrast(img).enhance(1.5)
    img = _vignette(img, 0.55)
    return img

def _clarendon(img):
    img = _cool(img, 0.08)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Color(img).enhance(1.3)
    img = _curve(img,
        r_pts=[(0,0),(85,70),(170,180),(255,255)],
        g_pts=[(0,0),(85,80),(170,175),(255,255)],
        b_pts=[(0,20),(85,90),(170,180),(255,245)],
    )
    return img

def _juno(img):
    img = _warm(img, 0.12)
    img = ImageEnhance.Color(img).enhance(1.2)
    img = _fade(img, 0.08)
    img = _curve(img,
        r_pts=[(0,10),(128,138),(255,255)],
        g_pts=[(0,0),(128,125),(255,250)],
        b_pts=[(0,0),(128,120),(255,240)],
    )
    return img

def _lark(img):
    img = _cool(img, 0.05)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = _fade(img, 0.1)
    # Desaturate reds slightly
    arr = np.array(img, dtype=np.float32)
    hsv_r_mask = (arr[:,:,0] > arr[:,:,1]) & (arr[:,:,0] > arr[:,:,2])
    arr[hsv_r_mask, 1] = arr[hsv_r_mask, 1] * 1.05
    img = Image.fromarray(arr.clip(0,255).astype(np.uint8))
    return img

def _ludwig(img):
    img = _warm(img, 0.07)
    img = _fade(img, 0.12)
    img = ImageEnhance.Contrast(img).enhance(0.9)
    return img

def _perpetua(img):
    img = _cool(img, 0.12)
    arr = np.array(img, dtype=np.float32)
    arr[:,:,1] = np.clip(arr[:,:,1] * 1.04, 0, 255)  # slight green
    img = Image.fromarray(arr.astype(np.uint8))
    img = _fade(img, 0.08)
    return img

def _rise(img):
    img = _warm(img, 0.2)
    img = _fade(img, 0.2)
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = _vignette(img, 0.2)
    return img

def _slumber(img):
    img = _desaturate(img, 0.35)
    img = _cool(img, 0.08)
    img = _fade(img, 0.18)
    img = ImageEnhance.Contrast(img).enhance(0.85)
    return img

def _toaster(img):
    img = _warm(img, 0.25)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = _vignette(img, 0.6)
    img = _curve(img,
        r_pts=[(0,20),(128,148),(255,255)],
        g_pts=[(0,0),(128,118),(255,230)],
        b_pts=[(0,0),(128,100),(255,200)],
    )
    return img

def _valencia(img):
    img = _warm(img, 0.1)
    img = _fade(img, 0.15)
    img = ImageEnhance.Color(img).enhance(1.1)
    return img

def _walden(img):
    img = _fade(img, 0.2)
    img = _cool(img, 0.1)
    img = _curve(img,
        r_pts=[(0,0),(128,120),(255,235)],
        g_pts=[(0,5),(128,128),(255,248)],
        b_pts=[(0,20),(128,140),(255,255)],
    )
    return img

def _willow(img):
    img = _desaturate(img, 0.7)
    img = ImageEnhance.Contrast(img).enhance(1.1)
    img = _fade(img, 0.1)
    return img

def _xpro2(img):
    img = _warm(img, 0.05)
    img = ImageEnhance.Contrast(img).enhance(1.6)
    img = ImageEnhance.Color(img).enhance(1.4)
    img = _vignette(img, 0.65)
    img = _curve(img,
        r_pts=[(0,0),(90,100),(200,215),(255,255)],
        g_pts=[(0,0),(90,85),(200,200),(255,245)],
        b_pts=[(0,10),(90,80),(200,190),(255,235)],
    )
    return img


# ── Registry ───────────────────────────────────────────────────────────────────

FILTERS: dict[str, callable] = {
    "Lo-Fi":     _lofi,
    "Clarendon": _clarendon,
    "Juno":      _juno,
    "Lark":      _lark,
    "Ludwig":    _ludwig,
    "Perpetua":  _perpetua,
    "Rise":      _rise,
    "Slumber":   _slumber,
    "Toaster":   _toaster,
    "Valencia":  _valencia,
    "Walden":    _walden,
    "Willow":    _willow,
    "Xpro2":     _xpro2,
}
