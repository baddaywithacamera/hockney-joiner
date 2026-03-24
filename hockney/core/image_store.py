"""
image_store.py — Tiered image cache with scratch disk backend.

Three tiers:
  Tier 1 (RAM):         300px thumbnails for Tray View. Whole set, always live.
  Tier 2 (Scratch):     ~1500px medium previews. LRU cache. Swapped per viewport.
  Tier 3 (Source disk): Full-resolution originals. Never touched until export.

With 1,000 images at ~300px thumbnails (JPEG ~15KB each), the RAM footprint
for the full thumbnail set is around 15MB. Manageable.

Medium previews at 1500px (~200KB each) for 1,000 images would be 200MB on
the scratch disk — again, fine. We only keep a window of them decoded in RAM.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

THUMB_LONG_EDGE = 300        # px — Tier 1, always in RAM
PREVIEW_LONG_EDGE = 1500     # px — Tier 2, on scratch disk / LRU in RAM
PROCESS_LONG_EDGE = 300      # px — same as thumbnail, used for LightGlue input
MEDIUM_RAM_CACHE_SIZE = 64   # how many medium previews to keep decoded in RAM

SESSION_DIR_PREFIX = "hj_session_"
THUMB_DIR = "thumbs"
PREVIEW_DIR = "previews"
META_FILE = "image_store_meta.json"

# Supported formats (rawpy formats added dynamically if rawpy available)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

try:
    import rawpy  # noqa: F401
    SUPPORTED_EXTENSIONS |= {
        ".raw", ".cr2", ".cr3", ".nef", ".arw", ".orf", ".rw2", ".dng", ".raf"
    }
    RAW_AVAILABLE = True
except ImportError:
    RAW_AVAILABLE = False

# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class ImageRecord:
    """Everything the app knows about one source image."""
    id: str                        # stable SHA-1 of the original file path
    source_path: Path              # original file on user's disk, never modified
    thumb_path: Path               # Tier 1 — 300px JPEG on scratch disk
    preview_path: Path             # Tier 2 — 1500px JPEG on scratch disk
    width: int = 0                 # original pixel dimensions
    height: int = 0
    file_size: int = 0
    loaded_at: float = field(default_factory=time.time)
    is_raw: bool = False

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "source_path": str(self.source_path),
            "thumb_path": str(self.thumb_path),
            "preview_path": str(self.preview_path),
            "width": self.width,
            "height": self.height,
            "file_size": self.file_size,
            "is_raw": self.is_raw,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImageRecord":
        return cls(
            id=d["id"],
            source_path=Path(d["source_path"]),
            thumb_path=Path(d["thumb_path"]),
            preview_path=Path(d["preview_path"]),
            width=d.get("width", 0),
            height=d.get("height", 0),
            file_size=d.get("file_size", 0),
            is_raw=d.get("is_raw", False),
        )


# ── Scratch disk session ───────────────────────────────────────────────────────

class ScratchSession:
    """
    Manages a temporary working directory on the user-chosen scratch disk.
    Created fresh each application session. Cleaned up on exit (or left for
    crash recovery if cleanup fails).
    """

    def __init__(self, scratch_root: Path):
        self.scratch_root = scratch_root
        self.session_dir = scratch_root / f"{SESSION_DIR_PREFIX}{int(time.time())}"
        self.thumb_dir = self.session_dir / THUMB_DIR
        self.preview_dir = self.session_dir / PREVIEW_DIR
        self._create_dirs()

    def _create_dirs(self):
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.preview_dir.mkdir(parents=True, exist_ok=True)
        log.info("Scratch session: %s", self.session_dir)

    def thumb_path_for(self, image_id: str) -> Path:
        return self.thumb_dir / f"{image_id}.jpg"

    def preview_path_for(self, image_id: str) -> Path:
        return self.preview_dir / f"{image_id}.jpg"

    def cleanup(self):
        try:
            shutil.rmtree(self.session_dir)
            log.info("Scratch session cleaned up: %s", self.session_dir)
        except Exception as e:
            log.warning("Could not clean up scratch session %s: %s", self.session_dir, e)

    def available_bytes(self) -> int:
        """Free space on the scratch disk in bytes."""
        return shutil.disk_usage(self.scratch_root).free

    def used_bytes(self) -> int:
        total = 0
        for p in self.session_dir.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        return total


# ── LRU RAM cache for decoded medium previews ──────────────────────────────────

class LRUImageCache:
    """
    Keeps the N most-recently-accessed medium previews decoded in RAM as
    numpy arrays. Evicts the least-recently-used when full.
    """

    def __init__(self, maxsize: int = MEDIUM_RAM_CACHE_SIZE):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.maxsize = maxsize

    def get(self, image_id: str) -> Optional[np.ndarray]:
        if image_id not in self._cache:
            return None
        self._cache.move_to_end(image_id)
        return self._cache[image_id]

    def put(self, image_id: str, arr: np.ndarray):
        if image_id in self._cache:
            self._cache.move_to_end(image_id)
        self._cache[image_id] = arr
        if len(self._cache) > self.maxsize:
            evicted_id, _ = self._cache.popitem(last=False)
            log.debug("LRU evicted: %s", evicted_id)

    def invalidate(self, image_id: str):
        self._cache.pop(image_id, None)

    def clear(self):
        self._cache.clear()

    def __len__(self):
        return len(self._cache)


# ── Main image store ───────────────────────────────────────────────────────────

class ImageStore:
    """
    Central registry and cache for all images in a session.

    Usage:
        store = ImageStore(scratch_session)
        records = store.load_folder(Path("/path/to/photos"))
        thumb_arr = store.get_thumbnail(record.id)    # always fast, always in RAM
        preview_arr = store.get_preview(record.id)    # fast if cached, disk read if not
        full_pil = store.get_full_res(record.id)      # slow — reads original from disk
    """

    def __init__(self, session: ScratchSession):
        self.session = session
        self._records: dict[str, ImageRecord] = {}   # id → record
        self._thumb_cache: dict[str, np.ndarray] = {}  # Tier 1: always populated
        self._preview_lru = LRUImageCache(MEDIUM_RAM_CACHE_SIZE)  # Tier 2: LRU

    # ── Loading ────────────────────────────────────────────────────────────────

    def load_folder(self, folder: Path) -> list[ImageRecord]:
        """
        Discover all supported images in folder (non-recursive by default).
        Generate thumbnails and medium previews on scratch disk.
        Returns list of ImageRecord in the order they were found.
        """
        paths = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        log.info("Found %d images in %s", len(paths), folder)
        records = []
        for path in paths:
            record = self._load_single(path)
            if record:
                records.append(record)
        return records

    def load_files(self, paths: list[Path]) -> list[ImageRecord]:
        """Load an explicit list of files (e.g. drag-and-drop)."""
        records = []
        for path in paths:
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                record = self._load_single(path)
                if record:
                    records.append(record)
            else:
                log.warning("Skipping unsupported file: %s", path)
        return records

    def _load_single(self, source_path: Path) -> Optional[ImageRecord]:
        image_id = _make_id(source_path)

        # Already loaded this session?
        if image_id in self._records:
            return self._records[image_id]

        try:
            pil_full = _open_image(source_path)
        except Exception as e:
            log.error("Failed to open %s: %s", source_path, e)
            return None

        w, h = pil_full.size
        thumb_path = self.session.thumb_path_for(image_id)
        preview_path = self.session.preview_path_for(image_id)

        # Generate and save thumbnails to scratch disk
        thumb_pil = _resize_long_edge(pil_full, THUMB_LONG_EDGE)
        thumb_pil.save(thumb_path, format="JPEG", quality=85, optimize=True)

        preview_pil = _resize_long_edge(pil_full, PREVIEW_LONG_EDGE)
        preview_pil.save(preview_path, format="JPEG", quality=90, optimize=True)

        # Cache thumbnail in RAM immediately (Tier 1)
        self._thumb_cache[image_id] = np.array(thumb_pil.convert("RGB"))

        record = ImageRecord(
            id=image_id,
            source_path=source_path,
            thumb_path=thumb_path,
            preview_path=preview_path,
            width=w,
            height=h,
            file_size=source_path.stat().st_size,
            is_raw=source_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"},
        )
        self._records[image_id] = record
        log.debug("Loaded: %s (%dx%d)", source_path.name, w, h)
        return record

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def get_thumbnail(self, image_id: str) -> Optional[np.ndarray]:
        """
        Tier 1: 300px thumbnail as numpy RGB array.
        Always in RAM after load. Returns None only if id is unknown.
        """
        return self._thumb_cache.get(image_id)

    def get_preview(self, image_id: str) -> Optional[np.ndarray]:
        """
        Tier 2: 1500px medium preview as numpy RGB array.
        Served from RAM LRU if hot; loaded from scratch disk if cold.
        """
        arr = self._preview_lru.get(image_id)
        if arr is not None:
            return arr

        record = self._records.get(image_id)
        if not record or not record.preview_path.exists():
            return None

        arr = np.array(Image.open(record.preview_path).convert("RGB"))
        self._preview_lru.put(image_id, arr)
        return arr

    def get_full_res(self, image_id: str):
        """
        Tier 3: Full-resolution PIL Image from original source file.
        Slow. Only call at export time. Never cache — too large.
        """
        record = self._records.get(image_id)
        if not record:
            return None
        return _open_image(record.source_path)

    def get_process_array(self, image_id: str) -> Optional[np.ndarray]:
        """
        300px numpy array for LightGlue feature extraction.
        Same resolution as thumbnail — reuse Tier 1 cache.
        """
        return self.get_thumbnail(image_id)

    def get_record(self, image_id: str) -> Optional[ImageRecord]:
        return self._records.get(image_id)

    def all_records(self) -> list[ImageRecord]:
        return list(self._records.values())

    def count(self) -> int:
        return len(self._records)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save_meta(self, path: Path):
        """Save record metadata to JSON (for project files / crash recovery)."""
        data = [r.as_dict() for r in self._records.values()]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_meta(self, path: Path):
        """Restore records from saved metadata. Does not reload images from disk."""
        data = json.loads(path.read_text(encoding="utf-8"))
        for d in data:
            record = ImageRecord.from_dict(d)
            self._records[record.id] = record
            # Reload thumbnail into RAM from scratch disk
            if record.thumb_path.exists():
                self._thumb_cache[record.id] = np.array(
                    Image.open(record.thumb_path).convert("RGB")
                )

    # ── Cleanup ────────────────────────────────────────────────────────────────

    def clear(self):
        self._records.clear()
        self._thumb_cache.clear()
        self._preview_lru.clear()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_id(path: Path) -> str:
    """Stable ID from the file's absolute path. SHA-1 hex, truncated to 12 chars."""
    return hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:12]


def _resize_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    """Resize so the longest dimension equals long_edge. Preserves aspect ratio."""
    w, h = img.size
    if max(w, h) <= long_edge:
        return img.copy()
    scale = long_edge / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _open_image(path: Path) -> Image.Image:
    """Open an image file. Uses rawpy for RAW formats if available."""
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
        img = Image.open(path)
        img = ImageOps.exif_transpose(img)  # honour EXIF rotation
        return img.convert("RGB")

    if RAW_AVAILABLE:
        import rawpy
        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, output_bps=8)
        return Image.fromarray(rgb)

    raise ValueError(f"RAW format not supported without rawpy: {path.suffix}")


def choose_scratch_disk_default() -> Path:
    """
    Fallback scratch disk location if the user hasn't chosen one yet.
    Uses the system temp directory — works on Windows and Linux.
    """
    import tempfile
    return Path(tempfile.gettempdir()) / "hockney-joiner"


def check_scratch_disk_space(scratch_root: Path, n_images: int) -> tuple[bool, str]:
    """
    Estimate whether the scratch disk has enough space.
    Rule of thumb: 1MB per image (thumbnails + previews).
    Returns (ok: bool, message: str).
    """
    required = n_images * 1 * 1024 * 1024  # 1MB per image estimate
    try:
        free = shutil.disk_usage(scratch_root).free
    except Exception:
        return True, "Could not check disk space."

    if free < required:
        needed_mb = required / (1024 * 1024)
        free_mb = free / (1024 * 1024)
        return False, (
            f"Scratch disk may be too full. "
            f"Estimated need: {needed_mb:.0f} MB, available: {free_mb:.0f} MB."
        )
    return True, "OK"
