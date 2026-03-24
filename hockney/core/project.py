"""
project.py — Full project save and load.

A project file is a single JSON document that captures everything needed
to restore a session exactly: which images were loaded, where they are
placed, what processing was applied, and what surface effect is active.

Format is human-readable and version-stamped so future versions of the
tool can handle old project files gracefully.

Schema (v1):
{
  "version": 1,
  "created": "2026-03-23T...",
  "images": [
    {
      "id": "abc123",
      "source_path": "/absolute/path/to/photo.jpg",
      "width": 6000,
      "height": 4000,
      "file_size": 8392847
    }, ...
  ],
  "placements": [
    {
      "image_id": "abc123",
      "x": 120.5,
      "y": 34.0,
      "rotation": -2.3,
      "z_order": 0,
      "auto_x": 120.5,
      "auto_y": 34.0,
      "auto_rotation": -2.3
    }, ...
  ],
  "removed_ids": ["def456", ...],
  "processing": {
    "histogram_eq": false,
    "filter": "None",
    "surface_effect": "None",
    "surface_intensity": 30
  }
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hockney.core.models import ImagePlacement

log = logging.getLogger(__name__)

PROJECT_VERSION = 1


# ── Save ───────────────────────────────────────────────────────────────────────

def save_project(
    path: Path,
    store,                          # ImageStore
    placements: list[ImagePlacement],
    removed_ids: set[str],
    processing: dict[str, Any],
):
    """
    Write a complete project file to path.
    processing dict keys: histogram_eq, filter, surface_effect, surface_intensity.
    """
    doc = {
        "version": PROJECT_VERSION,
        "created": datetime.now(timezone.utc).isoformat(),
        "images": [r.as_dict() for r in store.all_records()],
        "placements": [p.as_dict() for p in placements],
        "removed_ids": list(removed_ids),
        "processing": processing,
    }

    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    log.info("Project saved: %s (%d images, %d placements)",
             path, len(doc["images"]), len(doc["placements"]))


# ── Load ───────────────────────────────────────────────────────────────────────

class ProjectLoadResult:
    def __init__(self):
        self.placements: list[ImagePlacement] = []
        self.removed_ids: set[str] = set()
        self.processing: dict[str, Any] = {}
        self.missing_files: list[str] = []   # source files that couldn't be found
        self.version: int = 0


def load_project(path: Path, store) -> ProjectLoadResult:
    """
    Load a project file. Re-loads images into the store, rebuilds placements.
    Returns a ProjectLoadResult — check missing_files for any images that
    have moved or been deleted since the project was saved.
    """
    result = ProjectLoadResult()

    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("version", 0)
    result.version = version

    if version > PROJECT_VERSION:
        log.warning(
            "Project file version %d is newer than this tool (v%d). "
            "Some features may not restore correctly.",
            version, PROJECT_VERSION,
        )

    # ── Re-load images ─────────────────────────────────────────────────────────
    for img_data in raw.get("images", []):
        source = Path(img_data["source_path"])
        if not source.exists():
            result.missing_files.append(str(source))
            log.warning("Missing source file: %s", source)
            continue

        record = store._load_single(source)
        if record is None:
            result.missing_files.append(str(source))

    # ── Restore placements ─────────────────────────────────────────────────────
    for p_data in raw.get("placements", []):
        p = ImagePlacement(
            image_id=p_data["image_id"],
            x=p_data.get("x", 0.0),
            y=p_data.get("y", 0.0),
            rotation=p_data.get("rotation", 0.0),
            z_order=p_data.get("z_order", 0),
            auto_x=p_data.get("auto_x", p_data.get("x", 0.0)),
            auto_y=p_data.get("auto_y", p_data.get("y", 0.0)),
            auto_rotation=p_data.get("auto_rotation", p_data.get("rotation", 0.0)),
        )
        result.placements.append(p)

    # ── Restore removed set ────────────────────────────────────────────────────
    result.removed_ids = set(raw.get("removed_ids", []))

    # ── Processing settings ────────────────────────────────────────────────────
    result.processing = raw.get("processing", {
        "histogram_eq": False,
        "filter": "None",
        "surface_effect": "None",
        "surface_intensity": 30,
    })

    log.info(
        "Project loaded: %d placements, %d missing files",
        len(result.placements),
        len(result.missing_files),
    )
    return result
