"""
placement_log.py — Structured log of placement results for engine comparison.

Appends one JSON record per Auto-Place run to a log file in the project
session directory.  Each record captures: engine used, per-image match stats,
timing, and overall placement rate.  This lets us compare engines and tune
parameters across runs.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)


class PlacementLog:
    """Accumulate stats for one placement run, then flush to disk."""

    def __init__(self, session_dir: Path, project_name: str = ""):
        self._session_dir = session_dir
        self._project_name = project_name
        self._log_path = session_dir / "placement_log.jsonl"
        self._start_time = time.monotonic()
        self._engine = "unknown"
        self._total_images = 0
        self._placed = 0
        self._odds_and_ends = 0
        self._per_image: dict[str, dict] = {}
        self._notes = ""

    def set_engine(self, engine: str):
        self._engine = engine

    def set_counts(self, total: int, placed: int, odds_and_ends: int):
        self._total_images = total
        self._placed = placed
        self._odds_and_ends = odds_and_ends

    def add_image_stats(self, image_id: str, stats: dict):
        """Add per-image match stats (inliers, scale, slot, etc.)."""
        self._per_image[image_id] = stats

    def merge_image_stats(self, stats_dict: dict[str, dict]):
        """Merge a batch of per-image stats from an engine."""
        self._per_image.update(stats_dict)

    def set_notes(self, notes: str):
        self._notes = notes

    def flush(self):
        """Write the record to the log file (JSONL — one JSON object per line)."""
        elapsed = time.monotonic() - self._start_time
        record = {
            "timestamp": datetime.now().isoformat(),
            "project": self._project_name,
            "engine": self._engine,
            "total_images": self._total_images,
            "placed": self._placed,
            "odds_and_ends": self._odds_and_ends,
            "placement_rate": (self._placed / max(self._total_images, 1)) * 100,
            "elapsed_seconds": round(elapsed, 2),
            "per_image": self._per_image,
            "notes": self._notes,
        }

        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
            log.info("Placement log written: %s (%.0f%% placed, %.1fs, engine=%s)",
                     self._log_path.name,
                     record["placement_rate"],
                     elapsed,
                     self._engine)
        except Exception as e:
            log.warning("Failed to write placement log: %s", e)

    @staticmethod
    def read_log(session_dir: Path) -> list[dict]:
        """Read all log records from a session directory."""
        log_path = session_dir / "placement_log.jsonl"
        if not log_path.exists():
            return []
        records = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records
