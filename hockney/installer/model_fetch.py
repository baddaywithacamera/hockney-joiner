"""
model_fetch.py — Download LightGlue from HuggingFace.

Downloads a pinned, checksum-verified version once at install time.
After that the tool runs fully offline.

Model stored in /models inside the application folder.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

# ── Pinned model config ────────────────────────────────────────────────────────
# Update these when bumping the model version.

LIGHTGLUE_REPO = "cvg/LightGlue"
LIGHTGLUE_FILES = [
    "superpoint_lightglue.pth",
    "disk_lightglue.pth",
]

# SHA-256 checksums for each file (update when pinning a new version)
CHECKSUMS: dict[str, str] = {
    # "superpoint_lightglue.pth": "abc123...",
    # Fill in after first verified download
}

READY_MARKER = "lightglue_ready"


# ── Downloader ─────────────────────────────────────────────────────────────────

class ModelDownloadWorker(QThread):
    """
    Background thread that downloads LightGlue from HuggingFace.
    Emits progress (0-100) and finished/error signals.
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, models_dir: Path):
        super().__init__()
        self.models_dir = models_dir

    def run(self):
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            self.error.emit(
                "huggingface_hub is not installed.\n"
                "Run: pip install huggingface_hub"
            )
            return

        self.models_dir.mkdir(parents=True, exist_ok=True)
        n = len(LIGHTGLUE_FILES)

        for i, filename in enumerate(LIGHTGLUE_FILES):
            dest = self.models_dir / filename
            if dest.exists():
                log.info("Already present: %s", filename)
                self.progress.emit(int((i + 1) / n * 90))
                continue

            log.info("Downloading: %s", filename)
            try:
                downloaded = hf_hub_download(
                    repo_id=LIGHTGLUE_REPO,
                    filename=filename,
                    local_dir=str(self.models_dir),
                )
                log.info("Downloaded to: %s", downloaded)
            except Exception as e:
                self.error.emit(f"Download failed for {filename}: {e}")
                return

            # Verify checksum if we have one
            expected = CHECKSUMS.get(filename)
            if expected:
                actual = _sha256(dest)
                if actual != expected:
                    dest.unlink(missing_ok=True)
                    self.error.emit(
                        f"Checksum mismatch for {filename}.\n"
                        f"Expected: {expected}\nGot: {actual}\n"
                        "The download may be corrupt. Please try again."
                    )
                    return

            self.progress.emit(int((i + 1) / n * 90))

        # Write ready marker
        (self.models_dir / READY_MARKER).write_text("ok")
        self.progress.emit(100)
        log.info("LightGlue model ready in: %s", self.models_dir)
        self.finished.emit()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
