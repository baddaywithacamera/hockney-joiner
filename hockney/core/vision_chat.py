"""
vision_chat.py — Moondream2 vision-language interface.

Moondream2 is a small (1.6B param, ~1.7GB) vision-language model that
runs fully offline. It analyses images and answers questions about them
in plain language.

In the Hockney Joiner context it does two jobs:
  1. Answer open questions about the current composite or individual images.
  2. Return structured image index lists when asked about redundancy or quality,
     so the tray view can highlight flagged images for the photographer to review.

The photographer always decides. Moondream advises.

Download: triggered once via installer/moondream_fetch.py.
Runtime:  fully offline, runs on CPU. GPU if available.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-06-21"   # latest stable release tag
MOONDREAM_FILENAME_GZ = "moondream-2b-int8.mf.gz"   # compressed on HF
MOONDREAM_FILENAME    = "moondream-2b-int8.mf"       # decompressed local name
READY_MARKER = "moondream_ready"


def is_moondream_ready(models_dir: Path) -> bool:
    return (models_dir / READY_MARKER).exists()


# ── Async query worker ─────────────────────────────────────────────────────────

class VisionQueryWorker(QThread):
    """
    Background thread: sends an image + question to moondream, returns answer.
    The answer may contain image index references like "images 3, 7, and 12"
    which the caller can parse and use to highlight tiles in the tray view.
    """

    finished = pyqtSignal(str)       # plain text answer
    indices = pyqtSignal(list)       # list[int] of 1-based image indices mentioned
    error = pyqtSignal(str)

    def __init__(self, image, question: str, models_dir: Path):
        """
        image: PIL Image (the contact sheet or composite)
        question: plain language question from the user
        """
        super().__init__()
        self._image = image
        self._question = question
        self._models_dir = models_dir

    def run(self):
        try:
            import moondream as md
        except ImportError:
            log.info("moondream package missing — installing automatically…")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "moondream"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                import moondream as md  # noqa: F811
            except Exception as install_err:
                self.error.emit(
                    f"Could not install moondream package automatically.\n"
                    f"Please run:  pip install moondream\n\nDetail: {install_err}"
                )
                return

        try:
            model_path = str(self._models_dir / MOONDREAM_FILENAME)
            model = md.vl(model=model_path)
            encoded = model.encode_image(self._image)
            answer = model.query(encoded, self._question)["answer"]
        except Exception as e:
            self.error.emit(f"Moondream query failed: {e}")
            return

        log.info("Moondream answer: %s", answer[:120])
        self.finished.emit(answer)

        # Extract any image index numbers mentioned in the response
        found = _extract_indices(answer)
        if found:
            self.indices.emit(found)


def _extract_indices(text: str) -> list[int]:
    """
    Parse image index numbers from moondream's response.
    Handles: "images 3, 7 and 12", "image 4", "#5", "number 2", etc.
    Returns sorted list of 1-based integers.
    """
    numbers = re.findall(r"\b(\d+)\b", text)
    indices = sorted(set(int(n) for n in numbers if 1 <= int(n) <= 9999))
    return indices


# ── Moondream download worker ──────────────────────────────────────────────────

class MoondreamDownloadWorker(QThread):
    """
    Downloads the moondream2 model file (~1.7GB) using the moondream
    Python package's built-in download utility.
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, models_dir: Path):
        super().__init__()
        self.models_dir = models_dir

    def run(self):
        self.progress.emit(5)
        try:
            import moondream as md  # noqa: F401
        except ImportError:
            log.info("moondream package missing — installing automatically…")
            self.progress.emit(8)
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", "moondream"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except Exception as install_err:
                self.error.emit(
                    f"Could not install moondream package automatically.\n"
                    f"Please run:  pip install moondream\n\nDetail: {install_err}"
                )
                return
            log.info("moondream package installed.")

        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / MOONDREAM_FILENAME

        if model_path.exists():
            log.info("Moondream model already present.")
            self.progress.emit(100)
            (self.models_dir / READY_MARKER).write_text("ok")
            self.finished.emit()
            return

        try:
            import gzip
            import shutil

            log.info("Downloading moondream2 (~1.7 GB compressed)…")
            self.progress.emit(10)

            from huggingface_hub import hf_hub_download
            gz_path = hf_hub_download(
                repo_id=MOONDREAM_MODEL_ID,
                filename=MOONDREAM_FILENAME_GZ,
                revision=MOONDREAM_REVISION,
                local_dir=str(self.models_dir),
            )
            log.info("Downloaded compressed model to: %s", gz_path)
            self.progress.emit(80)

            # Decompress .mf.gz → .mf
            log.info("Decompressing model…")
            with gzip.open(gz_path, "rb") as f_in, open(model_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            Path(gz_path).unlink(missing_ok=True)   # remove the .gz to save space
            log.info("Decompressed to: %s", model_path)
            self.progress.emit(95)

        except Exception as e:
            self.error.emit(f"Download failed: {e}")
            return

        (self.models_dir / READY_MARKER).write_text("ok")
        self.progress.emit(100)
        self.finished.emit()
