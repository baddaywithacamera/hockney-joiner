"""
vision_chat.py — Moondream2 vision-language interface.

Moondream2 is a small (1.6B param, ~2GB) vision-language model that
runs fully offline. It analyses images and answers questions about them
in plain language.

In the Hockney Joiner context it does two jobs:
  1. Answer open questions about the current composite or individual images.
  2. Return structured image index lists when asked about redundancy or quality,
     so the tray view can highlight flagged images for the photographer to review.

The photographer always decides. Moondream advises.

Backend: transformers + vikhyatk/moondream2 (standard HuggingFace safetensors).
Revision 2024-08-26 — last stable release using only PIL (no pyvips/libvips).
No proprietary file formats, no API keys, no cloud connection at runtime.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

MOONDREAM_MODEL_ID = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2024-08-26"   # last stable release; PIL only, no pyvips
READY_MARKER = "moondream_ready"


def is_moondream_ready(models_dir: Path) -> bool:
    return (models_dir / READY_MARKER).exists()


def _ensure_deps() -> bool:
    """Install transformers + einops if missing. Returns True on success."""
    missing = []
    for pkg, import_name in [
        ("transformers", "transformers"),
        ("einops", "einops"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return True

    log.info("Installing missing packages: %s", missing)
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install"] + missing,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except Exception as e:
        log.error("Failed to install packages %s: %s", missing, e)
        return False


def _load_model(models_dir: Path):
    """Load moondream2 from the local cache, returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = str(models_dir)
    model = AutoModelForCausalLM.from_pretrained(
        MOONDREAM_MODEL_ID,
        revision=MOONDREAM_REVISION,
        trust_remote_code=True,
        cache_dir=cache_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MOONDREAM_MODEL_ID,
        revision=MOONDREAM_REVISION,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model.eval()
    return model, tokenizer


# ── Async query worker ─────────────────────────────────────────────────────────

class VisionQueryWorker(QThread):
    """
    Background thread: sends an image + question to moondream2, returns answer.
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
        if not _ensure_deps():
            self.error.emit(
                "Could not install required packages.\n"
                "Please run:  pip install transformers einops"
            )
            return

        try:
            model, tokenizer = _load_model(self._models_dir)
            enc_image = model.encode_image(self._image)
            answer = model.answer_question(enc_image, self._question, tokenizer)
        except Exception as e:
            self.error.emit(f"Moondream query failed: {e}")
            return

        log.info("Moondream answer: %s", answer[:120])
        self.finished.emit(answer)

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
    Downloads moondream2 weights from HuggingFace using the transformers cache.
    Uses standard safetensors format — no custom file formats, no pyvips.
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, models_dir: Path):
        super().__init__()
        self.models_dir = models_dir

    def run(self):
        self.progress.emit(5)

        if not _ensure_deps():
            self.error.emit(
                "Could not install required packages.\n"
                "Please run:  pip install transformers einops"
            )
            return

        self.progress.emit(10)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Check if already downloaded
        if is_moondream_ready(self.models_dir):
            log.info("Moondream already downloaded.")
            self.progress.emit(100)
            self.finished.emit()
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            log.info("Downloading moondream2 weights (~2 GB)…")
            self.progress.emit(15)

            # Download tokenizer first (small)
            AutoTokenizer.from_pretrained(
                MOONDREAM_MODEL_ID,
                revision=MOONDREAM_REVISION,
                trust_remote_code=True,
                cache_dir=str(self.models_dir),
            )
            self.progress.emit(30)

            # Download model weights (the big one)
            AutoModelForCausalLM.from_pretrained(
                MOONDREAM_MODEL_ID,
                revision=MOONDREAM_REVISION,
                trust_remote_code=True,
                cache_dir=str(self.models_dir),
                torch_dtype=torch.float32,
            )
            self.progress.emit(95)

        except Exception as e:
            self.error.emit(f"Download failed: {e}")
            return

        # Write ready marker
        (self.models_dir / READY_MARKER).write_text("ok")
        self.progress.emit(100)
        self.finished.emit()
