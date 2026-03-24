"""
vision_chat.py — BLIP-VQA vision-language interface.

BLIP-VQA-Large is a ~1.2 GB vision-language model by Salesforce that
runs fully offline. It answers questions about images in plain language.

In the Hockney Joiner context it does two jobs:
  1. Answer open questions about the current composite or individual images.
  2. Return structured image index lists when asked about redundancy or quality,
     so the tray view can highlight flagged images for the photographer to review.

The photographer always decides. The AI advises.

Backend: transformers + Salesforce/blip-vqa-large (standard HuggingFace model).
Uses BlipForQuestionAnswering — a first-class transformers class.
No trust_remote_code, no custom code from HuggingFace, no API keys.
Fully offline after the one-time ~1.2 GB download.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

VQA_MODEL_ID = "Salesforce/blip-vqa-large"
READY_MARKER = "vision_ready"


def is_moondream_ready(models_dir: Path) -> bool:
    """Check if the vision model has been downloaded."""
    return (models_dir / READY_MARKER).exists()


def _ensure_deps() -> bool:
    """Install transformers if missing. Returns True on success."""
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        pass

    log.info("Installing transformers…")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "transformers"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except Exception as e:
        log.error("Failed to install transformers: %s", e)
        return False


def _load_model(models_dir: Path):
    """Load BLIP-VQA from the local cache, returns (model, processor)."""
    import torch
    from transformers import BlipForQuestionAnswering, BlipProcessor

    cache_dir = str(models_dir)
    processor = BlipProcessor.from_pretrained(
        VQA_MODEL_ID,
        cache_dir=cache_dir,
        local_files_only=True,
    )
    model = BlipForQuestionAnswering.from_pretrained(
        VQA_MODEL_ID,
        cache_dir=cache_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.eval()
    return model, processor


# ── Async query worker ─────────────────────────────────────────────────────────

class VisionQueryWorker(QThread):
    """
    Background thread: sends an image + question to BLIP-VQA, returns answer.
    The answer may contain image index references like "3, 7, 12"
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
                "Could not install transformers.\n"
                "Please run:  pip install transformers"
            )
            return

        try:
            import torch
            model, processor = _load_model(self._models_dir)
            inputs = processor(self._image, self._question, return_tensors="pt")
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=100)
            answer = processor.decode(output[0], skip_special_tokens=True).strip()
        except Exception as e:
            self.error.emit(f"Vision query failed: {e}")
            return

        log.info("BLIP answer: %s", answer[:120])
        self.finished.emit(answer)

        found = _extract_indices(answer)
        if found:
            self.indices.emit(found)


def _extract_indices(text: str) -> list[int]:
    """
    Parse image index numbers from the model's response.
    Handles: "images 3, 7 and 12", "image 4", "#5", "number 2", etc.
    Returns sorted list of 1-based integers.
    """
    numbers = re.findall(r"\b(\d+)\b", text)
    indices = sorted(set(int(n) for n in numbers if 1 <= int(n) <= 9999))
    return indices


# ── Download worker ───────────────────────────────────────────────────────────

class MoondreamDownloadWorker(QThread):
    """
    Downloads BLIP-VQA-Large weights from HuggingFace.
    Uses standard BlipForQuestionAnswering — no custom code, no trust_remote_code.
    Class name kept as MoondreamDownloadWorker for backward compat with UI wiring.
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
                "Could not install transformers.\n"
                "Please run:  pip install transformers"
            )
            return

        self.progress.emit(10)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        if is_moondream_ready(self.models_dir):
            log.info("Vision model already downloaded.")
            self.progress.emit(100)
            self.finished.emit()
            return

        try:
            import torch
            from transformers import BlipForQuestionAnswering, BlipProcessor

            log.info("Downloading BLIP-VQA-Large (~1.2 GB)…")
            self.progress.emit(15)

            # Download processor (tokenizer + image processor, small)
            BlipProcessor.from_pretrained(
                VQA_MODEL_ID,
                cache_dir=str(self.models_dir),
            )
            self.progress.emit(30)

            # Download model weights
            BlipForQuestionAnswering.from_pretrained(
                VQA_MODEL_ID,
                cache_dir=str(self.models_dir),
                torch_dtype=torch.float32,
            )
            self.progress.emit(95)

        except Exception as e:
            self.error.emit(f"Download failed: {e}")
            return

        (self.models_dir / READY_MARKER).write_text("ok")
        self.progress.emit(100)
        self.finished.emit()
