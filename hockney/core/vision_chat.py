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

VQA_MODEL_ID = "Salesforce/blip-vqa-capfilt-large"
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

    def __init__(self, image, question: str, models_dir: Path, n_images: int = 0):
        """
        image: PIL Image (the contact sheet or composite)
        question: plain language question from the user
        models_dir: location of the local BLIP cache (used only for the local path)
        n_images: number of tiles on the contact sheet, used to bound/validate
                  any tile indices the model returns. 0 means "unknown".
        """
        super().__init__()
        self._image = image
        self._question = question
        self._models_dir = models_dir
        self._n_images = n_images

    def run(self):
        from hockney.core.vision_provider import get_provider, ProviderError

        # Cloud path (Claude / Gemini) when one is selected & configured.
        provider = None
        try:
            provider = get_provider()
        except Exception as e:  # config resolution should never hard-fail
            log.warning("Provider resolution failed, falling back to local: %s", e)

        if provider is not None:
            try:
                result = provider.query(self._image, self._question, self._n_images)
            except ProviderError as e:
                self.error.emit(str(e))
                return
            except Exception as e:  # noqa: BLE001 - surface anything unexpected to the UI
                self.error.emit(f"{provider.name.title()} query failed: {e}")
                return
            log.info("%s answer: %s", provider.name, result["answer"][:120])
            self.finished.emit(result["answer"])
            if result["indices"]:
                self.indices.emit(result["indices"])
            return

        # Local offline path (BLIP-VQA).
        self._run_local()

    def _run_local(self):
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

        found = _extract_indices(answer, self._n_images)
        if found:
            self.indices.emit(found)


def _extract_indices(text: str, n_images: int = 0) -> list[int]:
    """
    Parse image index numbers from the local model's response, bounded by the
    number of tiles so stray numbers (years, f-stops) aren't treated as tiles.
    """
    bound = n_images if n_images and n_images > 0 else 9999
    numbers = re.findall(r"\b(\d+)\b", text)
    return sorted({int(n) for n in numbers if 1 <= int(n) <= bound})


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
