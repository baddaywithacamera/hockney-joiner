"""
model_fetch.py — Download LightGlue weights.

LightGlue downloads its own weights from GitHub releases on first use.
We trigger that download explicitly here so it happens once, with a
progress dialog, rather than silently mid-session.

Weights land in torch's hub cache (~/.cache/torch/hub/checkpoints/).
Once downloaded, the tool runs fully offline.

Download size:
  SuperPoint extractor:   ~5 MB
  LightGlue matcher:      ~45 MB
  Total:                  ~50 MB
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

READY_MARKER = "lightglue_ready"


class ModelDownloadWorker(QThread):
    """
    Background thread that triggers the LightGlue weight download.
    The lightglue library fetches from GitHub releases and caches in
    ~/.cache/torch/hub/checkpoints/ — we just need to instantiate it.
    """

    progress = pyqtSignal(int)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, models_dir: Path):
        super().__init__()
        self.models_dir = models_dir

    def run(self):
        self.progress.emit(5)

        # Check lightglue is installed
        try:
            import torch
            import lightglue
            from lightglue import LightGlue, SuperPoint
        except ImportError as e:
            self.error.emit(
                f"LightGlue is not installed: {e}\n\n"
                "Run: pip install git+https://github.com/cvg/LightGlue.git"
            )
            return

        self.progress.emit(20)

        # Instantiating SuperPoint + LightGlue triggers weight download
        # if not already cached. This is the download step.
        try:
            log.info("Downloading/verifying SuperPoint weights…")
            device = "cpu"
            extractor = SuperPoint(max_num_keypoints=512).eval().to(device)
            self.progress.emit(60)

            log.info("Downloading/verifying LightGlue weights…")
            matcher = LightGlue(features="superpoint").eval().to(device)
            self.progress.emit(90)

            # Quick sanity check — run on a tiny blank image
            import torch
            dummy = torch.zeros(1, 1, 64, 64).to(device)
            with torch.no_grad():
                feats = extractor.extract(dummy)
            log.info("LightGlue sanity check passed.")

        except Exception as e:
            self.error.emit(f"Model download or initialisation failed:\n{e}")
            return

        # Write ready marker so future launches skip the download prompt
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / READY_MARKER).write_text("ok")

        self.progress.emit(100)
        log.info("LightGlue ready.")
        self.finished.emit()


def is_model_ready(models_dir: Path) -> bool:
    return (models_dir / READY_MARKER).exists()
