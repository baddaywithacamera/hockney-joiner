"""
main.py — Entry point for the Hockney Joiner Assembly Tool.

Handles:
  - Scratch disk selection (asked once, remembered in prefs)
  - First-run LightGlue model download prompt
  - First-run moondream2 model download prompt
  - PyQt application startup
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# PyQt6 imports
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from hockney.core.image_store import (
    ScratchSession,
    choose_scratch_disk_default,
    check_scratch_disk_space,
)
from hockney.installer.model_fetch import is_model_ready
from hockney.core.vision_chat import is_moondream_ready
from hockney.ui.main_window import MainWindow

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

APP_NAME = "Hockney Joiner"
APP_ORG = "HockneyJoiner"
APP_VERSION = "0.1.0"


# ── Scratch disk selection ─────────────────────────────────────────────────────

def get_scratch_disk(app: QApplication) -> Path:
    """
    Return the user's chosen scratch disk path.

    Logic:
      1. If a saved path exists and its drive is mounted → use it.
      2. If a saved path exists but the drive is missing → warn and offer to
         wait/replug or pick a different location.
      3. First run (no saved path) → show the selection dialog.
    """
    settings = QSettings(APP_ORG, APP_NAME)
    saved = settings.value("scratch_disk", "")

    if saved:
        saved_path = Path(saved)
        if saved_path.exists():
            log.info("Scratch disk found: %s", saved_path)
            return saved_path

        # Drive is known but not currently present
        drive_label = str(saved_path.anchor) if saved_path.anchor else str(saved_path)
        result = QMessageBox.warning(
            None,
            "Scratch Drive Not Found",
            (
                f"<b>Your scratch drive is not connected.</b><br><br>"
                f"Last used location:<br><code>{saved_path}</code><br><br>"
                f"Please plug in the drive and click <b>Retry</b>, "
                f"or choose a different scratch location."
            ),
            QMessageBox.StandardButton.Retry
            | QMessageBox.StandardButton.Open
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Retry,
        )

        if result == QMessageBox.StandardButton.Retry:
            # Re-check — drive may now be plugged in
            if saved_path.exists():
                log.info("Scratch disk found after retry: %s", saved_path)
                return saved_path
            # Still missing — fall through to picker below
            QMessageBox.information(
                None,
                "Drive Still Not Found",
                "Drive still not detected. Please choose a different scratch location.",
            )
        elif result == QMessageBox.StandardButton.Cancel:
            # Fall back to system temp silently
            fallback = choose_scratch_disk_default()
            log.info("User cancelled scratch selection — using system temp: %s", fallback)
            fallback.mkdir(parents=True, exist_ok=True)
            return fallback
        # StandardButton.Open or retry failed → drop through to picker

    # First run or user wants to pick a new location
    dialog = ScratchDiskDialog()
    if dialog.exec() == QDialog.DialogCode.Accepted:
        chosen = dialog.chosen_path
    else:
        chosen = choose_scratch_disk_default()
        log.info("No scratch disk chosen — using system temp: %s", chosen)

    chosen.mkdir(parents=True, exist_ok=True)
    settings.setValue("scratch_disk", str(chosen))
    return chosen


class ScratchDiskDialog(QDialog):
    """
    First-run dialog asking the user to choose a scratch disk.
    Explains what it's for and how much space to expect.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.chosen_path: Path = choose_scratch_disk_default()
        self.setWindowTitle("Choose Scratch Disk")
        self.setMinimumWidth(480)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("<b>Scratch Disk</b>")
        title.setStyleSheet("font-size: 16px;")
        layout.addWidget(title)

        explanation = QLabel(
            "Hockney Joiner uses a scratch disk to cache preview images while you work.\n\n"
            "For 1,000 photographs, expect roughly 1–2 GB of scratch space.\n"
            "An SSD is strongly recommended — a spinning hard disk will feel slow.\n\n"
            "This folder is cleaned up automatically when you quit the application.\n"
            "Choose a drive with plenty of free space."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self._path_label = QLabel(f"Current selection: {self.chosen_path}")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("color: #555; font-size: 11px;")
        layout.addWidget(self._path_label)

        browse_btn = QPushButton("Browse…")
        browse_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        browse_btn.clicked.connect(self._browse)
        layout.addWidget(browse_btn)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Scratch Disk Folder",
            str(self.chosen_path),
        )
        if chosen:
            self.chosen_path = Path(chosen)
            self._path_label.setText(f"Current selection: {self.chosen_path}")


def prompt_moondream_download(parent: QWidget | None = None) -> bool:
    """
    Ask the user whether to download moondream2 for the composition chat panel.
    Returns True if they want to proceed.
    """
    result = QMessageBox.question(
        parent,
        "Composition Chat Model",
        (
            "<b>moondream2 vision model not found.</b><br><br>"
            "Hockney Joiner includes a chat panel that lets you ask questions about "
            "your composite — things like <i>\"which images look redundant?\"</i> "
            "or <i>\"what's the overall mood?\"</i><br><br>"
            "moondream2 is open source, runs fully offline, and is about 1.7 GB. "
            "It downloads once and is stored locally — no cloud connection needed.<br><br>"
            "You can skip this now and download later via <b>Help → Download Composition AI</b>.<br><br>"
            "Download now?"
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No,
    )
    return result == QMessageBox.StandardButton.Yes


def prompt_model_download(parent: QWidget | None = None) -> bool:
    """
    Inform the user that LightGlue needs to be downloaded and ask for consent.
    Returns True if they want to proceed.
    """
    result = QMessageBox.question(
        parent,
        "AI Model Required",
        (
            "<b>LightGlue feature matching model not found.</b><br><br>"
            "Hockney Joiner uses LightGlue (open source, runs fully offline) "
            "to automatically detect overlapping areas between your photographs.<br><br>"
            "The download is several hundred MB (PyTorch + LightGlue weights). "
            "It happens once and is stored locally — no cloud connection needed at runtime.<br><br>"
            "Download now?"
        ),
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    return result == QMessageBox.StandardButton.Yes


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    # High-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_ORG)
    app.setApplicationVersion(APP_VERSION)

    # ── Scratch disk ──────────────────────────────────────────────────────────
    scratch_root = get_scratch_disk(app)
    log.info("Scratch disk: %s", scratch_root)

    # ── Scratch session (lives for the duration of this run) ──────────────────
    session = ScratchSession(scratch_root)

    # ── Model check ───────────────────────────────────────────────────────────
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_ready = is_model_ready(models_dir)
    if not model_ready:
        if prompt_model_download():
            download_model = True
        else:
            download_model = False
            log.info("User declined model download — auto-place will be disabled.")
    else:
        download_model = False

    # ── Moondream check ───────────────────────────────────────────────────────
    moondream_ready = is_moondream_ready(models_dir)
    if not moondream_ready:
        if prompt_moondream_download():
            download_moondream = True
        else:
            download_moondream = False
            log.info("User declined moondream download — composition chat will be disabled.")
    else:
        download_moondream = False

    # ── Main window ───────────────────────────────────────────────────────────
    window = MainWindow(
        session=session,
        models_dir=models_dir,
        model_ready=model_ready,
        download_model_on_open=download_model,
        download_moondream_on_open=download_moondream,
    )
    window.show()

    # ── Cleanup on exit ───────────────────────────────────────────────────────
    exit_code = app.exec()
    session.cleanup()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
