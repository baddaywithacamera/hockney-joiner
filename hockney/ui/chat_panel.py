"""
chat_panel.py — Composition chat panel.

A simple Q&A panel in the sidebar. The user types a question about their
composition; the vision model looks at a contact sheet of all thumbnails
and responds.

If the response mentions image numbers, those images are highlighted in the
tray view. The photographer reviews and decides what to do.

The AI advises. The photographer decides.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

PLACEHOLDER_QUESTIONS = [
    "Which images look redundant or nearly identical?",
    "Which images feel strongest in this composition?",
    "Does this look like a Hockney joiner or a failed panorama?",
    "Which tiles feel disconnected from the rest?",
]


class ChatPanel(QWidget):
    """
    Composition Q&A panel. Parent must connect tray_view so we can:
      - call render_contact_sheet() to get the image to analyse
      - call highlight_by_index() with the model's flagged indices
    """

    indices_ready = pyqtSignal(list)   # list[int] — connected to tray_view.highlight_by_index

    def __init__(self, models_dir: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.models_dir = models_dir
        self._tray_view = None
        self._worker = None
        self._build_ui()

    def set_tray_view(self, tray_view):
        self._tray_view = tray_view
        self.indices_ready.connect(tray_view.highlight_by_index)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QLabel("Composition Chat")
        header.setStyleSheet("font-weight: bold; font-size: 12px;")
        layout.addWidget(header)

        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self._status)

        # Chat history
        self._history = QTextEdit()
        self._history.setReadOnly(True)
        self._history.setMinimumHeight(120)
        self._history.setStyleSheet(
            "background: #1a1a1a; color: #ddd; font-size: 11px; border: none;"
        )
        layout.addWidget(self._history)

        # Input row
        input_row = QHBoxLayout()
        self._input = QLineEdit()
        self._input.setPlaceholderText(PLACEHOLDER_QUESTIONS[0])
        self._input.returnPressed.connect(self._send)
        input_row.addWidget(self._input)

        self._send_btn = QPushButton("Ask")
        self._send_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._send_btn.clicked.connect(self._send)
        input_row.addWidget(self._send_btn)
        layout.addLayout(input_row)

        clear_btn = QPushButton("Clear highlights")
        clear_btn.clicked.connect(self._clear_highlights)
        clear_btn.setStyleSheet("font-size: 10px;")
        layout.addWidget(clear_btn)

    def _send(self):
        question = self._input.text().strip()
        if not question:
            return
        if not self._tray_view:
            self._set_status("No tray view connected.")
            return

        from hockney.core.vision_chat import is_moondream_ready, VisionQueryWorker

        if not is_moondream_ready(self.models_dir):
            self._append("⚠ Composition AI not downloaded. Use Help → Download Composition AI.")
            return

        sheet = self._tray_view.render_contact_sheet()
        if sheet is None:
            self._append("⚠ No images loaded.")
            return

        self._append(f"You: {question}")
        self._input.clear()
        self._send_btn.setEnabled(False)
        self._set_status("Thinking…")

        self._worker = VisionQueryWorker(sheet, question, self.models_dir)
        self._worker.finished.connect(self._on_answer)
        self._worker.indices.connect(self._on_indices)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_answer(self, text: str):
        self._append(f"AI: {text}")
        self._send_btn.setEnabled(True)
        self._set_status("Ready.")

    def _on_indices(self, indices: list[int]):
        if indices:
            self.indices_ready.emit(indices)
            self._set_status(
                f"Highlighted image{'s' if len(indices) > 1 else ''} "
                f"{', '.join(str(i) for i in indices)}. "
                "Press any key or click 'Clear highlights' to dismiss."
            )

    def _on_error(self, msg: str):
        self._append(f"Error: {msg}")
        self._send_btn.setEnabled(True)
        self._set_status("Error.")

    def _clear_highlights(self):
        if self._tray_view:
            self._tray_view.clear_highlight()
        self._set_status("Highlights cleared.")

    def _append(self, text: str):
        self._history.append(text)
        self._history.verticalScrollBar().setValue(
            self._history.verticalScrollBar().maximum()
        )

    def _set_status(self, msg: str):
        self._status.setText(msg)
