"""
export_dialog.py — Export options dialog.

Lets the user choose output path, format, and scale before committing
to what could be a multi-minute render.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ExportDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Export Composite")
        self.setMinimumWidth(460)

        self.output_path: str = ""
        self.scale_mode = "medium"
        self.transparent_bg = False

        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        form = QFormLayout()
        form.setSpacing(10)

        # ── Output path ───────────────────────────────────────────────────────
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Choose output file…")
        path_row.addWidget(self._path_edit)

        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        path_row.addWidget(browse_btn)
        form.addRow("Output file:", path_row)

        # ── Scale ─────────────────────────────────────────────────────────────
        self._scale_combo = QComboBox()
        self._scale_combo.addItems([
            "Medium  —  ~1500px per image  (recommended)",
            "Full resolution  —  original pixel dimensions",
            "Screen  —  thumbnail size  (fast preview)",
        ])
        self._scale_combo.setCurrentIndex(0)
        form.addRow("Resolution:", self._scale_combo)

        # ── Transparent background ───────────────────────────────────────────
        self._transparent_cb = QCheckBox("Transparent background  (PNG only)")
        self._transparent_cb.setToolTip(
            "Export with no background — useful for layering in Photoshop.\n"
            "Only works with PNG output; JPEG/TIFF will use the canvas colour."
        )
        form.addRow("", self._transparent_cb)

        layout.addLayout(form)

        # ── Info label ────────────────────────────────────────────────────────
        self._info = QLabel(
            "TIFF output is lossless and recommended for archival or further editing.\n"
            "Full resolution can produce very large files with many high-res originals."
        )
        self._info.setWordWrap(True)
        self._info.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._info)

        # ── Buttons ───────────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Composite As",
            "joiner.tif",
            "TIFF (*.tif *.tiff);;PNG (*.png);;JPEG (*.jpg *.jpeg)",
        )
        if path:
            self._path_edit.setText(path)

    def _accept(self):
        self.output_path = self._path_edit.text().strip()
        if not self.output_path:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No output file", "Please choose an output file.")
            return

        idx = self._scale_combo.currentIndex()
        self.scale_mode = ["medium", "full", "screen"][idx]
        self.transparent_bg = self._transparent_cb.isChecked()
        self.accept()
