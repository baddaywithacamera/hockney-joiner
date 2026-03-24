"""
sidebar.py — Right-hand panel: image list, batch processing settings.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from pathlib import Path

from hockney.core.image_store import ImageStore

FILTER_OPTIONS = [
    "None", "Lo-Fi", "Clarendon", "Juno", "Lark", "Ludwig",
    "Perpetua", "Rise", "Slumber", "Toaster", "Valencia",
    "Walden", "Willow", "Xpro2",
]


class Sidebar(QWidget):
    process_requested = pyqtSignal(dict)

    def __init__(self, store: ImageStore, models_dir: Path | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store
        self.models_dir = models_dir
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # ── Image list ────────────────────────────────────────────────────────
        list_group = QGroupBox("Images")
        list_layout = QVBoxLayout(list_group)
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(200)
        list_layout.addWidget(self.image_list)
        layout.addWidget(list_group)

        # ── Batch processing ──────────────────────────────────────────────────
        batch_group = QGroupBox("Batch Processing")
        batch_layout = QVBoxLayout(batch_group)

        self.eq_checkbox = QCheckBox("Histogram Equalization")
        self.eq_checkbox.setToolTip(
            "Normalize exposure across all images. Reduces visible seams."
        )
        batch_layout.addWidget(self.eq_checkbox)

        batch_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(FILTER_OPTIONS)
        batch_layout.addWidget(self.filter_combo)

        layout.addWidget(batch_group)

        # ── Surface texture ───────────────────────────────────────────────────
        surface_group = QGroupBox("Surface Texture")
        surface_layout = QVBoxLayout(surface_group)

        surface_layout.addWidget(QLabel("Effect:"))
        self.surface_combo = QComboBox()
        self.surface_combo.addItems(["None", "Random Curves", "Surface Lighting", "Combined"])
        surface_layout.addWidget(self.surface_combo)

        surface_layout.addWidget(QLabel("Intensity:"))
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(30)
        surface_layout.addWidget(self.intensity_slider)

        layout.addWidget(surface_group)

        # ── Process button ────────────────────────────────────────────────────
        self.process_btn = QPushButton("Auto-Place")
        self.process_btn.setToolTip("Run LightGlue feature matching and place images.")
        self.process_btn.clicked.connect(self._on_process_clicked)
        layout.addWidget(self.process_btn)

        # ── Moondream chat ────────────────────────────────────────────────────
        if self.models_dir:
            from hockney.ui.chat_panel import ChatPanel
            self.chat_panel = ChatPanel(self.models_dir, parent=self)
            layout.addWidget(self.chat_panel)
        else:
            self.chat_panel = None

        layout.addStretch()

        # ── Info label ────────────────────────────────────────────────────────
        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.info_label)

    def _on_process_clicked(self):
        settings = {
            "histogram_eq": self.eq_checkbox.isChecked(),
            "filter": self.filter_combo.currentText(),
            "surface_effect": self.surface_combo.currentText(),
            "surface_intensity": self.intensity_slider.value(),
        }
        self.process_requested.emit(settings)

    def set_active_image(self, image_id: str):
        record = self.store.get_record(image_id)
        if record:
            self.info_label.setText(
                f"{record.source_path.name}\n"
                f"{record.width} × {record.height} px\n"
                f"{record.file_size / 1024 / 1024:.1f} MB"
            )

    def refresh(self):
        self.image_list.clear()
        for record in self.store.all_records():
            self.image_list.addItem(record.source_path.name)
