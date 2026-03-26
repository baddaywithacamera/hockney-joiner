"""
sidebar.py — Right-hand panel: image list, batch processing settings.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from pathlib import Path

from hockney.core.image_store import ImageStore
from hockney.core.models import ProjectConfig
from hockney.ui.reference_panel import ReferencePanel

FILTER_OPTIONS = [
    "None", "Lo-Fi", "Clarendon", "Juno", "Lark", "Ludwig",
    "Perpetua", "Rise", "Slumber", "Toaster", "Valencia",
    "Walden", "Willow", "Xpro2",
]


class Sidebar(QWidget):
    process_requested = pyqtSignal(dict)
    reference_changed = pyqtSignal()
    bg_color_changed = pyqtSignal(QColor)

    def __init__(self, store: ImageStore, models_dir: Path | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store
        self.models_dir = models_dir
        self._build_ui()

    def _build_ui(self):
        # Outer layout holds the scroll area
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)

        # Inner widget holds all sidebar content
        self._inner = QWidget()
        scroll.setWidget(self._inner)
        layout = QVBoxLayout(self._inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # ── Reference images (hidden until New Project is created) ─────────
        self._ref_panel: ReferencePanel | None = None

        # ── Image list ────────────────────────────────────────────────────────
        list_group = QGroupBox("Images")
        list_layout = QVBoxLayout(list_group)
        self.image_list = QListWidget()
        self.image_list.setMaximumHeight(200)
        list_layout.addWidget(self.image_list)
        layout.addWidget(list_group)

        # ── Tile size ─────────────────────────────────────────────────────────
        tile_group = QGroupBox("Tile Size")
        tile_layout = QVBoxLayout(tile_group)
        tile_row = QHBoxLayout()
        self.tile_slider = QSlider(Qt.Orientation.Horizontal)
        self.tile_slider.setRange(100, 300)
        self.tile_slider.setSingleStep(50)
        self.tile_slider.setPageStep(50)
        self.tile_slider.setValue(300)
        self.tile_slider.setToolTip(
            "Thumbnail tile size in pixels. Smaller = less memory for large batches."
        )
        self._tile_label = QLabel("300 px")
        tile_row.addWidget(self.tile_slider)
        tile_row.addWidget(self._tile_label)
        tile_layout.addLayout(tile_row)
        tile_layout.addWidget(QLabel(
            "Set before loading. Smaller tiles save memory for 500+ image batches."
        ))
        self.tile_slider.valueChanged.connect(self._on_tile_size_changed)
        layout.addWidget(tile_group)

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

        # ── Background colour ─────────────────────────────────────────────────
        bg_group = QGroupBox("Background")
        bg_layout = QHBoxLayout(bg_group)

        self._bg_color = QColor(28, 28, 28)   # default dark gray
        self._bg_swatch = QPushButton()
        self._bg_swatch.setFixedSize(36, 36)
        self._bg_swatch.setFlat(True)
        self._bg_swatch.setAutoFillBackground(True)
        self._update_swatch()
        self._bg_swatch.setToolTip("Click to pick background colour")
        self._bg_swatch.clicked.connect(self._pick_bg_color)
        bg_layout.addWidget(self._bg_swatch)

        bg_layout.addWidget(QLabel("Canvas colour"))
        bg_layout.addStretch()

        layout.addWidget(bg_group)

        # ── Matching engine selector ─────────────────────────────────────────
        engine_group = QGroupBox("Matching Engine")
        engine_layout = QVBoxLayout(engine_group)
        self.engine_combo = QComboBox()
        self.engine_combo.addItems([
            "Auto (try all, keep best)",
            "DISK + LightGlue",
            "SuperPoint + LightGlue",
            "ALIKED + LightGlue",
            "SIFT + LightGlue",
            "SIFT (classic)",
            "ORB (fast)",
            "AKAZE (edges)",
            "BRISK (fastest)",
        ])
        self.engine_combo.setToolTip(
            "Auto tries GPU engines + SIFT, picks best.\n"
            "DISK: learned blob features, good all-rounder.\n"
            "SuperPoint: structural edges and corners.\n"
            "ALIKED: adaptive descriptors, varied textures.\n"
            "SIFT+LightGlue: classic features + learned matcher.\n"
            "SIFT: classic CPU-only, scale-invariant.\n"
            "ORB: fast CPU, good for strong corners.\n"
            "AKAZE: nonlinear diffusion, good on blurry images.\n"
            "BRISK: fastest CPU option."
        )
        engine_layout.addWidget(self.engine_combo)
        layout.addWidget(engine_group)

        # ── Process button ────────────────────────────────────────────────────
        self.process_btn = QPushButton("Auto-Place")
        self.process_btn.setToolTip("Run feature matching and place images on the reference.")
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

    def _update_swatch(self):
        """Update the colour swatch button to show the current bg colour."""
        c = self._bg_color
        # Use QPushButton selector for specificity so the app theme can't override
        self._bg_swatch.setStyleSheet(
            f"QPushButton {{ background-color: rgb({c.red()},{c.green()},{c.blue()});"
            " border: 2px solid #666; border-radius: 4px; }"
            f" QPushButton:hover {{ border-color: #aaa; }}"
        )

    def _pick_bg_color(self):
        color = QColorDialog.getColor(
            self._bg_color, self, "Background Colour",
        )
        if color.isValid():
            self._bg_color = color
            self._update_swatch()
            self.bg_color_changed.emit(color)

    @property
    def bg_color(self) -> QColor:
        return self._bg_color

    def _on_tile_size_changed(self, value: int):
        # Snap to nearest 50
        snapped = round(value / 50) * 50
        snapped = max(100, min(300, snapped))
        if snapped != value:
            self.tile_slider.blockSignals(True)
            self.tile_slider.setValue(snapped)
            self.tile_slider.blockSignals(False)
        self._tile_label.setText(f"{snapped} px")
        self.store.thumb_long_edge = snapped

    # Engine combo index → config key mapping
    _ENGINE_MAP = {
        0: "auto",
        1: "disk_lightglue",
        2: "superpoint",
        3: "aliked",
        4: "sift_lightglue",
        5: "sift",
        6: "orb",
        7: "akaze",
        8: "brisk",
    }

    def selected_engine(self) -> str:
        """Return the matching_engine key for ProjectConfig."""
        return self._ENGINE_MAP.get(self.engine_combo.currentIndex(), "auto")

    def _on_process_clicked(self):
        settings = {
            "histogram_eq": self.eq_checkbox.isChecked(),
            "filter": self.filter_combo.currentText(),
            "surface_effect": self.surface_combo.currentText(),
            "surface_intensity": self.intensity_slider.value(),
            "matching_engine": self.selected_engine(),
        }
        self.process_requested.emit(settings)

    def get_processing_settings(self) -> dict:
        c = self._bg_color
        return {
            "histogram_eq": self.eq_checkbox.isChecked(),
            "filter": self.filter_combo.currentText(),
            "surface_effect": self.surface_combo.currentText(),
            "surface_intensity": self.intensity_slider.value(),
            "tile_size": self.tile_slider.value(),
            "bg_color": (c.red(), c.green(), c.blue()),
        }

    def apply_processing_settings(self, settings: dict):
        self.eq_checkbox.setChecked(settings.get("histogram_eq", False))
        f = settings.get("filter", "None")
        idx = self.filter_combo.findText(f)
        if idx >= 0:
            self.filter_combo.setCurrentIndex(idx)
        s = settings.get("surface_effect", "None")
        idx = self.surface_combo.findText(s)
        if idx >= 0:
            self.surface_combo.setCurrentIndex(idx)
        self.intensity_slider.setValue(settings.get("surface_intensity", 30))
        tile = settings.get("tile_size", 300)
        self.tile_slider.setValue(tile)
        self._on_tile_size_changed(tile)

    def set_active_image(self, image_id: str):
        record = self.store.get_record(image_id)
        if record:
            self.info_label.setText(
                f"{record.source_path.name}\n"
                f"{record.width} × {record.height} px\n"
                f"{record.file_size / 1024 / 1024:.1f} MB"
            )

    def set_project_config(self, config: ProjectConfig | None):
        """Show or update the reference panel for the current project config."""
        inner_layout = self._inner.layout()
        if config is None:
            # Remove existing reference panel
            if self._ref_panel:
                inner_layout.removeWidget(self._ref_panel)
                self._ref_panel.deleteLater()
                self._ref_panel = None
            return

        if self._ref_panel:
            self._ref_panel.set_config(config)
        else:
            self._ref_panel = ReferencePanel(config, parent=self._inner)
            self._ref_panel.reference_changed.connect(self.reference_changed)
            # Insert at top of sidebar (index 0)
            inner_layout.insertWidget(0, self._ref_panel)

    def refresh(self):
        self.image_list.clear()
        for record in self.store.all_records():
            self.image_list.addItem(record.source_path.name)
