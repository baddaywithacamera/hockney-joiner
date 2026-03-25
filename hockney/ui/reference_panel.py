"""
reference_panel.py — Reference image slot panel for the sidebar.

Shows labelled slots based on the project type.  Each slot has a
thumbnail preview, a Load button, and a Clear button.  Perspective
projects also have a "Show Advanced" toggle for Down Low / Up High.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from hockney.core.models import (
    PERSPECTIVE_ADVANCED_SLOTS,
    PERSPECTIVE_SLOTS,
    SLOT_LABELS,
    ProjectConfig,
    ReferenceImage,
)

log = logging.getLogger(__name__)

THUMB_SIZE = 80


class _SlotWidget(QWidget):
    """One reference slot: label + thumbnail + Load/Clear buttons."""

    loaded = pyqtSignal(str, str)    # slot_name, file_path
    cleared = pyqtSignal(str)        # slot_name

    def __init__(self, slot_name: str, parent=None):
        super().__init__(parent)
        self.slot_name = slot_name

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._thumb = QLabel()
        self._thumb.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._thumb.setStyleSheet(
            "border: 1px dashed #555; background: #1a1a1a; border-radius: 3px;"
        )
        self._thumb.setText("—")
        layout.addWidget(self._thumb)

        info_col = QVBoxLayout()
        info_col.setSpacing(4)
        lbl = QLabel(SLOT_LABELS.get(slot_name, slot_name))
        lbl.setStyleSheet("font-weight: bold; font-size: 11px;")
        info_col.addWidget(lbl)

        self._file_label = QLabel("empty")
        self._file_label.setStyleSheet("color: #888; font-size: 10px;")
        self._file_label.setWordWrap(True)
        info_col.addWidget(self._file_label)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load…")
        load_btn.setFixedWidth(60)
        load_btn.clicked.connect(self._on_load)
        btn_row.addWidget(load_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(self._on_clear)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()

        info_col.addLayout(btn_row)
        layout.addLayout(info_col, stretch=1)

    def _on_load(self):
        # Remember last directory across sessions via QSettings
        settings = QSettings("HockneyJoiner", "HockneyJoiner")
        last_dir = settings.value("last_image_dir", "", type=str)

        path, _ = QFileDialog.getOpenFileName(
            self, f"Load Reference — {SLOT_LABELS.get(self.slot_name, self.slot_name)}",
            last_dir,
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.cr2 *.cr3 *.nef *.arw *.dng)",
        )
        if path:
            settings.setValue("last_image_dir", str(Path(path).parent))
            self.set_image(path)
            self.loaded.emit(self.slot_name, path)

    def _on_clear(self):
        self._thumb.setPixmap(QPixmap())
        self._thumb.setText("—")
        self._file_label.setText("empty")
        self.cleared.emit(self.slot_name)

    def set_image(self, path: str):
        """Display a thumbnail and filename for a loaded reference."""
        pm = QPixmap(path)
        if not pm.isNull():
            scaled = pm.scaled(
                THUMB_SIZE, THUMB_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._thumb.setPixmap(scaled)
            self._thumb.setText("")
        self._file_label.setText(Path(path).name)


class ReferencePanel(QGroupBox):
    """
    Sidebar group showing reference image slots for the current project type.
    """

    reference_changed = pyqtSignal()   # emitted when any slot is loaded/cleared

    def __init__(self, config: ProjectConfig, parent=None):
        super().__init__("Reference Images", parent)
        self._config = config
        self._slot_widgets: dict[str, _SlotWidget] = {}
        self._advanced_visible = False
        self._build_ui()

    def _build_ui(self):
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(6)

        # Standard slots
        for slot in self._config.slots_for_type(include_advanced=False):
            w = self._make_slot(slot)
            self._layout.addWidget(w)

        # Advanced toggle (perspective only)
        if self._config.project_type == "perspective":
            self._adv_check = QCheckBox("Show Advanced")
            self._adv_check.setStyleSheet("color: #888; font-size: 10px;")
            self._adv_check.toggled.connect(self._toggle_advanced)
            self._layout.addWidget(self._adv_check)

            for slot in PERSPECTIVE_ADVANCED_SLOTS:
                w = self._make_slot(slot)
                w.setVisible(False)
                self._layout.addWidget(w)

        # Restore any existing references
        for ref in self._config.references:
            w = self._slot_widgets.get(ref.slot)
            if w and ref.source_path:
                w.set_image(ref.source_path)

    def _make_slot(self, slot_name: str) -> _SlotWidget:
        w = _SlotWidget(slot_name, parent=self)
        w.loaded.connect(self._on_slot_loaded)
        w.cleared.connect(self._on_slot_cleared)
        self._slot_widgets[slot_name] = w
        return w

    def _toggle_advanced(self, show: bool):
        for slot in PERSPECTIVE_ADVANCED_SLOTS:
            w = self._slot_widgets.get(slot)
            if w:
                w.setVisible(show)

    def _on_slot_loaded(self, slot: str, path: str):
        self._config.set_reference(slot, path)
        self.reference_changed.emit()
        log.info("Reference loaded: %s → %s", slot, path)

    def _on_slot_cleared(self, slot: str):
        self._config.clear_reference(slot)
        self.reference_changed.emit()
        log.info("Reference cleared: %s", slot)

    def set_config(self, config: ProjectConfig):
        """Rebuild the panel for a new project config."""
        self._config = config
        # Clear existing widgets
        for w in self._slot_widgets.values():
            w.setParent(None)
            w.deleteLater()
        self._slot_widgets.clear()
        # Remove all items from layout
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._build_ui()
