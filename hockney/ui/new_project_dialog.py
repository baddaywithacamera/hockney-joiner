"""
new_project_dialog.py — New Project wizard dialog.

Collects: project name, project type, subject type.
Returns a ProjectConfig on accept.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QLabel,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
)

from hockney.core.models import (
    PROJECT_TYPES,
    SUBJECT_TYPES,
    ProjectConfig,
)

_TYPE_LABELS = {
    "perspective": "Perspective  (multiple angles of the same scene)",
    "time_of_day": "Time of Day  (same angle, different times)",
    "seasonal": "Seasonal  (same angle, different seasons)",
}

_SUBJECT_LABELS = {
    "landscape": "Landscape",
    "skylife": "Skylife",
    "urban": "Urban",
    "indoor": "Indoor",
    "people": "People",
}


class NewProjectDialog(QDialog):
    """Three-section dialog: name, type, subject."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        # ── Project name ──────────────────────────────────────────
        layout.addWidget(QLabel("Project Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Downtown Portland Spring")
        layout.addWidget(self._name_edit)

        layout.addSpacing(8)

        # ── Project type ──────────────────────────────────────────
        type_group = QGroupBox("Project Type")
        type_layout = QVBoxLayout(type_group)
        self._type_buttons = QButtonGroup(self)

        for i, ptype in enumerate(PROJECT_TYPES):
            rb = QRadioButton(_TYPE_LABELS.get(ptype, ptype))
            rb.setProperty("project_type", ptype)
            self._type_buttons.addButton(rb, i)
            type_layout.addWidget(rb)
            if i == 0:
                rb.setChecked(True)

        layout.addWidget(type_group)

        # ── Subject type ──────────────────────────────────────────
        layout.addWidget(QLabel("Subject Type:"))
        self._subject_combo = QComboBox()
        for stype in SUBJECT_TYPES:
            self._subject_combo.addItem(_SUBJECT_LABELS.get(stype, stype), stype)
        layout.addWidget(self._subject_combo)

        layout.addSpacing(12)

        # ── Buttons ───────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> ProjectConfig:
        """Return a ProjectConfig from the dialog values."""
        checked = self._type_buttons.checkedButton()
        ptype = checked.property("project_type") if checked else "perspective"
        return ProjectConfig(
            project_name=self._name_edit.text().strip() or "Untitled",
            project_type=ptype,
            subject_type=self._subject_combo.currentData() or "landscape",
        )
