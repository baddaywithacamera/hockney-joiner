"""
composition_ai_dialog.py — Settings for the composition chat backend.

Lets the photographer choose who answers composition questions:
  • Local (offline)  — on-device BLIP-VQA, no data leaves the machine.
  • Claude / Gemini  — a cloud LLM. Sends ONE contact-sheet image + the
                       question to the provider when you ask something.

API keys are read from the environment first (ANTHROPIC_API_KEY /
GEMINI_API_KEY); anything entered here is a fallback stored via QSettings.
"""

from __future__ import annotations

import os

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from hockney.core import vision_provider as vp


class CompositionAISettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Composition AI Settings")
        self.setMinimumWidth(460)
        self._settings = QSettings(vp.SETTINGS_ORG, vp.SETTINGS_APP)
        self._build_ui()
        self._load()
        self._update_enabled()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Choose who answers composition questions. <b>Local</b> runs fully "
            "offline. <b>Claude</b> and <b>Gemini</b> are cloud services — asking "
            "a question sends one contact-sheet image plus your question to that "
            "provider. Placement always stays offline."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(intro)

        prov_box = QGroupBox("Provider")
        prov_form = QFormLayout(prov_box)
        self._provider = QComboBox()
        self._provider.addItem("Local (offline, BLIP-VQA)", "local")
        self._provider.addItem("Claude (Anthropic)", "claude")
        self._provider.addItem("Gemini (Google)", "gemini")
        self._provider.currentIndexChanged.connect(self._update_enabled)
        prov_form.addRow("Backend:", self._provider)
        layout.addWidget(prov_box)

        # Claude group
        self._claude_box = QGroupBox("Claude")
        c_form = QFormLayout(self._claude_box)
        self._claude_model = QLineEdit()
        self._claude_model.setPlaceholderText(vp.DEFAULT_CLAUDE_MODEL)
        self._claude_key = QLineEdit()
        self._claude_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._claude_key.setPlaceholderText("sk-ant-…  (or set ANTHROPIC_API_KEY)")
        self._claude_env = QLabel()
        self._claude_env.setStyleSheet("color: #6c6; font-size: 10px;")
        c_form.addRow("Model:", self._claude_model)
        c_form.addRow("API key:", self._claude_key)
        c_form.addRow("", self._claude_env)
        layout.addWidget(self._claude_box)

        # Gemini group
        self._gemini_box = QGroupBox("Gemini")
        g_form = QFormLayout(self._gemini_box)
        self._gemini_model = QLineEdit()
        self._gemini_model.setPlaceholderText(vp.DEFAULT_GEMINI_MODEL)
        self._gemini_key = QLineEdit()
        self._gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._gemini_key.setPlaceholderText("AIza…  (or set GEMINI_API_KEY)")
        self._gemini_env = QLabel()
        self._gemini_env.setStyleSheet("color: #6c6; font-size: 10px;")
        g_form.addRow("Model:", self._gemini_model)
        g_form.addRow("API key:", self._gemini_key)
        g_form.addRow("", self._gemini_env)
        layout.addWidget(self._gemini_box)

        note = QLabel(
            "Model IDs change often — these are editable. Cost/latency is lowest "
            "on the default fast tier, which is plenty for contact-sheet Q&A."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── Load / save ───────────────────────────────────────────────────────────
    def _load(self):
        provider = str(self._settings.value(vp.KEY_PROVIDER, vp.DEFAULT_PROVIDER))
        idx = self._provider.findData(provider)
        self._provider.setCurrentIndex(idx if idx >= 0 else 0)

        self._claude_model.setText(
            str(self._settings.value(vp.KEY_CLAUDE_MODEL, vp.DEFAULT_CLAUDE_MODEL))
        )
        self._gemini_model.setText(
            str(self._settings.value(vp.KEY_GEMINI_MODEL, vp.DEFAULT_GEMINI_MODEL))
        )
        self._claude_key.setText(str(self._settings.value(vp.KEY_CLAUDE_API, "")))
        self._gemini_key.setText(str(self._settings.value(vp.KEY_GEMINI_API, "")))

        if os.environ.get(vp.ENV_CLAUDE_KEY):
            self._claude_env.setText(f"✓ {vp.ENV_CLAUDE_KEY} is set — it overrides this field.")
        if os.environ.get(vp.ENV_GEMINI_KEY):
            self._gemini_env.setText(f"✓ {vp.ENV_GEMINI_KEY} is set — it overrides this field.")

    def _update_enabled(self):
        provider = self._provider.currentData()
        self._claude_box.setEnabled(provider == "claude")
        self._gemini_box.setEnabled(provider == "gemini")

    def _save_and_accept(self):
        self._settings.setValue(vp.KEY_PROVIDER, self._provider.currentData())
        self._settings.setValue(
            vp.KEY_CLAUDE_MODEL, self._claude_model.text().strip() or vp.DEFAULT_CLAUDE_MODEL
        )
        self._settings.setValue(
            vp.KEY_GEMINI_MODEL, self._gemini_model.text().strip() or vp.DEFAULT_GEMINI_MODEL
        )
        self._settings.setValue(vp.KEY_CLAUDE_API, self._claude_key.text().strip())
        self._settings.setValue(vp.KEY_GEMINI_API, self._gemini_key.text().strip())
        self._settings.sync()
        self.accept()
