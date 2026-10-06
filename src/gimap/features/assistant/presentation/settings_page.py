"""Settings ▸ Assistant: the brain (Claude Code on your plan, the Claude API, or another provider), effort, limits."""

from __future__ import annotations

from PyQt5 import sip
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..application import (
    BACKEND_API,
    BACKEND_CLAUDE_CODE,
    BACKEND_PROVIDER,
    EFFORTS,
    MODEL_CHOICES,
    PERMISSION_AUTO,
    PERMISSION_CONFIRM,
    PERMISSION_PREVIEW,
)
from src.gimap.app.presentation.i18n import tr

from . import preferences
from .code_section import ClaudeCodeSection
from .provider_section import ProviderSection
from .services import AssistantServices


def _muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("gimapRole", "muted")
    return label


class AssistantSettingsPage(QWidget):
    def __init__(self, settings, services: AssistantServices, tasks, parent=None):
        super().__init__(parent)
        self.setObjectName("assistantSettingsPage")
        self.settings = settings
        self.services = services
        self.tasks = tasks
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(_muted(tr(
            "“Process with AI” in Analyze lets an AI run the analysis tools on the open frame and "
            "report what it finds. Each run sends the frame's status and reduced curves (numbers, "
            "never the detector file) to the chosen provider."), self,
        ))

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)
        brain_row = QVBoxLayout()
        self.code_radio = QRadioButton(tr("Claude Code — uses your Claude plan (Pro / Max), no API key"), self)
        self.code_radio.setObjectName("assistantBrainCode")
        self.api_radio = QRadioButton(tr("Claude API — API key, billed per token"), self)
        self.api_radio.setObjectName("assistantBrainApi")
        self.provider_radio = QRadioButton(tr(
            "Other AI provider — DeepSeek, Qwen, OpenAI, Kimi, GLM, Gemini, Ollama … (see below)"), self
        )
        self.provider_radio.setObjectName("assistantBrainProvider")
        brains = QButtonGroup(self)
        self._brains = {BACKEND_CLAUDE_CODE: self.code_radio, BACKEND_API: self.api_radio, BACKEND_PROVIDER: self.provider_radio}
        for radio in self._brains.values():
            brains.addButton(radio)
            brain_row.addWidget(radio)
        chosen = preferences.read(settings, "backend")
        self._brains.get(chosen, self.code_radio).setChecked(True)
        form.addRow(tr("Brain"), brain_row)
        self.effort_combo = QComboBox(self)
        self.effort_combo.setObjectName("assistantEffortCombo")
        for effort in EFFORTS:  # the names are the setting itself: never translated
            self.effort_combo.addItem(effort, effort)
        self.effort_combo.setCurrentIndex(max(0, self.effort_combo.findData(str(preferences.read(settings, "effort")))))
        form.addRow(tr("Effort"), self.effort_combo)
        self.turns_spin = QSpinBox(self)
        self.turns_spin.setObjectName("assistantTurnsSpin")
        self.turns_spin.setRange(4, 60)
        self.turns_spin.setValue(int(preferences.read(settings, "max_turns")))
        form.addRow(tr("Most model turns per run"), self.turns_spin)
        self.permission_combo = QComboBox(self)
        self.permission_combo.setObjectName("assistantPermissionCombo")
        for text, value in (
            ("Preview first — changes come back as cards to apply", PERMISSION_PREVIEW),
            ("Ask before writing files or changing corrections", PERMISSION_CONFIRM),
            ("Fully automatic — everything logged and undoable", PERMISSION_AUTO),
        ):
            self.permission_combo.addItem(tr(text), value)
        self.permission_combo.setCurrentIndex(max(0, self.permission_combo.findData(preferences.read(settings, "permission"))))
        form.addRow(tr("Permissions"), self.permission_combo)
        layout.addLayout(form)
        layout.addWidget(_muted(tr(
            "Higher effort lets Claude think longer per step: slower, and more of your plan's "
            "usage or more tokens (other providers use their model's own setting). The start dialog "
            "can change the brain and the permission for a single run."), self,
        ))

        standing = QGroupBox(tr("Standing instructions for the AI"), self)
        standing_layout = QVBoxLayout(standing)
        self.standing_edit = QPlainTextEdit(standing)
        self.standing_edit.setObjectName("assistantStandingEdit")
        self.standing_edit.setPlaceholderText(tr(
            "e.g. Calibration images are in D:\\beamtime\\calib; AgBH was measured at every new distance; "
            "the energy is 11.8 keV; αi is in the .fio files as 'om'."
        ))
        self.standing_edit.setPlainText(str(preferences.read(settings, "standing_instructions") or ""))
        self.standing_edit.setFixedHeight(84)
        standing_layout.addWidget(self.standing_edit)
        standing_layout.addWidget(_muted(tr(
            "Added to every run, with the notes of the start dialog. Paths written here or in the notes can "
            "be read and searched by the AI directly."), standing,
        ))
        layout.addWidget(standing)
        self._standing_timer = QTimer(self)
        self._standing_timer.setSingleShot(True)
        self._standing_timer.setInterval(600)
        self._standing_timer.timeout.connect(self._remember_standing)
        self.standing_edit.textChanged.connect(self._standing_timer.start)

        self.code_section = ClaudeCodeSection(settings, services, tasks, self)
        layout.addWidget(self.code_section)
        layout.addWidget(self._api_section())
        self.provider_section = None
        if services.providers is not None:
            self.provider_section = ProviderSection(settings, services.providers, tasks, self)
            layout.addWidget(self.provider_section)
        else:
            layout.addWidget(_muted(tr("Other AI providers need the openai package: python -m pip install openai"), self))
            self.provider_radio.setEnabled(False)
        layout.addStretch(1)
        self._sync_source()
        for backend, radio in self._brains.items():
            radio.toggled.connect(lambda on, backend=backend: on and preferences.write(settings, "backend", backend))
        self.effort_combo.activated.connect(
            lambda _index: preferences.write(settings, "effort", self.effort_combo.currentData() or self.effort_combo.currentText())
        )
        self.turns_spin.valueChanged.connect(lambda value: preferences.write(settings, "max_turns", int(value)))
        self.permission_combo.currentIndexChanged.connect(
            lambda _index: preferences.write(settings, "permission", self.permission_combo.currentData())
        )
        self.model_combo.activated.connect(lambda _index: self._remember_model())
        self.model_combo.lineEdit().editingFinished.connect(self._remember_model)
        self.save_key_button.clicked.connect(self._save_key)
        self.remove_key_button.clicked.connect(self._remove_key)
        self.test_button.clicked.connect(self._test)

    def _api_section(self) -> QGroupBox:
        box = QGroupBox(tr("Claude API — API key"), self)
        box.setObjectName("assistantApiSection")
        form = QFormLayout(box)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)
        self.model_combo = QComboBox(box)
        self.model_combo.setObjectName("assistantModelCombo")
        self.model_combo.setEditable(True)
        self.model_combo.addItems(MODEL_CHOICES)
        self.model_combo.setEditText(str(preferences.read(self.settings, "model")))
        form.addRow(tr("Model"), self.model_combo)
        key_row = QHBoxLayout()
        self.key_edit = QLineEdit(box)
        self.key_edit.setObjectName("assistantKeyEdit")
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("sk-ant-…")
        self.save_key_button = QPushButton(tr("Save"), box)
        self.remove_key_button = QPushButton(tr("Remove"), box)
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.save_key_button)
        key_row.addWidget(self.remove_key_button)
        form.addRow(tr("API key"), key_row)
        self.source_label = QLabel("", box)
        self.source_label.setObjectName("assistantCredentialSource")
        self.source_label.setWordWrap(True)
        form.addRow(tr("Credentials"), self.source_label)
        test_row = QHBoxLayout()
        self.test_button = QPushButton(tr("Test Connection"), box)
        self.test_button.setObjectName("assistantTestButton")
        self.test_label = QLabel("", box)
        self.test_label.setWordWrap(True)
        test_row.addWidget(self.test_button)
        test_row.addWidget(self.test_label, 1)
        form.addRow("", test_row)
        form.addRow(_muted(tr(
            "An API key is billed per token and is separate from a Claude plan. A key saved here "
            "is kept in the user data folder (plain text, readable by your user account only) and "
            "takes precedence over the ANTHROPIC_API_KEY environment variable. The panel shows the "
            "tokens and the estimated cost of every run."), box,
        ))
        return box

    def _remember_standing(self) -> None:
        preferences.write(self.settings, "standing_instructions", self.standing_edit.toPlainText().strip())

    def hideEvent(self, event) -> None:
        if self._standing_timer.isActive():
            self._standing_timer.stop()
            self._remember_standing()
        super().hideEvent(event)

    def _remember_model(self) -> None:
        model = self.model_combo.currentText().strip()
        if model:
            preferences.write(self.settings, "model", model)

    def _sync_source(self) -> None:
        source = self.services.credentials()
        self.source_label.setText(source or tr("none found — save a key above or set ANTHROPIC_API_KEY"))
        self.remove_key_button.setEnabled(self.services.has_saved_key())
        self.test_button.setEnabled(bool(source))

    def _save_key(self) -> None:
        key = self.key_edit.text().strip()
        if not key:
            return
        try:
            self.services.save_key(key)
        except OSError as exc:
            QMessageBox.warning(self, tr("Claude API Key"), tr("The key could not be saved: {error}").format(error=exc))
            return
        self.key_edit.clear()
        self._sync_source()
        self.test_label.setText(tr("Key saved."))

    def _remove_key(self) -> None:
        answer = QMessageBox.question(
            self, tr("Claude API Key"), tr("Remove the API key saved in GIMaP?"),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.services.delete_key()
            self._sync_source()
            self.test_label.setText(tr("Key removed."))

    def _test(self) -> None:
        self._remember_model()
        model = str(preferences.read(self.settings, "model"))
        effort = str(preferences.read(self.settings, "effort"))
        self.test_button.setEnabled(False)
        self.test_label.setText(tr("Connecting…"))
        self.tasks.submit(
            "assistant-test",
            lambda: self.services.check(model, effort),
            on_done=self._tested,
            on_error=lambda message, _details: self._tested(None, message),
        )

    def _tested(self, name, error: str = "") -> None:
        if sip.isdeleted(self) or sip.isdeleted(self.test_label):
            return  # Settings was closed (and deleted) before the check came back
        self.test_button.setEnabled(True)
        self.test_label.setText(
            tr("Connected: {name} is available.").format(name=name) if name else tr("Failed: {error}").format(error=error)
        )


__all__ = ["AssistantSettingsPage"]
