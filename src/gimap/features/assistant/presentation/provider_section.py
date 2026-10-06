"""Settings ▸ Assistant ▸ other AI providers: any OpenAI-compatible service.

DeepSeek, Qwen, OpenAI, Kimi, GLM, Gemini, OpenRouter, SiliconFlow, Azure
OpenAI, or a model running on this computer (Ollama, LM Studio).  The person
picks a provider, keeps its key here (or in the provider's environment
variable), picks a model — typed, suggested, or fetched from the provider —
and tests that it can call GIMaP's tools.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ..application import PROVIDERS, provider
from src.gimap.app.presentation.i18n import tr

from . import preferences
from .services import ProviderServices


def _muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("gimapRole", "muted")
    return label


class ProviderSection(QGroupBox):
    def __init__(self, settings, providers: ProviderServices, tasks, parent=None):
        super().__init__(tr("Other AI providers — OpenAI-compatible API"), parent)
        self.setObjectName("assistantProviderSection")
        self.settings = settings
        self.providers = providers
        self.tasks = tasks
        form = QFormLayout(self)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(8)

        self.provider_combo = QComboBox(self)
        self.provider_combo.setObjectName("assistantProviderCombo")
        for preset in PROVIDERS:
            self.provider_combo.addItem(preset.name, preset.key)
        form.addRow(tr("Provider"), self.provider_combo)

        self.url_edit = QLineEdit(self)
        self.url_edit.setObjectName("assistantProviderUrl")
        form.addRow(tr("Address"), self.url_edit)

        key_row = QHBoxLayout()
        self.key_edit = QLineEdit(self)
        self.key_edit.setObjectName("assistantProviderKey")
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("sk-…")
        self.save_key_button = QPushButton(tr("Save"), self)
        self.remove_key_button = QPushButton(tr("Remove"), self)
        key_row.addWidget(self.key_edit, 1)
        key_row.addWidget(self.save_key_button)
        key_row.addWidget(self.remove_key_button)
        form.addRow(tr("API key"), key_row)
        self.source_label = QLabel("", self)
        self.source_label.setObjectName("assistantProviderSource")
        self.source_label.setWordWrap(True)
        form.addRow(tr("Credentials"), self.source_label)

        model_row = QHBoxLayout()
        self.model_combo = QComboBox(self)
        self.model_combo.setObjectName("assistantProviderModel")
        self.model_combo.setEditable(True)
        self.fetch_button = QPushButton(tr("Get List"), self)
        self.fetch_button.setObjectName("assistantProviderFetch")
        self.fetch_button.setToolTip(tr("Ask the provider which models this key can use"))
        model_row.addWidget(self.model_combo, 1)
        model_row.addWidget(self.fetch_button)
        form.addRow(tr("Model"), model_row)

        test_row = QHBoxLayout()
        self.test_button = QPushButton(tr("Test Connection"), self)
        self.test_button.setObjectName("assistantProviderTest")
        self.test_label = QLabel("", self)
        self.test_label.setObjectName("assistantProviderTestResult")
        self.test_label.setWordWrap(True)
        test_row.addWidget(self.test_button)
        test_row.addWidget(self.test_label, 1)
        form.addRow("", test_row)
        self.notes_label = _muted("", self)
        form.addRow(self.notes_label)
        form.addRow(_muted(tr(
            "The model must support tool calling (function calling): the test sends one tiny request "
            "with a tool. Keys saved here stay in the user data folder (readable by your account only) "
            "and take precedence over the provider's environment variable. Runs send the frame's status "
            "and reduced curves as numbers, never the detector file."), self,
        ))

        current = str(preferences.read(settings, "provider"))
        self.provider_combo.setCurrentIndex(max(0, self.provider_combo.findData(current)))
        self._show_provider()
        self.provider_combo.currentIndexChanged.connect(self._provider_chosen)
        self.url_edit.editingFinished.connect(self._remember_url)
        self.model_combo.activated.connect(lambda _index: self._remember_model())
        self.model_combo.lineEdit().editingFinished.connect(self._remember_model)
        self.save_key_button.clicked.connect(self._save_key)
        self.remove_key_button.clicked.connect(self._remove_key)
        self.fetch_button.clicked.connect(self._fetch)
        self.test_button.clicked.connect(self._test)

    # -- state -------------------------------------------------------------------------------

    def key(self) -> str:
        return str(self.provider_combo.currentData())

    def _stored(self, name: str) -> dict:
        return dict(preferences.read(self.settings, name) or {})  # a copy: never the shared default

    def model(self) -> str:
        preset = provider(self.key())
        return str(self._stored("provider_models").get(self.key()) or (preset.models[0] if preset.models else ""))

    def url(self) -> str:
        return str(self._stored("provider_urls").get(self.key()) or "")

    def _show_provider(self) -> None:
        preset = provider(self.key())
        self.url_edit.setPlaceholderText(preset.base_url or "https://…/v1")
        self.url_edit.setText(self.url())
        self.model_combo.clear()
        self.model_combo.addItems(list(preset.models))
        self.model_combo.setEditText(self.model())
        needs = preset.needs_key
        self.key_edit.setEnabled(True)
        self.key_edit.setPlaceholderText("sk-…" if needs else tr("optional for this provider"))
        self.notes_label.setText(tr(preset.notes) if preset.notes else "")
        self.notes_label.setVisible(bool(preset.notes))
        self.test_label.clear()
        self._sync_source()

    def _sync_source(self) -> None:
        preset = provider(self.key())
        source = self.providers.source(self.key())
        if source:
            text = source
        elif preset.needs_key:
            text = (tr("none found — save a key above or set {variable}").format(variable=preset.env_key)
                    if preset.env_key else tr("none found — save a key above"))
        else:
            text = tr("no key needed")
        self.source_label.setText(text)
        self.remove_key_button.setEnabled(self.providers.has_saved_key(self.key()))

    def _provider_chosen(self, _index: int) -> None:
        preferences.write(self.settings, "provider", self.key())
        self._show_provider()

    def _remember_url(self) -> None:
        urls = self._stored("provider_urls")
        text = self.url_edit.text().strip()
        if text:
            urls[self.key()] = text
        else:
            urls.pop(self.key(), None)
        preferences.write(self.settings, "provider_urls", urls)

    def _remember_model(self) -> None:
        model = self.model_combo.currentText().strip()
        if model:
            models = self._stored("provider_models")
            models[self.key()] = model
            preferences.write(self.settings, "provider_models", models)

    # -- actions -----------------------------------------------------------------------------

    def _save_key(self) -> None:
        key = self.key_edit.text().strip()
        if not key:
            return
        try:
            self.providers.save_key(self.key(), key)
        except OSError as exc:
            QMessageBox.warning(self, tr("API key"), tr("The key could not be saved: {error}").format(error=exc))
            return
        self.key_edit.clear()
        self._sync_source()
        self.test_label.setText(tr("Key saved."))

    def _remove_key(self) -> None:
        name = provider(self.key()).name
        answer = QMessageBox.question(
            self, tr("API key"), tr("Remove the {provider} key saved in GIMaP?").format(provider=name),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self.providers.delete_key(self.key())
            self._sync_source()
            self.test_label.setText(tr("Key removed."))

    def _fetch(self) -> None:
        self._remember_url()
        key, model, url = self.key(), self.model(), self.url()
        self.fetch_button.setEnabled(False)
        self.test_label.setText(tr("Asking the provider for its models…"))
        self.tasks.submit(
            "assistant-provider-models",
            lambda: self.providers.list_models(key, model, url),
            on_done=lambda names: self._fetched(key, names),
            on_error=lambda message, _details: self._fetched(key, None, message),
        )

    def _fetched(self, key: str, names, error: str = "") -> None:
        self.fetch_button.setEnabled(True)
        if key != self.key():
            return
        if names is None:
            self.test_label.setText(tr("Failed: {error}").format(error=error))
            return
        current = self.model_combo.currentText()
        self.model_combo.clear()
        self.model_combo.addItems(list(names))
        self.model_combo.setEditText(current)
        self.test_label.setText(tr("{count} models available.").format(count=len(names)))

    def _test(self) -> None:
        self._remember_url()
        self._remember_model()
        key, model, url = self.key(), self.model(), self.url()
        self.test_button.setEnabled(False)
        self.test_label.setText(tr("Connecting…"))
        self.tasks.submit(
            "assistant-provider-test",
            lambda: self.providers.check(key, model, url),
            on_done=lambda name: self._tested(name),
            on_error=lambda message, _details: self._tested(None, message),
        )

    def _tested(self, name, error: str = "") -> None:
        self.test_button.setEnabled(True)
        self.test_label.setText(
            tr("Connected: {name} answered with a tool call.").format(name=name) if name
            else tr("Failed: {error}").format(error=error)
        )


def chosen_provider(settings) -> tuple[str, str, str]:
    """(provider key, model, address override) chosen in the settings."""
    key = str(preferences.read(settings, "provider"))
    preset = provider(key)
    models = dict(preferences.read(settings, "provider_models") or {})
    urls = dict(preferences.read(settings, "provider_urls") or {})
    return key, str(models.get(key) or (preset.models[0] if preset.models else "")), str(urls.get(key) or "")


def describe_provider(settings, providers: ProviderServices | None) -> tuple[bool, str, str]:
    """(ready, status text, label) of the provider chosen in the settings."""
    key, model, _url = chosen_provider(settings)
    preset = provider(key)
    label = f"{preset.name} · {model or tr('no model')}"
    if providers is None:
        return False, tr("Other AI providers need the openai package (python -m pip install openai)."), label
    if not model:
        return False, tr("{provider}: choose a model in Set Up AI….").format(provider=preset.name), label
    source = providers.source(key)
    if preset.needs_key and not source:
        return False, tr("No {provider} key yet: use Set Up AI… to add one.").format(provider=preset.name), label
    return True, tr("{label} · credentials: {source}.").format(label=label, source=source or tr("none needed")), label


__all__ = ["ProviderSection", "chosen_provider", "describe_provider"]
