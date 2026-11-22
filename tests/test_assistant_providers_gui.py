"""Other AI providers in the GUI: the settings section, the start dialog's check, and a whole run."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow

from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.features.assistant.application import (
    BACKEND_PROVIDER,
    GOALS,
    PERMISSION_AUTO,
    RUN_COMPLETED,
    AnalysisGoals,
    RunAssistantTask,
)
from src.gimap.features.assistant.presentation import AssistantController, AssistantSettingsPage, AssistantStartDialog
from src.gimap.features.assistant.presentation.services import ProviderServices
from src.gimap.features.assistant.presentation.service_notice import ServiceNotice
from src.gimap.integrations.state import InMemorySettingsRepository
from tests.assistant_fakes import ScriptedLlm, call, report, turn
from tests.test_assistant_gui import _app, _services, _wait, analyze  # noqa: F401 - the fixture


class FakeProviders:
    def __init__(self, llm=None, keys=None):
        self.llm = llm
        self.keys = dict(keys or {})
        self.created: list[tuple] = []

    def services(self) -> ProviderServices:
        return ProviderServices(
            create=self.create,
            source=lambda key: "API key saved in GIMaP" if key in self.keys else "",
            has_saved_key=lambda key: key in self.keys,
            save_key=lambda key, value: self.keys.__setitem__(key, value),
            delete_key=lambda key: self.keys.pop(key, None) is not None,
            list_models=lambda key, model, url: ["qwen-max", "qwen-plus", "qwen3-coder"],
            check=lambda key, model, url: model,
        )

    def create(self, key, model, url):
        self.created.append((key, model, url))
        return self.llm


def _with(services, providers: FakeProviders):
    return dataclasses.replace(services, providers=providers.services())


def test_the_settings_keep_provider_address_key_and_model(tmp_path: Path, monkeypatch) -> None:
    _app()
    settings = InMemorySettingsRepository({})
    providers = FakeProviders()
    tasks = TaskRunner()
    page = AssistantSettingsPage(settings, _with(_services(tmp_path, None), providers), tasks)
    section = page.provider_section
    assert section is not None and section.key() == "deepseek"  # the default provider
    page.provider_radio.setChecked(True)
    assert settings.get("assistant", "backend") == BACKEND_PROVIDER

    section.provider_combo.setCurrentIndex(section.provider_combo.findData("qwen"))
    assert settings.get("assistant", "provider") == "qwen"
    assert section.url_edit.placeholderText().startswith("https://dashscope.aliyuncs.com")
    assert section.model_combo.currentText() == "qwen-plus" and "No" not in section.source_label.text()
    assert "DASHSCOPE_API_KEY" in section.source_label.text()
    section.key_edit.setText("sk-qwen")
    section.save_key_button.click()
    assert providers.keys == {"qwen": "sk-qwen"} and section.source_label.text() == "API key saved in GIMaP"
    assert section.key_edit.text() == ""  # the key is not left on screen

    notices = []

    def accept(dialog):
        notices.append(dialog.message.text())
        return QDialog.Accepted

    monkeypatch.setattr(ServiceNotice, "exec_", accept)
    section.fetch_button.click()
    assert tasks.wait(10)
    QApplication.processEvents()
    assert [section.model_combo.itemText(index) for index in range(section.model_combo.count())] == ["qwen-max", "qwen-plus", "qwen3-coder"]
    section.model_combo.setEditText("qwen-max")
    section.model_combo.lineEdit().editingFinished.emit()
    section.url_edit.setText("https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    section.url_edit.editingFinished.emit()
    assert settings.get("assistant", "provider_models") == {"qwen": "qwen-max"}
    assert settings.get("assistant", "provider_urls") == {"qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"}
    section.test_button.click()
    assert tasks.wait(10)
    QApplication.processEvents()
    assert section.test_label.text().startswith("Connected: qwen-max")
    assert len(notices) == 2 and "Qwen" in notices[0] and "dashscope-intl.aliyuncs.com" in notices[1]

    section.provider_combo.setCurrentIndex(section.provider_combo.findData("ollama"))
    assert section.source_label.text() == "no key needed" and section.url_edit.text() == ""
    tasks.shutdown()


def test_the_start_dialog_knows_whether_the_provider_can_run(tmp_path: Path) -> None:
    _app()
    window = QMainWindow()
    settings = InMemorySettingsRepository({})
    settings.set("assistant", "backend", BACKEND_PROVIDER)
    providers = FakeProviders(llm=object())
    controller = AssistantController(window, _with(_services(tmp_path, None), providers), settings=settings, automation=lambda: None)
    dialog = AssistantStartDialog(settings, status={"file": "film.tif", "measurement": "giwaxs"})
    assert dialog.backend() == BACKEND_PROVIDER
    controller.check_brain(dialog, BACKEND_PROVIDER)
    assert not dialog.start_button.isEnabled() and "No DeepSeek" in dialog.credentials_label.text()
    providers.keys["deepseek"] = "sk"
    controller.check_brain(dialog, BACKEND_PROVIDER)
    assert dialog.start_button.isEnabled() and "deepseek-chat" in dialog.credentials_label.text()
    settings.set("assistant", "provider", "ollama")
    settings.set("assistant", "provider_urls", {"ollama": "http://lab-server:11434/v1"})
    controller.check_brain(dialog, BACKEND_PROVIDER)
    assert dialog.start_button.isEnabled() and "none needed" in dialog.credentials_label.text()
    llm, label, task = controller._brain()
    assert providers.created[-1] == ("ollama", "qwen2.5:14b", "http://lab-server:11434/v1")
    assert task is RunAssistantTask and label.startswith("Ollama")
    window.close()


def test_a_run_with_another_provider_drives_analyze(analyze, tmp_path: Path, monkeypatch) -> None:  # noqa: F811
    window, page, context = analyze
    llm = ScriptedLlm([
        dataclasses.replace(turn(call("find_peaks", curve="radial")), model="returned-model"),
        dataclasses.replace(turn(report(("peaks", "done"), summary="Peaks found by another provider.")), model=""),
    ])
    providers = FakeProviders(llm=llm, keys={"deepseek": "sk"})
    context.settings.set("assistant", "backend", BACKEND_PROVIDER)
    controller = AssistantController(window, _with(_services(tmp_path, None), providers), settings=context.settings, automation=page.automation)
    answers = iter([QDialog.Rejected, QDialog.Accepted])
    notices = []

    def answer(dialog):
        notices.append(dialog.message.text())
        return next(answers)

    monkeypatch.setattr(ServiceNotice, "exec_", answer)
    assert not controller.run(AnalysisGoals(goals=("peaks",)))
    assert not providers.created and not llm.requests
    assert controller.run(AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO))
    _wait(lambda: controller.outcome is not None, 60)
    assert controller.outcome.state == RUN_COMPLETED
    assert providers.created == [("deepseek", "deepseek-chat", "")]
    assert controller._model == "DeepSeek · deepseek-chat"  # the panel and the record name the provider
    assert [step.tool for step in controller.outcome.steps][:2] == ["get_status", "find_peaks"]
    assert len(notices) == 2 and all("DeepSeek" in text and "image" in text for text in notices)
    context.settings.set("assistant", "backend", "api")
    saved, = (tmp_path / "data" / "assistant_runs").glob("*.json")
    for record in (json.loads(saved.read_text("utf-8")), controller._record(controller.outcome)):
        assert record["brain"] == BACKEND_PROVIDER
        assert record["provider"] == "DeepSeek" and record["model"] == "returned-model"
        assert [step["model"] for step in record["steps"]] == ["", "returned-model", "returned-model"]
    assert "DeepSeek · returned-model" in controller.report_document(controller.outcome)
    context.settings.set("assistant", "backend", BACKEND_PROVIDER)
    llm.turns.append(turn(report(("peaks", "done"))))
    assert controller.follow_up("What was the largest peak?")  # a third prompt would exhaust answers
    _wait(lambda: controller.outcome is not None, 60)
