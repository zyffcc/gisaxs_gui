"""“Process with Claude” end to end: a synthetic GIWAXS frame in Analyze, a scripted Claude.

The scripted model calls the same tools a real run would; the test checks that
Analyze follows each step on screen, that the numbers are right for the
synthetic film, and what the panel, the start dialog and the settings show.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox

from src.gimap.app import AppContext
from src.gimap.features.analyze.bootstrap import create_analyze_view_model
from src.gimap.features.analyze.domain.giwaxs import giwaxs_maps
from src.gimap.features.analyze.presentation.page import AnalyzePage
from src.gimap.features.assistant.application import (
    BACKEND_API,
    BACKEND_CLAUDE_CODE,
    BILLING_SUBSCRIPTION,
    GOALS,
    PERMISSION_AUTO,
    PERMISSION_CONFIRM,
    RUN_CANCELLED,
    RUN_COMPLETED,
    AnalysisGoals,
    LlmError,
)
from src.gimap.features.assistant.infrastructure import ClaudeCodeAgent, JsonResultStore, estimated_cost
from src.gimap.features.assistant.presentation import (
    AssistantController,
    AssistantServices,
    AssistantSettingsPage,
    AssistantStartDialog,
)
from src.gimap.integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.assistant_fakes import ScriptedLlm, call, report, tool_results, turn

SHAPE = (400, 400)
# 100 µm pixels 60 mm from the sample, beam near the bottom edge: q up to ≈ 2 Å⁻¹ in plane.
GEOMETRY = DetectorGeometry(100e-6, 100e-6, 0.06, 200.0, 380.0, 1.0, 0.2)
_APP = None


def _app() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _gauss(x, center, fwhm):
    return np.exp(-4.0 * np.log(2.0) * (x - center) ** 2 / fwhm**2)


def _write_frame(path: Path) -> Path:
    """Lamellar (100)/(200) along the surface normal, an isotropic ring, π–π in plane."""
    maps = giwaxs_maps(SHAPE, GEOMETRY)
    q = maps.q.astype(float)
    chi = np.abs(maps.chi_deg.astype(float))
    normal = np.exp(-(chi**2) / (2 * 15.0**2))
    in_plane = np.exp(-((chi - 90.0) ** 2) / (2 * 20.0**2))
    image = 30.0 + 300.0 * np.exp(-q / 0.25)
    image += 600.0 * _gauss(q, 0.40, 0.03) * normal
    image += 120.0 * _gauss(q, 0.80, 0.035) * normal
    image += 80.0 * _gauss(q, 1.20, 0.04)
    image += 150.0 * _gauss(q, 1.65, 0.08) * in_plane
    counts = np.random.default_rng(7).poisson(image).astype(np.float32)
    Image.fromarray(counts, mode="F").save(path)
    return path


def _context(tmp_path: Path) -> AppContext:
    profile = InstrumentProfile("Synthetic GIWAXS", GEOMETRY, None, SHAPE)
    return AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        instrument_profiles=InMemoryInstrumentProfileRepository([profile]),
    )


SIGNED_IN = {
    "version": "9.9.9 (Claude Code)", "logged_in": True, "auth_method": "claude.ai",
    "subscription": "max", "billing": "subscription", "error": "",
}
FAKE_CLI = Path(__file__).with_name("fake_claude_code.py")


def _services(
    tmp_path: Path, llm, *, credentials: str = "test key", code_info: dict | None = None, logins: list | None = None,
) -> AssistantServices:
    store = JsonResultStore(tmp_path / "data")
    saved: dict = {}

    def create_agent(cli, model, effort):
        environ = {"PATH": "", "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""), "FAKE_CLAUDE_SCENARIO": "report"}
        return ClaudeCodeAgent(command=[sys.executable, str(FAKE_CLI)], model=model, effort=effort, environ=environ)

    return AssistantServices(
        create_llm=lambda model, effort: llm,
        credentials=lambda: "API key saved in GIMaP" if saved else credentials,
        has_saved_key=lambda: bool(saved),
        save_key=lambda key: saved.update(key=key),
        delete_key=lambda: bool(saved.pop("key", None)),
        check=lambda model, effort: f"{model} ({effort})",
        store=store,
        save_text=store.save_text,
        save_run=store.save_run,
        cost=estimated_cost,
        create_agent=create_agent,
        find_cli=lambda configured: configured or "C:/Claude/claude.exe",
        code_status=lambda cli: dict(SIGNED_IN if code_info is None else code_info),
        code_login=lambda cli: (logins if logins is not None else []).append(cli),
    )


def _wait(condition, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        QApplication.processEvents()
        time.sleep(0.005)
        assert time.monotonic() < deadline, "timed out"
    QApplication.processEvents()


@pytest.fixture()
def analyze(tmp_path: Path):
    _app()
    frame = _write_frame(tmp_path / "film_giwaxs.tif")
    context = _context(tmp_path)
    page = AnalyzePage(create_analyze_view_model(context))
    window = QMainWindow()
    window.setCentralWidget(page)
    page.add_paths([frame])
    assert page.tasks.wait(60)
    assert page.view_model.state.analysis.kind == "giwaxs"
    yield window, page, context
    page.tasks.wait(60)
    window.close()


def _controller(window, page, context, services, backend: str = BACKEND_API) -> AssistantController:
    context.settings.set("assistant", "backend", backend)
    return AssistantController(window, services, settings=context.settings, automation=page.automation)


def test_claude_operates_analyze_and_reports_every_requested_result(analyze, tmp_path: Path) -> None:
    window, page, context = analyze
    llm = ScriptedLlm([
        turn(call("find_peaks", curve="radial"), text="Peaks first."),
        turn(call("compare_sectors")),
        turn(call("ring_orientation", q_center=0.40)),
        turn(call("crystallite_size", q_center=1.20)),
        turn(call("export_results", include_tables=True)),
        turn(report(("peaks", "done"), ("orientation", "done"), ("ring_orientation", "done"), ("crystallite_size", "done"))),
    ])
    controller = _controller(window, page, context, _services(tmp_path, llm))
    assert controller.run(AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO))
    _wait(lambda: controller.outcome is not None)

    outcome = controller.outcome
    assert outcome.state == RUN_COMPLETED, outcome.message
    assert all(step.ok for step in outcome.steps), [(step.tool, step.summary) for step in outcome.steps]
    results = outcome.results
    found = [peak.q for peak in results.peak_searches["radial"].peaks]
    for expected in (0.40, 0.80, 1.20, 1.65):
        assert min(abs(q - expected) for q in found) < 0.02, found

    def preference(q):
        return min(results.sector_rows, key=lambda row: abs(row.q - q)).preference

    assert preference(0.40) in ("mainly out-of-plane", "out-of-plane only")
    assert preference(1.65) in ("mainly in-plane", "in-plane only")
    ring = results.rings[0]
    assert "surface normal" in ring.texture and ring.herman > 0.3
    assert results.sizes[0].lower_bound and 8.0 < results.sizes[0].size / 10.0 < 25.0  # nm

    # Analyze followed the run on screen.
    status = page.automation().status()
    assert status["giwaxs"]["chi_q_window"] == pytest.approx(list(ring.q_window), abs=1e-6)
    assert page._lower_profile == "azimuthal"
    written = {path.name for path in (tmp_path / "gimap_analysis").iterdir()}
    assert "film_giwaxs_assistant.json" in written and any(name.endswith(".csv") for name in written)

    panel = controller.panel
    assert panel.state_label.text() == "Finished"
    texts = [panel.step_list.item(row).text() for row in range(panel.step_list.count())]
    # Steps in words; the raw call stays in the tooltip.
    assert texts[0].startswith("1. Reading the frame and its header — GIWAXS")
    assert any(text.startswith("2. Finding the peaks of I(q)") for text in texts)
    assert panel.step_list.item(0).toolTip().startswith("get_status(")
    assert not any("_" in text.split(" — ")[0] for text in texts if text[:1].isdigit())
    assert "Claude: Peaks first." in texts
    report_text = panel.report_view.toPlainText()
    assert "Peak table" in report_text and "Computed results" in report_text and "Crystallite size" in report_text
    assert "≈ $" in panel.usage_label.text()
    runs = list((tmp_path / "data" / "assistant_runs").iterdir())
    assert len(runs) == 1 and runs[0].name.endswith("_film_giwaxs.json")
    assert controller._dock.isVisibleTo(window)


def test_confirm_mode_asks_before_writing_and_a_no_is_respected(analyze, tmp_path: Path, monkeypatch) -> None:
    window, page, context = analyze
    asked: list[str] = []

    def answer_no(_parent, _title, text, *_args):
        asked.append(text)
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", staticmethod(answer_no))
    llm = ScriptedLlm([
        turn(call("export_results", include_tables=False)),
        turn(report(("peaks", "partial"))),
    ])
    controller = _controller(window, page, context, _services(tmp_path, llm))
    assert controller.run(AnalysisGoals(goals=("peaks",), permission=PERMISSION_CONFIRM))
    _wait(lambda: controller.outcome is not None)

    assert controller.outcome.state == RUN_COMPLETED
    assert asked and "Allow the AI" in asked[0]
    assert not (tmp_path / "gimap_analysis").exists()
    assert '"declined": true' in tool_results(llm.requests[1]["messages"])[0]["content"]


def test_stop_ends_a_run_that_is_waiting_for_claude(analyze, tmp_path: Path) -> None:
    window, page, context = analyze
    waiting = threading.Event()

    class SlowClaude(ScriptedLlm):
        def respond(self, *, system, tools, messages, cancelled=None, progress=None):
            waiting.set()
            while not cancelled():
                time.sleep(0.01)
            raise LlmError("Stopped by the user.")

    controller = _controller(window, page, context, _services(tmp_path, SlowClaude([])))
    assert controller.run(AnalysisGoals(goals=("peaks",)))
    _wait(waiting.is_set)
    assert controller.running() and controller.panel.stop_button.isVisibleTo(controller.panel)
    controller.panel.stop_button.click()
    _wait(lambda: controller.outcome is not None)
    assert controller.outcome.state == RUN_CANCELLED
    assert controller.panel.state_label.text() == "Stopped"
    assert controller.panel.again_button.isVisibleTo(controller.panel)


def test_shutdown_does_not_wait_for_a_stuck_request(analyze, tmp_path: Path) -> None:
    window, page, context = analyze
    release = threading.Event()

    class StuckClaude(ScriptedLlm):
        def respond(self, **_kwargs):
            release.wait(30)
            raise LlmError("gone")

    controller = _controller(window, page, context, _services(tmp_path, StuckClaude([])))
    assert controller.run(AnalysisGoals(goals=("peaks",)))
    _wait(lambda: controller.panel.step_list.count() >= 1)
    started = time.monotonic()
    controller.shutdown()
    assert time.monotonic() - started < 10.0
    release.set()


def test_start_dialog_turns_choices_into_goals_and_needs_a_ready_brain(tmp_path: Path) -> None:
    _app()
    settings = InMemorySettingsRepository({})
    status = {"file": "film.tif", "measurement": "giwaxs"}
    dialog = AssistantStartDialog(settings, status=status)
    assert dialog.backend() == BACKEND_CLAUDE_CODE  # the Claude plan is the default brain
    assert not dialog.start_button.isEnabled() and "Checking" in dialog.credentials_label.text()
    dialog.set_status(False, "Claude Code is not signed in.")
    assert not dialog.start_button.isEnabled() and "not signed in" in dialog.credentials_label.text()
    changed: list = []
    dialog.backendChanged.connect(changed.append)
    dialog.brain_combo.setCurrentIndex(dialog.brain_combo.findData(BACKEND_API))
    assert changed == [BACKEND_API] and settings.get("assistant", "backend") == BACKEND_API

    set_up: list = []
    configured = AssistantStartDialog(settings, status=status, configure=lambda: set_up.append(True))
    configured.backendChanged.connect(changed.append)
    configured.setup_button.click()
    assert set_up == [True] and changed[-1] == BACKEND_API  # asked to check again after the set-up
    configured.set_status(True, "Claude API · claude-opus-5 · credentials: key.")
    assert configured.start_button.isEnabled()
    configured.goal_checks["peaks"].setChecked(False)
    configured.goal_checks["orientation"].setChecked(False)
    configured.ring_spin.setValue(0.4)
    configured.notes_edit.setPlainText("P3HT on Si")
    configured.auto_radio.setChecked(True)
    goals = configured.goals()
    assert goals.goals == ("ring_orientation", "crystallite_size")
    assert goals.ring_q == pytest.approx(0.4) and goals.instructions == "P3HT on Si"
    assert goals.permission == PERMISSION_AUTO
    configured._accept()
    saved = settings.get("assistant", "goals")
    assert [goal for goal in saved if goal in configured.goal_checks] == ["ring_orientation", "crystallite_size"]
    assert set(saved) - set(configured.goal_checks) == {"gisaxs_cut", "in_plane_spacing", "gisaxs_fit"}  # kept
    assert settings.get("assistant", "permission") == PERMISSION_AUTO
    for check in configured.goal_checks.values():
        check.setChecked(False)
    assert not configured.start_button.isEnabled()

    # The next dialog starts from the remembered choices; without a frame it cannot start.
    again = AssistantStartDialog(settings, status={"file": None})
    again.set_status(True, "ready")
    assert [key for key, check in again.goal_checks.items() if check.isChecked()] == ["ring_orientation", "crystallite_size"]
    assert again.auto_radio.isChecked() and not again.start_button.isEnabled()
    assert again.backend() == BACKEND_API


def test_settings_page_saves_the_key_and_tests_the_connection(tmp_path: Path) -> None:
    _app()
    from src.gimap.app.presentation.task_runner import TaskRunner

    settings = InMemorySettingsRepository({})
    services = _services(tmp_path, None, credentials="")
    tasks = TaskRunner()
    page = AssistantSettingsPage(settings, services, tasks)
    assert not page.test_button.isEnabled() and not page.remove_key_button.isEnabled()
    page.key_edit.setText("sk-ant-test")
    page.save_key_button.click()
    assert page.key_edit.text() == "" and page.source_label.text() == "API key saved in GIMaP"
    assert page.remove_key_button.isEnabled()
    page.model_combo.setEditText("claude-sonnet-5")
    page.effort_combo.setCurrentIndex(page.effort_combo.findText("medium"))
    page.effort_combo.activated.emit(page.effort_combo.currentIndex())
    page.test_button.click()
    assert tasks.wait(10)
    assert page.test_label.text() == "Connected: claude-sonnet-5 (medium) is available."
    assert settings.get("assistant", "model") == "claude-sonnet-5"
    page.turns_spin.setValue(12)
    assert page.permission_combo.currentData() == "preview"  # the default for new users
    page.permission_combo.setCurrentIndex(page.permission_combo.findData(PERMISSION_AUTO))
    assert settings.get("assistant", "max_turns") == 12
    assert settings.get("assistant", "permission") == PERMISSION_AUTO
    tasks.shutdown()


def test_settings_dialog_shows_feature_pages() -> None:
    _app()
    from PyQt5.QtWidgets import QLabel

    from src.gimap.app.presentation.settings_dialog import SettingsDialog

    dialog = SettingsDialog(
        preferences=InMemoryUserPreferencesRepository(),
        extra_pages=(("Assistant", "Claude", lambda parent: QLabel("assistant page", parent)),),
    )
    categories = [dialog.category_list.item(row).text() for row in range(dialog.category_list.count())]
    assert categories[-1] == "Assistant" and dialog.pages.count() == len(categories)
    dialog.category_list.setCurrentRow(len(categories) - 1)
    assert dialog.pages.currentWidget().findChild(QLabel, "") is not None


def test_claude_code_on_the_claude_plan_runs_analyze_end_to_end(analyze, tmp_path: Path) -> None:
    window, page, context = analyze
    controller = _controller(window, page, context, _services(tmp_path, None), backend=BACKEND_CLAUDE_CODE)
    context.settings.set("assistant", "code_model", "opus")
    assert controller.run(AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO))
    _wait(lambda: controller.outcome is not None)

    outcome = controller.outcome
    assert outcome.state == RUN_COMPLETED, outcome.message
    assert outcome.billing == BILLING_SUBSCRIPTION and outcome.cost_usd == pytest.approx(0.42)
    tools = [step.tool for step in outcome.steps]
    assert tools[:5] == ["get_status", "find_peaks", "compare_sectors", "ring_orientation", "crystallite_size"]
    assert tools[-1] == "submit_report"
    found = [peak.q for peak in outcome.results.peak_searches["radial"].peaks]
    assert min(abs(q - 0.40) for q in found) < 0.02 and min(abs(q - 1.65) for q in found) < 0.03
    assert page._lower_profile == "azimuthal"  # Analyze followed Claude Code's ring_orientation

    panel = controller.panel
    assert panel.model_label.text() == "Claude Code · opus"
    texts = [panel.step_list.item(row).text() for row in range(panel.step_list.count())]
    assert "Claude Code · claude-opus-5 · Claude subscription" in texts
    assert "Claude plan limit (five hour): 82% used" in texts
    assert "your Claude plan (≈ $0.42 at API prices)" in panel.usage_label.text()
    assert "your Claude plan" in panel.report_view.toPlainText()


def test_the_start_check_reports_whether_claude_code_can_run(tmp_path: Path) -> None:
    _app()
    window = QMainWindow()
    settings = InMemorySettingsRepository({})
    status = {"file": "film.tif", "measurement": "giwaxs"}

    def check(code_info: dict | None) -> AssistantStartDialog:
        controller = AssistantController(
            window, _services(tmp_path, None, code_info=code_info), settings=settings, automation=lambda: None,
        )
        dialog = AssistantStartDialog(settings, status=status)
        controller.check_brain(dialog, BACKEND_CLAUDE_CODE)
        assert controller.tasks.wait(10)
        QApplication.processEvents()
        return dialog

    ready = check(None)
    assert ready.start_button.isEnabled()
    assert "signed in with your Claude plan (max)" in ready.credentials_label.text()
    signed_out = check({"version": "2.1.281 (Claude Code)", "logged_in": False, "error": ""})
    assert not signed_out.start_button.isEnabled()
    assert "Claude Code 2.1.281 is not signed in" in signed_out.credentials_label.text()
    api_login = check({"version": "2.1.281", "logged_in": True, "auth_method": "console", "billing": "api", "error": ""})
    assert api_login.start_button.isEnabled() and "billed per token" in api_login.credentials_label.text()
    missing = check({"error": "Claude Code was not found."})
    assert not missing.start_button.isEnabled() and "not found" in missing.credentials_label.text()
    window.close()


def test_settings_choose_the_brain_and_sign_in_to_claude_code(tmp_path: Path) -> None:
    _app()
    from src.gimap.app.presentation.task_runner import TaskRunner

    settings = InMemorySettingsRepository({})
    logins: list = []
    tasks = TaskRunner()
    page = AssistantSettingsPage(settings, _services(tmp_path, None, logins=logins), tasks)
    assert page.code_radio.isChecked() and not page.api_radio.isChecked()
    section = page.code_section
    assert section.cli_edit.placeholderText() == "found: C:/Claude/claude.exe"
    section.check_button.click()
    assert tasks.wait(10)
    QApplication.processEvents()
    assert "signed in with your Claude plan (max)" in section.account_label.text()
    section.cli_edit.setText("D:/tools/claude.exe")
    section.login_button.click()
    assert logins == ["D:/tools/claude.exe"] and settings.get("assistant", "code_cli") == "D:/tools/claude.exe"
    assert "Finish signing in" in section.account_label.text()
    section.code_model_combo.setEditText("sonnet")
    section.code_model_combo.lineEdit().editingFinished.emit()
    assert settings.get("assistant", "code_model") == "sonnet"
    page.api_radio.setChecked(True)
    assert settings.get("assistant", "backend") == BACKEND_API
    page.code_radio.setChecked(True)
    assert settings.get("assistant", "backend") == BACKEND_CLAUDE_CODE
    tasks.shutdown()


def test_the_panel_stays_narrow_and_uses_the_theme(tmp_path: Path) -> None:
    from src.gimap.features.assistant.application import LlmUsage, Operation
    from src.gimap.features.assistant.presentation.operation_cards import OperationCard

    _app()
    from src.gimap.features.assistant.presentation import AssistantPanel

    panel = AssistantPanel()
    panel.usage(LlmUsage(52000, 3100, 40000, 2000), 0.42, 48.0, BILLING_SUBSCRIPTION)
    assert panel.usage_label.text() == (
        "94.0k tokens in (40.0k cached) · 3.1k out · your Claude plan (≈ $0.42 at API prices) · 48 s")
    assert panel.minimumSizeHint().width() <= 420, panel.minimumSizeHint()  # the line wraps; Analyze keeps its room
    # A change without a picture (confirm and automatic modes) has no empty picture column.
    card = OperationCard(Operation("set_halves", {"side": "mean"}, "Average both halves", why="They agree."))
    assert card.picture.isHidden() and card.state_label.property("gimapRole") == "info" and not card.state_label.styleSheet()
    assert card.origin_label.text() == "Suggested by the AI" and card.origin_label.toolTip().startswith("set_halves(")
    # Colours come from the theme (light and dark), never from fixed hex values.
    folder = Path(__file__).parents[1] / "src" / "gimap" / "features" / "assistant" / "presentation"
    for name in ("guided_text.py", "panel.py", "operation_cards.py"):
        source = (folder / name).read_text(encoding="utf-8").lower()
        assert not any(colour in source for colour in ("#2e7d32", "#1e88e5", "#ef6c00", "#c62828")), name
    card.deleteLater()
    panel.deleteLater()


def test_the_ai_flow_follows_the_interface_language(analyze, tmp_path: Path, monkeypatch) -> None:
    from src.gimap.app.presentation import i18n
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, apply_to

    window, page, context = analyze
    for english, chinese in {"Finished": "已完成", "Undo All": "全部撤销", "Copy": "复制", "Ask": "提问",
                             "Suggested by the AI": "由 AI 建议", "AI Assistant": "AI 助手",
                             "Report language": "报告语言"}.items():
        monkeypatch.setitem(i18n.ZH, english, chinese)  # the keys the Chinese table gets for the AI flow
    apply_language("zh")
    try:
        # A fresh profile in Chinese asks for the report in Chinese; the language names are never translated.
        dialog = AssistantStartDialog(InMemorySettingsRepository({}), status={"file": "film.tif", "measurement": "giwaxs"})
        apply_to(dialog, "zh")
        assert [dialog.language_combo.itemText(row) for row in range(dialog.language_combo.count())] == ["English", "中文"]
        assert dialog.language_combo.currentText() == "中文" and dialog.goals().language == "中文"
        chosen = InMemorySettingsRepository({"assistant": {"language": "English"}})
        assert AssistantStartDialog(chosen, status={"file": "film.tif"}).goals().language == "English"
        llm = ScriptedLlm([
            turn(call("propose_operations", operations=[
                {"tool": "set_sector_widths", "arguments": {"in_plane_half_width_deg": 5.0, "out_of_plane_half_width_deg": 5.0},
                 "title": "Narrower sectors", "why": "Separate them."}])),
            turn(report(("peaks", "done"))),
        ])
        controller = _controller(window, page, context, _services(tmp_path, llm))
        assert controller.run(AnalysisGoals(goals=("peaks",), permission=PERMISSION_AUTO))
        _wait(lambda: controller.outcome is not None)
        panel = controller.panel
        assert controller._dock.windowTitle() == "AI 助手"
        assert panel.state_label.text() == "已完成" and panel.copy_button.text() == "复制" and panel.follow_button.text() == "提问"
        assert panel.operations.undo_all_button.text() == "全部撤销"
        assert panel.operations.cards[0].origin_label.text() == "由 AI 建议"  # cards are rebuilt for every run
    finally:
        apply_language(DEFAULT_LANGUAGE)


def test_the_ai_report_is_saved_next_to_the_data(analyze, tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QFileDialog

    from src.gimap.app.presentation.components import visible_toasts
    from src.gimap.features.assistant.presentation import guided_text

    window, page, context = analyze
    monkeypatch.setattr(guided_text, "_SAVE_FOLDERS", {})
    proposed = []
    target = tmp_path / "saved" / "film_giwaxs_ai_report.html"
    target.parent.mkdir()

    def save_dialog(_parent, _title, start, _filters):
        proposed.append(start)
        return str(target), "HTML report (*.html)"

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(save_dialog))
    llm = ScriptedLlm([turn(report(("peaks", "done")))])
    controller = _controller(window, page, context, _services(tmp_path, llm))
    assert controller.run(AnalysisGoals(goals=("peaks",), permission=PERMISSION_AUTO))
    _wait(lambda: controller.outcome is not None)
    assert controller.save_report() == str(target) and target.with_suffix(".json").exists()
    assert Path(proposed[0]) == tmp_path / "film_giwaxs_ai_report.html"  # the frame's folder (no gimap_analysis yet)
    toast = visible_toasts(window)[-1]
    assert toast.property("level") == "ok" and toast.action_button.text() == "Open Folder"
    controller.services = dataclasses.replace(controller.services, save_text=_refuse)
    assert controller.save_report() is None and visible_toasts(window)[-1].property("level") == "error"


def _refuse(_path, _text):
    raise PermissionError(13, "Permission denied")
