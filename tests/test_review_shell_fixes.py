"""Review fixes of the window shell: the Labs status bar, Clear and the automatic analysis, File ▸ Open Data on
Fitting, Recent projects, Process with AI, the quit question and a curve dropped on Fitting."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QEventLoop, QMimeData, QPoint, QPointF, Qt, QTimer, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

import src.gimap.app.main_window as main_window_module
import src.gimap.app.menus as menus_module


def _settle(seconds: float = 0.2) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture
def window():
    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )

    QApplication.instance() or QApplication([])
    context = AppContext(settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
                         preferences=InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner(),
                         instrument_profiles=InMemoryInstrumentProfileRepository([]))
    shown = MainWindow(context)
    shown.resize(1400, 900)
    shown.show()
    end = time.monotonic() + 40
    while not shown._initialization_completed and time.monotonic() < end:
        _settle(0.02)
    yield shown
    shown.close()
    _settle(0.05)


def _navigate(window, key: str) -> None:
    runtime = getattr(window, "runtime", None)
    if runtime is not None:
        runtime.navigate(key)
    else:
        window.components.show_page(key)


# -- the Labs status bar -----------------------------------------------------------------------------


def test_the_labs_status_bar_keeps_step_instructions_and_says_nothing_on_parameter_edits(window) -> None:
    from src.gimap.app.presentation.i18n import ZH, apply_language

    runtime = window.runtime
    statusbar = window.statusbar
    _navigate(window, "trainset")
    _settle()
    messages = []
    statusbar.messageChanged.connect(messages.append)
    runtime.trainset.status_updated.emit("Click the direct-beam position on the full detector")
    # No timeout: the instruction stays while its mode lasts (QStatusBar keeps no running timer).
    assert statusbar.currentMessage() == "Click the direct-beam position on the full detector"
    assert not [timer for timer in statusbar.findChildren(QTimer) if timer.isActive()]
    messages.clear()
    runtime.trainset.parameters_changed.emit("Trainset parameters", {"roi": 1})
    assert messages == [] and statusbar.currentMessage() == "Click the direct-beam position on the full detector"
    assert runtime.current_parameters["Trainset parameters"] == {"roi": 1}
    try:
        apply_language("zh")
        runtime.trainset.status_updated.emit("Trainset settings restored")
        assert statusbar.currentMessage() == ZH["Trainset settings restored"]
    finally:
        apply_language("en")


def test_the_status_bar_sits_under_the_page_and_the_sidebar_keeps_its_height(window) -> None:
    components = window.components
    assert window.statusbar.parentWidget() is window.mainContentWidget
    heights = {}
    for key in ("analyze", "trainset", "predict", "home"):
        components.show_page(key)
        _settle(0.1)
        heights[key] = components.sidebar.height()
        assert window.statusbar.isVisible() == (key in ("trainset", "predict")), key
    assert len(set(heights.values())) == 1, heights
    assert heights["trainset"] == window.centralwidget.height()
    statusbar = window.statusbar
    status_bottom = statusbar.mapTo(window, statusbar.rect().bottomLeft()).y()
    components.show_page("trainset")
    _settle(0.1)
    assert statusbar.mapTo(window, QPoint(0, 0)).x() >= components.sidebar.width()
    assert status_bottom <= window.centralwidget.geometry().bottom()


# -- Analyze ▸ Clear and the automatic analysis -------------------------------------------------------


def test_clear_in_analyze_tells_the_automatic_analysis(window, monkeypatch) -> None:
    components = window.components
    cleared = []
    monkeypatch.setattr(components.guided, "files_cleared", lambda: cleared.append(True), raising=False)
    components.analyze_page.clear_files()
    assert cleared == [True]


# -- File ▸ Open Data on Fitting -----------------------------------------------------------------------


def test_open_data_says_open_curve_on_fittings_single_analysis(window) -> None:
    components = window.components
    menu_bar = window.menus.menu_bar
    action = menu_bar.actions["open_files"]
    _navigate(window, "fitting")
    components.fitting_workspace.show_context("single")
    _settle()
    menu_bar.file_menu.aboutToShow.emit()
    assert action.text() == "Open Curve…" and "1D curve" in action.toolTip()
    components.fitting_workspace.show_context("insitu")
    _settle()
    menu_bar.file_menu.aboutToShow.emit()
    assert action.text() == "&Open Data…" and "detector frames" in action.toolTip()
    components.fitting_workspace.show_context("single")
    _navigate(window, "home")
    _settle()
    menu_bar.file_menu.aboutToShow.emit()
    assert action.text() == "&Open Data…"


# -- Recent projects -----------------------------------------------------------------------------------


def test_a_project_saved_and_opened_again_is_one_recent_entry(window, monkeypatch, tmp_path) -> None:
    menus = window.menus
    project = tmp_path / "A.gimap"
    assert menus._write_project(project)
    forward = str(project).replace("\\", "/")  # what the Open dialog returns on Windows
    monkeypatch.setattr(menus_module, "ask_open_json", lambda *_args, **_kwargs: forward)
    monkeypatch.setattr(menus_module, "show_toast", lambda *_args, **_kwargs: None)
    assert menus.open_project()
    assert menus._recent_projects() == [str(project)]
    assert menus.project_path == str(project)
    # Entries stored both ways before this fix collapse into one.
    window.app_context.preferences.set(menus_module.RECENT_PROJECTS_KEY, [forward, str(project), ""])
    assert menus._recent_projects() == [str(project)]
    assert [str(path) for path in menus.recent_paths()].count(str(project)) == 1


# -- Process with AI from the Tools menu ---------------------------------------------------------------


def test_process_with_ai_from_the_menu_takes_the_results_notes(window) -> None:
    components = window.components
    started = []
    components._assistant = SimpleNamespace(start=lambda notes="": started.append(notes), running=lambda: False,
                                            shutdown=lambda: None)
    try:
        components.guided.notes_edit.setPlainText("P03, GIWAXS, alpha_i = 0.4 deg")
        window.menus.menu_bar.actions["claude_assistant"].trigger()
        assert started == ["P03, GIWAXS, alpha_i = 0.4 deg"]
    finally:
        components._assistant = None


# -- quitting ------------------------------------------------------------------------------------------


def test_the_quit_question_names_a_series_map_as_such(window, monkeypatch) -> None:
    components = window.components
    page = components.analyze_page
    monkeypatch.setattr(page, "batch_running", lambda: True)
    monkeypatch.setattr(page, "_batch_map_only", True, raising=False)
    assert components.running_jobs() == ["Series map"]
    monkeypatch.setattr(page, "_batch_map_only", False, raising=False)
    assert components.running_jobs() == ["Batch Export"]


def test_the_wait_on_quit_takes_no_user_input_and_the_window_is_disabled(window, monkeypatch) -> None:
    components = window.components
    flags = []

    class _Core:
        @staticmethod
        def processEvents(*args):  # noqa: N802 - Qt API
            flags.append(args)

    calls = iter([True, True, False])
    monkeypatch.setattr(main_window_module, "QCoreApplication", _Core)
    monkeypatch.setattr(components, "guided", SimpleNamespace(stop=lambda: None, running=lambda: next(calls, False)))
    components._stop_automatic_analysis(timeout_s=5)
    assert flags and all(args == (QEventLoop.ExcludeUserInputEvents,) for args in flags)

    enabled = []
    shutdown = components.shutdown
    monkeypatch.setattr(components, "shutdown", lambda: (enabled.append(window.isEnabled()), shutdown()))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    window.close()
    assert enabled == [False]


# -- a curve dropped on Fitting ---------------------------------------------------------------------------


def _drop(widget, paths) -> None:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    pos = widget.rect().center()
    enter = QDragEnterEvent(pos, Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, enter)
    assert enter.isAccepted()
    QApplication.sendEvent(widget, QDropEvent(QPointF(pos), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))


def test_a_curve_dropped_on_fitting_keeps_the_halves_chosen_there(window, tmp_path: Path) -> None:
    curve = tmp_path / "cut_fit_input.dat"
    curve.write_text("-0.3 6 1\n-0.2 8 1\n-0.1 10 1\n0.1 10 1\n0.2 8 1\n0.3 6 1\n", encoding="utf-8")
    components = window.components
    fit_page = components.fitting_workspace.fit_page
    _navigate(window, "fitting")
    components.fitting_workspace.show_context("insitu")
    _settle()
    fit_page.session.side = "positive"
    _drop(window, [curve])
    _settle()
    assert fit_page.isVisible() and fit_page.session.curve is not None
    assert Path(fit_page.session.curve.path) == curve
    assert fit_page.session.side == "positive"
