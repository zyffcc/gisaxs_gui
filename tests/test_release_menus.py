"""The release menu bar: File, Edit, View, Analyze, Fitting, Tools, Help.

Mnemonics are unique in every menu (English and Chinese); the keys of the pages shown in the menus never make a
key ambiguous (Qt would run neither command); the enabled state follows the pages each time a menu opens and a
disabled entry says why; an Analyze or Fitting entry shows its page first, then does what the page's button does.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QApplication, QMainWindow, QShortcut

ROOT = Path(__file__).resolve().parents[1]
UNDO_KEY = "\t" + QKeySequence("Ctrl+Z").toString(QKeySequence.NativeText)  # the page's key, shown after a tab
GALAXI = ROOT / "tests" / "data" / "external" / "gisaxs_galaxi" / "galaxi_data.tif"
needs_galaxi = pytest.mark.skipif(not GALAXI.exists(), reason="GALAXI example not present")


def _settle(seconds: float = 0.2) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def _mnemonic(text: str):
    """The letter after a single ``&`` (``&&`` is a literal ampersand), upper case; ``None`` without one."""
    index = 0
    while index < len(text) - 1:
        if text[index] == "&":
            if text[index + 1] != "&":
                return text[index + 1].upper()
            index += 2
            continue
        index += 1
    return None


def _menus(menu_bar):
    """The menu bar's own row and every menu and submenu under it, as ``(name, actions)``."""
    yield "menu bar", menu_bar.window.menuBar().actions()
    pending = [action.menu() for action in menu_bar.window.menuBar().actions()]
    while pending:
        menu = pending.pop()
        yield menu.title(), menu.actions()
        pending += [action.menu() for action in menu.actions() if action.menu() is not None]


def _clashes(menu_bar) -> list:
    clashes = []
    for name, actions in _menus(menu_bar):
        seen: dict[str, str] = {}
        for action in actions:
            letter = None if action.isSeparator() else _mnemonic(action.text())
            if letter is None:
                continue
            if letter in seen:
                clashes.append((name, letter, seen[letter], action.text()))
            seen[letter] = action.text()
    return clashes


@pytest.fixture()
def menu_bar():
    from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands
    from src.gimap.app.presentation.navigation import NAVIGATION_ITEMS
    from src.gimap.app.presentation.workspace_menus import ANALYZE_MENU, EDIT_MENU, FITTING_MENU, MenuEntry, menu_keys

    QApplication.instance() or QApplication([])

    def nothing(*_arguments):
        return None

    window = QMainWindow()
    pages = {key: MenuEntry(nothing) for layout in (EDIT_MENU, ANALYZE_MENU, FITTING_MENU) for key in menu_keys(layout)}
    names = ("open_files", "open_folder", "open_recent", "clear_recent", "save_parameters", "load_parameters",
             "open_project", "save_project", "save_project_as", "quit", "show_workspace", "set_theme", "set_language",
             "change_font_size", "reset_font_size", "set_sidebar_collapsed", "toggle_full_screen",
             "geometry_calibration", "format_converter", "convert_current_file", "xrr_extractor", "sem_analysis",
             "ai_fitting_workspace", "claude_assistant", "settings", "open_data_folder",
             "open_example", "report_problem", "open_error_log", "citation")
    from src.gimap.app.help_commands import HelpCommands

    commands = MenuCommands(page_entries=pages, examples=HelpCommands(None).examples(),
                            **{name: nothing for name in names})
    yield MainMenuBar(window, commands, NAVIGATION_ITEMS)
    window.close()
    window.deleteLater()


# -- the menu bar --------------------------------------------------------------------------------------


def test_the_menu_bar_has_seven_menus_in_order(menu_bar) -> None:
    titles = [action.text() for action in menu_bar.window.menuBar().actions()]
    assert titles == ["&File", "&Edit", "&View", "&Analyze", "F&itting", "&Tools", "&Help"]
    view = [action for action in menu_bar.view_menu.actions() if action.isSeparator() and action.text()]
    assert [action.text() for action in view] == ["Workspaces", "Labs"]  # the sidebar's sections
    assert menu_bar.actions["sem_analysis"].text() == "SE&M Particle Analysis…"
    assert "AI assistant" in menu_bar.actions["settings"].toolTip()
    assert menu_bar.labs_menu.menuAction().toolTip() == ("The settings of 2D Prediction and Trainset Build, "
                                                         "and Fitting's model")  # Load… replaces Fitting's model


def test_a_menu_with_no_command_is_not_added() -> None:
    from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands
    from src.gimap.app.presentation.workspace_menus import MenuEntry

    QApplication.instance() or QApplication([])
    window = QMainWindow()
    try:
        MainMenuBar(window, MenuCommands(page_entries={"fitting_fit": MenuEntry(lambda: None)}))
        assert [action.text() for action in window.menuBar().actions()] == ["F&itting", "&Help"]
    finally:
        window.close()


def test_mnemonics_are_unique_in_every_menu_in_english_and_in_chinese(menu_bar) -> None:
    from src.gimap.app.presentation.i18n import apply_language

    assert _clashes(menu_bar) == []
    for key in ("open_files", "edit_undo", "analyze_run_automatic", "fitting_fit", "sem_analysis", "settings"):
        assert _mnemonic(menu_bar.actions[key].text()), key
    assert _mnemonic("&Open Curve…") == "O"  # File ▸ Open Data on Fitting's Single analysis keeps Alt+F, O
    try:
        apply_language("zh", [menu_bar.window])
        for menu in menu_bar.menus:
            menu.aboutToShow.emit()  # the texts composed as a menu opens (Undo: …)
        titles = [action.text() for action in menu_bar.window.menuBar().actions()]
        assert titles == ["文件(&F)", "编辑(&E)", "视图(&V)", "分析(&A)", "拟合(&I)", "工具(&T)", "帮助(&H)"]
        assert menu_bar.actions["sem_analysis"].text() == "SEM 粒径分析(&M)…"
        assert _clashes(menu_bar) == []
        for key in menu_bar.workspace_menus.specs:  # every new entry has a key of its own in Chinese too
            assert _mnemonic(menu_bar.actions[key].text()), menu_bar.actions[key].text()
    finally:
        apply_language("en", [menu_bar.window])
    assert menu_bar.actions["sem_analysis"].text() == "SE&M Particle Analysis…"


def test_page_keys_are_shown_but_never_taken_by_the_menu(menu_bar) -> None:
    for key, keys in (("edit_undo", "Ctrl+Z"), ("edit_redo", "Ctrl+Y"), ("analyze_repeat", "F5"),
                      ("analyze_export", "Ctrl+E"), ("analyze_batch", "Ctrl+Shift+E"), ("fitting_fit", "Ctrl+Return")):
        action = menu_bar.actions[key]
        assert action.shortcuts() == [], key  # shown as text in the key column: only the page's QShortcut takes it
        assert action.text().endswith("\t" + QKeySequence(keys).toString(QKeySequence.NativeText)), key


def test_entries_follow_their_state_as_the_menu_opens_and_say_why_not() -> None:
    from src.gimap.app.presentation.i18n import apply_language
    from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands
    from src.gimap.app.presentation.workspace_menus import MenuEntry

    QApplication.instance() or QApplication([])
    window = QMainWindow()
    state = {"why": "Open a frame first.", "what": ""}
    ran = []
    entries = {
        "analyze_send_fitting": MenuEntry(lambda: ran.append("send"), lambda: state["why"]),
        "edit_undo": MenuEntry(lambda: ran.append("undo"), lambda: "", lambda: state["what"]),
    }
    try:
        bar = MainMenuBar(window, MenuCommands(page_entries=entries))
        send, undo = bar.actions["analyze_send_fitting"], bar.actions["edit_undo"]
        bar.analyze_menu.aboutToShow.emit()
        assert not send.isEnabled() and send.toolTip() == "Open a frame first."
        state["why"] = ""
        bar.analyze_menu.aboutToShow.emit()
        assert send.isEnabled() and send.toolTip().startswith("The horizontal cut (GISAXS)")
        state["what"] = "cut regions"
        bar.edit_menu.aboutToShow.emit()
        assert undo.text() == "&Undo: cut regions" + UNDO_KEY
        try:
            apply_language("zh", [window])
            bar.edit_menu.aboutToShow.emit()
            assert undo.text() == "撤销(&U)：cut regions" + UNDO_KEY  # the page says what, in the interface language
            state["why"] = "Open a frame first."
            bar.analyze_menu.aboutToShow.emit()
            assert send.toolTip() == "请先打开一帧。"
        finally:
            apply_language("en", [window])
        state["what"] = ""
        bar.edit_menu.aboutToShow.emit()
        assert undo.text() == "&Undo" + UNDO_KEY
        undo.trigger()
        assert ran == ["undo"]
    finally:
        window.close()


# -- in the real window ------------------------------------------------------------------------------------


def _window():
    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

    QApplication.instance() or QApplication([])
    geometry = DetectorGeometry(172e-6, 172e-6, 1.73, 597.1, 719.6, 1.34, 0.463)  # tests/data/external/README.md
    profiles = InMemoryInstrumentProfileRepository([InstrumentProfile("GALAXI", geometry, None, (1043, 981))])
    context = AppContext(settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
                         preferences=InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner(),
                         instrument_profiles=profiles)
    window = MainWindow(context)
    window.resize(1400, 900)
    window.show()
    end = time.monotonic() + 40
    while not window._initialization_completed and time.monotonic() < end:
        _settle(0.02)
    return window


@pytest.fixture
def window():
    shown = _window()
    yield shown
    shown.close()
    _settle(0.05)


def _navigate(window, key: str) -> None:
    runtime = getattr(window, "runtime", None)
    if runtime is not None:
        runtime.navigate(key)
    else:
        window.components.show_page(key)


def _press(window, page, keys: str) -> tuple[list, list, list]:
    """``keys`` pressed with the focus in ``page``: the page shortcuts that ran, the menu entries that ran (by
    key), and the ambiguous shortcuts."""
    from PyQt5.QtTest import QTest

    sequence = QKeySequence(keys)
    ran, menu, ambiguous = [], [], []
    for shortcut in window.findChildren(QShortcut):
        shortcut.activatedAmbiguously.connect(lambda shortcut=shortcut: ambiguous.append(shortcut.key().toString()))
        if shortcut.key() == sequence:
            shortcut.activated.connect(lambda shortcut=shortcut: ran.append(shortcut.parent()))
    for name, action in window.menus.menu_bar.actions.items():
        if sequence in action.shortcuts():
            action.triggered.connect(lambda _checked=False, name=name: menu.append(name))
    window.activateWindow()
    QApplication.setActiveWindow(window)
    page.setFocus()
    _settle(0.05)
    target = QApplication.focusWidget()
    if target is None or not (target is page or page.isAncestorOf(target)):
        target = page
    key = sequence[0]
    QTest.keyClick(target, Qt.Key(key & ~int(Qt.KeyboardModifierMask)), Qt.KeyboardModifiers(key & int(Qt.KeyboardModifierMask)))
    _settle(0.1)
    return ran, menu, ambiguous


def test_the_keys_of_analyze_and_fitting_are_never_ambiguous(window) -> None:
    components = window.components
    analyze = components.analyze_page
    _navigate(window, "analyze")
    _settle()
    for keys in ("Ctrl+Z", "F5", "Ctrl+E", "Ctrl+Shift+E"):
        ran, menu, ambiguous = _press(window, analyze, keys)
        assert ambiguous == [] and menu == [], (keys, ambiguous, menu)
        assert len(ran) == 1 and ran[0] is analyze, keys  # the page's own shortcut, once
    fit_page = components.fitting_workspace.fit_page
    _navigate(window, "fitting")
    components.fitting_workspace.show_context("single")
    _settle()
    for keys in ("Ctrl+Z", "Ctrl+Return"):
        ran, menu, ambiguous = _press(window, fit_page, keys)
        assert ambiguous == [] and menu == [], (keys, ambiguous, menu)
        assert len(ran) == 1 and ran[0] is fit_page, keys


def test_an_entry_shows_its_page_first_then_does_what_the_page_does(window, monkeypatch) -> None:
    components = window.components
    actions = window.menus.menu_bar.actions
    analyze, workspace = components.analyze_page, components.fitting_workspace
    calls = []
    monkeypatch.setattr(analyze, "save_settings", lambda: calls.append(("settings", components.current_page_key())))
    monkeypatch.setattr(workspace.fit_page, "open_curve_dialog",
                        lambda: calls.append(("curve", components.current_page_key(), workspace.fit_page.isVisible())))
    monkeypatch.setattr(workspace.series_page, "choose_folder",
                        lambda: calls.append(("folder", components.current_page_key(), workspace.series_page.isVisible())))
    for key in ("home", "compare"):
        _navigate(window, key)
        _settle()
        actions["analyze_save_settings"].trigger()
    assert calls == [("settings", "analyze"), ("settings", "analyze")]
    calls.clear()
    _navigate(window, "home")
    workspace.show_context("insitu")
    _settle()
    actions["fitting_open_curve"].trigger()
    _settle()
    actions["fitting_fit_folder"].trigger()
    _settle()
    assert calls == [("curve", "fitting", True), ("folder", "fitting", True)]

    # Edit follows the page shown and does not navigate.
    undone = []
    monkeypatch.setattr(analyze, "undo_setup", lambda: undone.append("analyze"))
    monkeypatch.setattr(workspace.fit_page, "undo", lambda: undone.append("fitting"))
    menu_bar = window.menus.menu_bar
    _navigate(window, "home")
    _settle()
    menu_bar.edit_menu.aboutToShow.emit()
    assert not actions["edit_undo"].isEnabled() and "Analyze" in actions["edit_undo"].toolTip()
    actions["edit_undo"].trigger()
    assert undone == [] and components.current_page_key() == "home"
    _navigate(window, "fitting")
    workspace.show_context("single")
    _settle()
    actions["edit_undo"].trigger()
    _navigate(window, "analyze")
    _settle()
    monkeypatch.setattr(analyze, "undo_names", lambda: ("cut regions", ""))
    analyze.undo_button.setEnabled(True)
    menu_bar.edit_menu.aboutToShow.emit()
    assert actions["edit_undo"].isEnabled() and actions["edit_undo"].text() == "&Undo: cut regions" + UNDO_KEY
    actions["edit_undo"].trigger()
    assert undone == ["fitting", "analyze"]


@needs_galaxi
def test_the_analyze_menu_follows_the_frame_shown(window) -> None:
    components = window.components
    page = components.analyze_page
    menu_bar = window.menus.menu_bar
    actions = menu_bar.actions
    menu_bar.analyze_menu.aboutToShow.emit()
    for key in ("analyze_send_fitting", "analyze_export", "analyze_repeat", "analyze_run_automatic", "analyze_clear"):
        assert not actions[key].isEnabled(), key
    assert actions["analyze_send_fitting"].toolTip() == "Open a frame first."
    assert actions["analyze_batch_folder"].isEnabled()
    assert not actions["analyze_stop_automatic"].isEnabled()

    page.add_paths([str(GALAXI)])
    page.tasks.wait(120)
    _settle()
    assert page.view_model.state.analysis is not None
    menu_bar.analyze_menu.aboutToShow.emit()
    for key in ("analyze_send_fitting", "analyze_export", "analyze_repeat", "analyze_run_automatic",
                "analyze_clear", "analyze_batch"):
        assert actions[key].isEnabled(), (key, actions[key].toolTip())
    assert actions["analyze_send_fitting"].toolTip().startswith("The horizontal cut (GISAXS)")
    assert not actions["analyze_send_series"].isEnabled()  # one frame: no series
    assert not actions["analyze_send_compare"].isEnabled()  # no Series map built

    page.clear_files()
    _settle()
    menu_bar.analyze_menu.aboutToShow.emit()
    assert not actions["analyze_send_fitting"].isEnabled() and not actions["analyze_clear"].isEnabled()


def test_a_state_check_that_fails_is_reported_once_and_not_called_starting(monkeypatch) -> None:
    import sys

    from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands
    from src.gimap.app.presentation.workspace_menus import CHECK_FAILED, MenuEntry

    QApplication.instance() or QApplication([])
    reported = []
    monkeypatch.setattr(sys, "excepthook", lambda *exc_info: reported.append(exc_info[0]))

    def broken() -> str:
        raise AttributeError("'Page' object has no attribute 'run_pipeline_button'")  # a renamed attribute

    window = QMainWindow()
    try:
        bar = MainMenuBar(window, MenuCommands(page_entries={"analyze_repeat": MenuEntry(lambda: None, broken),
                                                             "analyze_export": MenuEntry(lambda: None)}))
        for _ in range(3):
            bar.analyze_menu.aboutToShow.emit()
        repeat, export = bar.actions["analyze_repeat"], bar.actions["analyze_export"]
        assert not repeat.isEnabled() and repeat.toolTip() == CHECK_FAILED and "starting" not in repeat.toolTip()
        assert reported == [AttributeError]  # once, to ErrorGuard; not on every open of the menu
        assert export.isEnabled()  # the other entries still follow their state
    finally:
        window.close()


def test_in_situ_start_follows_a_model_set_up_after_the_folder_was_listed(tmp_path) -> None:
    from types import SimpleNamespace

    from src.gimap.app.workspace_commands import NO_SERIES_CURVES, NO_SERIES_MODEL, WorkspaceCommands
    from src.gimap.features.fitting.application.single_fit import FitModel
    from tests.test_fit_series import _model, _pages, _series

    single, series = _pages()
    try:
        workspace = SimpleNamespace(series_page=series, fit_page=single)
        commands = WorkspaceCommands(SimpleNamespace(window=SimpleNamespace(
            components=SimpleNamespace(fitting_workspace=workspace))))
        single.set_model(FitModel(()))
        assert commands._series_start() == NO_SERIES_CURVES
        assert series.open_series(_series(tmp_path))
        assert commands._series_start() == NO_SERIES_MODEL
        single.set_model(_model(5.2))  # set up in Single analysis: the series page is not refreshed until shown
        assert commands._series_start() == ""
    finally:
        series.dispose()
        single.dispose()


def test_after_alt_the_page_keys_run_no_menu_entry_from_any_page(window, monkeypatch) -> None:
    from PyQt5.QtTest import QTest

    components = window.components
    analyze, fit_page = components.analyze_page, components.fitting_workspace.fit_page
    ran = []
    for owner, name in ((analyze, "run_analysis"), (analyze, "export_current"), (analyze, "batch_export_dialog"),
                        (analyze, "undo_setup"), (fit_page, "run_fit"), (fit_page, "undo")):
        monkeypatch.setattr(owner, name, lambda name=name: ran.append(name))
    menu_bar = window.menuBar()
    for page in ("compare", "home"):
        _navigate(window, page)
        _settle()
        for keys in ("F5", "Ctrl+E", "Ctrl+Shift+E", "Ctrl+Z", "Ctrl+Return"):
            window.activateWindow()
            QApplication.setActiveWindow(window)
            menu_bar.setFocus()  # as after Alt: the menu bar in keyboard mode has the focus
            QTest.keyClick(menu_bar, Qt.Key_Alt)
            _settle(0.05)
            key = QKeySequence(keys)[0]
            QTest.keyClick(QApplication.focusWidget() or menu_bar, Qt.Key(key & ~int(Qt.KeyboardModifierMask)),
                           Qt.KeyboardModifiers(key & int(Qt.KeyboardModifierMask)))
            _settle(0.05)
            QTest.keyClick(QApplication.focusWidget() or menu_bar, Qt.Key_Escape)
            _settle(0.05)
            assert ran == [] and components.current_page_key() == page, (page, keys, ran)


def test_a_frame_open_but_not_analysed_is_not_called_no_frame(window, tmp_path) -> None:
    from src.gimap.app.workspace_commands import NO_FRAME, NOT_ANALYSED

    page = window.components.analyze_page
    menu_bar = window.menus.menu_bar
    actions = menu_bar.actions
    broken = tmp_path / "not_a_detector_image.tif"
    broken.write_bytes(b"this is not a TIFF")  # invented: a file GIMaP cannot read
    page.add_paths([str(broken)])
    page.tasks.wait(60)
    _settle()
    assert page.view_model.current_path is not None and page.view_model.state.analysis is None
    for menu in [menu_bar.analyze_menu] + [action.menu() for action in menu_bar.analyze_menu.actions() if action.menu()]:
        menu.aboutToShow.emit()
    assert actions["analyze_repeat"].isEnabled()  # a frame is open
    for key in ("analyze_export", "analyze_send_fitting", "analyze_save_image", "analyze_save_map"):
        assert not actions[key].isEnabled(), key
        assert actions[key].toolTip() == NOT_ANALYSED != NO_FRAME, (key, actions[key].toolTip())
    page.clear_files()
    _settle()
