"""View ▸ Language (English / 中文) and the pages told after a language switch (wave 2b, shell-14 and the
``refresh_language()`` contract: the Labs view bindings and the tool windows are reached too)."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow, QMessageBox

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language, apply_to, current_language
from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands

LANGUAGE_MENU = {"&Language": "语言(&L)"}  # the shell's zh entry (zh_entries; the integrator merges it)


def _settle(seconds: float = 0.1) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def _keys(monkeypatch, entries: dict) -> None:
    for english, chinese in entries.items():
        if english not in i18n.ZH:
            monkeypatch.setitem(i18n.ZH, english, chinese)
            monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)


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
    apply_language(DEFAULT_LANGUAGE, [shown])
    shown.close()
    _settle(0.05)


def test_view_language_is_an_exclusive_choice_named_in_each_language(monkeypatch) -> None:
    QApplication.instance() or QApplication([])
    _keys(monkeypatch, LANGUAGE_MENU)
    host = QMainWindow()
    state = {"language": "en"}
    chosen = []
    menus = MainMenuBar(host, MenuCommands(set_language=lambda key: (chosen.append(key), state.update(language=key)),
                                           current_language=lambda: state["language"], set_theme=lambda _mode: None))
    english, chinese = menus.actions["language_en"], menus.actions["language_zh"]
    assert (english.text(), chinese.text()) == ("English", "中文")
    assert menus.language_menu.title() == "&Language"
    assert menus.language_menu.menuAction() in menus.view_menu.actions()  # View ▸ Language, after Theme
    assert menus.theme_menu.menuAction() in menus.view_menu.actions()
    assert english.isCheckable() and english.isChecked() and not chinese.isChecked()
    assert english.actionGroup() is chinese.actionGroup() and english.actionGroup().isExclusive()
    assert menus.language_menu.toolTipsVisible()

    chinese.trigger()
    assert chosen == ["zh"] and chinese.isChecked() and not english.isChecked()
    state["language"] = "en"  # switched back elsewhere (Settings ▸ Appearance)
    menus.sync()
    assert english.isChecked() and not chinese.isChecked()

    try:  # the names are never translated, in either direction; the menu title is
        apply_to(host, "zh")
        assert (english.text(), chinese.text()) == ("English", "中文")
        assert menus.language_menu.title() == "语言(&L)"
        apply_to(host, DEFAULT_LANGUAGE)
        assert (english.text(), chinese.text(), menus.language_menu.title()) == ("English", "中文", "&Language")
    finally:
        host.close()


def test_view_language_switches_like_settings_and_stays_in_step_with_it(window, monkeypatch) -> None:
    from src.gimap.app.presentation.settings_dialog import SettingsDialog
    from src.gimap.app.presentation.theme.appearance import Appearance

    bar = window.menus.menu_bar
    english, chinese = bar.actions["language_en"], bar.actions["language_zh"]
    preferences = window.app_context.preferences
    home = window.components.home_page
    assert current_language() == "en" and english.isChecked()

    chinese.trigger()  # View ▸ Language ▸ 中文
    assert current_language() == "zh" and Appearance(preferences).language == "zh"  # saved, as Settings does
    assert home.open_button.text() == i18n.ZH["Open Files…"]  # every open window, now
    assert chinese.isChecked() and not english.isChecked()
    assert (english.text(), chinese.text()) == ("English", "中文")

    dialog = SettingsDialog(window, preferences=preferences, settings=window.app_context.settings)
    try:
        assert dialog.language_combo.currentData() == "zh"  # Settings shows the choice made in the menu
        dialog.language_combo.setCurrentIndex(dialog.language_combo.findData("en"))  # and back from Settings
    finally:
        dialog.close()
    assert current_language() == "en" and Appearance(preferences).language == "en"
    assert english.isChecked() and not chinese.isChecked()  # the menu follows (language_changed)
    assert home.open_button.text() == "Open Files…"

    english.trigger()  # the language already shown: nothing changes
    assert current_language() == "en" and english.isChecked()


def test_a_language_switch_reaches_the_labs_bindings_and_the_tool_windows(window, monkeypatch) -> None:
    components, runtime = window.components, window.runtime
    refreshed = []

    class ToolWindow(QDialog):
        def refresh_language(self) -> None:
            refreshed.append("tool")

    tool = window.menus.tools.show("probe", lambda parent: ToolWindow(parent))
    pages = {"analyze": components.analyze_page, "compare": components.compare_page,
             "single": components.fitting_workspace.fit_page, "series": components.fitting_workspace.series_page,
             "prediction": runtime.prediction, "trainset": runtime.trainset}
    for name, page in pages.items():
        monkeypatch.setattr(page, "refresh_language", lambda name=name: refreshed.append(name), raising=False)
    # The Labs pages hand over to their bindings: told through the binding only, not twice.
    for name, page in {"predict page": components.predict_workspace, "trainset page": components.trainset_page}.items():
        monkeypatch.setattr(page, "refresh_language", lambda name=name: refreshed.append(name), raising=False)
    expected = sorted([*pages, "tool"])
    try:
        window.menus.set_language("zh")
        assert sorted(refreshed) == expected  # once each, after the walker
        window.menus.set_language("zh")  # not a switch
        assert len(refreshed) == len(expected)
        window.menus.set_language("en")
        assert len(refreshed) == 2 * len(expected)

        refreshed.clear()  # before the Labs runtime has started (the first moments after the window shows)
        with monkeypatch.context() as patch:
            patch.delattr(window, "runtime")
            window.menus.set_language("zh")
        assert {"predict page", "trainset page"} <= set(refreshed) and not {"prediction", "trainset"} & set(refreshed)
    finally:
        apply_language(DEFAULT_LANGUAGE, [window])
        tool.close()
    _settle(0.05)

    refreshed.clear()
    try:  # a closed tool window is gone (WA_DeleteOnClose in the real ones) or at least no longer listed
        tool.deleteLater()
        from PyQt5.QtCore import QCoreApplication, QEvent

        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert window.menus.tools.get("probe") is None
        window.menus.set_language("zh")
        assert "tool" not in refreshed and sorted(refreshed) == sorted(pages)
        refreshed.clear()
        window.components.refresh_language("zh")
        assert sorted(refreshed) == sorted(pages)  # the real Labs bindings define it: never the pages as well
    finally:
        apply_language(DEFAULT_LANGUAGE, [window])


def test_the_shells_messages_follow_the_interface_language(window, monkeypatch, tmp_path) -> None:
    import src.gimap.app.menus as menus_module

    _keys(monkeypatch, {"not a GIMaP project": "不是 GIMaP 项目", "Opened the project {name}.": "已打开项目 {name}。",
                        "GIMaP project (*.gimap);;All files (*)": "GIMaP 项目 (*.gimap);;所有文件 (*)"})
    warnings, toasts, filters = [], [], []
    monkeypatch.setattr(menus_module, "warn", lambda _parent, title, text: warnings.append((title, text)))
    monkeypatch.setattr(menus_module, "show_toast", lambda _parent, text, **_kwargs: toasts.append(text))
    monkeypatch.setattr(menus_module, "ask_open_json",
                        lambda _parent, _title, _folder, file_filter: filters.append(file_filter) or "")
    other = tmp_path / "other.gimap"
    other.write_text('{"format": "something else"}', encoding="utf-8")
    menus = window.menus
    try:
        menus.set_language("zh")
        assert menus.open_project(str(other)) is False
        assert warnings[-1] == (i18n.ZH["Open Project"], "无法打开 other.gimap：不是 GIMaP 项目")
        assert menus.open_project() is False and filters[-1] == "GIMaP 项目 (*.gimap);;所有文件 (*)"
        menus.set_language("en")
        menus.open_project(str(other))
        assert warnings[-1] == ("Open Project", "other.gimap could not be opened: not a GIMaP project")
        project = tmp_path / "sample.gimap"
        menus._write_project(project)
        assert menus.open_project(str(project)) is True
        assert toasts[-1] == "Opened the project sample.gimap."
    finally:
        apply_language(DEFAULT_LANGUAGE, [window])


def test_a_closed_main_window_is_not_told(window, monkeypatch) -> None:
    refreshed = []
    monkeypatch.setattr(window.runtime.prediction, "refresh_language", lambda: refreshed.append("prediction"),
                        raising=False)
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    window.close()
    try:
        apply_language("zh", [])
        assert refreshed == []
    finally:
        apply_language(DEFAULT_LANGUAGE, [])
