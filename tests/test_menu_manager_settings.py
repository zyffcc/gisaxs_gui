"""Main menu bar: File / View / Tools / Help built from injected commands."""

from __future__ import annotations

import ast
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication, QDialog, QMainWindow

from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands, ToolWindows
from src.gimap.app.presentation.navigation import NAVIGATION_ITEMS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_APP = None


def _app():
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _titles(menu) -> list[str]:
    return [action.text() for action in menu.actions() if not action.isSeparator()]


def test_menu_bar_has_four_menus_and_hides_missing_commands() -> None:
    _app()
    window = QMainWindow()
    calls = []
    menus = MainMenuBar(
        window,
        MenuCommands(
            open_files=lambda: calls.append("open"),
            show_workspace=lambda key: calls.append(key),
            settings=lambda: calls.append("settings"),
        ),
        NAVIGATION_ITEMS,
    )

    assert [a.text() for a in window.menuBar().actions()] == ["&File", "&View", "&Tools", "&Help"]
    assert _titles(menus.file_menu) == ["&Open Data…"]
    assert "&Geometry Calibration…" not in _titles(menus.tools_menu)
    assert _titles(menus.tools_menu) == ["&Settings…"]
    view_titles = _titles(menus.view_menu)
    assert view_titles[:3] == ["Start", "Analyze", "Fitting"]  # the automatic analysis is part of Analyze
    assert menus.actions["workspace_home"].shortcut().toString() == "Ctrl+1"
    assert menus.actions["workspace_analyze"].shortcut().toString() == "Ctrl+2"

    menus.actions["open_files"].trigger()
    menus.actions["workspace_predict"].trigger()
    menus.actions["settings"].trigger()
    assert calls == ["open", "predict", "settings"]
    window.close()


def test_theme_actions_are_exclusive_and_follow_the_current_theme() -> None:
    _app()
    window = QMainWindow()
    state = {"mode": "light"}
    menus = MainMenuBar(
        window,
        MenuCommands(set_theme=lambda mode: state.update(mode=mode), current_theme=lambda: state["mode"]),
    )
    assert menus.actions["theme_light"].isChecked()
    menus.actions["theme_dark"].trigger()
    assert state["mode"] == "dark"
    menus.sync()
    assert menus.actions["theme_dark"].isChecked()
    assert not menus.actions["theme_light"].isChecked()
    window.close()


def test_tool_windows_reuse_one_modeless_instance() -> None:
    _app()
    window = QMainWindow()
    created = []

    def factory(parent):
        dialog = QDialog(parent)
        dialog.setModal(False)
        created.append(dialog)
        return dialog

    tools = ToolWindows(window)
    first = tools.show("xrr", factory)
    second = tools.show("xrr", factory)
    assert first is second
    assert len(created) == 1
    first.close()
    window.close()


def test_menu_composition_opens_feature_dialogs_lazily() -> None:
    source = (PROJECT_ROOT / "src/gimap/app/menus.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert not any("gimap.features" in (node.module or "") for node in top_level if isinstance(node, ast.ImportFrom))
    for module in (
        "src.gimap.features.calibration.presentation.dialog",
        "src.gimap.features.format_converter.presentation.dialog",
        "src.gimap.features.xrr.presentation",
    ):
        assert module in source
