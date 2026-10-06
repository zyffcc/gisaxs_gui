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


def test_menus_show_their_descriptions_and_name_what_they_open(tmp_path) -> None:
    _app()
    window = QMainWindow()
    state = {"collapsed": False}
    menus = MainMenuBar(
        window,
        MenuCommands(
            open_files=lambda: None, open_recent=lambda path: None, recent_paths=lambda: [],
            load_parameters=lambda: None, save_parameters=lambda: None,
            set_theme=lambda mode: None, change_font_size=lambda step: None, reset_font_size=lambda: None,
            set_sidebar_collapsed=lambda on: state.update(collapsed=on), sidebar_collapsed=lambda: state["collapsed"],
            ai_fitting_workspace=lambda: None, settings=lambda: None,
        ),
        NAVIGATION_ITEMS,
    )
    for menu in (menus.file_menu, menus.view_menu, menus.tools_menu, menus.help_menu, menus.recent_menu,
                 menus.labs_menu, menus.theme_menu, menus.font_menu):
        assert menu.toolTipsVisible(), menu.title()
    tools = _titles(menus.tools_menu)
    assert "1D &Predict — Fit Many Curves…" in tools and not any("Fit Settings" in text for text in tools)
    assert menus.actions["ai_fitting_workspace"].toolTip().startswith("1D Predict")

    # Not checkable (no check column that indents it); the text says what a click does.
    sidebar = menus.actions["collapse_sidebar"]
    assert not sidebar.isCheckable() and sidebar.text() == "Collapse &Sidebar"
    sidebar.trigger()
    assert state["collapsed"] is True and sidebar.text() == "Expand &Sidebar"
    state["collapsed"] = False  # the sidebar's own button
    menus.sync()
    assert sidebar.text() == "Collapse &Sidebar"
    window.close()


def test_open_recent_rows_name_the_folder_and_tag_folders_and_projects(tmp_path) -> None:
    _app()
    window = QMainWindow()
    folder = tmp_path / "beamtime"
    folder.mkdir()
    frame = folder / "frame_001.tif"
    frame.write_bytes(b"x")
    project = tmp_path / "sample.gimap"
    project.write_text("{}", encoding="utf-8")
    menus = MainMenuBar(window, MenuCommands(open_recent=lambda path: None,
                                             recent_paths=lambda: [project, frame, folder]))
    menus._fill_recent()
    actions = menus.recent_menu.actions()
    assert [action.text() for action in actions] == [
        f"&1  sample.gimap — {tmp_path.name} (project)",
        "&2  frame_001.tif — beamtime",
        f"&3  beamtime — {tmp_path.name} (folder)",
    ]
    assert [action.toolTip() for action in actions] == [str(project), str(frame), str(folder)]
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
