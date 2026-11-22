"""The Analyze menu replays a saved record on a worker even when no detector frame is open."""

import threading
import time
from concurrent.futures import CancelledError
from types import SimpleNamespace

import pytest
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow

from src.gimap.app.presentation import reproduction
from src.gimap.app.presentation.menu_bar import MainMenuBar, MenuCommands
from src.gimap.app.workspace_commands import WorkspaceCommands


def test_reproduction_error_preserves_the_path_and_digests_in_chinese():
    from src.gimap.app.presentation.i18n import apply_language

    try:
        apply_language("zh")
        assert reproduction._error_text("SHA-256 mismatch for scan.tif: expected 1abc, got 2def") == (
            "文件 scan.tif 的 SHA-256 不符。记录值为 1abc，当前值为 2def。")
        assert reproduction._error_text("Invalid recorded mask bitmap: Invalid run lengths.") == (
            "记录中的掩膜位图无效：掩膜的游程长度无效。")
    finally:
        apply_language("en")


@pytest.mark.parametrize("outcome", ["identical", "different", "missing", "cancel"])
def test_reproduce_menu(qt_application, tmp_path, monkeypatch, outcome):
    window = QMainWindow()
    selected, notices = [], []
    record, out = str(tmp_path / "example.json"), str(tmp_path)
    monkeypatch.setattr(reproduction, "ask_open_json", lambda *a, **kw: record)
    monkeypatch.setattr(reproduction.QFileDialog, "getExistingDirectory", lambda *args: out)
    monkeypatch.setattr(reproduction, "show_toast", lambda _, msg, **kw: notices.append((msg, kw)))
    monkeypatch.setattr(reproduction, "warn", lambda _, title, msg: notices.append((msg, {})))

    def replay(path, folder, *, should_cancel):
        assert (path, folder) == (record, out)
        assert threading.current_thread() is not threading.main_thread()
        if outcome == "missing":
            raise ValueError("Missing referenced file: /example/detector.tif")
        if outcome == "cancel":
            deadline = time.monotonic() + 5
            while not should_cancel() and time.monotonic() < deadline:
                time.sleep(0.005)
            assert should_cancel()
            raise CancelledError()
        return {"status": outcome, "output_dir": out}

    menus = SimpleNamespace(show_workspace=selected.append,
                            reproduce_from_record=lambda: reproduction.choose_record(window, replay))
    key = "analyze_reproduce"
    commands = MenuCommands(page_entries={key: WorkspaceCommands(menus).entries()[key]})
    bar = MainMenuBar(window, commands)
    submenu = next(a.menu() for a in bar.analyze_menu.actions() if a.text() == "Tools")
    action = bar.actions[key]
    assert action in submenu.actions()
    if outcome == "cancel":
        QTimer.singleShot(0, lambda: QApplication.activeModalWidget().reject())
    try:
        action.trigger()
        assert selected == ["analyze"]
        assert len(notices) == 1
        message, options = notices[0]
        if outcome == "cancel":
            assert message == "Reproduction cancelled."
        elif outcome == "missing":
            assert "/example/detector.tif" in message
        else:
            assert options["level"] == ("ok" if outcome == "identical" else "warning")
            assert options["action"][0] == "Open Folder"
    finally:
        window.close()
        window.deleteLater()
