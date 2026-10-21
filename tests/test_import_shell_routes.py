"""GUI shell entries request background imports before any frame is decoded; projects keep their route."""
from __future__ import annotations

import pytest
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.gimap.app import main_window as main_window_module
from tests.test_project import _window


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_args, **_kwargs: QMessageBox.Discard))
    shown = _window()
    yield shown
    shown.components.analyze_page.tasks.wait(30)
    QApplication.processEvents()
    shown.close()


def _import_only(window, monkeypatch, *, accepted=True):
    page = window.components.analyze_page
    called = []

    def request(paths, **_kwargs):
        called.append(list(paths))
        if accepted:
            page.frameImportStarted.emit()
        return accepted

    monkeypatch.setattr(page, "import_paths", request)
    monkeypatch.setattr(page, "add_paths", lambda *_args, **_kwargs: pytest.fail("Shell must not scan synchronously."))
    monkeypatch.setattr(page, "show_paths", lambda *_args, **_kwargs: pytest.fail("Shell must not decode synchronously."))
    return called


def test_recent_source_starts_an_import_and_navigates_before_frames_exist(window, monkeypatch, tmp_path):
    called = _import_only(window, monkeypatch)
    window.components.show_page("home")
    path = str(tmp_path / "large.nxs")
    assert window.menus.open_recent(path) is True
    assert called == [[path]]
    assert window.components.current_page_key() == "analyze"
    assert window.components.analyze_page.view_model.state.files == []


def test_home_open_and_folder_show_background_progress_without_waiting_for_list_growth(window, monkeypatch, tmp_path):
    called = _import_only(window, monkeypatch)
    components = window.components
    path = str(tmp_path / "large.nxs")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *_args, **_kwargs: ([path], "")))
    components.show_page("home")
    components.home_page.openRequested.emit()
    assert called == [[path]] and components.current_page_key() == "analyze"
    assert components.analyze_page.view_model.state.files == []
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", staticmethod(lambda *_args, **_kwargs: str(tmp_path)))
    components.show_page("home")
    components.home_page.folderRequested.emit()
    assert called[-1] == [str(tmp_path)] and components.current_page_key() == "analyze"


def test_home_cancel_keeps_the_home_page(window, monkeypatch):
    called = _import_only(window, monkeypatch)
    components = window.components
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", staticmethod(lambda *_args, **_kwargs: ([], "")))
    components.show_page("home")
    components.home_page.openRequested.emit()
    assert called == [] and components.current_page_key() == "home"


def test_window_drop_accepts_a_source_request_before_decoding_and_filters_unsupported_paths(window, monkeypatch, tmp_path):
    called = _import_only(window, monkeypatch)
    components = window.components
    components.show_page("home")
    path, notes = tmp_path / "large.nxs", tmp_path / "notes.txt"
    assert components.open_dropped([path, notes]) is True
    assert called == [[str(path)]] and components.current_page_key() == "analyze"
    assert components.analyze_page.view_model.state.files == []
    called.clear()
    components.show_page("home")
    notices = []
    monkeypatch.setattr(main_window_module, "show_toast", lambda _parent, text, **_kwargs: notices.append(text))
    assert components.open_dropped([notes]) is False
    assert called == [] and components.current_page_key() == "home"
    assert len(notices) == 1 and ".gimap" in notices[0]


def test_project_recent_and_drop_bypass_detector_import(window, monkeypatch, tmp_path):
    called = _import_only(window, monkeypatch)
    opened = []
    monkeypatch.setattr(window.menus, "open_project", lambda path: opened.append(str(path)) or True)
    project = str(tmp_path / "sample.gimap")
    assert window.menus.open_recent(project) is True
    assert window.components.open_dropped([project, tmp_path / "detector.nxs"]) is True
    assert opened == [project, project] and called == []


def test_refused_import_is_not_reported_as_an_accepted_drop(window, monkeypatch, tmp_path):
    called = _import_only(window, monkeypatch, accepted=False)
    window.components.show_page("home")
    path = str(tmp_path / "large.nxs")
    assert window.components.open_dropped([path]) is False
    assert called == [[path]] and window.components.current_page_key() == "home"


def test_home_recent_without_menus_uses_the_same_background_route(window, monkeypatch, tmp_path):
    called = _import_only(window, monkeypatch)
    monkeypatch.setattr(window, "menus", None)
    window.components.show_page("home")
    path = str(tmp_path / "large.nxs")
    window.components._recent_requested(path)
    assert called == [[path]] and window.components.current_page_key() == "analyze"
