"""Cross-area review items of the window shell: Save Project As after Clear, the quit question and close with a
2D prediction and a Series map, a calibration from Analyze that leaves no window behind, the automatic
analysis told which frames are shown, and the pages' run-time texts after a language switch."""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

import src.gimap.app.main_window as main_window_module
import src.gimap.app.menus as menus_module

ROOT = Path(__file__).resolve().parents[1]
GALAXI = ROOT / "tests" / "data" / "external" / "gisaxs_galaxi" / "galaxi_data.tif"


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


# -- Save Project As after Clear ------------------------------------------------------------------


def test_save_project_as_with_nothing_listed_suggests_a_real_folder(window, monkeypatch, tmp_path) -> None:
    menus, page = window.menus, window.components.analyze_page
    assert not page.view_model.state.files and menus.project_path == ""
    asked = []
    monkeypatch.setattr(menus_module, "ask_save_json", lambda _parent, _title, suggested, *_a: asked.append(suggested) or "")
    for last, folder in ((str(tmp_path), tmp_path), ("", Path.home()), (str(tmp_path / "gone"), Path.home())):
        # Analyze's public last folder (a property on the page class; the contract of the Analyze package)
        monkeypatch.setattr(type(page), "last_folder", property(lambda _self, last=last: last), raising=False)
        assert menus.save_project_as() is False  # cancelled
        assert Path(asked[-1]) == folder / "sample.gimap" and Path(asked[-1]).is_absolute(), last


# -- the quit question and close ------------------------------------------------------------------


def test_the_quit_question_takes_the_batch_kind_from_analyze(window, monkeypatch) -> None:
    components, page = window.components, window.components.analyze_page
    monkeypatch.setattr(page, "batch_running", lambda: True)
    monkeypatch.setattr(page, "_batch_map_only", False, raising=False)  # the public accessor counts, not the flag
    monkeypatch.setattr(page, "batch_kind", lambda: "series_map", raising=False)
    assert components.running_jobs() == ["Series map"]
    monkeypatch.setattr(page, "batch_kind", lambda: "export", raising=False)
    assert components.running_jobs() == ["Batch Export"]


def test_a_running_2d_prediction_is_asked_about_and_stopped_before_the_jobs(window, monkeypatch) -> None:
    components = window.components
    prediction = window.runtime.prediction
    assert components.running_jobs() == []
    monkeypatch.setattr(prediction, "prediction_running", lambda: True)
    assert components.running_jobs() == ["2D Prediction"]
    with monkeypatch.context() as patch:  # the Labs runtime has not started yet: nothing to ask about or stop
        patch.delattr(window, "runtime")
        assert components.running_jobs() == []
        components._stop_predictions()

    calls = []
    stop, jobs = prediction.stop_predictions, window.app_context.jobs
    shutdown = jobs.shutdown
    monkeypatch.setattr(prediction, "stop_predictions", lambda: (calls.append("prediction"), stop()))
    monkeypatch.setattr(components, "_stop_automatic_analysis", lambda: calls.append("automatic"))
    components._assistant = SimpleNamespace(shutdown=lambda: calls.append("assistant"), running=lambda: False)
    monkeypatch.setattr(jobs, "shutdown", lambda: (calls.append("jobs"), shutdown()))
    questions = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **_kwargs: questions.append(args) or QMessageBox.Yes)
    window.close()
    assert window._closed and questions[0][2] == "Still running: 2D Prediction. Stop them and quit?"
    assert calls[:3] == ["prediction", "assistant", "automatic"] and calls[-1] == "jobs"


# -- Geometry Calibration from Analyze --------------------------------------------------------------


@pytest.mark.parametrize("path", [None, GALAXI])
def test_a_calibration_from_analyze_leaves_no_window_behind(window, monkeypatch, path) -> None:
    from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog

    if path is not None and not path.exists():
        pytest.skip("GALAXI example not present")
    monkeypatch.setattr(GeometryCalibrationDialog, "exec_", lambda _self: 0)  # closed at once
    window.components._calibrate_for_analyze(path)
    end = time.monotonic() + 30  # with a frame: deleted once its reading thread has ended
    while window.findChildren(GeometryCalibrationDialog) and time.monotonic() < end:
        QApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        time.sleep(0.02)
    assert window.findChildren(GeometryCalibrationDialog) == []


def test_a_calibration_whose_frame_cannot_be_loaded_leaves_no_window_behind(window, monkeypatch) -> None:
    from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog

    def unreadable(_self, _path):
        raise OSError("not a detector frame")

    monkeypatch.setattr(GeometryCalibrationDialog, "load_image", unreadable)
    monkeypatch.setattr(GeometryCalibrationDialog, "exec_", lambda _self: pytest.fail("not shown"))
    with pytest.raises(OSError):  # reported by the caller (Analyze), as before
        window.components._calibrate_for_analyze(Path("C:/data/broken.tif"))
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert window.findChildren(GeometryCalibrationDialog) == []


# -- the automatic analysis and the frames shown ---------------------------------------------------


def test_the_automatic_analysis_learns_the_frames_shown_and_analyze_the_frames_analysed(window, monkeypatch) -> None:
    components = window.components
    shown, finished = [], []
    monkeypatch.setattr(components.guided, "frame_shown",
                        lambda path, frame=None, summed=None: shown.append((path, frame, summed)), raising=False)
    analysis = SimpleNamespace(path=Path("C:/data/run.nxs"), frame_index=393, frame_total=10)
    components._frame_shown(components.guided, analysis)
    assert shown == [(str(Path("C:/data/run.nxs")), 394, 10)]  # 1-based first frame, frames summed
    components._frame_shown(components.guided, SimpleNamespace(path=Path("C:/data/b.tif")))  # nothing known
    assert shown[-1] == (str(Path("C:/data/b.tif")), None, None)

    def automatic_finished(state, detail, *, show_results=True, path=None, frames=None):
        finished.append((state, show_results, path, frames))

    monkeypatch.setattr(main_window_module, "_automatic_outcome", lambda _report: ("ok", "GISAXS"))
    monkeypatch.setattr(components.analyze_page, "automatic_finished", automatic_finished, raising=False)
    frames = {"first": 394, "summed": 10, "total": 800}
    report = {"ok": True, "procedure": "gisaxs", "frame": "C:/data/run.nxs", "frames": frames}
    components.guided.finished.emit(report)
    assert finished and finished[-1][1:] == (True, "C:/data/run.nxs", frames)
    components.guided.finished.emit({"ok": True, "procedure": "geometry", "frame": "C:/data/b.tif"})  # a single file
    assert finished[-1][1:] == (False, "C:/data/b.tif", None)


# -- a language switch -------------------------------------------------------------------------------


def test_a_language_switch_has_every_page_compose_its_texts_again(window, monkeypatch) -> None:
    from src.gimap.app.presentation.i18n import apply_language, current_language

    components = window.components
    workspace = components.fitting_workspace
    refreshed = []
    pages = {"analyze": components.analyze_page, "compare": components.compare_page,
             "single": workspace.fit_page, "series": workspace.series_page}
    for name, page in pages.items():
        monkeypatch.setattr(page, "refresh_language", lambda name=name: refreshed.append(name), raising=False)
    assert current_language() == "en"
    try:
        apply_language("zh", [window])
        assert sorted(refreshed) == sorted(pages)  # once each, after the walker
        apply_language("zh", [window])  # not a switch: nothing to compose again
        assert len(refreshed) == len(pages)
        apply_language("en", [window])
        assert len(refreshed) == 2 * len(pages)
    finally:
        apply_language("en", [window])

    def broken():
        raise KeyError("a bug in one page")

    monkeypatch.setattr(components.analyze_page, "refresh_language", broken, raising=False)
    refreshed.clear()
    with pytest.raises(KeyError):  # reported, after the other pages were refreshed
        components.refresh_language("en")
    assert sorted(refreshed) == sorted(set(pages) - {"analyze"})

    monkeypatch.setattr(components.analyze_page, "refresh_language", lambda: refreshed.append("analyze"), raising=False)
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    window.close()  # a closed window is no longer told
    refreshed.clear()
    try:
        apply_language("zh", [])
        assert refreshed == []
    finally:
        apply_language("en", [])
