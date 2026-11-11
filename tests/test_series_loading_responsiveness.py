"""Controlled slow I/O must leave the series GUI responsive and cancellable."""

import threading
import time

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from tests.test_fit_series import _model, _pages, _series, _wait


def _until(predicate, seconds=15):
    end = time.monotonic() + seconds
    while not predicate() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)
    assert predicate()


def _settle(series):
    _until(lambda: not getattr(series, "_listing_busy", False))


def test_start_reads_in_worker_and_stop_is_received_during_read(tmp_path, monkeypatch):
    single, series = _pages()
    release, entered = threading.Event(), threading.Event()
    timer = QTimer(series)
    timer.setSingleShot(True)
    try:
        folder = _series(tmp_path, (5., 5.4))
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_model(5.2))
        series.open_series(folder)
        _settle(series)
        load = series.view_model.read_series_curve
        reads, stopped = [], []

        def slow(request):
            reads.append(threading.current_thread())
            entered.set()
            assert release.wait(15), "test did not release the reader"
            return load(request)

        monkeypatch.setattr(series.view_model, "read_series_curve", slow)
        timer.timeout.connect(lambda: (series.stop(), stopped.append(True)))
        timer.start(30)
        series.start()
        _until(lambda: entered.is_set() and bool(stopped))
        assert series._stop.is_set() and not release.is_set()
        assert all(thread is not threading.main_thread() for thread in reads)
        release.set()
        _wait(series)
        assert not series.running and not series.fits
    finally:
        timer.stop()
        release.set()
        series.dispose()
        single.dispose()


def test_directory_discovery_returns_before_io_and_old_result_is_ignored(tmp_path, monkeypatch):
    single, series = _pages()
    release, entered = threading.Event(), threading.Event()
    try:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        old = _series(first, (5., 5.4))
        new = _series(second, (6.,))
        discover = series.view_model.storage.discover_insitu_frames

        def slow(request):
            if request.root == old:
                entered.set()
                assert release.wait(15), "test did not release the reader"
            return discover(request)

        monkeypatch.setattr(series.view_model.storage, "discover_insitu_frames", slow)
        assert series.open_series(old)
        _until(entered.is_set)
        assert series.open_series(new)
        _settle(series)
        assert series.paths == [str(new / "run_00001_fit_input.dat")]
        release.set()
        series.loading_tasks.wait()
        assert series.folder == str(new) and len(series.paths) == 1
    finally:
        release.set()
        series.dispose()
        single.dispose()


def test_stop_cancels_discovery_and_project_restore_cannot_leak_to_new_folder(tmp_path, monkeypatch):
    single, series = _pages()
    release = threading.Event()
    try:
        first, second = tmp_path / "first", tmp_path / "second"
        first.mkdir()
        second.mkdir()
        old, new = _series(first), _series(second, (6., 6.4))
        discover = series.view_model.storage.discover_insitu_frames

        def slow(request):
            if request.root == old:
                assert release.wait(15), "test did not release the reader"
            return discover(request)

        monkeypatch.setattr(series.view_model.storage, "discover_insitu_frames", slow)
        saved = {"folder": str(old), "first": 3, "last": 5, "every": 2, "side": "negative"}
        assert series.apply_project_state(saved) == []
        assert all(series.project_state()[key] == value for key, value in saved.items())
        assert not series.first_spin.isEnabled() and not series.side_combo.isEnabled()
        series.start_same.click()
        series.method_global.click()
        series.follow_check.setChecked(False)
        pending = series.project_state()
        assert pending["start"] == "same" and pending["method"] == "global" and not pending["follow"]
        series.stop()
        assert not series._listing_busy and not series.paths
        assert series.status_label.text() == "Series loading cancelled."
        series.open_series(new)
        pending = series.project_state()
        assert pending["folder"] == str(new) and pending["first"] == 1 and pending["last"] == 0
        _settle(series)
        release.set()
        series.loading_tasks.wait()
        assert series.folder == str(new) and series._listed() == [0, 1]
        assert series.halves_source() != "project"
    finally:
        release.set()
        series.dispose()
        single.dispose()


def test_headers_and_detector_cuts_use_workers_but_recipe_uses_gui_thread(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np

    from tests.test_fit_series import Q_NM

    calls = []
    recipe = SimpleNamespace(sum_count=1, halves="mean", record={})
    token = str(tmp_path / "frame.tif")
    (tmp_path / "frame.tif").write_bytes(b"synthetic detector placeholder")

    class Frames:
        def recipe(self):
            calls.append(("recipe", threading.current_thread()))
            return recipe, ""

        def list(self, _folder):
            calls.append(("discovery", threading.current_thread()))
            return [(token, ())]

        def grouped(self, items, _count):
            return items

        def cut(self, actual, _token, _summed):
            assert actual is recipe
            calls.append(("cut", threading.current_thread()))
            return {"q": Q_NM / 10, "intensity": np.ones(Q_NM.size), "sigma": np.ones(Q_NM.size),
                    "observation": {"source": "synthetic", "halves": "mean"}}

    single, series = _pages(Frames())
    try:
        assert series.open_series(tmp_path, "*.tif")
        _settle(series)
        assert series.paths == [token]
        assert any(name == "cut" for name, _ in calls)
        assert all((thread is threading.main_thread()) == (name == "recipe") for name, thread in calls)
        assert series._preview_curves[(token, "angstrom")].q.max() == Q_NM.max()
        folder = _series(tmp_path)
        read = series.view_model.stated_q_unit

        def header(path):
            calls.append(("header", threading.current_thread()))
            return read(path)

        monkeypatch.setattr(series.view_model, "stated_q_unit", header)
        series.open_series(folder, "*_fit_input.dat")
        _settle(series)
        assert any(name == "header" for name, _ in calls)
        assert all(thread is not threading.main_thread() for name, thread in calls if name == "header")
    finally:
        series.dispose()
        single.dispose()


def test_late_selected_frame_preview_does_not_replace_new_series(tmp_path, monkeypatch):
    single, series = _pages()
    release, entered = threading.Event(), threading.Event()
    try:
        first, second = tmp_path / "first", tmp_path / "second"
        first.mkdir()
        second.mkdir()
        old = _series(first, (5., 5.1, 5.2, 5.3, 5.4, 5.5))
        new = _series(second, (6.,))
        single.open_curve(new / "run_00001_fit_input.dat")
        current = series.view_model.state.current_curve
        series.open_series(old)
        _settle(series)
        series._stages_timer.stop()
        load = series.view_model.read_series_curve

        def slow(request):
            if request.path == old / "run_00003_fit_input.dat":
                entered.set()
                assert release.wait(15), "test did not release the reader"
            return load(request)

        monkeypatch.setattr(series.view_model, "read_series_curve", slow)
        series.frame_list.setCurrentRow(2)
        _until(entered.is_set)
        assert series.frame_plot.empty_overlay.label.text() == "Loading frame …"
        series.open_series(new)
        _settle(series)
        release.set()
        series.preview_tasks.wait()
        assert series.folder == str(new) and series._selected == 0
        assert all(path.startswith(str(new)) for path, _unit in series._preview_curves)
        assert series.frame_plot.figure_state()["title"] == "1 · run_00001_fit_input.dat"
        assert series.view_model.state.current_curve is current
    finally:
        release.set()
        series.dispose()
        single.dispose()


def test_watch_discovery_does_not_block_stop_or_add_frames_after_stop(tmp_path, monkeypatch):
    single, series = _pages()
    release, entered = threading.Event(), threading.Event()
    try:
        folder = _series(tmp_path, (5.,))
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_model(5.2))
        series.open_series(folder)
        _settle(series)
        series.watch_check.setChecked(True)
        series.start()
        _until(series._waiting_for_curves)
        discover = series.view_model.storage.discover_insitu_frames

        def slow(request):
            assert threading.current_thread() is not threading.main_thread()
            entered.set()
            assert release.wait(15), "test did not release the reader"
            return discover(request)

        monkeypatch.setattr(series.view_model.storage, "discover_insitu_frames", slow)
        series._look_for_new()
        _until(entered.is_set)
        series.stop()
        assert not series.running and series._stop.is_set()
        release.set()
        series.watch_tasks.wait()
        assert len(series.paths) == 1 and not series.queue
    finally:
        release.set()
        series.dispose()
        single.dispose()
