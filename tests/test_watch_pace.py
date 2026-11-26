"""A folder watch reduces a few new frames at a time (``bindings/watch_pace.py``): the others wait in line and every
frame still reaches the map, in order; stopping the watch leaves no frame of the map waiting. On the public GALAXI
frame (``tests/insitu_frames.py``)."""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.gimap.features.analyze.application.batch_job import watch_frames_at_once
from tests.insitu_frames import GALAXI, write_frame
from tests.test_analyze_workspace import _done
from tests.test_insitu_online import _english, _look, _watching  # noqa: F401 (the fixture)
from tests.test_live_following import _close


def test_how_many_new_frames_a_watch_reduces_at_once() -> None:
    two_megapixels = 1679 * 1475
    assert watch_frames_at_once(cores=32, free_bytes=8 * 2**30, frame_pixels=two_megapixels) == 4
    assert watch_frames_at_once(cores=8, free_bytes=8 * 2**30, frame_pixels=two_megapixels) == 2
    assert watch_frames_at_once(cores=32, free_bytes=600 * 2**20, frame_pixels=two_megapixels) == 1  # memory short
    assert watch_frames_at_once(cores=1, free_bytes=None, frame_pixels=two_megapixels) == 1


@pytest.mark.skipif(not GALAXI.is_file(), reason="public test frames (tests/data/external) are not available")
def test_a_watch_reduces_a_few_frames_at_once_and_every_frame_reaches_the_map(tmp_path, monkeypatch) -> None:
    page, folder = _watching(tmp_path)
    try:
        model = page.view_model
        monkeypatch.setattr(model, "resources", lambda: (8, None))  # two at once
        series_row, lock, running, most = model.series_row, threading.Lock(), [0], [0]

        def counted(request, key):
            with lock:
                running[0] += 1
                most[0] = max(most[0], running[0])
            try:
                time.sleep(0.05)
                return series_row(request, key)
            finally:
                with lock:
                    running[0] -= 1

        monkeypatch.setattr(model, "series_row", counted)
        for index in range(1, 10):
            write_frame(folder, index, jump_at=6)
        _look(page)
        assert most[0] == 2 and page.watch_frames_waiting() == 0
        assert page._series_map is not None and page._series_map.rows == 10
        assert "still to be added" not in page.series_title_label.text()  # (all in)
        page._watch_series_jobs = {"watch-series:a:0", "watch-series:b:0"}  # two frames still on their way
        page._series_describe()
        assert "2 new frames still to be added" in page.series_title_label.text()
        page._watch_series_jobs = set()
        names = [row[4][0].name for row in page._series_rows]
        assert names == sorted(names) and len(set(names)) == 10
    finally:
        page.stop_watch()
        _close(page)


@pytest.mark.skipif(not GALAXI.is_file(), reason="public test frames (tests/data/external) are not available")
def test_stopping_the_watch_leaves_no_frame_of_the_map_waiting(tmp_path, monkeypatch) -> None:
    page, folder = _watching(tmp_path)
    try:
        monkeypatch.setattr(page.view_model, "resources", lambda: (1, None))  # one at a time
        for index in range(1, 6):
            write_frame(folder, index, jump_at=6)
        for _ in range(3):
            page.poll_watch()
            if page.watch_frames_waiting():
                break
        assert page.watch_frames_waiting() > 0
        page.stop_watch()
        assert page.watch_frames_waiting() == 0
        _done(page)
    finally:
        _close(page)


@pytest.mark.skipif(not GALAXI.is_file(), reason="public test frames (tests/data/external) are not available")
def test_large_frames_are_reduced_in_worker_processes_and_give_the_rows_reduced_here(tmp_path, monkeypatch) -> None:
    import numpy as np
    from PyQt5.QtWidgets import QApplication

    from src.gimap.features.analyze.presentation.bindings import watch_pace

    page, folder = _watching(tmp_path)
    try:
        model = page.view_model
        monkeypatch.setattr(model, "resources", lambda: (8, None))  # two at once
        monkeypatch.setattr(watch_pace, "WORKER_PIXELS", 0)  # (the test frames are small: as a large one)
        for index in range(1, 6):
            write_frame(folder, index, jump_at=6)
        for _ in range(3):  # (a new file is taken once it is the same at two looks)
            page.poll_watch()
            _done(page)
            if getattr(page, "_watch_futures", None):
                break
        assert page._watch_workers is not None and page._watch_futures  # in the workers, not the page's tasks
        deadline = time.monotonic() + 120
        while (page._watch_futures or page.watch_frames_waiting()) and time.monotonic() < deadline:
            QApplication.processEvents()
            time.sleep(0.02)
        _look(page, times=1)
        assert len(page._series_rows) == 6
        names = [row[4][0].name for row in page._series_rows]
        assert names == sorted(names)
        here = model.series_row(model.request_for(page._series_rows[-1][4][0], 0, summed=()), page._series_key)
        np.testing.assert_array_equal(here[0], page._series_rows[-1][0])
        np.testing.assert_array_equal(here[1], page._series_rows[-1][1])
        page.stop_watch()
        assert page._watch_workers is None  # closed with the watch
    finally:
        page.stop_watch()
        _close(page)


@pytest.mark.skipif(not GALAXI.is_file(), reason="public test frames (tests/data/external) are not available")
def test_while_watching_the_frame_on_screen_is_analysed_one_at_a_time_and_ends_at_the_newest(tmp_path, monkeypatch) -> None:
    page, folder = _watching(tmp_path, frames=4)
    try:
        model = page.view_model
        analyze, lock, running, most, asked = model.analyze, threading.Lock(), [0], [0], []

        def slow(request):
            with lock:
                running[0] += 1
                most[0] = max(most[0], running[0])
                asked.append(request.path.name)
            try:
                time.sleep(0.3)
                return analyze(request)
            finally:
                with lock:
                    running[0] -= 1

        page.file_list.setCurrentRow(0)
        _done(page)
        monkeypatch.setattr(model, "analyze", slow)
        for row in (1, 2, 3):  # (the watch shows the newest frame at every look that finds files)
            page.file_list.setCurrentRow(row)
        _done(page)
        assert most[0] == 1
        assert asked == [model.state.files[1].name, model.state.files[3].name]  # (the one between: never wanted)
        assert model.state.analysis.path.name == model.state.files[3].name
        page.stop_watch()
        asked.clear()
        for row in (1, 2):  # (no watch: at once, as before)
            page.file_list.setCurrentRow(row)
        _done(page)
        assert len(asked) == 2 and model.state.analysis.path.name == model.state.files[2].name
    finally:
        page.stop_watch()
        _close(page)
