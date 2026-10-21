"""A watched series grows from the very same complete frames that exports use."""
import threading

import numpy as np
import pytest
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from tests.test_analyze_workspace import _context, _done, _page
from tests.test_analyze_nexus_sources import _ring_stack, _grow_rings
from tests.test_review_analyze_run_and_clear import _profiles
from tests.test_series_cut import _build
from tests.test_live_following import _close


def _look(page, times=3):
    for _ in range(times):
        page.poll_watch()
        _done(page)
    QTest.qWait(1200)
    _done(page)


def _watched_map(tmp_path, frames=4, *, before_map=False):
    path = _ring_stack(tmp_path / "run" / "live.h5", frames)
    page = _page(_context(_profiles()))
    page.set_mode_choice("giwaxs")
    page.add_paths([str(path)])
    _done(page)
    page.auto_export_check.setChecked(False)
    if before_map:
        page.start_watch(path.parent)
        page._watch_timer.stop()
    return page, path


def _start_watch(page, path):
    page.start_watch(path.parent)
    page._watch_timer.stop()  # the tests choose exactly when the real watch looks


def _table(path):
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    return np.loadtxt(lines[1:], delimiter=",")  # one CSV header after the provenance comments


@pytest.mark.parametrize("exporting", [False, True])
def test_a_built_map_follows_complete_new_frames_with_or_without_auto_export(tmp_path, exporting):
    path = _ring_stack(tmp_path / "run" / "live.h5", 10)
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path)])
        _done(page)
        _build(page, 10)
        before = np.array(page._series_map.image, copy=True)
        held = page._series_built["cut"]
        page.start_watch(path.parent)
        page._watch_timer.stop()
        page.auto_export_check.setChecked(exporting)
        _grow_rings(path, 15)
        _look(page)
        assert page._series_map.rows == 15
        assert len({(str(p), index) for p, index in page._series_map.refs}) == 15
        np.testing.assert_array_equal(page._series_map.image[:10], before)
        assert page._series_map.built_with["cut"] == held
        assert page._series_map.numbers == tuple(range(1, 16))
        assert page._series_map.x_label == "q (Å⁻¹)"
        _look(page)
        assert page._series_map.rows == 15  # no frame appended twice
        if exporting:
            tables = sorted((path.parent / "gimap_analysis").glob("*_radial.csv"))
            assert len(tables) == 5
            last = _table(tables[-1])
            # Native exported x/intensity agree with the newest map row on this fixed geometry.
            assert np.allclose(page._series_rows[-1][0], last[:, 0], rtol=1e-6)
            assert np.allclose(page._series_rows[-1][1], last[:, 1], rtol=1e-6)
    finally:
        page.stop_watch()
        _close(page)


def test_auto_export_initializes_a_map_and_keeps_all_whole_rows_once(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.start_watch(folder)
        page._watch_timer.stop()
        page.auto_export_check.setChecked(True)
        path = _ring_stack(folder / "live.h5", 3)
        _look(page)
        assert page._series_map is not None and page._series_map.rows == 3
        _grow_rings(path, 6)
        _look(page)
        assert page._series_map.rows == 6 and len(page._series_rows) == 6
        assert page._series_map.numbers == tuple(range(1, 7))
        assert len(list((folder / "gimap_analysis").glob("*_radial.csv"))) == 6
    finally:
        page.stop_watch()
        _close(page)


@pytest.mark.parametrize("action", ["stop", "clear", "settings", "curve"])
def test_late_native_rows_cannot_recreate_or_change_a_stopped_or_stale_map(tmp_path, monkeypatch, action):
    page, path = _watched_map(tmp_path)
    gate, entered = threading.Event(), threading.Event()
    try:
        _build(page, 4)
        original_map = page._series_map
        _start_watch(page, path)
        original = page.view_model.series_row

        def waiting(request, key):
            entered.set()
            assert gate.wait(10), "test did not release the native reduction"
            return original(request, key)

        monkeypatch.setattr(page.view_model, "series_row", waiting)
        _grow_rings(path, 7)
        for _ in range(3):
            page.poll_watch()
            QTest.qWait(30)
        assert entered.wait(2), "watch never scheduled the real curve reduction"
        if action == "stop":
            page.stop_watch()
        elif action == "clear":
            page.clear_files()
        elif action == "settings":
            page.incidence_spin.setValue(page.incidence_spin.value() + 0.1)
        else:
            page.series_curve_combo.setCurrentIndex(page.series_curve_combo.findData("azimuthal"))
        gate.set()
        _done(page)
        QTest.qWait(1200)
        _done(page)
        if action == "clear":
            assert page._series_map is None and page._series_rows == []
        else:
            assert page._series_map.rows == original_map.rows == 4
            np.testing.assert_array_equal(page._series_map.image, original_map.image)
    finally:
        gate.set()
        page.stop_watch()
        _close(page)


def test_a_picked_frame_q_window_and_q_zoom_stay_while_new_frames_arrive(tmp_path):
    page, path = _watched_map(tmp_path)
    try:
        _build(page, 4)
        _start_watch(page, path)
        page.show_right("series")
        x = page._series_map.x
        low, high = float(x[len(x) // 4]), float(x[len(x) // 2])
        page.series_map_view.positionClicked.emit(low, 1.5)  # the scientist clicked frame 2
        page.series_map_view.verticalBandChanged.emit(low, high)
        page.series_map_view.view_box.setXRange(low, high, padding=0)
        QApplication.processEvents()
        zoom = tuple(page.series_map_view.view_box.viewRange()[0])
        assert not page.batch_panel.follow_check.isChecked()
        _grow_rings(path, 7)
        _look(page)
        assert page._series_map.rows == 7 and page._series_row == 1
        assert page._series_q == (low, high)
        np.testing.assert_allclose(page.series_map_view.view_box.viewRange()[0], zoom, rtol=0, atol=1e-8)
    finally:
        page.stop_watch()
        _close(page)


@pytest.mark.parametrize("watch_before_build", [False, True])
@pytest.mark.parametrize("every", [1, 2])
def test_user_sums_keep_the_original_group_starts_and_replace_the_short_tail(tmp_path, watch_before_build, every):
    page, path = _watched_map(tmp_path, 10, before_map=watch_before_build)
    try:
        page.sum_spin.setValue(3)
        _done(page)
        page.series_step_spin.setValue(every)
        _build(page, 4 if every == 1 else 2)
        if not watch_before_build:
            _start_watch(page, path)
        before = np.array(page._series_map.image, copy=True)
        _grow_rings(path, 13)
        _look(page)
        series = page._series_map
        # Watch appends only complete groups; its new frame-13 group is still pending at 13.
        assert series.numbers == ((1, 4, 7, 10) if every == 1 else (1, 7))
        assert all(fact.summed == 3 for fact in series.facts)
        np.testing.assert_array_equal(series.image[:3 if every == 1 else 2], before[:3 if every == 1 else 2])
        # Compare to a fresh scientific reduction of each original group, never to an algebraic intensity merge.
        requests = {request.frame_index: request for request in page.view_model.batch_requests()}
        for row, ref in enumerate(series.refs):
            native = page.view_model.series_row(requests[ref[1]], "radial")
            np.testing.assert_array_equal(series.x, native[0])
            stored = next(value for value in page._series_rows if value[4] == ref)
            np.testing.assert_array_equal(stored[0], native[0])
            np.testing.assert_array_equal(stored[1], native[1])
            # The picture uses float32 cells; native scientific rows above keep the full precision.
            np.testing.assert_array_equal(series.image[row], np.asarray(native[1], dtype=series.image.dtype))
        _grow_rings(path, 15)
        _look(page)
        assert page._series_map.numbers == ((1, 4, 7, 10, 13) if every == 1 else (1, 7, 13))
        assert all(fact.summed == 3 for fact in page._series_map.facts)
    finally:
        page.stop_watch()
        _close(page)


def test_watch_before_build_keeps_follow_on_until_the_person_picks_a_frame(tmp_path):
    page, path = _watched_map(tmp_path, 6, before_map=True)
    try:
        _build(page, 6)
        assert page.batch_panel.follow_check.isChecked()
        _grow_rings(path, 9)
        _look(page)
        assert page._series_map.rows == 9
        assert page.batch_panel.follow_check.isChecked() and page._series_row == 8
        page.series_map_view.positionClicked.emit(float(page._series_map.x[len(page._series_map.x) // 2]), 1.5)
        _grow_rings(path, 12)
        _look(page)
        assert page._series_map.rows == 12 and page._series_row == 1
        assert not page.batch_panel.follow_check.isChecked()
    finally:
        page.stop_watch()
        _close(page)


def test_a_closed_frame_range_does_not_silently_expand_while_watching(tmp_path):
    page, path = _watched_map(tmp_path, 6)
    try:
        page.series_first_spin.setValue(2)
        page.series_last_spin.setValue(4)
        _build(page, 3)
        before = page._series_map
        _start_watch(page, path)
        _grow_rings(path, 9)
        _look(page)
        assert page._series_map.numbers == (2, 3, 4)
        np.testing.assert_array_equal(page._series_map.image, before.image)
    finally:
        page.stop_watch()
        _close(page)


def test_an_automatic_sum_is_removed_before_live_map_requests_are_grouped(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.sum_spin.setValue(3)
        page.view_model.set_sum_count(3, by="automatic")
        _start_watch(page, folder / "live.h5")
        page.auto_export_check.setChecked(True)
        _ring_stack(folder / "live.h5", 5)
        _look(page)
        assert page.view_model.sum_count == 1 and page.sum_spin.value() == 1
        assert page._series_map.rows == 5 and page._series_map.numbers == (1, 2, 3, 4, 5)
        assert all(fact.summed == 1 for fact in page._series_map.facts)
        assert len(list((folder / "gimap_analysis").glob("*_radial.csv"))) == 5
    finally:
        page.stop_watch()
        _close(page)
