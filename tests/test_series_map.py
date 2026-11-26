"""The Series tab: every frame's curve as intensity against frame and q, picked, opened and exported."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.analyze.domain import series_map, stack_curves
from tests.test_analyze_workspace import CBF, _app, _context, _pilatus_profile, requires_data
from tests.test_assistant_calibration import CENTER, DISTANCE_M, PIXEL, SHAPE, WAVELENGTH, save_tiff


def _wait(page, condition, timeout_s: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        page.tasks.wait(0.2)
        _app().processEvents()
    assert condition()


def test_curves_on_one_grid_are_stacked_as_they_are_and_others_interpolated() -> None:
    x = np.linspace(0.1, 1.0, 10)
    grid, image = stack_curves([(x, x * 1.0), (x, x * 2.0)])
    assert np.array_equal(grid, x) and image.shape == (2, 10) and image[1, 3] == pytest.approx(2 * x[3])
    shorter = np.linspace(0.3, 0.8, 6)  # the same points, fewer of them: each value stays where it was measured
    grid, image = stack_curves([(x, x), (shorter[::-1], shorter[::-1] * 3)])  # descending x is sorted first
    assert np.allclose(grid, x) and grid.size == 10
    assert np.isnan(image[1, 0]) and image[1, 4] == pytest.approx(3 * grid[4])  # never extrapolated
    between = np.linspace(0.33, 0.83, 6)  # points between the other curve's: one even grid with their step
    grid, image = stack_curves([(x, x), (between, between * 3)])
    assert grid[0] == pytest.approx(0.1) and np.allclose(np.diff(grid), 0.1) and grid.size == 10
    assert np.isnan(image[1, 0]) and image[1, 4] == pytest.approx(3 * grid[4]) and np.isnan(image[1, -1])
    series = series_map([(x, x), (x, 2 * x), (x, 3 * x)], ["a", "b", "c"], x_label="q (Å⁻¹)", curve="radial")
    assert series.rows == 3 and np.allclose(series.trace(0.5, 0.05), [0.5, 1.0, 1.5])
    assert np.allclose(series.profile(1), 2 * x)


def _ring_frame(height: float, q0: float = 1.1, seed: int = 0) -> np.ndarray:
    rows, columns = np.indices(SHAPE)
    radius = np.hypot((columns + 0.5 - CENTER[0]) * PIXEL, (rows + 0.5 - CENTER[1]) * PIXEL)
    q = 4 * math.pi * np.sin(np.arctan2(radius, DISTANCE_M) / 2) / WAVELENGTH
    image = 30.0 + 300.0 * np.exp(-q / 0.25) + height * np.exp(-4 * np.log(2) * (q - q0) ** 2 / 0.04**2)
    return np.random.default_rng(seed).poisson(image).astype(np.float32)


def test_the_series_map_grows_frame_by_frame_and_its_rows_open_and_export(tmp_path: Path) -> None:
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
    from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile

    _app()
    heights = [0.0, 100.0, 200.0, 300.0, 400.0, 500.0]
    paths = [save_tiff(tmp_path / f"insitu_{index:03d}.tif", _ring_frame(height, seed=index)) for index, height in enumerate(heights)]
    profile = InstrumentProfile("synthetic", DetectorGeometry(PIXEL, PIXEL, DISTANCE_M, CENTER[0], CENTER[1], WAVELENGTH, 0.2),
                                None, SHAPE)
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([profile]))))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path) for path in paths])
        page.tasks.wait(60)
        _app().processEvents()
        assert page.series_build_button.isEnabled() and page.series_curve_combo.currentData() == "radial"
        assert "6 files listed" in page.series_info_label.text()
        assert not page.series_empty.isHidden() and page.series_map_view.isHidden()
        page.series_build_button.click()
        assert page.current_right() == "series" and not page.cancel_button.isHidden()
        _wait(page, lambda: page._series_map is not None and page._series_map.rows == len(heights) and not page._series_queue)
        series = page._series_map
        assert series.labels[0] == "insitu_000.tif" and series.x_label.startswith("q")
        assert series.numbers == (1, 2, 3, 4, 5, 6) and series.title == "I(q)" and series.view == "both"
        assert not page.series_map_view.isHidden() and page.series_empty.isHidden() and page.cancel_button.isHidden()
        # The vertical band starts where the series changes most, the ring: its intensity grows with the frame.
        low, high = page._series_q
        assert low <= 1.1 + 0.03 and high >= 1.1 - 0.03
        trace = page.series_trace_plot.figure_state()["curves"][0][2]
        assert np.all(np.diff(trace) > 0)
        page._series_pick_row(3.4)
        assert page.series_profile_plot.title_label.text() == "Frame 4: insitu_003.tif"
        assert page.open_series_row() and page.file_list.currentRow() == 3
        written = page.export_series_csv(tmp_path / "map.csv")
        lines = written.read_text(encoding="ascii").splitlines()
        assert lines[0].startswith("# GIMaP Analyze series map") and "# curve: radial - I(q)" in lines
        assert f"# frame 1: {paths[0]}" in lines and f"# frame 6: {paths[5]}" in lines  # the file with its folder
        assert "# series: insitu; 6 of 6 listed frames" in lines
        header = next(line for line in lines if not line.startswith("#"))
        assert header.startswith("frame \\ q") and sum(1 for line in lines if not line.startswith("#")) == len(heights) + 1
        record = json.loads(written.with_suffix(".json").read_text(encoding="utf-8"))  # the settings, next to the table
        assert record["curve"] == "radial" and record["settings"]["mode"] == "giwaxs" and len(record["rows"]) == 6
        assert record["software"] == "GIMaP Analyze 1" and record["software_version"].startswith("GIMaP v")
        assert f"# software_version: {record['software_version']}" in lines
        assert record["rows"][3]["file"] == str(paths[3]) and record["rows"][3]["frame"] == 4
        figure = page.export_series_figure(tmp_path / "map.png")
        assert figure is not None and figure.stat().st_size > 1000
        page.series_step_spin.setValue(2)  # every second frame: a quick look at a long series
        page.build_series_map()
        _wait(page, lambda: not page._series_queue and page.cancel_button.isHidden() and page._series_map is not None)
        assert page._series_map.labels == ("insitu_000.tif", "insitu_002.tif", "insitu_004.tif")
        assert page._series_map.numbers == (1, 3, 5)  # the frames of the list, not the rows of the map
        assert page.series_profile_plot.title_label.text().startswith("Frame ")
    finally:
        page.tasks.wait(30)
        page.dispose()


@requires_data
def test_a_real_gisaxs_series_stacks_the_horizontal_cut_and_can_be_cancelled() -> None:
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.integrations.state import InMemoryInstrumentProfileRepository

    _app()
    frames = sorted(CBF.parent.glob("jg_gisaxs_4nm_old_3ml_insitu_ds03_*.cbf"))
    assert len(frames) >= 3
    page = AnalyzePage(create_analyze_view_model(_context(InMemoryInstrumentProfileRepository([_pilatus_profile()]))))
    try:
        page.set_mode_choice("gisaxs")
        page.add_paths([str(path) for path in frames])
        page.tasks.wait(300)
        _app().processEvents()
        assert page.series_curve_combo.currentData() == "horizontal"
        page.build_series_map()
        _wait(page, lambda: not page._series_queue and page._series_map is not None and page.cancel_button.isHidden(), 300)
        assert page._series_map.rows == len(frames) and page._series_map.x_label.startswith("qy")
        assert np.nanmin(page._series_map.x) < 0 < np.nanmax(page._series_map.x)  # both halves of the cut
        page.build_series_map()
        page.cancel_series_map()  # stops after the frame being reduced
        _wait(page, lambda: page.cancel_button.isHidden() and not page._series_queue, 120)
        assert page._series_map.rows == 1 and page.series_build_button.isEnabled()
    finally:
        page.tasks.wait(60)
        page.dispose()


def test_curves_on_the_same_points_stack_as_they_are_however_many_and_close_points_still_count_as_the_same() -> None:
    """A live map of a long run stacks tens of thousands of curves on each drawing: curves of one geometry have
    exactly the same points (compared as such first), and points the same within rounding still count as the
    same (no row of the map is put on another grid)."""
    x = np.linspace(-0.1, 0.1, 50)
    curves = [(x.copy(), np.full(50, float(row))) for row in range(300)]
    curves.append((x * (1 + 1e-12), np.full(50, 300.0)))  # the same within rounding
    grid, image = stack_curves(curves)
    assert np.array_equal(grid, x) and image.shape == (301, 50) and np.isfinite(image).all()
    assert np.array_equal(image[:, 7], np.arange(301, dtype=np.float32))


def test_a_live_map_of_many_rows_waits_longer_between_drawings() -> None:
    from src.gimap.features.analyze.presentation.bindings.batch_live import PACE, REDRAW_S, STACK_PACE, redraw_wait_s

    assert redraw_wait_s(0.05) == REDRAW_S  # a small map: once a second
    assert redraw_wait_s(1.5) == pytest.approx(PACE * 1.5)  # a drawing of 1.5 s: the window drawing a fifth of the time
    assert redraw_wait_s(-1.0) == REDRAW_S
    # Stacked on its own thread: that thread works at most half the time; the window's drawing still counts.
    assert redraw_wait_s(0.1, 1.2) == pytest.approx(STACK_PACE * 1.2)
    assert redraw_wait_s(0.6, 0.2) == pytest.approx(PACE * 0.6)


def _live_rows(first: int, count: int, points: int = 48) -> list:
    from src.gimap.features.analyze.application.series_map import RowLabel

    x = np.linspace(-0.1, 0.1, points)
    rows = []
    for number in range(first, first + count):
        name = f"live_{number:05d}.edf"  # (invented frames)
        label = RowLabel(name, name=name, source=f"C:/run/{name}", frame_index=0, exposure_s=0.1)
        rows.append((x, 100.0 + number + np.cos(x * 40.0), "qy (Å⁻¹)", label, (f"C:/run/{name}", 0)))
    return rows


def test_a_live_map_of_many_rows_is_stacked_on_its_own_thread_and_rows_that_arrive_meanwhile_come_next() -> None:
    from src.gimap.features.analyze.presentation.bindings.series_later import LATER_ROWS
    from tests.test_analyze_workspace import _page

    page = _page(_context(None))
    try:
        page._series_rows = _live_rows(1, LATER_ROWS + 40)
        page._series_key, page._series_built = "horizontal", {}
        page._batch_live, page._batch_last_draw, page._batch_rows_drawn = True, 0.0, 0
        page.batch_panel.follow_check.setChecked(True)

        page._batch_live_redraw()
        assert page._series_later_busy() and page._series_map is None  # being stacked: not drawn here
        assert not page.tasks.is_busy()  # (no work of the page's: its progress bar stays hidden)
        page._series_rows.extend(_live_rows(LATER_ROWS + 41, 5))  # frames that arrive meanwhile
        page._batch_last_draw = 0.0
        page._batch_live_redraw()  # one map at a time: these rows wait for the next
        assert page._series_later_runner.wait(60.0)
        _app().processEvents()
        assert not page._series_later_busy()
        assert page._series_map.rows == LATER_ROWS + 40 and page._series_row == LATER_ROWS + 39
        assert page._series_map.numbers[-1] == LATER_ROWS + 40
        page._batch_last_draw = 0.0
        page._batch_live_redraw()
        assert page._series_later_runner.wait(60.0)
        _app().processEvents()
        assert page._series_map.rows == LATER_ROWS + 45  # the rows that arrived while the map before was stacked

        page._batch_last_draw = 0.0
        page._series_rows.extend(_live_rows(LATER_ROWS + 46, 3))
        page._batch_live_redraw()
        assert page._series_later_busy()
        page._batch_live_redraw(force=True)  # the last map of a run: drawn here, the one being stacked is dropped
        drawn = page._series_map
        assert drawn.rows == LATER_ROWS + 48 and not page._series_later_busy()
        assert page._series_later_runner.wait(60.0)
        _app().processEvents()
        assert page._series_map is drawn

        calls = []  # a watched folder's redraw: the stages cleared and looked for again around the map shown
        page._series_rows.extend(_live_rows(LATER_ROWS + 49, 1))
        assert page._series_redraw_live(keep_view=True, keep_q=True, before=lambda: calls.append("before"),
                                        then=lambda: calls.append(("then", page._series_map.rows)))
        assert calls == [] and not page._series_redraw_live(keep_view=True, keep_q=True)  # (one at a time)
        assert page._series_later_runner.wait(60.0)
        _app().processEvents()
        assert calls == ["before", ("then", LATER_ROWS + 49)]

        page._batch_last_draw = 0.0
        page._series_rows.extend(_live_rows(LATER_ROWS + 50, 2))
        page._batch_live_redraw()
        page._reset_series()  # the list cleared while a map is being stacked: it is not shown
        assert page._series_later_runner.wait(60.0)
        _app().processEvents()
        assert page._series_map is None and page.series_map_view.isHidden()
    finally:
        page.dispose()


def test_the_rows_of_a_live_map_hold_an_equal_x_once() -> None:
    from src.gimap.features.analyze.application.series_map import SHARED_X, share_x

    known: list = []
    x = np.linspace(0.01, 0.2, 7)
    first = share_x((x, np.ones(7), "q (Å⁻¹)", "a", ("f.edf", 0)), known)
    second = share_x((x.copy(), np.full(7, 2.0), "q (Å⁻¹)", "b", ("f.edf", 1)), known)  # (a reduction's own copy)
    assert second[0] is first[0] is x and second[1][0] == 2.0 and second[2:] == ("q (Å⁻¹)", "b", ("f.edf", 1))
    other = share_x((x * 1.01, np.ones(7), "q (Å⁻¹)", "c", ("f.edf", 2)), known)  # another grid (another angle)
    assert other[0] is not x and len(known) == 2
    assert share_x((x.copy(), np.ones(7), "q (Å⁻¹)", "d", ("f.edf", 3)), known)[0] is x  # alternating angles
    assert share_x((x[:-1].copy(), np.ones(6), "q (Å⁻¹)", "e", ("f.edf", 4)), known)[0] is not x
    for step in range(SHARED_X + 2):
        share_x((x + step + 1.0, np.ones(7), "q (Å⁻¹)", "g", ("f.edf", 5 + step)), known)
    assert len(known) == SHARED_X and np.array_equal(x, np.linspace(0.01, 0.2, 7))  # no x changed


def test_the_frame_axis_ticks_the_rows_in_view() -> None:
    from src.gimap.features.analyze.presentation.bindings.series_display import MINOR_ROWS, frame_ticks

    numbers = range(1, 20001)
    major, minor = frame_ticks(numbers)  # every row of a long map at once: round numbers, not a tick per row
    assert [name for _at, name in major] == ["5000", "10000", "15000", "20000"]
    assert len(minor) == 36 and all(int(name) % 500 == 0 for _at, name in minor)
    major, minor = frame_ticks(numbers, shown=(99.6, 139.2))  # zoomed in on 40 rows: each of them
    assert sorted(at for at, _name in major + minor) == [row + 0.5 for row in range(99, 140)]
    assert dict(major + minor)[120.5] == "121"
    major, minor = frame_ticks(numbers, shown=(5000.0, 5000.0 + 2 * MINOR_ROWS))
    assert all(5000 <= at <= 5000 + 2 * MINOR_ROWS for at, _name in major + minor) and len(minor) < 60
