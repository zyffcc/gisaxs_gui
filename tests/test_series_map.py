"""The Series tab: every frame's curve as intensity against frame and q, picked, opened and exported."""

from __future__ import annotations

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
    shorter = np.linspace(0.3, 0.8, 6)
    grid, image = stack_curves([(x, x), (shorter[::-1], shorter[::-1] * 3)])  # descending x is sorted first
    assert grid[0] == pytest.approx(0.1) and grid[-1] == pytest.approx(1.0) and grid.size == 8
    assert np.isnan(image[1, 0]) and image[1, 4] == pytest.approx(3 * grid[4])  # never extrapolated
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
        assert lines[0].startswith("# GIMaP Analyze series map") and lines[1] == "# frame 1: insitu_000.tif"
        header = next(line for line in lines if not line.startswith("#"))
        assert header.startswith("frame \\ q") and sum(1 for line in lines if not line.startswith("#")) == len(heights) + 1
        figure = page.export_series_figure(tmp_path / "map.png")
        assert figure is not None and figure.stat().st_size > 1000
        page.series_step_spin.setValue(2)  # every second frame: a quick look at a long series
        page.build_series_map()
        _wait(page, lambda: not page._series_queue and page.cancel_button.isHidden() and page._series_map is not None)
        assert page._series_map.labels == ("insitu_000.tif", "insitu_002.tif", "insitu_004.tif")
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
