"""The Series map is right and says what it is: drawn at the true x, nothing invented across gaps, a frame
axis in real frame numbers, the halves of a signed cut, and exports that carry what they hold.

The first part needs no window (``analyze/domain/series_map.py`` and ``application/series_map.py``); the
second drives the Series tab on synthetic frames.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.features.analyze.application.series_map import (
    RowLabel, build_series_map, frame_text, row_facts, series_header, series_times,
)
from src.gimap.features.analyze.domain.series_map import (
    VIEWS, RowFacts, SeriesMap, display_grid, fold_signed, series_map, stack_curves, track_peak,
)

# -- the picture is drawn at the true x -------------------------------------------------------------------------------


def _uneven_axis() -> np.ndarray:
    """An x like a curve's: three merged bins at its start, then one point per detector column, a gap of
    forty columns (a module gap), and columns again."""
    fine = 0.10 + 0.002 * np.arange(300)
    return np.concatenate([[0.030, 0.055, 0.080], fine[:120], fine[160:]])


def test_the_picture_of_a_map_is_on_an_even_grid_at_the_true_x() -> None:
    x = _uneven_axis()
    peak = int(np.argmin(np.abs(x - 0.60)))
    image = np.ones((3, x.size), dtype=np.float32)
    image[:, peak] = 50.0
    shown, left, width = display_grid(x, image)
    step = width / shown.shape[1]
    assert step == pytest.approx(0.002, rel=1e-6) and left == pytest.approx(x[0] - 0.5 * step)
    centres = left + (np.arange(shown.shape[1]) + 0.5) * step
    column = int(np.nanargmax(shown[0]))
    assert abs(centres[column] - x[peak]) <= 0.5 * step + 1e-12  # (it was 0.066 off: 33 columns)
    assert centres[-1] == pytest.approx(x[-1], abs=0.5 * step)


def test_the_gap_of_the_detector_stays_blank_in_the_picture_and_merged_bins_do_not() -> None:
    x = _uneven_axis()
    shown, left, width = display_grid(x, np.ones((2, x.size), dtype=np.float32))
    step = width / shown.shape[1]
    centres = left + (np.arange(shown.shape[1]) + 0.5) * step
    gap = (centres > x[122] + step) & (centres < x[123] - step)  # between the columns either side of the gap
    assert gap.sum() >= 35 and np.isnan(shown[:, gap]).all()
    merged = (centres > 0.03) & (centres < 0.10)  # three wide bins: every column there shows the nearest one
    assert merged.sum() > 20 and np.isfinite(shown[:, merged]).all()
    measured = np.abs(centres[:, None] - x[None, :]).min(axis=1) <= 0.5 * step + 1e-12
    assert np.isfinite(shown[:, measured]).all()  # every column at a point of the curve shows it
    # A cell without a value stays without one; an axis of one point is drawn as it is.
    holed = np.ones((1, x.size), dtype=np.float32)
    holed[0, 200] = np.nan
    assert np.isnan(display_grid(x, holed)[0]).sum() == np.isnan(shown[0]).sum() + 1
    one, one_left, one_width = display_grid(np.array([0.5]), np.ones((2, 1)))
    assert one.shape == (2, 1) and one_left < 0.5 < one_left + one_width


# -- nothing is invented between the points --------------------------------------------------------------------------


def test_frames_that_differ_in_their_valid_columns_keep_their_own_points() -> None:
    """One geometry, one cut: the x of a detector column is the same in every frame, so nothing is
    interpolated — a frame simply has no value where its column was left out."""
    x = 0.01 * np.arange(1, 41)
    first = np.delete(np.arange(40), [7, 8])
    second = np.delete(np.arange(40), [20])
    grid, image = stack_curves([(x[first], 2.0 * x[first]), (x[second], 3.0 * x[second])])
    assert np.allclose(grid, x) and image.shape == (2, 40)
    assert np.isnan(image[0, [7, 8]]).all() and np.isnan(image[1, 20])
    assert np.allclose(image[0, first], 2.0 * x[first]) and np.allclose(image[1, second], 3.0 * x[second])


def test_curves_on_other_grids_are_not_joined_across_a_hole() -> None:
    """Frames cut at different rows have different x: they are put on one even grid, and the hole of a frame
    (a module gap) stays empty instead of a straight line between its two edges."""
    step = 0.001
    left, right = 0.010 + step * np.arange(60), 0.130 + step * np.arange(60)  # a hole of 60 steps
    x = np.concatenate([left, right])
    other = x + 0.0004  # the same detector, another row: every x a little different
    grid, image = stack_curves([(x, 1.0 + 10.0 * x), (other, 2.0 + 10.0 * other)])
    assert np.median(np.diff(grid)) == pytest.approx(step, rel=1e-3)  # the curves' own step, not a coarser one
    inside = (grid > left[-1] + step) & (grid < right[0] - step)
    assert inside.sum() >= 55 and np.isnan(image[:, inside]).all()
    measured = (grid >= other[0]) & (grid <= left[-1])
    assert np.allclose(image[0, measured], 1.0 + 10.0 * grid[measured], atol=1e-6)
    assert np.allclose(image[1, measured], 2.0 + 10.0 * grid[measured], atol=1e-6)
    assert np.isnan(image[1, 0]) and np.isfinite(image[0, 0])  # never beyond a curve's own ends


# -- where the series changes most --------------------------------------------------------------------------------------


def _growing(rows: int, noise: float = 1.0, seed: int = 3) -> SeriesMap:
    """A bright, steady line at x = 1.0 and a weak one at 1.7 that grows with the frame, with noise."""
    x = np.linspace(0.5, 2.5, 401)
    rng = np.random.default_rng(seed)
    curves = []
    for row in range(rows):
        y = 20.0 + 300.0 * np.exp(-0.5 * ((x - 1.0) / 0.02) ** 2) + 12.0 * row * np.exp(-0.5 * ((x - 1.7) / 0.02) ** 2)
        curves.append((x, y + rng.normal(0.0, noise * np.sqrt(y / 20.0))))
    return series_map(curves, [f"f{row}" for row in range(rows)], x_label="q (Å⁻¹)", curve="radial")


@pytest.mark.parametrize("rows", [3, 4, 8, 20])
def test_the_x_window_starts_on_the_line_that_changes_also_in_a_short_series(rows: int) -> None:
    for seed in range(5):
        assert _growing(rows, seed=seed).most_changing_x() == pytest.approx(1.7, abs=0.03), (rows, seed)


# -- the halves of a signed cut ---------------------------------------------------------------------------------------


def _signed_cut(scale: float = 1.0):
    """A horizontal cut with both halves: 61 points with qy > 0, and qy < 0 with a hole (a module gap)."""
    positive = 0.001 + 0.002 * np.arange(61)
    negative = -(0.002 + 0.002 * np.arange(60))
    negative = np.delete(negative, np.arange(20, 35))
    x = np.concatenate([negative[::-1], positive])
    y = np.where(x > 0, 100.0, 60.0) * scale * np.exp(-np.abs(x) / 0.05)
    return x, y


def test_a_signed_cut_is_shown_as_one_half_both_or_their_mean() -> None:
    x, y = _signed_cut()
    assert VIEWS == ("both", "positive", "negative", "folded", "mean")
    both_x, both_y = fold_signed(x, y, "both")
    assert np.array_equal(both_x, x) and np.array_equal(both_y, y)
    plus_x, plus_y = fold_signed(x, y, "positive")
    assert plus_x.size == 61 and plus_x.min() > 0 and np.allclose(plus_y, 100.0 * np.exp(-plus_x / 0.05))
    minus_x, minus_y = fold_signed(x, y, "negative")
    assert minus_x.size == 45 and minus_x.min() > 0 and np.all(np.diff(minus_x) > 0)
    assert np.allclose(minus_y, 60.0 * np.exp(-minus_x / 0.05))
    folded_x, folded_y = fold_signed(x, y, "folded")
    assert folded_x.size == 106 and np.all(np.diff(folded_x) >= 0)  # every measured point of both halves, on |x|
    mean_x, mean_y = fold_signed(x, y, "mean")
    assert np.all(np.diff(mean_x) > 0) and mean_x.min() > 0
    hole = (mean_x > 0.043) & (mean_x < 0.071)  # where the other half has no points: the one half as measured
    assert hole.sum() >= 12 and np.allclose(mean_y[hole], 100.0 * np.exp(-mean_x[hole] / 0.05))
    both = (mean_x > 0.004) & (mean_x < 0.038)
    assert np.allclose(mean_y[both], 80.0 * np.exp(-mean_x[both] / 0.05), rtol=2e-3)  # the mean of 100 and 60


def test_both_halves_on_the_modulus_keep_every_point_also_when_a_frame_lacks_a_column() -> None:
    """With the beam on a pixel edge the |x| of the two halves coincide: two points at one x. Frames whose valid
    columns differ are stacked on the union of their x, where each x has as many columns as one curve has points
    there — the q < 0 half does not overwrite the q > 0 half (review: 300 of 590 points were left)."""
    x = 0.004 * (np.arange(-50, 50) + 0.5)
    y = 100.0 * np.exp(-np.abs(x) / 0.05) * np.where(x > 0, 1.0, 0.6)
    lacking = np.delete(np.arange(100), 30)  # one more column left out in the second frame
    series = series_map([(x, y), (x[lacking], 1.1 * y[lacking])], ["a", "b"], x_label="qy (Å⁻¹)", curve="horizontal",
                        view="folded")
    assert series.x.size == 100 and np.all(np.diff(series.x) >= 0) and series.x.min() > 0
    assert np.isfinite(series.image).sum(axis=1).tolist() == [100, 99]
    for row, values in ((0, y), (1, 1.1 * y[lacking])):  # every measured value once, in its own row
        kept = np.sort(series.image[row][np.isfinite(series.image[row])])
        assert np.allclose(kept, np.sort(values), rtol=1e-6)
    pairs = series.x.reshape(50, 2)  # … two columns at every |x|, one for each half
    assert np.allclose(pairs[:, 0], pairs[:, 1]) and np.allclose(pairs[:, 0], x[50:])


def test_a_map_of_one_half_is_on_the_modulus_of_x_and_says_so() -> None:
    rows = [(*_signed_cut(1.0 + 0.1 * index), "qy (Å⁻¹)", f"f{index}.cbf", (Path(f"f{index}.cbf"), 0)) for index in range(4)]
    signed = build_series_map(rows, "horizontal")
    assert signed.view == "both" and signed.x_label == "qy (Å⁻¹)" and signed.x.min() < 0 < signed.x.max()
    for view, points in (("positive", 61), ("negative", 45), ("folded", 106), ("mean", 61)):
        series = build_series_map(rows, "horizontal", view=view)
        assert series.view == view and series.x_label == "|qy| (Å⁻¹)" and series.x.min() > 0
        assert series.x.size == points and series.rows == 4 and np.isfinite(series.image).all(), view
    one_sided = [(np.linspace(0.1, 1.0, 10), np.ones(10), "q (Å⁻¹)", "a", (Path("a"), 0))] * 2
    assert build_series_map(one_sided, "radial", view="mean").view == "both"  # an I(q) has no halves


def test_a_trace_window_in_a_gap_or_beyond_the_data_has_no_value() -> None:
    """Nothing measured in the window — a detector gap, or beyond the ends of the map — gives no value, not the
    value of the nearest column; a window narrower than the merged bins around it takes the nearest point,
    and says so."""
    x = _uneven_axis()  # merged bins at its start, a gap of forty columns after x[122]
    series = SeriesMap(x, np.vstack([np.full(x.size, 5.0), np.full(x.size, 7.0)]).astype(np.float32), ("a", "b"),
                       "q (Å⁻¹)", "radial")
    gap = 0.5 * (x[122] + x[123])
    for centre, half in ((gap, 0.01), (x[-1] + 0.5, 0.05), (x[0] - 0.2, 0.01)):
        assert np.isnan(series.trace(centre, half)).all(), centre
        assert series.trace_columns(centre, half)[1] == "none"
    assert np.allclose(series.trace(0.2, 0.01), [5.0, 7.0]) and series.trace_columns(0.2, 0.01)[1] == "window"
    columns, how = series.trace_columns(0.035, 0.001)  # between the merged bins at 0.030 and 0.055
    assert how == "nearest" and np.flatnonzero(columns).tolist() == [0]
    assert np.allclose(series.trace(0.035, 0.001), [5.0, 7.0])


# -- the peak in a window ---------------------------------------------------------------------------------------------


def test_the_area_of_a_peak_is_integrated_on_the_x_it_has() -> None:
    """Bins of different widths: the area is the integral over x, not a sum times the usual step."""
    x = np.concatenate([np.arange(0.80, 1.00, 0.02), np.arange(1.00, 1.20001, 0.004)])
    sigma = 0.03
    y = 5.0 + 100.0 * np.exp(-0.5 * ((x - 1.0) / sigma) ** 2)
    track = track_peak(series_map([(x, y), (x, y)], ["a", "b"], x_label="q (Å⁻¹)", curve="radial"), 0.80, 1.20)
    assert np.allclose(track.area, 100.0 * sigma * np.sqrt(2 * np.pi), rtol=0.03)  # (0.48 of it before)
    assert track.window == (0.80, 1.20) and [name for name, _values in track.table()] == ["position", "fwhm", "area", "height"]


# -- what a row is ----------------------------------------------------------------------------------------------------


def test_a_row_knows_its_frame_its_file_and_how_many_frames_it_sums() -> None:
    label = RowLabel("scan_00007 · frame 4 of 12", source="/data/scan_00007_m01.nxs", frame_index=3, frame_count=12,
                     summed=1, timestamp="", exposure_s=0.5)
    assert label == "scan_00007 · frame 4 of 12" and label.frame_index == 3 and label.exposure_s == 0.5
    import pickle

    back = pickle.loads(pickle.dumps(label))  # it comes back from a worker process
    assert back == label and (back.frame_index, back.frame_count, back.source) == (3, 12, "/data/scan_00007_m01.nxs")
    assert json.dumps({"frame": label}) == '{"frame": "scan_00007 \\u00b7 frame 4 of 12"}'
    assert frame_text(3, 12) == "frame 4 of 12" and frame_text(0, 1) == ""
    assert frame_text(5, 12, summed=5) == "frames 6–10 of 12" and frame_text(10, 12, summed=2) == "frames 11–12 of 12"


def _rows(count: int, *, step: int = 1, frames: int = 12, summed: int = 1, exposure=None, stamps=None) -> list:
    x = np.linspace(0.1, 1.0, 10)
    rows = []
    for row in range(count):
        index = row * step
        label = RowLabel(f"scan · frame {index + 1} of {frames}", source="/d/scan_m01.nxs", frame_index=index,
                         frame_count=frames, summed=summed, exposure_s=exposure,
                         timestamp="" if stamps is None else stamps[row])
        rows.append((x, x * (row + 1), "q (Å⁻¹)", label, (Path("/d/scan_m01.nxs"), index)))
    return rows


def test_the_frame_axis_counts_the_frames_of_the_file_from_one() -> None:
    rows = _rows(4, step=3)  # every third frame of a 12-frame file
    numbers = {("/d/scan_m01.nxs", index): index + 1 for index in range(12)}
    series = build_series_map(rows, "radial", numbers=numbers)
    assert series.numbers == (1, 4, 7, 10) and series.number(2) == 7
    assert [fact.frame_index for fact in series.facts] == [0, 3, 6, 9] and series.facts[0].frame_count == 12
    assert build_series_map(rows, "radial").numbers == (1, 2, 3, 4)  # nothing known of the list: the row
    plain = series_map([(np.arange(3.0), np.arange(3.0))] * 2, ["a", "b"], x_label="q", curve="radial")
    assert plain.numbers == (1, 2) and plain.times is None and plain.facts == ()


def test_the_last_incomplete_sum_is_marked() -> None:
    rows = _rows(3, step=5, summed=5)
    rows[-1] = (*rows[-1][:3], RowLabel("scan · frames 11–12 of 12", source="/d/scan_m01.nxs", frame_index=10,
                                        frame_count=12, summed=2), rows[-1][4])
    facts = row_facts(rows, group=5)
    assert [(fact.summed, fact.group, fact.incomplete) for fact in facts] == [(5, 5, False), (5, 5, False), (2, 5, True)]
    assert isinstance(facts[0], RowFacts) and row_facts(_rows(2))[0].incomplete is False


def test_the_time_of_a_row_comes_from_its_time_stamp_or_from_the_exposure() -> None:
    stamps = ["2024-10-26T01:38:05.208", "2024-10-26T01:38:06.708", "2024-10-26T01:38:09.208"]
    times, source = series_times(row_facts(_rows(3, stamps=stamps, exposure=0.995)))
    assert source == "timestamps" and np.allclose(times, [0.0, 1.5, 4.0])
    times, source = series_times(row_facts(_rows(4, step=3, exposure=0.5)))  # one file: frame × exposure
    assert source == "exposure" and np.allclose(times, [0.0, 1.5, 3.0, 4.5])
    same = ["2024-10-26T01:38:05"] * 3  # the time of the file, the same for every frame: no time axis from it
    assert series_times(row_facts(_rows(3, stamps=same)))[0] is None
    assert series_times(row_facts(_rows(3)))[0] is None and series_times(())[0] is None


def _scans(stamps, *, exposure=0.5, frames: int = 4) -> list:
    """Rows of several multi-frame files, each frame carrying the start time of its file (as a Lambda file does)."""
    x = np.linspace(0.1, 1.0, 10)
    rows = []
    for scan, stamp in enumerate(stamps):
        source = f"/d/scan{scan}_m01.nxs"
        for index in range(frames):
            label = RowLabel(f"scan{scan} frame {index + 1} of {frames}", source=source, frame_index=index,
                             frame_count=frames, exposure_s=exposure, timestamp=stamp)
            rows.append((x, x, "q (Å⁻¹)", label, (Path(source), index)))
    return rows


def test_the_start_time_of_a_file_is_not_the_time_of_each_of_its_frames() -> None:
    """Two scans of four frames, 14 minutes apart: not 0, 0, 0, 0, 840, 840 … ‘from the time stamps’, but the
    start of each file plus frame × exposure, said as such; without an exposure time no time axis at all."""
    start = ["2024-10-25T11:31:59", "2024-10-25T11:45:59"]
    times, source = series_times(row_facts(_scans(start)))
    assert source == "start_exposure" and np.allclose(times, [0.0, 0.5, 1.0, 1.5, 840.0, 840.5, 841.0, 841.5])
    series = build_series_map(_scans(start), "radial")
    assert series.time_from == "start_exposure" and any(
        line.startswith("time: seconds since the first row, start time of each file + frame number x exposure")
        for line in series_header(series))
    assert series_times(row_facts(_scans(start, exposure=None)))[0] is None
    assert series_times(row_facts(_scans(start[:1])))[1] == "exposure"  # one file: frame × exposure, as before
    # Time stamps of the P03 Lambda (an offset without a colon), and a fraction of any length.
    stamps = ["2024-10-25T11:31:59.36+0200", "2024-10-25T11:32:27.360787+0200", "2024-10-25 09:32:40.36Z"]
    times, source = series_times(row_facts(_scans(stamps, frames=1)))
    assert source == "timestamps" and np.allclose(times, [0.0, 28.0, 41.0], atol=1e-3)


def test_only_the_horizontal_cut_has_halves_and_the_stages_name_their_axis(tmp_path: Path) -> None:
    from src.gimap.features.analyze.application.series_map import view_words
    from src.gimap.features.analyze.application.series_stages import export_series_stages, stages_of, stages_record
    from src.gimap.features.analyze.infrastructure.adapters.csv_export import CsvCurveWriter

    chi = np.linspace(-90.0, 90.0, 181)
    rng = np.random.default_rng(1)
    rows = [100.0 + 50.0 * (1.0 if row < 6 else 3.0) * np.exp(-0.5 * ((chi - 30.0) / 5.0) ** 2) + rng.normal(0.0, 1.0, chi.size)
            for row in range(12)]
    ring = SeriesMap(chi, np.vstack(rows).astype(np.float32), tuple(f"f{row}" for row in range(12)), "χ (°)", "azimuthal")
    assert view_words(ring) == ""  # χ from −90 to 90 has no halves
    stages = stages_of(ring)
    written = export_series_stages(CsvCurveWriter(), ring, stages, tmp_path / "stages.csv")
    lines = written.read_text(encoding="utf-8").splitlines()
    changes = [line for line in lines if line.startswith("# Stage ")]
    assert changes and all(" at chi " in line and " at q " not in line for line in changes)
    assert not any(line.startswith("# halves") for line in lines)
    assert all(" at χ " in text for text in stages_record(ring, stages)["changes"])
    cut = SeriesMap(np.linspace(-0.1, 0.1, 21), np.ones((2, 21), dtype=np.float32), ("a", "b"), "qy (Å⁻¹)", "horizontal")
    assert view_words(cut) == "both halves, on the signed qy"


# == the Series tab ====================================================================================================

from types import SimpleNamespace

import h5py
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application.series_map import row_label
from src.gimap.features.analyze.presentation import series_texts
from src.gimap.features.analyze.presentation.bindings.series_display import frame_ticks, nice_step
from src.gimap.features.analyze.presentation.bindings.series_exports import ROW_AXIS, figure_rows
from src.gimap.features.analyze.presentation.views.series_view import CHANGE_FEW, VIEW_TIPS
from src.gimap.shared.detector_io import forget_layout
from tests.test_analyze_nexus_sources import _ring_stack
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_review_analyze_run_and_clear import _profiles
from tests.test_series_map import _ring_frame, _wait


@pytest.fixture(autouse=True)
def _english():
    forget_layout()
    apply_language("en")
    yield
    apply_language("en")
    forget_layout()


def _close(page) -> None:
    page.tasks.wait(30)
    page.dispose()
    page.close()


def _build(page, rows: int) -> None:
    page.build_series_map()
    _wait(page, lambda: page._series_map is not None and page._series_map.rows == rows and not page.batch_running())
    _done(page)


def _giwaxs_page(tmp_path: Path, count: int):
    """A page with ``count`` synthetic GIWAXS frames listed (the ring at q = 1.1 grows with the frame)."""
    from tests.test_assistant_calibration import save_tiff

    paths = [save_tiff(tmp_path / f"film_{index:02d}.tif", _ring_frame(100.0 * (index + 1), seed=index)) for index in range(count)]
    page = _page(_context(_profiles()))
    page.set_mode_choice("giwaxs")
    page.add_paths([str(path) for path in paths])
    _done(page)
    return page, paths


def _gisaxs_page(tmp_path: Path):
    from tests.test_series_cut import _gisaxs_page as page_of

    return page_of(tmp_path)


def _answers(monkeypatch, folder: Path) -> list:
    """Save dialogs answer with the name they propose, in ``folder``; returns what was proposed."""
    asked: list = []

    def answer(_parent, title, path, filters="", *_more, **_options):
        asked.append((title, Path(path).name))
        return str(folder / Path(path).name), ""

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(answer))
    return asked


# -- the frame axis ---------------------------------------------------------------------------------------------------


def test_the_ticks_of_the_frame_axis_are_whole_frames_in_the_middle_of_their_rows() -> None:
    assert [nice_step(span) for span in (3, 8, 9, 20, 45, 240, 1000)] == [1, 1, 2, 5, 10, 50, 200]
    major, minor = frame_ticks([1, 2, 3])
    assert major == [(0.5, "1"), (1.5, "2"), (2.5, "3")] and minor == []  # (it was 0, 0.5, 1, 1.5 … at the row edges)
    major, minor = frame_ticks(range(1, 21))
    assert major == [(4.5, "5"), (9.5, "10"), (14.5, "15"), (19.5, "20")] and len(minor) == 16
    major, minor = frame_ticks([1, 4, 7, 10])  # every third frame of a file: its frames, not the rows
    assert [name for _at, name in major + minor] and {name for _at, name in major + minor} == {"1", "4", "7", "10"}
    major, _minor = frame_ticks([0.0, 1.5, 3.0], whole=False)  # seconds, with the decimals their step needs
    assert major == [(0.5, "0.0"), (1.5, "1.5"), (2.5, "3.0")]
    assert frame_ticks([0.0, 0.998, 1.996], whole=False)[0] == [(0.5, "0"), (1.5, "1"), (2.5, "2")]
    assert [name for _at, name in frame_ticks([0.0, 0.998, 1.996, 2.994], whole=False)[0]] == ["0.00", "1.00", "2.00", "2.99"]
    assert frame_ticks([0.0, 27.999, 39.998], whole=False)[0] == [(0.5, "0"), (1.5, "28"), (2.5, "40")]
    assert figure_rows([1, 2, 3]) == (0.5, 3.5) and figure_rows([2, 5, 8]) == (0.5, 9.5)
    assert figure_rows([1, 2, 4]) is None and figure_rows([7]) == (6.5, 7.5)


def test_the_map_is_drawn_at_the_true_x_with_frames_counted_from_one(tmp_path: Path, monkeypatch) -> None:
    page, _paths = _giwaxs_page(tmp_path, 6)
    try:
        _build(page, 6)
        series, view = page._series_map, page.series_map_view
        picture = np.asarray(view._data, dtype=float)
        rect = view.image_item.mapRectToParent(view.image_item.boundingRect())
        column = rect.width() / picture.shape[1]
        assert rect.top() == 0.0 and rect.height() == pytest.approx(6.0)
        assert rect.left() == pytest.approx(series.x[0] - 0.5 * column) and rect.right() == pytest.approx(
            series.x[-1] + 0.5 * column, abs=column)
        steps = np.diff(series.x)
        assert steps.max() > 1.5 * np.median(steps)  # the x of this curve is not evenly spaced
        for row in range(series.rows):  # the strongest point of every row is drawn where it is
            drawn = rect.left() + (int(np.nanargmax(picture[row])) + 0.5) * column
            assert abs(drawn - series.x[int(np.nanargmax(series.image[row]))]) <= 0.5 * column + 1e-9
        axis = view.plot.getAxis("left")
        assert axis.labelText == "frame" and axis._tickLevels[0] == [(row + 0.5, str(row + 1)) for row in range(6)]
        # The read-out names the frame and the value at the true x under the pointer.
        peak = int(np.nanargmax(series.image[2]))
        assert f"q = {series.x[peak]:.4g}" in page._series_readout(float(series.x[peak]), 2.5)
        assert page._series_readout(float(series.x[peak]), 2.5).startswith("frame 3 (film_02.tif)")
        # The saved figure: the first frame on top (as on screen), frame numbers in the middle of the rows.
        saved = {}
        monkeypatch.setattr(page.view_model, "save_image_figure",
                            lambda path, image, **state: saved.update(state, image=image) or Path(path))
        monkeypatch.setattr(page, "notify_written", lambda *_args: None)
        assert page.export_series_figure(tmp_path / "map.png") == tmp_path / "map.png"
        left, right, bottom, top = saved["extent"]
        assert (bottom, top) == (6.5, 0.5) and saved["origin_upper"] is True and saved["y_label"] == "frame"
        assert left == pytest.approx(rect.left()) and right == pytest.approx(rect.right())
        assert saved["title"] == "I(q) — 6 frames" and page.series_title_label.text() == "I(q) · 6 frames of film"
        page.series_step_spin.setValue(4)  # frames 1 and 5: no figure axis names both rows evenly … it does (step 4)
        _build(page, 2)
        assert page._series_map.numbers == (1, 5) and view.plot.getAxis("left")._tickLevels[0] == [(0.5, "1"), (1.5, "5")]
        page.export_series_figure(tmp_path / "map.png")
        assert saved["extent"][2:] == (7.0, -1.0) and ROW_AXIS != "frame"
    finally:
        _close(page)


def test_a_map_of_a_file_with_exposure_times_can_be_read_against_time(tmp_path: Path) -> None:
    stack = _ring_stack(tmp_path / "stack.h5", 6)
    with h5py.File(stack, "a") as handle:
        handle.create_dataset("/entry/instrument/detector/count_time", data=np.asarray([500.0])).attrs["units"] = "ms"
    page = _page(_context(_profiles()))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(stack)])
        _done(page)
        _build(page, 6)
        series, combo = page._series_map, page.series_axis_combo
        assert series.time_from == "exposure" and np.allclose(series.times, 0.5 * np.arange(6))
        assert series.numbers == (1, 2, 3, 4, 5, 6) and series.labels[3] == "stack.h5 frame 4 of 6"
        assert page.series_profile_plot.title_label.text() == f"Frame {page._series_row + 1} of 6: stack.h5"
        time = combo.findData("time")
        assert combo.model().item(time).isEnabled() and combo.currentData() == "frame"
        combo.setCurrentIndex(time)
        combo.activated.emit(time)
        axis = page.series_map_view.plot.getAxis("left")
        assert axis.labelText == "time (s)" and axis._tickLevels[0][:3] == [(0.5, "0.0"), (1.5, "0.5"), (2.5, "1.0")]
        assert page.series_trace_plot.plot.getAxis("bottom").labelText == "time (s)"
        assert "time = frame × exposure" in page.series_title_label.text()
        assert np.allclose(page.series_trace_plot.figure_state()["curves"][0][1], series.times)
    finally:
        _close(page)


def test_frames_without_a_time_offer_no_time_axis(tmp_path: Path) -> None:
    page, _paths = _giwaxs_page(tmp_path, 3)
    try:
        _build(page, 3)
        combo = page.series_axis_combo
        item = combo.model().item(combo.findData("time"))
        assert page._series_map.times is None and not item.isEnabled() and item.toolTip() == series_texts.NO_TIME
        assert page.series_trace_plot.plot.getAxis("bottom").labelText == "frame"
    finally:
        _close(page)


# -- what the map is ----------------------------------------------------------------------------------------------------


def test_the_title_line_says_what_the_map_is_and_when_it_no_longer_is(tmp_path: Path) -> None:
    page, _paths = _giwaxs_page(tmp_path, 6)
    try:
        title, stale, combo = page.series_title_label, page.series_stale_label, page.series_curve_combo
        assert title.isHidden() and stale.isHidden()
        _build(page, 6)
        assert not title.isHidden() and title.text() == "I(q) · 6 frames of film" and stale.isHidden()
        assert page.series_info_label.text().startswith("6 frames × ")  # the how-to: one line, whole in its tooltip
        other = combo.findText("In-plane sector I(q)")
        combo.setCurrentIndex(other)  # another curve chosen, not built
        assert not stale.isHidden() and stale.text() == (
            "The Curve box shows another curve (In-plane sector I(q)) — Build Map stacks it.")
        assert title.text() == "I(q) · 6 frames of film"  # … the map is still the one it was
        combo.setCurrentIndex(combo.findData("radial"))
        assert stale.isHidden()
        page.view_model.set_incidence(0.35)  # the settings are no longer those of the map
        page.run_analysis()
        _done(page)
        assert not stale.isHidden() and stale.text() == "Built with other settings — Build Map again."
        apply_language("zh", [page])
        page.refresh_language()
        assert stale.text() == "这张热图是用另一套设置生成的——请再点“生成热图”。" and title.text() == "I(q) · film 的 6 帧"
        apply_language("en", [page])
        page.refresh_language()
        _build(page, 6)
        assert stale.isHidden() and title.text() == "I(q) · 6 frames of film"
        page.series_first_spin.setValue(2)  # a part of the series, every other frame of it
        page.series_last_spin.setValue(5)
        page.series_step_spin.setValue(2)
        _build(page, 2)
        assert page._series_map.numbers == (2, 4) and title.text() == "I(q) · 2 of 6 frames of film"
        page.series_first_spin.setValue(6)
        page.build_series_map()
        assert "Frames 6 to 5 hold fewer than two frames" in page.status_text() and page._series_map.rows == 2
        page.clear_files()
        assert title.isHidden() and stale.isHidden() and page.series_wide_button.isHidden()
    finally:
        _close(page)


def test_the_line_of_a_finished_map_follows_the_language(tmp_path: Path) -> None:
    page, _paths = _giwaxs_page(tmp_path, 3)
    try:
        _build(page, 3)
        assert page.status_text() == "Series map: 3 frames of I(q)"
        apply_language("zh", [page])
        page.refresh_language()
        assert page.status_text() == "序列热图：I(q)，3 帧"
        assert page.series_stages_label.text() == "寻找阶段至少需要 5 帧（这张热图只有 3 帧）。"
        apply_language("en", [page])
        page.refresh_language()
        assert page.status_text() == "Series map: 3 frames of I(q)"
    finally:
        _close(page)


def test_a_frame_range_is_said_and_ends_with_the_list(tmp_path: Path) -> None:
    """A range set for one map is said under the buttons and in the title of the folded options, and Clear
    ends it — it never holds a later list at “frames 1 to 1” (review: Build Map refused every later series)."""
    page, paths = _giwaxs_page(tmp_path, 6)
    title = page.series_options.toggle_button
    try:
        assert title.text() == "Options: frames, cut, axes" and "only" not in page.series_info_label.text()
        page.series_first_spin.setValue(3)
        page.series_last_spin.setValue(5)
        assert "Build Map takes frames 3 to 5 only (Options)." in page.series_info_label.text()
        assert title.text() == "Options: frames 3 to 5, cut, axes"
        page.series_last_spin.setValue(0)
        assert title.text() == "Options: frames 3 to the last, cut, axes"
        apply_language("zh", [page])
        page.refresh_language()
        assert title.text() == "选项：第 3 帧至最后一帧、切线、坐标轴" and "“生成热图”只取第 3 帧至最后一帧" in page.series_info_label.text()
        apply_language("en", [page])
        page.refresh_language()
        assert title.text() == "Options: frames 3 to the last, cut, axes"
        page.series_last_spin.setValue(5)
        page.clear_files()
        assert (page.series_first_spin.value(), page.series_last_spin.value()) == (1, 0)
        assert title.text() == "Options: frames, cut, axes" and page.series_view_combo.isHidden()
        page.add_paths([str(path) for path in paths])
        _done(page)
        assert page.series_last_spin.maximum() == 6 and "only" not in page.series_info_label.text()
        _build(page, 6)
    finally:
        _close(page)


def test_a_trace_window_without_a_measured_point_says_so_and_is_not_saved(tmp_path: Path, monkeypatch) -> None:
    page, _paths = _giwaxs_page(tmp_path, 3)
    try:
        _build(page, 3)
        end = float(page._series_map.x[-1])
        page._series_pick_q(end + 0.5, end + 0.6)  # beyond the end of the map
        assert page.series_trace_plot.figure_state()["curves"] == []  # no value, and no legend of a curve that is not there
        assert page.series_trace_plot.empty_overlay.text == series_texts.TRACE_NO_POINT
        assert not page.series_trace_plot.empty_overlay.label.isHidden()
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: pytest.fail("nothing to save")))
        assert page.export_series_trace_table() is None
        assert (page.status_text(), page.status_level()) == (series_texts.TRACE_NO_POINT, "warning")
        page._series_pick_q(1.0, 1.2)  # back on the data: values, and no sentence over them
        assert page.series_trace_plot.empty_overlay.label.isHidden() and np.isfinite(
            page.series_trace_plot.figure_state()["curves"][0][2]).all()
    finally:
        _close(page)


def test_a_failed_stage_search_leaves_no_stages_of_before(tmp_path: Path) -> None:
    page, _paths = _giwaxs_page(tmp_path, 6)
    try:
        _build(page, 6)
        _wait(page, lambda: page._series_stages is not None)
        assert page._stage_overlay.shown
        end = float(page._series_map.x[-1])
        page.series_map_view.view_box.setRange(xRange=(end + 1.0, end + 2.0), padding=0)  # nothing of the map there
        combo = page.series_stages_range_combo
        combo.setCurrentIndex(combo.findData("shown"))
        combo.activated.emit(combo.currentIndex())
        _wait(page, lambda: page.series_stages_label.text().startswith("No stages: "))
        _done(page)
        assert page._series_stages is None and not page._stage_overlay.shown and page.series_stages_details.isHidden()
        assert not page.series_export_stages_action.isEnabled() and page.export_series_stages() is None
        assert page.status_text() == page.series_stages_label.text()  # (why there are none, not “once a map is built”)
        apply_language("zh", [page])
        page.refresh_language()
        assert page.series_stages_label.text().startswith("没有阶段：") and not page._stage_overlay.shown
        apply_language("en", [page])
        page.refresh_language()
        assert page.series_stages_label.text().startswith("No stages: ")
    finally:
        _close(page)


# -- fewer than five frames ---------------------------------------------------------------------------------------------


def test_a_map_of_fewer_than_five_frames_has_no_stages_and_says_so(tmp_path: Path, monkeypatch) -> None:
    page, _paths = _giwaxs_page(tmp_path, 3)
    try:
        _build(page, 3)
        assert not page.series_stages_row.isHidden() and page.series_stages_combo.isHidden()  # no empty box
        assert page.series_stages_label.text() == "Stages need 5 frames or more (this map has 3)."
        assert page.series_stages_details.isHidden() and not page.series_export_stages_action.isEnabled()
        monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *_a, **_k: pytest.fail("no file to ask for")))
        assert page.export_series_stages() is None
        assert page.status_text() == "Stages are found in a map of 5 frames or more (this map has 3)."
        page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("change"))
        assert page.series_trace_plot.empty_overlay.text == CHANGE_FEW and "five frames" in CHANGE_FEW
    finally:
        _close(page)


def test_a_map_of_five_frames_has_its_stages_and_their_export(tmp_path: Path) -> None:
    page, _paths = _giwaxs_page(tmp_path, 6)
    try:
        _build(page, 6)
        _wait(page, lambda: page._series_stages is not None)
        assert not page.series_stages_combo.isHidden() and page.series_export_stages_action.isEnabled()
        written = page.export_series_stages(tmp_path / "stages.csv")
        lines = written.read_text(encoding="utf-8").splitlines()
        assert "# curve: radial - I(q)" in lines and any(line.startswith("# compared: q (1/A) from ") for line in lines)
        # Only the x range the map shows, when asked for: the stages follow its zoom.
        whole = (float(page._series_stages.q.min()), float(page._series_stages.q.max()))
        assert page._stages_x_range is None
        page.series_map_view.view_box.setRange(xRange=(0.9, 1.4), padding=0)
        combo = page.series_stages_range_combo
        combo.setCurrentIndex(combo.findData("shown"))
        combo.activated.emit(combo.currentIndex())
        _wait(page, lambda: page._stages_x_range is not None and page._series_stages is not None)
        low, high = float(page._series_stages.q.min()), float(page._series_stages.q.max())
        assert 0.9 <= low < high <= 1.4 and (low, high) != whole
    finally:
        _close(page)


# -- the halves of a signed cut ------------------------------------------------------------------------------------------


def test_a_gisaxs_map_is_stacked_in_the_halves_chosen_and_says_so(tmp_path: Path, monkeypatch) -> None:
    page = _gisaxs_page(tmp_path)
    try:
        _build(page, 5)
        combo = page.series_view_combo
        assert not combo.isHidden() and combo.currentData() == "both"
        assert [combo.itemData(index, 3) for index in range(combo.count())] == list(VIEW_TIPS)  # (Qt.ToolTipRole)
        signed = page._series_map
        assert signed.x.min() < 0 < signed.x.max() and signed.view == "both"
        mean = combo.findData("mean")
        combo.setCurrentIndex(mean)
        combo.activated.emit(mean)
        _done(page)
        series = page._series_map
        assert series.view == "mean" and series.x_label == "|qy| (Å⁻¹)" and series.x.min() > 0 and series.rows == 5
        assert "mean of both halves" in page.series_title_label.text()
        assert page.series_map_view.plot.getAxis("bottom").labelText == "|qy| (Å⁻¹)"
        asked = _answers(monkeypatch, tmp_path / "out")
        saved = {}
        monkeypatch.setattr(page.view_model, "save_image_figure", lambda path, image, **state: saved.update(state) or Path(path))
        assert page.export_series_figure().name == "film_horizontal_mean_map.png"
        assert saved["title"] == "Horizontal cut I(qy) — 5 frames · mean of both halves" and saved["x_label"] == "|qy| (Å⁻¹)"
        written = page.export_series_csv()
        assert asked[-1] == ("Export Series Map", "film_horizontal_mean_map.csv") and written.name == asked[-1][1]
        lines = written.read_text(encoding="utf-8").splitlines()
        assert "# halves: mean of both halves on |qy| (a point of one half alone where the other has none)" in lines
        assert "# cut of the series: horizontal cut: rows 202-206 for every frame" in lines  # the pixel rows of the band
        assert next(line for line in lines if not line.startswith("#")).startswith("frame \\ |qy| (1/A),")
        both = combo.findData("both")
        combo.setCurrentIndex(both)
        combo.activated.emit(both)
        assert page._series_map.view == "both" and np.array_equal(page._series_map.x, signed.x)
    finally:
        _close(page)


def test_a_curve_with_one_sign_has_no_halves_to_choose(tmp_path: Path) -> None:
    page, _paths = _giwaxs_page(tmp_path, 3)
    try:
        _build(page, 3)
        assert page.series_view_combo.isHidden() and page._series_map.view == "both"
    finally:
        _close(page)


# -- saving -------------------------------------------------------------------------------------------------------------


def test_the_files_of_a_map_are_named_after_it_and_say_what_they_hold(tmp_path: Path, monkeypatch) -> None:
    page, paths = _giwaxs_page(tmp_path, 6)
    try:
        _build(page, 6)
        page.file_list.setCurrentRow(0)  # the frame on screen is the first; the row picked is the fourth
        _done(page)
        page._series_pick_row(3.5)
        page._series_pick_q(1.0, 1.2)
        asked = _answers(monkeypatch, tmp_path / "out")
        saved = {}
        monkeypatch.setattr(page.view_model, "save_curves_figure",
                            lambda path, curves, **state: saved.update(state, curves=curves) or Path(path))
        # The frame is named as Batch Export names it (its file; 0-based in a multi-frame file): film_03.
        assert page.export_series_frame_figure().name == "film_03_radial.png"
        assert saved["title"] == "Frame 4: film_03.tif" and len(saved["curves"]) == 1
        assert page.export_series_trace_figure().name == "film_radial_trace_intensity.png"
        frame = page.export_series_frame_table()
        lines = frame.read_text(encoding="ascii").splitlines()
        assert frame.name == "film_03_radial.csv" and f"# frame 4: {paths[3]}" in lines
        table = [line for line in lines if not line.startswith("#")]
        assert table[0] == "q (1/A),intensity" and len(table) == page._series_map.x.size + 1
        trace = page.export_series_trace_table()
        lines = trace.read_text(encoding="ascii").splitlines()
        assert "# mean intensity in the window 1 - 1.2 (1/A)" in lines
        table = [line for line in lines if not line.startswith("#")]
        assert table[0] == "frame,file,intensity" and table[4].startswith("4,film_03.tif,") and len(table) == 7
        values = [float(line.split(",")[2]) for line in table[1:]]
        assert np.allclose(values, page._series_map.trace(1.1, 0.1), rtol=1e-6) and np.all(np.diff(values) > 0)
        peaks = page.export_series_track()
        lines = peaks.read_text(encoding="ascii").splitlines()  # the width is said for what it is
        assert any(line.startswith("# fwhm is the moment width: 2.355 x the standard deviation") for line in lines)
        assert "# position and fwhm in 1/A; area: intensity x 1/A (integrated over x); height: intensity above the background" in lines
        assert [name for _title, name in asked] == [
            "film_03_radial.png", "film_radial_trace_intensity.png", "film_03_radial.csv",
            "film_radial_trace_intensity.csv", "film_radial_peaks.csv"]
        # Every table has its record next to it: what the map holds and the settings it was built with.
        record = json.loads(frame.with_suffix(".json").read_text(encoding="utf-8"))
        assert (record["table"], record["frame"], record["settings"]["mode"]) == ("curve of one frame", 4, "giwaxs")
        assert json.loads(peaks.with_suffix(".json").read_text(encoding="utf-8"))["window"] == [1.0, 1.2]
        assert json.loads(trace.with_suffix(".json").read_text(encoding="utf-8"))["value"] == "intensity"
        page.series_trace_combo.setCurrentIndex(page.series_trace_combo.findData("fwhm"))
        assert page.series_trace_plot.plot.getAxis("left").labelText == "2.355 σ (Å⁻¹)"
        assert [curve[0] for curve in page.series_trace_plot.figure_state()["curves"]] == ["width"]  # (not “fwhm”)
        assert page.series_trace_combo.currentText() == "Peak width (2.355 σ)" and "moment width" in page.series_trace_combo.toolTip()
        assert page.export_series_trace_table().name == "film_radial_trace_fwhm.csv"
    finally:
        _close(page)


def test_a_row_of_a_set_of_module_files_is_named_by_its_scan() -> None:
    def frame(**more):
        values = dict(path=Path("/data/scan_00007_m01.nxs"), frame_index=3, frame_count=12, summed_frames=(),
                      metadata={"layout": "module_set", "exposure_time_s": 0.998, "timestamp": "2024-10-26T01:38:05"})
        return SimpleNamespace(**{**values, **more})

    label = row_label(frame())
    assert label == "scan_00007 frame 4 of 12" and label.name == "scan_00007" and label.source == "/data/scan_00007_m01.nxs"
    assert (label.frame_index, label.frame_count, label.summed, label.exposure_s) == (3, 12, 1, 0.998)
    summed = row_label(frame(frame_index=10, summed_frames=((Path("/data/scan_00007_m01.nxs"), 11),)))
    assert summed == "scan_00007 frame 11 of 12 (+1 summed)" and summed.summed == 2
    plain = row_label(frame(path=Path("/data/film_m1.nxs"), metadata={"layout": "stack"}))
    assert plain == "film_m1.nxs frame 4 of 12"  # only named like a module: its own file
    single = row_label(frame(path=Path("/data/film_003.tif"), frame_index=0, frame_count=1, metadata={}))
    assert single == "film_003.tif" and single.timestamp == "" and single.exposure_s is None


# -- the flow -----------------------------------------------------------------------------------------------------------


def test_the_card_offers_what_it_asks_for(tmp_path: Path, monkeypatch) -> None:
    page, paths = _giwaxs_page(tmp_path, 1)
    try:
        button = page.series_empty.action_button
        assert not button.isHidden() and button.text() == "Open Folder…" and page.series_options.isHidden()
        opened = []
        monkeypatch.setattr(page, "open_folder", lambda: opened.append(True))
        button.click()
        assert opened == [True]
        from tests.test_assistant_calibration import save_tiff

        page.add_paths([str(save_tiff(tmp_path / f"film_{index:02d}.tif", _ring_frame(200.0, seed=index))) for index in (1, 2)])
        _done(page)
        assert button.text() == "Build Map" and button.isEnabled() and not page.series_options.isHidden()
        button.click()
        _wait(page, lambda: page._series_map is not None and not page.batch_running())
        assert page._series_map.rows == 3 and page.series_empty.isHidden()
    finally:
        _close(page)


def test_a_picked_row_is_shown_in_analyze_when_that_is_asked_for(tmp_path: Path) -> None:
    page, _paths = _giwaxs_page(tmp_path, 4)
    try:
        _build(page, 4)
        page.file_list.setCurrentRow(0)
        _done(page)
        page.series_map_view.positionClicked.emit(1.0, 2.5)  # a click on the third row
        _app().processEvents()
        assert page._series_row == 2 and page.file_list.currentRow() == 0 and not page._series_follow_timer.isActive()
        page.series_follow_check.setChecked(True)
        page.series_map_view.positionClicked.emit(1.0, 3.5)
        assert page._series_follow_timer.isActive()  # not at once: dragging the band reads no frame
        _wait(page, lambda: page.file_list.currentRow() == 3)
        _done(page)
        assert page.view_model.state.analysis.path.name == "film_03.tif" and page.current_right() == "series"
    finally:
        _close(page)


def test_wide_gives_the_tab_the_room_of_the_image_and_gives_it_back(tmp_path: Path) -> None:
    page, _paths = _giwaxs_page(tmp_path, 6)
    try:
        page.resize(1096, 772)  # the page of a 1280 × 800 window
        page.show()
        page.show_right("series")
        _build(page, 6)
        for _ in range(8):
            _app().processEvents()
        button, splitter = page.series_wide_button, page.splitter
        before = splitter.sizes()
        assert not button.isHidden() and not button.isChecked() and page.series_plots.stacked()
        narrow_width = page.series_profile_plot.width()
        button.setChecked(True)
        for _ in range(8):
            _app().processEvents()
        wide = splitter.sizes()
        assert wide[0] == before[0] and wide[2] > before[2] and wide[1] == max(
            splitter.widget(1).minimumSizeHint().width(), splitter.widget(1).minimumWidth())
        assert page.series_plots.stacked()  # Wide enlarges each curve without putting them side by side.
        assert page.series_trace_plot.y() > page.series_profile_plot.y()
        assert page.series_profile_plot.width() > narrow_width
        assert page.series_trace_plot.width() == page.series_profile_plot.width()
        scroll = page.series_scroll
        plots_bottom = page.series_plots.mapTo(scroll, page.series_plots.rect().bottomLeft()).y()
        map_top = page.series_map_view.mapTo(scroll, page.series_map_view.rect().topLeft()).y()
        assert 0 <= map_top < plots_bottom <= scroll.widget().height()
        # The tab asks for no more height after it was tall (sections opened, another language, folded again):
        # the map and the plots ask for their least height, not for the height they last had.
        least = scroll.least_height(scroll.viewport().width())
        page.resize(1096, 1500)  # a window with room to spare: the map and the plots are tall
        for _ in range(8):
            _app().processEvents()
        assert page.series_map_view.height() > 300
        for section in (page.series_stages_details, page.series_options):
            section.set_expanded(True)
        from src.gimap.app.presentation.theme import apply_theme

        try:
            for mode, language in (("dark", "zh"), ("light", "en")):  # (every part is asked for its size again)
                apply_theme(mode, 9)
                apply_language(language, [page])
                page.refresh_language()
                for _ in range(8):
                    _app().processEvents()
        finally:
            apply_theme("light", 9)  # the tests' theme (``conftest.py``)
        for section in (page.series_stages_details, page.series_options):
            section.set_expanded(False)
        page.resize(1096, 772)
        for _ in range(8):
            _app().processEvents()
        assert scroll.least_height(scroll.viewport().width()) == least
        assert page.series_plots.mapTo(scroll, page.series_plots.rect().bottomLeft()).y() == plots_bottom
        button.setChecked(False)
        for _ in range(8):
            _app().processEvents()
        assert splitter.sizes() == before and page.series_plots.stacked()
    finally:
        _close(page)


# -- every word of it in both languages ---------------------------------------------------------------------------------


def test_every_text_of_the_series_tab_has_its_chinese() -> None:
    """The sentences composed at run time (their templates keep their fields), and every fixed text of the
    tab's own widgets: labels, buttons, check boxes, the items of its boxes, tooltips, the Export menu."""
    import string

    from PyQt5.QtWidgets import QAbstractButton, QAction, QComboBox, QLabel, QSpinBox, QWidget

    from src.gimap.app.presentation.i18n import ZH, translate
    from src.gimap.features.analyze.presentation import frames_model
    from src.gimap.features.analyze.presentation.views import series_view

    def fields(text: str) -> list:
        return sorted({name for _text, name, _spec, _conversion in string.Formatter().parse(text) if name})

    templates = [*series_texts.TEMPLATES, *frames_model.CUT_TEXTS, *series_view.VIEW_TIPS, series_view.CHANGE_FEW,
                 "Incomplete sums", "Export the Frame's Curve", "Export the Trace", "The trace has no values yet.",
                 "Could not export the curve: {error}", "Could not export the trace: {error}"]
    missing = [text for text in templates if text not in ZH]
    assert missing == []
    assert [text for text in templates if fields(text) != fields(ZH[text])] == []

    page = _page(_context(_profiles()))
    try:
        fixed = []
        for name, widget in vars(page).items():
            if not name.startswith("series_") or name in ("series_curve_combo",):  # (its items are the curves: data)
                continue
            if isinstance(widget, QAction):
                fixed += [widget.text(), widget.toolTip()]
            if not isinstance(widget, QWidget):
                continue
            fixed.append(widget.toolTip())
            if isinstance(widget, (QLabel, QAbstractButton)):
                fixed.append(widget.text())
            if isinstance(widget, QComboBox):
                fixed += [widget.itemText(index) for index in range(widget.count())]
            if isinstance(widget, QSpinBox):
                fixed += [widget.prefix(), widget.suffix(), widget.specialValueText()]
            for child in widget.findChildren(QLabel) if name in ("series_options", "series_stages_details") else ():
                fixed.append(child.text())
            for child in widget.findChildren(QAbstractButton) if name in ("series_options", "series_stages_details") else ():
                fixed += [child.text(), child.toolTip()]
        words = {text for text in fixed if text and any(letter.isalpha() for letter in text.replace("|x|", ""))}
        assert len(words) > 40
        assert sorted(text for text in words if translate(text, "zh") is None) == []
    finally:
        _close(page)
