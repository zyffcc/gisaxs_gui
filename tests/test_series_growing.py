"""A live Series map grows from the map before (``application/series_growing.py``, ``GrowingDisplay``,
``series_later.LiveStack``): the same map, value for value, as one made whole from the same rows — every step of a
run, every halves, with frame numbers, a Sum N, a row of other points, a row put in the place of another. Synthetic
rows with invented names."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.app.presentation.components.detector_display import GrowingDisplay, display_image
from src.gimap.features.analyze.application.series_growing import GrowingSeriesMap
from src.gimap.features.analyze.application.series_map import RowLabel, build_series_map, frame_key, share_x

X_SIGNED = np.linspace(-0.2, 0.2, 81)


def _row(index: int, x=X_SIGNED, *, summed: int = 1, known=None, holes: bool = False) -> tuple:
    rng = np.random.default_rng(index)
    y = 50 * np.exp(-((x - 0.03 * np.sin(index / 5)) / 0.04) ** 2) + rng.normal(0, 1, x.size)
    if holes:
        y[rng.integers(0, x.size, 3)] = np.nan  # (pixels left out of this frame)
    label = RowLabel(f"spray_{index:04d}.edf", name=f"spray_{index:04d}.edf", source=f"C:/runs/spray_{index:04d}.edf",
                     frame_index=0, summed=summed, timestamp=f"2027-01-20T10:{index // 60:02d}:{index % 60:02d}",
                     exposure_s=0.1, incidence_deg=0.4, file_time="")
    row = (x, y, "qy (Å⁻¹)", label, (f"C:/runs/spray_{index:04d}.edf", 0))
    return row if known is None else share_x(row, known)


def _assert_same_map(grown, rows, curve="horizontal", **options) -> None:
    whole = build_series_map(rows, curve, **options)
    series = grown.series
    np.testing.assert_array_equal(series.x, whole.x)
    np.testing.assert_array_equal(series.image, whole.image)
    assert series.image.dtype == whole.image.dtype
    assert (series.labels, series.refs, series.facts) == (whole.labels, whole.refs, whole.facts)
    assert (series.x_label, series.view, series.curve, series.title, series.time_from) == (
        whole.x_label, whole.view, whole.curve, whole.title, whole.time_from)
    if whole.times is None:
        assert series.times is None
    else:
        np.testing.assert_array_equal(series.times, whole.times)
    picture, left, width = whole.display()
    np.testing.assert_array_equal(grown.picture, picture)
    assert (grown.left, grown.width) == (left, width)


@pytest.mark.parametrize("view, holes", [("both", True), ("positive", False), ("negative", False), ("folded", False),
                                         ("mean", False), ("mean", True)])
def test_a_growing_map_is_the_map_made_whole_at_every_step(view, holes) -> None:
    """Rows with pixels left out (NaN) keep the points of the first on the signed qy; folded, their halves have other
    points (a point without value is in no half), and the map is made whole every time."""
    growing, known, rows = GrowingSeriesMap(), [], []
    shown_before = []
    grows = not (holes and view != "both")
    for step, more in enumerate((1, 3, 7, 1, 12)):
        rows.extend(_row(len(rows) + k, known=known, holes=holes) for k in range(more))
        grown = growing.build(list(rows), "horizontal", view=view, title="I(qy)")
        assert grown.same == (0 if step == 0 or not grows else len(rows) - more)
        _assert_same_map(grown, rows, view=view, title="I(qy)")
        shown_before.append((grown.series.image, grown.series.image.copy()))
    for image, copy in shown_before:  # (a map shown before stays as it was)
        np.testing.assert_array_equal(image, copy)


def test_frame_numbers_a_sum_and_rows_of_other_points() -> None:
    growing, rows = GrowingSeriesMap(), [_row(k, summed=2) for k in range(4)]
    numbers = {frame_key(row[4][0], 0): 10 + k for k, row in enumerate(rows)}
    grown = growing.build(rows, "horizontal", numbers=dict(numbers), group=2)
    rows = rows + [_row(4, summed=2), _row(5, summed=1)]  # (the last sum is short)
    numbers.update({frame_key(rows[4][4][0], 0): 14, frame_key(rows[5][4][0], 0): 15})
    grown = growing.build(rows, "horizontal", numbers=dict(numbers), group=2)
    assert grown.same == 4
    _assert_same_map(grown, rows, numbers=numbers, group=2)
    assert grown.series.facts[-1].incomplete
    numbers = {key: number + 1 for key, number in numbers.items()}  # (a frame listed before them: all move)
    grown = growing.build(rows, "horizontal", numbers=dict(numbers), group=2)
    assert grown.same == 0
    _assert_same_map(grown, rows, numbers=numbers, group=2)
    other = _row(6, x=np.linspace(-0.2, 0.2, 61), summed=2)  # (another geometry: merged onto one grid, made whole)
    grown = growing.build(rows + [other], "horizontal", numbers=dict(numbers), group=2)
    assert grown.same == 0
    _assert_same_map(grown, rows + [other], numbers=numbers, group=2)


def test_a_row_put_in_the_place_of_another_and_other_options_make_the_map_whole() -> None:
    growing, known = GrowingSeriesMap(), []
    rows = [_row(k, known=known) for k in range(6)]
    growing.build(rows, "horizontal")
    rows[2] = _row(20, known=known)
    grown = growing.build(rows, "horizontal")
    assert grown.same == 0
    _assert_same_map(grown, rows)
    grown = growing.build(rows + [_row(21, known=known)], "horizontal", view="positive")
    assert grown.same == 0
    _assert_same_map(grown, rows + [_row(21, known=known)], view="positive")
    one_sign = [_row(k, x=np.linspace(0.01, 0.2, 40), known=known) for k in range(3)]
    growing.build(one_sign, "horizontal", view="mean")  # (no halves: stacked as they are)
    signed = one_sign + [_row(30, known=known)]  # (the first signed curve: every row folded now)
    grown = growing.build(signed, "horizontal", view="mean")
    assert grown.same == 0
    with pytest.raises(ValueError):
        growing.build([], "horizontal")


@pytest.mark.parametrize("log", [False, True])
def test_the_growing_display_is_display_image_of_the_whole_picture(log) -> None:
    rng = np.random.default_rng(3)
    display, rows, shown_before = GrowingDisplay(), np.empty((0, 50), np.float32), []
    for step, more in enumerate((4, 1, 9, 30, 2)):
        added = rng.normal(5, 4, (more, 50)).astype(np.float32)  # (values at or below 0: the floor of a log display)
        added[:, 7] = np.nan
        if step == 2:
            added[0, :] = -1.0  # (a row of the floor only)
        same = rows.shape[0]
        rows = np.vstack([rows, added])
        valid, shown, sample = display.show(rows, same, log)
        expected_valid = np.isfinite(rows)
        expected_shown, expected_sample = display_image(rows, expected_valid, log)
        np.testing.assert_array_equal(valid, expected_valid)
        np.testing.assert_array_equal(shown, expected_shown)
        np.testing.assert_array_equal(sample, expected_sample)
        shown_before.append(shown)
    valid, shown, _sample = display.show(rows, 0, log)  # (all new: made whole)
    np.testing.assert_array_equal(shown, display_image(rows, np.isfinite(rows), log)[0])


def test_the_live_map_of_the_page_is_the_map_stacked_whole(tmp_path) -> None:
    from tests.test_analyze_workspace import _context, _page

    page = _page(_context(None))
    try:
        known = []
        page._series_rows = [page._series_live_row(_row(k)) for k in range(30)]
        page._series_key, page._series_built, page._series_view = "horizontal", {}, "mean"
        for more in (0, 5, 1):
            page._series_rows.extend(page._series_live_row(_row(100 + len(page._series_rows) + k, known=known))
                                     for k in range(more))
            live = page._series_stacking(live=True)()
            whole = page._series_stacking()()
            assert live[1] == whole[1]  # (the name)
            np.testing.assert_array_equal(live[0].image, whole[0].image)
            np.testing.assert_array_equal(live[2][0], whole[2][0])
            np.testing.assert_array_equal(live[3][0], whole[3][0])
            np.testing.assert_array_equal(live[3][1][1], whole[3][1][1])
        options = ("horizontal", "mean", getattr(page, "_series_curve_title", ""), 1)
        assert page._series_live_stack()._map.same_rows(page._series_rows, options, None) == 36
        page._series_live_forget()
        assert page._series_live_stack()._map.same_rows(page._series_rows, options, None) == 0
    finally:
        page.dispose()


def test_the_exposure_times_of_the_rows_are_kept_while_they_are_the_same_rows() -> None:
    from tests.test_analyze_workspace import _context, _page

    page = _page(_context(None))
    try:
        rows = [_row(k) for k in range(4)]
        page._series_rows = list(rows)
        assert page._series_exposures() == {0.1}
        longer = _row(9)
        longer[3].exposure_s = 1.0  # (a RowLabel carries its facts as attributes)
        page._series_rows = rows + [longer]
        assert page._series_exposures() == {0.1, 1.0}
        page._series_rows = [rows[0], _row(10), *rows[2:]]  # (a row put in the place of another: all again)
        assert page._series_exposures() == {0.1}
        page._series_live_forget()
        assert page._series_exposures_kept is None
    finally:
        page.dispose()


def _times_cases() -> dict:
    """Invented frames whose times come from each of the ways ``series_times`` reads them, and from one way at first
    and another once more rows arrived."""
    from src.gimap.features.analyze.domain.series_map import RowFacts

    def stamp(seconds: float) -> str:
        whole, part = divmod(seconds, 1.0)
        return f"2027-01-20T10:{int(whole) // 60:02d}:{int(whole) % 60:02d}" + (f".{round(part * 1000):03d}" if part else "")

    def files(n, *, back_at=None):  # one file a frame, whole-second stamps, file times 0.1 s apart
        return [RowFacts(number=i + 1, source=f"C:/runs/spray_{i:04d}.edf", exposure_s=0.1, timestamp=stamp(i // 10),
                         file_time=stamp(i * 0.1 + 0.03 - (5 if i == back_at else 0))) for i in range(n)]

    def scans(n, sources, *, stamps=True, frame_times=False):  # multi-frame files of 10 frames
        return [RowFacts(number=i + 1, source=f"C:/runs/scan_{i * sources // n}.nxs", frame_index=i % 10, frame_count=10,
                         exposure_s=0.5, timestamp=stamp(60 * (i * sources // n)) if stamps else "",
                         frame_time_s=0.5 * i if frame_times else None) for i in range(n)]

    late_source = scans(30, 1, stamps=False) + [
        RowFacts(number=31 + i, source="C:/runs/scan_9.nxs", frame_index=i, frame_count=10, exposure_s=0.5,
                 timestamp="") for i in range(5)]
    return {"file_times": files(40), "file times running back": files(40, back_at=25),
            "timestamps": scans(30, 1, frame_times=True), "exposure": scans(30, 1, stamps=False),
            "start_exposure": scans(30, 3), "a second source later": late_source}


@pytest.mark.parametrize("case", list(_times_cases()))
def test_the_times_of_a_growing_map_are_the_times_of_all_its_rows(case) -> None:
    """``RowTimes`` reads each row once and keeps what each way found of the rows so far: grown in pieces, its times
    are those of ``series_times`` of all the rows so far, value for value, whichever way they come from."""
    from src.gimap.features.analyze.application.series_times import RowTimes, series_times

    facts = _times_cases()[case]
    grown, at, ways = RowTimes(), 0, set()
    for more in (1, 2, 5, 3, 1, 8, 13, 40):
        grown.extend(facts[at:at + more])
        at = min(len(facts), at + more)
        (times, way), (expected, expected_way) = grown.times(), series_times(facts[:at])
        assert way == expected_way
        assert (times is None) == (expected is None)
        if times is not None:
            assert times.dtype == expected.dtype and times.tobytes() == expected.tobytes()
        ways.add(way)
    assert {"file_times": {"", "file_times"}, "file times running back": {"", "file_times", "timestamps"},
            "timestamps": {"", "timestamps"}, "exposure": {"", "exposure"}, "start_exposure": {"", "exposure", "start_exposure"},
            "a second source later": {"", "exposure"}}[case] == ways
