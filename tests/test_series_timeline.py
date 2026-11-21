"""The timeline of a Series map (``domain/series_timeline``): χ² of each frame to the one before and to a reference
frame, jumps where the series changes at once, and its table; the map's rows in a results database and back."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.gimap.features.analyze.application import (
    build_series_map,
    export_series_timeline,
    rows_from_store,
    stages_of,
    store_rows,
    timeline_of,
)
from src.gimap.features.analyze.application.series_map import RowLabel
from src.gimap.features.analyze.domain.series_timeline import column_noise, jump_rows, series_timeline
from src.gimap.features.analyze.infrastructure.adapters.csv_export import CsvCurveWriter
from src.gimap.shared.series_store import SeriesStore, store_path

X = np.linspace(0.01, 0.3, 200)


def _frames(count: int = 60, jump_at: int = 30, seed: int = 0) -> np.ndarray:
    """A peak that drifts slowly, then moves at once at ``jump_at``; Poisson noise of about 100 counts."""
    rng = np.random.default_rng(seed)
    image = []
    for row in range(count):
        centre = 0.1 + 0.0002 * row + (0.05 if row >= jump_at else 0.0)
        level = 100.0 + 400.0 * np.exp(-0.5 * ((X - centre) / 0.01) ** 2)
        image.append(rng.poisson(level).astype(float))
    return np.array(image)


def test_frames_that_differ_by_noise_give_about_one_and_a_jump_stands_out():
    image = _frames()
    timeline = series_timeline(X, image)
    quiet = np.delete(timeline.neighbour[1:], 29)
    assert 0.7 < float(np.median(quiet)) < 1.6
    assert timeline.neighbour[30] > 20 * float(np.median(quiet))
    assert timeline.jumps == (30,)
    assert timeline.reference[0] == 0.0 and timeline.reference[-1] > timeline.reference[10]
    assert np.isnan(timeline.neighbour[0])


def test_the_noise_of_a_column_comes_from_the_series_and_no_column_is_quieter_than_the_usual_one():
    image = _frames(jump_at=1000)
    noise, how = column_noise(image)
    assert how == "second differences along the frames"
    # the background of 100 counts: σ ≈ 10; the top of the peak (500 counts) is noisier, never quieter
    assert 7.0 < float(np.median(noise)) < 13.0 and float(noise.min()) >= float(np.median(noise)) - 1e-9
    few, how = column_noise(image[:3])
    assert how.startswith("square root") and np.all(few >= 9.0)


def test_a_quiet_series_has_no_jumps_and_a_window_compares_only_its_columns():
    image = _frames(jump_at=1000)
    assert series_timeline(X, image).jumps == ()
    assert jump_rows([1.0, 1.1, 0.9, 1.2, 1.0]) == ()
    moved = _frames()
    outside = series_timeline(X, moved, x_range=(0.2, 0.3))  # the peak never reaches this window
    assert outside.jumps == () and outside.columns == int(np.count_nonzero((X >= 0.2) & (X <= 0.3)))
    other = series_timeline(X, moved, reference_row=59)
    assert other.reference_row == 59 and other.reference[59] == 0.0


def _rows(image):
    return [(X, image[row], "q (Å⁻¹)", RowLabel(f"film_{row:03d}", source=f"C:/run/film_{row:03d}.tif",
                                                  timestamp=f"2026-01-01T00:00:{row:02d}"),
             (Path(f"C:/run/film_{row:03d}.tif"), 0)) for row in range(image.shape[0])]


def test_the_timeline_table_names_the_jump_the_stage_and_the_record(tmp_path):
    rows = _rows(_frames())
    series = build_series_map(rows, "radial")
    timeline = timeline_of(series)
    stages = stages_of(series)
    path = export_series_timeline(CsvCurveWriter(), series, timeline, tmp_path / "film_timeline.csv", stages=stages)
    text = path.read_text(encoding="utf-8")
    assert "chi2_to_previous,chi2_to_reference,jump,stage,odd,time_s" in text
    table = [line.split(",") for line in text.splitlines() if not line.startswith("#")][1:]
    assert table[30][4] == "True" and table[30][0] == "31"
    assert table[29][5] != table[31][5]  # a stage boundary at the jump
    record = (tmp_path / "film_timeline.json").read_text(encoding="utf-8")
    assert '"jumps": [\n    31\n  ]' in record or '"jumps": [31]' in record


def test_map_rows_go_into_the_database_and_come_back_as_rows_of_the_same_map(tmp_path):
    image = _frames(count=8, jump_at=4)
    rows = _rows(image)
    path = store_path(tmp_path)
    with SeriesStore(path) as store:
        assert store_rows(store, "film", rows[:5], "radial") == 5
        only = {(str(rows[5][4][0]), 0), (str(rows[6][4][0]), 0), (str(rows[7][4][0]), 0)}
        assert store_rows(store, "film", rows, "radial", only=only) == 3  # the new rows of a watch
    name, curve, back, _info = rows_from_store(path)
    assert (name, curve, len(back)) == ("film", "radial", 8)
    again = build_series_map(back, curve)
    np.testing.assert_allclose(again.image, build_series_map(rows, "radial").image)
    assert again.times is not None and again.times[-1] == 7.0
