"""The results database of a series folder (``shared/series_store``): curves and fits of every frame, the files a
watch skipped; written while it is read, and read back as it was written."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from src.gimap.shared.series_store import SCHEMA, SeriesStore, read_curves, stacked, store_path


def _curve(frame: int, shift: float = 0.0, points: int = 50) -> dict:
    x = np.linspace(0.01, 0.2, points)
    return dict(curve="horizontal", file=f"C:/run/film_{frame:03d}.tif", frame_index=0, frame=frame + 1,
                x=x, intensity=100.0 * np.exp(-0.5 * ((x - 0.1 - shift) / 0.01) ** 2) + 1.0,
                sigma=np.full(points, 0.5), x_label="|qy| (Å⁻¹)", label=f"film_{frame:03d}", time_s=2.0 * frame)


def test_curves_come_back_as_written_in_frame_order_and_a_frame_written_again_is_replaced(tmp_path):
    path = store_path(tmp_path)
    with SeriesStore(path) as store:
        assert store.put_curves("film", [_curve(2), _curve(0), _curve(1)]) == 3
        store.put_curve("film", **{**_curve(1, shift=0.01), "curve": "horizontal"})  # the same frame again
        store.set_info("film", {"curve_title": "Horizontal cut", "settings": {"mode": "gisaxs", "nan": float("nan")}})
        store.set_info("film", {"kind": "gisaxs"})
    with SeriesStore(path, read_only=True) as store:
        assert store.series_names() == ["film"]
        assert store.curve_keys("film") == [("horizontal", 3, "|qy| (Å⁻¹)")]
        curves = store.curves("film", "horizontal")
        assert [item.frame for item in curves] == [1, 2, 3]
        np.testing.assert_allclose(curves[1].intensity, _curve(1, shift=0.01)["intensity"])
        np.testing.assert_allclose(curves[0].sigma, 0.5)
        assert curves[2].time_s == 4.0 and curves[2].label == "film_002"
        assert store.info("film") == {"curve_title": "Horizontal cut", "settings": {"mode": "gisaxs", "nan": None},
                                      "kind": "gisaxs"}


def test_fits_and_skipped_files_are_kept_per_series(tmp_path):
    path = store_path(tmp_path)
    with SeriesStore(path) as store:
        store.put_fits("film", "model", [
            dict(file="a.dat", frame=1, state="ok", chi2=1.2, values={"R": 5.0}, errors={"R": float("nan")},
                 flags=["bounds: R"], start_from="previous"),
            dict(file="b.dat", frame=2, state="failed", record={"why": "no points"})])
        store.log_skipped("film", "c.tif", "held by its writer", what="waiting")
        store.log_skipped("other", "d.tif", "unreadable")
    with SeriesStore(path, read_only=True) as store:
        fits = store.fits("film", "model")
        assert [(fit.frame, fit.state, fit.chi2) for fit in fits] == [(1, "ok", 1.2), (2, "failed", None)]
        assert fits[0].values == {"R": 5.0} and fits[0].errors == {"R": None} and fits[0].flags == ("bounds: R",)
        assert fits[1].record == {"why": "no points"}
        assert [(item.file, item.what) for item in store.skipped("film")] == [("c.tif", "waiting")]
        assert len(store.skipped()) == 2
        assert store.summary()["series"][0]["fits"] == {"model": 2}


def test_a_database_can_be_read_while_it_is_written(tmp_path):
    path = store_path(tmp_path)
    writer = SeriesStore(path)
    writer.put_curves("live", [_curve(0)])
    with SeriesStore(path, read_only=True) as reader:
        assert len(reader.curves("live", "horizontal")) == 1
        writer.put_curves("live", [_curve(1)])
        assert len(reader.curves("live", "horizontal")) == 2
    writer.close()


def test_read_curves_takes_the_series_and_curve_named_or_the_last_and_first(tmp_path):
    path = store_path(tmp_path)
    with SeriesStore(path) as store:
        store.put_curves("one", [_curve(0), _curve(1)])
        store.put_curves("two", [{**_curve(0), "curve": "radial"}, {**_curve(1), "curve": "vertical"}])
    series, curve, curves = read_curves(path)
    assert (series, curve, len(curves)) == ("two", "radial", 1)
    assert read_curves(path, "one", "horizontal")[2][1].frame == 2
    with pytest.raises(ValueError, match="no curve"):
        read_curves(path, "one", "vertical")


def test_stacked_puts_curves_of_other_grids_on_the_first_without_bridging_gaps():
    first = _curve(0)
    other = _curve(1)
    other["x"] = np.concatenate([np.linspace(0.02, 0.08, 20), np.linspace(0.15, 0.25, 20)])  # a gap of 0.07
    other["intensity"] = np.ones(40)
    from src.gimap.shared.series_store import StoredCurve

    curves = [StoredCurve("s", "c", "f", 0, index + 1, item["x"], item["intensity"]) for index, item in
              enumerate((first, other))]
    x, image = stacked(curves)
    np.testing.assert_allclose(x, first["x"])
    assert np.isnan(image[1][x < 0.02]).all() and np.isnan(image[1][(x > 0.085) & (x < 0.145)]).all()
    assert np.allclose(image[1][(x > 0.02) & (x < 0.08)], 1.0)


def test_not_a_database_and_a_newer_schema_are_said(tmp_path):
    junk = tmp_path / "junk.sqlite"
    junk.write_text("not sqlite", encoding="utf-8")
    with pytest.raises(ValueError, match="not a GIMaP results database"):
        SeriesStore(junk, read_only=True).series_names()
    path = store_path(tmp_path / "new")
    SeriesStore(path).put_curves("s", [_curve(0)])
    with sqlite3.connect(path) as db:
        db.execute("UPDATE store SET value = ? WHERE key = 'schema'", (str(SCHEMA + 1),))
    with pytest.raises(ValueError, match="newer GIMaP"):
        SeriesStore(path, read_only=True).series_names()
