"""The In-situ series page with a chained fit: each frame from the result of the one before, Predict parameters (a
stand-in here, the page's ``predictor``: its method is not this test's) where the chain breaks, a backward pass, the
Flags column, and the fits saved as a table with flag columns, as NXcanSAS and into the results database."""

from __future__ import annotations

import csv

import pytest

from src.gimap.shared.nxcansas import read_nxcansas
from src.gimap.shared.series_store import SeriesStore
from tests.test_fit_series import _model, _open_series, _pages, _series, _wait

RADII = (5.0, 5.2, 5.4, 9.0, 9.2)
"""A sample that changes at once between frame 3 and 4: refined from 5.4 nm, frame 4 is a poor fit."""


def test_a_chained_series_asks_predict_only_where_the_chain_breaks_and_saves_its_flags(tmp_path, monkeypatch) -> None:
    single, series = _pages()
    try:
        folder = _series(tmp_path, RADII)
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_model(5.2))
        assert _open_series(series, folder)
        asked = []

        def predictor(settings, previous, new_stage):
            assert settings.method == "predict"
            return lambda data, start: asked.append(len(data.q)) or _model(9.0)

        monkeypatch.setattr(series, "predictor", predictor)
        series.skip_odd_check.setChecked(False)
        series.start_chain.setChecked(True)
        series.backward_check.setChecked(True)
        assert series._start_choice() == "chain"
        series.start()
        _wait(series)
        fits = series.fits
        assert len(fits) == 5 and all(frame.ok for frame in fits.values())
        assert asked  # Predict parameters was asked …
        jump = fits[3]
        assert jump.chain and jump.start_from == "predict"  # … where the chain broke
        assert jump.result.model.get((0, "R")).value == pytest.approx(9.0, rel=0.05)
        assert [fits[index].start_from for index in (1, 2)] in (["previous", "previous"], ["next", "next"])
        assert "Backward pass" in series.log_view.toPlainText()
        flags_column = series.results_table.columnCount() - 3  # (before Points and Halves)
        assert series.results_table.horizontalHeaderItem(flags_column).text() == "Flags"
        assert "chain broke" in series.results_table.item(3, flags_column).text()

        (tmp_path / "out").mkdir()
        table = series.save_table(tmp_path / "out" / "fits.csv")
        with open(table, encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert rows[3]["start_from"] == "predict" and "chain broke" in rows[3]["flags"]
        assert {row["fit_pass"] for row in rows} <= {"forward", "backward"}

        written = series.save_fits_nxcansas(tmp_path / "out" / "fits.h5")
        curves = read_nxcansas(written)
        assert len(curves) == 10 and curves[0].title.startswith("frame 1 ") and curves[1].title.endswith("the model")

        store = series.save_results_database(tmp_path / "out" / "gimap_series.sqlite")
        with SeriesStore(store, read_only=True) as database:
            (name,) = database.series_names()
            stored = database.fits(name, "model")
        assert len(stored) == 5 and stored[3].start_from == "predict"
        assert stored[3].values["1_Sphere_R_nm"] == pytest.approx(9.0, rel=0.05)
        assert any(flag.startswith("chain broke") for flag in stored[3].flags)
    finally:
        series.dispose()
        single.dispose()


def test_the_trend_is_drawn_against_time_when_the_curves_say_when_they_were_measured(tmp_path) -> None:
    import json

    import numpy as np

    from src.gimap.features.fitting.application.single_fit import evaluate
    from tests.test_fit_series import Q_NM

    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    for index, (radius, second) in enumerate(((5.0, 0), (5.2, 10), (5.4, 25))):
        exact = evaluate(_model(radius), Q_NM)
        sigma = np.sqrt(exact) + 0.5
        measured = exact + np.random.default_rng(index).normal(0.0, sigma)
        observation = {"source": "native_detector_columns", "timestamp": f"2026-01-01T10:00:{second:02d}+0100"}
        header = "GIMaP Analyze fit input" + chr(10) + "observation: " + json.dumps(observation)
        np.savetxt(folder / f"run_{index + 1:05d}_fit_input.dat", np.column_stack([Q_NM / 10.0, measured, sigma]),
                   header=header)
    single, series = _pages()
    try:
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_model(5.2))
        assert _open_series(series, folder)
        series.skip_odd_check.setChecked(False)
        series.start()
        _wait(series)
        assert not series.trend_axis_combo.isHidden()
        series.trend_axis_combo.setCurrentIndex(series.trend_axis_combo.findData("time"))
        (_name, x, _y), *_rest = series.trend_plot._source
        assert list(x) == [0.0, 10.0, 25.0]
        assert series.trend_plot.plot.getAxis("bottom").labelText == "time (s)"
        series.trend_axis_combo.setCurrentIndex(series.trend_axis_combo.findData("frame"))
        assert list(series.trend_plot._source[0][1]) == [1.0, 2.0, 3.0]
    finally:
        series.dispose()
        single.dispose()


def test_every_new_text_of_the_page_has_its_chinese() -> None:
    from src.gimap.app.presentation.i18n import ZH

    single, series = _pages()
    try:
        texts = [series.start_chain.text(), series.start_chain.toolTip(), series.backward_check.text(),
                 series.backward_check.toolTip(), series.save_nxcansas_action.text(), series.save_nxcansas_action.toolTip(),
                 series.save_store_action.text(), series.save_store_action.toolTip(), series.trend_axis_combo.toolTip(),
                 *(series.trend_axis_combo.itemText(index) for index in range(series.trend_axis_combo.count()))]
        from src.gimap.features.fitting.application.series_chain import FLAG_KINDS
        from src.gimap.features.fitting.presentation.single import series_chain, series_results

        texts += [*FLAG_KINDS, series_results.FLAGS_TIP, series_chain.BACKWARD_STARTED, series_chain.BACKWARD_KEPT,
                  series_chain.CHAIN_REFINE_ONLY, series_chain.NOTHING_TO_SAVE, series_chain.SAVED_NXCANSAS,
                  series_chain.SAVED_STORE, "Flags", "from the previous frame, Predict where the chain breaks"]
        assert sorted(text for text in texts if text not in ZH) == []
    finally:
        series.dispose()
        single.dispose()
