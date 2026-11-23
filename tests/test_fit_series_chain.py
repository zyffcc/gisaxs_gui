"""The In-situ series page with a chained fit: each frame from the result of the one before, Predict parameters (a
stand-in here, the page's ``predict_solutions``: its method is not this test's) where the chain breaks, a backward
pass, the Started from and Flags columns, and the fits saved as a table with flag columns, as NXcanSAS (measured curves
and models told apart) and into the results database; Fit New Frames in Fitting… ticks Watch and starts."""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

import pytest

from src.gimap.shared.nxcansas import read_nxcansas
from src.gimap.shared.series_store import SeriesStore
from tests.test_fit_series import _model, _open_series, _pages, _series, _wait

RADII = (5.0, 5.2, 5.4, 9.0, 9.2)
"""A sample that changes at once between frame 3 and 4: there a new stage begins (said so here: the stages are not this
test's), and the chain breaks."""


def test_a_chained_series_asks_predict_only_where_the_chain_breaks_and_saves_its_flags(tmp_path, monkeypatch) -> None:
    single, series = _pages()
    try:
        folder = _series(tmp_path, RADII)
        single.open_curve(folder / "run_00001_fit_input.dat")
        # The background held at 0: a value the chained fits keep; the stand-in's solutions have it free (better fits).
        single.set_model(_model(5.2).with_parameter(("globals", "background"), value=0.0, free=False))
        assert _open_series(series, folder)
        monkeypatch.setattr(series, "_new_stage", lambda index: index == 3)
        asked = []

        def solutions(data, start):  # far, closer and probable, close but too unlikely
            asked.append(len(data.q))
            return [SimpleNamespace(model=_model(radius), probability=probability)
                    for radius, probability in ((9.0, 1.0), (8.8, 0.5), (5.5, 0.01))]

        monkeypatch.setattr(series, "predict_solutions", lambda: solutions)
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
        # 8.8 nm, not 5.5 (too unlikely) nor 9.0: the closest to frame 3 (forward) or 5 (the backward pass, kept)
        assert jump.predict_choice.startswith("the closest to frame ")
        assert jump.result.model.get((0, "R")).value == pytest.approx(9.0, rel=0.05)
        assert [fits[index].start_from for index in (1, 2)] in (["previous", "previous"], ["next", "next"])
        assert "Backward pass" in series.log_view.toPlainText()
        flags_column = series.results_table.columnCount() - 3  # (before Points and Halves)
        assert series.results_table.horizontalHeaderItem(flags_column).text() == "Flags"
        started = flags_column - 1
        assert series.results_table.horizontalHeaderItem(started).text() == "Started from"
        assert series.results_table.item(3, started).text() == "restarted from Predict (a new stage begins)"
        assert "the closest to frame " in series.results_table.item(3, started).toolTip()
        assert series.results_table.item(1, started).text() in ("continued from frame 1",
                                                                 "continued from frame 3 (backward pass)")
        assert "chain broke" not in series.results_table.item(3, flags_column).text()  # (said where it started)

        (tmp_path / "out").mkdir()
        table = series.save_table(tmp_path / "out" / "fits.csv")
        with open(table, encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert rows[3]["start_from"] == "predict" and "chain broke" in rows[3]["flags"]
        assert rows[3]["predict_choice"].startswith("the closest to frame ") and rows[2]["start_frame"] in ("2", "4")
        assert {row["fit_pass"] for row in rows} <= {"forward", "backward"}
        assert int(rows[3]["chain_reference_frame"]) == 4
        assert float(rows[3]["chain_reference_chi2"]) == pytest.approx(jump.result.chi2_reduced)

        written = series.save_fits_nxcansas(tmp_path / "out" / "fits.h5")
        curves = read_nxcansas(written)
        assert len(curves) == 10 and curves[0].title.startswith("frame 1 ") and curves[1].title.endswith("the model")
        assert [curve.role for curve in curves[:2]] == ["data", "model"]

        store = series.save_results_database(tmp_path / "out" / "gimap_series.sqlite")
        with SeriesStore(store, read_only=True) as database:
            (name,) = database.series_names()
            stored = database.fits(name, "model")
        assert len(stored) == 5 and stored[3].start_from == "predict"
        assert stored[3].values["1_Sphere_R_nm"] == pytest.approx(9.0, rel=0.05)
        assert any(flag.startswith("chain broke") for flag in stored[3].flags)

        state = series.series_results_state()
        series.restore_series_results(state)
        assert series.fits[3].chain_reference_frame == 4
        assert series.fits[3].chain_reference_chi2 == jump.result.chi2_reduced
        assert "last chain start" in series.selection_summary.text()
        report = series.save_report(tmp_path / "out" / "report.md").read_text(encoding="utf-8")
        assert "Drift also breaks the chain when χ²ᵣ exceeds 2 and 3 times" in report
        assert any("Predict restart kept" in line and line.endswith(f"| 4 | {jump.result.chi2_reduced:.6g} |")
                   for line in report.splitlines())
        for frame in state["frames"]:
            del frame["chain_reference_frame"], frame["chain_reference_chi2"]
        del state["settings"]["chain_drift_recorded"]
        series.restore_series_results(state)
        assert len(series.fits) == len(RADII) and series.fits[3].chain == jump.chain
        assert all(f.chain_reference_frame is None and f.chain_reference_chi2 is None for f in series.fits.values())
        assert "the previous χ²ᵣ" in series.selection_summary.text()
        older = tmp_path / "out" / "older.csv"
        series.save_table(older)
        record = json.loads(older.with_suffix(".json").read_text(encoding="utf-8"))
        assert "chain_reference_ratio_above" not in record["series_policy"]["chain_breaks"]
    finally:
        series.dispose()
        single.dispose()


def test_chain_starts_from_single_after_a_frame_has_no_fit(tmp_path, monkeypatch):
    from src.gimap.features.fitting.application import series_chain
    from src.gimap.features.fitting.application.series_fit import FrameFit

    single, series = _pages()
    try:
        folder = _series(tmp_path, RADII[:3])
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_model(5.2))
        assert _open_series(series, folder)
        fit_frame = series_chain.fit_frame

        def fit(index, path, *args):
            if index == 1:
                return FrameFit(index, path, error="too few points in the fitting range")
            return fit_frame(index, path, *args)

        monkeypatch.setattr(series_chain, "fit_frame", fit)
        series.start_chain.setChecked(True)
        series.skip_odd_check.setChecked(False)
        series.start()
        _wait(series)
        failed, following = series.fits[1], series.fits[2]
        assert failed.chain and failed.chain_reference_frame == 2 and failed.chain_reference_chi2 is None
        assert following.start_from == "single" and following.chain_reference_frame == 3
        assert following.chain_reference_chi2 == following.result.chi2_reduced
    finally:
        series.dispose()
        single.dispose()


def test_fit_new_frames_ticks_watch_and_starts_once_the_model_is_set_up(tmp_path) -> None:
    from src.gimap.features.fitting.presentation.single.series_watch import WATCHING_NOT_SET_UP

    single, series = _pages()
    try:
        folder = _series(tmp_path, RADII[:3])
        series.open_series(folder)
        series.watch_and_start()  # (asked while the curves are still being listed)
        _wait_listed(series)
        assert series.watch_check.isChecked() and not series.running  # the model is still the starting guess
        assert series.status_label.text() == WATCHING_NOT_SET_UP
        single.open_curve(folder / "run_00001_fit_input.dat")
        single.set_model(_model(5.2))
        series.watch_and_start()  # listed already: at once
        assert series.running and series._watch.isActive()
        series.stop()
        _wait(series)
    finally:
        series.dispose()
        single.dispose()


def _wait_listed(series) -> None:
    import time

    from PyQt5.QtWidgets import QApplication

    end = time.monotonic() + 30
    while (series._listing_busy or series._start_when_listed) and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)
    QApplication.processEvents()


def test_the_trend_is_drawn_against_time_when_the_curves_say_when_they_were_measured(tmp_path) -> None:
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
        pair = next(i for i in range(series.trend_combo.count()) if series.trend_combo.itemData(i) == (0, "R+D"))
        series.trend_combo.setCurrentIndex(pair)
        radius, distance = series.trend_plot._source
        assert radius[0].startswith("R · ") and distance[0].startswith("D · ")
        bars = series.trend_plot._items[-2:]
        for key, curve, bar in zip(("R", "D"), (radius, distance), bars):
            assert list(curve[1]) == [0.0, 10.0, 25.0]
            assert list(curve[2]) == [frame.result.model.get((0, key)).value for frame in series.fits.values()]
            _x, bounds = bar.getOriginalDataset()
            errors = [frame.result.errors[(0, key)] for frame in series.fits.values()]
            np.testing.assert_allclose(bounds[0::2], curve[2] - errors)
            np.testing.assert_allclose(bounds[1::2], curve[2] + errors)
        assert len(series.trend_plot._items) == 4  # two estimates and their two sets of vertical bars
        assert series.quick_table_button.isVisible() and series.quick_trend_button.isEnabled()
        csv_path = tmp_path / "trend.csv"
        series.save_table(str(csv_path))
        with csv_path.open(encoding="utf-8") as stream:
            assert [float(row["elapsed_time_s"]) for row in csv.DictReader(stream)] == [0.0, 10.0, 25.0]
        rows = json.loads(csv_path.with_suffix(".json").read_text(encoding="utf-8"))["per_frame"]
        assert [row["elapsed_time_s"] for row in rows] == [0.0, 10.0, 25.0]
        assert rows[0]["acquisition"]["timestamp"].endswith("+0100")
        older = series.series_results_state()
        for row in older["frames"]:
            row.pop("acquisition")
        series.restore_series_results(older)
        assert series.fits[0].acquisition == rows[0]["acquisition"]
        series.trend_combo.setCurrentIndex(pair)
        figure = tmp_path / "trend.svg"
        series._save_plot(series.trend_plot, "trend", str(figure))
        picture = figure.read_text(encoding="utf-8")
        assert "R, D (nm)" in picture and "time (s)" in picture
        assert "R · Refined fit" in picture and "D · Refined fit" in picture
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
        from src.gimap.app import document_session  # noqa: F401 - (the window's session texts load with it)
        from src.gimap.features.analyze.presentation.bindings import project, watch_online
        from src.gimap.features.fitting.application.series_chain import FLAG_KINDS
        from src.gimap.features.fitting.presentation import curve_import
        from src.gimap.features.fitting.presentation.single import (
            page,
            run,
            series_chain,
            series_results,
            series_watch,
            start_texts,
        )

        texts += [*FLAG_KINDS, series_results.FLAGS_TIP, series_chain.BACKWARD_STARTED, series_chain.BACKWARD_KEPT,
                  series_chain.CHAIN_REFINE_ONLY, series_chain.NOTHING_TO_SAVE, series_chain.SAVED_NXCANSAS,
                  series_chain.SAVED_STORE, "Flags", "from the previous frame, Predict where the chain breaks",
                  *start_texts.TEXTS, series_watch.WATCHING_NOT_SET_UP, run.RESTARTED, watch_online.FIT_HANDED,
                  page.CHOOSE_MEASURED, curve_import.NXCANSAS_FITS_NOTE, project.MAP_NOT_KEPT, project.MAP_UNREADABLE]
        assert sorted(text for text in texts if text not in ZH) == []
    finally:
        series.dispose()
        single.dispose()
