"""The In-situ series page of Fitting: every curve of a folder fitted with the model of Single analysis."""

from __future__ import annotations

import csv
import json
import time

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.fitting.application.single_fit import FitModel, evaluate, new_component
from tests.test_assistant_gui import _app

Q_NM = np.linspace(0.08, 2.0, 200)
RADII = (5.0, 5.4, 5.8, 6.2, 6.6)


def _model(radius: float) -> FitModel:
    model = FitModel((new_component("sphere", radius=radius, distance=40.0),))
    return model.with_values({(0, "Int"): 2000.0, (0, "sigma_R"): 0.15, (0, "sigma_D"): 0.3,
                              ("globals", "background"): 3.0})


def _write(path, radius: float, seed: int) -> None:
    exact = evaluate(_model(radius), Q_NM)
    sigma = np.sqrt(exact) + 0.5
    measured = exact + np.random.default_rng(seed).normal(0.0, sigma)
    np.savetxt(path, np.column_stack([Q_NM / 10.0, measured, sigma]))


def _series(tmp_path, radii=RADII):
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    for index, radius in enumerate(radii):
        _write(folder / f"run_{index + 1:05d}_fit_input.dat", radius, index)
    return folder


def _pages():
    from src.gimap.app import AppContext
    from src.gimap.features.fitting.bootstrap import create_fitting_view_model
    from src.gimap.features.fitting.presentation.single.page import FitPage
    from src.gimap.features.fitting.presentation.single.series_page import FitSeriesPage
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )

    _app()
    context = AppContext(settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
                         preferences=InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner(),
                         instrument_profiles=InMemoryInstrumentProfileRepository([]))
    view_model = create_fitting_view_model(context)
    single = FitPage(view_model, preferences=context.preferences)
    series = FitSeriesPage(view_model, single, preferences=context.preferences)
    series.resize(1400, 900)
    series.show()
    return single, series


def _wait(series, limit: float = 120.0) -> None:
    end = time.monotonic() + limit
    while series.running and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def test_a_series_is_fitted_frame_after_frame_from_the_previous_result(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    assert series.step_rail.state("curves") == "pending" and not series.start_button.isEnabled()
    assert series.open_series(folder)
    assert len(series.paths) == 5 and series.frame_list.count() == 5 and series.start_button.isEnabled()
    assert "Sphere" in series.model_summary.text() and "Mean of both halves" in series.model_summary.text()
    series.start()
    assert series.running and series.stop_button.isVisible() and series.pause_button.isVisible()
    _wait(series)
    assert not series.running and len(series.fits) == 5 and all(frame.ok for frame in series.fits.values())
    radii = [series.fits[index].result.model.get((0, "R")).value for index in range(5)]
    assert radii == pytest.approx(RADII, rel=0.05)  # the trend of the series
    assert series.step_rail.state("results") == "ok" and series.results_table.rowCount() == 5
    assert series.trend_combo.currentData() == (0, "R") and series.trend_plot.curve_count() == 2  # R and its ±1σ bars
    assert series.results_table.horizontalHeaderItem(2).text() == "1·Sphere Scale"
    assert series.results_table.horizontalHeaderItem(3).text() == "1·Sphere R (nm)"
    series.trend_plot.log_check.setChecked(not series.trend_plot.log_check.isChecked())
    assert series.trend_plot.curve_count() == 2  # the bars are drawn again after the axis changes
    assert series.frame_list.item(4).text().startswith("✓")
    path = series.save_table(str(tmp_path / "series.csv"))
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows) == 5 and float(rows[4]["1_Sphere_R_nm"]) == pytest.approx(6.6, rel=0.05)
    assert "1_Sphere_R_nm_error" in rows[0]
    record = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert record["frames"] == 5 and record["settings"]["start"] == "previous" and record["failed"] == []
    series.dispose()
    single.dispose()


def test_a_bad_file_does_not_stop_the_series_and_stop_keeps_what_was_done(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    (folder / "run_00003_fit_input.dat").write_text("not a curve\n", encoding="utf-8")
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.start()
    _wait(series)
    assert len(series.fits) == 5 and not series.fits[2].ok and series.fits[3].ok
    assert series.frame_list.item(2).text().startswith("✗") and "1 failed" in series.results_summary.text()
    series.every_spin.setValue(2)  # frames 1, 3, 5
    assert series.frame_list.count() == 3
    series.start()
    series.stop()
    _wait(series)
    assert not series.running and len(series.fits) <= 3
    series.dispose()
    single.dispose()


def test_watch_fits_curves_that_appear_and_a_frame_opens_in_single_analysis(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path, RADII[:2])
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.watch_check.setChecked(True)
    series.start()
    end = time.monotonic() + 60
    while len(series.fits) < 2 and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    _write(folder / "run_00003_fit_input.dat", 5.9, 7)  # written during the run
    series._look_for_new()
    while len(series.fits) < 3 and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    assert len(series.paths) == 3 and len(series.fits) == 3 and series.running  # still watching
    series.stop()
    _wait(series)
    opened = []
    series.openFrameRequested.connect(lambda path, model: opened.append((path, model)))
    series.frame_list.setCurrentRow(2)
    series.to_single_button.click()
    assert opened and opened[0][0].endswith("run_00003_fit_input.dat") and opened[0][1] is not None
    series.dispose()
    single.dispose()
