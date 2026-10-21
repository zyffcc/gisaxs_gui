"""The In-situ series page of Fitting: every curve of a folder fitted with the model of Single analysis."""

from __future__ import annotations

import csv
import json
import time

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.components.toast import visible_toasts
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


def _pages(frames=None):
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
    series = FitSeriesPage(view_model, single, preferences=context.preferences, frames=frames)
    series.resize(1400, 900)
    series.show()
    return single, series


def _wait(series, limit: float = 120.0) -> None:
    end = time.monotonic() + limit
    while series.running and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)


def _answer(monkeypatch, button) -> list:
    """The question about results that were not saved is answered with ``button``; returns what was asked."""
    from PyQt5.QtWidgets import QMessageBox

    asked = []
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda _parent, title, text, *_rest: asked.append((title, text)) or button))
    return asked


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
    assert series.results_table.horizontalHeaderItem(2).text() == "Scale"
    assert series.results_table.horizontalHeaderItem(3).text() == "R (nm)"
    series.trend_plot.log_check.setChecked(not series.trend_plot.log_check.isChecked())
    assert series.trend_plot.curve_count() == 2  # the bars are drawn again after the axis changes
    assert series.frame_list.item(4).text().startswith("✓")
    path = series.save_table(str(tmp_path / "series.csv"))
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows) == 5 and float(rows[4]["1_Sphere_R_nm"]) == pytest.approx(6.6, rel=0.05)
    assert "1_Sphere_R_nm_error" in rows[0]
    assert [row["points"] for row in rows] == ["200"] * 5 and rows[0]["file"] == "run_00001_fit_input.dat"
    assert {row["halves_used"] for row in rows} == {"positive"}  # curves with one side: what there is
    record = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert record["frames"] == 5 and record["settings"]["start"] == "previous" and record["failed"] == []
    assert record["q_unit_of_files"] == "angstrom" and record["settings"]["side"] == "mean"
    assert record["listing"]["first"] == 1 and record["listing"]["last"] == 5 and record["listing"]["frames_to_fit"] == 5
    saved = [toast for toast in visible_toasts(series.window()) if toast.text().startswith("Saved series.csv")]
    assert saved and saved[0].action_button.text() == "Open Folder"
    series.dispose()
    single.dispose()


def test_series_saves_report_errors_and_leave_no_table_without_its_record(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    single, series = _pages()
    folder = _series(tmp_path, RADII[:3])
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.start()
    _wait(series)
    write_text = Path.write_text

    def no_record(self, *args, **kwargs):
        if ".json" in self.name:  # the record (also under its temporary name)
            raise PermissionError(13, "Permission denied", str(self))
        return write_text(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", no_record)
        assert series.save_table(str(tmp_path / "table.csv")) is None  # no exception: no error dialog
    assert series.status_label.text().startswith("Could not save:")
    assert not (tmp_path / "table.csv").exists() and not (tmp_path / "table.json").exists()
    assert not list(tmp_path.glob("table*"))  # nor a temporary one
    assert series._save_plot(series.trend_plot, "trend", str(tmp_path / "missing" / "trend.png")) is None
    assert series.status_label.text().startswith("Could not save:")
    assert series._save_plot(series.frame_plot, "frame", str(tmp_path / "frame.png")) == str(tmp_path / "frame.png")
    assert (tmp_path / "frame.png").stat().st_size > 1000 and series.status_label.text() == "Saved frame.png."
    series.dispose()
    single.dispose()


def test_before_start_the_frames_and_the_trend_choices_are_shown(tmp_path) -> None:
    from pathlib import Path

    from src.gimap.features.fitting.application.single_fit import Curve, prepare_curve
    from src.gimap.features.fitting.presentation.single import series_page
    from src.gimap.features.fitting.presentation.single.series_stages import TREND_EMPTY

    single, series = _pages()
    folder = _series(tmp_path)
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    single.set_range((0.5, 1.5))  # the preview fits what Single analysis would fit
    assert series.open_series(folder)
    assert series.frame_list.currentRow() == 0 and series.frame_plot.curve_count() >= 1
    assert series.frame_plot.figure_state()["title"].startswith("1 · run_00001")
    series.frame_list.setCurrentRow(3)
    state = series.frame_plot.figure_state()
    assert state["title"] == "4 · run_00004_fit_input.dat"
    q, intensity, sigma = np.loadtxt(folder / "run_00004_fit_input.dat", unpack=True)
    expected = prepare_curve(Curve.from_arrays(q, intensity, sigma, name="run_00004_fit_input.dat"), "mean", (0.5, 1.5))
    name, x, y = state["curves"][0]
    assert np.allclose(x, expected.q) and np.allclose(y, expected.intensity) and x.min() >= 0.5 and x.max() <= 1.5
    assert series.trend_combo.count() > 1 and series.trend_combo.currentData() == (0, "R")
    assert series.trend_plot.figure_state()["title"] == "" and series.trend_plot.curve_count() == 0
    assert series.trend_plot.empty_overlay.label.text() == TREND_EMPTY  # said once, over the empty plot
    end = time.monotonic() + 60
    while series._series_stages is None and time.monotonic() < end:  # the stage search keeps the curves it read
        series.stage_tasks.wait(0.1)
        QApplication.processEvents()
    assert len(series._preview_curves) == 5 and series.frame_list.currentRow() == 3  # still selected
    assert len(Path(series_page.__file__).read_text(encoding="utf-8").splitlines()) <= 600
    series.dispose()
    single.dispose()


def test_after_start_a_frame_not_fitted_yet_is_drawn_on_the_points_of_the_run(tmp_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    _answer(monkeypatch, QMessageBox.Discard)  # the frame fitted before the stop is not saved
    single, series = _pages()
    folder = _series(tmp_path)
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    single.set_range((0.5, 1.5))
    series.open_series(folder)
    series.start()
    series.stop()  # stopped early: the frames after the first are not fitted
    _wait(series)
    assert 3 not in series.fits and series._settings.q_range == (0.5, 1.5)
    single.set_range((0.2, 1.9))  # Single analysis moves on
    series.frame_list.setCurrentRow(3)
    name, x, _y = series.frame_plot.figure_state()["curves"][0]
    assert x.min() >= 0.5 and x.max() <= 1.5  # the points the run fits, not Single's new range
    series.open_series(folder)  # listed again: before Start, the frames follow Single analysis
    series.frame_list.setCurrentRow(3)
    name, x, _y = series.frame_plot.figure_state()["curves"][0]
    assert x.min() < 0.5 and x.max() > 1.5
    series.dispose()
    single.dispose()


def test_the_curves_step_fits_its_panel_before_a_folder_is_chosen() -> None:
    single, series = _pages()
    assert series.every_spin.prefix() == "" and series.every_spin.maximum() == 1  # set once the curves are listed
    series.splitter.setSizes([340, 1060])  # the narrowest steps panel
    for _ in range(3):
        QApplication.processEvents()
    scroll = series.step_pages["curves"]
    assert scroll.widget().minimumSizeHint().width() <= scroll.viewport().width()
    series.dispose()
    single.dispose()


def test_stages_and_odd_frames_steer_the_run(tmp_path) -> None:
    from src.gimap.features.fitting.application.series_fit import SeriesSettings, next_start

    radii = [5.0 + 0.1 * i for i in range(10)] + [6.0] * 10  # grows, then holds: two stages
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    for index, radius in enumerate(radii):
        exact = evaluate(_model(radius), Q_NM)
        if index == 5:
            exact = exact * np.exp(-Q_NM)  # frame 6: a curve unlike its neighbours
        sigma = 0.01 * exact
        measured = exact + np.random.default_rng(index).normal(0.0, sigma)
        np.savetxt(folder / f"run_{index + 1:05d}_fit_input.dat", np.column_stack([Q_NM / 10.0, measured, sigma]))
    single, series = _pages()
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.0))
    series.open_series(folder)
    end = time.monotonic() + 60
    while series._series_stages is None and time.monotonic() < end:
        series.stage_tasks.wait(0.1)
        QApplication.processEvents()
    stages = series._series_stages
    assert stages is not None and [frame.row for frame in stages.odd] == [5]
    assert stages.count >= 2 and abs(stages.boundaries[stages.count][0] - 10) <= 2
    assert series.stages_label.text().startswith(f"{stages.count} stages: frames 1–") and "odd frames: 6" in series.stages_label.text()
    assert not series.skip_odd_check.isHidden() and series.skip_odd_check.isChecked()
    assert series.frame_list.item(5).text().endswith("odd") and series.stage_of_frame(15) == stages.count
    series.start_stages.setChecked(True)
    series.start()
    _wait(series)
    assert 5 not in series.fits and len(series.fits) == 19  # the odd frame was left out
    assert series._settings.start == "stages"
    names = [curve[0] for curve in series.trend_plot.figure_state()["curves"]]
    assert any(name.endswith("stage 1") for name in names) and any(name.endswith(f"stage {stages.count}") for name in names)
    # A new stage starts from the model of Single analysis, within a stage from the previous result.
    previous = series.fits[9]
    settings = SeriesSettings(start="stages")
    assert next_start(previous, single.session.model, settings, new_stage=True) is single.session.model
    assert next_start(previous, single.session.model, settings, new_stage=False) is previous.result.model
    series.dispose()
    single.dispose()


def test_stages_are_found_after_a_run_that_started_while_they_were_looked_for(tmp_path, monkeypatch) -> None:
    """Start pressed while the search still reads the curves: its result is not taken during the run, and
    nothing looked again after it — the Curves step said “Looking for stages and odd frames …” for good."""
    import threading

    radii = [5.0 + 0.1 * i for i in range(8)] + [6.0] * 8  # grows, then holds: two stages
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    for index, radius in enumerate(radii):
        exact = evaluate(_model(radius), Q_NM)
        measured = exact + np.random.default_rng(index).normal(0.0, 0.01 * exact)
        np.savetxt(folder / f"run_{index + 1:05d}_fit_input.dat", np.column_stack([Q_NM / 10.0, measured, 0.01 * exact]))
    single, series = _pages()
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.0))
    load, release = series.view_model.load_curve, threading.Event()

    def held(request):  # the search (in the background) reads a long series: it is held until the run is on
        if threading.current_thread() is not threading.main_thread():
            release.wait(30)
        return load(request)

    monkeypatch.setattr(series.view_model, "load_curve", held)
    try:
        series.open_series(folder)
        series.start_stages.setChecked(True)  # “afresh at each stage”
        end = time.monotonic() + 30
        while not series.stage_tasks.is_busy() and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
        assert series.stage_tasks.is_busy() and series.stages_label.text() == "Looking for stages and odd frames …"
        series.watch_check.setChecked(True)  # the run stays on until it is stopped
        series.start()
        assert series.running and series._series_stages is None
        said = "No stages are known: every frame starts from the previous frame's result."
        assert said in series.log_view.toPlainText()  # “afresh at each stage” cannot be kept: the run says so
        release.set()
        while (series.stage_tasks.is_busy() or not series._waiting_for_curves()) and time.monotonic() < end:
            series.stage_tasks.wait(0.05)
            QApplication.processEvents()
        QApplication.processEvents()
        assert series.running and series._series_stages is None  # the run keeps what it started with: no stages
        assert series.stages_label.text() == "No stages for this run: it started before they were found."
        series.stop()
        _wait(series)
        end = time.monotonic() + 30
        QApplication.processEvents()
        while (series._stages_timer.isActive() or series.stage_tasks.is_busy()) and time.monotonic() < end:
            series.stage_tasks.wait(0.05)
            QApplication.processEvents()
        QApplication.processEvents()
        stages = series._series_stages
        assert stages is not None and stages.count >= 2 and series.stages_label.text().startswith(f"{stages.count} stages: frames 1–")
        assert series.stage_of_frame(0) == 1 and series.stage_of_frame(15) == stages.count
        names = [curve[0] for curve in series.trend_plot.figure_state()["curves"]]  # the trend: in their colours now
        assert any(name.endswith("stage 1") for name in names) and any(name.endswith(f"stage {stages.count}") for name in names)
        assert series._run_listing["stages"] is None and series._settings.start == "stages"  # the record: none steered the run
    finally:  # (a failure above must not leave the held search and a running series behind)
        release.set()
        series.stop()
        _wait(series, 30)
        series.dispose()
        single.dispose()


def test_a_bad_file_does_not_stop_the_series_and_stop_keeps_what_was_done(tmp_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    _answer(monkeypatch, QMessageBox.Discard)  # the second Start, with the first run not saved
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
    assert len(series.paths) == 2  # seen once: it may still be written
    series._look_for_new()  # the same size and time at the next look: taken
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


# -- the halves of a series, and what every frame was fitted on --------------------------------------

def _cut(radius: float, seed: int, *, only=None, negative: float = 1.0, noise: float = 1.0, odd: bool = False):
    """One frame of a series of cuts through the beam: signed q (Å⁻¹), I and σ; the q < 0 half ``negative``
    times as intense; ``only``: the one half the frame holds."""
    exact = evaluate(_model(radius), Q_NM) * (np.exp(-Q_NM) if odd else 1.0)
    rng = np.random.default_rng(seed)
    halves = {}
    for name, factor in (("negative", negative), ("positive", 1.0)):
        sigma = noise * (np.sqrt(factor * exact) + 0.5)
        halves[name] = (Q_NM / 10.0, factor * exact + rng.normal(0.0, sigma), sigma)
    q, intensity, sigma = halves["negative"]
    left = (-q[::-1], intensity[::-1], sigma[::-1])
    parts = [part for name, part in (("negative", left), ("positive", halves["positive"])) if only in (None, name)]
    return tuple(np.concatenate(column) for column in zip(*parts))


def _curve_of(cut, name="frame.dat"):
    from src.gimap.features.fitting.application.single_fit import Curve

    return Curve.from_arrays(*cut, name=name)


def test_a_series_is_fitted_on_the_mean_or_one_half_never_on_both_together() -> None:
    from src.gimap.features.fitting.application.series_fit import SeriesSettings, fit_frame, prepare_frame

    assert SeriesSettings().side == "mean"
    assert [SeriesSettings(side=side).side for side in ("mean", "positive", "negative")] == ["mean", "positive", "negative"]
    assert SeriesSettings(side="both").side == SeriesSettings(side="both_abs").side == SeriesSettings(side=None).side == "mean"
    curve = _curve_of(_cut(5.4, 1, negative=1.2))
    assert prepare_frame(curve, SeriesSettings(side="both")).points == 200  # one curve: not 400 points on |q|
    start = _model(5.2)
    for side, used, averaged, level in (("mean", "mean", 200, 1.1), ("positive", "positive", 0, 1.0),
                                        ("negative", "negative", 0, 1.2)):
        frame = fit_frame(0, "frame.dat", curve, start, SeriesSettings(side=side))
        assert frame.ok and frame.state == "ok" and frame.reason == ""
        assert (frame.points, frame.halves_used, frame.averaged, frame.left_out) == (200, used, averaged, 0)
        assert frame.result.points == 200 and frame.result.model.get((0, "Int")).value == pytest.approx(2000 * level, rel=0.1)
    ranged = fit_frame(0, "frame.dat", curve, start, SeriesSettings(q_range=(0.5, 1.5), excluded=frozenset({float(f"{Q_NM[100]:.9g}")})))
    assert ranged.left_out == 1 and ranged.points == int(((Q_NM >= 0.5) & (Q_NM <= 1.5)).sum()) - 1


def test_a_frame_without_the_chosen_half_fails_and_is_not_fitted_on_the_other() -> None:
    from dataclasses import replace

    from src.gimap.features.fitting.application.series_fit import (
        SeriesSettings,
        fit_frame,
        next_start,
        parameter_columns,
        series_counts,
        series_record,
        series_table,
    )

    curves = [_curve_of(_cut(radius, seed, only="negative" if seed == 2 else None, negative=1.3))
              for seed, radius in enumerate(RADII)]  # the third frame has lost its q > 0 half
    start = _model(5.2)
    settings = SeriesSettings(side="positive")
    frames = [fit_frame(index, f"run_{index + 1:05d}.dat", curve, start, settings) for index, curve in enumerate(curves)]
    lost = frames[2]
    assert not lost.ok and lost.error == lost.reason == "No points on the q > 0 half" and lost.state == "failed"
    assert (lost.points, lost.halves_used) == (0, "") and all(frame.ok for index, frame in enumerate(frames) if index != 2)
    assert [frame.halves_used for frame in frames] == ["positive", "positive", "", "positive", "positive"]
    assert next_start(lost, start, settings) is start  # the next frame does not start from a frame without a fit
    assert series_counts(frames) == {"frames": 5, "ok": 4, "not_converged": 0, "stopped": 0, "failed": 1}
    # The mean of a frame with one half is that half — fitted, and named: the q < 0 half is 1.3 times as intense.
    mean = [fit_frame(index, f"run_{index + 1:05d}.dat", curve, start, SeriesSettings()) for index, curve in enumerate(curves)]
    assert [frame.halves_used for frame in mean] == ["mean", "mean", "negative", "mean", "mean"]
    assert [frame.averaged for frame in mean] == [200, 200, 0, 200, 200] and all(frame.ok for frame in mean)
    scale = [frame.result.model.get((0, "Int")).value for frame in mean]
    assert scale[2] == pytest.approx(2600, rel=0.1) and scale[1] == pytest.approx(2300, rel=0.1)

    # The table as the page fills its own from it, and with what each frame was fitted on for the file that is saved.
    names = [name for _path, name in parameter_columns(start)]
    header, rows = series_table(frames, start)
    assert header[:5] == ["frame", "file", "chi2_reduced", "log_rmse", "converged"] and header[5::2] == names
    header, rows = series_table(frames, start, details=True)
    assert header[:7] == ["frame", "file", "chi2_reduced", "log_rmse", "converged", "points", "halves_used"]
    assert header[7::2] == names and all(len(row) == len(header) for row in rows)
    below = series_table([replace(frames[0], path="/data/run/a/fit_input/f_1.dat")], start, folder="/data/run")
    assert below[1][0][1] == "a/fit_input/f_1.dat"  # frames of several subfolders stay apart
    table = [dict(zip(header, row)) for row in rows]
    assert [(row["points"], row["halves_used"]) for row in table] == [(200, "positive")] * 2 + [(0, "")] + [(200, "positive")] * 2
    assert np.isnan(table[2]["chi2_reduced"]) and table[3]["1_Sphere_R_nm"] == pytest.approx(RADII[3], rel=0.05)
    record = series_record(start, settings, "folder", "*.dat", frames, q_unit="angstrom", saved="2026-01-02T03:04:05+01:00")
    assert record["schema"] == "gimap_series_fit_v2" and record["software"].startswith("GIMaP v")
    assert record["saved"] == "2026-01-02T03:04:05+01:00" and record["q_unit_of_files"] == "angstrom"
    assert record["settings"]["side"] == "positive" and record["frames"] == 5 and record["counts"]["failed"] == 1
    assert record["failed"] == [{"frame": 3, "file": "run_00003.dat", "why": "No points on the q > 0 half"}]
    assert record["per_frame"][2] == {"frame": 3, "file": "run_00003.dat", "state": "failed", "reason": "No points on the q > 0 half",
                                      "points": 0, "halves_used": "", "averaged": 0, "left_out": 0, "weighting": ""}
    assert record["per_frame"][0] == {"frame": 1, "file": "run_00001.dat", "state": "ok", "reason": "", "points": 200,
                                      "halves_used": "positive", "averaged": 0, "left_out": 0, "weighting": "sigma"}
    assert json.loads(json.dumps(record)) == record
    from datetime import datetime

    assert datetime.fromisoformat(series_record(start, settings, "folder", "*.dat", frames)["saved"]).year >= 2026


def test_the_halves_of_a_series_come_from_what_there_is_to_go_by() -> None:
    """The person's choice for the series, else what the fit input was sent with, else Single analysis' — and a
    series of curves with one side has no halves: a half left from other curves would fail every frame."""
    from src.gimap.features.fitting.application.series_fit import SeriesSettings, fit_frame, series_side

    assert series_side() == "mean" and series_side("negative", "positive", "mean") == "negative"
    assert series_side(None, "positive", "negative") == "positive" and series_side(None, None, "negative") == "negative"
    assert series_side(None, "both_abs", "both") == "mean" and series_side("both", None, "positive") == "positive"
    assert series_side("negative", "negative", "negative", signed=False) == "mean"
    # Curves with points on one side only (an I(q), a cut of one half), and a half left over from Single analysis:
    one_sided = [_curve_of(_cut(radius, seed, only="positive")) for seed, radius in enumerate(RADII)]
    assert not any(curve.signed for curve in one_sided)
    start = _model(5.2)
    left_over = [fit_frame(index, "frame.dat", curve, start, SeriesSettings(side="negative")) for index, curve in enumerate(one_sided)]
    assert [frame.reason for frame in left_over] == ["No points on the q < 0 half"] * 5  # as a chosen half must fail
    settings = SeriesSettings(side=series_side(None, None, "negative", signed=any(curve.signed for curve in one_sided)))
    frames = [fit_frame(index, "frame.dat", curve, start, settings) for index, curve in enumerate(one_sided)]
    assert settings.side == "mean" and all(frame.ok for frame in frames)
    assert [(frame.points, frame.halves_used) for frame in frames] == [(200, "positive")] * 5


def test_a_stopped_frame_is_not_a_result_and_is_counted_apart() -> None:
    from src.gimap.features.fitting.application.series_fit import (
        STATES,
        SeriesSettings,
        fit_frame,
        next_start,
        series_counts,
        series_record,
    )

    curve = _curve_of(_cut(5.8, 3))
    start, settings = _model(5.0), SeriesSettings(method="global")
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 5

    stopped = fit_frame(1, "b.dat", curve, start, settings, stop)
    done = fit_frame(0, "a.dat", curve, start, SeriesSettings())
    few = fit_frame(2, "c.dat", curve, start, SeriesSettings(q_range=(0.5, 0.51)))
    assert STATES == ("ok", "not converged", "stopped", "failed")
    assert (done.state, stopped.state, few.state) == ("ok", "stopped", "failed")
    assert stopped.ok and "Stopped" in stopped.reason and few.reason == "too few points in the fitting range" and few.points < 3
    assert next_start(stopped, start, settings) is start and next_start(done, start, settings) is done.result.model
    frames = [stopped, done, few]
    assert series_counts(frames) == {"frames": 3, "ok": 1, "not_converged": 0, "stopped": 1, "failed": 1}
    record = series_record(start, settings, "folder", "*.dat", frames)
    assert record["stopped"] == [2] and record["not_converged"] == [] and [item["state"] for item in record["per_frame"]] == [
        "ok", "stopped", "failed"]


def test_a_curve_that_cannot_be_used_does_not_end_the_search_for_stages() -> None:
    from src.gimap.features.fitting.application.series_fit import SeriesSettings, stages_of_curves
    from src.gimap.shared.series_stages import SeriesStages

    radii = [5.0 + 0.1 * i for i in range(10)] + [6.0] * 10  # grows, then holds: two stages
    cuts = [_cut(radius, index, noise=0.02, odd=index == 15, only="negative" if index == 3 else None)
            for index, radius in enumerate(radii)]  # frame 16 is unlike its neighbours; frame 4 has lost its q > 0 half
    curves = [_curve_of(cut, f"run_{index + 1:05d}.dat") for index, cut in enumerate(cuts)]
    settings = SeriesSettings(side="positive")
    good = [curve for index, curve in enumerate(curves) if index not in (3, 12)]
    alone = stages_of_curves(good, settings)  # what the 18 curves that can be used say
    assert isinstance(alone, SeriesStages) and alone.skipped == () and alone.rows == 18
    assert [frame.row for frame in alone.odd] == [13] and alone.count >= 2

    listed = list(curves)
    listed[12] = None  # this file could not be read
    stages = stages_of_curves(listed, settings, unreadable={12: "No (q, I[, σ]) rows could be read."})
    assert stages.skipped == ((3, "No points on the q > 0 half"), (12, "No (q, I[, σ]) rows could be read."))
    # Every row is the place of the curve in the list it was given, the skipped ones included.
    assert stages.rows == 20 and [frame.row for frame in stages.odd] == [15] and stages.count == alone.count
    place = [index for index in range(20) if index not in (3, 12)]
    assert stages.boundaries == {count: tuple(place[row] for row in rows) for count, rows in alone.boundaries.items()}
    assert abs(stages.boundaries[stages.count][0] - 10) <= 2 and stages.ranges()[0][0] == 0 and stages.ranges()[-1][1] == 19
    assert stages.stage_of(3) == 1 and stages.stage_of(12) == stages.stage_of(13) and stages.odd_rows == frozenset({15})
    assert np.isnan(stages.level[[3, 12]]).all() and np.allclose(np.delete(stages.level, [3, 12]), alone.level)
    assert stages.scores.shape == (20, alone.scores.shape[1]) and np.isnan(stages.scores[3]).all()
    assert stages.with_count(1).skipped == stages.skipped and stages.with_count(1).count == 1
    assert stages_of_curves(listed, settings).skipped[1] == (12, "the curve could not be read")
    # On the mean nothing but the unreadable file is left out: the frame with one half is that half.
    assert stages_of_curves(listed, SeriesSettings()).skipped == ((12, "the curve could not be read"),)
    with pytest.raises(ValueError, match="No points on the q < 0 half"):  # no frame can be used: said with the reason
        stages_of_curves([_curve_of(_cut(5.0, seed, only="positive")) for seed in range(6)], SeriesSettings(side="negative"))


def test_in_the_page_a_frame_without_the_chosen_half_fails_with_its_reason(tmp_path) -> None:
    single, series = _pages()
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    for index, radius in enumerate(RADII):
        np.savetxt(folder / f"run_{index + 1:05d}_fit_input.dat",
                   np.column_stack(_cut(radius, index, only="negative" if index == 2 else None, negative=1.3)))
    single.open_curve(folder / "run_00001_fit_input.dat", "positive")
    single.set_model(_model(5.2))
    assert series.open_series(folder)
    series.start()
    _wait(series)
    assert series._settings.side == "positive" and len(series.fits) == 5 and not series.running
    assert [series.fits[index].ok for index in range(5)] == [True, True, False, True, True]
    assert series.fits[2].error == "No points on the q > 0 half" and series.fits[3].halves_used == "positive"
    assert series.frame_list.item(2).text().startswith("✗") and series.frame_list.item(2).text().endswith("No points on the q > 0 half")
    assert [series.fits[index].result.model.get((0, "R")).value for index in (0, 1, 3, 4)] == pytest.approx(
        [RADII[index] for index in (0, 1, 3, 4)], rel=0.05)
    path = series.save_table(str(tmp_path / "series.csv"))
    record = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert path and record["failed"][0]["why"] == "No points on the q > 0 half" and record["settings"]["side"] == "positive"
    assert [item["state"] for item in record["per_frame"]] == ["ok", "ok", "failed", "ok", "ok"]
    # Both halves on |q|, chosen in Single analysis, is not what a series fits: every frame on the mean.
    single.set_side("both")
    series.start()
    _wait(series)
    assert series._settings.side == "mean" and all(series.fits[index].ok for index in range(5))
    assert [series.fits[index].halves_used for index in range(5)] == ["mean", "mean", "negative", "mean", "mean"]
    assert series.fits[0].points == 200 and series.fits[0].averaged == 200
    series.dispose()
    single.dispose()


def test_the_sentences_about_the_halves_of_a_series_have_their_chinese() -> None:
    from src.gimap.app.presentation.i18n import ZH
    from src.gimap.features.fitting.application.series_fit import TOO_FEW_POINTS, UNREADABLE
    from src.gimap.features.fitting.application.single_fit import BOTH_USES_MEAN, SERIES_SIDES
    from src.gimap.shared.signed_cut import MISSING_HALF

    for text in (*MISSING_HALF.values(), TOO_FEW_POINTS, UNREADABLE, BOTH_USES_MEAN, "No curves to find stages in.",
                 *(text for _key, text in SERIES_SIDES)):
        assert ZH.get(text) not in (None, text), text
    assert ZH["No points on the q > 0 half"] != ZH["No points on the q < 0 half"]
    # What the note of a prepared cut says of a half, without a value in it: entries as they are written.
    for side in ("qy > 0", "qy < 0"):
        for text in (f"the {side} half has no points", f"the {side} half is too sparse to average with"):
            assert side in ZH[text] and ZH[text] != text, text


# -- a series of curves with one side (from the signed-cut package: the page takes the halves from series_side) ----

def test_a_series_of_one_sided_curves_is_fitted_whatever_half_single_analysis_was_left_on(tmp_path) -> None:
    """Analyze's Halves is "qy < 0 half" and an I(q) with one side is sent to Fitting: Single analysis shows no
    choice of halves for it (and fits its 200 points); the series of such curves must fit every frame."""
    single, series = _pages()
    folder = _series(tmp_path)  # five curves, q > 0 only
    single.open_curve(folder / "run_00001_fit_input.dat", "negative")  # as main_window sends it: side=fit_side
    single.set_model(_model(5.2))
    assert single.side_combo.isHidden() and single._data().q.size == 200
    assert series.open_series(folder)
    series.start()
    _wait(series)
    assert [series.fits[index].ok for index in range(5)] == [True] * 5, sorted({fit.error for fit in series.fits.values()})
    assert series._settings.side == "mean" and {fit.halves_used for fit in series.fits.values()} == {"positive"}
    series.dispose()
    single.dispose()


def test_a_half_chosen_for_a_cut_in_single_analysis_does_not_fail_a_series_of_one_sided_curves(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    cut = tmp_path / "cut_fit_input.dat"
    np.savetxt(cut, np.column_stack(_cut(RADII[0], 0, negative=1.3)))
    single.open_curve(cut, "negative")  # a signed cut on its q < 0 half
    single.set_model(_model(5.2))
    assert not single.side_combo.isHidden() and single.session.side == "negative"
    assert series.open_series(folder)
    series.start()
    _wait(series)
    assert [series.fits[index].ok for index in range(5)] == [True] * 5, sorted({fit.error for fit in series.fits.values()})
    series.dispose()
    single.dispose()


def test_a_first_frame_that_lost_a_half_does_not_make_the_series_one_without_halves(tmp_path) -> None:
    """Before the stage search has read the curves only the first frame is known, and it has one half. The
    series was sent with “q < 0 half”: a look at a few more curves, when they are listed, finds the cuts."""
    import json

    single, series = _pages()
    folder = tmp_path / "gimap_analysis" / "fit_input"
    folder.mkdir(parents=True)
    for index, radius in enumerate(RADII):
        cut = _cut(radius, index, only="negative" if index == 0 else None, negative=1.3)
        lines = ["# GIMaP Analyze fit input (q in 1/A)", '# observation: ' + json.dumps({"halves": "negative"})]
        lines += [f"{q:.8g} {intensity:.8g} {sigma:.8g} 5" for q, intensity, sigma in zip(*cut)]
        (folder / f"cut_{index + 1:05d}_fit_input.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    single.set_model(_model(5.2))
    assert series.open_series(folder) and series.curves_signed()  # a look at a few more: these are cuts
    assert series.series_side() == "negative" and series.halves_source() == "files"  # … said before Start
    series.start()
    assert series._settings.side == "negative" and series._side_from == "files"
    _wait(series)
    assert all(frame.ok and frame.halves_used == "negative" for frame in series.fits.values())
    series.dispose()
    single.dispose()


# -- Watch ---------------------------------------------------------------------------------------------

def test_watch_loses_no_frame_when_the_timer_and_a_finished_fit_coincide(tmp_path) -> None:
    """A finished fit queued the next one; the watch's look in between started one too, and the runner dropped
    the first one's result: the frame stayed “being fitted” for ever."""
    single, series = _pages()
    folder = _series(tmp_path, RADII[:3])
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.watch_check.setChecked(True)
    done, written = series._frame_done, []

    def racing(frame, curve=None):
        done(frame, curve)
        if len(written) < 4:  # a curve arrives, and the watch looks, between a finished fit and the queued next one
            written.append(1)
            _write(folder / f"run_{len(written) + 3:05d}_fit_input.dat", 5.9, 7)
            series._look_for_new()
            series._look_for_new()

    series._frame_done = racing
    series.start()
    end = time.monotonic() + 60
    while time.monotonic() < end and (len(series.fits) < 7 or series.tasks.is_busy()):
        QApplication.processEvents()
        time.sleep(0.01)
    assert len(series.paths) == 7 and sorted(series.fits) == list(range(7)) and all(f.ok for f in series.fits.values())
    assert series.frame_list.count() == 7 and all(series.frame_list.item(row).text().startswith("✓") for row in range(7))
    QApplication.processEvents()  # the run goes on from the last fit: nothing is left, it says what it waits for
    assert series.running and series._waiting_for_curves()
    assert series.status_label.text() == f"Waiting for new curves in {tmp_path.name} … (Stop to end)"  # the folder that holds gimap_analysis
    assert series.step_rail.state("results") == "busy" and series.step_rail.detail("results") == "7 frames · waiting for new curves"
    series.stop()
    _wait(series)
    assert series.status_label.text() == "Series stopped: 7 frames fitted, 0 failed."
    series.dispose()
    single.dispose()


def test_watch_reads_a_curve_again_that_was_still_being_written(tmp_path) -> None:
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
    whole = folder / "whole.txt"
    _write(whole, 5.9, 7)
    rows = whole.read_text(encoding="utf-8").splitlines()
    whole.unlink()
    target = folder / "run_00003_fit_input.dat"
    with open(target, "w", encoding="utf-8") as stream:  # the writer is in the middle of the file
        stream.write("\n".join(rows[:60]) + "\n")
        stream.flush()
        series._look_for_new()
        series._look_for_new()  # … and pauses there for a moment: the file looks settled, and is read
        while len(series.fits) < 3 and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
        early = series.fits[2]
        assert early.points == 60
        stream.write("\n".join(rows[60:]) + "\n")
    series._look_for_new()  # changed since it was read; it may still change
    assert series.fits[2] is early and 2 not in series.queue
    series._look_for_new()  # settled: read and fitted again
    while (series.fits[2] is early or series.tasks.is_busy()) and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    assert series.fits[2].points == 200 and series.fits[2].ok and series.results_table.rowCount() == 3
    assert series.results_table.item(2, series.results_table.columnCount() - 2).text() == "200"  # its row made again
    assert "fitted again" in series.log_view.toPlainText()
    series._look_for_new()
    series._look_for_new()
    assert series.queue == [] and series.running  # nothing changed: nothing is fitted twice
    # A new file is not taken at the first look, nor an empty one at any.
    (folder / "run_00004_fit_input.dat").write_text("", encoding="utf-8")
    series._look_for_new()
    series._look_for_new()
    assert len(series.paths) == 3
    series.stop()
    _wait(series)
    series.dispose()
    single.dispose()


def test_the_watch_box_works_when_ticked_or_unticked_during_a_run(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path, RADII[:2])
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    assert "Can be switched while the series runs" in series.watch_check.toolTip()
    series.start()
    assert series.running and not series._watch.isActive()
    series.watch_check.setChecked(True)  # ticked while the series runs: from now on
    assert series._watch.isActive()
    end = time.monotonic() + 60
    while not series._waiting_for_curves() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    assert series.running and len(series.fits) == 2  # it stays, for the curves to come
    assert series.status_label.text().startswith("Waiting for new curves in")
    _write(folder / "run_00003_fit_input.dat", 5.9, 7)
    series._look_for_new()
    series._look_for_new()
    while len(series.fits) < 3 and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    assert len(series.fits) == 3
    while not series._waiting_for_curves() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    series.watch_check.setChecked(False)  # unticked while it only waits: the run is over
    assert not series.running and not series._watch.isActive()
    assert series.status_label.text() == "Series done: 3 frames fitted, 0 failed."
    series.dispose()
    single.dispose()


def test_watch_keeps_to_the_frames_that_were_chosen(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path, RADII[:4])
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.every_spin.setValue(2)  # frames 1 and 3
    series.watch_check.setChecked(True)
    series.start()
    end = time.monotonic() + 60
    while not series._waiting_for_curves() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    for number in (5, 6):
        _write(folder / f"run_{number:05d}_fit_input.dat", 6.0, number)
    series._look_for_new()
    series._look_for_new()
    while (len(series.fits) < 3 or series.tasks.is_busy()) and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    assert len(series.paths) == 6 and sorted(series.fits) == [0, 2, 4]  # every second frame, as chosen
    assert series._listed() == [0, 2, 4] and series.frame_list.count() == 3  # a line per listed frame, in its place
    assert [series.frame_list.item(row).text().split()[1] for row in range(3)] == ["1", "3", "5"]
    series.stop()
    _wait(series)
    listing = series._run_listing  # the record: every second frame up to the fifth, of six curves
    assert (listing["first"], listing["last"], listing["every"]) == (1, 5, 2)
    assert listing["curves_in_folder"] == 6 and listing["frames_to_fit"] == 3 and listing["added_by_watch"] == 1
    series.dispose()
    single.dispose()


def test_the_record_of_a_run_that_watch_extended_says_what_was_fitted(tmp_path) -> None:
    """``listing`` of the record was made at Start and written as it was: three curves, for a run of five."""
    single, series = _pages()
    folder = _series(tmp_path, RADII[:3])
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.start()
    series.watch_check.setChecked(True)  # ticked during the run: the record says that new curves were watched for
    end = time.monotonic() + 60
    while not series._waiting_for_curves() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    for number in (4, 5):
        _write(folder / f"run_{number:05d}_fit_input.dat", 6.0, number)
    series._look_for_new()
    series._look_for_new()
    while (len(series.fits) < 5 or series.tasks.is_busy()) and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    series.stop()
    _wait(series)
    assert series.save_table(str(tmp_path / "watched.csv"))
    record = json.loads((tmp_path / "watched.json").read_text(encoding="utf-8"))
    listing = record["listing"]
    assert record["frames"] == 5 and record["counts"]["ok"] == 5
    assert (listing["first"], listing["last"], listing["every"]) == (1, 5, 1)
    assert listing["curves_in_folder"] == 5 and listing["frames_to_fit"] == 5
    assert listing["watch"] is True and listing["added_by_watch"] == 2
    series.dispose()
    single.dispose()


def test_watch_says_when_the_folder_cannot_be_read_and_goes_on_when_it_is_back(tmp_path) -> None:
    """A share that drops for a while: the page does not go on saying that it waits for new curves in it."""
    single, series = _pages()
    folder = _series(tmp_path, RADII[:2])
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.watch_check.setChecked(True)
    series.start()
    end = time.monotonic() + 60
    while not series._waiting_for_curves() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    waiting = f"Waiting for new curves in {tmp_path.name} … (Stop to end)"
    assert series.status_label.text() == waiting
    away = tmp_path / "away"
    folder.rename(away)
    series._look_for_new()
    assert series.status_label.text() == "The folder of the curves cannot be read: waiting for it … (Stop to end)"
    assert series.running and len(series.fits) == 2  # nothing is lost, the run stays
    away.rename(folder)
    series._look_for_new()
    assert series.status_label.text() == waiting
    series._look_for_new()
    assert series.queue == [] and len(series.paths) == 2  # the same curves: nothing is fitted again
    series.stop()
    _wait(series)
    series.dispose()
    single.dispose()


@pytest.mark.parametrize("language", ["en", "zh"])
def test_a_curve_that_is_gone_when_its_turn_comes_says_so_in_the_interface_language(tmp_path, language) -> None:
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, ZH, apply_language
    from src.gimap.features.fitting.application.models import CURVE_NOT_FOUND

    single, series = _pages()
    folder = _series(tmp_path, RADII[:3])
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    try:
        apply_language(language, [series, single])
        series.refresh_language()
        series.open_series(folder)
        gone = folder / "run_00002_fit_input.dat"
        path = series.paths[1]
        gone.unlink()  # deleted after the curves were listed
        series.start()
        _wait(series)
        assert [series.fits[index].ok for index in range(3)] == [True, False, True]
        assert series.fits[1].error == CURVE_NOT_FOUND.format(path=path)  # the record: as the reader said it
        said = (CURVE_NOT_FOUND if language == "en" else ZH[CURVE_NOT_FOUND]).format(path=path)
        assert ZH[CURVE_NOT_FOUND] != CURVE_NOT_FOUND and path in said
        assert series.results_table.item(1, 1).text() == f"✗ {said}"
        assert series.frame_list.item(1).text().endswith(said) and said in series.log_view.toPlainText()
        series.frame_list.setCurrentRow(1)
        assert said in series.frame_note.text()
    finally:
        apply_language(DEFAULT_LANGUAGE, [series, single])
        series.dispose()
        single.dispose()


# -- listing the curves ----------------------------------------------------------------------------------

def test_the_hint_names_fit_input_and_a_chosen_gimap_analysis_folder_finds_it(tmp_path) -> None:
    single, series = _pages()
    assert "gimap_analysis/fit_input folder" in series.step_intro["curves"].text()
    assert "gimap_analysis/fit_input folder" in series.folder_button.toolTip()
    chosen = tmp_path / "gimap_analysis"
    folder = chosen / "fit_input"
    folder.mkdir(parents=True)
    for index, radius in enumerate(RADII[:3]):
        _write(folder / f"run_{index + 1:05d}_fit_input.dat", radius, index)
    (chosen / "curves").mkdir()  # the other folders of a Batch Export hold no fit inputs
    assert series.open_series(chosen)  # the folder Batch Export was given, not the one it wrote the curves into
    assert len(series.paths) == 3 and series.folder == str(folder) and series.folder_chip.toolTip() == str(folder)
    assert series.status_label.text() == "3 curves listed, in fit_input of the chosen folder."
    # Several sends below one folder: nothing is guessed, the places are named.
    for name in ("first", "second"):
        (tmp_path / "sends" / name / "fit_input").mkdir(parents=True)
        _write(tmp_path / "sends" / name / "fit_input" / "run_00001_fit_input.dat", 5.0, 1)
    assert series.open_series(tmp_path / "sends") is False and series.paths == []
    assert series.status_label.text() == ("No curves here, but in first/fit_input, second/fit_input: choose one of "
                                          "them, or tick “Also in subfolders”.")
    assert series.frame_plot.figure_state()["title"] == "" and series.frame_plot.curve_count() == 0  # not the frame of before
    assert series.frame_note.text() == ""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert series.open_series(empty) is False
    assert series.status_label.text() == f"No curves matching *_fit_input.dat in {empty}."
    series.dispose()
    single.dispose()


def test_file_names_are_shortened_to_what_differs() -> None:
    from src.gimap.features.fitting.presentation.single.series_text import short_names

    real = [f"jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_{frame:05d}_fit_input.dat" for frame in (5, 33, 45, 100)]
    assert short_names(real) == ["…00005", "…00033", "…00045", "…00100"]  # the number whole, the rest said once
    assert short_names([f"run_{frame:05d}_fit_input.dat" for frame in (1, 2, 12)]) == ["run_00001", "run_00002", "run_00012"]
    assert short_names(["film_frame0003_sum10_fit_input.dat", "film_frame0013_sum10_fit_input.dat"]) == [
        "…0003", "…0013"]  # never a part of a number: not “…03”, “…13”
    assert short_names(["film_0003_sum10_fit_input.dat", "film_0013_sum10_fit_input.dat"]) == ["film_0003", "film_0013"]
    two_runs = [f"run_{run}/fit_input/frame_{frame:05d}_fit_input.dat" for run in "AB" for frame in (1, 2)]
    assert short_names(two_runs) == ["run_A/fit_input/frame_00001", "run_A/fit_input/frame_00002",
                                     "run_B/fit_input/frame_00001", "run_B/fit_input/frame_00002"]
    assert short_names(two_runs[::2]) == ["run_A", "run_B"]  # what differs, also when it is the folder
    assert short_names(["only_one_fit_input.dat"]) == ["only_one_fit_input.dat"] and short_names([]) == []
    assert short_names(["cut.dat", "other.txt"]) == ["cut.dat", "other.txt"]  # nothing shared: as they are
    assert short_names(["a_fit_input.dat", "ab_fit_input.dat"]) == ["a", "ab"]


def test_a_line_of_the_list_leads_with_the_state_and_wraps_instead_of_being_cut(tmp_path) -> None:
    from PyQt5.QtCore import Qt

    single, series = _pages()
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    names = [f"jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_{index + 1:05d}_fit_input.dat" for index in range(5)]
    for index, name in enumerate(names):
        _write(folder / name, RADII[index], index)
    (folder / names[2]).write_text("not a curve\n", encoding="utf-8")
    single.open_curve(folder / names[0])
    single.set_model(_model(5.2))
    series.open_series(folder)
    assert series.frame_list.item(0).text() == "· 1  …00001"
    assert series.frame_list.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff and series.frame_list.wordWrap()
    assert series.frame_plot.figure_state()["title"] == f"1 · {names[0]}"  # the whole name: where the frame is shown
    series.start()
    _wait(series)
    texts = [series.frame_list.item(row).text() for row in range(5)]
    assert texts[0].startswith("✓ 1  …00001  χ²ᵣ ") and texts[2].startswith("✗ 3  …00003  No data rows found")
    assert series.frame_list.item(2).toolTip().startswith("No data rows found")  # in full on hover too
    QApplication.processEvents()
    wide = series.frame_list.viewport().width()
    assert all(series.frame_list.visualItemRect(series.frame_list.item(row)).width() <= wide for row in range(5))
    assert series.frame_list.visualItemRect(series.frame_list.item(2)).height() > series.frame_list.visualItemRect(
        series.frame_list.item(0)).height()  # the long reason takes a second line
    series.dispose()
    single.dispose()


def test_an_unreadable_file_does_not_end_the_search_for_stages_and_is_marked_before_start(tmp_path) -> None:
    radii = [5.0 + 0.1 * i for i in range(10)] + [6.0] * 10
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    for index, radius in enumerate(radii):
        exact = evaluate(_model(radius), Q_NM)
        measured = exact + np.random.default_rng(index).normal(0.0, 0.01 * exact)
        np.savetxt(folder / f"run_{index + 1:05d}_fit_input.dat", np.column_stack([Q_NM / 10.0, measured, 0.01 * exact]))
    (folder / "run_00014_fit_input.dat").write_bytes(b"# a write cut short\n-0.1985 0.2" + bytes(range(256)))
    single, series = _pages()
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.0))
    series.open_series(folder)
    end = time.monotonic() + 60
    while series._series_stages is None and time.monotonic() < end:
        series.stage_tasks.wait(0.1)
        QApplication.processEvents()
    stages = series._series_stages
    assert stages is not None and stages.count >= 2 and list(series._skipped) == [13]  # found on the other 19
    assert series.stages_label.text().startswith(f"{stages.count} stages: frames 1–")
    assert series.stages_label.text().endswith("cannot be used: frames 14")
    item = series.frame_list.item(13)
    assert item.text().startswith("✗ 14") and series.frame_list.item(12).text().startswith("· 13")
    assert series.stage_of_frame(13) == series.stage_of_frame(14)  # in a stage by its place
    series.frame_list.setCurrentRow(13)
    assert series.frame_plot.curve_count() == 0 and series.frame_note.text().startswith("Frame 14 cannot be fitted: ")
    series.dispose()
    single.dispose()


def test_first_is_never_after_last(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    series.open_series(folder)
    series.last_spin.setValue(2)
    series.first_spin.setValue(4)  # First past Last: Last follows, the range is never empty
    assert series.last_spin.value() == 4 and series._listed() == [3] and series.frame_list.count() == 1
    assert series.step_rail.detail("curves") == "1 of 5 curves" and series.start_button.isEnabled()
    series.first_spin.setValue(1)
    series.last_spin.setValue(5)
    assert series._listed() == [0, 1, 2, 3, 4]
    series.dispose()
    single.dispose()


def test_links_to_curves_that_lie_elsewhere_are_listed(tmp_path) -> None:
    import os

    from src.gimap.features.fitting.application.models import DiscoverInSituFramesRequest
    from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalInSituFrameRepository

    store = _series(tmp_path)
    linked = tmp_path / "linked"
    linked.mkdir()
    try:
        for number in (10, 2, 1):
            os.symlink(store / f"run_{min(number, 5):05d}_fit_input.dat", linked / f"frame_{number}_fit_input.dat")
    except (OSError, NotImplementedError):
        pytest.skip("links cannot be made here")
    frames = LocalInSituFrameRepository().discover_insitu_frames(DiscoverInSituFramesRequest(linked))
    assert [frame.path.name for frame in frames] == ["run_00001_fit_input.dat", "run_00002_fit_input.dat",
                                                     "run_00005_fit_input.dat"]  # in the order of the links: 1, 2, 10


def test_the_new_sentences_of_the_series_page_have_their_chinese() -> None:
    from src.gimap.app.presentation.i18n import ZH
    from src.gimap.features.fitting.application.models import CURVE_NOT_FOUND
    from src.gimap.features.fitting.presentation.single import (
        series_halves,
        series_listing,
        series_page,
        series_results,
        series_stages,
        series_watch,
    )
    from src.gimap.features.fitting.presentation.single.series_text import HALF_TEXT, ONE_HALF
    from src.gimap.features.fitting.presentation.views import fit_series_view

    texts = [series_halves.HALVES_TIP, series_halves.ONE_SIDED, series_halves.BOTH_IN_SINGLE, *series_halves.SOURCE_TEXT.values(),
             series_listing.CURVES_HINT, series_page.NOT_SET_UP, series_page.NOT_SET_UP_NOTE, series_results.STALE,
             series_results.UNTOUCHED, fit_series_view.FOLDER_HINT, *ONE_HALF.values(), HALF_TEXT["mean"],
             "{side} ({source})", "{side}; {range}", "Halves of the cut", "Log scale", "Logarithmic axis of the value", "Points", "Halves",
             "{count} fitted", "{count} failed", "{count} not converged", "{count} stopped", "{count} poor fits",
             "Poor fit (χ²ᵣ above {limit}): frames {frames}", "{points} points per frame", "{low}–{high} points per frame",
             "{points} points", "{count} left out", "{single} from one half alone",
             "{points} points, {single} of them from one half alone",
             "poor fit", "not converged", "stopped", "cannot be used: frames {frames}",
             "Frame {frame} cannot be fitted: {reason}", "Set Up on Frame {n} in Single Analysis",
             "Waiting for new curves in {folder} … (Stop to end)", "{done} frames · waiting for new curves",
             "{count} curves changed after they were read: fitted again.",
             "{count} of them are outside the chosen frames and are not fitted.",
             "Series stopped: {count} frames fitted, {failed} failed.",
             "{count} curves listed, in {place} of the chosen folder.",
             "No curves here, but in {places}: choose one of them, or tick “Also in subfolders”.",
             "The results of the last run are not saved.\nSave the table of every frame first?", "In-situ series",
             "The points the frame was fitted on", "What those points are: the mean of both halves of the cut, or one half",
             "measured (+q)", "measured (−q)",
             series_stages.LOOKING, series_stages.NOT_FOR_THIS_RUN, series_stages.NO_STAGES_AT_START,
             series_watch.FOLDER_GONE, CURVE_NOT_FOUND]
    missing = [text for text in texts if ZH.get(text) in (None, text)]
    assert not missing, missing
    for template in (text for text in texts if "{" in text):
        import re

        assert sorted(re.findall(r"\{(\w+)\}", template)) == sorted(re.findall(r"\{(\w+)\}", ZH[template])), template
