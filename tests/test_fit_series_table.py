"""The Results step of Fitting ▸ In-situ series: the table of every frame (one row per finished frame, never
a freeze), what a row says of a frame that failed or fits badly, what the run came to, and saving."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import replace

import numpy as np
import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHeaderView, QMessageBox

from src.gimap.app.presentation.theme import theme_manager
from src.gimap.features.fitting.application.single_fit import evaluate
from tests.test_fit_series import Q_NM, RADII, _answer, _curve_of, _cut, _model, _pages, _series, _wait, _write


def _settle(seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.005)


def _fitted(tmp_path, radii=RADII, *, bad=None, model=5.2):
    """A series fitted to the end: ``bad`` names a frame (0-based) whose file is no curve."""
    single, series = _pages()
    folder = _series(tmp_path, radii)
    if bad is not None:
        (folder / f"run_{bad + 1:05d}_fit_input.dat").write_text("not a curve\n", encoding="utf-8")
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(model))
    assert series.open_series(folder)
    series.start()
    _wait(series)
    return single, series, folder


def _header(table) -> list[str]:
    return [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]


def _row(table, row: int) -> list[str]:
    return [table.item(row, column).text() if table.item(row, column) is not None else "" for column in range(table.columnCount())]


def test_three_hundred_frames_do_not_freeze_the_results_step(tmp_path) -> None:
    """The table was filled whole at every frame with columns that measured every cell again: 300 rows took
    more than a minute at each refresh (the end of a run, a return to the page, a click on an option)."""
    from src.gimap.features.fitting.application.series_fit import SeriesSettings, fit_frame

    single, series = _pages()
    folder = tmp_path / "gimap_analysis"
    folder.mkdir()
    for index in range(301):
        (folder / f"run_{index + 1:05d}_fit_input.dat").write_text("0.01 1 1\n0.02 1 1\n0.03 1 1\n", encoding="utf-8")
    single.set_model(_model(5.2))
    assert series.open_series(folder)
    series.show_step("results")
    _settle()
    header = series.results_table.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.Interactive and not header.stretchLastSection()
    frame = fit_frame(0, "x", _curve_of(_cut(5.4, 1)), _model(5.2), SeriesSettings())
    series._start_model = _model(5.2)
    series.fits = {index: replace(frame, index=index, path=series.paths[index]) for index in range(300)}
    started = time.monotonic()
    series.refresh()  # the table is of other fits: filled once
    first = time.monotonic() - started
    _settle()
    assert series.results_table.rowCount() == 300 and series.results_table.item(299, 0).text() == "300"
    started = time.monotonic()
    series.refresh()
    again = time.monotonic() - started
    started = time.monotonic()
    series._frame_done(replace(frame, index=300, path=series.paths[300]))  # one more frame: one more row
    one_frame = time.monotonic() - started
    _settle()
    assert series.results_table.rowCount() == 301 and series.results_table.item(300, 0).text() == "301"
    assert first < 0.5 and again < 0.5 and one_frame < 0.2, (first, again, one_frame)
    series.dispose()
    single.dispose()


def test_the_table_is_filled_one_row_per_finished_frame(tmp_path, monkeypatch) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    rebuilt, rows = [], []
    rebuild, show = series._rebuild_results, series._show_result_row
    monkeypatch.setattr(series, "_rebuild_results", lambda: rebuilt.append(len(series.fits)) or rebuild())
    monkeypatch.setattr(series, "_show_result_row", lambda frame: show(frame) or rows.append(series.results_table.rowCount()))
    series.start()
    _wait(series)
    assert rebuilt == [0]  # emptied at Start, and never filled whole again
    assert rows == [1, 2, 3, 4, 5] and series._table_frames == [0, 1, 2, 3, 4]
    series.refresh()
    series.show_step("curves")
    series.show_step("results")
    assert rebuilt == [0] and series.results_table.rowCount() == 5
    series.dispose()
    single.dispose()


def test_the_table_shows_every_value_the_points_and_the_halves_and_copies(tmp_path) -> None:
    from src.gimap.app.presentation.components.table_copy import table_text
    from src.gimap.features.fitting.application.series_fit import parameter_columns

    single, series, _folder = _fitted(tmp_path)
    table = series.results_table
    values = parameter_columns(series._start_model)
    header = _header(table)
    assert len(values) >= 6 and table.columnCount() == len(values) + 4  # every free value has its column
    assert header[:4] == ["#", "χ²ᵣ", "Scale", "R (nm)"] and header[-2:] == ["Points", "Halves"]
    assert table.horizontalScrollBarPolicy() != Qt.ScrollBarAlwaysOff  # the columns that do not fit are scrolled to
    first = _row(table, 0)
    assert first[0] == "1" and float(first[1]) == pytest.approx(1.0, abs=0.5) and " ± " in first[3]
    assert first[-2:] == ["200", "q > 0"]  # what the frame was fitted on (curves with one side)
    assert all(table.columnWidth(column) >= table.sizeHintForIndex(table.model().index(0, column)).width()
               for column in range(table.columnCount()))  # no value cut by its column
    assert "200 points" in table.item(0, 3).toolTip() and "run_00001_fit_input.dat" in table.item(0, 3).toolTip()
    copied = table_text(table).splitlines()
    assert copied[0].split("\t") == header and len(copied) == 6 and copied[1].split("\t")[-2:] == ["200", "q > 0"]
    names = [name for _path, name in values]  # the columns of the saved table: plain text, units as the record's
    assert "res_width_nm^-1" in names and all(name.isascii() for name in names)
    # A frame selected in the list is the table's current row, and a row selected shows its frame.
    series.frame_list.setCurrentRow(3)
    assert table.currentRow() == 3
    table.setCurrentCell(1, 0)
    assert series._selected == 1 and series.frame_plot.figure_state()["title"].startswith("2 · run_00002")
    # The value the trend follows comes into view.
    series.resize(900, 700)
    series.show_step("results")
    _settle()
    last_value = len(values) + 1
    series.trend_combo.setCurrentIndex(series.trend_combo.count() - 1)
    _settle()
    shown = table.viewport().rect()
    assert shown.intersects(table.visualRect(table.model().index(1, last_value)))
    series.dispose()
    single.dispose()


def test_a_failed_row_says_why_and_the_counts_are_of_what_was_fitted(tmp_path) -> None:
    single, series, _folder = _fitted(tmp_path, bad=2)
    table = series.results_table
    assert table.rowCount() == 5 and table.item(2, 0).text() == "3"
    assert table.item(2, 1).text().startswith("✗ No data rows found") and table.columnSpan(2, 1) == table.columnCount() - 1
    assert table.item(2, 1).foreground().color().name() == theme_manager().color("danger").name()
    assert table.columnSpan(1, 1) == 1 and table.item(1, 1).data(Qt.ForegroundRole) is None
    assert table.columnWidth(1) < 120  # the reason spans its row: it does not widen the χ²ᵣ column
    summary = series.results_summary.text()
    assert summary.startswith("4 fitted · 1 failed") and "5 frames fitted" not in summary  # the failed one is not “fitted”
    assert "200 points per frame" in summary
    assert series.step_rail.state("results") == "warn" and series.step_rail.detail("results") == "4 fitted · 1 failed"
    assert series.status_label.text() == "Series done: 4 frames fitted, 1 failed."
    # The failed frame, selected: why, where the plot would be (not “the selected frame appears here”).
    table.setCurrentCell(2, 0)
    assert series.frame_plot.curve_count() == 0
    assert series.frame_plot.empty_overlay.label.text().startswith("Frame 3 cannot be fitted: No data rows found")
    assert series.frame_note.text().startswith("Frame 3 cannot be fitted") and series.frame_note.property("gimapRole") == "warning"
    table.setCurrentCell(3, 0)
    assert series.frame_plot.curve_count() >= 2 and series.frame_note.property("gimapRole") == "muted"
    series.dispose()
    single.dispose()


def _ripple(folder, name: str, radius: float = 5.4) -> None:
    """A curve the model cannot follow, measured precisely: its fit converges with a large χ²ᵣ."""
    exact = evaluate(_model(radius), Q_NM) * (1.0 + 0.6 * np.sin(9.0 * Q_NM))
    np.savetxt(folder / name, np.column_stack([Q_NM / 10.0, exact, 0.01 * exact]))


def test_a_fit_with_chi2_far_from_one_is_not_a_green_tick(tmp_path) -> None:
    from src.gimap.features.fitting.application.series_fit import POOR_CHI2, poor_fit

    single, series = _pages()
    folder = _series(tmp_path)
    _ripple(folder, "run_00004_fit_input.dat")
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.start_same.setChecked(True)
    series.start()
    _wait(series)
    poor = series.fits[3]
    assert poor.ok and poor.state == "ok" and poor.result.chi2_reduced > POOR_CHI2 and poor_fit(poor)
    assert not poor_fit(series.fits[0]) and series.fits[0].result.chi2_reduced < 2
    item = series.frame_list.item(3)
    assert item.text().startswith("!") and item.text().endswith("poor fit") and "χ²ᵣ" in item.text()
    assert item.foreground().color().name() == theme_manager().color("warning").name()
    assert "poor fit" in item.toolTip()
    assert series.frame_list.item(0).text().startswith("✓") and series.frame_list.item(0).data(Qt.ForegroundRole) is None
    row = series.results_table.item(3, 1)
    assert row.foreground().color().name() == theme_manager().color("warning").name() and "poor fit" in row.toolTip()
    assert series.step_rail.state("results") == "warn"
    summary = series.results_summary.text()
    assert "1 poor fits" in summary and "Poor fit (χ²ᵣ above 10): frames 4" in summary
    # (the end of the run counts it: no sentence of a success — tests/test_glue.py)
    assert series.status_label.text() == "Series done: 5 frames fitted, 1 of them poor fits (χ²ᵣ above 10), 0 failed."
    path = series.save_table(str(tmp_path / "series.csv"))
    record = json.loads((tmp_path / "series.json").read_text(encoding="utf-8"))
    assert path and record["poor_fit"] == {"chi2_reduced_above": POOR_CHI2, "frames": [4]}
    # Without σ in the file χ²ᵣ is the scatter of ln I: it says nothing of this.
    relative = replace(poor, result=replace(poor.result, weighting="relative"))
    assert not poor_fit(relative) and not poor_fit(replace(poor, result=None, error="no fit"))
    series.dispose()
    single.dispose()


def test_a_stopped_frame_is_marked_and_is_no_point_of_the_trend(tmp_path) -> None:
    from src.gimap.features.fitting.application.series_fit import SeriesSettings, fit_frame

    single, series, _folder = _fitted(tmp_path, RADII[:3])
    calls = []
    stopped = fit_frame(1, series.paths[1], series._curves[1], _model(5.0), SeriesSettings(method="global"),
                        lambda: calls.append(1) or len(calls) > 5)
    assert stopped.state == "stopped"
    series.fits[1] = stopped
    series._show_result_row(stopped)
    series._frames_changed()
    item = series.frame_list.item(1)
    assert item.text().startswith("■") and item.text().endswith("stopped")
    assert item.foreground().color().name() == theme_manager().color("warning").name()
    assert "2 fitted · 1 stopped" in series.results_summary.text() and series.step_rail.state("results") == "warn"
    series._draw_trend()
    _name, x, _y = series.trend_plot.figure_state()["curves"][0]
    assert list(x) == [1.0, 3.0]  # where a stopped fit was is not a result
    series.dispose()
    single.dispose()


def test_the_trend_has_whole_frames_on_its_axis_and_no_legend_over_one_value(tmp_path) -> None:
    single, series, _folder = _fitted(tmp_path)
    plot = series.trend_plot
    assert not plot.legend.isVisible()  # one value: named in the list above the plot and on the axis
    (low, high), _y = plot.plot.getViewBox().viewRange()
    assert low == pytest.approx(0.5) and high == pytest.approx(5.5)  # no point on the edge of the plot
    axis = plot.plot.getAxis("bottom")
    ticks = [value for _spacing, values in axis.tickValues(low, high, 600) for value in values]
    assert ticks and all(float(value).is_integer() for value in ticks)
    assert plot.log_check.text() == "Log scale"  # of a value or χ²ᵣ: not “Log I”
    series.dispose()
    single.dispose()


def test_the_legend_of_the_stages_goes_where_the_fewest_points_are() -> None:
    single, series = _pages()
    frames = np.arange(1.0, 7.0)
    rising = [("stage 1", frames[:3], np.array([4.0, 4.5, 5.0])), ("stage 2", frames[3:], np.array([5.5, 6.0, 6.5]))]
    assert series._legend_corner(rising) == (-10, -10)  # the bottom right: off the late frames of a value that grows
    falling = [(name, x, 10.0 - y) for name, x, y in rising]
    assert series._legend_corner(falling) == (10, -10)  # … of one that falls: the bottom left
    scattered = [("stage 1", frames[:3], np.array([1.2, 0.9, 1.0])), ("stage 2", frames[3:], np.array([1.1, 0.9, 0.95]))]
    assert series._legend_corner(scattered) == (-10, 10)  # χ²ᵣ: wherever no point is (here the top right)
    assert series._legend_corner([("stage 1", frames[:1], np.array([np.nan]))]) == (-10, -10)
    series.dispose()
    single.dispose()


def test_an_untouched_model_is_said_before_start_and_its_run_is_not_green(tmp_path) -> None:
    """Analyze ▸ Send Series to Fitting leaves Single analysis without a curve: the model is its starting guess."""
    single, series = _pages()
    folder = _series(tmp_path)
    assert series.open_series(folder)
    assert single.session.curve is None and not single.session.edited
    assert series.step_rail.state("start") == "warn"
    assert series.step_rail.detail("start") == "Model not set up yet: fit one frame in Single analysis"
    assert not series.setup_note.isHidden() and "starting guess" in series.setup_note.text()
    assert series.start_button.isEnabled() and "Model not set up yet" in series.start_button.toolTip()
    # A listed frame can be sent to Single analysis to set the model up — before any run.
    assert series.edit_model_button.text() == "Set Up on Frame 1 in Single Analysis"
    series.frame_list.setCurrentRow(2)
    assert series.edit_model_button.text() == "Set Up on Frame 3 in Single Analysis"
    opened, edit = [], []
    series.openFrameRequested.connect(lambda path, model: opened.append((path, model)))
    series.editModelRequested.connect(lambda: edit.append(1))
    series.edit_model_button.click()
    assert opened == [(series.paths[2], None)] and not edit
    series.start()
    _wait(series)
    assert all(frame.ok for frame in series.fits.values())
    assert series.step_rail.state("results") == "warn"  # whatever χ²ᵣ came out: nobody set this model up
    assert "never set up or fitted" in series.results_summary.text()
    record_path = series.save_table(str(tmp_path / "guess.csv"))
    assert record_path and "starting guess" in json.loads((tmp_path / "guess.json").read_text(encoding="utf-8"))["listing"]["model"]
    # With a curve and a model set up in Single analysis the step is ready, and the button goes there.
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.refresh()
    assert series.step_rail.state("start") == "ok" and series.setup_note.isHidden()
    assert series.edit_model_button.text() == "Edit in Single Analysis"
    series.edit_model_button.click()
    assert edit == [1] and len(opened) == 1
    series.dispose()
    single.dispose()


def test_a_second_start_asks_before_results_that_were_not_saved_go(tmp_path, monkeypatch) -> None:
    single, series, folder = _fitted(tmp_path, RADII[:3])
    before = dict(series.fits)
    asked = _answer(monkeypatch, QMessageBox.Cancel)
    series.start()
    assert len(asked) == 1 and "not saved" in asked[0][1] and not series.running and series.fits == before  # kept
    assert series.open_series(folder) is False and len(asked) == 2 and series.fits == before  # nor by listing anew
    # Listing the same curves again (the pattern field merely lost the focus) asks nothing and changes nothing.
    series.pattern_edit.editingFinished.emit()
    assert len(asked) == 2 and series.fits == before and series.results_table.rowCount() == 3
    # Save: the table is written (here: to the place the dialog names), then the run starts.
    from PyQt5.QtWidgets import QFileDialog

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *_args, **_kwargs: (str(tmp_path / "kept.csv"), "")))
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *_args: QMessageBox.Save))
    series.start()
    assert series.running and (tmp_path / "kept.csv").is_file() and (tmp_path / "kept.json").is_file()
    _wait(series)
    # Saved results go without a question; so do those the person lets go (Discard).
    assert series.save_table(str(tmp_path / "second.csv"))
    asked = _answer(monkeypatch, QMessageBox.Discard)
    series.start()
    _wait(series)
    assert asked == [] and len(series.fits) == 3
    series.start()
    _wait(series)
    assert len(asked) == 1 and len(series.fits) == 3
    series.dispose()
    single.dispose()


def test_saved_plots_say_what_they_show(tmp_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QFileDialog

    single, series, folder = _fitted(tmp_path, RADII[:3])
    starts = []
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda _parent, _title, start="", *_rest: starts.append(start) or ("", "")))
    series.frame_list.setCurrentRow(1)
    series._save_plot(series.trend_plot, "trend")
    series._save_plot(series.frame_plot, "frame")
    series.trend_combo.setCurrentIndex(0)  # χ²ᵣ
    series._save_plot(series.trend_plot, "trend")
    assert starts == [str(folder / "series_trend_1_Sphere_R_nm.png"), str(folder / "series_frame_00002.png"),
                      str(folder / "series_trend_chi2_reduced.png")]  # a second trend does not overwrite the first
    assert series._plot_words("frame")[1] == "2 · run_00002_fit_input.dat"
    assert series._plot_words("trend")[1] == f"χ²ᵣ · {tmp_path.name} · Sphere"  # the folder that holds gimap_analysis
    titles = []
    from src.gimap.features.fitting.presentation.single import series_results

    write = series_results.write_plot
    monkeypatch.setattr(series_results, "write_plot", lambda item, path: titles.append(
        (item.titleLabel.text, item.titleLabel.geometry().height())) or write(item, path))
    assert series._save_plot(series.frame_plot, "frame", str(tmp_path / "frame.png")) == str(tmp_path / "frame.png")
    assert "run_00002_fit_input.dat" in titles[0][0]  # written into the picture …
    assert titles[0][1] >= 20  # … with a row of its own (not half of it above the edge)
    assert not series.frame_plot.plot.titleLabel.isVisible()  # … and gone from the page, where the header says it
    assert (tmp_path / "frame.png").stat().st_size > 1000
    series.dispose()
    single.dispose()


def test_the_results_step_gives_the_table_room_and_gives_it_back(tmp_path) -> None:
    from src.gimap.features.fitting.presentation.views.fit_series_view import RESULTS_WIDTH

    single, series, _folder = _fitted(tmp_path, RADII[:2])
    series.resize(1280, 760)
    series.show_step("curves")
    _settle()
    narrow = series.splitter.sizes()
    assert narrow[0] < RESULTS_WIDTH
    series.show_step("results")
    assert series.splitter.sizes()[0] >= RESULTS_WIDTH - 2
    series.resize(1700, 900)  # a wide window: half of it, more of the columns at once
    _settle()
    wide = series.splitter.sizes()
    assert wide[0] >= 800 and wide[1] >= 800
    series.resize(1280, 760)  # … and back, the plots keep their room
    _settle()
    assert RESULTS_WIDTH - 2 <= series.splitter.sizes()[0] <= RESULTS_WIDTH + 80 and series.splitter.sizes()[1] >= 460
    series.show_step("start")
    assert series.splitter.sizes() == narrow
    assert "splitter" not in (series.preferences.get("fitting_series_page", {}) or {})  # not the person's width
    series.show_step("results")
    series.splitter.setSizes([640, sum(narrow) - 640])
    series.splitter.splitterMoved.emit(640, 1)  # the person moves the splitter: that width holds on every step
    series.show_step("curves")
    assert series.splitter.sizes()[0] == pytest.approx(640, abs=2)
    series.dispose()
    single.dispose()


def test_the_saved_table_and_record_name_the_files_of_subfolders_apart(tmp_path, monkeypatch) -> None:
    _answer(monkeypatch, QMessageBox.Discard)
    single, series = _pages()
    root = tmp_path / "gimap_analysis"
    for run in ("run_A", "run_B"):
        (root / run / "fit_input").mkdir(parents=True)
        for index, radius in enumerate(RADII[:2]):
            _write(root / run / "fit_input" / f"frame_{index + 1:05d}_fit_input.dat", radius, index)
    single.open_curve(root / "run_A" / "fit_input" / "frame_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.subfolders_check.setChecked(True)
    assert series.open_series(root) and len(series.paths) == 4
    texts = [series.frame_list.item(row).text() for row in range(4)]
    assert "run_A/fit_input/frame_00001" in texts[0] and "run_B/fit_input/frame_00001" in texts[2]
    series.start()
    _wait(series)
    path = series.save_table(str(tmp_path / "two_runs.csv"))
    files = [row["file"] for row in csv.DictReader(open(path, encoding="utf-8"))]
    assert files == ["run_A/fit_input/frame_00001_fit_input.dat", "run_A/fit_input/frame_00002_fit_input.dat",
                     "run_B/fit_input/frame_00001_fit_input.dat", "run_B/fit_input/frame_00002_fit_input.dat"]
    record = json.loads((tmp_path / "two_runs.json").read_text(encoding="utf-8"))
    assert [item["file"] for item in record["per_frame"]] == files and record["listing"]["subfolders"] is True
    series.dispose()
    single.dispose()


# -- after the review: links, a run without a result, the name of a series, the bars ------------------

def test_frames_of_subfolders_stay_apart_when_the_folder_is_reached_through_a_link(tmp_path, monkeypatch) -> None:
    """The listing follows links (a linked folder, a linked mount point, a mapped drive): the frames then lie
    below the place the link leads to, not below the folder as it was chosen."""
    import os

    from src.gimap.features.fitting.application.series_fit import file_name

    _answer(monkeypatch, QMessageBox.Discard)
    real = tmp_path / "two_runs_real"
    for run in ("run_A", "run_B"):
        (real / run / "fit_input").mkdir(parents=True)
        for index, radius in enumerate(RADII[:2]):
            _write(real / run / "fit_input" / f"frame_{index + 1:05d}_fit_input.dat", radius, index)
    link = tmp_path / "link_to_runs"
    try:
        os.symlink(real, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("links cannot be made here")
    names = ["run_A/fit_input/frame_00001_fit_input.dat", "run_A/fit_input/frame_00002_fit_input.dat",
             "run_B/fit_input/frame_00001_fit_input.dat", "run_B/fit_input/frame_00002_fit_input.dat"]
    lying = real.resolve() / "run_B" / "fit_input" / "frame_00001_fit_input.dat"
    assert file_name(lying, link) == names[2] and file_name(str(lying), str(link)) == names[2]
    assert file_name(lying, tmp_path / "elsewhere") == "frame_00001_fit_input.dat"  # not below it: the name
    assert file_name(lying) == "frame_00001_fit_input.dat"
    single, series = _pages()
    single.open_curve(real / "run_A" / "fit_input" / "frame_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.subfolders_check.setChecked(True)
    assert series.open_series(link) and len(series.paths) == 4 and series.folder == str(link)
    texts = [series.frame_list.item(row).text() for row in range(4)]
    assert "run_A/fit_input/frame_00001" in texts[0] and "run_B/fit_input/frame_00001" in texts[2]
    series.start()
    _wait(series)
    assert names[2] in series.results_table.item(2, 0).toolTip()
    path = series.save_table(str(tmp_path / "linked.csv"))
    assert [row["file"] for row in csv.DictReader(open(path, encoding="utf-8"))] == names
    record = json.loads((tmp_path / "linked.json").read_text(encoding="utf-8"))
    assert [item["file"] for item in record["per_frame"]] == names
    series.dispose()
    single.dispose()


def test_a_run_that_fitted_no_frame_leaves_nothing_to_ask_about(tmp_path, monkeypatch) -> None:
    """Stop during the first frame: “0 fitted · 1 stopped”. Where a stopped fit was is not a result."""
    single, series = _pages()
    folder = _series(tmp_path, RADII[:3])
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    assert series.open_series(folder)
    series.method_global.setChecked(True)  # a search of the ranges: long enough to be stopped in
    asked = _answer(monkeypatch, QMessageBox.Cancel)
    series.start()
    series.stop()
    _wait(series)
    assert [frame.state for frame in series.fits.values()] == ["stopped"]
    assert series.status_label.text() == "Series stopped: 0 frames fitted, 0 failed."
    assert series.results_summary.text().startswith("0 fitted · 1 stopped")
    series.start()
    assert asked == [] and series.running  # a second Start: nothing to save, nothing asked
    series.stop()
    _wait(series)
    assert [frame.state for frame in series.fits.values()] == ["stopped"]
    assert series.open_series(folder) and asked == []  # nor when the curves are listed anew
    series.dispose()
    single.dispose()


def test_a_series_is_named_by_the_folder_that_holds_it() -> None:
    """Every series Analyze writes lies in “gimap_analysis/fit_input”: those two names tell no series from
    another (the title of a saved trend, the line about the curves, what Watch waits in)."""
    from src.gimap.features.fitting.presentation.single.series_text import series_name

    assert series_name("/data/beamtime/sampleA/gimap_analysis/fit_input") == "sampleA"
    assert series_name("/data/beamtime/sampleA/gimap_analysis/handover/fit_input") == "sampleA · handover"  # a Batch Export
    assert series_name("/data/beamtime/sampleA/gimap_analysis") == "sampleA"  # listed with its subfolders
    assert series_name("/data/beamtime/sampleA/GIMaP_Analysis/Fit_Input") == "sampleA"
    assert series_name("/data/beamtime/curves") == "beamtime/curves"  # any other folder: its last two names
    assert series_name("/data/beamtime/sampleA/fit_input") == "beamtime/sampleA"
    assert series_name("/gimap_analysis/fit_input") == "gimap_analysis"  # nothing above it to name it by
    assert series_name("curves") == "curves" and series_name("") == ""


def test_the_page_and_a_saved_trend_name_the_series_of_analyze(tmp_path, monkeypatch) -> None:
    single, series = _pages()
    folder = tmp_path / "sampleA" / "gimap_analysis" / "fit_input"
    folder.mkdir(parents=True)
    for index, radius in enumerate(RADII[:3]):
        _write(folder / f"run_{index + 1:05d}_fit_input.dat", radius, index)
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    assert series.open_series(folder)
    assert series.step_intro["curves"].text() == "3 of 3 curves · sampleA"
    assert series.folder_chip.toolTip() == str(folder)  # the whole path stays where it was
    series.watch_check.setChecked(True)
    series.start()
    end = time.monotonic() + 60
    while not series._waiting_for_curves() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
    assert series.status_label.text() == "Waiting for new curves in sampleA … (Stop to end)"
    series.stop()
    _wait(series)
    series.watch_check.setChecked(False)
    assert series._plot_words("trend")[1] == "1·Sphere R (nm) · sampleA · Sphere"
    other = tmp_path / "sampleB" / "gimap_analysis" / "fit_input"  # another series of Analyze: another title
    other.mkdir(parents=True)
    for index, radius in enumerate(RADII[:3]):
        _write(other / f"run_{index + 1:05d}_fit_input.dat", radius, index)
    _answer(monkeypatch, QMessageBox.Discard)
    assert series.open_series(other)
    series.start()
    _wait(series)
    assert series._plot_words("trend")[1] == "1·Sphere R (nm) · sampleB · Sphere"
    series.dispose()
    single.dispose()


def test_the_bars_of_the_trend_stay_at_a_switch_of_the_theme(tmp_path) -> None:
    """The plot draws its own curves again for the new theme and clears what the page added to them."""
    from src.gimap.app.presentation.theme import apply_theme

    single, series, _folder = _fitted(tmp_path, RADII[:3])
    assert series.trend_plot.curve_count() == 2  # R and its ±1σ bars
    try:
        apply_theme("dark", 9)
        QApplication.processEvents()
        assert series.trend_plot.curve_count() == 2
    finally:
        apply_theme("light", 9)
    QApplication.processEvents()
    assert series.trend_plot.curve_count() == 2
    series.dispose()
    single.dispose()
