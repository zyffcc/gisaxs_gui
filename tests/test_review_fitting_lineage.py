"""Review of Fitting: what the rail says of solutions and stale fits, read-only saves, and the In-situ
series reading its frames as it was started (points, unit) and never changing under a run."""

from __future__ import annotations

import os
import threading
import time

import pytest
from PyQt5.QtWidgets import QApplication

from tests.test_fit_page import _curve_file, _page, _shape_solution
from tests.test_fit_page import _wait as _wait_fit
from tests.test_fit_series import _model, _pages, _series
from tests.test_fit_series import _wait as _wait_series

STALE = "Changed after the fit: Fit again"


def test_solutions_hold_only_on_the_points_they_were_searched_on(tmp_path) -> None:
    gate = threading.Event()

    def quick_fit(*_args, **_kwargs):
        gate.wait(30)  # the search runs while the band is dragged
        return [_shape_solution(radius) for radius in (5.0, 6.0, 7.0)]

    page = _page(quick_fit=quick_fit)
    page.open_curve(_curve_file(tmp_path))
    assert page.method() == "shapes"
    searched_on = page.session.selection()
    page.run_fit()
    assert page.fit_running()
    page.set_range((0.5, 0.7))  # during the search: its χ² is still of every point
    gate.set()
    _wait_fit(page)
    assert page.session.solutions and page.session.solutions_selection == searched_on
    assert page.step_rail.state("fit") == "warn" and page.step_rail.detail("fit") == STALE
    assert page.step_rail.state("results") == "warn" and page.step_rail.detail("results") == STALE
    assert "changed after the search" in page.warnings_label.text()
    page.set_range(None)  # the points of the search again
    assert page.step_rail.state("fit") == "ok" and page.step_rail.detail("fit").startswith("3 solutions · best")
    assert page.step_rail.state("results") == "ok" and page.step_rail.detail("results") == "3 solutions to compare"
    assert page.warnings_label.text() == ""
    full = page.session.all_points()
    page.exclude_button.setChecked(True)
    page._clicked(float(full.q[100]), float(full.intensity[100]))  # a point left out
    assert page.step_rail.state("fit") == "warn" and page.step_rail.state("results") == "warn"
    page._clicked(float(full.q[100]), float(full.intensity[100]))  # taken back
    assert page.step_rail.state("fit") == "ok"
    page.open_curve(_curve_file(tmp_path, name="other_fit_input.dat"))  # another curve: no solutions
    assert page.session.solutions == [] and page.session.solutions_selection is None
    page.dispose()


def test_a_solution_of_analyze_is_a_start_not_a_fit(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    row = dict(_shape_solution(6.0), best_chi2_weighted=0.9)
    assert page.show_solution(row)
    assert page.session.solutions_selection is None and page.solutions_table.rowCount() == 1
    assert page.step_rail.state("fit") == "pending" and page.step_rail.detail("fit") == "From Analyze: Fit to refine it"
    assert page.step_rail.state("results") == "pending"
    page.dispose()


def test_results_warn_when_the_fit_no_longer_holds(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    page.set_model(page.session.model.with_values({(0, "R"): 5.5, (0, "D"): 42.0}))
    page.method_buttons["local"].setChecked(True)
    page.run_fit()
    _wait_fit(page)
    assert page.session.result_is_current() and page.step_rail.state("results") == "ok"
    page.set_range((0.5, 1.5))
    assert page.step_rail.state("results") == "warn" and page.step_rail.detail("results") == STALE
    page.set_range(None)
    assert page.step_rail.state("results") == "ok" and page.step_rail.detail("results") == "Errors, solutions, Save"
    page.dispose()


def test_a_read_only_table_is_not_replaced_and_a_stray_old_one_never_blocks(tmp_path) -> None:
    from src.gimap.features.fitting.presentation.single.results import write_pair, write_text_whole

    target = tmp_path / "ro.csv"
    target.write_text("old", encoding="utf-8")
    os.chmod(target, 0o444)
    try:
        with pytest.raises(PermissionError):
            write_pair(target, lambda path: path.write_text("new", encoding="utf-8"), {"a": 1})
        assert target.read_text(encoding="utf-8") == "old"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["ro.csv"]  # no record, no .old, no .part
        model = tmp_path / "model.json"
        model.write_text("{}", encoding="utf-8")
        os.chmod(model, 0o444)
        with pytest.raises(PermissionError):
            write_text_whole(model, '{"new": 1}')
        assert model.read_text(encoding="utf-8") == "{}"
        page = _page()
        page.open_curve(_curve_file(tmp_path))
        assert page.export_data_dialog(str(target)) is None  # the page says so, no exception
        assert page.status_label.text().startswith("Could not save:") and "read-only" in page.status_label.text()
        assert target.read_text(encoding="utf-8") == "old"
        page.dispose()
    finally:
        for path in tmp_path.iterdir():
            os.chmod(path, 0o666)
    table = tmp_path / "t.csv"
    table.write_text("first", encoding="utf-8")
    stray = tmp_path / "t.csv.old"  # left by an interrupted save, read-only: it may be the only copy
    stray.write_text("stray", encoding="utf-8")
    os.chmod(stray, 0o444)
    try:
        write_pair(table, lambda path: path.write_text("second", encoding="utf-8"), {"a": 1})
        assert table.read_text(encoding="utf-8") == "second" and stray.read_text(encoding="utf-8") == "stray"
        assert sorted(path.name for path in tmp_path.glob("t.*")) == ["t.csv", "t.csv.old", "t.json"]
    finally:
        os.chmod(stray, 0o666)


def test_before_start_the_frame_is_drawn_again_when_single_fits_other_points(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.frame_list.setCurrentRow(2)
    _name, x, _y = series.frame_plot.figure_state()["curves"][0]
    assert x.min() < 0.1 and x.max() > 1.9
    single.set_range((0.5, 1.5))
    series.refresh()  # as showing In-situ series does; no wait for the stage search
    _name, x, _y = series.frame_plot.figure_state()["curves"][0]
    assert x.min() >= 0.5 and x.max() <= 1.5 and series.frame_list.currentRow() == 2
    single.open_curve(folder / "run_00001_fit_input.dat", unit="nm")  # q read in nm⁻¹: 10× smaller
    series.refresh()
    _name, x, _y = series.frame_plot.figure_state()["curves"][0]
    assert x.max() == pytest.approx(0.2, rel=0.01)
    series.dispose()
    single.dispose()


def test_a_running_series_keeps_the_unit_of_q_it_was_started_with(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    single.open_curve(folder / "run_00001_fit_input.dat")  # Å⁻¹, as Analyze writes them
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.start()
    assert series._unit == "angstrom"
    single.open_curve(folder / "run_00003_fit_input.dat", unit="nm")  # another curve looked at meanwhile
    assert series._preview_unit() == "angstrom"  # the frames are drawn as the run reads them
    _wait_series(series)
    assert len(series.fits) == 5 and all(frame.ok for frame in series.fits.values())
    assert all(float(series._curves[index].q.max()) == pytest.approx(2.0, rel=0.01) for index in series.fits)
    radii = [series.fits[index].result.model.get((0, "R")).value for index in sorted(series.fits)]
    assert max(radii) < 8.0  # no tenfold jump in R
    series.dispose()
    single.dispose()


def test_a_project_does_not_change_a_running_series(tmp_path) -> None:
    single, series = _pages()
    folder = _series(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    single.open_curve(folder / "run_00001_fit_input.dat")
    single.set_model(_model(5.2))
    series.open_series(folder)
    series.start()
    assert series.running
    before = series.project_state()
    notes = series.apply_project_state({"folder": str(other), "first": 2, "last": 3, "every": 1, "start": "same",
                                        "method": "global", "subfolders": True})
    assert notes == ["the In-situ series was not restored: a series is running"]
    assert series.project_state() == before and series.running
    series.stop()
    end = time.monotonic() + 60
    while series.running and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    assert not series.running
    series.dispose()
    single.dispose()


def test_the_series_preview_unit_follows_single_before_start(tmp_path) -> None:
    single, series = _pages()
    assert series._preview_unit() == "angstrom" and not series._started()
    folder = _series(tmp_path, (5.0, 5.4))
    single.open_curve(folder / "run_00001_fit_input.dat", unit="nm")
    assert series._preview_unit() == "nm"
    series.dispose()
    single.dispose()
