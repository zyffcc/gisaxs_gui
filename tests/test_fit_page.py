"""The single-curve page of Fitting: open a curve, fit it (four methods, one button), results, save."""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.fitting.application.single_fit import FitModel, evaluate, new_component
from tests.test_assistant_gui import _app

Q_NM = np.linspace(0.08, 2.0, 240)


def _truth() -> FitModel:
    model = FitModel((new_component("sphere", radius=6.0, distance=40.0),))
    return model.with_values({(0, "Int"): 2000.0, (0, "sigma_R"): 0.15, (0, "sigma_D"): 0.3,
                              ("globals", "background"): 3.0})


def _curve_file(tmp_path, *, signed=False, sigma=True, name="sample_fit_input.dat"):
    exact = evaluate(_truth(), Q_NM)
    noise = np.sqrt(exact) + 0.5
    measured = exact + np.random.default_rng(2).normal(0.0, noise)
    q_a = Q_NM / 10.0
    if signed:  # a horizontal cut through the beam: both halves
        q_a = np.concatenate([-q_a[::-1], q_a])
        measured = np.concatenate([measured[::-1] * 1.02, measured])
        noise = np.concatenate([noise[::-1], noise])
    columns = [q_a, measured] + ([noise] if sigma else [])
    path = tmp_path / name
    np.savetxt(path, np.column_stack(columns))
    return path


def _page(**kwargs):
    from src.gimap.app import AppContext
    from src.gimap.features.fitting.bootstrap import create_fitting_view_model
    from src.gimap.features.fitting.presentation.single.page import FitPage
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
    page = FitPage(create_fitting_view_model(context), preferences=context.preferences, **kwargs)
    page.resize(1400, 900)
    page.show()
    return page


def _wait(page, limit: float = 120.0) -> None:
    end = time.monotonic() + limit
    while page.fit_running() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    page.tasks.wait(5)
    QApplication.processEvents()


def test_a_curve_opens_on_the_first_step_that_needs_the_person(tmp_path) -> None:
    page = _page(quick_fit=lambda *args, **kwargs: [])
    assert page.step_rail.state("curve") == "pending" and not page.fit_button.isEnabled()
    assert page.open_curve(_curve_file(tmp_path))
    assert page.step_rail.state("curve") == "ok" and page.fit_button.isEnabled()
    assert "240 points" in page.curve_info.text() and "σ from the file" in page.sigma_note.text()
    assert page.side_combo.isHidden()  # one half only: nothing to choose
    model = page.session.model
    assert model.components[0].family == "sphere" and not page.session.edited
    # The starting model is brought to the level of the curve (its scales solved), not three decades below.
    level = evaluate(model, Q_NM[:5]).mean() / evaluate(_truth(), Q_NM[:5]).mean()
    assert 0.1 < level < 10
    assert page.step_stack.currentWidget() is page.step_pages["fit"]  # a first curve: find the shape first
    page.dispose()


def test_refine_finds_the_values_with_errors_and_undo_brings_the_start_back(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    start = page.session.model.with_values({(0, "R"): 5.3, (0, "D"): 44.0})
    page.set_model(start)
    page.method_buttons["local"].setChecked(True)
    page.run_fit()
    assert page.fit_running() and page.stop_button.isVisible()
    _wait(page)
    result = page.session.result
    assert result is not None and result.converged and not page.stop_button.isVisible()
    fitted = page.session.model
    assert fitted.get((0, "R")).value == pytest.approx(6.0, rel=0.05)
    assert result.chi2_reduced == pytest.approx(1.0, abs=0.3)
    row = page.model_editor.row((0, "R"))
    assert row.error.text().startswith("± ") and "nm" in row.error.text()
    assert page.parameters_table.rowCount() >= 5 and page.step_rail.state("fit") == "ok"
    assert page.step_stack.currentWidget() is page.step_pages["results"]
    page.undo()  # the fit is one step of the history
    assert page.session.model == start and page.session.result is not None and not page.session.result_is_current()
    assert "changed after this fit" in page.warnings_label.text()
    page.redo()
    assert page.session.model == fitted
    page.dispose()


def test_the_range_and_the_halves_choose_the_points(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path, signed=True), "mean")
    assert not page.side_combo.isHidden() and page.session.side == "mean"
    assert page._data().q.size == 240
    page.set_side("both")
    assert page._data().q.size == 480
    page.set_side("positive")
    assert page._data().q.size == 240
    page.set_range((0.5, 1.5))
    data = page._data()
    assert data.q.min() >= 0.5 and data.q.max() <= 1.5 and data.q.size < 240
    assert "in the fitting range" in page.curve_info.text()
    page._band_dragged(0.2, 1.0)  # dragging the band on the plot
    assert page.session.q_range == pytest.approx((0.2, 1.0))
    page.dispose()


def test_a_solution_from_analyze_becomes_the_model(tmp_path) -> None:
    from src.gimap.features.fitting.infrastructure.adapters.experimental_fit import forward

    page = _page()
    page.open_curve(_curve_file(tmp_path))
    params = {"R": 6.0, "sigma_R": 0.15, "D": 40.0, "sigma_D": 0.3}
    components = [{"type": "sphere", "type_id": 1, "weight": 1.0, "amplitude": 2000.0, "params": params}]
    globals_ = {"background": 3.0, "resolution_amplitude": 0.0, "sigma_Res": 0.02, "nu_Res": 3.0}
    row = {"combination": "sphere", "components": components, "global_params": globals_, "best_chi2_weighted": 1.1,
           "native_q": Q_NM.tolist(), "native_fit": forward(Q_NM, components, globals_).tolist()}
    assert page.show_solution(row)
    model = page.session.model
    assert model.get((0, "R")).value == 6.0 and model.get((0, "sigma_R")).value == pytest.approx(0.15)
    assert "draws the same curve" in page.status_label.text()
    assert page.solutions_table.rowCount() == 1
    page.dispose()


def test_save_the_fit_the_plot_and_the_model_and_load_a_model(tmp_path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QFileDialog

    page = _page()
    page.open_curve(_curve_file(tmp_path))
    page.run_fit()  # method: find the shape is not available without a quick fit → refine
    _wait(page)
    table = page.export_data_dialog(str(tmp_path / "fit.csv"))
    header = (tmp_path / "fit.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("q_nm^-1,q_A^-1,I,sigma,model,residual")
    record = json.loads((tmp_path / "fit.json").read_text(encoding="utf-8"))
    assert record["model"]["spreads"] == "relative" and record["points"]["weighting"] == "sigma"
    assert table and "fit" in record and record["fit"]["method"] == "local"
    assert (tmp_path / "plot.png").exists() is False
    page.export_plot_dialog(str(tmp_path / "plot.png"))
    assert (tmp_path / "plot.png").stat().st_size > 5000
    page.save_model_dialog(str(tmp_path / "model.json"))
    saved = page.session.model
    page.set_model(FitModel((new_component("cylinder"),)))
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(tmp_path / "model.json"), "")))
    page.load_model_dialog()
    assert page.session.model == saved
    page.dispose()


def test_find_the_particle_shape_lists_solutions_and_puts_the_best_in_the_model(tmp_path) -> None:
    from src.gimap.features.fitting.bootstrap import create_quick_fit

    page = _page(quick_fit=create_quick_fit())
    page.open_curve(_curve_file(tmp_path))
    assert page.method() == "shapes" and page.fit_step_button.text() == "Find the particle shape"
    page.run_fit()
    _wait(page, 300)
    solutions = page.session.solutions
    assert solutions and solutions == sorted(solutions, key=lambda item: item.chi2)
    assert page.session.model == solutions[0].model and page.method() == "local"  # next: refine it here
    assert page.solutions_table.rowCount() == len(solutions) and not page.solutions_table.isHidden()
    best = solutions[0].model.components[0]
    assert best.family == "sphere" and best.value("R") == pytest.approx(6.0, rel=0.1)
    page.solutions_table.selectRow(min(1, len(solutions) - 1))
    page.use_selected_solution()
    assert page.session.model == solutions[min(1, len(solutions) - 1)].model
    page.dispose()


def test_in_the_window_analyze_sends_curves_here_and_in_situ_takes_the_model(tmp_path) -> None:
    from tests.test_fitting_presentation import _fitting_window

    app, window = _fitting_window()
    components = window.components
    path = _curve_file(tmp_path, signed=True)
    components._send_curve_to_fitting(path, "mean")
    app.processEvents()
    page = components.fitting_workspace.fit_page
    assert page.session.curve is not None and page.session.side == "mean"
    assert window.runtime.fitting.current_1d_data is not None  # the former page follows (In-situ series)
    page.set_model(FitModel((new_component("cylinder", radius=4.0),)))
    components.fitting_workspace.insitu_series_page.capture_recipe_requested.emit()
    binding = window.runtime.fitting
    widget = binding._iter_particle_widget_ids()[0]
    assert binding.get_particle_shape(widget) == "Cylinder"
    assert binding._get_particle_parameter(widget, "R", 0.0) == pytest.approx(4.0)
    window.close()


def test_the_page_comes_back_with_the_model_and_the_curve_of_last_time(tmp_path) -> None:
    from src.gimap.features.fitting.application.single_fit import model_to_dict
    from src.gimap.features.fitting.presentation.single.page import PREFERENCES_KEY

    page = _page()
    path = _curve_file(tmp_path, signed=True)
    page.open_curve(path, "positive")
    page.set_range((0.3, 1.2))
    mine = FitModel((new_component("vertical_cylinder", radius=3.0),))
    page.set_model(mine)
    stored = page.preferences.get(PREFERENCES_KEY, {})
    assert stored["curve"]["side"] == "positive" and stored["model"] == model_to_dict(mine)
    page.dispose()

    again = _page()
    again.preferences = page.preferences
    again._restore()
    for _ in range(5):
        QApplication.processEvents()
    assert again.session.model == mine and again.session.edited  # the person's model, not a starting guess
    assert again.session.curve is not None and again.session.side == "positive"
    assert again.session.q_range == pytest.approx((0.3, 1.2))
    again.dispose()


def test_a_fit_goes_stale_when_the_range_or_the_left_out_points_change(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    page.set_range((0.2, 1.9))
    page.method_buttons["local"].setChecked(True)
    page.run_fit()
    _wait(page)
    assert page.session.result_is_current() and page.step_rail.state("fit") == "ok"
    assert page.model_editor.row((0, "R")).error.text().startswith("± ")
    page.set_range((0.5, 1.5))  # a band drag: the result stays, but no longer holds
    assert page.session.result is not None and not page.session.result_is_current()
    assert "fitting range or the left-out points changed after this fit" in page.warnings_label.text()
    assert "model was changed" not in page.warnings_label.text()
    assert page.step_rail.state("fit") == "warn" and page.model_editor.row((0, "R")).error.text() == ""
    page.export_data_dialog(str(tmp_path / "stale.csv"))
    assert "fit" not in json.loads((tmp_path / "stale.json").read_text(encoding="utf-8"))
    page.set_range((0.2, 1.9))  # the range of the fit again
    assert page.session.result_is_current() and page.step_rail.state("fit") == "ok"
    assert "changed after this fit" not in page.warnings_label.text()
    assert page.model_editor.row((0, "R")).error.text().startswith("± ")
    full = page.session.all_points()
    page.exclude_button.setChecked(True)
    page._clicked(float(full.q[100]), float(full.intensity[100]))  # a point left out
    assert not page.session.result_is_current() and page.step_rail.state("fit") == "warn"
    assert "left-out points changed" in page.warnings_label.text()
    page.export_data_dialog(str(tmp_path / "left_out.csv"))
    assert "fit" not in json.loads((tmp_path / "left_out.json").read_text(encoding="utf-8"))
    page._clicked(float(full.q[100]), float(full.intensity[100]))  # taken back
    assert page.session.result_is_current() and page.step_rail.state("fit") == "ok"
    page.export_data_dialog(str(tmp_path / "current.csv"))
    assert "fit" in json.loads((tmp_path / "current.json").read_text(encoding="utf-8"))
    page.dispose()


def test_saves_report_errors_and_leave_no_table_without_its_record(tmp_path, monkeypatch) -> None:
    from pathlib import Path

    from src.gimap.app.presentation.components.toast import visible_toasts

    page = _page()
    page.open_curve(_curve_file(tmp_path))

    def refused(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    with monkeypatch.context() as patch:
        patch.setattr(np, "savetxt", refused)
        assert page.export_data_dialog(str(tmp_path / "a.csv")) is None  # no exception, so no error dialog
    assert page.status_label.text().startswith("Could not save:") and "Permission denied" in page.status_label.text()
    assert not (tmp_path / "a.csv").exists() and not (tmp_path / "a.json").exists()
    write_text = Path.write_text

    def no_record(self, *args, **kwargs):
        if ".json" in self.name:  # the record (also under its temporary name)
            raise PermissionError(13, "Permission denied", str(self))
        return write_text(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", no_record)
        assert page.export_data_dialog(str(tmp_path / "b.csv")) is None
        assert page.save_model_dialog(str(tmp_path / "model.json")) is None
    assert not (tmp_path / "b.csv").exists() and not (tmp_path / "b.json").exists()  # the table went with its record
    assert page.status_label.text().startswith("Could not save:")
    assert page.export_plot_dialog(str(tmp_path / "missing" / "plot.png")) is None  # QImage.save says False
    assert page.status_label.text().startswith("Could not save:") and "Saved" not in page.status_label.text()
    assert page.export_data_dialog(str(tmp_path / "ok.csv")) == str(tmp_path / "ok.csv")
    assert (tmp_path / "ok.csv").exists() and (tmp_path / "ok.json").exists()
    saved = [toast for toast in visible_toasts(page.window()) if toast.text().startswith("Saved ok.csv")]
    assert saved and saved[0].action_button is not None and saved[0].action_button.text() == "Open Folder"
    pair = {name: (tmp_path / name).read_bytes() for name in ("ok.csv", "ok.json")}
    page.set_range((0.5, 1.5))  # a new table, over the earlier pair
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", no_record)
        assert page.export_data_dialog(str(tmp_path / "ok.csv")) is None
    assert {name: (tmp_path / name).read_bytes() for name in pair} == pair  # the earlier pair as it was
    replace = os.replace

    def record_in_use(source, target):
        if str(target).endswith(".json"):
            raise PermissionError(13, "The file is in use", str(target))
        return replace(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", record_in_use)  # the table takes its name, then the record cannot
        assert page.export_data_dialog(str(tmp_path / "ok.csv")) is None
        assert page.export_data_dialog(str(tmp_path / "new.csv")) is None
    assert {name: (tmp_path / name).read_bytes() for name in pair} == pair  # the earlier table is back
    assert not (tmp_path / "new.csv").exists() and not (tmp_path / "new.json").exists()
    left = sorted(path.name for path in tmp_path.iterdir() if path.suffix in (".part", ".old"))
    assert left == []  # no temporary file stays behind
    page.dispose()


def _shape_solution(radius: float) -> dict:
    """A row of the quick physical fit (Find the particle shape): a sphere of this radius."""
    from src.gimap.features.fitting.infrastructure.adapters.experimental_fit import forward

    params = {"R": radius, "sigma_R": 0.15, "D": 40.0, "sigma_D": 0.3}
    components = [{"type": "sphere", "type_id": 1, "weight": 1.0, "amplitude": 2000.0, "params": params}]
    globals_ = {"background": 3.0, "resolution_amplitude": 0.0, "sigma_Res": 0.02, "nu_Res": 3.0}
    return {"combination": "sphere", "components": components, "global_params": globals_,
            "native_q": Q_NM.tolist(), "native_fit": forward(Q_NM, components, globals_).tolist()}


def test_a_fit_that_ends_after_another_curve_was_opened_is_not_kept(tmp_path) -> None:
    first = _curve_file(tmp_path, name="a_fit_input.dat")
    q, intensity, sigma = np.loadtxt(first, unpack=True)
    second = tmp_path / "b_fit_input.dat"
    np.savetxt(second, np.column_stack([q, intensity * 7.0 + 50, sigma * 3]))  # another curve, the same q
    page = _page()
    page.open_curve(first)
    page.method_buttons["global"].setChecked(True)
    page.run_fit()
    assert page.fit_running()
    page.open_curve(second)  # Open Curve, Send to Fitting, a unit change … while the fit runs
    _wait(page)
    assert not page.fit_running() and page.session.curve.name == "b_fit_input.dat"
    assert page.session.result is None and not page.session.result_is_current()  # a's fit is not b's
    assert page.step_rail.state("fit") == "pending" and page.step_rail.state("results") == "pending"
    assert "fit of a_fit_input.dat ended after another curve was opened" in page.status_label.text()
    page.export_data_dialog(str(tmp_path / "b.csv"))
    assert "fit" not in json.loads((tmp_path / "b.json").read_text(encoding="utf-8"))
    page.dispose()

    page = _page(quick_fit=lambda *args, **kwargs: [_shape_solution(radius) for radius in (5.0, 6.0, 7.0)])
    page.open_curve(first)
    assert page.method() == "shapes"
    page.run_fit()
    page.open_curve(second)
    model = page.session.model
    _wait(page)
    assert page.session.solutions == [] and page.session.model == model  # a's solutions, χ² on a's points
    assert page.step_rail.state("fit") == "pending" and not page.solutions_table.isVisibleTo(page)
    page.dispose()


def test_find_the_particle_shape_shows_on_the_rail_and_its_row_is_selected(tmp_path) -> None:
    solution = _shape_solution
    page = _page(quick_fit=lambda *args, **kwargs: [solution(radius) for radius in (5.0, 5.5, 6.0, 6.5, 7.0)])
    page.open_curve(_curve_file(tmp_path))

    def cells(name: str) -> tuple:
        table = page.parameters_table
        line = next(line for line in range(table.rowCount()) if table.item(line, 0).text() == name)
        return table.item(line, 1).text(), table.item(line, 2).text()

    assert cells("R")[0].endswith(" nm")  # before any fit: the unit in the value column
    assert page.method() == "shapes"
    page.run_fit()
    _wait(page)
    solutions = page.session.solutions
    assert len(solutions) == 5 and page.session.result is None
    assert page.step_rail.state("fit") == "ok" and page.step_rail.state("results") == "ok"
    assert page.step_rail.detail("fit").startswith("5 solutions · best Sphere χ²ᵣ")
    assert page.step_rail.detail("results") == "5 solutions to compare"
    assert page.solutions_table.horizontalHeaderItem(4).text() == "χ²ᵣ"
    best = next(line for line, item in enumerate(solutions) if item.model == page.session.model)
    assert page.solutions_table.currentRow() == best and page.use_solution_button.isEnabled()
    page.run_fit()  # Refine the chosen solution
    _wait(page)
    value, error = cells("R")
    assert value.endswith(" nm") and error.startswith("± ") and not error.endswith("nm")
    page.dispose()


def test_the_page_has_no_open_shortcut_of_its_own_and_short_residual_labels(tmp_path) -> None:
    from pathlib import Path

    from PyQt5.QtWidgets import QShortcut

    from src.gimap.features.fitting.presentation.single import page as page_module

    assert "Ctrl+O" not in Path(page_module.__file__).read_text(encoding="utf-8")  # File ▸ Open Data does it
    page = _page()
    keys = {shortcut.key().toString() for shortcut in page.findChildren(QShortcut)}
    assert {"Ctrl+Z", "Ctrl+Y", "Ctrl+Return"} <= keys and "Ctrl+O" not in keys
    page.open_curve(_curve_file(tmp_path))
    assert page.residual_plot.figure_state()["y_label"] == "Δ/σ" and "(I − model)/σ" in page.residual_plot.toolTip()
    page.open_curve(_curve_file(tmp_path, sigma=False, name="no_sigma_fit_input.dat"))
    assert page.residual_plot.figure_state()["y_label"] == "Δ ln I" and "ln(I / model)" in page.residual_plot.toolTip()
    assert page.add_component_button.text() == "Add Particle"
    assert page.model_editor.row(("globals", "res_width")).name.text() == "Peak w (nm⁻¹)"
    page.dispose()


def test_a_parameter_name_is_translated_and_its_unit_kept() -> None:
    from src.gimap.app.presentation.i18n import translator
    from src.gimap.features.fitting.presentation.single.model_editor import FitModelEditor

    _app()
    state = translator()
    language, state.language = state.language, "zh"
    try:
        editor = FitModelEditor()
        editor.set_model(_truth())
        assert editor.row(("globals", "res_width")).name.text() == "峰 w (nm⁻¹)"
    finally:
        state.language = language


def test_points_are_left_out_by_a_click_or_a_box_and_taken_back(tmp_path) -> None:
    page = _page()
    page.open_curve(_curve_file(tmp_path))
    full = page.session.all_points()
    page.exclude_button.setChecked(True)
    target = full.q[100]
    page._clicked(float(target), float(full.intensity[100]))  # a click on a point
    assert page._data().q.size == 239 and page.left_out() == 1 and not page.excluded_row.isHidden()
    assert page.excluded_label.text().endswith(": 1")
    page._clicked(float(target), float(full.intensity[100]))  # again: back in
    assert page._data().q.size == 240 and page.excluded_row.isHidden()
    from PyQt5.QtCore import QRectF

    xs, ys = page._shown(full.q[10:20], full.intensity[10:20])
    page._exclude_box(QRectF(float(xs.min()) - 1e-9, float(ys.min()) - 1e-9,
                             float(xs.max() - xs.min()) + 2e-9, float(ys.max() - ys.min()) + 2e-9))
    assert page.left_out() >= 10 and page._data().q.size <= 230
    page.export_data_dialog(str(tmp_path / "fit.csv"))
    record = json.loads((tmp_path / "fit.json").read_text(encoding="utf-8"))
    assert len(record["points"]["left_out_q_nm^-1"]) == page.left_out()
    page.include_all()
    assert page._data().q.size == 240 and page.left_out() == 0
    page.dispose()
