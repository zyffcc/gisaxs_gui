"""The windows of the joint fit, the substrate level and the posterior check (offscreen):

* Model's grazing card: the substrate combo sets the model's substrate, the level line says the four-term DWBA once
  the angles are there; Predict uses the substrate from Model;
* Fit Several Curves Together: a horizontal and a vertical cut (the vertical one read from a file with Analyze's
  geometry in its observation line), the substrate, the parameters' Shared boxes, Fit Together in the background,
  the plot, Use in Model and Save Record…;
* the Posterior check tab: a Refine fit checked by a short MCMC run (table, trace, verdict, Save Check…), a model
  changed afterwards is said; Analyze ▸ Send Both Cuts to Fitting writes the vertical cut with its geometry.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace

import numpy as np
from PyQt5.QtWidgets import QApplication, QComboBox, QLabel

from src.gimap.features.fitting.application.joint_fit import CurveSpec, curve_errors, hints
from src.gimap.features.fitting.application.single_fit import Curve, FitModel, evaluate, new_component
from src.gimap.features.fitting.application.single_predict import predict_request
from src.gimap.shared.numbers import format_number
from tests.test_fit_page import _page, _wait

LAMBDA, ALPHA_I = 0.1045, 0.4


def _settle(page, dialog_tasks=None, limit: float = 180.0) -> None:
    end = time.monotonic() + limit
    tasks = dialog_tasks or page.tasks
    while tasks.is_busy() and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.01)
    tasks.wait(5)
    QApplication.processEvents()


def _cylinders() -> FitModel:
    component = new_component("vertical_cylinder", radius=4.0, distance=14.0)
    model = FitModel((component,)).with_values({(0, "sigma_R"): 0.15, (0, "sigma_nn"): 0.2, (0, "Int"): 2e4,
                                                ("globals", "background"): 3.0})
    return model.with_parameter(("globals", "res_amplitude"), value=0.0, free=False)


def test_substrate_combo_sets_the_level_and_predicts_substrate():
    page = _page()
    try:
        editor = page.model_editor
        model = FitModel((new_component("sphere", radius=4.0, structure=False),)).with_values(
            {("globals", "alpha_i"): ALPHA_I, ("globals", "alpha_f"): 0.15, ("globals", "wavelength"): LAMBDA})
        page.set_model(model)
        combo = editor.findChild(QComboBox, "fitSubstrate")
        line = editor.findChild(QLabel, "fitLevel")
        assert "Two paths" in line.text() or "Born" in line.text()
        combo.setCurrentIndex(combo.findData("Si"))
        QApplication.processEvents()
        assert page.session.model.substrate is not None and page.session.model.substrate.material == "Si"
        line = editor.findChild(QLabel, "fitLevel")
        assert "Four-term DWBA" in line.text() and "δ =" in line.text()
        q = np.linspace(0.1, 2.0, 30)
        data = type("Data", (), {"q": q, "intensity": np.ones_like(q), "sigma": None})()
        with_substrate = predict_request(data, page.session.model, "model").payload
        without = predict_request(data, replace(page.session.model, substrate=None), "model").payload
        assert with_substrate.pop("substrate") == {"material": "Si"}
        assert with_substrate == without
        assert "four-term DWBA on Si" in page.predict_substrate_note.text()
        assert page.reflected_beam.isHidden()
        combo.setCurrentIndex(combo.findData(""))
        QApplication.processEvents()
        assert page.session.model.substrate is None
    finally:
        page.dispose()
        page.deleteLater()


def _vertical_file(tmp_path, truth: FitModel) -> object:
    """A vertical cut as Analyze writes it: q_z in Å⁻¹ and its geometry in the observation line."""
    from src.gimap.features.fitting.application.joint_fit import curve_model, substrate_of

    k = 2 * math.pi / LAMBDA
    qz = k * (math.sin(math.radians(ALPHA_I)) + np.sin(np.radians(np.linspace(0.05, 1.3, 150))))
    spec = CurveSpec("v", qz, np.ones_like(qz), None, "vertical", alpha_i=ALPHA_I, wavelength=LAMBDA, q_parallel=0.3)
    model = curve_model(spec, truth, substrate_of("Si"), True).with_values({(0, "h"): 6.0, (0, "sigma_h"): 0.1,
                                                                            (0, "Int"): 5e3})
    y = np.random.default_rng(5).poisson(evaluate(model, qz)).astype(float)
    observation = {"source": "native_detector_rows", "cut": "vertical", "alpha_i_deg": ALPHA_I,
                   "wavelength_nm": LAMBDA, "q_parallel_nm": 0.3, "critical_angle_deg": 0.0,
                   "horizontal_alpha_f_deg": 0.15}
    path = tmp_path / "frame_fit_input_qz.dat"
    lines = ["# GIMaP Analyze fit input (q in 1/A)", "# curve: Vertical cut I(qz)",
             f"# observation: {json.dumps(observation)}", "# columns: qz (1/A)  I  sigma  pixels"]
    lines += [f"{a / 10:.8g} {b:.8g} {math.sqrt(max(b, 1.0)):.8g} 10" for a, b in zip(qz, y)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert hints(observation)["kind"] == "vertical"
    return path


def test_joint_window_fits_a_horizontal_and_a_vertical_cut(tmp_path):
    page = _page()
    try:
        truth = _cylinders()
        q = np.linspace(0.05, 2.5, 200)
        y = np.random.default_rng(4).poisson(evaluate(truth, q)).astype(float)
        page.session.set_curve(Curve.from_arrays(q, y, np.sqrt(np.maximum(y, 1)), name="horizontal", unit="nm"))
        page.set_model(truth.with_values({(0, "R"): 5.0, (0, "D"): 16.0}).with_values(
            {("globals", "alpha_i"): ALPHA_I, ("globals", "alpha_f"): 0.15, ("globals", "wavelength"): LAMBDA}))
        from src.gimap.features.fitting.presentation.single.joint_fit_dialog import JointFitDialog

        dialog = JointFitDialog(page, page.view_model, page)
        assert dialog.add_page_curve()
        assert dialog.add_file(_vertical_file(tmp_path, truth))
        assert [s.kind for s in dialog.specs] == ["horizontal", "vertical"]
        assert dialog.specs[1].q_parallel == 0.3 and dialog.specs[1].alpha_i == ALPHA_I
        assert not dialog.joint_run.isEnabled() and "Choose substrate" in dialog.joint_substrate.currentText()
        dialog.joint_shown.setCurrentIndex(1)
        assert dialog.joint_plot.has_curves()
        np.testing.assert_equal(dialog.joint_plot.figure_state()["curves"][0][1], dialog.specs[1].q)
        dialog.set_substrate("Si")
        assert dialog.joint_run.isEnabled()
        assert dialog.joint_min_exit.value() == 2
        dialog._render()
        names = [dialog.joint_parameters.item(i, 0).text() for i in range(dialog.joint_parameters.rowCount())]
        assert any(" h" in name for name in names)  # the vertical cut makes the cylinders' height a parameter
        dialog.joint_method.setCurrentIndex(dialog.joint_method.findData("local"))
        dialog.run()
        _settle(page, dialog.tasks)
        assert dialog.result is not None, dialog.joint_status.text()
        assert abs(dialog.result.models[0].get((0, "h")).value - 6.0) < 0.5
        assert "χ²ᵣ" in dialog.joint_summary.text()
        height = dialog.result.models[0].get((0, "h")).value
        assert f"h = {format_number(height, curve_errors(dialog.result, 0).get((0, 'h')))} nm" in dialog.joint_dimensions.text()
        dialog.joint_min_exit.setValue(0)  # the completed fit retains its own range
        assert "2 αc" in dialog.joint_summary.text()
        assert dialog.joint_shown.count() == 2
        dialog.joint_shown.setCurrentIndex(1)
        dialog.use_in_model()
        assert page.session.model.get((0, "R")).value == dialog.result.models[0].get((0, "R")).value
        assert not page.session.model.get((0, "h")).free
        assert page.model_editor.row((0, "h")) is not None  # Model shows the height the joint fit found
        saved = dialog.save_record_dialog(tmp_path / "joint.json")
        record = json.loads((tmp_path / "joint.json").read_text(encoding="utf-8"))
        assert saved and record["kind"] == "gimap_joint_fit_v1" and len(record["curves"]) == 2
        assert record["substrate"] == {"material": "Si"} and "h_1" in record["shared"]
        vertical = record["curves"][1]
        assert vertical["vertical_range"]["min_exit_angle_ac"] == 2 and vertical["points"] < len(dialog.specs[1].q)
        assert dialog.save_report_dialog(tmp_path / "joint.md")
        report = (tmp_path / "joint.md").read_text(encoding="utf-8")
        assert "2 αc" in report and "correlations between height and radius" in report
        assert "dilute particle layer" in report and "at a cutoff of 0 αc" in report
        assert record["height_range_check"]["heights"][0]["height"] == "h_1"
        dialog.result = replace(dialog.result, curve_chi2=(0.1, 10.0))
        dialog._summary()
        assert "Needs attention" in dialog.joint_summary.text() and "likely underestimated" in dialog.joint_summary.text()
        assert dialog.save_report_dialog(tmp_path / "poor-fit.md")
        assert "6–20%" in (tmp_path / "poor-fit.md").read_text(encoding="utf-8")
        dialog.run()  # use the chosen zero limit and the cylinder height now set in Model
        _settle(page, dialog.tasks)
        assert dialog.result.points[1] == len(dialog.specs[1].q)
        assert dialog.result.start_curves[1].model.get((0, "h")).value == page.session.model.get((0, "h")).value
        dialog.dispose()
    finally:
        page.dispose()
        page.deleteLater()


def test_one_particle_shape_can_be_chosen_in_the_joint_window(tmp_path):
    from src.gimap.app.presentation.i18n import apply_language
    from src.gimap.features.fitting.presentation.single.joint_fit_dialog import JointFitDialog

    page = _page()
    dialog = JointFitDialog(page, page.view_model, page)
    try:
        assert dialog.add_file(_vertical_file(tmp_path, _cylinders()))
        index = dialog.joint_family.findData("vertical_cylinder")
        dialog.joint_family.activated.emit(index)
        assert page.session.model.components[0].family == "vertical_cylinder"
        names = [dialog.joint_parameters.item(i, 0).text() for i in range(dialog.joint_parameters.rowCount())]
        assert any(" h (nm)" in name for name in names)
        assert not dialog.joint_run.isEnabled()
        dialog.remove_selected()
        assert not dialog.joint_plot.has_curves()
        dialog._save_text(tmp_path / "notes.md", "Test")
        apply_language("zh", [dialog.joint_dialog])
        assert dialog.joint_status.text() == "已保存 notes.md。"
    finally:
        apply_language("en", [dialog.joint_dialog])
        dialog.dispose()
        page.dispose()
        page.deleteLater()


def test_posterior_check_of_a_refine_fit(tmp_path):
    page = _page()
    try:
        truth = FitModel((new_component("sphere", radius=4.0, structure=False),)).with_values(
            {(0, "Int"): 1e4, ("globals", "background"): 5.0})
        truth = truth.with_parameter(("globals", "res_amplitude"), value=0.0, free=False)
        for key in ("res_width", "res_exponent"):
            truth = truth.with_parameter(("globals", key), free=False)
        q = np.geomspace(0.1, 2.5, 100)
        y = np.random.default_rng(1).poisson(evaluate(truth, q)).astype(float)
        page.session.set_curve(Curve.from_arrays(q, y, np.sqrt(np.maximum(y, 1)), name="spheres", unit="nm"))
        page.set_model(truth.with_values({(0, "R"): 4.4}))
        page.run_fit()
        _wait(page)
        assert page.session.result is not None
        page.show_uncertainty()
        page.render_check()
        assert page.check_run.isEnabled(), page.check_what.text()
        assert "Refine or Search" in page.check_what.text()
        page.check_steps.setValue(400)
        page.run_check()
        # The independent ensembles can still be sampling after the usual GUI wait.
        _settle(page, limit=600)
        assert not page.tasks.is_busy(), page.check_verdict.text()
        assert page.check_table.rowCount() == 2
        assert {page.check_table.item(i, 0).text() for i in range(2)} == {"R_1", "sigma_R_1"}
        assert "R̂" in page.check_verdict.text() and "check only" in page.check_verdict.text()
        assert page.check_trace_value.count() == 2
        assert page.save_check_dialog(tmp_path / "check.json")
        record = json.loads((tmp_path / "check.json").read_text(encoding="utf-8"))
        assert record["checked"] == "fit" and record["steps"] == 400 and len(record["values"]) == 2
        assert "split R̂" in page._report_text("text")
        page.session.set_range((0.2, 1.0))
        page.render_check()
        assert not page.check_run.isEnabled() and "data selection changed" in page.check_verdict.text()
        assert page.check_report_blocks() == []
        page.session.set_range(None)
        page.set_model(page.session.model.with_values({(0, "R"): 4.2}))
        page.render_check()
        assert "model changed" in page.check_verdict.text()
    finally:
        page.uncertainty_dialog.close()
        page.dispose()
        page.deleteLater()


def test_both_cuts_are_written_with_the_vertical_cut_geometry(tmp_path):
    from types import SimpleNamespace

    from src.gimap.features.analyze.application import both_cuts
    from src.gimap.features.analyze.domain.models import Curve as AnalyzeCurve
    from src.gimap.shared.geometry import DetectorGeometry

    geometry = DetectorGeometry(distance_m=2.0, pixel_size_x_m=172e-6, pixel_size_y_m=172e-6, beam_center_x_px=500.0,
                                beam_center_y_px=900.0, wavelength_angstrom=1.045, incidence_deg=ALPHA_I)
    x = np.linspace(0.01, 0.1, 20)
    horizontal = AnalyzeCurve("horizontal", "Horizontal cut I(qy)", x, x * 0 + 5, x * 0 + 1, np.ones(20, int),
                              "qy (Å⁻¹)", region={"rows": (783, 793), "center_row": 788.0})
    vertical = AnalyzeCurve("vertical", "Vertical cut I(qz)", x, x * 0 + 5, x * 0 + 1, np.ones(20, int), "qz (Å⁻¹)",
                            region={"columns": (590, 600), "center_column": 595.0})
    reduction = SimpleNamespace(curve=lambda key: {"horizontal": horizontal, "vertical": vertical}.get(key))
    analysis = SimpleNamespace(reduction=reduction, geometry=geometry, kind="gisaxs")
    found = both_cuts.cut_geometry(analysis)
    k = 2 * math.pi / 1.045
    expected_qy = k * math.sin(math.atan(95 * 172e-6 / 2.0)) * 10  # nm⁻¹, on the horizon (cos α_f ≈ 1)
    assert found["cut"] == "vertical" and abs(found["q_parallel_nm"] - expected_qy) < 0.01 * expected_qy
    assert found["wavelength_nm"] == 0.1045 and found["alpha_i_deg"] == ALPHA_I
    assert abs(found["horizontal_alpha_f_deg"] - (math.degrees(math.atan(112 * 172e-6 / 2.0)) - ALPHA_I)) < 1e-9
    assert both_cuts.VERTICAL_SUFFIX.endswith("_qz.dat") and not both_cuts.VERTICAL_SUFFIX.endswith("_fit_input.dat")


def test_both_cuts_of_a_public_frame_reach_the_joint_fit(tmp_path):
    import pytest

    from tests.test_analyze_workspace import GALAXI, _done, _public_page

    if not GALAXI.is_file():
        pytest.skip("public test frames (tests/data/external) are not available")
    from src.gimap.features.analyze.application.both_cuts import write_both_cuts
    from src.gimap.features.fitting.application import LoadCurveRequest
    from src.gimap.features.fitting.infrastructure.adapters.local_files import LocalCurveRepository

    page = _public_page()
    try:
        page.choose_mode("gisaxs")
        page.add_paths([str(GALAXI)])
        _done(page)
        model = page.view_model
        horizontal, vertical, geometry = write_both_cuts(model.frame_tools().export, model.state.analysis, tmp_path,
                                                         halves=model.fit_side)
        assert horizontal.name.endswith("_fit_input.dat") and vertical.name.endswith("_fit_input_qz.dat")
        loaded = LocalCurveRepository().load(LoadCurveRequest(vertical, "angstrom"))
        found = hints(loaded.observation)
        assert found["kind"] == "vertical" and found["alpha_i"] == geometry["alpha_i_deg"] == 0.463
        assert found["wavelength"] == 0.134 and found["q_parallel"] > 0
        assert found["band_crosses_specular"] is False
        assert np.all(np.asarray(loaded.q) > 0)
    finally:
        page.dispose()
        page.close()


def test_joint_fit_warns_for_a_band_that_covers_the_rod_even_with_its_centre_off_rod(tmp_path):
    from src.gimap.features.fitting.presentation.single.joint_fit_dialog import JointFitDialog

    page = _page()
    try:
        path = _vertical_file(tmp_path, _cylinders())
        path.write_text(path.read_text(encoding="utf-8").replace('"cut": "vertical"',
                        '"cut": "vertical", "band_crosses_specular": true'), encoding="utf-8")
        dialog = JointFitDialog(page, page.view_model, page)
        assert dialog.add_file(path)
        assert dialog.specs[0].q_parallel == 0.3
        assert "crosses the specular rod" in dialog.joint_status.text()
        dialog.remove_selected()
        assert not dialog.joint_status.text()
        # A curve without band metadata still warns on the centre, including a manually typed position.
        arrays = np.ones(5)
        dialog.add_spec(CurveSpec("No band", arrays, arrays, kind="vertical", q_parallel=0.0))
        assert "on or near the specular rod" in dialog.joint_status.text()
        dialog.joint_q_parallel.setValue(0.3)
        assert not dialog.joint_status.text()
        dialog.dispose()
    finally:
        page.dispose()
        page.deleteLater()
