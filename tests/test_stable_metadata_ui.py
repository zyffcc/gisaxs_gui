"""Stable routing and CBF counting provenance, without loading neural weights."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.features.fitting.application.workflow_v5 import default_options
from src.gimap.features.fitting.domain.cbf_observations import column_observations
from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import prepare_sides
from src.gimap.features.fitting.presentation.bindings.ai_job_execution import (
    workflow_options_for_mode,
)
from src.gimap.features.fitting.presentation.bindings.insitu_cut_processing import (
    InsituCutProcessingMixin,
)
from src.gimap.features.fitting.presentation.bindings.workflow_v5_binding import (
    WorkflowV5BindingMixin,
)


def test_cbf_counts_are_actual_selected_pixels_including_zero_observations():
    image = np.full((3, 20), 9.0)
    image[:, 4] = np.nan
    image[0, 6] = np.nan
    image[:, 7] = 0
    q_mesh = np.broadcast_to(np.arange(20), image.shape)
    selected = np.ones_like(image, dtype=bool)
    selected[1:, 8] = False
    q, y, sigma, metadata = column_observations(
        image, q_mesh, (0, 2, 0, 19), selection_mask=selected
    )
    counts = np.asarray(metadata["valid_pixel_counts"])
    assert counts.shape == q.shape
    assert counts[q == 6] == 2
    assert counts[q == 8] == 1
    assert counts[q == 7] == 3 and y[q == 7] == 0
    assert sigma[q == 7] == 1 / 3
    assert 4 not in q
    assert metadata["intensity_unit"] == "counts_per_pixel"


def test_curve_selection_filters_counts_with_exact_observation_mask():
    binding = WorkflowV5BindingMixin()
    binding.ui = SimpleNamespace(fitCurrentDataCheckBox=SimpleNamespace(isChecked=lambda: True))
    binding.fitting_view_model = SimpleNamespace(insitu=SimpleNamespace(recipe=None))
    binding.current_cut_data = {}
    binding._workflow_options = default_options
    binding._workflow_input_selection = lambda: dict(
        axis_filter="all", roi=[-2.5, 3.1], excluded_q=["2.0"]
    )
    binding._ai_q_key = lambda value: str(float(value))
    binding._convert_q_values_for_model = lambda q, **_: q
    q = np.arange(-3, 4, dtype=float)

    def native(_):
        binding._workflow_observation_metadata = dict(valid_pixel_counts=list(range(2, 9)))
        return dict(x_coords=q, y_intensity=np.ones(7), err=np.ones(7))

    binding._native_cbf_input = native
    selected_q, _, _ = binding._current_ai_curve_arrays()
    np.testing.assert_array_equal(selected_q, [-1, 1, 3])
    assert binding._workflow_observation_metadata["valid_pixel_counts"] == [4.0, 6.0, 8.0]
    assert binding._workflow_observation_metadata["selected_points"] == 3


def test_side_sort_indices_align_count_metadata_without_sigma_inference():
    q = np.r_[np.arange(-8, 0), np.arange(8, 0, -1)] * 0.1
    counts = np.arange(2, 18)
    sides = prepare_sides(q, np.ones(16), np.ones(16) * 100, default_options())
    for item in sides:
        np.testing.assert_allclose(item["q"], np.arange(1, 9) * 0.1)
        expected = counts[8:][::-1] if item["sign"] == 1 else counts[:8][::-1]
        np.testing.assert_array_equal(counts[item["indices"]], expected)


@pytest.mark.parametrize("method", ["stable", "model", "experimental"])
def test_main_fit_preserves_saved_method(method):
    saved = {**default_options(), "method": method}
    assert workflow_options_for_mode(saved, "full")["method"] == method
    assert saved["method"] == method
    neural = workflow_options_for_mode(saved, "fast")
    assert neural["method"] == "model" and neural["numerical"] is False


@pytest.mark.parametrize(
    "method,numerical,expected",
    [("stable", False, "stable"), ("stable", True, "stable"),
     ("experimental", False, "experimental"), ("model", False, "fast"),
     ("model", True, "full")],
)
def test_insitu_routes_captured_method_without_legacy_coercion(method, numerical, expected):
    binding = InsituCutProcessingMixin()
    binding._insitu_workflow_settings = lambda: dict(
        use_previous=False, full_auto_fit=True, auto_refine=False
    )
    binding._log_insitu_workflow = lambda *args: None
    binding.ui = SimpleNamespace()
    binding.fitting_view_model = SimpleNamespace(
        insitu=SimpleNamespace(recipe=SimpleNamespace(
            model={"workflow_v5": dict(method=method, numerical=numerical)}
        ))
    )
    calls = []

    def start(mode):
        calls.append(mode)
        binding._ai_job_thread = object()

    binding._start_ai_prediction = start
    binding._run_insitu_workflow_fit({})
    assert calls == [expected]


def test_dialog_defaults_general_and_preserves_explicit_saved_methods():
    from PyQt5.QtWidgets import QApplication
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    app = QApplication.instance() or QApplication([])
    default = WorkflowV5Dialog()
    assert default.options()["method"] == "model"
    assert default.method.currentText() == "General V5 (experimental)"
    assert default.numerical.isEnabled()
    assert not default.amplitude_calibration.isEnabled()
    assert default.options()["amplitude_calibration"] is True
    default.method.setCurrentIndex(default.method.findData("stable"))
    assert default.method.currentText() == "Single RC specialist (experimental)"
    assert "known" in default.method.toolTip() or "Complete composition" in default.method.toolTip()
    assert default.amplitude_calibration.isEnabled()
    default.fix_sigma.setChecked(True)
    default.sigma_res.setValue(0.025)
    default.fix_nu.setChecked(True)
    default.nu_res.setValue(2.6)
    assert default.options()["sigma_res"] == 0.025
    assert default.options()["nu_res"] == 2.6
    default.close()
    for method in ("model", "stable", "experimental"):
        dialog = WorkflowV5Dialog(options=dict(method=method))
        assert dialog.options()["method"] == method
        assert dialog.numerical.isEnabled() is (method == "model")
        assert dialog.amplitude_calibration.isEnabled() is (method == "stable")
        dialog.close()
    app.processEvents()


def test_amplitude_calibration_is_independent_of_legacy_numerical_setting():
    from PyQt5.QtWidgets import QApplication
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    app = QApplication.instance() or QApplication([])
    dialog = WorkflowV5Dialog(options=dict(method="stable", numerical=False))
    assert not dialog.options()["numerical"]
    assert dialog.options()["amplitude_calibration"] is True
    dialog.amplitude_calibration.setChecked(False)
    assert dialog.options()["amplitude_calibration"] is False
    dialog.method.setCurrentIndex(dialog.method.findData("model"))
    assert not dialog.amplitude_calibration.isEnabled()
    dialog.numerical.setChecked(True)
    dialog.method.setCurrentIndex(dialog.method.findData("stable"))
    assert dialog.amplitude_calibration.isEnabled()
    assert dialog.options()["amplitude_calibration"] is False
    dialog.close()
    saved = WorkflowV5Dialog(options=dict(method="stable", amplitude_calibration=False))
    assert saved.options()["amplitude_calibration"] is False
    saved.close()
    app.processEvents()


@pytest.mark.parametrize(
    "method,label", [("stable", "Single RC specialist (experimental"), ("model", "General V5 (experimental)"),
                     ("experimental", "Numerical physical")]
)
def test_workspace_status_identifies_selected_method(method, label):
    binding = WorkflowV5BindingMixin()
    binding._workflow_options = lambda: {**default_options(), "method": method}
    statuses = []
    binding._set_ai_workspace_status = lambda message, _: statuses.append(message)
    binding._refresh_ai_fitting_models()
    assert label in statuses[0]


def test_recipe_preserves_scientific_geometry_precision_and_deliberate_edits():
    from PyQt5.QtWidgets import QApplication, QDoubleSpinBox
    from src.gimap.features.fitting.domain.detector_settings import DetectorSettings
    from src.gimap.features.fitting.presentation.bindings.insitu_recipe_binding import (
        _recipe_detector_geometry,
    )

    app = QApplication.instance() or QApplication([])
    exact = DetectorSettings(
        distance=1456.712345, grazing_angle=0.4123456, wavelength=0.103312345,
        beam_center_x=791.3190849, beam_center_y=370.753456,
        pixel_size_x=172.012345, pixel_size_y=172.023456,
    )
    controls = {}
    fields = (
        ("distance", "distance_spinbox", 1),
        ("grazing_angle", "angle_spinbox", 3),
        ("wavelength", "wavelength_spinbox", 4),
        ("beam_center_x", "beam_center_x_spinbox", 2),
        ("beam_center_y", "beam_center_y_spinbox", 2),
        ("pixel_size_x", "pixel_size_x_spinbox", 1),
        ("pixel_size_y", "pixel_size_y_spinbox", 1),
    )
    for attribute, name, decimals in fields:
        control = QDoubleSpinBox()
        control.setRange(0, 20000)
        control.setDecimals(decimals)
        control.setValue(getattr(exact, attribute))
        controls[name] = control
    panel = SimpleNamespace(**controls)
    panel.current_settings = lambda: DetectorSettings(**{
        attribute: controls[name].value() for attribute, name, _ in fields
    })
    assert panel.current_settings().beam_center_x == 791.32
    captured = _recipe_detector_geometry(panel, exact)
    assert captured == dict(
        distance_mm=exact.distance, grazing_angle_deg=exact.grazing_angle,
        wavelength_nm=exact.wavelength, beam_center_x_px=exact.beam_center_x,
        beam_center_y_px=exact.beam_center_y, pixel_size_x_um=exact.pixel_size_x,
        pixel_size_y_um=exact.pixel_size_y,
    )
    controls["beam_center_x_spinbox"].setValue(792.12)
    edited = _recipe_detector_geometry(panel, exact)
    assert edited["beam_center_x_px"] == 792.12
    assert edited["wavelength_nm"] == exact.wavelength
    controls["wavelength_spinbox"].setValue(0.1045)
    edited = _recipe_detector_geometry(panel, exact)
    assert edited["wavelength_nm"] == 0.1045
    assert _recipe_detector_geometry(panel, None)["beam_center_y_px"] == 370.75
    for control in controls.values():
        control.close()
    app.processEvents()


def test_stable_stage_labels_explain_fallback_without_changing_exports(tmp_path):
    from PyQt5.QtWidgets import QApplication
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    app = QApplication.instance() or QApplication([])
    dialog = WorkflowV5Dialog()
    stages = (
        ("stable_amplitude_calibrated", "Model + amplitude"),
        ("stable_neural", "Neural model"),
        ("stable_numerical_fallback", "Numerical fallback"),
    )
    rows = [dict(
        file="test", side="positive", rank=i + 1, combination="random_cylinder",
        best_source=stage, best_log_rmse=0.2, signed_weighted_rms=1,
        native_q=[1, 2], observed=[1, 1], sigma=[0.1, 0.1],
        display_q=[1, 2], display_fit=[1, 1],
        fallback_reason="Fixed resolution requires numerical fitting" if i == 2 else None,
        warnings=["Particle parameters may be non-unique"],
    ) for i, (stage, _) in enumerate(stages)]
    dialog.set_results(rows, tmp_path)
    for i, (stage, label) in enumerate(stages):
        cell = dialog.table.item(i, 5)
        assert cell.text() == label
        assert stage in cell.toolTip()
        assert "non-unique" in cell.toolTip()
        assert dialog.rows[i]["best_source"] == stage
    assert "Fixed resolution" in dialog.table.item(2, 5).toolTip()
    dialog.close()
    app.processEvents()


def test_insitu_mode_labels_do_not_promise_legacy_execution_for_stable():
    from PyQt5.QtWidgets import QApplication
    from src.gimap.features.fitting.presentation.views.insitu_workflow_controls import (
        InSituWorkflowControls,
    )

    app = QApplication.instance() or QApplication([])
    controls = InSituWorkflowControls(None)
    assert controls.workflowModeCombo.itemText(0) == "Fit curves · selected method"
    assert controls.workflowModeCombo.itemText(1) == "Fit curves · legacy correction off"
    assert "single-RC specialist" in controls.workflowModeCombo.toolTip()
    for index in (0, 1):
        controls.workflowModeCombo.setCurrentIndex(index)
        assert controls.autoFitCheckBox.isChecked()
        assert controls.fullAutoFitCheckBox.isChecked()
    controls.workflowModeCombo.setCurrentIndex(2)
    assert not controls.autoFitCheckBox.isChecked()
    controls.close()
    app.processEvents()
