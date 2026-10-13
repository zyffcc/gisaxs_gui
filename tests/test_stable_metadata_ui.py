"""Stable routing and CBF counting provenance, without loading neural weights."""

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from src.gimap.features.fitting.application.workflow_v5 import default_options
from src.gimap.features.fitting.infrastructure.adapters.workflow_v5 import prepare_sides
from src.gimap.features.fitting.presentation.bindings.ai_job_execution import (
    workflow_options_for_mode,
)
from src.gimap.features.fitting.presentation.bindings.insitu_curve_processing import (
    InsituCurveProcessingMixin,
)
from src.gimap.features.fitting.presentation.bindings.workflow_v5_binding import (
    WorkflowV5BindingMixin,
)


def test_curve_selection_filters_counts_with_exact_observation_mask():
    binding = WorkflowV5BindingMixin()
    binding.ui = SimpleNamespace()
    binding.fitting_view_model = SimpleNamespace(insitu=SimpleNamespace(recipe=None))
    q = np.arange(-3, 4, dtype=float)
    binding.current_1d_data = {  # native detector columns written by Analyze
        "q": q,
        "I": np.ones(7),
        "err": np.ones(7),
        "pixels": np.arange(2, 9, dtype=float),
        "observation": {"source": "native_detector_columns", "file_format": "cbf"},
        "q_source_unit": "angstrom",
    }
    binding._workflow_options = default_options
    binding._workflow_input_selection = lambda: dict(
        axis_filter="all", roi=[-2.5, 3.1], excluded_q=["2.0"]
    )
    binding._ai_q_key = lambda value: str(float(value))
    binding._convert_q_values_for_model = lambda q, **_: q
    selected_q, _, _ = binding._current_ai_curve_arrays()
    np.testing.assert_array_equal(selected_q, [-1, 1, 3])
    assert binding._workflow_observation_metadata["valid_pixel_counts"] == [4.0, 6.0, 8.0]
    assert binding._workflow_observation_metadata["selected_points"] == 3


@pytest.mark.parametrize(
    "axis_filter,expected_q",
    [("all", [-3, -2, -1, 1, 2, 3]), ("negative", [-3, -2, -1]), ("positive", [1, 2, 3])],
)
def test_folded_views_select_the_fitting_range_in_abs_q(axis_filter, expected_q):
    """|q| overlay, ±q average and −q as |q| show the range in |q|: both signs are kept."""
    binding = WorkflowV5BindingMixin()
    binding.ui = SimpleNamespace()
    binding.fitting_view_model = SimpleNamespace(insitu=SimpleNamespace(recipe=None))
    q = np.arange(-3, 4, dtype=float)
    binding.current_1d_data = {
        "q": q,
        "I": np.ones(7),
        "err": np.ones(7),
        "pixels": np.full(7, 5.0),
        "observation": {"source": "native_detector_columns", "file_format": "cbf"},
        "q_source_unit": "angstrom",
    }
    binding._workflow_options = default_options
    binding._workflow_input_selection = lambda: dict(
        axis_filter=axis_filter, roi=[0.5, 3.5], roi_abs=True, excluded_q=[]
    )
    binding._ai_q_key = lambda value: str(float(value))
    binding._convert_q_values_for_model = lambda q, **_: q
    selected_q, _, _ = binding._current_ai_curve_arrays()
    np.testing.assert_array_equal(selected_q, expected_q)
    assert binding._workflow_observation_metadata["selected_points"] == len(expected_q)


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
    binding = InsituCurveProcessingMixin()
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
    assert controls.workflowModeCombo.itemText(2) == "Plot curves only"
    controls.close()
    app.processEvents()
