from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ML_ROOT = Path(__file__).resolve().parents[1] / "utils" / "ML_Fitting_1D_GISAXS"
sys.path.insert(0, str(ML_ROOT))

from TrainSetBuild import constraints, schema
from TrainSetBuild.physics_adapter import component_array_to_dict
from Training.predict_topk import (
    candidate_refine_setup,
    enforce_size_distribution_constraints,
    load_curve,
    score_from_metrics,
)
from Training.prediction_curve_io import adapt_input_to_model
from Training.prediction_refinement import estimated_function_evaluations


def test_comment_header_with_separate_unit_does_not_shift_intensity_column(tmp_path):
    curve = tmp_path / "curve.dat"
    rows = ["# q (1/A) I err"]
    rows.extend(f"{0.01 * i:g} {1000.0 / i:g} {10.0 / i:g}" for i in range(1, 17))
    curve.write_text("\n".join(rows), encoding="utf-8")

    q, intensity, sigma, _debug = load_curve(curve)

    assert len(q) == 16
    assert np.isclose(intensity[0], 1000.0)
    assert np.isclose(sigma[0], 10.0)


def test_candidate_width_projection_matches_geometry_semantics():
    cylinder = enforce_size_distribution_constraints(
        np.array([10.0, 20.0, 5.0, 8.0, 20.0, 30.0]),
        schema.TYPE_CYLINDER,
    )
    vertical = enforce_size_distribution_constraints(
        np.array([10.0, 2.0, 0.0, 0.0, 0.0, 3.0]),
        schema.TYPE_VERTICAL_CYLINDER,
    )

    assert np.isclose(cylinder[1], 9.0)
    assert np.isclose(cylinder[3], 4.5)
    assert np.isclose(cylinder[5], 18.0)
    assert np.isclose(vertical[1], 0.9)
    assert np.all(vertical[4:6] == 0.0)


def test_refinement_uses_user_parameter_and_global_bounds():
    cons = constraints.from_json_dict(
        {
            "mode": "free",
            "parameter_ranges": {"slot_0": {"R": [5.0, 8.0]}},
            "global_ranges": {"BG": [1e-4, 1e-2]},
        }
    )
    params = np.array([20.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    item = {
        "components": [component_array_to_dict(schema.TYPE_SPHERE, params, 1.0)],
        "global_phys": np.array([1.0, 0.01, 2.0, 0.0, 1.0]),
    }
    x0, lower, upper, _setup = candidate_refine_setup(item, cons=cons)
    expected_low = schema.normalize_value(5.0, schema.PARAM_NORM_RANGES["R"])
    expected_high = schema.normalize_value(8.0, schema.PARAM_NORM_RANGES["R"])
    assert np.isclose(lower[0], expected_low)
    assert np.isclose(upper[0], expected_high)
    assert lower[0] <= x0[0] <= upper[0]
    global_start = 2  # sphere R and sigma_R
    assert lower[global_start] <= x0[global_start] <= upper[global_start]


def test_hybrid_score_penalizes_narrow_linear_overshoot():
    visually_spiky = {"log_rmse": 0.72, "relative_rmse": 5.0}
    smoother = {"log_rmse": 0.79, "relative_rmse": 0.75}

    assert score_from_metrics(smoother, "hybrid_log_relative") < score_from_metrics(
        visually_spiky, "hybrid_log_relative"
    )


@pytest.mark.parametrize("weight", [0.0, 1e-12, 0.25])
def test_refinement_tiny_weight_starts_inside_existing_logit_bounds(weight):
    from scipy.optimize import least_squares

    params = np.array([20.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    item = {
        "components": [component_array_to_dict(schema.TYPE_SPHERE, params, weight),
                       component_array_to_dict(schema.TYPE_SPHERE, params, 1.0 - weight)],
        "global_phys": np.array([1.0, 0.01, 2.0, 0.0, 1.0]),
    }
    x0, lower, upper, setup = candidate_refine_setup(item)
    index = setup["weight_start"]
    assert lower[index] == -20.0 and upper[index] == 20.0
    assert x0[index] == max(-20.0, np.log(max(weight, 1e-12)))
    assert np.all(x0 >= lower) and np.all(x0 <= upper)
    result = least_squares(lambda x: x - x0, x0, bounds=(lower, upper), max_nfev=2)
    assert np.all(np.isfinite(result.fun))


def test_current_spacing_rule_vector_is_projected_for_legacy_checkpoint():
    model = SimpleNamespace(inputs=[SimpleNamespace(name="d_spacing_rule:0", shape=(None, 3))])
    current_batch = {"d_spacing_rule": np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)}

    adapted = adapt_input_to_model(current_batch, model)

    assert adapted["d_spacing_rule"].shape == (1, 3)
    np.testing.assert_array_equal(adapted["d_spacing_rule"], [[0.0, 1.0, 0.0]])


def test_legacy_checkpoint_rejects_new_spacing_rule_instead_of_silently_remapping():
    model = SimpleNamespace(inputs=[SimpleNamespace(name="d_spacing_rule:0", shape=(None, 3))])
    current_batch = {"d_spacing_rule": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)}

    with pytest.raises(RuntimeError, match="not supported by this older model"):
        adapt_input_to_model(current_batch, model)


def test_refine_stall_patience_uses_function_evaluation_scale():
    assert estimated_function_evaluations(0, 12) == 0
    assert estimated_function_evaluations(1, 12) == 1
    assert estimated_function_evaluations(13, 12) == 1
    assert estimated_function_evaluations(14, 12) == 2
