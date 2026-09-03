import numpy as np
import pytest

from src.gimap.features.fitting.domain.constraints import (
    clamp_to_open_bounds,
    default_global_search_bounds,
    default_global_search_selected,
    default_refine_bounds,
    default_refine_selected,
)
from src.gimap.features.fitting.domain.scoring import (
    chi_square,
    log_rmse,
    optimize_scale_factor,
)


def test_default_manual_bounds_match_controller_rules():
    assert default_refine_bounds("BG", 0.2) == pytest.approx((0.1, 0.3))
    assert default_refine_bounds("R1", 10.0) == pytest.approx((8.0, 12.0))
    assert default_refine_bounds("sigma_R2", 0.1) == pytest.approx((0.05, 0.15))
    assert default_refine_bounds("sigma_R2", 0.0) == pytest.approx((0.0, 0.1))
    assert default_refine_bounds("nu_Res", 5.0) == pytest.approx((3.75, 6.25))
    assert default_refine_selected("Int3")
    assert default_refine_selected("R1")
    assert not default_refine_selected("k")


def test_global_search_bounds_are_broad_and_use_curve_scale_for_background():
    observed = np.array([40.0, 50.0, 60.0, 1000.0])
    q_model = np.array([0.5, 1.0, 20.0])

    assert default_global_search_bounds("Int1", 2.0, observed, q_model) == pytest.approx(
        (0.0, 20000.0)
    )
    assert default_global_search_bounds("R1", 10.0, observed, q_model) == pytest.approx(
        (0.0025, 40.0)
    )
    lower, upper = default_global_search_bounds("BG", 0.01, observed, q_model)
    assert lower == 0.0
    assert upper == pytest.approx(95.0)
    assert default_global_search_selected("sigma_R2")
    assert default_global_search_selected("D3")
    assert default_global_search_selected("int_Res")
    assert not default_global_search_selected("k")


def test_application_refinement_defaults_forward_name_and_current_value():
    from src.gimap.features.fitting.application import ManualRefinementCalculations

    refinement = ManualRefinementCalculations()

    assert refinement.default_bounds("R1", 10.0) == pytest.approx((8.0, 12.0))
    assert refinement.default_global_bounds("R1", 10.0) == pytest.approx((0.001, 100.0))
    assert refinement.default_selected("R1")
    assert refinement.default_global_selected("sigma_R1")


def test_open_bound_clamp_preserves_legacy_epsilon():
    actual = clamp_to_open_bounds([0.0, 12.0], [0.0, 1.0], [10.0, 12.0])
    assert actual[0] == pytest.approx(1e-15)
    assert actual[1] == pytest.approx(12.0 - 1e-15)


def test_scores_and_scale_match_direct_legacy_formulas():
    observed = np.array([2.0, 4.0, 8.0])
    fitted = np.array([1.0, 2.0, 4.0])
    result = optimize_scale_factor(observed, fitted, current_scale=1.0)

    assert result.scale == pytest.approx(2.0)
    assert result.method == "Analytical"
    assert result.residual_after == pytest.approx(0.0)
    assert chi_square(observed, fitted) == pytest.approx(np.mean((observed - fitted) ** 2))
    assert log_rmse(observed, observed) == pytest.approx(0.0)
