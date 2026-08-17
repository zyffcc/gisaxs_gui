import numpy as np
import pytest

from src.gimap.features.fitting.domain.constraints import (
    clamp_to_open_bounds,
    default_refine_bounds,
    default_refine_selected,
)
from src.gimap.features.fitting.domain.scoring import (
    chi_square,
    log_rmse,
    optimize_scale_factor,
)


def test_default_manual_bounds_match_controller_rules():
    assert default_refine_bounds("BG", 0.2) == (0.0, 2.0)
    assert default_refine_bounds("sigma_R2", 0.1) == (0.0, 1.0)
    assert default_refine_bounds("nu_Res", 5.0) == (0.1, 50.0)
    assert default_refine_selected("Int3")
    assert not default_refine_selected("R1")


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
