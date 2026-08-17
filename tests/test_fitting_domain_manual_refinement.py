import numpy as np
import pytest

from src.gimap.features.fitting.domain.manual_refinement import run_manual_refinement
from src.gimap.features.fitting.domain.models import (
    FittingParameterSet,
    ParameterValue,
)


def test_parameter_set_preserves_order_and_numeric_values():
    parameters = FittingParameterSet(
        (
            ParameterValue("Int1", 2.0, 0.0, 10.0),
            ParameterValue("BG", 0.5, 0.0, 1.0, scope="global"),
        )
    )

    assert parameters.names() == ("Int1", "BG")
    np.testing.assert_allclose(parameters.as_array(), [2.0, 0.5])
    assert parameters.as_dict() == {"Int1": 2.0, "BG": 0.5}


def test_manual_refinement_matches_fixed_linear_fixture():
    q = np.linspace(1.0, 4.0, 8)

    def model(x, scale, background):
        return scale * x + background

    setup = {
        "model_func": model,
        "q_model": q,
        "y": model(q, 3.0, 0.5),
        "params": [
            {"index": 0, "name": "scale", "value": 1.0},
            {"index": 1, "name": "background", "value": 0.2},
        ],
    }
    selected = [
        (setup["params"][0], 0.01, 10.0),
        (setup["params"][1], 0.01, 2.0),
    ]

    result = run_manual_refinement(
        setup,
        selected,
        {
            "max_nfev": 200,
            "ftol": 1e-12,
            "xtol": 1e-12,
            "gtol": 1e-12,
        },
    )

    np.testing.assert_allclose(result["params"], [3.0, 0.5], rtol=1e-5, atol=1e-5)
    assert result["final_log_rmse"] < result["initial_log_rmse"]
    assert result["stopped"] is False
