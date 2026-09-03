from pathlib import Path

import numpy as np
import pytest

from src.gimap.features.fitting.domain.constraints import (
    default_global_search_bounds,
    default_global_search_selected,
)
from src.gimap.features.fitting.domain.manual_refinement import run_manual_refinement
from src.gimap.features.fitting.domain.models import (
    FittingParameterSet,
    ParameterValue,
)
from src.gimap.features.fitting.domain.scattering_model import (
    make_mixed_model,
    params_template,
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


def test_refinement_preserves_exact_current_value_when_it_is_best_at_a_bound():
    q = np.linspace(0.1, 1.0, 8)

    def model(x, offset):
        return np.ones_like(x) + offset * 1e15

    descriptor = {"index": 0, "name": "BG", "value": 0.0}
    setup = {
        "model_func": model,
        "q_model": q,
        "y": model(q, 0.0),
        "params": [descriptor],
    }

    result = run_manual_refinement(
        setup,
        [(descriptor, 0.0, 1.0)],
        {"max_nfev": 1, "ftol": 1e-8, "xtol": 1e-8, "gtol": 1e-8},
    )

    assert result["params"][0] == 0.0
    assert result["final_log_rmse"] == 0.0


def test_local_refinement_normalizes_parameters_with_very_different_scales():
    q = np.linspace(0.1, 1.0, 40)

    def model(x, tiny, large):
        return 2.0 + tiny * 1e8 * x + large * 1e-4 * x * x

    setup = {
        "model_func": model,
        "q_model": q,
        "y": model(q, 4e-8, 8e4),
        "params": [
            {"index": 0, "name": "tiny", "value": 1e-8},
            {"index": 1, "name": "large", "value": 2e4},
        ],
    }
    selected = [
        (setup["params"][0], 1e-10, 1e-7),
        (setup["params"][1], 1e3, 1e5),
    ]

    result = run_manual_refinement(
        setup,
        selected,
        {"max_nfev": 100, "ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12},
    )

    np.testing.assert_allclose(result["params"], [4e-8, 8e4], rtol=1e-6)
    assert result["final_log_rmse"] < 1e-10


def test_global_search_improves_repository_three_sphere_cut_fixture():
    parameter_values = [
        38.01233451271194,
        88.78584664178646,
        21.071814192393536,
        0.048323380489745014,
        0.12439633154137274,
        0.007436776145520525,
        22.26340427420722,
        1.7012413822164394,
        1.1020109126498318e-6,
        7.69537111141774e-9,
        0.0030352697702965272,
        0.320136068256,
        0.0869803899281294,
        0.862587903718,
        7.660391371032999,
        0.014302908780172206,
        0.005707001289000001,
        6.979183197953,
        3.28177e-7,
        1258486.088644592,
    ]
    names = params_template(["sphere", "sphere", "sphere"])
    data_path = Path(__file__).parents[1] / "TestSAXSdata" / "Cut_Data.txt"
    q, observed = np.loadtxt(data_path, skiprows=1).T
    valid = np.isfinite(q) & np.isfinite(observed) & (observed > 0)
    # Import 1D treats the q column as Angstrom^-1 and converts it to the
    # model's nm^-1 unit.  This is the actual GUI path used for this fixture.
    q, observed = q[valid] * 10.0, observed[valid]
    setup = {
        "shapes": ["sphere", "sphere", "sphere"],
        "model_func": make_mixed_model(["sphere", "sphere", "sphere"]),
        "q_model": q,
        "y": observed,
        "params": [
            {"index": index, "name": name, "value": value}
            for index, (name, value) in enumerate(zip(names, parameter_values))
        ],
    }
    selected = [
        (
            descriptor,
            *default_global_search_bounds(descriptor["name"], descriptor["value"], observed, q),
        )
        for descriptor in setup["params"]
        if default_global_search_selected(descriptor["name"])
    ]

    result = run_manual_refinement(
        setup,
        selected,
        {
            "mode": "global",
            "global_samples": 16384,
            "global_starts": 3,
            "random_seed": 1729,
            "max_nfev": 80,
            "ftol": 1e-8,
            "xtol": 1e-8,
            "gtol": 1e-8,
        },
    )

    assert result["mode"] == "global"
    assert result["initial_log_rmse"] == pytest.approx(0.17402910283186818)
    assert result["final_log_rmse"] < 0.05
    assert result["final_log_rmse"] < result["initial_log_rmse"] * 0.3
