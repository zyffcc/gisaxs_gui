"""The single-curve model and fitting engine: the manual model's curve, relative spreads, fits and errors."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.gimap.features.fitting.domain.fit_engine import FitData, FitStopped, fit, residuals
from src.gimap.features.fitting.domain.fit_model import (
    FitModel,
    evaluate,
    from_manual,
    model_from_dict,
    model_to_dict,
    new_component,
    to_manual,
)
from src.gimap.features.fitting.domain.scattering_model import make_mixed_model

Q = np.linspace(0.05, 2.0, 300)


def _truth(family="sphere") -> FitModel:
    model = FitModel((new_component(family, radius=6.0, distance=40.0),))
    values = {(0, "Int"): 1000.0, (0, "sigma_R"): 0.15, (0, "sigma_D"): 0.3,
              ("globals", "background"): 2.0, ("globals", "res_amplitude"): 300.0,
              ("globals", "res_width"): 0.03, ("globals", "res_exponent"): 4.0}
    if family == "cylinder":
        values.update({(0, "h"): 15.0, (0, "sigma_h"): 0.2})
    return model.with_values(values)


def _manual_curve(model: FitModel, q) -> np.ndarray:
    mapping = to_manual(model)
    order = {"Sphere": ("intensity", "radius", "sigma_radius", "diameter", "sigma_diameter"),
             "Cylinder": ("intensity", "radius", "sigma_radius", "height", "sigma_height", "diameter", "sigma_diameter"),
             "Vertical Cylinder": ("intensity", "radius", "sigma_radius", "diameter", "sigma_diameter")}
    params = []
    for component in mapping.components:
        params += [component.parameters[key] for key in order[component.shape]]
    g = mapping.global_parameters
    params += [g["background"], g["sigma_res"], g["nu_res"], g["int_res"], g["k_value"]]
    spec = [component.shape.lower().replace(" ", "_") for component in mapping.components]
    return make_mixed_model(spec)(q, *params)


@pytest.mark.parametrize("family", ["sphere", "cylinder", "vertical_cylinder"])
def test_the_model_is_the_manual_model_with_relative_spreads(family) -> None:
    model = _truth(family)
    np.testing.assert_allclose(evaluate(model, Q), _manual_curve(model, Q), rtol=1e-12)
    back = from_manual(to_manual(model))  # spreads: relative → nm → relative
    np.testing.assert_allclose(evaluate(back, Q), evaluate(model, Q), rtol=1e-12)
    assert back.components[0].value("sigma_R") == pytest.approx(0.15)


def test_a_model_is_saved_and_read_back() -> None:
    model = _truth("cylinder").with_parameter((0, "D"), free=False, lower=10.0, upper=90.0)
    again = model_from_dict(json.loads(json.dumps(model_to_dict(model))))
    assert again.get((0, "D")) == model.get((0, "D"))
    np.testing.assert_allclose(evaluate(again, Q), evaluate(model, Q))


def _noisy(model: FitModel, seed: int = 4) -> FitData:
    exact = evaluate(model, Q)
    sigma = np.sqrt(exact) + 0.5
    return FitData.prepare(Q, exact + np.random.default_rng(seed).normal(0.0, sigma), sigma)


def test_a_local_fit_finds_the_parameters_with_errors_that_cover_them() -> None:
    truth = _truth()
    data = _noisy(truth)
    start = truth.with_values({(0, "R"): 5.0, (0, "D"): 46.0, (0, "sigma_R"): 0.25, (0, "Int"): 1.0})
    result = fit(start, data, method="local")
    assert result.converged and result.weighting == "sigma"
    assert result.chi2_reduced == pytest.approx(1.0, abs=0.25)  # noise at the level of σ
    for key in ("R", "D", "sigma_R"):
        value, error = result.model.get((0, key)).value, result.errors[(0, key)]
        assert 0 < error < 0.2 * truth.get((0, key)).value
        assert abs(value - truth.get((0, key)).value) < 4 * error, (key, value, error)
    assert result.model.get((0, "Int")).value == pytest.approx(1000.0, rel=0.05)  # solved, not searched


def test_a_huge_parameter_does_not_wipe_out_the_errors_of_the_others() -> None:
    # In a series the resolution peak can run off to a vanishing width with an amplitude of 1e28 or more:
    # it must not set the finite-difference step (or the cut-off of the inverse) of R.
    from src.gimap.features.fitting.domain.fit_engine import _errors, _Problem

    truth = _truth().with_values({("globals", "res_amplitude"): 0.0})
    data = _noisy(truth)
    runaway = truth.with_values({("globals", "res_amplitude"): 1e28, ("globals", "res_width"): 0.0021,
                                 ("globals", "res_exponent"): 20.0})
    reference, _ = _errors(truth, data, _Problem(truth, data, None, None, 1))
    errors, _ = _errors(runaway, data, _Problem(runaway, data, None, None, 1))
    for key in ("R", "sigma_R", "Int"):
        assert reference[(0, key)] / 3 < errors[(0, key)] < 3 * reference[(0, key)], key


def test_a_wide_search_escapes_a_bad_start() -> None:
    truth = _truth()
    data = _noisy(truth, seed=9)
    start = truth.with_values({(0, "R"): 1.5, (0, "D"): 300.0})
    local = fit(start, data, method="local")
    wide = fit(start, data, method="global", max_evaluations=1500)
    assert wide.chi2_reduced < 1.5 and wide.chi2_reduced <= local.chi2_reduced
    assert wide.model.get((0, "R")).value == pytest.approx(6.0, rel=0.05)


def test_without_sigma_the_residuals_are_relative() -> None:
    truth = _truth()
    data = FitData.prepare(Q, evaluate(truth, Q))
    assert data.weighting == "relative"
    result = fit(truth.with_values({(0, "R"): 5.5}), data)
    assert result.log_rmse < 1e-4 and np.max(np.abs(residuals(result.model, data))) < 1e-3


def test_fixed_parameters_stay_and_a_stop_keeps_the_best_so_far() -> None:
    truth = _truth()
    data = _noisy(truth)
    start = truth.with_values({(0, "R"): 5.0}).with_parameter((0, "D"), free=False, value=41.0)
    result = fit(start, data)
    assert result.model.get((0, "D")).value == 41.0 and (0, "D") not in result.errors
    calls = []

    def stop():
        calls.append(1)
        return len(calls) > 5

    stopped = fit(start, data, method="global", stop=stop)
    assert stopped.stopped and not stopped.converged and "Stopped" in stopped.message
    with pytest.raises(ValueError, match="finite range"):
        fit(start.with_parameter((0, "R"), upper=math.inf), data)
    assert FitStopped.__name__ == "FitStopped"
