"""Refine goes on from its own result while χ²ᵣ falls (``single_fit.converged_again``): a least squares that stopped
early is run again, at most ``REFINE_RESTARTS`` times, and says how often (``FitResult.restarts``); a fit at its
minimum is returned as it was. The least squares behind a solution of Predict parameters is not run again, so its
numbers stay those of the engine (Predict's own refinement, ``posterior/refine.py``, calls the engine directly)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.gimap.features.fitting.application.fit_state import result_from_dict, result_to_dict
from src.gimap.features.fitting.application.series_fit import SeriesSettings, fit_frame
from src.gimap.features.fitting.application.single_fit import (
    REFINE_RESTARTS,
    FitData,
    converged_again,
    fit,
    fit_record,
    fit_resolving,
)
from tests.test_fit_series import _model
from tests.test_series_chain import _curve


def _data(radius: float = 5.0) -> FitData:
    curve = _curve(radius)
    return FitData.prepare(curve.q, curve.intensity, curve.sigma)


def test_a_least_squares_that_stopped_early_goes_on_from_its_own_result() -> None:
    data, start = _data(), _model(7.0)
    once = fit(start, data, method="local", max_evaluations=12)  # stopped by its budget, above the minimum
    result, _actions, _left = fit_resolving(start, data, method="local", max_evaluations=12)
    assert 1 <= result.restarts <= REFINE_RESTARTS
    assert result.chi2_reduced < once.chi2_reduced * (1 - 1e-3)
    assert result.evaluations > once.evaluations and result.method == "local"
    record = fit_record(result.model, data, None, side="mean", q_range=None, result=result)
    assert record["fit"]["restarts"] == result.restarts
    assert result_from_dict(result_to_dict(result)).restarts == result.restarts


def test_a_fit_at_its_minimum_is_kept_as_it_was() -> None:
    data = _data()
    done = fit(_model(5.1), data, method="local")
    again = converged_again(done, data)
    assert again is done and again.restarts == 0
    assert converged_again(replace(done, stopped=True), data).restarts == 0  # a stopped fit: as it is


def test_the_least_squares_after_predict_is_the_engine_s_own() -> None:
    curve = _curve(5.0, seed=7)
    data = FitData.prepare(curve.q, curve.intensity, curve.sigma)
    solution = _model(6.2)
    direct = fit(solution, data, method="local")
    frame = fit_frame(0, "a.dat", curve, solution, SeriesSettings(method="predict"), predict=lambda _data, _start: solution)
    assert frame.ok and frame.result.restarts == 0
    assert frame.result.chi2_reduced == direct.chi2_reduced
    values = lambda model: [parameter.value for _path, parameter in model.parameters()]  # noqa: E731
    assert np.array_equal(values(frame.result.model), values(direct.model))
