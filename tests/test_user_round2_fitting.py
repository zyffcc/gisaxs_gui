"""Refine / Search on curves a user test broke: a start that nothing tried could beat, a σ column with invalid
entries, fewer points than free values, a curve without a positive intensity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.gimap.features.fitting.application import curve_intake
from src.gimap.features.fitting.application.single_fit import (
    NO_POSITIVE_POINT,
    START_KEPT,
    TOO_FEW_FOR_FREE,
    Curve,
    FitData,
    FitModel,
    evaluate,
    fit,
    fit_resolving,
    new_component,
    prepare_points,
)

Q = np.linspace(0.08, 2.5, 200)


def _model(scale: float = 200.0) -> FitModel:
    model = FitModel((new_component("sphere", radius=6.0, distance=24.0),))
    for key in ("res_amplitude", "res_width", "res_exponent"):
        model = model.with_parameter(("globals", key), free=False)
    return model.with_values({(0, "Int"): scale, ("globals", "background"): 1.0})


def _counts(model: FitModel, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).poisson(evaluate(model, Q)).astype(float)


def test_refine_that_keeps_its_start_says_so_and_claims_no_errors():
    truth = _model()
    data = prepare_points(Curve.from_arrays(Q, _counts(truth), None, name="synthetic", unit="nm")).data
    assert data.sigma is None
    engine = fit(truth, data, method="local")
    assert engine.start_kept and not engine.converged and engine.message == START_KEPT
    assert engine.model is truth  # the values are the start's, unchanged
    result, _actions, _remaining = fit_resolving(truth, data, method="local")
    assert result.start_kept and not result.converged
    assert result.errors and all(math.isnan(value) for value in result.errors.values())
    assert result.correlated == ()


def test_refine_from_a_poor_start_still_converges_with_errors():
    truth = _model()
    sigma = np.sqrt(np.maximum(evaluate(truth, Q), 1.0))
    data = FitData.prepare(Q, evaluate(truth, Q) + 0.3 * sigma * np.sin(37.0 * Q), sigma)
    start = truth.with_values({(0, "R"): 5.2, (0, "D"): 21.0})
    result, _actions, _remaining = fit_resolving(start, data, method="local")
    assert result.converged and not result.start_kept
    assert math.isfinite(result.errors[(0, "R")])
    again, _actions, _remaining = fit_resolving(result.model, data, method="local")
    assert not again.start_kept and again.converged  # a start at the minimum is found again, not "kept unimproved"
    assert math.isfinite(again.errors[(0, "R")])


def test_state_text_of_a_kept_start_is_not_converged():
    from src.gimap.features.fitting.presentation.single.results import FitResultsMixin

    truth = _model()
    data = prepare_points(Curve.from_arrays(Q, _counts(truth), None, name="synthetic", unit="nm")).data
    result, _actions, _remaining = fit_resolving(truth, data, method="local")
    assert FitResultsMixin._state_text(result) == "did not improve on the start values"
    assert "converged" not in FitResultsMixin._rail_quality(result)


def test_points_with_an_invalid_sigma_are_left_out_and_counted():
    truth = _model(2000.0)
    intensity = evaluate(truth, Q)
    sigma = np.sqrt(intensity)
    sigma[[3, 50]] = 0.0
    sigma[77] = -1.0
    sigma[120] = np.nan
    made = prepare_points(Curve.from_arrays(Q, intensity, sigma, name="synthetic", unit="nm"))
    assert made.data.weighting == "sigma"
    assert made.invalid_sigma == 4 and made.points == Q.size - 4
    assert np.all(made.data.sigma > 0)
    notes = curve_intake.intake_notes(Q, level=None, points=made.points, invalid_sigma=made.invalid_sigma)
    assert [note.kind for note in notes] == ["sigma_invalid"]
    assert "4 points" in notes[0].text and str(Q.size - 4) in notes[0].text
    assert curve_intake.note_from_dict(notes[0].as_dict()) == notes[0]


def test_valid_and_absent_sigma_are_prepared_as_before():
    intensity = evaluate(_model(), Q)
    whole = FitData.prepare(Q, intensity, np.sqrt(intensity))
    assert whole.q.size == Q.size and whole.weighting == "sigma"
    assert FitData.prepare(Q, intensity, np.zeros_like(Q)).sigma is None  # no valid σ at all: ln I, as before
    assert prepare_points(Curve.from_arrays(Q, intensity, None, name="synthetic", unit="nm")).invalid_sigma == 0


def test_fewer_points_than_free_values_is_refused():
    model = _model()
    free = sum(1 for _path, parameter in model.parameters() if parameter.free)
    intensity = evaluate(model, Q)
    data = FitData.prepare(Q[:5], intensity[:5], np.sqrt(intensity[:5]))
    assert free >= 5
    with pytest.raises(ValueError) as raised:
        fit_resolving(model, data, method="local")
    assert str(raised.value) == TOO_FEW_FOR_FREE.format(points=5, free=free, minimum=free + 1)


def test_a_curve_without_a_positive_intensity_is_refused():
    model = _model()
    intensity = evaluate(model, Q)
    data = FitData.prepare(Q, -intensity, np.sqrt(intensity))
    assert data.q.size == Q.size
    with pytest.raises(ValueError, match="above zero"):
        fit_resolving(model, data, method="local")
    assert NO_POSITIVE_POINT


def test_refusals_are_said_in_chinese():
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.fitting.presentation.single.method_hint import _TOO_FEW

    for text in (TOO_FEW_FOR_FREE, NO_POSITIVE_POINT, START_KEPT, curve_intake.SIGMA_INVALID,
                 "did not improve on the start values"):
        assert text in ZH
    assert _TOO_FEW.fullmatch(TOO_FEW_FOR_FREE.format(points=5, free=9, minimum=10)).groupdict() == {
        "points": "5", "free": "9", "minimum": "10"}
