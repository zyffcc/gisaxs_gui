"""After a manual fit: the correlation matrix, the χ² profile, the fit report, and the beam tail held to the curve
(``application.fit_uncertainty``, ``application.fit_report``, ``domain.fit_profile``, ``domain.beam_tail``)."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from src.gimap.features.fitting.application import fit_report
from src.gimap.features.fitting.application.fit_uncertainty import (
    chi2_profile,
    correlation,
    profile_values,
    strong_pairs,
    uncertainty_record,
)
from src.gimap.features.fitting.application.series_fit import FrameFit, flag_names, flagged_fit
from src.gimap.features.fitting.application.single_fit import (
    Curve,
    FitData,
    FitModel,
    evaluate,
    fit,
    fit_scales,
    new_component,
    tail_for_curve,
)
from src.gimap.features.fitting.domain import beam_tail

Q = np.geomspace(0.05, 2.0, 220)


def _sphere_curve(seed=0, tail=0.0):
    truth = FitModel((new_component("sphere", radius=6.0, distance=25.0),)).with_values(
        {(0, "Int"): 5000.0, (0, "sigma_R"): 0.12, (0, "sigma_nn"): 0.22, ("globals", "background"): 3.0,
         ("globals", "res_amplitude"): tail, ("globals", "res_width"): 0.1})
    counts = np.random.default_rng(seed).poisson(evaluate(truth, Q)).astype(float)
    return truth, FitData.prepare(Q, counts, np.sqrt(np.maximum(counts, 1.0)))


def test_the_correlation_matrix_comes_from_the_covariance():
    truth, data = _sphere_curve()
    result = fit(truth.with_values({(0, "R"): 5.6}), data)
    paths, matrix = correlation(result)
    covariance = np.array(result.covariance)
    assert paths == result.covariance_paths and matrix.shape == covariance.shape
    finite = np.isfinite(np.diag(covariance)) & (np.diag(covariance) > 0)
    assert np.allclose(np.diag(matrix)[finite], 1.0) and np.allclose(matrix, matrix.T, equal_nan=True)
    i, j = paths.index((0, "R")), paths.index((0, "sigma_R"))
    assert matrix[i, j] == pytest.approx(covariance[i, j] / math.sqrt(covariance[i, i] * covariance[j, j]))
    assert result.errors[(0, "R")] == pytest.approx(math.sqrt(covariance[i, i]))
    assert all(abs(r) > 0.9 for _a, _b, r in strong_pairs(result))
    assert strong_pairs(result, 0.0)[0][2] ** 2 >= strong_pairs(result, 0.0)[-1][2] ** 2
    record = uncertainty_record(result)
    assert record["correlation"]["parameters"][i] == "R_1" and len(record["covariance"]) == len(paths)


def test_the_profile_interval_agrees_with_the_covariance_error():
    truth, data = _sphere_curve(seed=3)
    result = fit(truth.with_values({(0, "R"): 5.8}), data)
    error = result.errors[(0, "R")]
    values = profile_values(result.model, (0, "R"), error, 11)
    assert values.min() == pytest.approx(result.model.get((0, "R")).value - 3 * error)
    profile = chi2_profile(result.model, data, (0, "R"), values)
    low, high = profile.interval()
    best = result.model.get((0, "R")).value
    assert low < best < high
    assert (high - low) / 2 == pytest.approx(error, rel=0.3)  # Δχ² = 1 ↔ 1σ for a well-determined value
    assert np.nanmin(profile.delta) > -0.1  # the others fitted again from the grid's neighbour: close to the best fit
    stopped = chi2_profile(result.model, data, (0, "R"), values, stop=lambda: True)
    assert stopped.stopped and np.all(np.isnan(stopped.totals))


def test_the_report_names_the_data_the_model_every_value_and_the_pairs(tmp_path):
    truth, data = _sphere_curve()
    result = fit(truth.with_values({(0, "R"): 5.6}), data)
    path = tmp_path / "sample_fit_input.dat"
    np.savetxt(path, np.column_stack([Q / 10, data.intensity, data.sigma]))
    curve = Curve.from_arrays(Q / 10, data.intensity, data.sigma, name=path.name, path=str(path))
    text = fit_report.report(result.model, data, curve, side="mean", q_range=(0.05, 2.0), result=result)
    for needle in ("# GIMaP fit report", "sample_fit_input.dat", "SHA-256", "Points fitted: 220", "χ²ᵣ",
                   "| 1·Sphere R (nm) | R_1 |", "Correlations (|r| > 0.9)", "Contact: particles do not overlap"):
        assert needle in text, needle
    r_error = f"{result.errors[(0, 'R')]:.3g}"
    assert r_error in text
    html = fit_report.report(result.model, data, curve, side="mean", q_range=None, result=result, fmt="html")
    assert html.startswith("<!doctype html>") and "<table>" in html and "R_1" in html
    plain = fit_report.report(result.model, data, curve, side="mean", q_range=None, result=result, fmt="text")
    assert "GIMaP fit report\n================" in plain
    assert fit_report.format_of("x.HTML") == "html" and fit_report.format_of("x.txt") == "text"
    assert fit_report.format_of("x.md") == "markdown"
    unfitted = fit_report.report(truth, data, None, side="mean", q_range=None, result=None)
    assert "No fit of these values yet" in unfitted


def test_the_beam_tail_of_a_manual_fit_stays_within_the_curve():
    """The curve has no tail; with the generic ranges a search used to trade A for a width far below the first point
    (A 10¹⁷–10²², errors NaN). With the ranges of the curve A stays below 10× the largest intensity."""
    truth, data = _sphere_curve(seed=5)
    start = FitModel((new_component("sphere", radius=5.0, distance=20.0),))
    assert beam_tail.is_generic(start)
    tied = tail_for_curve(start, data, start=True)
    limits = beam_tail.limits(start, data)
    assert tied.globals["res_amplitude"].upper == pytest.approx(10 * data.intensity.max())
    assert tied.globals["res_width"].lower == pytest.approx(0.3 * data.q.min())
    assert tied.globals["res_width"].upper == pytest.approx(min(30 * data.q.min(), data.q.max()))
    assert limits["res_exponent"] == (1.0, 6.0) and not beam_tail.is_generic(tied)
    assert tail_for_curve(tied, data) is tied  # ranges set once are the person's
    result = fit(fit_scales(tied, data), data, method="global", max_evaluations=1500)
    amplitude = result.model.globals["res_amplitude"]
    assert 0 <= amplitude.value <= amplitude.upper <= 10 * data.intensity.max()
    predicted = FitModel(start.components).with_parameter(("globals", "res_width"), lower=0.01, upper=0.2)
    assert tail_for_curve(predicted, data) is predicted  # not the generic ranges (a Predict solution): left alone


def test_a_runaway_tail_is_flagged_and_a_series_counts_it():
    truth, data = _sphere_curve(seed=5, tail=800.0)
    result = fit(tail_for_curve(truth, data), data)
    assert not beam_tail.tail_flags(result) and not beam_tail.flagged(result)
    g = result.model.globals
    at_top = replace(result, model=result.model.with_values({("globals", "res_amplitude"): g["res_amplitude"].upper}),
                     at_bounds=(("globals", "res_amplitude"),))
    assert beam_tail.tail_flags(at_top) == ((("globals", "res_amplitude"), "bound"),)
    blind = replace(result, errors={**result.errors, ("globals", "res_width"): math.nan})
    assert beam_tail.tail_flags(blind) == ((("globals", "res_width"), "undetermined"),) and beam_tail.flagged(blind)
    off = replace(blind, model=result.model.with_values({("globals", "res_amplitude"): 0.0}))
    assert not beam_tail.tail_flags(off)  # no tail: its shape changes nothing
    loose = replace(result, errors={**result.errors, (0, "sigma_nn"): 50 * g["res_width"].value + 100})
    assert (0, "sigma_nn") in beam_tail.undetermined(loose) and not beam_tail.flagged(loose)
    frames = [FrameFit(0, "a.dat", result), FrameFit(1, "b.dat", at_top), FrameFit(2, "c.dat", error="no data")]
    assert [flagged_fit(frame) for frame in frames] == [False, True, False]
    assert flag_names(frames[1]) == ["res_amplitude (bound)"]
