"""The MCMC check (``domain.mcmc``, ``domain.posterior_check``, ``application.posterior_check``):

* the ensemble against a correlated Gaussian (its quantiles), the autocorrelation time of an AR(1) series against
  (1 + φ)/(1 − φ), split R̂ of chains that disagree, a stopped run keeping its steps;
* the check of a manual fit: its density at the fit's values, intervals close to the fit's errors on a curve whose
  posterior is nearly Gaussian;
* Predict's posterior of a solution: the density is that of ``posterior.infer.Likelihood`` and the prior, the values
  reported leave out the model error, a short run is not called converged; the rows, verdict and record of the window.

A longer comparison (a grid posterior, Predict's own importance sampling): ``tools/benchmarks/mcmc_check.py``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.gimap.features.fitting.application import posterior_check as app
from src.gimap.features.fitting.domain import fit_engine as fe
from src.gimap.features.fitting.domain import fit_model as fm
from src.gimap.features.fitting.domain import mcmc
from src.gimap.features.fitting.domain import posterior_check as pc
from src.gimap.features.fitting.domain.posterior import encoding as E
from src.gimap.features.fitting.domain.posterior.infer import Likelihood, physical
from src.gimap.features.fitting.domain.posterior.refine import model_values


def test_ensemble_samples_a_correlated_gaussian():
    rng = np.random.default_rng(0)
    cov = np.array([[1.0, 0.9, 0.0], [0.9, 1.0, 0.2], [0.0, 0.2, 4.0]])
    precision = np.linalg.inv(cov)
    chain = mcmc.sample(lambda x: -0.5 * np.einsum("bi,ij,bj->b", x, precision, x), 0.1 * rng.normal(size=(16, 3)),
                        2500, rng)
    found = mcmc.quantiles(chain.kept().reshape(-1, 3))
    exact = np.array([[0.0, -1.0, 1.0, -1.959964, 1.959964]]).T * np.sqrt(np.diag(cov))[None, :]
    assert np.max(np.abs(found - exact) / np.sqrt(np.diag(cov))) < 0.25
    diagnostics = mcmc.diagnose(chain)
    assert diagnostics.converged and max(diagnostics.rhat) < 1.05 and 0.15 < diagnostics.acceptance < 0.7


def test_autocorrelation_time_and_split_rhat():
    rng = np.random.default_rng(1)
    phi, n, walkers = 0.9, 20000, 8
    series = np.zeros((n, walkers))
    noise = rng.normal(size=(n, walkers))
    for t in range(1, n):
        series[t] = phi * series[t - 1] + noise[t]
    assert mcmc.autocorrelation_time(series) == pytest.approx((1 + phi) / (1 - phi), rel=0.15)
    agree = rng.normal(size=(500, 8))
    apart = agree + np.arange(8)[None, :]
    assert mcmc.split_rhat(agree) < 1.02 and mcmc.split_rhat(apart) > 1.5
    assert mcmc.walkers_for(3) == 16 and mcmc.walkers_for(9) == 36 and mcmc.walkers_for(9) % 2 == 0


def test_a_stopped_run_keeps_its_steps_and_bad_starts_are_refused():
    rng = np.random.default_rng(2)
    calls = []
    chain = mcmc.sample(lambda x: -0.5 * np.sum(x * x, axis=1), rng.normal(size=(8, 2)), 100, rng,
                        progress=lambda done, total: calls.append(done), stop=lambda: len(calls) >= 10)
    assert chain.stopped and chain.steps == 10 and calls[-1] == 10
    with pytest.raises(ValueError):
        mcmc.sample(lambda x: np.full(x.shape[0], -np.inf), rng.normal(size=(8, 2)), 10, rng)
    with pytest.raises(ValueError):
        mcmc.sample(lambda x: -np.sum(x * x, axis=1), rng.normal(size=(7, 2)), 10, rng)


def _sphere_fit():
    rng = np.random.default_rng(1)
    q = np.geomspace(0.1, 2.5, 60)
    truth = fm.FitModel((fm.new_component("sphere", radius=4.0, structure=False),)).with_values(
        {(0, "Int"): 1e4, ("globals", "background"): 5.0})
    truth = truth.with_parameter(("globals", "res_amplitude"), value=0.0, free=False)
    for key in ("res_width", "res_exponent"):
        truth = truth.with_parameter(("globals", key), free=False)
    y = rng.poisson(fm.evaluate(truth, q)).astype(float)
    data = fe.FitData.prepare(q, y, np.sqrt(np.maximum(y, 1.0)))
    return fe.fit(truth.with_values({(0, "R"): 4.4}), data), data, (q, y)


def test_check_of_a_manual_fit_matches_its_errors():
    fit, data, _curve = _sphere_fit()
    target = pc.FitTarget(fit.model, data)
    assert target.names == ["R_1", "sigma_R_1"]
    residual = fe.residuals(fit.model, data)
    t = np.clip(target.problem.start(), 1e-6, 1 - 1e-6)
    expected = -0.5 * float(residual @ residual) / target.variance + float(np.sum(np.log(t) + np.log1p(-t)))
    assert target(target.z_star[None, :])[0] == pytest.approx(expected, rel=1e-9)
    assert target.variance == pytest.approx(fit.chi2_reduced, rel=1e-6)
    result = pc.run_check(target, steps=1000, seed=3)
    for index, path in enumerate(target.paths):
        half = 0.5 * (result.quantiles[2, index] - result.quantiles[1, index])
        assert half == pytest.approx(fit.errors[path], rel=0.3)
        assert abs(result.quantiles[0, index] - fit.model.get(path).value) < fit.errors[path]


def test_predict_target_is_predicts_posterior():
    fit, _data, (q, y) = _sphere_fit()
    composition = E.composition_index([("sphere", False)])
    centre = model_values(fit.model, composition)
    target = pc.PredictTarget(q, y, np.sqrt(np.maximum(y, 1.0)), composition=composition, centre=centre)
    assert "model_error" in target.names and "model_error" not in target.reported
    assert "reflection_re" not in target.names  # no two paths without both angles
    z = target.z_star[None, :]
    full = np.zeros((1, E.DIMS))
    full[:, target.dims] = z
    values = physical(E.from_z(full, [target.data.ratio], composition), target.data.q_max)
    log_like = Likelihood(target.data, composition)(values)[0]
    assert target(z)[0] == pytest.approx(float(log_like[0] + E.log_prior(full, composition)[0]), rel=1e-12)
    assert target.best[target.names.index("R_1")] == pytest.approx(fit.model.get((0, "R")).value, rel=1e-6)
    assert np.all(pc.widths(target) > 0)
    result = pc.run_check(target, steps=60, seed=1)
    assert not result.diagnostics.converged
    assert np.all(np.isfinite(result.quantiles))


def test_window_rows_verdict_and_record():
    fit, data, _curve = _sphere_fit()
    checked = app.target_for(fit.model, data, (), fit)
    assert checked.kind == "fit" and set(checked.reference) == {"R_1", "sigma_R_1"}
    with pytest.raises(ValueError, match="Nothing to check"):
        app.target_for(fit.model, data, (), None)
    result = app.run(checked, steps=200, seed=4)
    rows = app.rows(result, checked.reference)
    assert [row["name"] for row in rows] == ["R_1", "sigma_R_1"]
    assert all(row["overlap"] is not None for row in rows)
    template, values = app.verdict(result)
    assert template in (app.CONVERGED, app.NOT_CONVERGED) and "{" not in template.format(**values)
    record = app.record(result, checked)
    json.dumps(record, allow_nan=False)
    assert record["checked"] == "fit" and record["steps"] == 200 and record["note"] == app.CHECK_ONLY
    assert math.isfinite(record["values"][0]["median"])
