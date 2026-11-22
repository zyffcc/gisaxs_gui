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
    start = rng.normal(size=(2, 16, 3)) + np.array([-2, 2])[:, None, None]
    chain = mcmc.sample(lambda x: -0.5 * np.einsum("bi,ij,bj->b", x, precision, x), start, 2500, rng)
    found = mcmc.quantiles(chain.kept().reshape(-1, 3))
    exact = np.array([[0.0, -1.0, 1.0, -1.959964, 1.959964]]).T * np.sqrt(np.diag(cov))[None, :]
    assert np.max(np.abs(found - exact) / np.sqrt(np.diag(cov))) < 0.25
    diagnostics = mcmc.diagnose(chain)
    assert diagnostics.converged and max(diagnostics.rhat) < 1.05 and 0.15 < diagnostics.acceptance < 0.7


def test_one_mode_can_look_mixed_while_dispersed_ensembles_disagree():
    class Bimodal:
        names = reported = ["x"]
        kind = "predict"
        best = np.array([-8.0])
        # Two clusters of distinct importance samples, with equal mass and negligible probability between them.
        importance_z = (np.array([-8.0, 8.0])[:, None] + np.linspace(-0.15, 0.15, 40)).reshape(-1, 1)

        @staticmethod
        def values(z):
            return z

        def __call__(self, z):
            return np.logaddexp(-0.5 * ((z[:, 0] + 8) / 0.15)**2, -0.5 * ((z[:, 0] - 8) / 0.15)**2)

    target = Bimodal()
    result = pc.run_check(target, steps=1500, walkers=12, seed=31)
    assert result.starts_from == "importance samples"
    assert np.all(result.trace[:, :12, 0] < 0) != np.all(result.trace[:, 12:, 0] < 0)
    assert result.diagnostics.ensembles == 2
    assert result.diagnostics.rhat[0] > 1.1 and not result.diagnostics.converged
    rng = np.random.default_rng(19)
    single = mcmc.sample(target, target.importance_z[:12], 1500, rng)
    assert abs(mcmc.diagnose(single).rhat[0] - 1) < 0.05
    assert not mcmc.diagnose(single).converged


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


def _sphere_fit(*, local_box=False):
    rng = np.random.default_rng(1)
    q = np.geomspace(0.1, 2.5, 60)
    truth = fm.FitModel((fm.new_component("sphere", radius=4.0, structure=False),)).with_values(
        {(0, "Int"): 1e4, ("globals", "background"): 5.0})
    if local_box:
        # The covariance is local to the R=4 nm mode. The default box reaches 500 nm and includes distant
        # oscillatory solutions. These bounds are at least six local standard errors from the generating values.
        truth = truth.with_parameter((0, "R"), lower=3.8, upper=4.2)
        truth = truth.with_parameter((0, "sigma_R"), lower=0.16, upper=0.24)
    truth = truth.with_parameter(("globals", "res_amplitude"), value=0.0, free=False)
    for key in ("res_width", "res_exponent"):
        truth = truth.with_parameter(("globals", key), free=False)
    y = rng.poisson(fm.evaluate(truth, q)).astype(float)
    data = fe.FitData.prepare(q, y, np.sqrt(np.maximum(y, 1.0)))
    return fe.fit(truth.with_values({(0, "R"): 4.15 if local_box else 4.4}), data), data, (q, y)


def test_check_of_a_manual_fit_matches_its_errors():
    fit, data, _curve = _sphere_fit(local_box=True)
    target = pc.FitTarget(fit.model, data)
    assert target.names == ["R_1", "sigma_R_1"]
    residual = fe.residuals(fit.model, data)
    t = np.clip(target.problem.start(), 1e-6, 1 - 1e-6)
    expected = -0.5 * float(residual @ residual) / target.variance + float(np.sum(np.log(t) + np.log1p(-t)))
    assert target(target.z_star[None, :])[0] == pytest.approx(expected, rel=1e-9)
    assert target.variance == pytest.approx(fit.chi2_reduced, rel=1e-6)
    result = pc.run_check(target, steps=2000, seed=3)
    assert result.starts_from == "prior box" and result.diagnostics.converged
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
    result = pc.run_check(target, steps=60, seed=1)
    assert not result.diagnostics.converged
    assert result.starts_from == "prior box" and result.diagnostics.ensembles == 2
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
    assert record["ensembles"] == 2 and record["walkers_per_ensemble"] == result.walkers
    assert math.isfinite(record["values"][0]["median"])


def test_predict_check_uses_importance_samples_and_labels_its_reference():
    from types import SimpleNamespace

    fit, data, (q, _y) = _sphere_fit()
    composition = E.composition_index([("sphere", False)])
    rng = np.random.default_rng(10)
    z = rng.normal(size=(48, E.DIMS))
    samples = physical(E.from_z(z, np.full(48, q.max() / q.min()), composition), q.max())
    reference = {"median": 4.1, "low68": 4.05, "high68": 4.12, "low95": 4.01, "high95": 4.2}
    solution = SimpleNamespace(model=fit.model, intervals={"R_1": (4.0, 4.0, 3.8, 4.2, 3.6, 4.4, 0.2)},
                               flags=("bic",), importance_starts=samples, posterior_summary={"R_1": reference},
                               diagnostics={"posterior_kind": "tempered", "interval_kind": "bic"})
    checked = app.target_for(fit.model, data, [solution])
    assert checked.reference_kind == "tempered" and checked.reference["R_1"] == tuple(reference.values())
    result = app.run(checked, steps=12, walkers=16)
    assert result.starts_from == "importance samples"
    record = app.record(result, checked)
    assert record["note"] == app.PREDICT_CHECK
    assert all(math.isfinite(record["values"][0][k]) for k in ("tau", "rhat"))
    assert app.PREDICT_CHECK in str(app.report_blocks(result, checked))
    assert "split R̂" in str(app.report_blocks(result, checked))
    solution.posterior_summary = {}
    solution.importance_starts = []
    checked = app.target_for(fit.model, data, [solution])
    assert checked.reference_kind == "least_squares" and checked.reference["R_1"] == tuple(solution.intervals["R_1"][1:6])


def test_born_only_check_uses_the_recorded_geometry_and_error_floor():
    from types import SimpleNamespace

    fit, data, _curve = _sphere_fit()
    model = fit.model
    for key, value in (("alpha_i", 0.12), ("alpha_f", 0.12), ("wavelength", 0.1)):
        model = model.with_parameter(("globals", key), value=value)
    # The angles reconstructed in Model give k_i − k_f = 0. The sampled cut had no reflected path.
    solution = SimpleNamespace(model=model, intervals={"R_1": (4, 4, 3.8, 4.2, 3.5, 4.5, 0.2)},
                               diagnostics={"cut_qz": 0.3, "cut_qz_minus": None, "model_error_floor": 0.08})
    target = app.target_for(model, data, [solution]).target
    assert target.data.qz == 0.3 and not target.data.reflected
    assert "reflection_re" not in target.names
    assert np.allclose(target.data.sigma, np.hypot(data.sigma, 0.08 * data.intensity))
    tiny_width = 0.00001
    model = model.with_parameter(("globals", "res_width"), value=tiny_width)
    solution.model = model
    target = app.target_for(model, data, [solution]).target
    index = target.names.index("tail_width")
    assert target.best[index] == tiny_width and target.values(target.z_star)[0, index] > tiny_width
