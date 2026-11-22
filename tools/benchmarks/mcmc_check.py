"""The MCMC check of a fit (``domain.mcmc``, ``domain.posterior_check``) against posteriors known otherwise
(Physics and Methods 9.19).

1. A correlated Gaussian in 4 dimensions: the quantiles of every coordinate against the exact ones.
2. A synthetic cut of spheres (R = 4 nm, σ_R/R = 0.2, number-weighted Gaussian, 10⁴ counts at q → 0, a background
   of 5 counts, Poisson noise, 120 points over q = 0.1–2.5 nm⁻¹), a manual fit of R and σ_R/R (the scale and the
   background solved): the MCMC check of the fit (``FitTarget``) against the same posterior integrated on a 241 × 241
   grid in the fit's coordinates (both ranges logarithmic), the marginal quantiles of R and σ_R/R.
3. The same cut and Predict's posterior of its solution (``PredictTarget``: the model error and the scales
   integrated out, the prior of the network's box) against Predict's own importance sampling around the solution
   (``posterior.laplace.local_posterior``, 8 rounds of 400 samples): the 68 % and 95 % intervals, of every value
   and of the populations' alone (a beam tail the curve does not need is held by the prior only, with a long upper
   tail that importance sampling of a few hundred effective samples reaches only roughly).

The deviations are given in units of each posterior's standard deviation (σ of the grid, or half the 68 % interval).
usage: python tools/benchmarks/mcmc_check.py   (writes mcmc_check.json next to itself; about two minutes)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from src.gimap.features.fitting.domain import fit_engine as fe  # noqa: E402
from src.gimap.features.fitting.domain import fit_model as fm  # noqa: E402
from src.gimap.features.fitting.domain import mcmc  # noqa: E402
from src.gimap.features.fitting.domain import posterior_check as pc  # noqa: E402
from src.gimap.features.fitting.domain.posterior import encoding as E  # noqa: E402
from src.gimap.features.fitting.domain.posterior import refine  # noqa: E402
from src.gimap.features.fitting.domain.posterior.laplace import local_posterior  # noqa: E402

LEVELS = (0.5, 0.16, 0.84, 0.025, 0.975)


def gaussian(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, 0.9, 0.0, 0.2], [0.9, 1.0, 0.1, 0.0], [0.0, 0.1, 4.0, -1.0], [0.2, 0.0, -1.0, 0.5]])
    precision = np.linalg.inv(cov)
    chain = mcmc.sample(lambda x: -0.5 * np.einsum("bi,ij,bj->b", x, precision, x), 0.1 * rng.normal(size=(16, 4)),
                        4000, rng)
    kept = chain.kept().reshape(-1, 4)
    exact = np.array([[0.0, -1.0, 1.0, -1.959964, 1.959964]]).T * np.sqrt(np.diag(cov))[None, :]
    found = mcmc.quantiles(kept, LEVELS)
    diagnostics = mcmc.diagnose(chain)
    return dict(steps=4000, walkers=16, max_quantile_error_sigma=float(np.max(np.abs(found - exact) / np.sqrt(np.diag(cov)))),
                tau=list(diagnostics.tau), rhat=list(diagnostics.rhat), acceptance=diagnostics.acceptance,
                converged=diagnostics.converged)


def sphere_curve(seed: int = 1):
    rng = np.random.default_rng(seed)
    q = np.geomspace(0.1, 2.5, 120)
    truth = fm.FitModel((fm.new_component("sphere", radius=4.0, structure=False),)).with_values(
        {(0, "Int"): 1e4, ("globals", "background"): 5.0})
    truth = truth.with_parameter(("globals", "res_amplitude"), value=0.0, free=False)
    for key in ("res_width", "res_exponent"):
        truth = truth.with_parameter(("globals", key), free=False)
    y = rng.poisson(fm.evaluate(truth, q)).astype(float)
    return q, y, np.sqrt(np.maximum(y, 1.0)), truth


def grid_posterior(target: pc.FitTarget, points: int = 241, span: float = 6.0) -> np.ndarray:
    """Marginal quantiles (LEVELS × 2) of the fit's posterior on a grid of the fit's coordinates (both logarithmic,
    so the prior is uniform there) ± ``span`` widths around the best values."""
    problem = target.problem
    t0 = problem.start()

    def log_like(t) -> float:
        residual = problem.solve(np.asarray(t, float))[1]
        return -0.5 * float(residual @ residual) / target.variance

    # the marginal widths (the inverse of the Hessian in t by central differences): R and σ_R/R are correlated, and
    # a box of the conditional widths would cut the posterior off
    h = 1e-4
    hessian = np.empty((2, 2))
    for a in range(2):
        for b in range(2):
            ea, eb = np.eye(2)[a] * h, np.eye(2)[b] * h
            hessian[a, b] = (log_like(t0 + ea + eb) - log_like(t0 + ea - eb) - log_like(t0 - ea + eb)
                             + log_like(t0 - ea - eb)) / (4 * h * h)
    widths = np.sqrt(np.diag(np.linalg.inv(-hessian)))
    axes = [np.linspace(t0[k] - span * widths[k], t0[k] + span * widths[k], points) for k in range(2)]
    surface = np.array([[log_like([a, b]) for b in axes[1]] for a in axes[0]])
    weight = np.exp(surface - surface.max())
    out = np.empty((len(LEVELS), 2))
    for k, axis in enumerate(axes):
        marginal = weight.sum(axis=1 - k)
        cumulative = np.cumsum(marginal) / marginal.sum()
        t_levels = np.interp(LEVELS, cumulative, axis)
        out[:, k] = [problem.decode(np.where(np.arange(2) == k, t, t0))[k] for t in t_levels]
    return out


def main():
    rows = {"gaussian_4d": gaussian()}
    print("Gaussian:", rows["gaussian_4d"], flush=True)
    q, y, sigma, truth = sphere_curve()
    data = fe.FitData.prepare(q, y, sigma)
    fit = fe.fit(truth.with_values({(0, "R"): 4.5, (0, "sigma_R"): 0.25}), data)
    target = pc.FitTarget(fit.model, data)
    check = pc.run_check(target, steps=4000, seed=2)
    grid = grid_posterior(target)
    sd = 0.5 * (grid[2] - grid[1])
    rows["fit_target"] = dict(
        names=check.names, fit=list(check.best), fit_errors=[fit.errors[(0, "R")], fit.errors[(0, "sigma_R")]],
        grid_quantiles=grid.tolist(), mcmc_quantiles=check.quantiles.tolist(), steps=check.steps,
        walkers=check.walkers, seconds=check.seconds,
        max_quantile_error_sigma=float(np.max(np.abs(check.quantiles - grid) / sd)),
        tau=list(check.diagnostics.tau), rhat=list(check.diagnostics.rhat), converged=check.diagnostics.converged)
    print("FitTarget:", rows["fit_target"], flush=True)
    composition = E.composition_index([("sphere", False)])
    centre = refine.model_values(fit.model, composition)
    predict = pc.PredictTarget(q, y, sigma, composition=composition, centre=centre)
    check = pc.run_check(predict, steps=4000, seed=3)
    local = local_posterior(predict.data, composition, centre, pc.TAU_START, np.random.default_rng(7), 400)
    names = list(predict.names)
    columns = list(predict.dims)
    sampled = local.quantiles(LEVELS, refine.ESS_MIN)[:, columns]
    half = 0.5 * (check.quantiles[2] - check.quantiles[1])
    reported = [names.index(n) for n in predict.reported]
    populations = [i for i, d in enumerate(predict.dims) if d < E.GLOBAL.start]
    rows["predict_target"] = dict(
        names=names, reported=predict.reported, mcmc_quantiles=check.quantiles.tolist(),
        importance_quantiles=sampled.tolist(), importance_ess=local.ess, steps=check.steps, walkers=check.walkers,
        seconds=check.seconds,
        max_quantile_difference_sigma=float(np.max(np.abs(check.quantiles[:, reported] - sampled[:, reported])
                                                   / half[reported])),
        max_quantile_difference_sigma_populations=float(np.max(
            np.abs(check.quantiles[:, populations] - sampled[:, populations]) / half[populations])),
        tau=list(check.diagnostics.tau), rhat=list(check.diagnostics.rhat), converged=check.diagnostics.converged)
    print("PredictTarget:", rows["predict_target"], flush=True)
    (HERE / "mcmc_check.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
