"""The posterior of a solution or of a manual fit checked by a short MCMC run (``mcmc``; Physics and Methods 9.19).

A check only: it reports intervals and how far the run can be trusted beside the values it checks, and changes none.

* ``PredictTarget``: the posterior that Predict parameters samples for one of its solutions (``posterior.laplace``):
  the curve's σ with the model error floor added (σ² + (f I)², f = ``refine.MODEL_ERROR_FLOOR``), the likelihood of
  ``posterior.infer.Likelihood`` (the smooth multiplicative model error of amplitude τ and the scales integrated out)
  and its prior, uniform in the unit box of ``posterior.encoding``, in the network's coordinates z = logit(v) of the
  composition's values (the effective reflection coefficient only on a cut with both angles).
* ``FitTarget``: a manual fit (Refine or Search). Its free non-linear values, uniform in the fit's coordinates t ∈ [0, 1]
  (in ln for a range spanning a decade or more; the contact D ≥ 2R and the inequalities as in the fit), sampled in
  u = logit(t); the scales solved at every point, as the fit does (a profile, not a marginal); and
  log L = −Σ r² / (2 s²), s² = the fit's χ²ᵣ: the likelihood whose curvature the fit's errors linearise (9.8), so a
  curve described only to χ²ᵣ > 1 widens the intervals as it widens the errors.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np

from . import fit_engine, fit_links, mcmc
from .posterior import encoding as E
from .posterior.infer import PHYSICAL_NAMES, Cut, Likelihood, dimensionless, physical
from .posterior.refine import MODEL_ERROR_FLOOR, NUISANCE

TAU_START = 0.02
ENSEMBLES = 2
TRACE_STEPS = 400
"""Steps kept for the trace plot (evenly thinned)."""


def _logit(v):
    v = np.clip(np.asarray(v, float), 1e-9, 1.0 - 1e-9)
    return np.log(v) - np.log1p(-v)


def _expit(u):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(u, float)))


class PredictTarget:
    """Predict's posterior of one solution (module docstring)."""

    kind = "predict"

    def __init__(self, q, intensity, sigma, *, composition: int, centre, qz: float = 0.0,
                 qz_minus: Optional[float] = None, law: str = "number_gauss", coupling: str = "lma",
                 model_error_floor: float = MODEL_ERROR_FLOOR, importance_starts=()):
        counts = Cut.prepare(q, intensity, sigma, qz, qz_minus)
        self.data = (replace(counts, sigma=np.hypot(counts.sigma, model_error_floor * np.abs(counts.intensity)))
                     if model_error_floor > 0 else counts)
        self.composition = int(composition)
        self.likelihood = Likelihood(self.data, self.composition, law, coupling)
        active = E.active(self.composition)[0]
        reflection = {E.GLOBAL.start + 4, E.GLOBAL.start + 5}
        self.dims = [d for d in np.flatnonzero(active) if self.data.reflected or d not in reflection]
        self.names = [PHYSICAL_NAMES[d] for d in self.dims]
        self.reported = [n for d, n in zip(self.dims, self.names) if E.NAMES[d] not in NUISANCE]
        centre = np.array(centre, float)
        self.best = centre[self.dims].copy()  # the checked physical values, including any outside the sampler's box
        tau = E.GLOBAL.start + 3
        if not np.isfinite(centre[tau]):
            centre[tau] = TAU_START
        for d in reflection:
            if not np.isfinite(centre[d]):
                centre[d] = 0.0
        unit = E.to_unit(dimensionless(centre[None, :], self.data.q_max), [self.data.ratio], self.composition)
        self.z_star = _logit(np.clip(unit, 1e-6, 1 - 1e-6))[0][self.dims]
        samples = np.asarray(importance_starts, float)
        self.importance_z = np.empty((0, len(self.dims)))
        if samples.size:
            samples = np.where(active, samples, np.nan)  # the job's JSON uses zero for inactive coordinates
            unit = E.to_unit(dimensionless(samples, self.data.q_max), np.full(len(samples), self.data.ratio),
                             self.composition)[:, self.dims]
            self.importance_z = np.unique(_logit(unit[np.all(np.isfinite(unit), axis=1)]), axis=0)

    def _full(self, z) -> np.ndarray:
        z = np.atleast_2d(np.asarray(z, float))
        full = np.zeros((z.shape[0], E.DIMS))
        full[:, self.dims] = z
        return full

    def __call__(self, z) -> np.ndarray:
        full = self._full(z)
        values = physical(E.from_z(full, np.full(full.shape[0], self.data.ratio), self.composition), self.data.q_max)
        log_like, _amplitudes, _chi2 = self.likelihood(values)
        out = log_like + E.log_prior(full, self.composition)
        return np.where(np.isfinite(out), out, -np.inf)

    def values(self, z) -> np.ndarray:
        """Physical values (B, len(names)) of positions z (B, k)."""
        full = self._full(z)
        values = physical(E.from_z(full, np.full(full.shape[0], self.data.ratio), self.composition), self.data.q_max)
        return values[:, self.dims]



class FitTarget:
    """A manual fit's posterior (module docstring)."""

    kind = "fit"

    def __init__(self, model, data: fit_engine.FitData):
        self.problem = fit_engine._Problem(model, data, None, None, 1)
        if not self.problem.nonlinear:
            raise ValueError("The fit has no free value besides the scales: nothing to sample")
        self.paths = [path for path, _p in self.problem.nonlinear]
        self.names = [fit_links.parameter_name(path) for path in self.paths]
        self.reported = list(self.names)
        start = np.clip(self.problem.start(), 1e-6, 1 - 1e-6)
        residual = self.problem.solve(start)[1]
        free = len(self.paths) + len(self.problem.linear)
        dof = residual.size - free
        if dof <= 0:
            raise ValueError("The curve has no more points than free values")
        self.variance = max(float(residual @ residual) / dof, 1e-300)
        self.z_star = _logit(start)
        self.best = self.values(self.z_star[None, :])[0]

    def __call__(self, u) -> np.ndarray:
        u = np.atleast_2d(np.asarray(u, float))
        out = np.empty(u.shape[0])
        for row, point in enumerate(u):
            t = _expit(point)
            residual = self.problem.solve(t)[1]
            prior = float(np.sum(np.log(np.maximum(t, 1e-300)) + np.log(np.maximum(1.0 - t, 1e-300))))
            value = -0.5 * float(residual @ residual) / self.variance + prior
            out[row] = value if math.isfinite(value) else -np.inf
        return out

    def values(self, u) -> np.ndarray:
        u = np.atleast_2d(np.asarray(u, float))
        return np.array([self.problem.decode(_expit(point)) for point in u])



@dataclass
class CheckResult:
    kind: str
    names: list
    reported: list
    best: np.ndarray
    """The values checked (the solution's or the fit's), in ``names`` order."""
    quantiles: np.ndarray
    """(5, k): median, 16 %, 84 %, 2.5 %, 97.5 % of the kept steps."""
    diagnostics: mcmc.Diagnostics
    trace: np.ndarray
    """(≤ TRACE_STEPS, ensembles × W, k) physical values for the trace plot."""
    trace_steps: np.ndarray
    steps: int
    walkers: int
    seconds: float
    stopped: bool
    seed: int
    starts_from: str = "prior box"


def _dispersed_groups(points: np.ndarray) -> list[np.ndarray]:
    """Separate the starting points along their broadest direction in standardised sampler coordinates."""
    if len(points) < 2:
        return list(np.array_split(points, ENSEMBLES))
    spread = points.std(axis=0)
    standard = (points - points.mean(axis=0)) / np.where(spread > 0, spread, 1.0)
    direction = np.linalg.svd(standard, full_matrices=False)[2][0]
    return list(np.array_split(points[np.argsort(standard @ direction)], ENSEMBLES))


def starts(target, walkers: int, rng: np.random.Generator, tries: int = 20) -> np.ndarray:
    """Draw distinct walkers from dispersed subsets of this mode's importance samples, filling missing starts
    from the prior, uniform before the logit transformation for both targets."""
    dims = len(target.names)
    samples = getattr(target, "importance_z", np.empty((0, dims)))
    groups = _dispersed_groups(samples)
    if any(len(group) < walkers for group in groups):
        priors = _dispersed_groups(_logit(rng.uniform(size=(ENSEMBLES * walkers, dims))))
        groups = [np.concatenate((group, prior[:max(0, walkers - len(group))]))
                  for group, prior in zip(groups, priors)]
    ensembles = np.stack([group[rng.choice(len(group), walkers, replace=False)] for group in groups])
    flat = ensembles.reshape(-1, dims)
    bad = ~np.isfinite(target(flat))
    for _ in range(tries):
        if not bad.any():
            return ensembles
        flat[bad] = _logit(rng.uniform(size=(int(bad.sum()), dims)))
        bad[bad] = ~np.isfinite(target(flat[bad]))
    if bad.any():
        raise ValueError("Could not find finite posterior density for every starting walker")
    return ensembles


def run_check(target, *, steps: int = 2000, walkers: Optional[int] = None, seed: int = 0,
              progress: Optional[Callable[[int, int], None]] = None,
              stop: Optional[Callable[[], bool]] = None) -> CheckResult:
    """Run independent ensembles on the target. Pool their kept draws for marginal quantiles in physical units."""
    tick = time.perf_counter()
    rng = np.random.default_rng(seed)
    count = walkers or mcmc.walkers_for(len(target.names))
    start = starts(target, count, rng)
    from_importance = set(map(tuple, getattr(target, "importance_z", ())))
    n_importance = sum(tuple(row) in from_importance for ensemble in start for row in ensemble)
    source = ("prior box" if n_importance == 0 else
              "importance samples and prior box" if n_importance < ENSEMBLES * count else "importance samples")
    chain = mcmc.sample(target, start, int(steps), rng, progress=progress, stop=stop)
    if chain.steps < 4:
        raise ValueError("The run was stopped before it had steps to report")
    flat = chain.positions.reshape(-1, chain.positions.shape[-1])
    physical_values = target.values(flat).reshape(chain.steps, ENSEMBLES, count, -1)
    burn = int(math.floor(mcmc.BURN * chain.steps))
    kept = physical_values[burn:].reshape(-1, physical_values.shape[-1])
    every = max(1, math.ceil(chain.steps / TRACE_STEPS))
    trace = physical_values[::every].reshape(-1, ENSEMBLES * count, physical_values.shape[-1]).copy()
    return CheckResult(target.kind, list(target.names), list(target.reported), np.asarray(target.best, float),
                       mcmc.quantiles(kept), mcmc.diagnose(chain, coordinates=physical_values),
                       trace, np.arange(0, chain.steps, every), chain.steps, count,
                       time.perf_counter() - tick, chain.stopped, int(seed), source)


__all__ = ["CheckResult", "FitTarget", "PredictTarget", "run_check", "starts"]
