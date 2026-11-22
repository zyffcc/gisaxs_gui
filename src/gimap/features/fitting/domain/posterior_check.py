"""The posterior of a solution or of a manual fit checked by a short MCMC run (``mcmc``; Physics and Methods 9.19).

A check only: it reports intervals and how far the run can be trusted beside the values it checks, and changes none.

* ``PredictTarget``: the posterior that Predict parameters samples for one of its solutions (``posterior.laplace``):
  the curve's σ with the model error floor added (σ² + (f I)², f = ``refine.MODEL_ERROR_FLOOR``), the likelihood of
  ``posterior.infer.Likelihood`` (the smooth multiplicative model error of amplitude τ and the scales integrated out)
  and its prior, uniform in the unit box of ``posterior.encoding``, in the network's coordinates z = logit(v) of the
  composition's values (the effective reflection coefficient only on a cut with both angles). Started around the
  solution's values (τ around ``TAU_START``).
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
BALL = 0.5
"""The walkers start around the values checked, each coordinate spread by this fraction of the target's own width
there, 1/√(−∂² ln p/∂z²) by central differences (``widths``; between ``WIDTH_RANGE``): an estimate the run does not
depend on (its first half is discarded), only how soon it reaches the bulk."""
WIDTH_RANGE = (1e-3, 1.0)
CURVATURE_STEP = 1e-2
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
                 model_error_floor: float = MODEL_ERROR_FLOOR):
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
        tau = E.GLOBAL.start + 3
        if not np.isfinite(centre[tau]):
            centre[tau] = TAU_START
        for d in reflection:
            if not np.isfinite(centre[d]):
                centre[d] = 0.0
        unit = E.to_unit(dimensionless(centre[None, :], self.data.q_max), [self.data.ratio], self.composition)
        self.z_star = _logit(np.clip(unit, 1e-6, 1 - 1e-6))[0][self.dims]
        self.best = self.values(self.z_star[None, :])[0]

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
    """(≤ TRACE_STEPS, W, k) physical values for the trace plot."""
    trace_steps: np.ndarray
    steps: int
    walkers: int
    seconds: float
    stopped: bool
    seed: int


def widths(target) -> np.ndarray:
    """1/√(−∂² ln p/∂z²) of every coordinate at the values checked (central differences), within WIDTH_RANGE."""
    centre = np.asarray(target.z_star, float)
    points = np.repeat(centre[None, :], 2 * centre.size + 1, axis=0)
    for k in range(centre.size):
        points[2 * k, k] += CURVATURE_STEP
        points[2 * k + 1, k] -= CURVATURE_STEP
    values = target(points)
    with np.errstate(invalid="ignore", divide="ignore"):
        second = (values[0:-1:2] + values[1:-1:2] - 2.0 * values[-1]) / CURVATURE_STEP**2
        width = 1.0 / np.sqrt(np.where(np.isfinite(second) & (second < 0), -second, 0.0))
    return np.clip(np.where(np.isfinite(width), width, WIDTH_RANGE[1]), *WIDTH_RANGE)


def starts(target, walkers: int, rng: np.random.Generator, tries: int = 20) -> np.ndarray:
    """The walkers' start: a ball around the values checked (``BALL`` × ``widths``), redrawn (and shrunk) where the
    posterior is zero."""
    centre, spread = np.asarray(target.z_star, float), BALL * widths(target)
    x = centre + spread * rng.standard_normal((walkers, centre.size))
    lp = target(x)
    for attempt in range(tries):
        bad = ~np.isfinite(lp)
        if not bad.any():
            return x
        scale = 0.5 ** (attempt + 1)
        x[bad] = centre + scale * spread * rng.standard_normal((int(bad.sum()), centre.size))
        lp[bad] = target(x[bad])
    if not np.all(np.isfinite(lp)):
        raise ValueError("The posterior is zero around the values checked: no walker could start")
    return x


def run_check(target, *, steps: int = 2000, walkers: Optional[int] = None, seed: int = 0,
              progress: Optional[Callable[[int, int], None]] = None,
              stop: Optional[Callable[[], bool]] = None) -> CheckResult:
    """``steps`` steps of the ensemble on ``target`` (``PredictTarget`` or ``FitTarget``); quantiles and diagnostics of
    the second half, in physical values."""
    tick = time.perf_counter()
    rng = np.random.default_rng(seed)
    count = walkers or mcmc.walkers_for(len(target.names))
    chain = mcmc.sample(target, starts(target, count, rng), int(steps), rng, progress=progress, stop=stop)
    if chain.steps < 4:
        raise ValueError("The run was stopped before it had steps to report")
    flat = chain.positions.reshape(-1, chain.positions.shape[2])
    physical_values = target.values(flat).reshape(chain.positions.shape[0], chain.positions.shape[1], -1)
    burn = int(math.floor(mcmc.BURN * chain.steps))
    kept = physical_values[burn:].reshape(-1, physical_values.shape[2])
    every = max(1, chain.steps // TRACE_STEPS)
    return CheckResult(target.kind, list(target.names), list(target.reported), np.asarray(target.best, float),
                       mcmc.quantiles(kept), mcmc.diagnose(chain, coordinates=physical_values),
                       physical_values[::every], np.arange(0, chain.steps, every), chain.steps, count,
                       time.perf_counter() - tick, chain.stopped, int(seed))


__all__ = ["CheckResult", "FitTarget", "PredictTarget", "run_check", "starts"]
