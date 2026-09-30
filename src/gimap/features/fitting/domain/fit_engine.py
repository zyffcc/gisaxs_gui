"""Fitting one curve with a ``FitModel``: bounded least squares, a wide search first, and the errors.

* The free scales (component Int, background, resolution-peak A) enter linearly: at every step
  they are solved exactly by non-negative least squares for the other parameters (variable
  projection), so the search only moves sizes, spreads, distances and the peak shape.
* ``local``: bounded least squares (trust region) from the current values.
* ``global``: differential evolution over the ranges of the free non-linear parameters (a range
  spanning 10× or more is searched on a log scale), then ``local`` from the best three starts.
* Residuals: (I − model)/σ when σ is known, else ln(I/model) (every point the same relative weight).
* Errors: from the Jacobian of all free parameters at the solution, cov = (JᵀJ)⁻¹·s² with
  s² = Σr²/(N − p) — reported with the correlated pairs (|ρ| ≥ 0.95) and parameters left at a
  bound of their range, where the error is not meaningful.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import numpy as np
from scipy.optimize import differential_evolution, least_squares, nnls
from scipy.stats import qmc

from .fit_model import LINEAR, FitModel, component_form, evaluate, resolution_shape

TINY = 1e-300
CORRELATED = 0.95
EDGE = 1e-3
LOCAL_STEP = 0.02
"""First steps of a local fit, as a fraction of each parameter's range (log range when log-scaled)."""
"""A non-linear parameter within this fraction of its range from a bound is reported as at the bound."""


class FitStopped(Exception):
    """Raised inside the optimiser when the person stops the fit; the best values so far are kept."""


@dataclass(frozen=True)
class FitData:
    q: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray] = None

    @classmethod
    def prepare(cls, q, intensity, sigma=None) -> "FitData":
        """|q| > 0 in nm⁻¹, finite points, sorted; σ kept only when every kept point has σ > 0."""
        q = np.abs(np.asarray(q, dtype=float)).reshape(-1)
        y = np.asarray(intensity, dtype=float).reshape(-1)
        keep = np.isfinite(q) & np.isfinite(y) & (q > 0)
        s = None if sigma is None else np.asarray(sigma, dtype=float).reshape(-1)
        if s is not None and s.shape == q.shape:
            keep_sigma = keep & np.isfinite(s) & (s > 0)
            s = s if keep_sigma.sum() == keep.sum() else None
        else:
            s = None
        if s is None:
            keep &= y > 0  # relative residuals need I > 0
        order = np.argsort(q[keep])
        return cls(q[keep][order], y[keep][order], None if s is None else s[keep][order])

    @property
    def weighting(self) -> str:
        return "sigma" if self.sigma is not None else "relative"


@dataclass(frozen=True)
class FitResult:
    model: FitModel
    method: str
    weighting: str
    points: int
    free: int
    chi2_reduced: float
    log_rmse: float
    evaluations: int
    seconds: float
    converged: bool
    stopped: bool
    message: str
    errors: Mapping[tuple, float] = field(default_factory=dict)
    correlated: tuple = ()
    at_bounds: tuple = ()


def residuals(model: FitModel, data: FitData) -> np.ndarray:
    """(I − model)/σ, or ln(I/model) without σ (the residual plot)."""
    return _residual(evaluate(model, data.q), data)


def _residual(predicted: np.ndarray, data: FitData) -> np.ndarray:
    if data.sigma is not None:
        return (data.intensity - predicted) / data.sigma
    return np.log(data.intensity) - np.log(np.maximum(predicted, TINY))


def log_rmse(model: FitModel, data: FitData) -> float:
    predicted = evaluate(model, data.q)
    good = (data.intensity > 0) & (predicted > 0)
    if not good.any():
        return math.nan
    return float(np.sqrt(np.mean(np.log(data.intensity[good] / predicted[good]) ** 2)))


class _Problem:
    """The free parameters of a model split into non-linear (searched) and linear (solved)."""

    def __init__(self, model: FitModel, data: FitData, stop, progress, total: int):
        self.model, self.data, self.stop, self.progress, self.total = model, data, stop, progress, max(1, total)
        free = [(path, parameter) for path, parameter in model.parameters() if parameter.free]
        self.nonlinear = [(path, p) for path, p in free if path[1] not in LINEAR]
        self.linear = [path for path, p in free if path[1] in LINEAR]
        for path, parameter in self.nonlinear:
            if not (math.isfinite(parameter.lower) and math.isfinite(parameter.upper) and parameter.upper > parameter.lower):
                raise ValueError(f"Give {path[1]} a finite range (min < max) to fit it")
        self.lower = np.array([p.lower for _path, p in self.nonlinear], dtype=float)
        self.upper = np.array([p.upper for _path, p in self.nonlinear], dtype=float)
        self.logarithmic = (self.lower > 0) & (self.upper >= 10.0 * np.maximum(self.lower, TINY))
        self.weights = 1.0 / data.sigma if data.sigma is not None else 1.0 / data.intensity
        self.evaluations = 0
        self.cache: dict = {}
        self.best = (math.inf, model)
        self._reported = 0.0
        self.phase = ""

    # -- the unit cube ↔ values -----------------------------------------------------

    def decode(self, t) -> np.ndarray:
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        values = self.lower + t * (self.upper - self.lower)
        if self.logarithmic.any():
            lo, hi = np.log(self.lower[self.logarithmic]), np.log(self.upper[self.logarithmic])
            values[self.logarithmic] = np.exp(lo + t[self.logarithmic] * (hi - lo))
        return values

    def encode(self, values) -> np.ndarray:
        values = np.clip(np.asarray(values, dtype=float), self.lower, self.upper)
        t = (values - self.lower) / (self.upper - self.lower)
        if self.logarithmic.any():
            lo, hi = np.log(self.lower[self.logarithmic]), np.log(self.upper[self.logarithmic])
            t[self.logarithmic] = (np.log(values[self.logarithmic]) - lo) / (hi - lo)
        return np.clip(t, 1e-9, 1.0 - 1e-9)

    def start(self) -> np.ndarray:
        return self.encode([p.value for _path, p in self.nonlinear])

    # -- one evaluation: set the non-linear values, solve the scales -----------------

    def solve(self, t) -> tuple[FitModel, np.ndarray]:
        if self.stop is not None and self.stop():
            raise FitStopped()
        self.evaluations += 1
        model = self.model.with_values({path: value for (path, _p), value in zip(self.nonlinear, self.decode(t))})
        if self.linear:
            model = self._scales(model)
        try:
            with np.errstate(all="ignore"):
                residual = _residual(evaluate(model, self.data.q, cache=self.cache), self.data)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            residual = np.full(self.data.q.shape, 1e6)
        if not np.all(np.isfinite(residual)):
            residual = np.where(np.isfinite(residual), residual, 1e6)
        cost = float(residual @ residual)
        if cost < self.best[0]:
            self.best = (cost, model)
        self._report(cost)
        return model, residual

    def _scales(self, model: FitModel) -> FitModel:
        q, g = self.data.q, model.globals
        k = float(g["k"].value)
        columns, fixed = [], np.full_like(q, 0.0 if ("globals", "background") in self.linear else g["background"].value)
        forms = {}
        for index, component in enumerate(model.components):
            forms[index] = component_form(component, q, self.cache)
            if (index, "Int") in self.linear:
                continue
            fixed = fixed + k * component.value("Int") * forms[index]
        peak = resolution_shape(model, q) if (g["res_amplitude"].value or ("globals", "res_amplitude") in self.linear) else None
        if peak is not None and ("globals", "res_amplitude") not in self.linear:
            fixed = fixed + k * g["res_amplitude"].value * peak
        for path in self.linear:
            owner, key = path
            columns.append(np.ones_like(q) if key == "background" else k * (peak if key == "res_amplitude" else forms[owner]))
        matrix = np.column_stack(columns) * self.weights[:, None]
        target = (self.data.intensity - fixed) * self.weights
        if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(target))):
            return model
        coefficients = nnls(matrix, target, maxiter=50 * matrix.shape[1])[0]
        return model.with_values(dict(zip(self.linear, coefficients)))

    def residual(self, t) -> np.ndarray:
        return self.solve(t)[1]

    def cost(self, t) -> float:
        residual = self.solve(t)[1]
        return float(residual @ residual)

    def _report(self, cost: float) -> None:
        now = time.perf_counter()
        if self.progress is None or now - self._reported < 0.25:
            return
        self._reported = now
        points = max(1, self.data.q.size - len(self.nonlinear) - len(self.linear))
        self.progress(min(0.99, self.evaluations / self.total),
                      f"{self.phase} — {self.evaluations} evaluations, best χ²ᵣ {self.best[0] / points:.4g}")


def fit(model: FitModel, data: FitData, *, method: str = "local", max_evaluations: Optional[int] = None,
        seed: int = 1729, progress: Optional[Callable[[float, str], None]] = None,
        stop: Optional[Callable[[], bool]] = None) -> FitResult:
    """Fit the free parameters of ``model`` to ``data`` (q in nm⁻¹)."""
    if method not in ("local", "global"):
        raise ValueError(f"Unknown fitting method {method!r}")
    if data.q.size < 3:
        raise ValueError("Too few points to fit (at least 3 in the fitting range)")
    tick = time.perf_counter()
    count = sum(1 for _path, p in model.parameters() if p.free)
    budget = int(max_evaluations or (4000 if method == "global" else 150 * (count + 1)))
    problem = _Problem(model, data, stop, progress, budget)
    stopped, converged, message = False, True, ""
    try:
        if not problem.nonlinear:
            problem.phase = "Solving the scales"
            problem.solve(np.zeros(0))
            message = "Only scales were free: solved exactly."
        else:
            starts = [problem.start()]
            if method == "global":
                starts = _search(problem, budget, seed)
            problem.phase = "Refining"
            local_budget = max(20, (budget if method == "local" else budget // 4) // len(starts))
            steps = max(5, local_budget // (len(problem.nonlinear) + 1))  # each step also costs a Jacobian
            best = None
            for start in starts:
                # Offsets from the start: the trust region begins at ~LOCAL_STEP of every range and
                # grows only while the steps pay off, so Refine stays near the values it was given.
                result = least_squares(lambda u, t0=start: problem.residual(t0 + u), np.zeros_like(start),
                                       bounds=(-start, 1.0 - start), x_scale=LOCAL_STEP, method="trf",
                                       max_nfev=steps, ftol=1e-6, xtol=1e-6, gtol=1e-8)
                if best is None or result.cost < best.cost:
                    best = result
            converged = bool(best.status > 0)
            message = str(best.message)
    except FitStopped:
        stopped, converged, message = True, False, "Stopped: the best values so far are kept."
    fitted = problem.best[1]
    errors, correlated = _errors(fitted, data, problem)
    return FitResult(
        model=fitted, method=method, weighting=data.weighting, points=int(data.q.size), free=count,
        chi2_reduced=_chi2(fitted, data, count), log_rmse=log_rmse(fitted, data), evaluations=problem.evaluations,
        seconds=time.perf_counter() - tick, converged=converged and not stopped, stopped=stopped, message=message,
        errors=errors, correlated=correlated, at_bounds=_at_bounds(fitted, problem),
    )


def fit_scales(model: FitModel, data: FitData) -> FitModel:
    """Only the free scales (Int, background, peak A) solved for the other values as they are:
    a starting model brought to the level of the curve."""
    held = model
    for path, parameter in model.parameters():
        if parameter.free and path[1] not in LINEAR:
            held = held.with_parameter(path, free=False)
    problem = _Problem(held, data, None, None, 1)
    if not problem.linear:
        return model
    solved = problem._scales(held)
    return model.with_values({path: solved.get(path).value for path in problem.linear})


def _search(problem: _Problem, budget: int, seed: int) -> list[np.ndarray]:
    """Differential evolution over the unit cube; the best three distinct members as starts."""
    problem.phase = "Searching the ranges"
    dimension = len(problem.nonlinear)
    size = max(8, 5 * dimension)
    population = qmc.LatinHypercube(d=dimension, seed=seed).random(size)
    population[0] = problem.start()
    generations = max(1, (3 * budget // 4) // size - 1)
    evolution = differential_evolution(problem.cost, [(0.0, 1.0)] * dimension, init=population, maxiter=generations,
                                       seed=seed, polish=False, tol=0.0, atol=0.0, updating="immediate")
    order = np.argsort(evolution.population_energies)
    starts: list[np.ndarray] = []
    for index in order:
        candidate = np.clip(evolution.population[index], 1e-9, 1 - 1e-9)
        if all(np.max(np.abs(candidate - other)) > 0.05 for other in starts):
            starts.append(candidate)
        if len(starts) == 3:
            break
    return starts or [problem.start()]


def _chi2(model: FitModel, data: FitData, free: int) -> float:
    residual = residuals(model, data)
    dof = data.q.size - free
    return float(residual @ residual / dof) if dof > 0 else math.nan


def _errors(model: FitModel, data: FitData, problem: _Problem) -> tuple[dict, tuple]:
    """1σ errors of every free parameter, and the strongly correlated pairs."""
    paths = [path for path, _p in problem.nonlinear] + list(problem.linear)
    if not paths or data.q.size <= len(paths):
        return {path: math.nan for path in paths}, ()
    values = np.array([model.get(path).value for path in paths], dtype=float)
    # The size of every parameter on its own (its range when it is 0): the finite-difference step
    # and the unit of its column. A parameter of 1e30 next to R ≈ 5 nm must not set R's step.
    scale = np.array([abs(value) if value else _width(model.get(path)) for path, value in zip(paths, values)])
    jacobian = np.empty((data.q.size, len(paths)))
    for column, (path, value) in enumerate(zip(paths, values)):
        step = 1e-4 * scale[column]
        with np.errstate(all="ignore"):
            upper = residuals(model.with_values({path: value + step}), data)
            lower = residuals(model.with_values({path: value - step}), data)
        jacobian[:, column] = (upper - lower) / (2.0 * step)
    residual = residuals(model, data)
    variance = float(residual @ residual) / (data.q.size - len(paths))
    if not np.all(np.isfinite(jacobian)):
        return {path: math.nan for path in paths}, ()
    # Columns in relative units, so that no parameter pushes the others below the cut-off of the inverse.
    scaled = jacobian * scale
    covariance = np.linalg.pinv(scaled.T @ scaled) * variance * np.outer(scale, scale)
    diagonal = np.diag(covariance)
    errors = {path: float(np.sqrt(d)) if d > 0 else math.nan for path, d in zip(paths, diagonal)}
    correlated = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if diagonal[i] > 0 and diagonal[j] > 0:
                rho = covariance[i, j] / math.sqrt(diagonal[i] * diagonal[j])
                if abs(rho) >= CORRELATED:
                    correlated.append((paths[i], paths[j], float(rho)))
    return errors, tuple(correlated)


def _width(parameter) -> float:
    width = parameter.upper - parameter.lower
    return float(width) if math.isfinite(width) and width > 0 else 1.0


def _at_bounds(model: FitModel, problem: _Problem) -> tuple:
    edges = []
    if problem.nonlinear:
        t = problem.encode([model.get(path).value for path, _p in problem.nonlinear])
        edges += [path for (path, _p), value in zip(problem.nonlinear, t) if value <= EDGE or value >= 1 - EDGE]
    edges += [path for path in problem.linear if model.get(path).value <= 0]
    return tuple(edges)


__all__ = ["FitData", "FitResult", "FitStopped", "fit", "fit_scales", "log_rmse", "residuals"]
