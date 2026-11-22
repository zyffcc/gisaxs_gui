"""Fitting one curve with a ``FitModel``: bounded least squares, a wide search first, and the errors.

* The free scales (component Int, background, resolution-peak A) enter linearly: at every step
  they are solved by bounded weighted least squares for the other parameters (variable
  projection), so the search only moves sizes, spreads, distances and the peak shape.
* ``local``: bounded least squares (trust region) from the current values.
* ``global``: differential evolution over the ranges of the free non-linear parameters (a range
  spanning 10× or more is searched on a log scale), then ``local`` from the best three starts.
* Residuals: (I − model)/σ when σ is known, else ln(I/model) (every point the same relative weight).
* Errors: from the Jacobian of all free parameters at the solution, cov = (JᵀJ)⁻¹·s² with
  s² = Σr²/(N − p) — reported with the correlated pairs (|ρ| ≥ 0.95) and parameters left at a
  bound of their range, where the error is not meaningful. The covariance itself is kept
  (``FitResult.covariance``) for the correlation matrix.
* Links and inequalities (``fit_links``): a linked parameter is not searched; its value follows the others at every
  evaluation (a linked scale that reads fitted scales enters their columns, so the scales are still solved exactly),
  and its error is propagated from the covariance. An inequality maps its parameter between the bound and the far end
  of its range, like the contact D ≥ 2R. A model without them is fitted exactly as before.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional

import numpy as np
from scipy.optimize import differential_evolution, least_squares, lsq_linear
from scipy.stats import qmc

from . import fit_links, linked_fit
from .fit_resolution import INVALID_DQ
from .fit_model import LINEAR, NETWORK_FAMILIES, FitModel, evaluate, particle_forms, resolution_shape

TINY = 1e-300
CORRELATED = 0.95
EDGE = 1e-3
"""A non-linear parameter within this fraction of its range from a bound is reported as at the bound."""
CONTACT = 2.0
"""The paracrystal's distance D of a component is at least CONTACT · R: particles of the mean radius do not overlap."""
LOCAL_STEP = 0.02
"""First steps of a local fit, as a fraction of each parameter's range (log range when log-scaled)."""
SOLVING, SEARCHING, REFINING = PHASES = ("Solving the scales", "Searching the ranges", "Refining")
PROGRESS_LINE = "{phase} — {evaluations} evaluations, best χ²ᵣ {chi2}"
"""What a running fit says of itself (``progress(fraction, line)``): ``PROGRESS_LINE`` with one of ``PHASES``.
English, like every text of the domain; the page says it in the interface language."""


START_KEPT = "The fit did not improve on the start values: they are kept."
"""``FitResult.message`` when ``start_kept``."""
KEPT_MARGIN = 1e-6
"""The best curve tried must be worse than the start by more than this fraction of the start's Σr² for the start to
count as kept unimproved (a start already at the minimum is found again within rounding, and is converged)."""


class FitStopped(Exception):
    """Raised inside the optimiser when the person stops the fit; the best values so far are kept."""


@dataclass(frozen=True)
class FitData:
    q: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray] = None
    dq: Optional[np.ndarray] = None
    """Gaussian q-resolution widths in nm⁻¹ when using resolution from the file."""
    weight_mode: str = "file"

    @classmethod
    def prepare(cls, q, intensity, sigma=None, *, dq=None, weighting="file") -> "FitData":
        """|q| > 0 in nm⁻¹, finite points, sorted. With σ, the points whose σ is not a number above 0 are left out
        (``PreparedCurve.invalid_sigma`` counts them); σ is dropped only when no point has one."""
        q = np.abs(np.asarray(q, dtype=float)).reshape(-1)
        y = np.asarray(intensity, dtype=float).reshape(-1)
        keep = np.isfinite(q) & np.isfinite(y) & (q > 0)
        if weighting == "file":
            s = None if sigma is None else np.asarray(sigma, dtype=float).reshape(-1)
            if s is not None and s.shape == q.shape and np.any(keep & np.isfinite(s) & (s > 0)):
                keep &= np.isfinite(s) & (s > 0)
            else:
                s = None  # the existing fallback for a file with no usable σ
        elif weighting in ("relative", "poisson", "none"):
            keep &= y > 0
            s = None
            if weighting == "relative":
                s = y
            elif weighting == "poisson":
                s = np.sqrt(np.maximum(y, 0))
        else:
            raise ValueError(f"Unknown weighting {weighting!r}")
        if s is None:
            keep &= y > 0  # logarithmic residuals need I > 0
        if dq is not None:
            dq = np.asarray(dq, dtype=float).reshape(-1)
            if dq.shape != q.shape or not np.all(np.isfinite(dq[keep]) & (dq[keep] >= 0)):
                raise ValueError(INVALID_DQ)
        order = np.argsort(q[keep])
        return cls(q[keep][order], y[keep][order], None if s is None else s[keep][order],
                   None if dq is None else dq[keep][order], weighting)

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
    start_kept: bool = False
    """No curve the fit tried was better than the starting one (by the reported objective), so the start values are
    returned unchanged: not a converged fit, and ``errors`` are not those of a minimum that was found."""
    covariance: tuple = ()
    """The covariance matrix of the free parameters (rows of ``covariance_paths``; NaN in the rows of a direction
    the curve does not see); empty when it could not be computed."""
    covariance_paths: tuple = ()
    restarts: int = 0
    """How many times the least squares was run again from its own result and improved it (a manual fit,
    ``single_fit.fit_resolving``); 0 for a fit made once."""


def stopped_result(result: FitResult) -> FitResult:
    """Keep completed numerical work while marking the whole request as interrupted."""
    return replace(result, stopped=True, converged=False, message="Stopped: the best values so far are kept.")


def residuals(model: FitModel, data: FitData) -> np.ndarray:
    """(I − model)/σ, or ln(I/model) without σ (the residual plot)."""
    return _residual(evaluate(model, data.q, dq=data.dq), data)


def _residual(predicted: np.ndarray, data: FitData) -> np.ndarray:
    if data.sigma is not None:
        return (data.intensity - predicted) / data.sigma
    return np.log(data.intensity) - np.log(np.maximum(predicted, TINY))


def log_rmse(model: FitModel, data: FitData) -> float:
    predicted = evaluate(model, data.q, dq=data.dq)
    good = (data.intensity > 0) & (predicted > 0)
    if not good.any():
        return math.nan
    return float(np.sqrt(np.mean(np.log(data.intensity[good] / predicted[good]) ** 2)))


class _Problem:
    """The free parameters of a model split into non-linear (searched) and linear (solved)."""

    def __init__(self, model: FitModel, data: FitData, stop, progress, total: int):
        self.links = fit_links.Links(model)
        if self.links.ties:
            model = self.links.apply(model)
        self.model, self.data, self.stop, self.progress, self.total = model, data, stop, progress, max(1, total)
        free = [(path, parameter) for path, parameter in model.parameters() if parameter.free and not parameter.tie]
        for path, parameter in free:
            if not (math.isfinite(parameter.lower) and parameter.lower < parameter.upper):
                raise ValueError(f"Give {path[1]} a valid range (min < max) to fit it")
        self.nonlinear = [(path, p) for path, p in free if path[1] not in LINEAR]
        self.linear = [path for path, p in free if path[1] in LINEAR]
        for path, parameter in self.nonlinear:
            if not (math.isfinite(parameter.lower) and math.isfinite(parameter.upper) and parameter.upper > parameter.lower):
                raise ValueError(f"Give {path[1]} a finite range (min < max) to fit it")
        self.lower = np.array([p.lower for _path, p in self.nonlinear], dtype=float)
        self.upper = np.array([p.upper for _path, p in self.nonlinear], dtype=float)
        self.pairs = self._contact(model) if model.contact else []
        self.logarithmic = (self.lower > 0) & (self.upper >= 10.0 * np.maximum(self.lower, TINY))
        self.bounded = linked_fit.inequalities(self, model)
        self.linear_ties = linked_fit.linear_ties(self)
        self.weights = 1.0 / data.sigma if data.sigma is not None else 1.0 / data.intensity
        self.evaluations = 0
        self.cache: dict = {}
        self.best = (math.inf, model)
        self.lowest = math.inf  # the best Σr² among the curves tried (bookkeeping only: ``fit`` reports with it)
        self._reported = 0.0
        self.phase = ""

    def _contact(self, model: FitModel) -> list[tuple[int, int]]:
        """D ≥ CONTACT · R in every component with a paracrystal. With one of the two fixed, the other's range is
        narrowed here; with both free, the pairs ``(index of D, index of R)`` among the non-linear values are returned and
        D is searched between CONTACT · R and its maximum (``decode``). Values the person fixed are left as they are."""
        where = {path: index for index, (path, _p) in enumerate(self.nonlinear)}
        pairs = []
        for owner, component in enumerate(model.components):
            if "D" not in component.keys() or component.family not in NETWORK_FAMILIES:
                continue  # (the families of ``model_library``: ``_inequalities``)
            d_index, r_index = where.get((owner, "D")), where.get((owner, "R"))
            if d_index is not None and r_index is not None:
                pairs.append((d_index, r_index))
            elif d_index is not None:
                self.lower[d_index] = min(max(self.lower[d_index], CONTACT * component.value("R")),
                                          np.nextafter(self.upper[d_index], -np.inf))
            elif r_index is not None:
                self.upper[r_index] = max(min(self.upper[r_index], component.value("D") / CONTACT),
                                          np.nextafter(self.lower[r_index], np.inf))
        return pairs

    # -- the unit cube ↔ values -----------------------------------------------------

    def _floor(self, values, d_index: int, r_index: int) -> float:
        """The lowest D of a pair at the radius in ``values`` (just below D's maximum when 2R is beyond it)."""
        return min(max(self.lower[d_index], CONTACT * values[r_index]), self.upper[d_index] * (1.0 - 1e-9))

    def _to_value(self, t: float, low: float, high: float, logarithmic: bool) -> float:
        if logarithmic:
            return float(np.exp(np.log(low) + t * (np.log(high) - np.log(low))))
        return low + t * (high - low)

    def _to_unit(self, value: float, low: float, high: float, logarithmic: bool) -> float:
        value = min(max(value, low), high)
        if logarithmic:
            return float((np.log(value) - np.log(low)) / (np.log(high) - np.log(low)))
        return (value - low) / (high - low)

    def decode(self, t) -> np.ndarray:
        t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
        values = self.lower + t * (self.upper - self.lower)
        if self.logarithmic.any():
            lo, hi = np.log(self.lower[self.logarithmic]), np.log(self.upper[self.logarithmic])
            values[self.logarithmic] = np.exp(lo + t[self.logarithmic] * (hi - lo))
        values = np.clip(values, self.lower, self.upper)
        for d_index, r_index in self.pairs:
            values[d_index] = self._to_value(t[d_index], self._floor(values, d_index, r_index), self.upper[d_index],
                                             bool(self.logarithmic[d_index]))
        if self.bounded:
            linked_fit.bound(self, values, t)
        return values

    def encode(self, values) -> np.ndarray:
        values = np.clip(np.asarray(values, dtype=float), self.lower, self.upper)
        t = (values - self.lower) / (self.upper - self.lower)
        if self.logarithmic.any():
            lo, hi = np.log(self.lower[self.logarithmic]), np.log(self.upper[self.logarithmic])
            t[self.logarithmic] = (np.log(values[self.logarithmic]) - lo) / (hi - lo)
        for d_index, r_index in self.pairs:
            t[d_index] = self._to_unit(values[d_index], self._floor(values, d_index, r_index), self.upper[d_index],
                                       bool(self.logarithmic[d_index]))
        if self.bounded:
            units = linked_fit.bound(self, np.array(values, dtype=float))
            for index, _relation, _expression in self.bounded:
                t[index] = units[index]
        return np.clip(t, 1e-9, 1.0 - 1e-9)

    def start(self) -> np.ndarray:
        return self.encode([p.value for _path, p in self.nonlinear])

    # -- one evaluation: set the non-linear values, solve the scales -----------------

    def solve(self, t) -> tuple[FitModel, np.ndarray]:
        if self.stop is not None and self.stop():
            raise FitStopped()
        self.evaluations += 1
        model = self.model.with_values({path: value for (path, _p), value in zip(self.nonlinear, self.decode(t))})
        valid = True
        if self.links.ties:
            try:
                model = self.links.apply(model)
            except ValueError:  # a link that cannot be evaluated here (log of a negative value …): a bad point
                valid = False
        if self.linear and valid:
            model = linked_fit.scales_linked(self, model) if self.linear_ties else self._scales(model)
        try:
            if not valid:
                raise ValueError("a link cannot be evaluated")
            with np.errstate(all="ignore"):
                residual = _residual(evaluate(model, self.data.q, cache=self.cache, dq=self.data.dq), self.data)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            residual = np.full(self.data.q.shape, 1e6)
        if not np.all(np.isfinite(residual)):
            residual = np.where(np.isfinite(residual), residual, 1e6)
        cost = float(residual @ residual)
        self.lowest = min(self.lowest, cost)
        if cost < self.best[0] and valid:
            self.best = (cost, model)
        self._report(cost)
        return model, residual

    def _scales(self, model: FitModel) -> FitModel:
        q, g = self.data.q, model.globals
        k = float(g["k"].value)
        columns, fixed = [], np.full_like(q, 0.0 if ("globals", "background") in self.linear else g["background"].value)
        forms = particle_forms(model, q, self.data.dq, self.cache, scale_columns=True)
        for index, component in enumerate(model.components):
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
            return model.with_values({path: np.clip(model.get(path).value, model.get(path).lower,
                                                    model.get(path).upper) for path in self.linear})
        # Normalise columns: particle scales and background can differ by many decades.
        scales = np.maximum(np.linalg.norm(matrix, axis=0), np.finfo(float).tiny)
        low = np.array([model.get(path).lower for path in self.linear])
        high = np.array([model.get(path).upper for path in self.linear])
        solved = lsq_linear(matrix / scales, target, bounds=(low * scales, high * scales),
                            method="bvls", tol=1e-10, max_iter=100 * len(self.linear))
        coefficients = np.clip(solved.x / scales, low, high)
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
        self.progress(min(0.99, self.evaluations / self.total), PROGRESS_LINE.format(
            phase=self.phase, evaluations=self.evaluations, chi2=f"{self.best[0] / points:.4g}"))


def fit(model: FitModel, data: FitData, *, method: str = "local", max_evaluations: Optional[int] = None,
        seed: int = 1729, progress: Optional[Callable[[float, str], None]] = None,
        stop: Optional[Callable[[], bool]] = None, seeds: tuple = ()) -> FitResult:
    """Fit the free parameters of ``model`` to ``data`` (q in nm⁻¹). ``seeds``: models of the same parameters whose
    values join the first population of ``global`` (the proposal network's samples)."""
    if method not in ("local", "global"):
        raise ValueError(f"Unknown fitting method {method!r}")
    if data.q.size < 3:
        raise ValueError("Too few points to fit (at least 3 in the fitting range)")
    tick = time.perf_counter()
    model = fit_links.apply_links(model)  # the model itself when it has no link
    count = sum(1 for _path, p in model.parameters() if p.free and not p.tie)
    budget = int(max_evaluations or (4000 if method == "global" else 150 * (count + 1)))
    problem = _Problem(model, data, stop, progress, budget)
    # The starting curve is a valid contender. Profiling linear scales uses relative
    # weights without sigma, whereas the reported objective is logarithmic.
    # A completed refinement must never replace a better feasible starting curve.
    if all(not p.free or p.lower <= p.value <= p.upper for _path, p in model.parameters()):
        initial = residuals(model, data)
        if np.all(np.isfinite(initial)):
            problem.best = (float(initial @ initial), model)
    stopped, converged, message = False, True, ""
    start_cost = problem.best[0]
    try:
        if not problem.nonlinear:
            problem.phase = SOLVING
            problem.solve(np.zeros(0))
            message = "Only scales were free: solved exactly."
        else:
            starts = [problem.start()]
            if method == "global":
                starts = _search(problem, budget, seed,
                                 [problem.encode([m.get(path).value for path, _p in problem.nonlinear]) for m in seeds])
            problem.phase = REFINING
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
    # Reported, never acted on: the values returned are the same either way.
    kept = (not stopped and fitted is model and count > 0 and math.isfinite(start_cost)
            and problem.lowest > start_cost * (1.0 + KEPT_MARGIN))
    if kept:
        converged, message = False, START_KEPT
    errors, correlated, covariance, paths = _covariance(fitted, data, problem)
    return FitResult(
        model=fitted, method=method, weighting=data.weighting, points=int(data.q.size), free=count,
        chi2_reduced=_chi2(fitted, data, count), log_rmse=log_rmse(fitted, data), evaluations=problem.evaluations,
        seconds=time.perf_counter() - tick, converged=converged and not stopped, stopped=stopped, message=message,
        errors=errors, correlated=correlated, at_bounds=_at_bounds(fitted, problem), start_kept=kept,
        covariance=() if covariance is None else tuple(tuple(float(v) for v in row) for row in covariance),
        covariance_paths=tuple(paths),
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
    solved = linked_fit.scales_linked(problem, problem.model) if problem.linear_ties else problem._scales(held)
    fitted = model.with_values({path: solved.get(path).value for path in problem.linear})
    return problem.links.apply(fitted) if problem.links.ties else fitted


def _search(problem: _Problem, budget: int, seed: int, extra=()) -> list[np.ndarray]:
    """Differential evolution over the unit cube (the start and ``extra`` members among the first population, the rest
    a Latin hypercube); the best three distinct members as starts."""
    problem.phase = SEARCHING
    dimension = len(problem.nonlinear)
    size = max(8, 5 * dimension)
    population = qmc.LatinHypercube(d=dimension, seed=seed).random(size)
    population[0] = problem.start()
    for row, member in enumerate(list(extra)[:size // 2], start=1):
        population[row] = member
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


UNSEEN = 1e-12
"""Eigenvalues of JᵀJ (columns in relative units) below this fraction of the largest: directions without error."""


def _errors(model: FitModel, data: FitData, problem: _Problem) -> tuple[dict, tuple]:
    """1σ errors of every free parameter, and the strongly correlated pairs."""
    return _covariance(model, data, problem)[:2]


def _covariance(model: FitModel, data: FitData, problem: _Problem) -> tuple[dict, tuple, Optional[np.ndarray], list]:
    """1σ errors of every free parameter (and of the linked ones, propagated), the strongly correlated pairs, the
    covariance matrix (None when it could not be computed) and its parameters."""
    paths = [path for path, _p in problem.nonlinear] + list(problem.linear)
    if not paths or data.q.size <= len(paths):
        return {path: math.nan for path in paths}, (), None, paths
    links = problem.links
    values = np.array([model.get(path).value for path in paths], dtype=float)
    # The size of every parameter on its own (its range when it is 0): the finite-difference step
    # and the unit of its column. A parameter of 1e30 next to R ≈ 5 nm must not set R's step.
    scale = np.array([abs(value) if value else _width(model.get(path)) for path, value in zip(paths, values)])
    jacobian = np.empty((data.q.size, len(paths)))
    for column, (path, value) in enumerate(zip(paths, values)):
        step = 1e-4 * scale[column]
        with np.errstate(all="ignore"):
            upper = residuals(links.apply(model.with_values({path: value + step})), data)
            lower = residuals(links.apply(model.with_values({path: value - step})), data)
        jacobian[:, column] = (upper - lower) / (2.0 * step)
    residual = residuals(model, data)
    variance = float(residual @ residual) / (data.q.size - len(paths))
    if not np.all(np.isfinite(jacobian)):
        return {path: math.nan for path in paths}, (), None, paths
    # Columns in relative units, so that no parameter pushes the others below the cut-off of the inverse.
    scaled = jacobian * scale
    normal = scaled.T @ scaled
    covariance = np.linalg.pinv(normal) * variance * np.outer(scale, scale)
    diagonal = np.array(np.diag(covariance))
    # A direction the curve does not see (the shape of a population of amplitude 0, two values that only enter
    # together): the pseudo-inverse leaves it out, which would read as an error of 0. Its values have no error.
    size, vectors = np.linalg.eigh(normal)
    unseen = vectors[:, size <= UNSEEN * max(float(size[-1]), 1e-300)]
    diagonal[np.sum(unseen * unseen, axis=1) > 1e-6] = math.nan
    errors = {path: float(np.sqrt(d)) if d > 0 else math.nan for path, d in zip(paths, diagonal)}
    correlated = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if diagonal[i] > 0 and diagonal[j] > 0:
                rho = covariance[i, j] / math.sqrt(diagonal[i] * diagonal[j])
                if abs(rho) >= CORRELATED:
                    correlated.append((paths[i], paths[j], float(rho)))
    shown = np.array(covariance)
    blind = ~(diagonal > 0)
    shown[blind, :] = math.nan
    shown[:, blind] = math.nan
    if links.ties:
        errors.update(linked_fit.propagated(model, links, paths, values, scale, shown))
    return errors, tuple(correlated), shown, paths


def _width(parameter) -> float:
    width = parameter.upper - parameter.lower
    return float(width) if math.isfinite(width) and width > 0 else 1.0


def _at_bounds(model: FitModel, problem: _Problem) -> tuple:
    edges = []
    if problem.nonlinear:
        t = problem.encode([model.get(path).value for path, _p in problem.nonlinear])
        edges += [path for (path, _p), value in zip(problem.nonlinear, t) if value <= EDGE or value >= 1 - EDGE]
    for path in problem.linear:
        p = model.get(path)
        tolerance = EDGE * (p.upper - p.lower) if math.isfinite(p.upper) else 1e-8 * max(1, abs(p.value))
        if p.value <= p.lower + tolerance or p.value >= p.upper - tolerance:
            edges.append(path)
    return tuple(edges)


def errors(model: FitModel, data: FitData) -> tuple[dict, tuple]:
    """The errors of the free parameters at the model's values for ``data`` (and the strongly correlated pairs), as
    a fit reports them, without fitting: e.g. with σ that include a model error."""
    return _errors(model, data, _Problem(model, data, None, None, 1))


contact_inequalities = linked_fit.contact_inequalities

__all__ = ["PHASES", "PROGRESS_LINE", "START_KEPT", "FitData", "FitResult", "FitStopped", "contact_inequalities", "errors",
           "fit", "fit_scales",
           "log_rmse", "residuals", "stopped_result"]
