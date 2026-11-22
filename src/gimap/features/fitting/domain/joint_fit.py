"""Several curves fitted together with one χ² (Physics and Methods 9.18): SasView's simultaneous fit, IsGISAXS's joint
fit of a horizontal and a vertical cut.

Every curve has a model of its own (``FitModel``) with the same components: its own background, beam tail, factor k
and cut (a horizontal cut's angles and level; a vertical cut, ``FitModel.cut``). A parameter of ``shared`` (a path
``(component, key)`` or ``("globals", key)``) takes the first curve's value, freedom and range in every curve and is
fitted once; the others are each curve's own. The scales (Int, BG, tail A) are always each curve's own and solved
exactly for it at every step (variable projection, ``fit_engine``): every curve has its own scale and background, and
with several populations their ratio is not tied between the curves. The objective is the sum over the curves of
Σ r², r as in ``fit_engine`` ((I − m)/σ, or ln(I/m) for a curve without σ). Refine (``local``) and Search (``global``)
are those of ``fit_engine`` on the joint vector of the free non-linear values; the errors come from the Jacobian of
all residuals with respect to every free value (shared, per curve, the scales), cov = (JᵀJ)⁺ s², s² = Σr²/(N − p).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from . import fit_engine
from .fit_engine import LOCAL_STEP, PROGRESS_LINE, REFINING, SOLVING, FitData, FitStopped
from .fit_model import LINEAR, FitModel

SHARED = "shared"


@dataclass(frozen=True)
class JointCurve:
    name: str
    data: FitData
    model: FitModel


@dataclass(frozen=True)
class JointResult:
    models: tuple
    """The fitted model of every curve (shared values equal in all)."""
    shared: frozenset
    method: str
    points: tuple
    free: int
    """Free values fitted: shared and per-curve non-linear values and every curve's free scales."""
    chi2_reduced: float
    """Σ r² over every point of every curve / (N − free)."""
    curve_chi2: tuple
    """Σ r² / N of each curve (its share of the objective per point)."""
    weightings: tuple
    evaluations: int
    seconds: float
    converged: bool
    stopped: bool
    message: str
    errors: Mapping = field(default_factory=dict)
    """1σ errors: key ``("shared", path)`` or ``(curve index, path)``."""
    at_bounds: tuple = ()
    covariance: tuple = ()
    covariance_keys: tuple = ()
    start_curves: tuple = field(default=(), compare=False, repr=False)
    evaluation_budget: Optional[int] = None
    seed: int = 1729
    curve_dof: tuple = ()
    """Residual degrees of freedom per curve, N_c minus its share of the fitted Jacobian's leverage."""

    def reduced_chi2_for(self, index: int) -> float:
        """The cut's reduced χ² with supplied or Poisson σ. Relative and log residuals have no calibrated σ."""
        if self.weightings[index] != "sigma" or self.models[index].weighting == "relative":
            return math.nan
        dof = self.curve_dof[index]
        return self.curve_chi2[index] * self.points[index] / dof if dof > 1e-8 else math.nan


def default_shared(model: FitModel) -> frozenset:
    """The particles' shape and arrangement: every component parameter but its scale."""
    return frozenset((owner, key) for (owner, key), _p in model.parameters() if owner != "globals" and key != "Int")


def shared_curves(curves: Sequence[JointCurve], shared) -> list[JointCurve]:
    """The curves with the first curve's shared parameters (value, freedom, range, link) in every model. Refuses curves
    whose components differ, a shared scale and a shared parameter a model does not have."""
    if not curves:
        raise ValueError("Give at least one curve")
    reference = curves[0].model
    layout = [(c.family, c.keys()) for c in reference.components]
    out = []
    for curve in curves:
        if [(c.family, c.keys()) for c in curve.model.components] != layout:
            raise ValueError(f"{curve.name}: its particles differ from those of {curves[0].name}")
        model = curve.model
        for path in shared:
            if path[1] in LINEAR:
                raise ValueError("Scales (Int, background, tail A) are each curve's own: they cannot be shared")
            source = reference.get(path)
            model = model.with_parameter(path, value=source.value, free=source.free, lower=source.lower,
                                         upper=source.upper, tie=source.tie)
        out.append(replace(curve, model=model))
    return out


class _JointProblem:
    """The curves' problems (``fit_engine._Problem``) on one vector of non-linear values."""

    def __init__(self, curves: Sequence[JointCurve], shared, stop, progress, total: int):
        self.curves, self.shared, self.stop, self.progress, self.total = curves, frozenset(shared), stop, progress, total
        self.problems = [fit_engine._Problem(c.model, c.data, stop, None, 1) for c in curves]
        index: dict = {}
        self.keys, self.maps = [], []
        for number, problem in enumerate(self.problems):
            local = []
            for path, _p in problem.nonlinear:
                key = (SHARED, path) if path in self.shared else (number, path)
                if key not in index:
                    index[key] = len(self.keys)
                    self.keys.append(key)
                local.append(index[key])
            self.maps.append(np.array(local, int))
        self.nonlinear = self.keys
        self.evaluations = 0
        self.best = (math.inf, tuple(p.model for p in self.problems))
        self.phase = ""
        self._reported = 0.0
        self.points = sum(int(c.data.q.size) for c in curves)
        self.linear = sum(len(p.linear) for p in self.problems)

    def start(self) -> np.ndarray:
        t = np.full(len(self.keys), 0.5)
        for problem, where in zip(self.problems, self.maps):
            t[where] = problem.start()
        return t

    def solve(self, t) -> tuple[tuple, np.ndarray]:
        t = np.asarray(t, float)
        self.evaluations += 1
        models, parts = [], []
        for problem, where in zip(self.problems, self.maps):
            model, residual = problem.solve(t[where])
            models.append(model)
            parts.append(residual)
        residual = np.concatenate(parts) if parts else np.zeros(0)
        cost = float(residual @ residual)
        if cost < self.best[0]:
            self.best = (cost, tuple(models))
        self._report()
        return tuple(models), residual

    def residual(self, t) -> np.ndarray:
        return self.solve(t)[1]

    def cost(self, t) -> float:
        residual = self.solve(t)[1]
        return float(residual @ residual)

    def _report(self) -> None:
        now = time.perf_counter()
        if self.progress is None or now - self._reported < 0.25:
            return
        self._reported = now
        dof = max(1, self.points - len(self.keys) - self.linear)
        self.progress(min(0.99, self.evaluations / max(1, self.total)), PROGRESS_LINE.format(
            phase=self.phase, evaluations=self.evaluations, chi2=f"{self.best[0] / dof:.4g}"))


def joint_residuals(models: Sequence[FitModel], curves: Sequence[JointCurve]) -> np.ndarray:
    return np.concatenate([fit_engine.residuals(m, c.data) for m, c in zip(models, curves)])


def fit_curves(curves: Sequence[JointCurve], shared=None, *, method: str = "local",
               max_evaluations: Optional[int] = None, seed: int = 1729,
               progress: Optional[Callable[[float, str], None]] = None,
               stop: Optional[Callable[[], bool]] = None) -> JointResult:
    """Fit the curves together (module docstring); ``shared``: paths of the first curve (default ``default_shared``)."""
    if method not in ("local", "global"):
        raise ValueError(f"Unknown fitting method {method!r}")
    tick = time.perf_counter()
    shared = default_shared(curves[0].model) if shared is None else frozenset(tuple(p) for p in shared)
    start_curves = tuple(curves)
    curves = shared_curves(curves, shared)
    for curve in curves:
        if curve.data.q.size < 3:
            raise ValueError(f"{curve.name}: too few points to fit (at least 3)")
    probe = _JointProblem(curves, shared, stop, progress, 1)
    count = len(probe.keys) + probe.linear
    if probe.points <= count:
        raise ValueError("The curves have no more points than free values")
    budget = int(max_evaluations or (4000 if method == "global" else 150 * (count + 1)))
    problem = _JointProblem(curves, shared, stop, progress, budget)
    start_models = tuple(p.model for p in problem.problems)
    feasible = all(not p.free or p.lower <= p.value <= p.upper for m in start_models for _path, p in m.parameters())
    if feasible:
        initial = joint_residuals(start_models, curves)
        if np.all(np.isfinite(initial)):
            problem.best = (float(initial @ initial), start_models)
    stopped, converged, message = False, True, ""
    try:
        if not problem.keys:
            problem.phase = SOLVING
            problem.solve(np.zeros(0))
            message = "Only scales were free: solved exactly."
        else:
            starts = [problem.start()]
            if method == "global":
                starts = fit_engine._search(problem, budget, seed)
            problem.phase = REFINING
            local_budget = max(20, (budget if method == "local" else budget // 4) // len(starts))
            steps = max(5, local_budget // (len(problem.keys) + 1))
            best = None
            for start in starts:
                result = least_squares(lambda u, t0=start: problem.residual(t0 + u), np.zeros_like(start),
                                       bounds=(-start, 1.0 - start), x_scale=LOCAL_STEP, method="trf",
                                       max_nfev=steps, ftol=1e-6, xtol=1e-6, gtol=1e-8)
                if best is None or result.cost < best.cost:
                    best = result
            converged, message = bool(best.status > 0), str(best.message)
    except FitStopped:
        stopped, converged, message = True, False, "Stopped: the best values so far are kept."
    models = problem.best[1]
    residual = joint_residuals(models, curves)
    errors, covariance, keys, curve_dof = _covariance(models, curves, problem)
    edges = []
    for number, (model, sub) in enumerate(zip(models, problem.problems)):
        for path in fit_engine._at_bounds(model, sub):
            key = (SHARED, path) if path in shared else (number, path)
            if key not in edges:
                edges.append(key)
    dof = residual.size - count
    curve_chi2 = []
    for model, curve in zip(models, curves):
        r = fit_engine.residuals(model, curve.data)
        curve_chi2.append(float(r @ r) / max(1, r.size))
    return JointResult(
        models=tuple(models), shared=shared, method=method, points=tuple(int(c.data.q.size) for c in curves),
        free=count, chi2_reduced=float(residual @ residual) / dof if dof > 0 else math.nan,
        curve_chi2=tuple(curve_chi2), weightings=tuple(c.data.weighting for c in curves),
        evaluations=problem.evaluations, seconds=time.perf_counter() - tick, converged=converged and not stopped,
        stopped=stopped, message=message, errors=errors, at_bounds=tuple(edges),
        covariance=() if covariance is None else tuple(tuple(float(v) for v in row) for row in covariance),
        covariance_keys=tuple(keys), start_curves=start_curves, evaluation_budget=budget, seed=seed,
        curve_dof=curve_dof)


def _covariance(models, curves, problem: _JointProblem):
    """Return the 1σ errors, covariance, parameter keys and residual degrees of freedom per curve."""
    keys = list(problem.keys)
    for number, sub in enumerate(problem.problems):
        keys += [(number, path) for path in sub.linear]
    residual = joint_residuals(models, curves)
    points = [curve.data.q.size for curve in curves]
    if not keys:
        return {}, None, keys, tuple(points)

    def value_of(key) -> tuple[float, object]:
        owner, path = key
        model = models[0] if owner == SHARED else models[owner]
        return float(model.get(path).value), model.get(path)

    def moved(key, value):
        owner, path = key
        out = []
        for number, model in enumerate(models):
            if owner == SHARED or owner == number:
                model = model.with_values({path: value})
            out.append(problem.problems[number].links.apply(model) if problem.problems[number].links.ties else model)
        return out

    values, scale = [], []
    for key in keys:
        value, parameter = value_of(key)
        width = parameter.upper - parameter.lower
        values.append(value)
        scale.append(abs(value) if value else (float(width) if math.isfinite(width) and width > 0 else 1.0))
    scale = np.array(scale)
    jacobian = np.empty((residual.size, len(keys)))
    for column, key in enumerate(keys):
        step = 1e-4 * scale[column]
        with np.errstate(all="ignore"):
            upper = joint_residuals(moved(key, values[column] + step), curves)
            lower = joint_residuals(moved(key, values[column] - step), curves)
        jacobian[:, column] = (upper - lower) / (2.0 * step)
    if not np.all(np.isfinite(jacobian)):
        return {key: math.nan for key in keys}, None, keys, (math.nan,) * len(curves)
    variance = float(residual @ residual) / (residual.size - len(keys))
    scaled = jacobian * scale
    normal = scaled.T @ scaled
    covariance = np.linalg.pinv(normal) * variance * np.outer(scale, scale)
    diagonal = np.array(np.diag(covariance))
    size, vectors = np.linalg.eigh(normal)
    seen = size > fit_engine.UNSEEN * max(float(size[-1]), 1e-300)
    unseen = vectors[:, ~seen]
    diagonal[np.sum(unseen * unseen, axis=1) > 1e-6] = math.nan
    blind = ~(diagonal > 0)
    covariance[blind, :] = math.nan
    covariance[:, blind] = math.nan
    # Diagonal of the linearised hat matrix J J⁺, without forming an N × N matrix.
    basis = (scaled @ vectors[:, seen]) / np.sqrt(size[seen])
    leverage = np.sum(basis**2, axis=1)
    curve_dof = tuple(float(len(part) - part.sum()) for part in np.split(leverage, np.cumsum(points)[:-1]))
    return {key: float(np.sqrt(d)) if d > 0 else math.nan for key, d in zip(keys, diagonal)}, covariance, keys, curve_dof


def curve_errors(result: JointResult, number: int) -> dict:
    """The errors of curve ``number``'s values by path (shared ones included)."""
    out = {}
    for (owner, path), error in result.errors.items():
        if owner == SHARED or owner == number:
            out[path] = error
    return out


__all__ = ["JointCurve", "JointResult", "SHARED", "curve_errors", "default_shared", "fit_curves", "joint_residuals",
           "shared_curves"]
