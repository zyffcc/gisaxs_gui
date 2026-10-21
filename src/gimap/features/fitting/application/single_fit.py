"""The single-curve fit of Fitting (the API its page uses): the curve to fit, the model, the fit,
solutions from the other fitting methods, and what is exported.

* ``prepare_curve``: a curve with signed q (a horizontal GISAXS cut) becomes the points that are
  fitted — both halves on |q|, their mean (``shared/signed_cut.mean_of_halves``: where the other half
  has measured points; in a hole of it, and where one half reaches further, the measured half alone),
  or one half — with σ carried along (σ of a mean: ½·√(σ₊² + σ₋²)); then the fitting range and the
  points left out. ``prepare_points`` also says what the points are made of.
* ``model_from_solution``: a solution of the quick physical fit or of 1D Predict (``native_v5``)
  as a model, and how closely the model reproduces its curve.
* ``export_table`` / ``fit_record``: the points, the model and its terms, the residuals; and the
  JSON record of the model, the fit and its settings written next to them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

from src.gimap.shared.signed_cut import MISSING_HALF, MissingHalf, local_steps, mean_of_halves

from ..domain.fit_engine import (
    PHASES,
    PROGRESS_LINE,
    FitData,
    FitResult,
    FitStopped,
    fit,
    fit_scales,
    log_rmse,
    residuals,
)
from ..domain.fit_model import (
    FAMILIES,
    GLOBALS,
    INFO,
    LINEAR,
    Component,
    FitModel,
    Parameter,
    evaluate,
    from_manual,
    model_from_dict,
    model_to_dict,
    new_component,
    to_manual,
)
from ..domain import fit_edges
from ..domain.native_solution import native_solution_mapping

NM_PER_A = 10.0
SIDES = (
    ("mean", "Mean of both halves"),
    ("both", "Both halves on |q|"),
    ("positive", "q > 0 half"),
    ("negative", "q < 0 half"),
)
SERIES_SIDES = tuple(item for item in SIDES if item[0] != "both")
"""What a series offers (``shared/signed_cut.FIT_HALVES``): every frame is fitted on one curve."""
ANALYZE_SIDES = {"both_abs": "both", "mean": "mean", "positive": "positive", "negative": "negative"}
ONE_CURVE_METHODS = ("shapes", "ai", "fit1d")
"""The methods that take one curve with one measurement per |q| (Refine and Search also fit both halves)."""
BOTH_USES_MEAN = "Both halves on |q| cannot be fitted by this method: the mean of both halves is used."
"""Said when one of ``ONE_CURVE_METHODS`` runs while both halves are chosen: the page goes over to the mean,
so that the points it shows and records are the points that were fitted."""
QUICK_FIT_MIN_POINTS = 8
"""The quick physical fit needs this many points of the curve it fits (more than ``MAX_FIT_POINTS`` are merged)."""
TOO_FEW_FOR_QUICK_FIT = {
    "positive": "Only {points} points on the q > 0 half: this fit needs at least {minimum}",
    "negative": "Only {points} points on the q < 0 half: this fit needs at least {minimum}",
    "mean": "Only {points} points in the mean of both halves: this fit needs at least {minimum}",
    "": "Only {points} points in the curve: this fit needs at least {minimum}",
}
"""Why the quick physical fit refuses a curve, by what it would have fitted (``preparation["halves"]`` of
its rows; ``""``: a curve with one side, taken as it came). Templates: ``{points}``, ``{minimum}``."""


@dataclass(frozen=True)
class Curve:
    """A loaded curve: signed q in nm⁻¹, I, σ (or None), where it came from."""

    q: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray]
    name: str
    path: str = ""
    source_unit: str = "angstrom"

    @property
    def signed(self) -> bool:
        return bool(np.any(self.q < 0) and np.any(self.q > 0))

    @classmethod
    def from_arrays(cls, q, intensity, sigma=None, *, name: str, path: str = "", unit: str = "angstrom") -> "Curve":
        factor = NM_PER_A if unit == "angstrom" else 1.0
        q = np.asarray(q, dtype=float).reshape(-1) * factor
        intensity = np.asarray(intensity, dtype=float).reshape(-1)
        sigma = None if sigma is None else np.asarray(sigma, dtype=float).reshape(-1)
        if sigma is not None and (sigma.shape != q.shape or not np.any(np.isfinite(sigma) & (sigma > 0))):
            sigma = None
        keep = np.isfinite(q) & np.isfinite(intensity)
        return cls(q[keep], intensity[keep], None if sigma is None else sigma[keep], name, path, unit)


def _half(curve: Curve, positive: bool):
    keep = curve.q > 0 if positive else curve.q < 0
    q = np.abs(curve.q[keep])
    order = np.argsort(q)
    sigma = None if curve.sigma is None else curve.sigma[keep][order]
    return q[order], curve.intensity[keep][order], sigma


def point_key(q: float) -> float:
    """How a point is named when it is left out: its |q| (nm⁻¹) to nine significant digits."""
    return float(f"{abs(float(q)):.9g}")


def left_out_points(q, excluded) -> np.ndarray:
    """Which of the points ``q`` (|q| in nm⁻¹) the names in ``excluded`` (``point_key`` values) mean.

    A name means the point nearest to it when that is no further than half the step there — the same
    detector column in another frame of a series, whose q differs in the last digits — and any other
    point at the same |q| (both halves on |q|). A name in a hole of the curve means no point."""
    q = np.asarray(q, dtype=float)
    names = np.array(sorted(excluded), dtype=float)
    if not names.size or not q.size:
        return np.zeros(q.shape, dtype=bool)
    grid = np.unique(q)
    right = np.clip(np.searchsorted(grid, names), 1, max(grid.size - 1, 1))
    if grid.size == 1:
        near, step = np.zeros(names.shape, dtype=int), np.zeros(names.shape)
    else:
        near = np.where(names - grid[right - 1] <= grid[right] - names, right - 1, right)
        step = local_steps(grid)[right - 1]
    tolerance = np.maximum(0.5 * step, 1e-8 * np.abs(names))  # a name is its |q| to nine digits
    return np.isin(q, grid[near[np.abs(grid[near] - names) <= tolerance]])


@dataclass(frozen=True)
class PreparedCurve:
    """The points of a curve that are fitted, and what they are."""

    data: FitData
    halves_used: str
    """``mean``, ``positive``, ``negative`` or ``both``: what the points are made of. Under ``mean`` a curve
    with one half only — or a fitting range that only one half covers — is that half."""
    averaged: int = 0
    """Points that are the mean of two measurements (``mean`` only); the others come from one half alone."""
    left_out: int = 0
    """Points in the fitting range that the left-out names took away."""

    @property
    def points(self) -> int:
        return int(self.data.q.size)


def prepare_points(curve: Curve, side: str = "mean", q_range: Optional[tuple] = None, excluded=(), *,
                   strict: bool = False) -> PreparedCurve:
    """The points fitted (|q| in nm⁻¹) — in ``q_range`` (nm⁻¹) when given, without the ``excluded`` ones
    (``left_out_points``) — and what they are made of.

    ``side``: ``mean`` (a curve with one half only: that half), ``both`` (every point on |q|), ``positive``
    or ``negative``. A chosen half without points is ``MissingHalf`` when ``strict`` (a frame of a series,
    where the other half must never stand in for it); else the half there is is used — Single analysis,
    where a curve with points on one side only offers no choice of halves."""
    positive, negative = _half(curve, True), _half(curve, False)
    there = [name for name, half in (("positive", positive), ("negative", negative)) if half[0].size]
    halves = None
    if side == "both":
        q, intensity, sigma = np.abs(curve.q), curve.intensity, curve.sigma
        used = "both" if len(there) == 2 else (there[0] if there else "both")
    elif side in ("positive", "negative"):
        used = side
        if side not in there:
            if strict:
                raise MissingHalf(MISSING_HALF[side])
            used = there[0] if there else side
        q, intensity, sigma = positive if used == "positive" else negative
    elif len(there) == 2:
        mean, used = mean_of_halves(positive, negative), "mean"
        q, intensity, sigma, halves = mean.q, mean.intensity, mean.sigma, mean.halves
    else:
        used = there[0] if there else "mean"
        q, intensity, sigma = positive if used == "positive" else negative
    keep = np.ones(q.shape, dtype=bool)
    if q_range is not None:
        low, high = sorted(float(value) for value in q_range)
        keep = (q >= low) & (q <= high)
    left_out = left_out_points(q, excluded) & keep if len(excluded) else np.zeros(q.shape, dtype=bool)
    keep &= ~left_out
    data = FitData.prepare(q[keep], intensity[keep], None if sigma is None else sigma[keep])
    averaged = 0
    if halves is not None:
        halves = halves[keep][np.isin(q[keep], data.q)]  # of the points that are fitted
        averaged = int(np.count_nonzero(halves == 2))
        if halves.size and not averaged and np.all(halves == halves[0]):
            used = "positive" if halves[0] > 0 else "negative"
    return PreparedCurve(data, used, averaged, int(left_out.sum()))


def prepare_curve(curve: Curve, side: str = "mean", q_range: Optional[tuple] = None, excluded=(), *,
                  strict: bool = False) -> FitData:
    """The points fitted (|q| in nm⁻¹): ``prepare_points`` without the account of what they are."""
    return prepare_points(curve, side, q_range, excluded, strict=strict).data


def model_from_solution(row: Mapping, base: Optional[FitModel] = None) -> tuple[FitModel, float]:
    """A quick-fit or 1D Predict solution as a model (free/fixed and ranges from ``base``), and the
    largest relative difference between the model's curve and the solution's own curve."""
    converted = native_solution_mapping(row)
    return from_manual(converted.mapping, base), converted.max_deviation


def fit_resolving(model: FitModel, data: FitData, *, method: str = "local", max_evaluations=None, progress=None,
                  stop=None, resolve: bool = True) -> tuple:
    """A fit, then its parameters at an edge of their range answered and the model refined again
    (``domain.fit_edges``): (the result kept, notes of what was done, notes of what is still at an edge)."""
    result = fit(model, data, method=method, max_evaluations=max_evaluations, progress=progress, stop=stop)
    if not resolve:
        return result, [], []
    again = lambda changed: fit(changed, data, method="local", max_evaluations=max_evaluations,  # noqa: E731
                                progress=progress, stop=stop)
    return fit_edges.resolve(again, result, cancelled=stop or (lambda: False))


def parameter_text(key: str, value: float, error: Optional[float] = None) -> str:
    """``value ± error unit`` with the error's significant digits deciding the value's."""
    unit = INFO[key].unit
    if error is None or not math.isfinite(error) or error <= 0:
        text = f"{value:.4g}"
    else:
        digits = max(0, 1 - int(math.floor(math.log10(error)))) if error < 10 else 0
        text = f"{value:.{digits}f} ± {error:.{digits}f}" if digits < 8 else f"{value:.3g} ± {error:.2g}"
    return f"{text} {unit}".strip()


def export_table(model: FitModel, data: FitData) -> tuple[list[str], np.ndarray]:
    """Columns: q (nm⁻¹), q (Å⁻¹), I, σ, model, residual, then each term of the model."""
    total, parts = evaluate(model, data.q, parts=True)
    names = ["q_nm^-1", "q_A^-1", "I", "sigma", "model", "residual"]
    columns = [data.q, data.q / NM_PER_A, data.intensity,
               data.sigma if data.sigma is not None else np.full_like(data.q, np.nan), total, residuals(model, data)]
    for name, curve in parts.items():
        names.append(name.replace(" ", "_").replace(":", ""))
        columns.append(curve)
    return names, np.column_stack(columns)


def fit_record(model: FitModel, data: FitData, curve: Optional[Curve], *, side: str, q_range, result: Optional[FitResult],
               excluded=()) -> dict:
    """What the export writes next to the table: the model, the fit and how it was made."""
    made_of = None if curve is None else prepare_points(curve, side, q_range, excluded)
    record = {
        "model": model_to_dict(model),
        "equation": "I(q) = BG + k * [sum_i Int_i * P_i(q) * S_i(q; D_i, sigma_D_i/D_i) + A / (1 + (|q|/w)^nu)]",
        "curve": None if curve is None else {"name": curve.name, "path": curve.path, "q_unit_of_file": curve.source_unit},
        "points": {"side": side, "q_range_nm^-1": None if q_range is None else [float(v) for v in q_range],
                   "count": int(data.q.size), "weighting": data.weighting,
                   # what the points are (a curve with one half only is that half, whatever ``side`` says)
                   "halves_used": None if made_of is None else made_of.halves_used,
                   "averaged": None if made_of is None else made_of.averaged,
                   "left_out_q_nm^-1": sorted(float(value) for value in excluded)},
        "log_rmse": log_rmse(model, data),
    }
    if result is not None:
        record["fit"] = {
            "method": result.method, "chi2_reduced": result.chi2_reduced, "converged": result.converged,
            "evaluations": result.evaluations, "seconds": round(result.seconds, 3), "message": result.message,
            "errors": {f"{owner}:{key}": value for (owner, key), value in result.errors.items()},
            "at_bounds": [f"{owner}:{key}" for owner, key in result.at_bounds],
            "correlated": [[f"{a[0]}:{a[1]}", f"{b[0]}:{b[1]}", rho] for a, b, rho in result.correlated],
        }
    return record


__all__ = [
    "ANALYZE_SIDES", "FAMILIES", "GLOBALS", "INFO", "LINEAR", "NM_PER_A", "PHASES", "PROGRESS_LINE", "SIDES",
    "BOTH_USES_MEAN", "ONE_CURVE_METHODS", "QUICK_FIT_MIN_POINTS", "SERIES_SIDES", "TOO_FEW_FOR_QUICK_FIT",
    "Component", "Curve", "FitData", "FitModel", "FitResult", "FitStopped", "MissingHalf", "Parameter", "PreparedCurve",
    "evaluate", "export_table", "fit", "fit_record", "fit_resolving", "fit_scales", "from_manual", "left_out_points", "log_rmse",
    "model_from_dict", "model_from_solution", "model_to_dict", "new_component", "parameter_text", "point_key",
    "prepare_curve", "prepare_points", "residuals", "to_manual",
]
