"""The single-curve fit of Fitting (the API its page uses): the curve to fit, the model, the fit,
solutions from the other fitting methods, and what is exported.

* ``prepare_curve``: a curve with signed q (a horizontal GISAXS cut) becomes the points that are
  fitted — both halves on |q|, their mean (where both exist; beyond, the longer half alone), or
  one half — with σ carried along (σ of a mean: ½·√(σ₊² + σ₋²)); then the fitting range.
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

from ..domain.fit_engine import FitData, FitResult, FitStopped, fit, fit_scales, log_rmse, residuals
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
from ..domain.native_solution import native_solution_mapping

NM_PER_A = 10.0
SIDES = (
    ("mean", "Mean of both halves"),
    ("both", "Both halves on |q|"),
    ("positive", "q > 0 half"),
    ("negative", "q < 0 half"),
)
ANALYZE_SIDES = {"both_abs": "both", "mean": "mean", "positive": "positive", "negative": "negative"}


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


def prepare_curve(curve: Curve, side: str = "mean", q_range: Optional[tuple] = None, excluded=()) -> FitData:
    """The points fitted (|q| in nm⁻¹), in ``q_range`` (nm⁻¹) when given, without the ``excluded`` ones
    (``point_key`` values)."""
    if not curve.signed:
        q, intensity, sigma = np.abs(curve.q), curve.intensity, curve.sigma
    elif side in ("positive", "negative"):
        q, intensity, sigma = _half(curve, side == "positive")
    elif side == "both":
        q, intensity, sigma = np.abs(curve.q), curve.intensity, curve.sigma
    else:
        q, intensity, sigma = _mean_of_halves(curve)
    if q_range is not None:
        low, high = sorted(float(value) for value in q_range)
        keep = (q >= low) & (q <= high)
        q, intensity = q[keep], intensity[keep]
        sigma = None if sigma is None else sigma[keep]
    if excluded:
        keep = ~np.isin(np.array([point_key(value) for value in q]), np.array(sorted(excluded), dtype=float))
        q, intensity = q[keep], intensity[keep]
        sigma = None if sigma is None else sigma[keep]
    return FitData.prepare(q, intensity, sigma)


def _mean_of_halves(curve: Curve):
    qp, ip, sp = _half(curve, True)
    qn, i_n, sn = _half(curve, False)
    if qp.size < 2 or qn.size < 2:
        return (qp, ip, sp) if qp.size >= qn.size else (qn, i_n, sn)
    low, high = max(qp.min(), qn.min()), min(qp.max(), qn.max())
    main, other = ((qp, ip, sp), (qn, i_n, sn)) if qp.size >= qn.size else ((qn, i_n, sn), (qp, ip, sp))
    q, intensity = main[0].copy(), main[1].copy()
    sigma = None if main[2] is None or other[2] is None else main[2].copy()
    both = (q >= low) & (q <= high)
    intensity[both] = 0.5 * (intensity[both] + np.interp(q[both], other[0], other[1]))
    if sigma is not None:
        sigma[both] = 0.5 * np.hypot(sigma[both], np.interp(q[both], other[0], other[2]))
    beyond = other[0] > q.max()  # the longer half continues alone
    if beyond.any():
        q = np.concatenate([q, other[0][beyond]])
        intensity = np.concatenate([intensity, other[1][beyond]])
        if sigma is not None:
            sigma = np.concatenate([sigma, other[2][beyond]])
    return q, intensity, sigma


def model_from_solution(row: Mapping, base: Optional[FitModel] = None) -> tuple[FitModel, float]:
    """A quick-fit or 1D Predict solution as a model (free/fixed and ranges from ``base``), and the
    largest relative difference between the model's curve and the solution's own curve."""
    converted = native_solution_mapping(row)
    return from_manual(converted.mapping, base), converted.max_deviation


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
    record = {
        "model": model_to_dict(model),
        "equation": "I(q) = BG + k * [sum_i Int_i * P_i(q) * S_i(q; D_i, sigma_D_i/D_i) + A / (1 + (|q|/w)^nu)]",
        "curve": None if curve is None else {"name": curve.name, "path": curve.path, "q_unit_of_file": curve.source_unit},
        "points": {"side": side, "q_range_nm^-1": None if q_range is None else [float(v) for v in q_range],
                   "count": int(data.q.size), "weighting": data.weighting,
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
    "ANALYZE_SIDES", "FAMILIES", "GLOBALS", "INFO", "LINEAR", "NM_PER_A", "SIDES",
    "Component", "Curve", "FitData", "FitModel", "FitResult", "FitStopped", "Parameter",
    "evaluate", "export_table", "fit", "fit_record", "fit_scales", "from_manual", "log_rmse", "model_from_dict",
    "model_from_solution", "model_to_dict", "new_component", "parameter_text", "point_key", "prepare_curve",
    "residuals", "to_manual",
]
