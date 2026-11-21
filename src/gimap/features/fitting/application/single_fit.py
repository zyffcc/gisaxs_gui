"""The single-curve fit of Fitting (the API its page uses): the curve to fit, the model, the fit,
solutions from the other fitting methods, and what is exported.

* ``prepare_curve``: a curve with signed q (a horizontal GISAXS cut) becomes the points that are
  fitted — both halves on |q|, their mean (``shared/signed_cut.mean_of_halves``: where the other half
  has measured points; in a hole of it, and where one half reaches further, the measured half alone),
  or one half — with σ carried along (σ of a mean: ½·√(σ₊² + σ₋²)); then the fitting range and the
  points left out. ``prepare_points`` also says what the points are made of.
* ``model_from_solution``: a solution of Predict parameters (a row of 1D Predict or of Analyze, ``native_v5``)
  as a model, and how closely the model reproduces its curve.
* ``export_table`` / ``fit_record``: the points, the model and its terms, the residuals; and the
  JSON record of the model, the fit and its settings written next to them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional

import numpy as np

from src.gimap.shared.signed_cut import MISSING_HALF, MissingHalf, local_steps, mean_of_halves
from src.gimap.shared.version import SOFTWARE

from ..domain.fit_engine import (
    PHASES,
    PROGRESS_LINE,
    FitData,
    FitResult,
    START_KEPT,
    FitStopped,
    fit,
    fit_scales,
    log_rmse,
    residuals,
    stopped_result,
)
from ..domain.fit_model import (
    BASE_GLOBALS,
    DEFAULT_INTERFERENCE,
    FAMILIES,
    INTERFERENCES,
    NETWORK_FAMILIES,
    GLOBALS,
    GRAZING,
    INFO,
    LINEAR,
    RESOLUTION,
    SIZE_LAWS,
    Component,
    FitModel,
    Parameter,
    evaluate,
    from_manual,
    grazing,
    model_from_dict,
    model_to_dict,
    new_component,
    intensity_size,
    number_size,
    qz_plus_r,
    sigma_nn_over_d,
    to_manual,
    with_interference,
)
from ..domain import beam_tail, fit_edges
from ..domain.fit_limits import constrain_proposal, explicit_limits
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
ONE_CURVE_METHODS = ("predict",)
"""The methods that take one curve with one measurement per |q| (Refine and Search also fit both halves)."""
BOTH_USES_MEAN = "Both halves on |q| cannot be fitted by this method: the mean of both halves is used."
"""Said when one of ``ONE_CURVE_METHODS`` runs while both halves are chosen: the page goes over to the mean,
so that the points it shows and records are the points that were fitted."""
CURVE_FIT_MIN_POINTS = 8
"""Predict parameters on one cut (``bootstrap.create_curve_fit``) needs this many points (more than ``MAX_FIT_POINTS``
are merged)."""
TOO_FEW_FOR_CURVE_FIT = {
    "positive": "Only {points} points on the q > 0 half: this fit needs at least {minimum}",
    "negative": "Only {points} points on the q < 0 half: this fit needs at least {minimum}",
    "mean": "Only {points} points in the mean of both halves: this fit needs at least {minimum}",
    "": "Only {points} points in the curve: this fit needs at least {minimum}",
}
"""Why the fit of one cut (``bootstrap.create_curve_fit``) refuses a curve, by what it would have fitted (``preparation["halves"]`` of
its rows; ``""``: a curve with one side, taken as it came). Templates: ``{points}``, ``{minimum}``."""


TOO_FEW_FOR_FREE = ("Only {points} points in the fitting range for {free} free values: a fit needs at least "
                    "{minimum}. Fix some values or widen the fitting range.")
NO_POSITIVE_POINT = "No point of the curve has an intensity above zero: there is nothing a model can be fitted to."
"""Why Refine / Search refuse a curve (``fit_resolving``). ``TOO_FEW_FOR_FREE``: ``{points}``, ``{free}``, ``{minimum}``."""


@dataclass(frozen=True)
class Curve:
    """A loaded curve: signed q in nm⁻¹, I, σ (or None), where it came from."""

    q: np.ndarray
    intensity: np.ndarray
    sigma: Optional[np.ndarray]
    name: str
    path: str = ""
    source_unit: str = "angstrom"
    unit_from: str = ""
    """Where ``source_unit`` came from (``curve_units``: stated in the file, the setting for files without one …)."""
    rows_left_out: int = 0
    """Rows of the file that are not in the curve (not numbers, NaN or inf): said with the curve."""
    columns: Optional[dict] = None
    """The columns of a table read as q, I and σ (``ColumnMapping.to_dict``; a CSV: chosen by its header's names, or
    by the person); ``None``: a file read by place or an NXcanSAS curve."""

    @property
    def signed(self) -> bool:
        return bool(np.any(self.q < 0) and np.any(self.q > 0))

    @classmethod
    def from_arrays(cls, q, intensity, sigma=None, *, name: str, path: str = "", unit: str = "angstrom",
                    unit_from: str = "", rows_left_out: int = 0, columns: Optional[dict] = None) -> "Curve":
        factor = NM_PER_A if unit == "angstrom" else 1.0
        q = np.asarray(q, dtype=float).reshape(-1) * factor
        intensity = np.asarray(intensity, dtype=float).reshape(-1)
        sigma = None if sigma is None else np.asarray(sigma, dtype=float).reshape(-1)
        if sigma is not None and (sigma.shape != q.shape or not np.any(np.isfinite(sigma) & (sigma > 0))):
            sigma = None
        keep = np.isfinite(q) & np.isfinite(intensity)
        return cls(q[keep], intensity[keep], None if sigma is None else sigma[keep], name, path, unit, unit_from,
                   int(rows_left_out) + int(q.size - np.count_nonzero(keep)), columns)


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
    negative: int = 0
    zero: int = 0
    """Points of the selection with a negative / zero intensity: fitted with their σ, not fitted without σ (ln I)."""
    invalid_sigma: int = 0
    """Points left out because their σ is zero, negative or not a number while other points have one."""

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
        # the q > 0 half gives the |q| of the mean, as in Analyze's curve and the series map: one cut, one mean curve
        mean, used = mean_of_halves(positive, negative, grid="positive"), "mean"
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
    measured = intensity[keep][np.isfinite(q[keep]) & (q[keep] > 0)]
    measured = measured[np.isfinite(measured)]
    candidates = int(np.count_nonzero(np.isfinite(q[keep]) & (q[keep] > 0) & np.isfinite(intensity[keep])))
    invalid = candidates - int(data.q.size) if data.sigma is not None else 0
    return PreparedCurve(data, used, averaged, int(left_out.sum()), int(np.count_nonzero(measured < 0)),
                         int(np.count_nonzero(measured == 0)), max(0, invalid))


def prepare_curve(curve: Curve, side: str = "mean", q_range: Optional[tuple] = None, excluded=(), *,
                  strict: bool = False) -> FitData:
    """The points fitted (|q| in nm⁻¹): ``prepare_points`` without the account of what they are."""
    return prepare_points(curve, side, q_range, excluded, strict=strict).data


CUT_MODEL_FAMILIES = {"sphere": "sphere", "random_cylinder": "cylinder", "cylinder": "cylinder",
                      "vertical_cylinder": "vertical_cylinder"}


def cut_model_solution(row: Mapping) -> FitModel:
    """A solution in Fitting's own conventions (``physics_backend`` "cut_model": Predict parameters): its values as they
    are, amplitudes as Int."""
    components = []
    for item in row.get("components") or ():
        p = item["params"]
        family = CUT_MODEL_FAMILIES[str(item["type"])]
        component = new_component(family, radius=float(p["R"]), distance=float(p["D"]),
                                  structure=bool(item.get("structure_factor", True)), law=row.get("law", "number_gauss"))
        params = dict(component.params)
        for key in ("sigma_R", "h", "sigma_h", "sigma_nn"):
            if key in params and p.get(key) is not None:
                params[key] = _within(params[key], float(p[key]))
        params["Int"] = _within(params["Int"], float(item.get("amplitude", 1.0)))
        components.append(Component(family, params, component.structure, component.coupling, component.law))
    model = FitModel(tuple(components))
    g = row.get("global_params") or {}
    for key, source in (("background", "background"), ("res_amplitude", "resolution_amplitude"),
                        ("res_width", "sigma_Res"), ("res_exponent", "nu_Res"), ("q_resolution", "q_resolution"),
                        ("alpha_i", "alpha_i"), ("alpha_f", "alpha_f"), ("wavelength", "wavelength")):
        if g.get(source) is not None:
            model = model.with_parameter(("globals", key), **_within(model.globals[key], float(g[source])).__dict__)
    return model


def _within(parameter: Parameter, value: float) -> Parameter:
    """``parameter`` at ``value``, its range widened to hold it."""
    return Parameter(value, parameter.free, min(parameter.lower, value), max(parameter.upper, value))


def model_from_solution(row: Mapping, base: Optional[FitModel] = None) -> tuple[FitModel, float]:
    """A quick-fit or 1D Predict solution as a model (free/fixed and ranges from ``base``), and the
    largest relative difference between the model's curve and the solution's own curve."""
    if row.get("physics_backend") == "cut_model":
        proposed = cut_model_solution(row)
        model = constrain_proposal(proposed, base) if base is not None else proposed
        return model, solution_deviation(model, row)
    converted = native_solution_mapping(row)
    proposed = from_manual(converted.mapping)
    model = constrain_proposal(proposed, base) if base is not None else proposed
    return model, solution_deviation(model, row)


def solution_deviation(model: FitModel, row: Mapping) -> float:
    """Difference from the source curve after conversion, constraints or refinement."""
    q = np.asarray(row.get("native_q") or (), dtype=float)
    target = np.asarray(row.get("native_fit") or (), dtype=float)
    if not q.size or q.shape != target.shape:
        return float("nan")
    keep = np.isfinite(q) & np.isfinite(target) & (target > 0)
    return float(np.max(np.abs(evaluate(model, np.abs(q[keep])) / target[keep] - 1))) if keep.any() else float("nan")


def fit_resolving(model: FitModel, data: FitData, *, method: str = "local", max_evaluations=None, progress=None,
                  stop=None, resolve: bool = True) -> tuple:
    """Fit under hard Model limits; return the result, no automatic changes, and boundary diagnostics.

    Refused (``ValueError``): fewer points than free values plus one, and a curve without a positive intensity.
    A fit that kept its start values (``FitResult.start_kept``) claims no errors."""
    free = sum(1 for _path, parameter in model.parameters() if parameter.free)
    if not np.any(data.intensity > 0):
        raise ValueError(NO_POSITIVE_POINT)
    if data.q.size <= free:
        raise ValueError(TOO_FEW_FOR_FREE.format(points=int(data.q.size), free=free, minimum=free + 1))
    result = fit(model, data, method=method, max_evaluations=max_evaluations, progress=progress, stop=stop)
    if result.start_kept:
        result = replace(result, errors={path: math.nan for path in result.errors}, correlated=())
    if stop is not None and stop():
        return stopped_result(result), [], []
    # Model min/max are a contract, including when the old Resolve checkbox was saved on.
    # Retain edge diagnostics without widening or fixing any parameter behind the user's back.
    return fit_edges.resolve(lambda changed: result, result, rounds=0, cancelled=stop or (lambda: False))


def tail_for_curve(model: FitModel, data: FitData, *, start: bool = False) -> FitModel:
    """The beam tail's ranges tied to the curve (``domain.beam_tail``) while they are the generic ones and the tail
    is in use (A free or above 0); ``start``: also w and ν from the first points (a starting model). Applied where a
    model meets a curve (the page opens it, a fit starts from generic ranges, a frame of a series), never inside
    ``fit_resolving``: the fit keeps every range of the model it is given."""
    tail = model.globals["res_amplitude"]
    if not data.q.size or not beam_tail.is_generic(model) or not (tail.free or tail.value > 0):
        return model
    return beam_tail.tied_to_curve(model, data, with_start=start)


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


def derived_values(component: Component) -> dict:
    size = number_size(component)  # None: not defined for this intensity-weighted width (fit_model.number_size)
    by_intensity = intensity_size(component)
    return {"R_number_mean_nm": None if size is None else round(size[0], 6),
            "sigma_R_number_rel": None if size is None else round(size[1], 6),
            "R_intensity_mean_nm": round(by_intensity[0], 6), "sigma_R_intensity_rel": round(by_intensity[1], 6),
            "sigma_nn_over_D": None if not component.structure else round(sigma_nn_over_d(component), 6)}


def fit_record(model: FitModel, data: FitData, curve: Optional[Curve], *, side: str, q_range, result: Optional[FitResult],
               excluded=()) -> dict:
    """What the export writes next to the table: the model, the fit and how it was made."""
    from .curve_sources import source_record
    from .fit_uncertainty import describe_links, uncertainty_record

    made_of = None if curve is None else prepare_points(curve, side, q_range, excluded)
    record = {
        "software_version": SOFTWARE,
        "model": model_to_dict(model),
        "equation": "I(q) = BG + k * [sum_i Int_i * P_i(q) * S_i(q; D_i, sigma_nn_i/D_i) + A / (1 + (|q|/w)^nu)]; "
                    "P_i(0) = 1; S_i the Hosemann paracrystal; a component with coupling 'da' has "
                    "P_i + |<F_i>|^2 (S_i - 1) in place of P_i * S_i; sizes per component ('sizes')",
        "curve": None if curve is None else {"name": curve.name, "path": curve.path, "q_unit_of_file": curve.source_unit,
                                             "q_unit_from": curve.unit_from, "columns_read": curve.columns},
        "points": {"side": side, "q_range_nm^-1": None if q_range is None else [float(v) for v in q_range],
                   "count": int(data.q.size), "weighting": data.weighting,
                   # what the points are (a curve with one half only is that half, whatever ``side`` says)
                   "halves_used": None if made_of is None else made_of.halves_used,
                   "averaged": None if made_of is None else made_of.averaged,
                   "left_out_q_nm^-1": sorted(float(value) for value in excluded)},
        "log_rmse": log_rmse(model, data),
        # spheres of a horizontal cut: q_z = 0 (effective interface) or the two-path model with the cut's angles
        "sphere_model": "effective interface (q_z = 0)" if grazing(model) is None else "two paths (direct + exit-reflected)",
        "qz_plus_R": {f"{index}": round(value, 4) for index, value in qz_plus_r(model)},
        # every particle's radius by number (as microscopy and fit1d count) and by intensity (V²), and σ_nn/D
        "derived": [derived_values(component) for component in model.components],
        "links": describe_links(model),
    }
    if curve is not None and curve.path:
        record["curve"]["source"] = source_record(curve.path)
    if result is not None:
        record["fit"] = {
            "method": result.method, "chi2_reduced": result.chi2_reduced, "converged": result.converged,
            "evaluations": result.evaluations, "seconds": round(result.seconds, 3), "message": result.message,
            "errors": {f"{owner}:{key}": value for (owner, key), value in result.errors.items()},
            "at_bounds": [f"{owner}:{key}" for owner, key in result.at_bounds],
            "correlated": [[f"{a[0]}:{a[1]}", f"{b[0]}:{b[1]}", rho] for a, b, rho in result.correlated],
            **uncertainty_record(result),  # correlation matrix and covariance (names as in links), flags
        }
    return record


__all__ = [
    "ANALYZE_SIDES", "BASE_GLOBALS", "DEFAULT_INTERFERENCE", "FAMILIES", "INTERFERENCES", "NETWORK_FAMILIES",
    "tail_for_curve", "with_interference", "GLOBALS", "GRAZING", "INFO", "RESOLUTION", "grazing", "qz_plus_r", "derived_values",
    "intensity_size", "number_size", "sigma_nn_over_d", "SIZE_LAWS", "LINEAR", "NM_PER_A", "PHASES", "PROGRESS_LINE", "SIDES",
    "BOTH_USES_MEAN", "NO_POSITIVE_POINT", "START_KEPT", "TOO_FEW_FOR_FREE", "ONE_CURVE_METHODS", "CURVE_FIT_MIN_POINTS", "SERIES_SIDES", "TOO_FEW_FOR_CURVE_FIT",
    "Component", "Curve", "FitData", "FitModel", "FitResult", "FitStopped", "MissingHalf", "Parameter", "PreparedCurve",
    "evaluate", "export_table", "fit", "fit_record", "fit_resolving", "fit_scales", "from_manual", "left_out_points", "log_rmse",
    "model_from_dict", "model_from_solution", "model_to_dict", "new_component", "parameter_text", "point_key",
    "prepare_curve", "prepare_points", "residuals", "to_manual", "explicit_limits", "solution_deviation",
]
