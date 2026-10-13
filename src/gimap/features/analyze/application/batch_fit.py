"""Fitting every frame of a batch (optional): peaks of the ring regions (GIWAXS) or a particle model
of the horizontal cut (GISAXS), with the starting values chosen by the person.

* **Peaks** — every cut region with a q window (a ring or a spot picked in Cuts) is one peak: the
  I(q) of the region's χ range is fitted with ``fit_peak`` (Gaussian, Lorentzian or pseudo-Voigt on a
  straight background) in the region's window widened by its own width on each side
  (``FIT_MARGIN``), at the detector's q step — a window of only ± FWHM leaves too little background
  to tell it from the tails of a pseudo-Voigt. The table has, per frame and peak, the centre q, d = 2π/q, FWHM, area,
  height, η, χ²ᵣ and their errors.
* **Model** — the curve that would be sent to Fitting (``fit_input_curve``: the chosen half of the
  horizontal cut) goes to the quick physical fit of Fitting (sphere / random cylinder / vertical
  cylinder with size dispersity and a spacing D; injected, since Analyze does not own it). The table
  has the best solution per frame.

Starting values (``start``):

* ``previous`` — each frame starts from the previous frame's result (a series that changes slowly:
  the fit follows the peak or the particles);
* ``first`` — every frame starts from the first frame's result (the same start for all);
* ``fresh`` — every frame is fitted on its own, the starting values found in its own data.

A frame whose fit fails is kept in the table with its reason; the next frame then starts from the
last good result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..domain import GISAXS, PeakFit, fit_peak, q_from_two_theta
from ..domain.regions import window_profile
from .frame_preprocessing import floating
from .models import FrameAnalysis
from .ports import CurveWriter
from .use_cases import SOFTWARE, fit_input_curve

FIT_NONE, FIT_PEAKS, FIT_MODEL = "none", "peaks", "model"
FIT_MARGIN = 1.0
"""The fit window is the region's q window widened by this many times its width on each side."""
START_PREVIOUS, START_FIRST, START_FRESH = "previous", "first", "fresh"
STARTS = (START_PREVIOUS, START_FIRST, START_FRESH)
MODEL_COMPONENTS = {"auto": (), "sphere": (1,), "random_cylinder": (2,), "vertical_cylinder": (3,)}
FAMILY_COMPONENTS = {"sphere": 1, "random_cylinder": 2, "vertical_cylinder": 3, "cylinder": 3}


@dataclass(frozen=True)
class PeakTarget:
    key: str
    name: str
    window: tuple[float, float]
    chi_range: tuple[float, float] = (0.0, 90.0)
    both_sides: bool = True

    @property
    def fit_window(self) -> tuple[float, float]:
        low, high = self.window
        margin = FIT_MARGIN * (high - low)
        return max(0.0, low - margin), high + margin


def peak_targets(analysis: FrameAnalysis) -> list[PeakTarget]:
    """The regions of the frame with a q window (their I(q) curves ``regionN``), in order."""
    reduction = analysis.reduction
    if reduction is None or reduction.kind == GISAXS:
        return []
    targets = []
    for curve in reduction.curves:
        region = curve.region or {}
        if curve.key.endswith("_chi") or "cut_region" not in region or curve.is_empty:
            continue
        low, high = region.get("q") or (None, None)
        if low is None or high is None:
            continue
        chi = region.get("chi_deg") or (0.0, 90.0)
        targets.append(PeakTarget(
            curve.key, str(region["cut_region"]), (float(low), float(high)), (float(chi[0]), float(chi[1])),
            bool(region.get("both_sides", True)),
        ))
    return targets


def _in_q(analysis: FrameAnalysis, curve) -> np.ndarray:
    x = np.asarray(curve.x, dtype=float)
    if curve.x_label.startswith("2θ") and analysis.geometry is not None:
        x = q_from_two_theta(x, analysis.geometry.wavelength_angstrom)
    return x


class FitStarts:
    """The starting values of the next frame, per peak (or for the model), for one batch."""

    def __init__(self, start: str = START_PREVIOUS):
        if start not in STARTS:
            raise ValueError(f"Unknown start {start!r}.")
        self.start = start
        self._first: dict[str, dict] = {}
        self._previous: dict[str, dict] = {}

    def for_key(self, key: str) -> Optional[dict]:
        if self.start == START_PREVIOUS:
            return self._previous.get(key)
        if self.start == START_FIRST:
            return self._first.get(key)
        return None

    def remember(self, key: str, values: dict) -> None:
        self._first.setdefault(key, dict(values))
        self._previous[key] = dict(values)


@dataclass(frozen=True)
class PeakInput:
    """What one peak fit needs: the target and the I(q) in its fit window (small; crosses processes)."""

    target: PeakTarget
    x: np.ndarray
    y: np.ndarray
    sigma: Optional[np.ndarray]
    window: tuple[float, float]


def peak_inputs(analysis: FrameAnalysis, *, maps=None) -> list[PeakInput]:
    """The data of every peak target of the frame, ready to fit.

    With the frame's q ``maps`` the widened window (``PeakTarget.fit_window``) straight from the pixels;
    without them, the region's own I(q) curve.
    """
    inputs = []
    counts = not floating(analysis.metadata) and not analysis.corrections.background_path
    if counts and analysis.intensity_scale is not None:
        counts = analysis.intensity_scale  # corrected counts: their variance is scaled too
    for target in peak_targets(analysis):
        if maps is not None:
            usable = np.asarray(analysis.valid, dtype=bool) & maps.above_horizon
            x, y, sigma = window_profile(maps, analysis.data, usable, target.fit_window, target.chi_range,
                                         target.both_sides, counts=counts)
            inputs.append(PeakInput(target, x, y, sigma, target.fit_window))
        else:
            curve = analysis.reduction.curve(target.key)
            inputs.append(PeakInput(target, _in_q(analysis, curve), np.asarray(curve.intensity, dtype=float),
                                    None if curve.sigma is None else np.asarray(curve.sigma, dtype=float), target.window))
    return inputs


def fit_peak_inputs(
    inputs: Sequence[PeakInput], profile: str, starts: Optional[FitStarts] = None,
) -> list[tuple[PeakTarget, PeakFit]]:
    """Fit every input in turn (``starts`` supplies and records the starting values)."""
    results = []
    for item in inputs:
        start = starts.for_key(item.target.key) if starts is not None else None
        fit = fit_peak(item.x, item.y, item.sigma, item.window, profile=profile, start=start)
        if fit.ok and starts is not None:
            starts.remember(item.target.key, fit.as_start())
        results.append((item.target, fit))
    return results


def fit_peaks(
    analysis: FrameAnalysis, profile: str, starts: Optional[FitStarts] = None, *, maps=None,
) -> list[tuple[PeakTarget, PeakFit]]:
    """Every peak target of the frame fitted (``peak_inputs`` then ``fit_peak_inputs``)."""
    return fit_peak_inputs(peak_inputs(analysis, maps=maps), profile, starts)


def peak_row(label: str, fits: Sequence[tuple[PeakTarget, PeakFit]]) -> dict[str, Any]:
    """One row of the peak table: per peak ``<name> q``, ``… q err``, d, FWHM, area, height, η, χ²ᵣ, note."""
    row: dict[str, Any] = {"frame": label}
    for target, fit in fits:
        name = target.name
        d = 2.0 * math.pi / fit.center if fit.ok and fit.center > 0 else math.nan
        values = (
            ("q (1/A)", fit.center), ("q err", fit.center_err), ("d (A)", d), ("FWHM (1/A)", fit.fwhm),
            ("FWHM err", fit.fwhm_err), ("area", fit.area), ("area err", fit.area_err), ("height", fit.height),
            ("eta", fit.eta), ("chi2 red", fit.chi2_red),
        )
        for column, value in values:
            row[f"{name} {column}"] = float(value) if fit.ok else math.nan
        row[f"{name} note"] = fit.message
    return row


def model_row(label: str, rows: Sequence[dict], *, error: str = "") -> dict[str, Any]:
    """One row of the model table: the best solution (model, χ², log RMSE, parameters in nm)."""
    row: dict[str, Any] = {"frame": label}
    if not rows:
        row["note"] = error or "no solution"
        return row
    best = rows[0]
    row["model"] = str(best.get("combination") or "")
    row["chi2"] = float(best.get("best_chi2_weighted", math.nan))
    row["log rmse"] = float(best.get("best_log_rmse", math.nan))
    for index, component in enumerate(best.get("components") or (), start=1):
        prefix = f"c{index} {component.get('type', '')}"
        row[f"{prefix} weight"] = float(component.get("weight", math.nan))
        for key, value in (component.get("params") or {}).items():
            try:
                row[f"{prefix} {key}"] = float(value)
            except (TypeError, ValueError):
                continue
    row["note"] = "; ".join(str(item) for item in best.get("warnings") or ())
    return row


def _family(row: dict) -> Optional[int]:
    components = row.get("components") or ()
    if not components:
        return None
    return FAMILY_COMPONENTS.get(str(components[0].get("type", "")).lower())


def _distance(row: dict) -> Optional[float]:
    for component in row.get("components") or ():
        value = (component.get("params") or {}).get("D")
        if value is not None and np.isfinite(float(value)):
            return float(value)
    return None


def fit_model(
    analysis: FrameAnalysis, fitter: Callable, model: str = "auto", starts: Optional[FitStarts] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> list[dict]:
    """The solutions of the quick physical fit for the frame's fit curve, best first."""
    return fit_model_curve(fit_input_curve(analysis), fitter, model, starts, cancelled)


def fit_model_curve(
    curve, fitter: Callable, model: str = "auto", starts: Optional[FitStarts] = None,
    cancelled: Optional[Callable[[], bool]] = None,
) -> list[dict]:
    """The solutions of the quick physical fit for one curve (the frame's ``fit_input_curve``), best first."""
    if curve is None or curve.is_empty:
        raise ValueError("This frame has no curve to fit.")
    components = MODEL_COMPONENTS.get(model, ())
    start = starts.for_key("model") if starts is not None else None
    distance = None
    if start is not None:
        distance = start.get("D")
        if not components and start.get("family"):
            components = (int(start["family"]),)
    rows = fitter(curve.x, curve.intensity, curve.sigma, components=tuple(components), distance_nm=distance,
                  cancelled=cancelled)
    rows = list(rows or ())
    if rows and starts is not None:
        starts.remember("model", {"family": _family(rows[0]), "D": _distance(rows[0])})
    return rows


class FitTable:
    """Rows of a fit table (one per frame), written at the end of the batch."""

    def __init__(self, kind: str):
        self.kind = kind
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def columns(self) -> list[str]:
        columns: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        return columns

    def write(self, writer: CurveWriter, path: Path, comments: Sequence[str] = ()) -> Path:
        columns = self.columns()
        title = "peaks of the ring regions fitted in every frame" if self.kind == FIT_PEAKS else (
            "a particle model fitted to the horizontal cut of every frame")
        lines = [f"GIMaP Analyze batch fit ({SOFTWARE}): {title}", *comments]
        body = [[row.get(column, "") for column in columns] for row in self.rows]
        return writer.write_table(Path(path), lines, columns, body)

    def trend(self, column_suffix: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """``(name, frame number, value)`` of every column ending in ``column_suffix`` (for a trend plot)."""
        curves = []
        for column in self.columns():
            if column.endswith(column_suffix):
                values = np.array([float(row.get(column, math.nan)) if row.get(column, "") != "" else math.nan
                                   for row in self.rows], dtype=float)
                curves.append((column[: -len(column_suffix)].strip() or column, np.arange(1, len(values) + 1, dtype=float), values))
        return curves


__all__ = [
    "FAMILY_COMPONENTS", "FIT_MARGIN", "FIT_MODEL", "FIT_NONE", "FIT_PEAKS", "FitStarts", "FitTable", "MODEL_COMPONENTS",
    "PeakInput", "PeakTarget", "START_FIRST", "START_FRESH", "START_PREVIOUS", "STARTS", "fit_model",
    "fit_model_curve", "fit_peak_inputs", "fit_peaks", "model_row", "peak_inputs", "peak_row", "peak_targets",
]
