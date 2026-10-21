"""Fitting every frame of a batch (optional): peaks of the ring regions (GIWAXS) or a particle model
of the horizontal cut (GISAXS), with the starting values chosen by the person.

* **Peaks** — every cut region with a q window (a ring or a spot picked in Cuts) is one peak: the
  I(q) of the region's χ range is fitted with ``fit_peak`` (Gaussian, Lorentzian or pseudo-Voigt on a
  straight background) in the region's window widened by its own width on each side
  (``FIT_MARGIN``), at the detector's q step — a window of only ± FWHM leaves too little background
  to tell it from the tails of a pseudo-Voigt. The table has, per frame and peak, the centre q, d = 2π/q, FWHM, area,
  height, η, χ²ᵣ and their errors.
* **Model** — the curve that would be sent to Fitting (``fit_input_curve``: the whole horizontal cut,
  both halves with signed qy) goes to the quick physical fit of Fitting (sphere / random cylinder /
  vertical cylinder with size dispersity and a spacing D; injected, since Analyze does not own it)
  together with the halves to fit (``halves``: ``mean`` / ``positive`` / ``negative`` — a model is a
  function of |q| and a fit takes one curve, never both halves together). The fit prepares the curve
  itself (``shared/signed_cut``: the mean without drawing across holes, neighbouring points merged
  above 1000) and says what it fitted. The table has the best solution per frame, with the halves and
  the number of points it was fitted on; a frame without the chosen half is a failed fit with that
  reason, never the other half in its place. A cut that holds no intensity above zero on what was to be
  fitted (a closed shutter) is not given to the fit: its row says so (``NO_INTENSITY``).

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

from src.gimap.shared.signed_cut import fit_halves

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
HALVES_WORDS = {"mean": "mean of both halves", "positive": "q > 0 half", "negative": "q < 0 half"}
"""The halves a fit takes of the cut, in the words of the tables and the README (plain ASCII)."""
NO_CURVE = "This frame has no curve to fit."
NO_FITTER = "no particle-model fit in this session"
NO_SOLUTION = "no solution"
"""Why a frame has no model fit, besides the reasons the fit itself gives (a ``ValueError``'s text)."""
NO_INTENSITY = {
    "mean": "The horizontal cut has no intensity above zero",
    "positive": "No intensity above zero on the q > 0 half",
    "negative": "No intensity above zero on the q < 0 half",
}
"""A frame without counts on what was to be fitted: the whole cut (``mean``: a closed shutter; said for any
choice of halves) or the chosen half alone (behind an absorber). Said by the batch before the fit is asked —
the fit's own words for it name a requirement of its algorithm."""


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


def _made_of(rows: Sequence[dict], halves: str) -> tuple[str, Any, dict]:
    """``(halves, points, preparation)``: what the best solution was fitted on, as the fit says it
    (``preparation`` of its rows) — under ``mean`` the one half of a cut that has only one. Without a
    solution: the halves asked for and no point; a fit that does not say: the halves asked for, points blank."""
    made_of = dict(rows[0].get("preparation") or {}) if rows else {}
    points = made_of.get("points")
    return (str(made_of.get("halves") or fit_halves(halves)),
            int(points) if points is not None else ("" if rows else 0), made_of)


def model_row(label: str, rows: Sequence[dict], *, error: str = "", halves: str = "mean") -> dict[str, Any]:
    """One row of the model table: the halves and the number of points the frame was fitted on, then the best
    solution (model, χ², log RMSE, parameters in nm); without a solution the reason as the note."""
    used, points, _ = _made_of(rows, halves)
    row: dict[str, Any] = {"frame": label, "halves": used, "points": points}
    if not rows:
        row["note"] = error or NO_SOLUTION
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


def model_fit_frame(label: str, rows: Sequence[dict], *, error: str = "", halves: str = "mean") -> dict[str, Any]:
    """What the batch record keeps of one frame's model fit: ``state`` (``ok`` / ``failed``), the ``halves``
    and ``points`` fitted, the note of the preparation as the fit gave it (``prepared``) with its numbers
    (``averaged``: points with two measurements, ``one_half``: with one, ``merged``: neighbouring points per
    point of the fit) — or the ``reason`` there is no solution."""
    used, points, made_of = _made_of(rows, halves)
    if not rows:
        return {"frame": label, "state": "failed", "reason": error or NO_SOLUTION, "halves": used, "points": 0,
                "prepared": ""}
    return {
        "frame": label, "state": "ok", "halves": used, "points": points, "prepared": str(rows[0].get("prepared") or ""),
        "averaged": int(made_of.get("averaged") or 0), "one_half": int(made_of.get("one_half") or 0),
        "merged": int(made_of.get("merged") or 1),
    }


def model_fit_curve(label: str, rows: Sequence[dict], *, halves: str = "mean") -> tuple[list[str], list[str], list[list[float]]]:
    """``(comments, header, rows)`` of one frame's fitted curve: the points the best solution was fitted on —
    |q| in nm⁻¹, I and σ of the mean of the halves or of the one half, after merging — and the model at them,
    so the file shows what was compared with what. A fit that does not return its points: the model alone,
    on its own q grid."""
    best = rows[0]
    name = str(best.get("combination") or "")
    q, data, fit = best.get("native_q") or (), best.get("observed") or (), best.get("native_fit") or ()
    if len(q) and len(q) == len(data) == len(fit):
        used = _made_of(rows, halves)[0]
        sigma = best.get("sigma") or ()
        sigma = sigma if len(sigma) == len(q) else [math.nan] * len(q)
        comments = [f"{label}: the points fitted ({HALVES_WORDS.get(used, used)}, {len(q)} points) and the best model "
                    f"({name}) at them; q in 1/nm"]
        if best.get("prepared"):
            comments.append(f"points: {best['prepared']}")
        return comments, ["q (1/nm)", "I", "sigma", "fit"], [
            [abs(float(a)), float(b), float(c), float(d)] for a, b, c, d in zip(q, data, sigma, fit)]
    grid, curve = best.get("display_q") or (), best.get("display_fit") or ()
    return ([f"{label}: the best model ({name}) on its own q grid"], ["q (1/nm)", "fit"],
            [[abs(float(a)), float(b)] for a, b in zip(grid, curve)])


def model_fit_record(halves: str, frames: Sequence[dict]) -> dict[str, Any]:
    """The model fits of a batch for its record: the halves chosen, how many frames were fitted, every frame."""
    return {
        "halves": fit_halves(halves), "frames": len(frames),
        "fitted": sum(1 for frame in frames if frame.get("state") == "ok"), "per_frame": [dict(frame) for frame in frames],
    }


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


def _no_intensity(curve, halves: str) -> str:
    """Why the cut cannot be fitted for want of counts (``NO_INTENSITY``), or ``""``: its points (finite, off the
    beam) are there and none of them is above zero — the whole cut, said whichever halves were chosen, since
    the other half would not help; else the chosen half. A half without any point is the fit's to say."""
    q, intensity = np.asarray(curve.x, dtype=float), np.asarray(curve.intensity, dtype=float)
    measured = np.isfinite(q) & np.isfinite(intensity) & (q != 0)
    lit = measured & (intensity > 0)
    if measured.any() and not lit.any():
        return NO_INTENSITY["mean"]
    half = {"positive": q > 0, "negative": q < 0}.get(halves)
    if half is not None and (measured & half).any() and not (lit & half).any():
        return NO_INTENSITY[halves]
    return ""


def fit_model(
    analysis: FrameAnalysis, fitter: Callable, model: str = "auto", starts: Optional[FitStarts] = None,
    cancelled: Optional[Callable[[], bool]] = None, *, halves: str = "mean",
) -> list[dict]:
    """The solutions of the quick physical fit for the frame's fit curve, best first."""
    return fit_model_curve(fit_input_curve(analysis), fitter, model, starts, cancelled, halves=halves)


def fit_model_curve(
    curve, fitter: Callable, model: str = "auto", starts: Optional[FitStarts] = None,
    cancelled: Optional[Callable[[], bool]] = None, *, halves: str = "mean",
) -> list[dict]:
    """The solutions of the quick physical fit for one curve (the frame's ``fit_input_curve``), best first.

    ``halves``: what the fit takes of the signed cut — ``mean``, ``positive`` or ``negative`` (anything else,
    like the display choice ``both_abs``, is the mean). The fitter gets the whole cut and this choice, and
    raises ``ValueError`` with the reason when the chosen half has no points or too few. A cut whose points
    to fit hold no intensity above zero is not given to the fitter: ``ValueError`` with one of ``NO_INTENSITY``."""
    if curve is None or curve.is_empty:
        raise ValueError(NO_CURVE)
    halves = fit_halves(halves)
    blank = _no_intensity(curve, halves)
    if blank:
        raise ValueError(blank)
    components = MODEL_COMPONENTS.get(model, ())
    start = starts.for_key("model") if starts is not None else None
    distance = None
    if start is not None:
        distance = start.get("D")
        if not components and start.get("family"):
            components = (int(start["family"]),)
    rows = fitter(curve.x, curve.intensity, curve.sigma, components=tuple(components), distance_nm=distance,
                  cancelled=cancelled, halves=halves)
    rows = list(rows or ())
    if rows and starts is not None:
        starts.remember("model", {"family": _family(rows[0]), "D": _distance(rows[0])})
    return rows


NUMBER_COLUMN = "frame number"
"""The column of a fit table with the place of each frame among the listed frames, counted from 1: the x of
the trend plot (``frame`` is the frame's name)."""


class FitTable:
    """Rows of a fit table (one per frame), written at the end of the batch."""

    def __init__(self, kind: str):
        self.kind = kind
        self.rows: list[dict[str, Any]] = []
        self.numbers: list[Optional[int]] = []
        """The place of each row's frame among the listed frames, from 1 (``None``: not known)."""

    def add(self, row: dict[str, Any], number: Optional[int] = None) -> None:
        self.rows.append(row)
        self.numbers.append(None if number is None else int(number))

    def columns(self) -> list[str]:
        """Every column of the rows, in the order they first appear; the model's ``note`` last — also when the
        first frame has no fit and its row holds nothing but the note."""
        columns: list[str] = []
        for row in self.rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        if "note" in columns:
            columns.append(columns.pop(columns.index("note")))
        return columns

    def frame_numbers(self) -> np.ndarray:
        """The x of a trend: the frames' places in the list (every n-th frame: 1, n + 1, …; a frame that failed
        to be read leaves its number out), or 1, 2, … when a row came without its number."""
        if self.numbers and all(number is not None for number in self.numbers):
            return np.asarray(self.numbers, dtype=float)
        return np.arange(1, len(self.rows) + 1, dtype=float)

    def write(self, writer: CurveWriter, path: Path, comments: Sequence[str] = ()) -> Path:
        columns = self.columns()
        title = "peaks of the ring regions fitted in every frame" if self.kind == FIT_PEAKS else (
            "a particle model fitted to the horizontal cut of every frame")
        lines = [f"GIMaP Analyze batch fit ({SOFTWARE}): {title}", *comments]
        body = [[row.get(column, "") for column in columns] for row in self.rows]
        if self.numbers and all(number is not None for number in self.numbers) and columns[:1] == ["frame"]:
            columns = ["frame", NUMBER_COLUMN, *columns[1:]]
            body = [[cells[0], number, *cells[1:]] for cells, number in zip(body, self.numbers)]
            lines.append(f"{NUMBER_COLUMN}: the place of the frame among the listed frames, counted from 1")
        return writer.write_table(Path(path), lines, columns, body)

    def _values(self, column: str) -> np.ndarray:
        return np.array([float(row.get(column, math.nan)) if row.get(column, "") != "" else math.nan
                         for row in self.rows], dtype=float)

    def trend(self, column_suffix: str, error_suffix: str = "") -> list[tuple]:
        """``(name, frame number, value, error or None)`` of every column ending in ``column_suffix`` that holds
        a value (for a trend plot); the error is the column ``<name><error_suffix>`` when there is one."""
        curves = []
        columns = self.columns()
        for column in columns:
            if not column.endswith(column_suffix):
                continue
            values = self._values(column)
            if not np.isfinite(values).any():
                continue  # a peak that was fitted in no frame: no empty entry in the legend
            name = column[: -len(column_suffix)].strip()
            error = f"{name}{error_suffix}" if error_suffix else ""
            curves.append((name or column, self.frame_numbers(), values, self._values(error) if error in columns else None))
        return curves


__all__ = [
    "FAMILY_COMPONENTS", "FIT_MARGIN", "FIT_MODEL", "FIT_NONE", "FIT_PEAKS", "FitStarts", "FitTable", "HALVES_WORDS",
    "MODEL_COMPONENTS", "NO_CURVE", "NO_FITTER", "NO_INTENSITY", "NO_SOLUTION", "NUMBER_COLUMN", "PeakInput", "PeakTarget",
    "START_FIRST",
    "START_FRESH",
    "START_PREVIOUS", "STARTS", "fit_model", "fit_model_curve", "fit_peak_inputs", "fit_peaks", "model_fit_curve",
    "model_fit_frame", "model_fit_record", "model_row", "peak_inputs", "peak_row", "peak_targets",
]
