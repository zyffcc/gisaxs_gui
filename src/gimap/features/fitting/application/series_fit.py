"""Fitting a series of curves (in-situ): every frame with the model of Single analysis, each frame
starting from the previous frame's result (or every frame from the same start), and the table of
the results — every parameter with its error and the quality of each frame's fit.

A frame is fitted on one curve: the mean of the halves of its cut, or one half (``SeriesSettings.side``,
never both halves together). A frame without the chosen half fails with the sentence of ``MissingHalf``;
the other half does not stand in for it. Every frame says how many points it was fitted on and what
they are (``FrameFit.points``, ``halves_used``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from src.gimap.shared.series_stages import SeriesStages, find_stages
from src.gimap.shared.signed_cut import FIT_HALVES, fit_halves
from src.gimap.shared.version import SOFTWARE

from ..domain.fit_engine import FitResult
from ..domain.fit_model import FAMILIES, INFO, FitModel, model_to_dict
from .single_fit import Curve, PreparedCurve, fit_resolving, prepare_points
from .models import InSituSourceFrame
from .single_fit1d import fit1d_result

STARTS = (("previous", "Each frame starts from the previous frame's result"),
          ("same", "Every frame starts from the model in Single analysis"),
          ("stages", "From the previous frame's result, and from the model in Single analysis at every new stage"))
TOO_FEW_POINTS = "too few points in the fitting range"
UNREADABLE = "the curve could not be read"
STATES = ("ok", "not converged", "stopped", "failed")
"""What became of a frame (``FrameFit.state``): fitted and converged; fitted, but the fit did not converge;
stopped by the person in the middle of its fit (its values are where the fit was, not a result); no fit."""
POOR_CHI2 = 10.0
"""A converged fit weighted by σ whose χ²ᵣ is above this is a poor fit (``poor_fit``): the model misses the
curve by far more than its noise — a model that was never set up, another stage of the sample."""


@dataclass(frozen=True)
class SeriesSettings:
    side: str = "mean"
    """``mean``, ``positive`` or ``negative``; anything else (``both`` of Single analysis) becomes the mean."""
    q_range: Optional[tuple] = None
    excluded: frozenset = frozenset()
    method: str = "local"
    start: str = "previous"
    resolve_edges: bool = True
    """Answer the parameters that stop at an edge of their range (``single_fit.fit_resolving``)."""
    follow: bool = True
    """``fit1d``: keep the solution that continues the previous frame when it fits as well (``domain.fit1d_follow``)."""
    fit1d: Optional[Mapping] = None
    """``fit1d``: its settings (``fit1d_*`` of Single analysis' Fit step; the particles are those of the start model)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", fit_halves(self.side))


def series_side(chosen=None, observed=None, single=None, *, signed: bool = True) -> str:
    """The halves a series is fitted on (``SeriesSettings.side``), from what there is to go by, in this
    order: the person's own choice for the series, the ``halves`` the first listed fit input was sent with,
    the halves of Single analysis — each only when it is one a fit can take (not ``both``) — else the mean.

    ``signed``: whether a listed curve is a cut through the beam (points on both sides of q = 0). A series
    of curves with one side has no halves to choose from: it is fitted on the mean, which for such a curve is
    the curve itself — whatever is left of a choice made for other curves (a half would fail every frame)."""
    if not signed:
        return "mean"
    return next((choice for choice in (chosen, observed, single) if choice in FIT_HALVES), "mean")


@dataclass(frozen=True)
class FrameFit:
    """One frame of the series: its file and the fit, or why there is none."""

    index: int
    path: str
    result: Optional[FitResult] = None
    error: str = ""
    points: int = 0
    """The points the frame was fitted on (in the fitting range, without the left-out ones)."""
    halves_used: str = ""
    """What those points are: ``mean``, ``positive`` or ``negative`` (a frame with one half only, fitted
    under ``mean``, is that half); empty when the curve could not be prepared."""
    averaged: int = 0
    """How many of the points are the mean of two measurements (the others come from one half alone)."""
    left_out: int = 0
    """How many points the left-out names of the series took away in this frame."""
    resolved: tuple = ()
    """What was done to parameters that stopped at an edge (English sentences, ``domain.fit_edges``)."""
    native: Optional[dict] = None
    """``fit1d``: the kit's own result of the frame (its v2 curve, errors, bounds, whether it followed the previous)."""

    @property
    def ok(self) -> bool:
        return self.result is not None and not self.error

    @property
    def fitted(self) -> bool:
        """Fitted to the end — converged or not: a result to keep. Where a stopped fit was is none."""
        return self.ok and not self.result.stopped

    @property
    def state(self) -> str:
        """One of ``STATES``."""
        if not self.ok:
            return "failed"
        if not self.fitted:
            return "stopped"
        return "ok" if self.result.converged else "not converged"

    @property
    def reason(self) -> str:
        """Why the frame is not ``ok``: the error, or what the fit said when it ended."""
        return self.error if not self.ok else ("" if self.state == "ok" else str(self.result.message or ""))


def poor_fit(frame) -> bool:
    """Whether a frame's fit ended as a result (``ok``) that does not describe its curve: weighted by σ, with
    χ²ᵣ above ``POOR_CHI2``. Without σ in the file χ²ᵣ is the scatter of ln I and says nothing of this."""
    result = frame.result if frame.ok else None
    if result is None or not result.converged or getattr(result, "weighting", "") != "sigma":
        return False
    return bool(math.isfinite(result.chi2_reduced) and result.chi2_reduced > POOR_CHI2)


def prepare_frame(curve: Curve, settings: SeriesSettings) -> PreparedCurve:
    """The points a frame is fitted on (the halves, range and left-out points of the series) and what they
    are. ``MissingHalf`` when the frame has no points on the chosen half."""
    return prepare_points(curve, settings.side, settings.q_range, settings.excluded, strict=True)


def fit_frame(index: int, path: str, curve: Curve, start: FitModel, settings: SeriesSettings,
              stop: Optional[Callable[[], bool]] = None, predict: Optional[Callable] = None) -> FrameFit:
    """The fit of one frame (``prepare_frame``). ``fit1d``: ``predict(data, start) -> value`` runs fit1d's job on
    the frame's points (the page knows the job runner and the previous frame)."""
    try:
        prepared = prepare_frame(curve, settings)
    except ValueError as exc:  # no points on the chosen half
        return FrameFit(index, path, error=str(exc) or type(exc).__name__)
    made_of = dict(points=prepared.points, halves_used=prepared.halves_used, averaged=prepared.averaged,
                   left_out=prepared.left_out)
    if prepared.points < 3:
        return FrameFit(index, path, error=TOO_FEW_POINTS, **made_of)
    try:
        if settings.method == "fit1d":
            value = predict(prepared.data, start)
            result = fit1d_result(value, start, prepared.data)
            actions = (value["native"].get("fit1d_bounds") or {}).get("actions") or ()
            return FrameFit(index, path, result, resolved=tuple(note["text"] for note in actions),
                            native=value["native"], **made_of)
        result, actions, _remaining = fit_resolving(start, prepared.data, method=settings.method, stop=stop,
                                                    resolve=settings.resolve_edges)
        return FrameFit(index, path, result, resolved=tuple(note.text for note in actions), **made_of)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, KeyError, RuntimeError) as exc:
        return FrameFit(index, path, error=str(exc) or type(exc).__name__, **made_of)


def series_counts(frames: Sequence[FrameFit]) -> dict:
    """How many frames are in each of ``STATES`` (``not_converged`` for the second), and ``frames`` in all.
    Only the ``ok`` and ``not_converged`` ones were fitted to the end."""
    states = [frame.state for frame in frames]
    return {"frames": len(states), **{state.replace(" ", "_"): states.count(state) for state in STATES}}


def next_start(previous: Optional[FrameFit], single: FitModel, settings: SeriesSettings, *,
               new_stage: bool = False) -> FitModel:
    """Where the next frame starts: the previous frame's result (when it converged) or the Single model —
    also at the first frame of a new stage when ``start`` is ``"stages"``."""
    if settings.method == "fit1d":  # its particles and constraints are the model's; it follows by itself
        return single
    follows = settings.start == "previous" or (settings.start == "stages" and not new_stage)
    if follows and previous is not None and previous.ok and previous.result.converged:
        return previous.result.model
    return single


@dataclass(frozen=True)
class FrameStages(SeriesStages):
    """The stages of a series whose rows are the curves it was asked for, in their order — also when some
    of them could not be used."""

    skipped: tuple = ()
    """``(row, why)`` of the curves left out of the search: they could not be read or prepared (no points
    on the chosen half, too few points in the fitting range). They lie in a stage by their place in the
    series; their ``scores`` and ``level`` are NaN, and they are never odd frames."""


def _on_every_row(found: SeriesStages, rows: Sequence[int], total: int, skipped) -> FrameStages:
    """``found`` (its rows: the curves at ``rows`` of ``total``) with every row counted among all curves."""
    place = np.asarray(rows, dtype=int)
    scores = np.full((total, *found.scores.shape[1:]), np.nan)
    level = np.full(total, np.nan)
    scores[place], level[place] = found.scores, found.level
    at = lambda row: None if row is None else int(place[row])  # noqa: E731
    values = {item.name: getattr(found, item.name) for item in fields(found)}
    values.update(
        rows=total, scores=scores, level=level, odd=tuple(replace(frame, row=at(frame.row)) for frame in found.odd),
        boundaries={count: tuple(at(row) for row in starts) for count, starts in found.boundaries.items()},
        representatives={count: tuple(at(row) for row in chosen) for count, chosen in found.representatives.items()},
        half_row=at(found.half_row), ninety_row=at(found.ninety_row),
    )
    return FrameStages(**values, skipped=tuple(skipped))


def stages_of_curves(curves: Sequence[Optional[Curve]], settings: SeriesSettings,
                     unreadable: Optional[Mapping[int, str]] = None) -> FrameStages:
    """Stages and odd frames of the curves as they are fitted (``prepare_frame``), on the q (nm⁻¹) of the
    first curve that can be used.

    A curve that cannot be used does not end the search: ``None`` (it could not be read; ``unreadable``
    may say why, by its place), no points on the chosen half, fewer than two points in the range. The
    stages are found on the others and ``skipped`` names the ones left out, with the reason; the rows of
    the result are still the places in ``curves``."""
    prepared, skipped = _usable_curves(curves, settings, unreadable)
    if not prepared:
        raise ValueError(skipped[0][1] if skipped else "No curves to find stages in.")
    grid = prepared[0][1]
    image = np.full((len(prepared), grid.size), np.nan)
    for line, (_row, q, intensity) in enumerate(prepared):
        inside = (grid >= q[0]) & (grid <= q[-1])
        image[line, inside] = np.interp(grid[inside], q, intensity)
    return _on_every_row(find_stages(grid, image), [row for row, _q, _intensity in prepared], len(curves), skipped)


def unusable_curves(curves: Sequence[Optional[Curve]], settings: SeriesSettings,
                    unreadable: Optional[Mapping[int, str]] = None) -> tuple:
    """``(row, why)`` of the curves that cannot be used as the series stands — as ``FrameStages.skipped``
    names them — also when there are too few curves to find stages in."""
    return tuple(_usable_curves(curves, settings, unreadable)[1])


def _usable_curves(curves, settings: SeriesSettings, unreadable) -> tuple[list, list]:
    """``(row, q, I)`` of every curve as it is fitted (ascending q), and ``(row, why)`` of the others."""
    prepared, skipped = [], []
    for row, curve in enumerate(curves):
        if curve is None:
            skipped.append((row, str((unreadable or {}).get(row) or UNREADABLE)))
            continue
        try:
            data = prepare_frame(curve, settings).data
        except ValueError as exc:
            skipped.append((row, str(exc)))
            continue
        if data.q.size < 2:
            skipped.append((row, TOO_FEW_POINTS))
            continue
        order = np.argsort(data.q)
        prepared.append((row, np.asarray(data.q, dtype=float)[order], np.asarray(data.intensity, dtype=float)[order]))
    return prepared, skipped


def parameter_columns(model: FitModel) -> list[tuple[tuple, str]]:
    """The free parameters of the model, as (path, column name) — the columns of the table. The names are
    plain text for any program that reads the table (``res_width_nm^-1``, as the record writes its units)."""
    columns = []
    for path, parameter in model.parameters():
        if not parameter.free:
            continue
        owner, key = path
        prefix = "" if owner == "globals" else f"{owner + 1}_{FAMILIES[model.components[owner].family][0].replace(' ', '_')}_"
        unit = INFO[key].unit.replace("⁻¹", "^-1")
        columns.append((path, f"{prefix}{key}" + (f"_{unit}" if unit else "")))
    return columns


MADE_OF_COLUMNS = ("points", "halves_used")


def file_name(path, folder=None) -> str:
    """How a table and a record name a frame's file: its path below ``folder`` (with ``/``) when it lies
    there — frames of several subfolders with equal names stay apart — else its name.

    The listing follows links, so a frame may lie below the place ``folder`` leads to and not below
    ``folder`` as it was chosen (a linked folder, a linked mount point, a mapped drive): both are tried."""
    frame = InSituSourceFrame.from_token(str(path))  # a frame past the first of a detector source: “· frame N”
    suffix = f" · frame {frame.frame_index + 1}" if frame.frame_index else ""
    if folder:
        for root in (Path(folder), _resolved(str(folder))):
            try:
                return frame.path.relative_to(root).as_posix() + suffix
            except ValueError:
                pass
    return frame.path.name + suffix


@lru_cache(maxsize=16)
def _resolved(folder: str) -> Path:
    """Where ``folder`` leads to (asked of the file system once per folder: every frame of a series on a
    mapped drive comes here)."""
    try:
        return Path(folder).expanduser().resolve()
    except (OSError, RuntimeError, ValueError):  # a folder that cannot be followed: as it is
        return Path(folder)


def series_table(frames: list[FrameFit], model: FitModel, *, details: bool = False,
                 folder=None) -> tuple[list[str], list[list]]:
    """Header and rows: frame, file, χ²ᵣ, log RMSE, converged, then value and error of every free parameter.

    ``details``: with what each frame was fitted on after ``converged`` — ``points`` and ``halves_used``
    (``MADE_OF_COLUMNS``) — for the table that is saved. A frame without a fit has NaN values.
    ``folder``: the folder of the series, for the ``file`` column (``file_name``)."""
    columns = parameter_columns(model)
    header = ["frame", "file", "chi2_reduced", "log_rmse", "converged"] + (list(MADE_OF_COLUMNS) if details else [])
    for _path, name in columns:
        header += [name, f"{name}_error"]
    rows = []
    for frame in sorted(frames, key=lambda item: item.index):
        row = [frame.index + 1, file_name(frame.path, folder)]
        made_of = [frame.points, frame.halves_used] if details else []
        if not frame.ok:
            rows.append(row + [math.nan, math.nan, False] + made_of + [math.nan] * (2 * len(columns)))
            continue
        result = frame.result
        row += [result.chi2_reduced, result.log_rmse, bool(result.converged)] + made_of
        for path, _name in columns:
            row += [result.model.get(path).value, result.errors.get(path, math.nan)]
        rows.append(row)
    return header, rows


def series_record(model: FitModel, settings: SeriesSettings, folder: str, pattern: str, frames: list[FrameFit], *,
                  q_unit: Optional[str] = None, saved: Optional[str] = None, side_from: str = "",
                  listing: Optional[Mapping] = None) -> dict:
    """What is written next to the table: the start model, how the series was fitted and, for every frame,
    what became of it and on which points.

    ``q_unit``: the unit the q of the files was read in (``angstrom`` / ``nm``; the values of the record
    are in nm⁻¹); ``saved``: the time written into the record (now, when not given); ``side_from``: where
    the halves of the series came from (the person's own choice, the fit inputs, Single analysis …);
    ``listing``: which curves of the folder were fitted (first, last, every n-th, the frames left out …),
    written as it is given. Files are named as in the table (``file_name``)."""
    ordered = sorted(frames, key=lambda item: item.index)
    failed = [{"frame": frame.index + 1, "file": file_name(frame.path, folder), "why": frame.error}
              for frame in ordered if not frame.ok]
    record = {
        "schema": "gimap_series_fit_v2", "software": SOFTWARE,
        "saved": saved or datetime.now().astimezone().isoformat(timespec="seconds"),
        "folder": folder, "pattern": pattern, "q_unit_of_files": q_unit,
        "start_model": model_to_dict(model),
        "settings": {"side": settings.side, "q_range_nm^-1": None if settings.q_range is None else list(settings.q_range),
                     "left_out_q_nm^-1": sorted(settings.excluded), "method": settings.method, "start": settings.start,
                     "resolve_edges": settings.resolve_edges,
                     **({"follow": settings.follow, "fit1d": dict(settings.fit1d or {})} if settings.method == "fit1d" else {})},
        "frames": len(frames), "counts": series_counts(frames), "failed": failed,
        "not_converged": [frame.index + 1 for frame in ordered if frame.state == "not converged"],
        "stopped": [frame.index + 1 for frame in ordered if frame.state == "stopped"],
        "poor_fit": {"chi2_reduced_above": POOR_CHI2, "frames": [frame.index + 1 for frame in ordered if poor_fit(frame)]},
        "per_frame": [{"frame": frame.index + 1, "file": file_name(frame.path, folder), "state": frame.state,
                       "reason": frame.reason, "points": frame.points, "halves_used": frame.halves_used,
                       "averaged": frame.averaged, "left_out": frame.left_out,
                       "weighting": frame.result.weighting if frame.result is not None else "",
                       **({"resolved": list(frame.resolved)} if frame.resolved else {}),
                       **({"followed_previous": bool((frame.native.get("fit1d_follow") or {}).get("followed"))}
                          if frame.native and frame.native.get("fit1d_follow") else {})} for frame in ordered],
    }
    if side_from:
        record["settings"]["side_from"] = side_from
    if listing is not None:
        record["listing"] = dict(listing)
    return record


__all__ = ["FrameFit", "FrameStages", "MADE_OF_COLUMNS", "POOR_CHI2", "STARTS", "STATES", "SeriesSettings", "TOO_FEW_POINTS",
           "UNREADABLE", "file_name", "fit_frame", "next_start", "parameter_columns", "poor_fit", "prepare_frame",
           "series_counts", "series_record", "series_side", "series_table", "stages_of_curves", "unusable_curves"]
