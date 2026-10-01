"""Fitting a series of curves (in-situ): every frame with the model of Single analysis, each frame
starting from the previous frame's result (or every frame from the same start), and the table of
the results — every parameter with its error and the quality of each frame's fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from src.gimap.shared.series_stages import SeriesStages, find_stages

from ..domain.fit_engine import FitResult, fit
from ..domain.fit_model import FAMILIES, INFO, FitModel, model_to_dict
from .single_fit import Curve, prepare_curve

STARTS = (("previous", "Each frame starts from the previous frame's result"),
          ("same", "Every frame starts from the model in Single analysis"),
          ("stages", "From the previous frame's result, and from the model in Single analysis at every new stage"))


@dataclass(frozen=True)
class SeriesSettings:
    side: str = "mean"
    q_range: Optional[tuple] = None
    excluded: frozenset = frozenset()
    method: str = "local"
    start: str = "previous"


@dataclass(frozen=True)
class FrameFit:
    """One frame of the series: its file and the fit, or why there is none."""

    index: int
    path: str
    result: Optional[FitResult] = None
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.result is not None and not self.error


def fit_frame(index: int, path: str, curve: Curve, start: FitModel, settings: SeriesSettings,
              stop: Optional[Callable[[], bool]] = None) -> FrameFit:
    """The fit of one frame (the halves, range and left-out points of Single analysis)."""
    try:
        data = prepare_curve(curve, settings.side, settings.q_range, settings.excluded)
        if data.q.size < 3:
            return FrameFit(index, path, error="too few points in the fitting range")
        return FrameFit(index, path, fit(start, data, method=settings.method, stop=stop))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return FrameFit(index, path, error=str(exc) or type(exc).__name__)


def next_start(previous: Optional[FrameFit], single: FitModel, settings: SeriesSettings, *,
               new_stage: bool = False) -> FitModel:
    """Where the next frame starts: the previous frame's result (when it converged) or the Single model —
    also at the first frame of a new stage when ``start`` is ``"stages"``."""
    follows = settings.start == "previous" or (settings.start == "stages" and not new_stage)
    if follows and previous is not None and previous.ok and previous.result.converged:
        return previous.result.model
    return single


def stages_of_curves(curves: Sequence[Curve], settings: SeriesSettings) -> SeriesStages:
    """Stages and odd frames of the curves as they are fitted (the halves, range and left-out points),
    on the first curve's q (nm⁻¹)."""
    prepared = [prepare_curve(curve, settings.side, settings.q_range, settings.excluded) for curve in curves]
    grid = np.sort(np.asarray(prepared[0].q, dtype=float))
    image = np.full((len(prepared), grid.size), np.nan)
    for row, data in enumerate(prepared):
        order = np.argsort(data.q)
        q, intensity = np.asarray(data.q, dtype=float)[order], np.asarray(data.intensity, dtype=float)[order]
        if q.size < 2:
            continue
        inside = (grid >= q[0]) & (grid <= q[-1])
        image[row, inside] = np.interp(grid[inside], q, intensity)
    return find_stages(grid, image)


def parameter_columns(model: FitModel) -> list[tuple[tuple, str]]:
    """The free parameters of the model, as (path, column name) — the columns of the table."""
    columns = []
    for path, parameter in model.parameters():
        if not parameter.free:
            continue
        owner, key = path
        prefix = "" if owner == "globals" else f"{owner + 1}_{FAMILIES[model.components[owner].family][0].replace(' ', '_')}_"
        unit = INFO[key].unit
        columns.append((path, f"{prefix}{key}" + (f"_{unit}" if unit else "")))
    return columns


def series_table(frames: list[FrameFit], model: FitModel) -> tuple[list[str], list[list]]:
    """Header and rows: frame, file, χ²ᵣ, log RMSE, converged, then value and error of every free parameter."""
    columns = parameter_columns(model)
    header = ["frame", "file", "chi2_reduced", "log_rmse", "converged"]
    for _path, name in columns:
        header += [name, f"{name}_error"]
    rows = []
    for frame in sorted(frames, key=lambda item: item.index):
        row = [frame.index + 1, Path(frame.path).name]
        if not frame.ok:
            rows.append(row + [math.nan, math.nan, False] + [math.nan] * (2 * len(columns)))
            continue
        result = frame.result
        row += [result.chi2_reduced, result.log_rmse, bool(result.converged)]
        for path, _name in columns:
            row += [result.model.get(path).value, result.errors.get(path, math.nan)]
        rows.append(row)
    return header, rows


def series_record(model: FitModel, settings: SeriesSettings, folder: str, pattern: str, frames: list[FrameFit]) -> dict:
    """What is written next to the table: the start model and how every frame was fitted."""
    failed = [{"frame": frame.index + 1, "file": Path(frame.path).name, "why": frame.error}
              for frame in frames if not frame.ok]
    return {
        "schema": "gimap_series_fit_v1", "folder": folder, "pattern": pattern,
        "start_model": model_to_dict(model),
        "settings": {"side": settings.side, "q_range_nm^-1": None if settings.q_range is None else list(settings.q_range),
                     "left_out_q_nm^-1": sorted(settings.excluded), "method": settings.method, "start": settings.start},
        "frames": len(frames), "failed": failed,
        "not_converged": [frame.index + 1 for frame in frames if frame.ok and not frame.result.converged],
    }


__all__ = ["FrameFit", "STARTS", "SeriesSettings", "fit_frame", "next_start", "parameter_columns",
           "series_record", "series_table", "stages_of_curves"]
