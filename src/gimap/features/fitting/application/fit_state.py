"""Fit results as plain JSON values, for a project (File ▸ Save Project) and back: a fit's ``FitResult``, the points
it was made on (``selection``), a frame of a series (``FrameFit``) and a series' settings.

A path of the model is written ``"<owner>:<key>"`` (``"0:R"``, ``"globals:background"``), as the fit's record does
(``single_fit.fit_record``); a number that is not finite is written ``null`` and read back as NaN. Reading never
fails on a damaged entry: it gives ``None`` (``ValueError``/``KeyError``/``TypeError`` are caught by the callers that
say what could not be read).
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Optional

from ..domain.fit_engine import FitResult
from ..domain.fit_model import model_from_dict, model_to_dict
from .series_fit import FrameFit, SeriesSettings

STATE_SCHEMA = "gimap_fit_state_v1"


def number(value) -> Optional[float]:
    """A JSON number (``None`` for NaN and ±inf)."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def read_number(value) -> float:
    return math.nan if value is None else float(value)


def path_key(path) -> str:
    owner, key = path
    return f"{owner}:{key}"


def path_of(text: str) -> tuple:
    owner, _, key = str(text).partition(":")
    if not key:
        raise ValueError(f"not a parameter path: {text!r}")
    return ("globals" if owner == "globals" else int(owner)), key


def result_to_dict(result: Optional[FitResult]) -> Optional[dict]:
    if result is None:
        return None
    return {
        "model": model_to_dict(result.model), "method": result.method, "weighting": result.weighting,
        "points": int(result.points), "free": int(result.free), "chi2_reduced": number(result.chi2_reduced),
        "log_rmse": number(result.log_rmse), "evaluations": int(result.evaluations), "seconds": number(result.seconds),
        "converged": bool(result.converged), "stopped": bool(result.stopped), "message": str(result.message),
        "errors": {path_key(path): number(value) for path, value in result.errors.items()},
        "correlated": [[path_key(a), path_key(b), number(rho)] for a, b, rho in result.correlated],
        "at_bounds": [path_key(path) for path in result.at_bounds], "start_kept": bool(result.start_kept),
    }


def result_from_dict(data: Optional[dict]) -> Optional[FitResult]:
    if not data:
        return None
    return FitResult(
        model_from_dict(data["model"]), str(data["method"]), str(data["weighting"]), int(data["points"]),
        int(data["free"]), read_number(data.get("chi2_reduced")), read_number(data.get("log_rmse")),
        int(data.get("evaluations") or 0), read_number(data.get("seconds")), bool(data.get("converged")),
        bool(data.get("stopped")), str(data.get("message") or ""),
        errors={path_of(key): read_number(value) for key, value in (data.get("errors") or {}).items()},
        correlated=tuple((path_of(a), path_of(b), read_number(rho)) for a, b, rho in data.get("correlated") or ()),
        at_bounds=tuple(path_of(key) for key in data.get("at_bounds") or ()),
        start_kept=bool(data.get("start_kept")))


def selection_to_dict(selection) -> Optional[dict]:
    """The points a fit was made on: (halves, range, left-out points) of ``FitSession.selection``."""
    if selection is None:
        return None
    side, q_range, excluded = selection
    return {"side": side, "q_range": None if q_range is None else [float(v) for v in q_range],
            "excluded": sorted(float(value) for value in excluded)}


def selection_from_dict(data: Optional[dict]):
    if not data:
        return None
    q_range = data.get("q_range")
    return (str(data.get("side") or "mean"), None if q_range is None else tuple(sorted(float(v) for v in q_range)),
            frozenset(float(value) for value in data.get("excluded") or ()))


def frame_to_dict(frame: FrameFit) -> dict:
    out = {item.name: getattr(frame, item.name) for item in fields(frame) if item.name != "result"}
    out["resolved"] = list(frame.resolved)
    out["result"] = result_to_dict(frame.result)
    return out


def frame_from_dict(data: dict) -> FrameFit:
    known = {item.name for item in fields(FrameFit)} - {"result", "resolved"}
    values = {key: value for key, value in data.items() if key in known}
    return FrameFit(**values, result=result_from_dict(data.get("result")), resolved=tuple(data.get("resolved") or ()))


def settings_to_dict(settings: SeriesSettings) -> dict:
    return {"side": settings.side, "q_range": None if settings.q_range is None else [float(v) for v in settings.q_range],
            "excluded": sorted(float(v) for v in settings.excluded), "method": settings.method, "start": settings.start,
            "resolve_edges": bool(settings.resolve_edges), "follow": bool(settings.follow)}


def settings_from_dict(data: dict) -> SeriesSettings:
    q_range = data.get("q_range")
    return SeriesSettings(side=str(data.get("side") or "mean"),
                          q_range=None if q_range is None else tuple(float(v) for v in q_range),
                          excluded=frozenset(float(v) for v in data.get("excluded") or ()),
                          method=str(data.get("method") or "local"), start=str(data.get("start") or "previous"),
                          resolve_edges=bool(data.get("resolve_edges", True)), follow=bool(data.get("follow", True)))


__all__ = ["STATE_SCHEMA", "frame_from_dict", "frame_to_dict", "number", "path_key", "path_of", "read_number",
           "result_from_dict", "result_to_dict", "selection_from_dict", "selection_to_dict", "settings_from_dict",
           "settings_to_dict"]
