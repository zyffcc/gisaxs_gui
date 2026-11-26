"""Fitting ▸ Rank Incidence Angles…: which angle to measure next, from the fit in Single analysis
(``domain/angle_information.py``; Physics and Methods 9.20), and the record of what the operator chose.

The points of the cut give the pixels each one averages from its error: a Poisson count averaged over n pixels has
σ² = I / n, so n = I / σ² (at least 1); a curve without σ counts every point as one pixel. The record of a choice is
one JSON line appended to ``angle_decisions.jsonl`` in the folder of the curve: when, the curve, the model's
parameters, every candidate with its scores, the angle chosen and the operator's note — the trail of an adaptive run.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from src.gimap.shared.version import SOFTWARE

from ..domain.angle_information import BURIAL, MODEL_FLOOR, Burial, Footprint, ModelCut, score_angles
from ..domain.fit_engine import FitData
from ..domain.fit_model import FitModel

DECISIONS_NAME = "angle_decisions.jsonl"
NEEDS_ANGLES = "The fit has no incidence angle: Rank Incidence Angles needs αi of the curve (Model ▸ Geometry)."
NEEDS_SUBSTRATE = ("Rank Incidence Angles needs a known substrate (Model ▸ Substrate): the four-term DWBA gives how the "
                   "intensity changes with the incidence angle.")
NEEDS_PARAMETERS = "The model has no free shape or spacing parameter to rank angles for."
PARAMETER_NAMES = {"R": "R", "sigma_R": "σR/R", "D": "D", "sigma_nn": "σnn/D", "h": "h", "sigma_h": "σh/h",
                   "fraction": "buried fraction", "depth_nm": "burial depth ζ"}


@dataclass(frozen=True)
class AngleRequest:
    angles: tuple[float, ...]
    exposure_s: float = 1.0
    """Of each candidate exposure."""
    curve_exposure_s: float = 1.0
    """The exposure the fitted curve is per."""
    made: tuple[tuple[float, float], ...] = ()
    """``(angle, exposure)`` measured so far; empty: the curve's own angle and exposure."""
    floor: float = MODEL_FLOOR
    beam_height_mm: float = 0.03
    sample_length_mm: float = 20.0
    burial: Optional[Burial] = None


def pixels_of(data: FitData) -> np.ndarray:
    """Pixels behind each point, from its Poisson error (n = I / σ², at least 1)."""
    if data.sigma is None:
        return np.ones(data.q.size)
    sigma = np.asarray(data.sigma, float)
    intensity = np.clip(np.asarray(data.intensity, float), 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = np.where(sigma > 0, intensity / sigma**2, 1.0)
    return np.clip(np.nan_to_num(pixels, nan=1.0, posinf=1.0), 1.0, None)


def cut_of(model: FitModel, data: FitData, request: AngleRequest) -> ModelCut:
    alpha = float(model.globals["alpha_i"].value)
    if not alpha > 0:
        raise ValueError(NEEDS_ANGLES)
    if model.substrate is None:
        raise ValueError(NEEDS_SUBSTRATE)
    cut = ModelCut(model, np.asarray(data.q, float), pixels_of(data), exposure_s=float(request.curve_exposure_s),
                   burial=request.burial)
    if not cut.wanted():
        raise ValueError(NEEDS_PARAMETERS)
    return cut


def parameter_label(key: str, components: int) -> str:
    """“R”, “D₂”, “burial depth ζ”: a column of the table (``target:name``)."""
    target, name = key.split(":", 1)
    label = PARAMETER_NAMES.get(name, name)
    return label if target == BURIAL or components < 2 else f"{label} ({int(target) + 1})"


def rank_angles(model: FitModel, data: FitData, request: AngleRequest) -> list[dict]:
    """One row per candidate angle, in the order asked: ``alpha_i_deg``, ``counts``, ``gain``, ``bounds`` (relative
    Cramér–Rao bounds by ``target:name``) and ``rank`` (1: the largest gain)."""
    cut = cut_of(model, data, request)
    made = request.made or ((cut.reference_deg, float(request.curve_exposure_s)),)
    footprint = Footprint(request.beam_height_mm, request.sample_length_mm)
    scores = score_angles([cut], request.angles, request.exposure_s, made=made, floor=request.floor,
                          footprint=footprint)
    order = sorted(range(len(scores)), key=lambda i: -scores[i].gain if math.isfinite(scores[i].gain) else math.inf)
    ranks = {index: place + 1 for place, index in enumerate(order)}
    return [{"alpha_i_deg": score.alpha_i_deg, "counts": score.counts, "gain": score.gain, "bounds": dict(score.bounds),
             "rank": ranks[index]} for index, score in enumerate(scores)]


def decision_record(model: FitModel, curve_path: str, request: AngleRequest, rows: Sequence[dict],
                    chosen_deg: float, note: str = "") -> dict:
    parameters = {f"{index}:{key}": float(component.params[key].value)
                  for index, component in enumerate(model.components) for key in component.keys()}
    burial = None if request.burial is None else {
        "film_delta": request.burial.film_delta, "film_beta": request.burial.film_beta,
        "fraction": request.burial.fraction, "depth_nm": request.burial.depth_nm}
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "software": SOFTWARE,
        "curve": curve_path, "alpha_i_fitted_deg": float(model.globals["alpha_i"].value),
        "parameters": parameters, "burial": burial, "made": [list(item) for item in request.made],
        "exposure_s": request.exposure_s, "model_floor": request.floor,
        "footprint_mm": {"beam_height": request.beam_height_mm, "sample_length": request.sample_length_mm},
        "candidates": [_finite(dict(row)) for row in rows], "chosen_deg": float(chosen_deg), "note": note,
        "method": "Physics and Methods 9.20: Fisher information of the fitted horizontal cut",
    }


def append_decision(folder: Path, record: dict) -> Path:
    """Append the record to ``angle_decisions.jsonl`` in ``folder`` (made when missing); returns its path."""
    path = Path(folder) / DECISIONS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False, default=_plain) + "\n")
    return path


def _finite(value):
    """The same, with every number that is not finite (an angle that fixes nothing) as ``None``."""
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _plain(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Not JSON: {type(value).__name__}")


def with_burial(request: AngleRequest, *, delta: float, beta: float, fraction: float, depth_nm: float) -> AngleRequest:
    return replace(request, burial=Burial(float(delta), float(beta), float(fraction), float(depth_nm)))


__all__ = ["AngleRequest", "DECISIONS_NAME", "NEEDS_ANGLES", "NEEDS_PARAMETERS", "NEEDS_SUBSTRATE", "append_decision",
           "cut_of", "decision_record", "parameter_label", "pixels_of", "rank_angles", "with_burial"]
