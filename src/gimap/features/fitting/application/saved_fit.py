"""Read an exported single-curve fit as the starting model and selection for another fit."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.gimap.shared.reproduction import verified_path
from .curve_sources import ColumnMapping
from .models import InSituSourceFrame
from .single_fit import SIDES, FitModel, model_from_dict


@dataclass(frozen=True)
class SavedFit:
    path: Path
    sha256: str
    source: InSituSourceFrame
    source_sha256: str
    unit: str
    model: FitModel
    side: str
    q_range: tuple | None
    excluded: frozenset
    count: int
    quality: float | None
    weighting: str
    method: str
    bounds: frozenset
    errors: dict


def read_saved_fit(path) -> SavedFit:
    path = Path(path)
    content = path.read_bytes()
    record = json.loads(content)
    if not isinstance(record, dict) or not all(isinstance(record.get(key), dict) for key in ("curve", "model", "points")):
        raise ValueError("Choose the JSON written by Save Curve and Parameters.")
    curve, points = record["curve"], record["points"]
    source = curve.get("source")
    if not isinstance(source, dict) or not source.get("path") or not source.get("sha256"):
        raise ValueError("This fit record has no source curve with a SHA-256 digest.")
    physical = verified_path(source, path.parent)
    columns = ColumnMapping.from_dict(source["columns"]) if source.get("columns") else None
    frame = InSituSourceFrame(physical, frame_index=source.get("frame_index", 0), columns=columns)
    side, unit = points["side"], curve["q_unit_of_file"]
    q_range = points.get("q_range_nm^-1")
    if q_range is not None:
        q_range = tuple(float(v) for v in q_range)
        if len(q_range) != 2 or not np.isfinite(q_range).all() or q_range[0] >= q_range[1]:
            raise ValueError("Invalid fitting range in the saved record.")
    excluded = frozenset(float(v) for v in points.get("left_out_q_nm^-1") or ())
    if side not in dict(SIDES) or unit not in ("angstrom", "nm") or not all(np.isfinite(v) for v in excluded):
        raise ValueError("Invalid curve selection in the saved record.")
    fit = record.get("fit") or {}
    method = fit.get("method") or ("predict" if record.get("predict", {}).get("solutions") else "local")
    quality = float(fit["chi2_reduced"]) if fit.get("chi2_reduced") is not None else None
    errors = {key: None if value is None else float(value) for key, value in (fit.get("errors") or {}).items()}
    return SavedFit(path.resolve(), hashlib.sha256(content).hexdigest(), frame, source["sha256"], unit,
                    model_from_dict(record["model"]),
                    side, q_range, excluded, int(points["count"]), quality, points.get("weighting", ""),
                    method, frozenset(fit.get("at_bounds") or ()), errors)
