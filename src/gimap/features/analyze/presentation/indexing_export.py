"""The displayed pole figure as CSV and its analysis, phases and fitted models as JSON."""

import csv
import json
import math
from pathlib import Path

import numpy as np

from ..application.export_record import analysis_metadata
from ..application.pole_figure import pole_figure_csv


def write_pole_export(path: Path, analysis, result, indexing: dict) -> list[Path]:
    """The pole figure as CSV at ``path`` and ``<stem>_indexing.json`` beside it (phases, reflections, fit)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = analysis_metadata(analysis)
    header = {"source": record.get("source") or str(analysis.path), "frame": record.get("frame"),
              "geometry": json.dumps(record.get("geometry"), ensure_ascii=False, default=str)}
    path.write_text(pole_figure_csv(result, header), encoding="utf-8")
    json_path = path.with_name(path.stem.replace("_pole_figure", "") + "_indexing.json")
    payload = {"kind": "GIMaP indexing and pole figure", "analysis": record, "indexing": indexing,
               "pole_figure": result.record()}
    json_path.write_text(json.dumps(_finite(payload), ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return [path, json_path]


def _finite(value):
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    return value


def write_crystal_export(path, analysis, indexing, records, phase, ring):
    record = analysis_metadata(analysis)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_path = path.with_suffix(".json")
    if json_path == path:
        json_path = path.with_name(path.stem + "_record.json")
    payload = {"kind": "GIMaP crystal orientation", "analysis": record, "indexing": indexing,
               "crystal_orientation": records}
    with path.open("w", encoding="utf-8", newline="") as out:
        out.write(f"# source: {analysis.path}\n# frame: {record.get('frame')}\n")
        out.write(f"# phase: {phase['phase']}\n# q_window_A^-1: {ring['q_window']}\n")
        out.write(f"# settings and joint fit: {json_path.name}\n")
        writer = csv.writer(out)
        components = ring["components"]
        writer.writerow(["chi_deg", "observed_N", "sigma", "measured", "fit_mask", "fitted_N",
                         *(family["hkl"] for family in components)])
        writer.writerows(zip(ring["chi_deg"], ring["observed"], ring["sigma"], ring["measured"],
                                 ring.get("fit_mask", ring["measured"]), ring["fitted"],
                                 *(family["values"] for family in components)))
    json_path.write_text(json.dumps(_finite(payload), ensure_ascii=False, indent=1, default=str, allow_nan=False), encoding="utf-8")
    return [path, json_path]
