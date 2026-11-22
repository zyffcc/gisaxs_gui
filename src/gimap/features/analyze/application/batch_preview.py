"""Small in-memory fit previews, independent of whether fitted curves are written to disk.

Particle-model fits return their actual prepared points in nm⁻¹. The preview copies just the most probable
solution and converts that axis to Å⁻¹ for Analyze; intensities and the fitted values are unchanged.
No detector frames or discarded candidate solutions are retained.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .batch_fit import model_name


@dataclass(frozen=True)
class FitCurvePreview:
    name: str
    x: np.ndarray
    observed: np.ndarray
    fitted: np.ndarray


@dataclass(frozen=True)
class FrameFitPreview:
    label: str
    number: Optional[int]
    ref: tuple[Path, int]
    ok: bool
    values: dict
    curves: tuple[FitCurvePreview, ...] = ()
    note: str = ""
    preparation: dict = field(default_factory=dict)
    cache_key: Optional[int] = None
    intervals: dict = field(default_factory=dict)


def _array(values) -> np.ndarray:
    return np.asarray(() if values is None else values, dtype=float).reshape(-1).copy()


def _ref(outcome) -> tuple[Path, int]:
    return Path(outcome.metadata.get("source_file") or outcome.label), int(outcome.metadata.get("frame_index") or 0)


def component_intervals(solution):
    """The reported 68% intervals of Predict, keyed by their columns in the batch parameter table."""
    intervals = {}
    posterior = solution.get("posterior") or {}
    for index, component in enumerate(solution.get("components") or (), 1):
        prefix = f"c{index} {component.get('type', '')}"
        for key in component.get("params") or {}:
            source = "sigma_nn" if key == "sigma_D" else key
            values = posterior.get(f"{source}_{index}") or ()
            if len(values) < 4 or values[2] is None or values[3] is None:
                continue
            low, high = values[2:4]
            if math.isfinite(low) and math.isfinite(high) and low <= high:
                scale = 1 / math.sqrt(2 * math.pi) if key == "sigma_D" else 1
                intervals[f"{prefix} {key}"] = (low * scale, high * scale)
    return intervals


def model_preview(outcome, rows, values: dict, *, error: str = "") -> FrameFitPreview:
    curves, preparation = [], {}
    if rows:
        best = rows[0]
        preparation = dict(best.get("preparation") or {})
        q, observed, fitted = (_array(best.get(key)) for key in ("native_q", "observed", "native_fit"))
        if q.size and q.size == observed.size == fitted.size:
            curves.append(FitCurvePreview(model_name(best), np.abs(q) * 0.1, observed, fitted))
        else:
            q, fitted = (_array(best.get(key)) for key in ("display_q", "display_fit"))
            if q.size and q.size == fitted.size:
                curves.append(FitCurvePreview(model_name(best), np.abs(q) * 0.1,
                                              np.array([], dtype=float), fitted))
    return FrameFitPreview(outcome.label, outcome.number, _ref(outcome), bool(rows), dict(values), tuple(curves),
                           error or str(values.get("note") or ""), preparation,
                           intervals=component_intervals(rows[0]) if rows else {})


def peak_preview(outcome, fits, values: dict) -> FrameFitPreview:
    curves = tuple(FitCurvePreview(target.name, _array(fit.x), _array(fit.y), _array(fit.fitted))
                   for target, fit in fits if fit.ok and fit.x.size)
    failed = [f"{target.name}: {fit.message}" for target, fit in fits if not fit.ok]
    return FrameFitPreview(outcome.label, outcome.number, _ref(outcome), bool(curves), dict(values), curves,
                           "; ".join(failed))


def failed_preview(label: str, number: Optional[int], ref, reason: str) -> FrameFitPreview:
    return FrameFitPreview(label, number, (Path(ref[0]), int(ref[1])), False,
                           {"frame": label, "points": 0, "note": reason}, note=reason)


__all__ = ["FitCurvePreview", "FrameFitPreview", "component_intervals", "failed_preview", "model_preview", "peak_preview"]
