"""A series as one image: the same curve of every frame, stacked (intensity against frame and q)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..domain import SeriesMap, q_from_two_theta, series_map
from .models import FrameAnalysis
from .ports import CurveWriter
from .use_cases import SOFTWARE

SERIES_TITLE = "GIMaP Analyze series map: rows = frames in list order, columns = x; mean intensity per bin, blank = no data"


def series_row(analysis: FrameAnalysis, key: str) -> tuple[np.ndarray, np.ndarray, str]:
    """``(x, intensity, x_label)`` of one curve of a frame; 2θ is converted to q (Å⁻¹)."""
    reduction = analysis.reduction
    curve = reduction.curve(key) if reduction is not None else None
    if curve is None or curve.is_empty:
        raise ValueError(f"{Path(analysis.path).name} has no '{key}' curve (it needs a geometry and this reduction).")
    x, label = np.asarray(curve.x, dtype=float), curve.x_label
    if label.startswith("2θ") and analysis.geometry is not None:
        x, label = q_from_two_theta(x, analysis.geometry.wavelength_angstrom), "q (Å⁻¹)"
    return x, np.asarray(curve.intensity, dtype=float), label


def row_label(analysis: FrameAnalysis) -> str:
    """The file, the frame of a multi-frame file, and how many frames were summed."""
    name = Path(analysis.path).name
    frame = f" #{analysis.frame_index + 1}" if int(analysis.frame_count or 1) > 1 else ""
    summed = len(analysis.summed_frames or ())
    return name + frame + (f" (+{summed} summed)" if summed else "")


def build_series_map(rows: Sequence[tuple[np.ndarray, np.ndarray, str, str, tuple]], curve: str) -> SeriesMap:
    """``rows``: ``(x, intensity, x_label, label, ref)`` in frame order."""
    if not rows:
        raise ValueError("No frame gave the curve.")
    return series_map(
        [(x, y) for x, y, _label, _name, _ref in rows], [name for *_rest, name, _ref in rows],
        x_label=rows[0][2], curve=curve, refs=[ref for *_rest, ref in rows],
    )


def export_series_track(writer: CurveWriter, series: SeriesMap, track, path: Path) -> Path:
    """The peak followed through the series (``track_peak``) as a CSV table, one row per frame."""
    unit = series.x_label[series.x_label.find("("):] if "(" in series.x_label else ""
    low, high = track.window
    comments = [
        "GIMaP Analyze series: the peak in one window of every frame (centroid and width above a straight background)",
        f"curve: {series.curve}; window: {low:.6g} - {high:.6g} {unit}",
    ]
    rows = [
        [index + 1, label, float(track.position[index]), float(track.fwhm[index]), float(track.area[index]),
         float(track.height[index])]
        for index, label in enumerate(series.labels)
    ]
    return writer.write_table(Path(path), comments, ["frame", "file", "position", "fwhm", "area", "height"], rows)


def export_series_map(writer: CurveWriter, series: SeriesMap, path: Path) -> Path:
    """CSV: the x axis in the first row, the frame number in the first column, a ``#`` line per frame."""
    metadata = {
        "software": SOFTWARE, "title": SERIES_TITLE, "curve": series.curve,
        "row_names": [f"{index + 1}: {label}" for index, label in enumerate(series.labels)],
    }
    frames = np.arange(1, series.rows + 1, dtype=float)
    return writer.write_map(series.image, series.x, frames, Path(path), (series.x_label, "frame"), metadata)


__all__ = ["SERIES_TITLE", "build_series_map", "export_series_map", "export_series_track", "row_label", "series_row"]
