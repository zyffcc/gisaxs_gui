"""Keep the reduction behind each Series row until its map or selected profile is exported."""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from src.gimap.shared.reproduction import replay_header


def spool_row(folder, analysis, key, label, metadata=None):
    """Store each distinct row recipe once on disk. The growing map retains only its filename."""
    if not folder:
        return label
    if metadata is None:
        from .export_record import analysis_metadata
        metadata = analysis_metadata(analysis, curve_keys=[])
    recipe = metadata.get("reproduction")
    if recipe is None:
        return label
    fields = ("source_file", "frame_index", "summed_frames", "instrument_profile", "reference_peak_scale",
              "error_model", "binning", "normalisation", "detector_name", "frame_shape", "header_values", "curves",
              "geometry_resolution", "intensity_corrections")
    saved = {field: metadata[field] for field in fields if field in metadata}
    saved["reproduction"] = {k: v for k, v in recipe.items() if k not in ("reference", "peak_fits")}
    saved["reproduction"]["curve_keys"] = [key]
    saved["native_x_label"] = analysis.reduction.curve(key).x_label
    raw = json.dumps(saved, sort_keys=True, separators=(",", ":")).encode("utf-8")
    path = Path(folder) / (hashlib.sha256(raw).hexdigest() + ".json")
    if not path.exists():
        with NamedTemporaryFile(dir=folder, delete=False) as stream:
            stream.write(raw)
        os.replace(stream.name, path)
    label.recipe_path = str(path)
    return label


def _csv_values(column):
    return np.array([float(format(value, ".8g")) for value in column])


def series_columns(series, row=None, *, numbers=None):
    """Match the original CSV precision and orientation. Preserve repeated x bins in the axis table."""
    x = _csv_values(series.x)
    if row is not None:
        return {"series_curve": {"x": x, "intensity": _csv_values(series.profile(row))}}
    frames = {"frame": np.array(series.numbers if numbers is None else numbers, dtype=np.int64)}
    for i, column in enumerate(series.image.T, 1):
        frames[f"bin_{i:04d}"] = _csv_values(column)
    return {"series_map": frames, "series_axis": {"x": x}}


def series_recipe(series, row=None):
    paths = [getattr(label, "recipe_path", None) for label in series.labels]
    if not all(paths):
        return None  # A caller can supply an in-memory SeriesMap without native detector inputs.
    frames = [json.loads(Path(path).read_bytes()) for path in paths]
    return {**replay_header("series", series_columns(series, row)), "frames": frames,
            "curve": series.curve, "view": series.view, "row": row}
