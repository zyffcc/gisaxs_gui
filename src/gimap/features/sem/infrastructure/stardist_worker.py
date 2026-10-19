"""Standalone local StarDist worker. Never imported to load models in the GUI."""

from __future__ import annotations

import importlib.metadata
import json
import math
from pathlib import Path
import sys
import traceback

import numpy as np


def normalize_roi(raw, *, bright_particles=True, low=1.0, high=99.8):
    values = np.asarray(raw, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("SEM ROI has no finite pixels for StarDist")
    values = values.copy()
    values[~finite] = np.median(values[finite])
    if not bright_particles:
        values = -values
    lower, upper = (float(v) for v in np.percentile(values[finite], [low, high]))
    if upper <= lower:
        raise ValueError("SEM ROI has no contrast for StarDist")
    normalized = (values - lower) / (upper - lower)
    return normalized, finite, {
        "low_percentile": float(low), "high_percentile": float(high),
        "low_value": lower, "high_value": upper, "clip": False,
        "formula": "(oriented_intensity - low_value) / (high_value - low_value)",
        "polarity": "stored_grayscale" if bright_particles else "negate_stored_grayscale_before_normalization",
        "non_finite_count": int((~finite).sum()),
        "non_finite_model_input_fill": "median of finite ROI pixels; labels at invalid pixels removed",
        "model_input_dtype": str(normalized.dtype), "channel_strategy": "single_channel_YX",
    }


def predict(request):
    # The optional heavy packages stay confined to this process. Model loading
    # uses a local directory only: no from_pretrained/download calls here.
    import tensorflow as tf
    from stardist.models import StarDist2D

    tf.config.set_visible_devices([], "GPU")
    settings = request["settings"]
    raw = np.load(request["input"], allow_pickle=False)
    normalized, finite, normalization = normalize_roi(
        raw, bright_particles=settings["bright_particles"],
        low=settings["normalize_low"], high=settings["normalize_high"],
    )
    folder = Path(request["model_directory"])
    model = StarDist2D(None, name=folder.name, basedir=str(folder.parent))
    if model.config.n_channel_in != 1:
        raise ValueError("Choose a single-channel 2D StarDist model for grayscale SEM images")
    probability = settings["probability_threshold"]
    overlap = settings["overlap_threshold"]
    probability = float(model.thresholds.prob if probability is None else probability)
    overlap = float(model.thresholds.nms if overlap is None else overlap)
    scale = float(settings["ai_scale"])
    tiles = tuple(max(1, int(math.ceil(length * scale / 768))) for length in raw.shape)
    labels, details = model.predict_instances(
        normalized, axes="YX", scale=(scale, scale), n_tiles=tiles,
        prob_thresh=probability, nms_thresh=overlap, show_tile_progress=False,
    )
    if labels.shape != raw.shape:
        raise ValueError("The StarDist runtime did not restore the original ROI label shape")
    labels[~finite] = 0
    np.save(request["output"], labels, allow_pickle=False)
    score = np.asarray(details.get("prob", []), float)
    record = {
        "algorithm": "StarDist2D star-convex instance segmentation",
        "versions": {
            **{name: importlib.metadata.version(name) for name in ("stardist", "csbdeep", "numpy")},
            # Windows installs the tensorflow-intel distribution, which still
            # imports as tensorflow. Record the runtime we actually imported.
            "tensorflow": str(tf.__version__),
        },
        "python_version": sys.version, "execution_device": "CPU",
        "normalization": normalization, "input_roi_shape": list(raw.shape),
        "prediction_scale": [scale, scale], "n_tiles": list(tiles),
        "probability_threshold": probability, "overlap_nms_threshold": overlap,
        "output_roi_shape": list(labels.shape),
        "coordinate_restoration": "StarDist predict_instances scale maps polygon coordinates to original ROI before rasterizing labels",
        "measurement_grid": "original_image_pixels; inference scale is not a physical calibration",
        "instance_count_before_filters": int(len(np.unique(labels[labels > 0]))),
        "object_score_summary": {"count": int(score.size),
                                 "min": float(score.min()) if score.size else None,
                                 "max": float(score.max()) if score.size else None},
        "model_config": {"n_channel_in": int(model.config.n_channel_in),
                         "n_rays": int(model.config.n_rays), "grid": list(model.config.grid)},
    }
    Path(request["record"]).write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    try:
        predict(request)
    except Exception as exc:
        Path(request["error"]).write_text(json.dumps({
            "message": f"Local StarDist failed: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }), encoding="utf-8")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
