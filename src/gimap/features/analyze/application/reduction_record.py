"""Inputs and full-precision reference columns for one reduction."""

from dataclasses import asdict

import numpy as np

from src.gimap.shared.reproduction import replay_header

from ..domain.mask_bitmap import bitmap_record
from .export_record import _plain, fit_input_curve
from .settings_file import AnalyzeSettings, settings_record


def reduction_tables(analysis, keys):
    tables = {}
    for key in keys:
        if key == "fit_input":
            curve = fit_input_curve(analysis)
        else:
            curve = analysis.reduction.curve("vertical" if key == "fit_input_qz" else key)
        if curve is None:
            raise ValueError(f"The reduction did not produce the recorded curve {key!r}.")
        tables[key] = {name: getattr(curve, name) for name in ("x", "intensity", "sigma", "pixels")}
    return tables


def reduction_record(analysis, inputs, keys=None):
    request = analysis.request
    settings = settings_record(AnalyzeSettings(
        sample_roll_deg=request.sample_roll_deg, sample_roll_measurement=analysis.sample_roll_measurement,
        mode=analysis.kind, gisaxs=request.gisaxs, giwaxs=request.giwaxs, corrections=analysis.corrections,
    ))
    if keys is None:
        keys = [curve.key for curve in analysis.reduction.curves if not curve.is_empty]
    tables = reduction_tables(analysis, keys)
    return {
        **replay_header("reduction", tables),
        "inputs": inputs,
        "geometry": asdict(analysis.geometry),
        "settings": _plain({key: settings[key] for key in ("format", "version", "mode", "gisaxs", "giwaxs", "corrections")}),
        "mask": bitmap_record(analysis.drawn_mask, analysis.shape),
        "excluded_pixels": bitmap_record(~np.asarray(analysis.valid, dtype=bool), analysis.shape),
        "series_cut": _plain(request.series_cut),
        "curve_keys": list(keys),
    }
