"""Probability-mass comparison with recorded optional display transforms."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .models import SemComparison, SemComparisonSettings, SemMeasurements, SemPrediction


def prediction_edges(axis: np.ndarray) -> np.ndarray:
    edges = np.r_[axis[0] - (axis[1] - axis[0]) / 2,
                  (axis[:-1] + axis[1:]) / 2,
                  axis[-1] + (axis[-1] - axis[-2]) / 2]
    edges[0] = max(0.0, edges[0])
    return edges


def _rebin_prediction(prediction, edges):
    source_edges = prediction_edges(prediction.radius_nm)
    probability = prediction.probability / prediction.probability.sum()
    cdf = np.r_[0.0, np.cumsum(probability)]
    mapped = np.interp(edges, source_edges, cdf, left=0.0, right=1.0)
    masses = np.maximum(0, np.diff(mapped))
    retained = float(masses.sum())
    if retained <= 0:
        raise ValueError("Chosen radius range does not contain prediction probability")
    return masses / retained, float(np.clip(1 - retained, 0, 1))


def _histogram(values, edges):
    counts = np.histogram(values, bins=edges)[0]
    retained = int(counts.sum())
    if retained <= 0:
        raise ValueError("Chosen radius range does not contain any accepted SEM particles")
    return counts / retained, int(values.size - retained)


def compare_distributions(measurements: SemMeasurements, prediction: SemPrediction | None,
                          settings: SemComparisonSettings) -> SemComparison:
    if not isinstance(settings.bins, int) or not 2 <= settings.bins <= 1000:
        raise ValueError("Histogram bins must be an integer between 2 and 1000")
    if not np.isfinite(settings.radius_scale) or settings.radius_scale <= 0:
        raise ValueError("SEM radius scale must be finite and positive")
    if settings.display_mode not in {"probability", "paper"}:
        raise ValueError("Unknown comparison display mode")
    values = measurements.radius_nm * settings.radius_scale
    if values.size == 0:
        raise ValueError("No accepted SEM particles are available")
    low = 0.0 if settings.radius_min_nm is None else float(settings.radius_min_nm)
    high = float(values.max()) if settings.radius_max_nm is None else float(settings.radius_max_nm)
    if prediction is not None and settings.radius_max_nm is None:
        high = max(high, float(prediction_edges(prediction.radius_nm)[-1]))
    if not np.isfinite(low) or not np.isfinite(high) or low < 0 or high <= low:
        raise ValueError("Radius limits must be finite, non-negative and increasing")
    edges = np.linspace(low, high, settings.bins + 1)
    sem, excluded_sem = _histogram(values, edges)
    model, excluded_model = None, None
    if prediction is not None:
        model, excluded_model = _rebin_prediction(prediction, edges)
    centers = (edges[:-1] + edges[1:]) / 2
    display_edges, display_sem = edges, sem
    display_model_axis = centers if model is not None else None
    display_model = model
    amplitude_scale = 1.0
    if settings.display_mode == "paper":
        display_edges = np.linspace(low, high, 61)
        display_sem, _ = _histogram(values, display_edges)
        if prediction is not None:
            display_model_axis = prediction.radius_nm
            display_model = prediction.probability / prediction.probability.sum()
            display_model = display_model - display_model.min()
            if display_model.max() > 0:
                amplitude_scale = float(sem.max() / display_model.max())
                display_model = display_model * amplitude_scale
    return SemComparison(
        bin_edges_nm=edges, sem_probability=sem, prediction_probability=model,
        display_sem_edges_nm=display_edges, display_sem_values=display_sem,
        display_prediction_radius_nm=display_model_axis,
        display_prediction_values=display_model,
        metadata={
            "settings": asdict(settings), "quantity": "radius", "unit": "nm",
            "sem_source": dict(measurements.source), "sem_measurement_settings": dict(measurements.settings),
            "sem_summary": dict(measurements.summary),
            "prediction_source": prediction.source if prediction is not None else None,
            "prediction_metadata": dict(prediction.metadata) if prediction is not None else None,
            "prediction_input_radius_nm": prediction.radius_nm.tolist() if prediction is not None else None,
            "prediction_input_probability": prediction.probability.tolist() if prediction is not None else None,
            "prediction_included": prediction is not None,
            "normalization": "probability_mass_per_common_bin; normalize retained observations/mass to sum=1",
            "prediction_rebin": "piecewise_uniform_probability_mass_between_midpoint_bin_edges",
            "sem_count_total": int(values.size), "sem_count_outside_range": excluded_sem,
            "prediction_probability_outside_range": excluded_model,
            "sem_radius_scale": float(settings.radius_scale),
            "display_mode": settings.display_mode,
            "display_sem_bins": len(display_edges) - 1,
            "display_model_minimum_subtracted": settings.display_mode == "paper" and prediction is not None,
            "display_model_amplitude_scale": amplitude_scale,
            "display_model_is_probability": settings.display_mode == "probability",
            "display_axis_shift_nm": 0.0,
        },
    )
