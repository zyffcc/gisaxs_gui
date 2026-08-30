"""Scientific primitives for WAXS batch curve preprocessing."""

from __future__ import annotations

import numpy as np


def locate_reference_peak(
    x: np.ndarray,
    intensity: np.ndarray,
    target: float,
    half_width: float,
) -> tuple[float, float]:
    """Return the strongest finite, unlogged point near a reference q."""
    q = np.asarray(x, dtype=float)
    values = np.asarray(intensity, dtype=float)
    valid = (
        np.isfinite(q)
        & np.isfinite(values)
        & (q >= float(target) - float(half_width))
        & (q <= float(target) + float(half_width))
    )
    if not np.any(valid):
        raise RuntimeError(
            f"No finite peak data within {target:.6g} ± {half_width:.6g} Å⁻¹."
        )
    indices = np.flatnonzero(valid)
    peak_index = int(indices[np.argmax(values[valid])])
    return float(q[peak_index]), float(values[peak_index])


def aligned_detector_distance(
    current_distance: float,
    observed_peak_q: float,
    target_q: float,
) -> float:
    """Scale SDD so the observed radial-q peak moves toward the target q."""
    distance = float(current_distance)
    observed = float(observed_peak_q)
    target = float(target_q)
    if distance <= 0.0 or observed <= 0.0 or target <= 0.0:
        raise ValueError("Calibration distance and q values must be positive.")
    return distance * observed / target


def peak_normalization_factor(
    peak_intensity: float, target_intensity: float = 1.0
) -> float:
    """Return the linear multiplier mapping an unlogged peak to a target."""
    peak = float(peak_intensity)
    target = float(target_intensity)
    if not np.isfinite(peak) or peak <= 0.0 or not np.isfinite(target) or target <= 0.0:
        raise RuntimeError(
            "Normalization peak and target intensities must be finite and positive."
        )
    return target / peak
