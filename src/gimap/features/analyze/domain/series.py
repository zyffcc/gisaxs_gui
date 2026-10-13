"""Corrections across an in-situ series, keyed to a reference diffraction peak.

During a heating or drying run the sample height (and therefore the
sample–detector distance) and the incident flux drift.  A peak that does not
change — a substrate or internal standard reflection — measures both: its q
position gives the distance, its height the flux.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeriesCorrection:
    reference_q: float = 2.132
    """q of the reference peak (Å⁻¹)."""
    half_width: float = 0.035
    """The peak is searched within reference_q ± half_width."""
    align_distance: bool = False
    """Scale the detector distance of each frame so the peak sits at reference_q."""
    normalize: bool = False
    """Scale intensities so the peak has ``target_intensity``."""
    target_intensity: float = 1.0
    per_frame: bool = False
    """Normalise every frame to its own peak (``False``: the first frame's factor for all)."""

    @property
    def is_identity(self) -> bool:
        return not (self.align_distance or self.normalize)


def locate_reference_peak(
    q: np.ndarray, intensity: np.ndarray, target: float, half_width: float
) -> tuple[float, float]:
    """``(q, I)`` of the strongest finite point within ``target ± half_width``."""
    q = np.asarray(q, dtype=float)
    values = np.asarray(intensity, dtype=float)
    window = (
        np.isfinite(q) & np.isfinite(values)
        & (q >= float(target) - float(half_width))
        & (q <= float(target) + float(half_width))
    )
    if not window.any():
        raise ValueError(f"No data within {target:.4g} ± {half_width:.4g} Å⁻¹ for the reference peak.")
    indices = np.flatnonzero(window)
    best = int(indices[np.argmax(values[window])])
    return float(q[best]), float(values[best])


def aligned_distance(distance_m: float, observed_q: float, target_q: float) -> float:
    """Distance that moves a peak seen at ``observed_q`` to ``target_q`` (small-angle scaling)."""
    if distance_m <= 0 or observed_q <= 0 or target_q <= 0:
        raise ValueError("Distance and peak positions must be positive.")
    return float(distance_m) * float(observed_q) / float(target_q)


def normalization_factor(peak_intensity: float, target_intensity: float = 1.0) -> float:
    if not (np.isfinite(peak_intensity) and peak_intensity > 0 and target_intensity > 0):
        raise ValueError("The reference peak must have a positive intensity to normalise.")
    return float(target_intensity) / float(peak_intensity)


__all__ = [
    "SeriesCorrection",
    "aligned_distance",
    "locate_reference_peak",
    "normalization_factor",
]
