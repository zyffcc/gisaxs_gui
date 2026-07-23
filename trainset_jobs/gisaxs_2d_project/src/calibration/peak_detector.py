from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths

from .radial_profile import RadialProfile


@dataclass
class DetectedPeak:
    radius_px: float
    intensity: float
    prominence: float
    width_px: float
    coverage: float


def detect_radial_peaks(profile: RadialProfile, min_radius_px: float = 5.0) -> list[DetectedPeak]:
    y = gaussian_filter1d(np.asarray(profile.intensity, dtype=np.float32), sigma=1.2)
    finite = np.isfinite(y)
    if not finite.any():
        return []
    # Polar profiles have a long all-invalid tail for centers near an edge.
    # Excluding poorly sampled radii prevents that tail from inflating the
    # noise estimate and hiding weak inner powder rings.
    max_count = float(np.max(profile.counts)) if profile.counts.size else 0.0
    if max_count <= 1000.0:
        statistical = finite & (profile.counts >= max(4.0, 0.05 * max_count))
    else:
        statistical = finite & (profile.counts >= 4.0)
    if statistical.sum() < 15:
        statistical = finite
    baseline = float(np.median(y[statistical]))
    mad = float(np.median(np.abs(y[statistical] - baseline))) + 1e-6
    prominence = max(2.3 * mad, float(np.percentile(y[statistical], 85) - baseline) * 0.08, 1e-5)
    indices, props = find_peaks(y, prominence=prominence, distance=3, width=(0.7, None))
    widths = peak_widths(y, indices, rel_height=0.5)[0] if indices.size else np.array([])
    peaks: list[DetectedPeak] = []
    for idx, width, prom in zip(indices, widths, props.get("prominences", [])):
        radius = float(profile.radius_px[idx])
        coverage = float(profile.coverage[idx])
        if radius < min_radius_px or profile.counts[idx] < 4 or coverage < 0.015:
            continue
        peaks.append(DetectedPeak(radius, float(y[idx]), float(prom), float(width), coverage))
    peaks.sort(key=lambda peak: peak.radius_px)
    return peaks[:60]
