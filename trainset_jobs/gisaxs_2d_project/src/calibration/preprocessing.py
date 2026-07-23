from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .models import DetectorImage


@dataclass
class AnalysisImage:
    signal: np.ndarray
    valid: np.ndarray
    valid_fraction: float


def preprocess_detector_image(image: DetectorImage, subtract_background: bool = True) -> AnalysisImage:
    source = np.asarray(image.data)
    valid = np.isfinite(source)
    if image.mask is not None:
        valid &= ~np.asarray(image.mask, dtype=bool)
    # Zero-filled regions in CBF files include module gaps, the area outside
    # the illuminated detector aperture, and inactive electronics.  They carry
    # no ring-position information and otherwise dominate radial averages.
    valid &= source > 0
    if not np.any(valid):
        raise ValueError("The detector image contains no usable pixels.")

    sample = source[valid][:: max(1, int(valid.sum()) // 250_000)]
    high = float(np.nanpercentile(sample, 99.95))
    maximum = float(np.nanmax(sample))
    if maximum > max(high * 20.0, 1e6):
        valid &= source < maximum
    clip_high = max(0.0, float(np.nanpercentile(sample, 99.9)))
    signal = np.zeros(source.shape, dtype=np.float32)
    np.clip(source, 0.0, clip_high if clip_high > 0 else None, out=signal, where=valid)
    np.log1p(signal, out=signal)
    if subtract_background and min(signal.shape) >= 64:
        # Normalized smoothing avoids detector gaps bleeding into the background.
        sigma = max(6.0, min(signal.shape) / 90.0)
        weights = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
        background = gaussian_filter(signal, sigma=sigma, mode="nearest")
        background /= np.maximum(weights, 1e-3)
        signal -= background.astype(np.float32, copy=False)
        signal[signal < 0] = 0
    signal[~valid] = 0
    return AnalysisImage(signal=signal, valid=valid, valid_fraction=float(np.mean(valid)))
