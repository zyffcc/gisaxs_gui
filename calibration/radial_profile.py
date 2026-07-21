from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class RadialProfile:
    radius_px: np.ndarray
    intensity: np.ndarray
    coverage: np.ndarray
    counts: np.ndarray


def calculate_radial_profile(
    signal: np.ndarray,
    valid: np.ndarray,
    center_x_px: float,
    center_y_px: float,
    *,
    bin_width_px: float = 1.0,
) -> RadialProfile:
    height, width = signal.shape
    yy, xx = np.ogrid[:height, :width]
    radii = np.hypot(xx - float(center_x_px), yy - float(center_y_px))
    bins = np.floor(radii / float(bin_width_px)).astype(np.int32)
    selected = np.asarray(valid, dtype=bool)
    flat_bins = bins[selected].ravel()
    values = np.asarray(signal, dtype=np.float32)[selected].ravel()
    if flat_bins.size == 0:
        raise ValueError("No valid pixels are available for a radial profile.")
    length = int(flat_bins.max()) + 1
    counts = np.bincount(flat_bins, minlength=length).astype(np.float32)
    sums = np.bincount(flat_bins, weights=values, minlength=length).astype(np.float32)
    intensity = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    radius = (np.arange(length, dtype=np.float32) + 0.5) * float(bin_width_px)
    ideal = np.maximum(1.0, 2.0 * np.pi * radius * float(bin_width_px))
    coverage = np.clip(counts / ideal, 0.0, 1.0)
    return RadialProfile(radius, intensity, coverage, counts)


def calculate_azimuthal_profile(
    signal: np.ndarray,
    valid: np.ndarray,
    center_x_px: float,
    center_y_px: float,
    *,
    angle_bins: int = 360,
    percentile: float = 60.0,
    radial_step_px: float = 1.0,
) -> RadialProfile:
    """Robust polar profile that rejects localized GISAXS streaks and gaps.

    Powder rings stay at one radius over many azimuths, whereas the specular
    rod, Yoneda band, module gaps, cables, and aperture edges occupy limited or
    radius-varying azimuths.  A polar percentile therefore preserves partial
    rings without allowing a few bright pixels to dominate the profile.
    """
    try:
        import cv2
    except ImportError:
        return calculate_radial_profile(signal, valid, center_x_px, center_y_px)
    height, width = signal.shape
    corners = ((0.0, 0.0), (width - 1.0, 0.0), (0.0, height - 1.0), (width - 1.0, height - 1.0))
    max_radius = int(np.ceil(max(np.hypot(x - center_x_px, y - center_y_px) for x, y in corners)))
    radii = np.arange(0.0, max_radius + radial_step_px, radial_step_px, dtype=np.float32)
    angles = np.linspace(-np.pi, np.pi, int(angle_bins), endpoint=False, dtype=np.float32)
    map_x = (float(center_x_px) + np.cos(angles)[:, None] * radii[None, :]).astype(np.float32)
    map_y = (float(center_y_px) + np.sin(angles)[:, None] * radii[None, :]).astype(np.float32)
    polar = cv2.remap(
        np.asarray(signal, dtype=np.float32), map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )
    polar_valid = cv2.remap(
        np.asarray(valid, dtype=np.uint8), map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    counts = polar_valid.sum(axis=0).astype(np.float32)
    coverage_valid = counts / float(angle_bins)
    masked = np.where(polar_valid, polar, np.nan)
    populated = counts > 0
    intensity = np.zeros(radii.shape, dtype=np.float32)
    if np.any(populated):
        intensity[populated] = np.nanpercentile(
            masked[:, populated], float(percentile), axis=0
        ).astype(np.float32)

    # Estimate how much azimuth actually supports a local radial maximum. This
    # is distinct from merely having valid detector pixels at that radius.
    filled = np.where(polar_valid, polar, 0.0).astype(np.float32, copy=False)
    weights = gaussian_filter1d(polar_valid.astype(np.float32), 8.0, axis=1, mode="nearest")
    radial_background = gaussian_filter1d(filled, 8.0, axis=1, mode="nearest")
    radial_background /= np.maximum(weights, 1e-3)
    excess = np.where(polar_valid, polar - radial_background, np.nan)
    median = np.zeros(radii.shape, dtype=np.float32)
    mad = np.zeros(radii.shape, dtype=np.float32)
    if np.any(populated):
        median[populated] = np.nanmedian(excess[:, populated], axis=0)
        mad[populated] = np.nanmedian(
            np.abs(excess[:, populated] - median[None, populated]), axis=0
        )
    threshold = median + np.maximum(1.5 * mad, 0.005)
    supported = polar_valid & (excess > threshold[None, :])
    signal_coverage = supported.sum(axis=0).astype(np.float32) / float(angle_bins)
    coverage = np.minimum(coverage_valid, signal_coverage)
    return RadialProfile(radii, intensity, coverage, counts)


def annular_azimuthal_coverage(
    valid: np.ndarray,
    center_x_px: float,
    center_y_px: float,
    radius_px: float,
    half_width_px: float = 2.5,
    angle_bins: int = 72,
) -> float:
    height, width = valid.shape
    x0 = max(0, int(center_x_px - radius_px - half_width_px))
    x1 = min(width, int(center_x_px + radius_px + half_width_px) + 1)
    y0 = max(0, int(center_y_px - radius_px - half_width_px))
    y1 = min(height, int(center_y_px + radius_px + half_width_px) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance = np.hypot(xx - center_x_px, yy - center_y_px)
    annulus = (np.abs(distance - radius_px) <= half_width_px) & valid[y0:y1, x0:x1]
    if not np.any(annulus):
        return 0.0
    angles = np.arctan2((yy - center_y_px) + np.zeros_like(xx), (xx - center_x_px) + np.zeros_like(yy))[annulus]
    occupied = np.unique(np.floor((angles + np.pi) * angle_bins / (2.0 * np.pi)).astype(int) % angle_bins)
    return float(len(occupied) / angle_bins)
