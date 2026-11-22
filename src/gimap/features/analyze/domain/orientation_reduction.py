"""Detector pixel means for ring orientation, with the same intensity units for the radial background."""

from __future__ import annotations

import math

import numpy as np

from src.gimap.shared.geometry.conventions import index_center_from_canonical
from src.gimap.shared.geometry.q_mapping import exit_directions, lab_rays, refracted_qz

from .binning import BinnedMean
from .intensity import polarization_factor
from .models import Curve
from .peak_fit import GAUSSIAN, fit_peak

DEFAULT_POLARIZATION = 0.99
BACKGROUND_PEAK = "local linear background of a Gaussian fit to I(q)"
BACKGROUND_FLANKS = "mean of I(q) beside the window"


def minimum_exit(geometry, override):
    if override is not None:
        return float(override), "user setting"
    if geometry.refraction:
        return 2 * geometry.critical_angle_deg, "2 × αc"
    return 2 * geometry.incidence_deg, "2 × αi"


def _background(q, intensity, scale, counting, low, high, step):
    half = 0.5 * (high - low)
    left, right = low - 2*half, high + 2*half
    bins = min(3000, max(1, round((right-left) / max(step, 1e-10))))
    radial = BinnedMean.linear(left, right, bins)
    radial.add(q, intensity, scale)
    x, y, sigma, _ = radial.result(counts=counting)
    fit = fit_peak(x, y, sigma, (left, right), profile=GAUSSIAN)
    if fit.ok and low <= fit.center <= high and np.isfinite(fit.background_center):
        error = fit.background_center_err
        return fit.background_center, BACKGROUND_PEAK, error if np.isfinite(error) else None
    beside = ((x >= low-half) & (x < low)) | ((x > high) & (x <= high+half))
    if beside.any():
        return float(y[beside].mean()), BACKGROUND_FLANKS, None
    return 0.0, "none", None


def ring_profiles(maps, image, valid, counts, geometry, window, step, corrections):
    """Reduce actual pixels inside the ring at each exit-angle threshold. Image excludes L and mirror fill."""
    low, high = sorted(window)
    width = high - low
    near = valid & maps.above_horizon & (maps.q >= low-width) & (maps.q <= high+width)
    row, column = np.nonzero(near)
    q, chi = maps.q[near], maps.chi_deg[near]
    cx, cy = index_center_from_canonical(geometry.beam_center_x_px, geometry.beam_center_y_px)
    along, x, y = lab_rays(geometry, (column - cx) * geometry.pixel_size_x_m,
                           (cy - row) * geometry.pixel_size_y_m)
    _, _, sin_exit = exit_directions(x, y, along, geometry.incidence_rad)
    signal = image[near]
    counting = isinstance(counts, np.ndarray) or bool(counts)
    scale = counts[near] if isinstance(counts, np.ndarray) else None
    polar = corrections.polarization
    if polar is None and corrections.orientation_polarization:
        polar = DEFAULT_POLARIZATION
        factor = polarization_factor(along, x, y, polar)
        signal = signal / factor
        if counting:
            scale = (1.0 if scale is None else scale) / factor
    threshold, source = minimum_exit(geometry, corrections.orientation_min_exit_deg)
    nominal = sin_exit >= math.sin(math.radians(min(90.0, threshold)))
    background, method, error = _background(q[nominal], signal[nominal], None if scale is None else scale[nominal],
                                             counting, low, high, step)
    profiles = {}
    ring = (q >= low) & (q <= high) & (np.abs(chi) <= 90)
    error_model = "pixel_scatter" if not counting else "scaled_poisson" if scale is not None else "poisson"
    for multiplier in (0.5, 1.0, 2.0):
        limit = multiplier * threshold
        selected = ring & (sin_exit >= math.sin(math.radians(min(90.0, limit)))) & (limit <= 90.0)
        acc = BinnedMean.linear(-90.0, 90.0, 180)
        acc.add(chi[selected], signal[selected], None if scale is None else scale[selected])
        region = {"q_window": (low, high), "min_exit_deg": limit, "exit_angle_source": source,
                  "min_qz": float(refracted_qz(math.sin(math.radians(min(90.0, limit))), math.sin(geometry.incidence_rad),
                                               geometry.critical_rad, geometry.wavevector_inv_angstrom)),
                  "polarization": polar, "lorentz": False, "mirror_fill": False, "background": background,
                  "background_source": method, "background_sigma": error, "error_model": error_model}
        profiles[multiplier] = Curve("azimuthal", f"I(χ), q = {low:.3f}–{high:.3f} Å⁻¹",
                                     *acc.result(counts=counting), "χ (°)", region=region)
    return profiles, np.ravel_multi_index((row[ring & nominal], column[ring & nominal]), image.shape)
