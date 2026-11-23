"""Detector pixel means for ring orientation, with the same intensity units for the radial background."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from src.gimap.shared.geometry.conventions import index_center_from_canonical
from src.gimap.shared.geometry.q_mapping import exit_directions, lab_rays, refracted_qz
from src.gimap.shared.ring_profile import HORIZON_FRACTION, fold_ring

from .binning import BinnedMean
from .curve_errors import pixel_errors
from .intensity import polarization_factor
from .models import Curve
from .peak_fit import GAUSSIAN, fit_peak
from .refracted_film import internal_qz

DEFAULT_POLARIZATION = 0.99
BACKGROUND_PEAK = "local linear background of a Gaussian fit to I(q)"
BACKGROUND_FLANKS = "mean of I(q) beside the window"


def minimum_exit(geometry, override):
    if override is not None:
        return float(override), "user setting"
    if geometry.refraction:
        return 2 * geometry.critical_angle_deg, "2 × αc"
    return 2 * geometry.incidence_deg, "2 × αi"


def _background(q, intensity, errors, low, high, step):
    half = 0.5 * (high - low)
    left, right = low - 2*half, high + 2*half
    bins = min(3000, max(1, round((right-left) / max(step, 1e-10))))
    radial = BinnedMean.linear(left, right, bins)
    radial.add(q, intensity, errors=errors)
    x, y, sigma, _ = radial.result(counts=errors)
    fit = fit_peak(x, y, sigma, (left, right), profile=GAUSSIAN)
    if fit.ok and low <= fit.center <= high and np.isfinite(fit.background_center):
        error = fit.background_center_err
        return fit.background_center, BACKGROUND_PEAK, error if np.isfinite(error) else None
    beside = ((x >= low-half) & (x < low)) | ((x > high) & (x <= high+half))
    if beside.any():
        return float(y[beside].mean()), BACKGROUND_FLANKS, None
    return 0.0, "none", None


def ring_profiles(maps, image, valid, counts, geometry, window, step, corrections):
    profiles, indices, _, _ = measured_ring_profiles(maps, image, valid, counts, geometry, window, step, corrections)
    return profiles, indices


def measured_ring_profiles(maps, image, valid, counts, geometry, window, step, corrections, *, film_index=None):
    """Reduce actual pixels inside the ring at each exit-angle threshold. Image excludes L and mirror fill."""
    low, high = sorted(window)
    width = high - low
    near = valid & maps.above_horizon & (maps.q >= low-width) & (maps.q <= high+width)
    row, column = np.nonzero(near)
    q, chi = maps.q[near], maps.chi_deg[near]
    cx, cy = index_center_from_canonical(geometry.beam_center_x_px, geometry.beam_center_y_px)
    along, x, y = lab_rays(geometry, (column - cx) * geometry.pixel_size_x_m,
                           (cy - row) * geometry.pixel_size_y_m)
    _, _, sin_exit = exit_directions(x, y, along, geometry.incidence_rad, sample_roll_rad=geometry.sample_roll_rad)
    if film_index is not None:
        chi = np.degrees(np.arctan2(maps.q_parallel[near], internal_qz(geometry, sin_exit, *film_index)))
    signal = image[near]
    errors = pixel_errors(counts).subset(near)
    polar = corrections.polarization
    if polar is None and corrections.orientation_polarization:
        polar = DEFAULT_POLARIZATION
        factor = polarization_factor(along, x, y, polar)
        signal = signal / factor
        if errors.model != "scatter":
            scale = errors.scale
            if errors.model == "poisson":
                scale = (1.0 if scale is None else scale) / factor
            errors = replace(errors, scale=scale,
                             variance=None if errors.variance is None else errors.variance / factor**2,
                             extra_variance=None if errors.extra_variance is None else errors.extra_variance / factor**2)
    threshold, source = minimum_exit(geometry, corrections.orientation_min_exit_deg)
    nominal = sin_exit >= math.sin(math.radians(min(90.0, threshold)))
    background, method, error = _background(q[nominal], signal[nominal], errors.subset(nominal), low, high, step)
    profiles = {}
    ring = (q >= low) & (q <= high) & (np.abs(chi) <= 90)
    propagated = errors.scale is not None or errors.extra_variance is not None
    error_model = {"sigma": "sigma_frame", "scatter": "pixel_scatter",
                   "poisson": "scaled_poisson" if propagated else "poisson"}[errors.model]
    for multiplier in (0.5, 1.0, 2.0):
        limit = multiplier * threshold
        selected = ring & (sin_exit >= math.sin(math.radians(min(90.0, limit)))) & (limit <= 90.0)
        acc = BinnedMean.linear(-90.0, 90.0, 180)
        acc.add(chi[selected], signal[selected], errors=errors.subset(selected))
        azimuth, means, sigma, pixels = acc.result(counts=errors)
        min_qz = float(refracted_qz(math.sin(math.radians(min(90.0, limit))), math.sin(geometry.incidence_rad),
                                    geometry.critical_rad, geometry.wavevector_inv_angstrom))
        # At the outer edge of the q window, cos χ = qz / q gives the largest accessible |χ|.
        horizon_chi = math.degrees(math.acos(np.clip(min_qz / high, 0., 1.))) if high > 0 else None
        if film_index is not None:
            parallel = math.sqrt(max(0., high**2 - min_qz**2))
            min_qz = float(internal_qz(geometry, math.sin(math.radians(min(90., limit))), *film_index))
            horizon_chi = math.degrees(math.atan2(parallel, min_qz))
        *_, horizon = fold_ring(azimuth, means, sigma, background, horizon_chi_deg=horizon_chi)
        region = {"q_window": (low, high), "min_exit_deg": limit, "exit_angle_source": source,
                  "min_qz": min_qz, "horizon_chi_deg": horizon_chi, "horizon_excluded_deg": horizon,
                  "horizon_fraction": HORIZON_FRACTION,
                  "polarization": polar, "lorentz": False, "mirror_fill": False, "background": background,
                  "background_source": method, "background_sigma": error, "error_model": error_model}
        region["pole_angle_coordinate"] = "inside film" if film_index is not None else "active q map"
        profiles[multiplier] = Curve("azimuthal", f"I(χ), q = {low:.3f}–{high:.3f} Å⁻¹",
                                     azimuth, means, sigma, pixels, "χ (°)", region=region)
    selected = ring & nominal
    indices = np.ravel_multi_index((row[selected], column[selected]), image.shape)
    return profiles, indices, sin_exit[selected], chi[selected]
