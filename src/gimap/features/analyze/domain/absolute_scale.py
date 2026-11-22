"""Absolute intensity from the radial mean of an isotropic secondary standard."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .normalisation import mm_to_cm


# Forward X-ray cross section of liquid water (cm⁻¹), at integer °C, 0 through 100.
# Graz water table reproduced in BioXTAS RAW's RAWSettings.water_scattering_table.
# Orthaber, Bergmann & Glatter (2000), doi:10.1107/S0021889899015216.
_WATER = (
    .01692, .01686, .01680, .01675, .01670, .01665, .01661, .01657, .01653, .01650,
    .01647, .01645, .01642, .01640, .01638, .01637, .01635, .01634, .01633, .01633,
    .01632, .01632, .01632, .01632, .01632, .01633, .01634, .01635, .01636, .01637,
    .01638, .01640, .01641, .01643, .01645, .01647, .01650, .01652, .01655, .01658,
    .01660, .01663, .01666, .01670, .01673, .01677, .01680, .01684, .01688, .01692,
    .01696, .01700, .01704, .01709, .01713, .01718, .01723, .01728, .01732, .01738,
    .01743, .01748, .01753, .01759, .01764, .01770, .01776, .01781, .01787, .01793,
    .01800, .01806, .01812, .01818, .01825, .01831, .01838, .01845, .01852, .01859,
    .01866, .01873, .01880, .01887, .01895, .01902, .01909, .01917, .01925, .01932,
    .01940, .01948, .01956, .01964, .01973, .01981, .01989, .01998, .02006, .02015,
    .02023,
)


def water_intensity(temperature_c: float) -> float:
    """Low-q water plateau at ambient pressure, linearly interpolated in temperature."""
    if not 0.0 <= temperature_c <= 100.0:
        raise ValueError("Water temperature must be between 0 and 100 °C.")
    return float(np.interp(temperature_c, np.arange(len(_WATER)), _WATER))


@dataclass(frozen=True)
class ScaleFit:
    factor: float
    sigma_factor: float
    q: np.ndarray
    scaled: np.ndarray
    reference: np.ndarray
    residual: np.ndarray


def _curve_arrays(q, intensity, sigma):
    q = np.asarray(q, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    sigma = np.zeros_like(q) if sigma is None else np.asarray(sigma, dtype=float)
    if q.ndim != 1 or q.shape != intensity.shape or q.shape != sigma.shape:
        raise ValueError("The q, intensity and uncertainty columns must have the same length.")
    if np.any(sigma < 0):
        raise ValueError("Standard uncertainties must be nonnegative.")
    order = np.argsort(q)
    return q[order], intensity[order], sigma[order]


def _reference_arrays(reference):
    q, y, sigma = _curve_arrays(*reference)
    if (len(q) < 2 or not np.all(np.isfinite((q, y, sigma))) or np.any(q < 0)
            or np.any(np.diff(q) <= 0) or np.any(y < 0)):
        raise ValueError("The reference needs distinct, nonnegative q and finite, nonnegative intensities and uncertainties.")
    return q, y, sigma


def _weighted_slope(measured, target, measured_sigma, reference_variance):
    """Fit measured = b * target using the estimated variance on both axes."""
    if not np.any(target > 0):
        raise ValueError("The standard and reference must give a positive scale.")
    slope = (target @ measured) / (target @ target)
    weights = np.ones_like(measured)
    latent = target
    known_variance = np.any(measured_sigma > 0) or np.any(reference_variance > 0)
    for _ in range(100):
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError("The standard and reference must give a positive scale.")
        if known_variance:
            weights = 1.0 / (measured_sigma**2 + slope**2 * reference_variance)
            # The derivative of the effective-variance χ² contains the latent
            # (noise-free) reference. This also gives the correct inverse fit when
            # only the reference has pointwise uncertainties.
            latent = weights * (measured_sigma**2 * target + slope * reference_variance * measured)
        weighted_latent = weights * latent
        cross = weighted_latent @ target
        if not np.isfinite(cross) or cross <= 0:
            raise ValueError("The standard and reference must give a positive scale.")
        update = (weighted_latent @ measured) / cross
        if abs(update - slope) <= 1e-12 * slope:
            denominator = weighted_latent @ latent
            influence = weighted_latent / denominator
            return update, influence, weights, denominator, known_variance
        slope = update
    raise ValueError("The weighted scale fit did not converge.")


def fit_standard(q, intensity, sigma=None, *, exposure_s: float = 1.0, monitor: float = 1.0,
                 transmission: float = 1.0, thickness_mm: float, reference=None,
                 temperature_c: float = 20.0, q_range=None, reference_fractional_sigma: float = 0.0) -> ScaleFit:
    """Fit an unnormalised, corrected radial mean to water or a (q, cross section, σ) table.

    q is in Å⁻¹. The certificate's pointwise standard uncertainties are independent at
    the supplied q nodes. A separate fractional reference scale uncertainty is common
    to every point. Residuals and the two fitted curves are in cm⁻¹.
    """
    terms = np.asarray([exposure_s, monitor, transmission, thickness_mm], dtype=float)
    if not np.all(np.isfinite(terms) & (terms > 0)) or transmission > 1:
        raise ValueError("Enter positive standard normalisation terms and a transmission of at most 1.")
    if not np.isfinite(reference_fractional_sigma) or reference_fractional_sigma < 0:
        raise ValueError("The common reference uncertainty must be finite and nonnegative.")
    q, measured, measured_sigma = _curve_arrays(q, intensity, sigma)
    divisor = exposure_s * monitor * transmission * mm_to_cm(thickness_mm)
    if not np.isfinite(divisor) or divisor <= 0:
        raise ValueError("The standard normalisation is outside the numerical range.")
    measured = measured / divisor
    measured_sigma = measured_sigma / divisor
    valid = (q >= 0) & np.isfinite(q) & np.isfinite(measured) & np.isfinite(measured_sigma)
    if q_range is not None:
        lo, hi = q_range
        if not (np.isfinite(lo) and np.isfinite(hi) and 0 <= lo < hi):
            raise ValueError("Enter increasing, nonnegative q limits.")
        valid &= (q >= lo) & (q <= hi)
    if reference is None:
        level = water_intensity(temperature_c)
    else:
        nodes, levels, node_sigma = _reference_arrays(reference)
        valid &= (q >= nodes[0]) & (q <= nodes[-1])
    q, measured, measured_sigma = q[valid], measured[valid], measured_sigma[valid]
    if reference is None:
        target = np.full_like(q, level)
        reference_variance = np.zeros_like(q)
    else:
        left = np.searchsorted(nodes, q, side="right").clip(1, len(nodes) - 1) - 1
        right_weight = (q - nodes[left]) / (nodes[left + 1] - nodes[left])
        left_weight = 1.0 - right_weight
        target = left_weight * levels[left] + right_weight * levels[left + 1]
        reference_variance = (left_weight * node_sigma[left])**2 + (right_weight * node_sigma[left + 1])**2

    # If some bins have usable errors, zero-error placeholders do not get infinite weight.
    weighted = (measured_sigma > 0) | (reference_variance > 0)
    if np.any(weighted):
        q, measured, measured_sigma = q[weighted], measured[weighted], measured_sigma[weighted]
        target, reference_variance = target[weighted], reference_variance[weighted]
        if reference is not None:
            left, left_weight, right_weight = left[weighted], left_weight[weighted], right_weight[weighted]
    if len(q) < 2 or q[0] == q[-1]:
        raise ValueError("The fit needs at least two distinct q points in the shared range.")
    slope, influence, weights, denominator, known = _weighted_slope(measured, target, measured_sigma, reference_variance)
    variance = np.sum((influence * measured_sigma)**2)
    if reference is not None:
        # Accumulate sensitivities at original nodes before squaring: interpolants share data.
        node_influence = np.bincount(left, influence * left_weight, minlength=len(nodes))
        node_influence += np.bincount(left + 1, influence * right_weight, minlength=len(nodes))
        variance += np.sum((slope * node_influence * node_sigma)**2)
    reduced_scatter = weights @ (measured - slope * target)**2 / (len(q) - 1)
    variance += max(0.0, reduced_scatter - float(known)) / denominator
    factor = float(1.0 / slope)
    sigma_factor = float(np.hypot(np.sqrt(variance) * factor**2, factor * reference_fractional_sigma))
    if not (np.isfinite(factor) and np.isfinite(sigma_factor)):
        raise ValueError("The fitted scale or uncertainty is outside the numerical range.")
    scaled = factor * measured
    return ScaleFit(factor, sigma_factor, q, scaled, target, scaled - target)
