"""Orientation distributions of a fibre texture for the pole figure of one ring.

χ is the polar angle of the plane normal from the surface normal (0 out of plane, 90° in the plane), folded to
[0°, 90°]: a plane normal and its opposite are the same plane, and the texture is isotropic about the surface
normal. ρ(χ) is the pole density per unit solid angle, normalised over the hemisphere,

    ∫₀^{π/2} ρ(χ) 2π sin χ dχ = 1,

and N(χ) = 2π ρ(χ) sin χ is the fraction of crystallites per radian of χ (what a Lorentz-corrected I(χ) is
proportional to, Physics and Methods 5.6). Each model is an oriented part plus an isotropic fraction φ:

    ρ(χ) = (1 − φ) g(χ) + φ / 2π

* ``gaussian``: g ∝ Σ_c exp(−(χ − c)² / 2s²) over c = χ₀, −χ₀ and 180° − χ₀ (the images that fold onto
  [0°, 90°]), s = FWHM / 2√(2 ln 2): crystallites tilted by χ₀ from the normal with a Gaussian mosaic spread;
* ``maier_saupe``: g ∝ exp(m cos²(χ − χ₀)) + exp(m cos²(χ + χ₀)), the Maier–Saupe form SciAnalysis fits to I(χ)
  (m > 0: about χ₀; m < 0: a girdle 90° away).

1 − φ is then the oriented fraction of the crystallites, χ₀ their direction, the FWHM their mosaicity, and the
Hermans factor of the whole distribution

    ⟨cos²χ⟩ = ∫ ρ cos²χ sin χ dχ / ∫ ρ sin χ dχ,    f = (3⟨cos²χ⟩ − 1) / 2

includes the missing wedge as the model fills it (the isotropic part adds f = 0, so f = (1 − φ) f_oriented).
"""

from __future__ import annotations

import math

import numpy as np

GAUSSIAN, MAIER_SAUPE = "gaussian", "maier_saupe"
MODELS = (GAUSSIAN, MAIER_SAUPE)
SIGMA_PER_FWHM = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
GRID = np.radians(np.linspace(0.0, 90.0, 3601))
"""Fine χ grid (0.025°) for the normalisation and the moments."""
_integrate = getattr(np, "trapezoid", None) or np.trapz


def oriented_shape(chi_rad, model: str, width: float, chi0_deg: float) -> np.ndarray:
    """g(χ) before normalisation; ``width`` is the FWHM in degrees (Gaussian) or m (Maier–Saupe)."""
    chi = np.asarray(chi_rad, dtype=float)
    centre = math.radians(chi0_deg)
    if model == GAUSSIAN:
        sigma = max(math.radians(width) * SIGMA_PER_FWHM, 1e-6)
        return sum(np.exp(-0.5 * ((chi - c) / sigma) ** 2) for c in (centre, -centre, math.pi - centre))
    if model == MAIER_SAUPE:
        m = float(width)
        offset = max(m, 0.0)  # exp(m cos²) ≤ exp(max(m, 0)): no overflow
        return np.exp(m * np.cos(chi - centre) ** 2 - offset) + np.exp(m * np.cos(chi + centre) ** 2 - offset)
    raise ValueError(f"Unknown orientation model {model!r}.")


def _normalisation(model: str, width: float, chi0_deg: float) -> float:
    shape = oriented_shape(GRID, model, width, chi0_deg)
    return float(_integrate(shape * 2.0 * math.pi * np.sin(GRID), GRID))


def pole_density(chi_rad, model: str, fraction_iso: float, width: float, chi0_deg: float) -> np.ndarray:
    """ρ(χ) per steradian, normalised over the hemisphere."""
    norm = _normalisation(model, width, chi0_deg)
    oriented = oriented_shape(chi_rad, model, width, chi0_deg) / norm if norm > 0 else 0.0 * np.asarray(chi_rad)
    return (1.0 - fraction_iso) * oriented + fraction_iso / (2.0 * math.pi)


def crystallites_per_radian(chi_rad, model: str, fraction_iso: float, width: float, chi0_deg: float) -> np.ndarray:
    """N(χ) = 2π ρ(χ) sin χ: the fraction of crystallites per radian of χ (integrates to 1 over [0, π/2])."""
    chi = np.asarray(chi_rad, dtype=float)
    return 2.0 * math.pi * pole_density(chi, model, fraction_iso, width, chi0_deg) * np.sin(chi)


def binned(chi_low_deg, chi_high_deg, model: str, fraction_iso: float, width: float, chi0_deg: float,
           samples: int = 12) -> np.ndarray:
    """N(χ) averaged over each bin [low, high) (degrees), as a binned measurement sees it."""
    low = np.asarray(chi_low_deg, dtype=float)
    high = np.asarray(chi_high_deg, dtype=float)
    steps = (np.arange(samples) + 0.5) / samples
    points = np.radians(low[:, None] + (high - low)[:, None] * steps[None, :])
    values = crystallites_per_radian(points.ravel(), model, fraction_iso, width, chi0_deg).reshape(points.shape)
    return values.mean(axis=1)


def hermans(model: str, fraction_iso: float, width: float, chi0_deg: float) -> float:
    """f of the whole distribution, wedge included (see the module docstring)."""
    rho = pole_density(GRID, model, fraction_iso, width, chi0_deg)
    weight = rho * np.sin(GRID)
    mean_cos2 = float(_integrate(weight * np.cos(GRID) ** 2, GRID) / _integrate(weight, GRID))
    return 0.5 * (3.0 * mean_cos2 - 1.0)


def order_2d(model: str, fraction_iso: float, width: float, chi0_deg: float) -> float:
    """2⟨cos²χ⟩ − 1 of the pole density along the ring, without the sin χ weight (SciAnalysis' orientation factor)."""
    rho = pole_density(GRID, model, fraction_iso, width, chi0_deg)
    mean_cos2 = float(_integrate(rho * np.cos(GRID) ** 2, GRID) / _integrate(rho, GRID))
    return 2.0 * mean_cos2 - 1.0


def fwhm_deg(model: str, width: float) -> float:
    """Full width at half maximum (°) of the oriented part about its maximum; NaN when it has no half maximum."""
    if model == GAUSSIAN:
        return float(width)
    m = float(width)
    if m > math.log(2.0):  # exp(m cos²δ) falls to half at cos²δ = 1 − ln 2 / m
        return 2.0 * math.degrees(math.acos(math.sqrt(1.0 - math.log(2.0) / m)))
    if m < -math.log(2.0):  # maximum at δ = 90°; half at cos²δ = ln 2 / |m|
        return 2.0 * (90.0 - math.degrees(math.acos(math.sqrt(math.log(2.0) / -m))))
    return math.nan


__all__ = [
    "GAUSSIAN", "MAIER_SAUPE", "MODELS", "binned", "crystallites_per_radian", "fwhm_deg", "hermans",
    "order_2d", "oriented_shape", "pole_density",
]
