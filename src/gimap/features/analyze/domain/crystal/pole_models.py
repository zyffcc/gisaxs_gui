"""Orientation distributions of a fibre texture for the pole figure of one ring.

χ is the polar angle of the plane normal from the surface normal (0 out of plane, 90° in the plane), folded to
[0°, 90°]: a plane normal and its opposite are the same plane, and the texture is isotropic about the surface
normal. ρ(χ) is the pole density per unit solid angle, normalised over the hemisphere,

    ∫₀^{π/2} ρ(χ) 2π sin χ dχ = 1,

and N(χ) = 2π ρ(χ) sin χ is the fraction of crystallites per radian of χ.
Each model is an oriented part plus an isotropic fraction φ:

    ρ(χ) = (1 − φ) g(χ) + φ / 2π

* ``gaussian``: g ∝ Σ_c exp(−(χ − c)² / 2s²) over c = χ₀, −χ₀ and 180° − χ₀ (the images that fold onto
  [0°, 90°]), s = FWHM / 2√(2 ln 2): a Gaussian in polar angle about a cone at χ₀;
* ``gaussian_mosaic``: the azimuthal average of antipodal spherical Gaussian lobes about a director at χ₀;
* ``maier_saupe``: the azimuthal average of exp(m cos²γ), where γ is the angle to that director.

1 − φ is then the oriented fraction of the crystallites, χ₀ their direction, the FWHM their mosaicity, and the
Hermans factor of the whole distribution

    ⟨cos²χ⟩ = ∫ ρ cos²χ sin χ dχ / ∫ ρ sin χ dχ,    f = (3⟨cos²χ⟩ − 1) / 2

includes the missing wedge as the model fills it (the isotropic part adds f = 0, so f = (1 − φ) f_oriented).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import dawsn, erf, roots_legendre

GAUSSIAN, GAUSSIAN_MOSAIC, MAIER_SAUPE = "gaussian", "gaussian_mosaic", "maier_saupe"
MODELS = (GAUSSIAN, GAUSSIAN_MOSAIC, MAIER_SAUPE)
SIGMA_PER_FWHM = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
GRID = np.radians(np.linspace(0.0, 90.0, 3601))
"""Fine χ grid (0.025°) for the normalisation and the moments."""
_integrate = getattr(np, "trapezoid", None) or np.trapz


@lru_cache(maxsize=64)
def _azimuth_nodes(count):
    # Reflection about π gives the full fibre average from the interval [0, π].
    x, w = roots_legendre(count)
    return np.cos(0.5 * math.pi * (1 + x)), w / 2


def _axial_shape(dot, model, width):
    if model == MAIER_SAUPE:
        return np.exp(width * dot**2 - max(width, 0.))
    gamma = np.arccos(np.clip(dot, -1., 1.))
    sigma = math.radians(width) * SIGMA_PER_FWHM
    return np.exp(-0.5 * (gamma / sigma)**2) + np.exp(-0.5 * ((math.pi - gamma) / sigma)**2)


def _narrow_average(chi, centre, model, width, min_dot):
    """Integrate the two lobes on the ψ intervals where their scaled density exceeds exp(−40).

    On [0, π], u=a+b cos ψ is monotone. Reflecting the antipodal lobe about π
    puts its maximum at ψ=0 too. Scaling each interval to [0,1] resolves even
    m=60000 with the same 48 nodes as a moderate width.
    """
    a, b = np.cos(chi) * math.cos(centre), np.abs(np.sin(chi) * math.sin(centre))
    total = np.zeros(chi.size)
    nodes = (_MOMENT_X + 1.) / 2.
    for parallel in (a.ravel(), -a.ravel()):
        active = np.flatnonzero(parallel + b.ravel() >= min_dot)
        across = b.ravel()[active]
        lower_cos = np.divide(min_dot - parallel[active], across, where=across > 0., out=np.full(active.size, -1.))
        upper = np.arccos(np.clip(lower_cos, -1., 1.))
        dot = parallel[active, None] + across[:, None] * np.cos(upper[:, None] * nodes)
        if model == MAIER_SAUPE:
            density = np.exp(width * (dot**2 - 1.))
        else:
            sigma = math.radians(width) * SIGMA_PER_FWHM
            density = np.exp(-0.5 * (np.arccos(np.clip(dot, -1., 1.)) / sigma)**2)
        total[active] += upper * np.sum(density * _MOMENT_W, axis=1) / (2. * math.pi)
    return total.reshape(chi.shape)


def oriented_shape(chi_rad, model: str, width: float, chi0_deg: float) -> np.ndarray:
    """g(χ) before normalisation; ``width`` is the FWHM in degrees (Gaussian) or m (Maier–Saupe)."""
    chi = np.asarray(chi_rad, dtype=float)
    centre = math.radians(chi0_deg)
    if model == GAUSSIAN:
        sigma = max(math.radians(width) * SIGMA_PER_FWHM, 1e-6)
        return sum(np.exp(-0.5 * ((chi - c) / sigma) ** 2) for c in (centre, -centre, math.pi - centre))
    if model not in (MAIER_SAUPE, GAUSSIAN_MOSAIC):
        raise ValueError(f"Unknown orientation model {model!r}.")
    if chi0_deg == 0.:
        return _axial_shape(np.cos(chi), model, width)
    if model == MAIER_SAUPE and width > 40.:
        return _narrow_average(chi, centre, model, width, math.sqrt(1. - 40. / width))
    if model == GAUSSIAN_MOSAIC:
        limit = math.sqrt(80.) * math.radians(width) * SIGMA_PER_FWHM
        if limit < math.pi / 2.:
            return _narrow_average(chi, centre, model, width, math.cos(limit))
    azimuth, weights = _azimuth_nodes(96)
    dot = (np.cos(chi)[..., None] * math.cos(centre)
           + np.sin(chi)[..., None] * math.sin(centre) * azimuth)
    return np.sum(_axial_shape(dot, model, width) * weights, axis=-1)


def _normalisation(model: str, width: float, chi0_deg: float) -> float:
    # The integral of an antipodally symmetric density over the hemisphere is
    # invariant under rotation of its director. Integrate in director coordinates.
    if model == MAIER_SAUPE:
        if width == 0.:
            return 2. * math.pi
        root = math.sqrt(abs(width))
        integral = dawsn(root) / root if width > 0 else math.sqrt(math.pi) * erf(root) / (2. * root)
        return 2. * math.pi * integral
    if model == GAUSSIAN_MOSAIC:
        return 2. * math.pi * _mosaic_integral(width)
    shape = oriented_shape(GRID, model, width, chi0_deg)
    return float(_integrate(shape * 2.0 * math.pi * np.sin(GRID), GRID))


def _mosaic_integral(width, moment=False):
    # Integrating one Gaussian over the full sphere accounts for both antipodes
    # on the hemisphere. Scale by s so a narrow mosaic is fully resolved.
    sigma = math.radians(width) * SIGMA_PER_FWHM
    upper = min(12., math.pi / sigma)
    z = (_MOMENT_X + 1.) * (upper / 2.)
    theta = sigma * z
    factor = 0.5 * (3. * np.cos(theta)**2 - 1.) if moment else 1.
    return float(0.5 * upper * sigma * np.sum(_MOMENT_W * np.exp(-0.5 * z**2) * np.sin(theta) * factor))


_MOMENT_X, _MOMENT_W = roots_legendre(48)


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
    if model in (GAUSSIAN_MOSAIC, MAIER_SAUPE):
        # Addition theorem for P₂: averaging the director azimuth leaves P₂(cos χ₀).
        if model == GAUSSIAN_MOSAIC:
            aligned = _mosaic_integral(width, moment=True) / _mosaic_integral(width)
        elif abs(width) < 1e-4:
            aligned = 2 * width / 15. + 4 * width**2 / 315.
        else:
            integral = _normalisation(model, width, 0.) / (2. * math.pi)
            cos2 = (math.exp(min(width, 0.)) / integral - 1.) / (2. * width)
            aligned = 0.5 * (3. * cos2 - 1.)
        return (1. - fraction_iso) * aligned * (3. * math.cos(math.radians(chi0_deg))**2 - 1.) / 2.
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
    if model in (GAUSSIAN, GAUSSIAN_MOSAIC):
        return float(width)
    m = float(width)
    if m > math.log(2.0):  # exp(m cos²δ) falls to half at cos²δ = 1 − ln 2 / m
        return 2.0 * math.degrees(math.acos(math.sqrt(1.0 - math.log(2.0) / m)))
    if m < -math.log(2.0):  # maximum at δ = 90°; half at cos²δ = ln 2 / |m|
        return 2.0 * (90.0 - math.degrees(math.acos(math.sqrt(math.log(2.0) / -m))))
    return math.nan


__all__ = [
    "GAUSSIAN", "GAUSSIAN_MOSAIC", "MAIER_SAUPE", "MODELS", "binned", "crystallites_per_radian", "fwhm_deg", "hermans",
    "order_2d", "oriented_shape", "pole_density",
]
