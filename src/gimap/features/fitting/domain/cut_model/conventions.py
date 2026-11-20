"""How the conventions of GIMaP's earlier 1D models map onto ``cut_model`` (Physics and Methods, section 9.2).

| Quantity | cut_model | earlier Fitting | fit1d kit | 1D Predict V5 / quick fit |
|---|---|---|---|---|
| q | nm⁻¹ | nm⁻¹ (Å⁻¹ × 10) | nm⁻¹ | nm⁻¹ |
| sizes | ``law``: number_gauss, lognormal, intensity_gauss | intensity Gaussian (sphere, random cylinder); number Gaussian ±3σ (vertical cylinder) | number Gaussian | intensity Gaussian (all three) |
| spacing disorder | σ_nn/D (Hosemann) | σ_D/D = (σ_nn/D)/√(2π) | w = σ_nn (nm) | σ_D/D as Fitting |
| vertical cylinder scale | T(0) = 1 | 10⁻⁶ ⟨R⁴⟩ₙ / 4 (R in nm) | T(0) = 1 | T(0) = 1 |
| low-q term | A / (1 + (q/w)^ν) | the same | C / (1 + (q/w)^ν), ν = 2 by default | ρ_res · max(p[:5]) / max(g[:5]) · g |
| background | BG | BG | B | ρ_BG · median(p) |

Only these functions convert; nothing else in GIMaP converts these quantities.
"""

from __future__ import annotations

import math

import numpy as np

from . import sizes

SQRT_2PI = math.sqrt(2.0 * math.pi)
ANGSTROM_TO_NM = 10.0
"""q in Å⁻¹ × 10 = q in nm⁻¹."""
LEGACY_VC_NODES, LEGACY_VC_SPAN = 26, 3.0


def sigma_nn_from_sigma_d(sigma_d_over_d):
    """σ_nn/D from Fitting's (and V5's) σ_D/D: their paracrystal damps with exp(−π q² σ_D²), Hosemann's with
    exp(−q² σ_nn²/2), so σ_nn = √(2π) σ_D."""
    return SQRT_2PI * np.asarray(sigma_d_over_d, float)


def sigma_d_from_sigma_nn(sigma_nn_over_d):
    return np.asarray(sigma_nn_over_d, float) / SQRT_2PI


def fit1d_w(distance, sigma_nn_over_d):
    """fit1d's w (nm), the absolute standard deviation of the neighbour distance."""
    return np.asarray(distance, float) * np.asarray(sigma_nn_over_d, float)


def legacy_vertical_cylinder_scale(radius: float, spread: float) -> float:
    """What Fitting's vertical-cylinder form factor in earlier versions was at q = 0, 10⁻⁶ ⟨R⁴⟩/4, on its own grid (26 equally
    spaced radii over ±3σ, number weights). A legacy scale Int times this is the forward intensity c of
    ``cut_model``: c = Int · 10⁻⁶ ⟨R⁴⟩/4."""
    if spread <= 0:
        return 1e-6 * radius**4 / 4.0
    sigma = radius * spread
    x = np.linspace(max(0.0, radius - LEGACY_VC_SPAN * sigma), radius + LEGACY_VC_SPAN * sigma, LEGACY_VC_NODES)
    w = np.exp(-0.5 * ((x - radius) / sigma) ** 2)
    return float(1e-6 * np.sum(w * x**4) / w.sum() / 4.0)


def number_from_intensity(radius: float, spread: float, power: float) -> tuple[float, float] | None:
    """Mean and relative spread by number of an intensity Gaussian (``None`` when not defined, sizes.TAIL_SHARE)."""
    return sizes.size_report("intensity_gauss", radius, spread, power)["number"]


def intensity_from_number(radius: float, spread: float, power: float, law: str = "number_gauss") -> tuple[float, float]:
    """Mean and relative spread by V² (∝ R^power) of a number distribution."""
    return sizes.size_report(law, radius, spread, power)["intensity"]


def v5_background(rho_bg: float, particle_curve, scale: float = 1.0) -> float:
    """V5's background ρ_BG · median(p) in the units of the curve the particles were scaled to (``scale``)."""
    return float(rho_bg * np.median(np.asarray(particle_curve, float)) * scale)


def v5_tail_amplitude(rho_res: float, particle_curve, tail_curve, scale: float = 1.0) -> float:
    """V5's beam-tail amplitude ρ_res · max(p[:5]) / max(g[:5]), as A of A/(1+(q/w)^ν)."""
    p, g = np.asarray(particle_curve, float), np.asarray(tail_curve, float)
    return float(rho_res * p[:5].max() / max(g[:5].max(), 1e-30) * scale)


__all__ = ["ANGSTROM_TO_NM", "SQRT_2PI", "fit1d_w", "intensity_from_number", "legacy_vertical_cylinder_scale",
           "number_from_intensity", "sigma_d_from_sigma_nn", "sigma_nn_from_sigma_d", "v5_background",
           "v5_tail_amplitude"]
