"""X-ray optics of the fringe analysis: wavelength, critical angle from an electron density, the z-wavevector
inside a layer and the height of a layer from the position of an order-n minimum.

Units: angles in degrees, lengths in nm, wavevectors in nm^-1, electron densities in electrons per cubic angstrom.

The vertical form factor of a uniform layer (box profile) of height H is sinc(Q H / 2) with Q the z-component of
the momentum transfer inside the layer. Its minima are at Q_n = 2 pi n / H. With refraction (DWBA-like, inside the
layer of critical angle alpha_c) Q = kz(alpha_i) + kz(alpha_f), kz(a) = k sqrt(sin^2 a - sin^2 alpha_c); below the
critical angle kz is imaginary (evanescent) and its real part, 0, is used. In vacuum (Born) Q = k (sin alpha_i +
sin alpha_f), the usual q_z.
"""

from __future__ import annotations

import math

import numpy as np

HC_EV_NM = 1239.841984
"""h c in eV nm (photon energy to wavelength)."""
ELECTRON_RADIUS_NM = 2.8179403262e-6
ELECTRON_DENSITY = {"Cu": 2.46, "Ag": 2.77, "Au": 4.66}
"""Bulk electron densities of the metals (e/A^3), from Z and the bulk density, without anomalous dispersion."""
HOST_ELECTRON_DENSITY = 0.70
"""Si (e/A^3): an upper bound for a polymer film on Si, the host whose Yoneda region is left out."""


def wavelength_nm(energy_ev: float) -> float:
    return HC_EV_NM / float(energy_ev)


def wavenumber(energy_ev: float) -> float:
    """k = 2 pi / lambda (nm^-1)."""
    return 2.0 * math.pi / wavelength_nm(energy_ev)


def critical_angle_deg(electron_density: float, energy_ev: float) -> float:
    """alpha_c = sqrt(2 delta), delta = r_e lambda^2 rho_e / (2 pi) (rho_e in e/A^3 = 1000 e/nm^3)."""
    lam = wavelength_nm(energy_ev)
    delta = ELECTRON_RADIUS_NM * lam * lam * float(electron_density) * 1000.0 / (2.0 * math.pi)
    return math.degrees(math.sqrt(2.0 * delta))


def bulk_critical_angle_deg(metal: str, energy_ev: float):
    """The bulk critical angle of ``metal`` (None for a metal without a tabulated electron density)."""
    rho = ELECTRON_DENSITY.get(str(metal or ""))
    return None if rho is None else critical_angle_deg(rho, energy_ev)


def kz_inside(alpha_deg, alpha_c_deg, k: float):
    """Real part of the z-wavevector inside a layer of critical angle ``alpha_c_deg`` (0 below it)."""
    s = np.sin(np.radians(alpha_deg)) ** 2 - np.sin(np.radians(alpha_c_deg)) ** 2
    return k * np.sqrt(np.clip(s, 0.0, None))


def q_inside(alpha_f_deg, alpha_i_deg: float, alpha_c_deg: float, k: float):
    """Q = kz(alpha_i) + kz(alpha_f) inside the layer (refraction-corrected q_z)."""
    return kz_inside(alpha_i_deg, alpha_c_deg, k) + kz_inside(alpha_f_deg, alpha_c_deg, k)


def qz_vacuum(alpha_f_deg, alpha_i_deg: float, k: float):
    """q_z = k (sin alpha_i + sin alpha_f) (vacuum, Born)."""
    return k * (np.sin(np.radians(alpha_i_deg)) + np.sin(np.radians(alpha_f_deg)))


def height_nm(order, alpha_f_deg, alpha_i_deg: float, alpha_c_deg: float, k: float):
    """H_n = 2 pi n / Q(alpha_f): the height of a uniform layer whose order-n minimum is at ``alpha_f_deg``."""
    q = q_inside(alpha_f_deg, alpha_i_deg, alpha_c_deg, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(q > 0, 2.0 * math.pi * np.asarray(order, dtype=float) / np.where(q > 0, q, 1.0), np.nan)


def alpha_f_of_q(q, alpha_i_deg: float, alpha_c_deg: float, k: float):
    """The exit angle at which Q (inside the layer) takes the value ``q`` (NaN where it cannot)."""
    kz_f = np.asarray(q, dtype=float) - kz_inside(alpha_i_deg, alpha_c_deg, k)
    s = (kz_f / k) ** 2 + np.sin(np.radians(alpha_c_deg)) ** 2
    with np.errstate(invalid="ignore"):
        return np.where((kz_f >= 0) & (s <= 1), np.degrees(np.arcsin(np.sqrt(np.clip(s, 0, 1)))), np.nan)


__all__ = ["ELECTRON_DENSITY", "HC_EV_NM", "HOST_ELECTRON_DENSITY", "alpha_f_of_q", "bulk_critical_angle_deg",
           "critical_angle_deg", "height_nm", "kz_inside", "q_inside", "qz_vacuum", "wavelength_nm", "wavenumber"]
