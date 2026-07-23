from __future__ import annotations

import math

import numpy as np


HC_KEV_ANGSTROM = 12.398419843320026


def energy_to_wavelength(energy_kev: float) -> float:
    energy = float(energy_kev)
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("X-ray energy must be greater than zero.")
    return HC_KEV_ANGSTROM / energy


def q_to_two_theta(q_inv_angstrom: float | np.ndarray, wavelength_angstrom: float) -> np.ndarray:
    q = np.asarray(q_inv_angstrom, dtype=float)
    argument = q * float(wavelength_angstrom) / (4.0 * math.pi)
    if np.any(np.abs(argument) > 1.0):
        raise ValueError("The selected q value is not physically reachable at this energy.")
    return 2.0 * np.arcsin(argument)


def q_to_ring_radius_px(
    q_inv_angstrom: float | np.ndarray,
    energy_kev: float,
    distance_mm: float,
    pixel_size_m: float,
) -> np.ndarray:
    wavelength = energy_to_wavelength(energy_kev)
    angle = q_to_two_theta(q_inv_angstrom, wavelength)
    return (float(distance_mm) * 1e-3 * np.tan(angle)) / float(pixel_size_m)


def q_to_ring_radius_m(
    q_inv_angstrom: float | np.ndarray,
    wavelength_angstrom: float,
    distance_mm: float,
) -> np.ndarray:
    return float(distance_mm) * 1e-3 * np.tan(q_to_two_theta(q_inv_angstrom, wavelength_angstrom))


def distance_from_ring_radius(
    radius_px: float,
    q_inv_angstrom: float,
    wavelength_angstrom: float,
    pixel_size_m: float,
) -> float:
    tangent = math.tan(float(q_to_two_theta(q_inv_angstrom, wavelength_angstrom)))
    if tangent <= 0:
        raise ValueError("A positive q value is required.")
    return float(radius_px) * float(pixel_size_m) / tangent * 1000.0
