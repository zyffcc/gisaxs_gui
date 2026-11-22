"""Tabulated material optical constants for grazing-incidence X-ray scattering (Physics and Methods 9.17).

The refractive index n = 1 − δ + iβ of a material of density ρ and mass fractions w_j of elements of atomic weight
A_j and atomic scattering factors f₁ + i f₂ (forward scattering):

    δ + iβ = (r_e λ² / 2π) Σ_j n_j (f₁,j + i f₂,j),    n_j = ρ N_A w_j / A_j

f₁ and f₂ are those of Henke, Gullikson & Davis (1993, At. Data Nucl. Data Tables 54, 181) as tabulated by CXRO, every
fourth point of their tables from 4.9 to 30 keV (and every point between the L edges of gold); between the points f₁
is interpolated linearly in ln E and f₂ linearly in ln f₂ against ln E (within 0.15 % of f₁ and 0.35 % of f₂ of the
full tables; the tables themselves are good to a few per cent near an absorption edge). Outside 4.9–30 keV the table
refuses: δ and β are then typed. Materials (``MATERIALS``): Si (2.33 g/cm³), SiO₂ (2.20 g/cm³, thermal oxide or fused
silica), plate glass (NIST's composition, 2.40 g/cm³) and Au (19.32 g/cm³). Each entry represents a homogeneous
material. A layered surface needs the thickness and optical constants of each film.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from src.gimap.shared.xray_scattering import ENERGY_RANGE_KEV, HC_KEV_NM, R_E_NM
from src.gimap.shared.xray_scattering import scattering_factors as _forward_factors

AVOGADRO = 6.02214076e23

ATOMIC_WEIGHT = {"O": 15.999, "Na": 22.98977, "Si": 28.0855, "Ca": 40.078, "Au": 196.96657}


@dataclass(frozen=True)
class Material:
    label: str
    density: float
    """g/cm³"""
    mass_fractions: Mapping[str, float]


MATERIALS = {
    "Si": Material("Si", 2.33, {"Si": 1.0}),
    "SiO2": Material("SiO₂", 2.20, {"Si": 0.467435, "O": 0.532565}),
    "glass": Material("Glass (plate)", 2.40, {"O": 0.4598, "Na": 0.096441, "Si": 0.336553, "Ca": 0.107205}),
    "Au": Material("Au", 19.32, {"Au": 1.0}),
}
"""The materials of the table (key: label, density, mass fractions). Plate glass: NIST's composition of “Glass,
plate” (soda-lime)."""



def scattering_factors(element: str, energy_kev: float) -> tuple[float, float]:
    """Forward scattering factors for the tabulated substrates."""
    try:
        return _forward_factors(element, energy_kev)
    except ValueError as error:
        lo, hi = ENERGY_RANGE_KEV
        raise ValueError(f"The table of substrates covers {lo:.1f}–{hi:.0f} keV (here {energy_kev:.3g} keV): "
                         "type δ and β of the substrate") from error


def optical_constants(material: str, wavelength_nm: float) -> tuple[float, float]:
    """(δ, β) of a material of ``MATERIALS`` at ``wavelength_nm``."""
    if material not in MATERIALS:
        raise ValueError(f"Unknown substrate {material!r} (one of {', '.join(MATERIALS)})")
    if not wavelength_nm > 0:
        raise ValueError("The substrate's index needs a wavelength > 0")
    entry = MATERIALS[material]
    energy = HC_KEV_NM / wavelength_nm
    total = 0j
    for element, fraction in entry.mass_fractions.items():
        f1, f2 = scattering_factors(element, energy)
        atoms_per_nm3 = entry.density * AVOGADRO * fraction / ATOMIC_WEIGHT[element] * 1e-21
        total += atoms_per_nm3 * complex(f1, f2)
    factor = R_E_NM * wavelength_nm**2 / (2.0 * math.pi)
    return factor * total.real, factor * total.imag


def critical_angle_deg(material: str, wavelength_nm: float) -> float:
    """Small-angle critical angle √(2δ), in degrees, at the supplied wavelength."""
    delta, _ = optical_constants(material, wavelength_nm)
    return math.degrees(math.sqrt(2.0 * delta))


def normal_wavevector_ratio(sine, delta: float, beta: float):
    """Refracted kz/k for sin(α) in vacuum and n = 1 − δ + iβ, using the decaying root."""
    real = sine * sine - 2 * delta
    if isinstance(sine, np.ndarray):
        return np.sqrt(real + 2j * beta)
    return cmath.sqrt(complex(real, 2 * beta))
