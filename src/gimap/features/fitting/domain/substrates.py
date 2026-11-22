"""Substrates of known refractive index, for the four-term DWBA of particles on a surface (Physics and Methods 9.17).

The refractive index n = 1 − δ + iβ of a material of density ρ and mass fractions w_j of elements of atomic weight
A_j and atomic scattering factors f₁ + i f₂ (forward scattering):

    δ + iβ = (r_e λ² / 2π) Σ_j n_j (f₁,j + i f₂,j),    n_j = ρ N_A w_j / A_j

f₁ and f₂ are those of Henke, Gullikson & Davis (1993, At. Data Nucl. Data Tables 54, 181) as tabulated by CXRO, every
fourth point of their tables from 4.9 to 30 keV (and every point between the L edges of gold); between the points f₁
is interpolated linearly in ln E and f₂ linearly in ln f₂ against ln E (within 0.15 % of f₁ and 0.35 % of f₂ of the
full tables; the tables themselves are good to a few per cent near an absorption edge). Outside 4.9–30 keV the table
refuses: δ and β are then typed. Materials (``MATERIALS``): Si (2.33 g/cm³), SiO₂ (2.20 g/cm³, thermal oxide or fused
silica), plate glass (NIST's composition, 2.40 g/cm³) and Au (19.32 g/cm³). A rough or oxidised surface, a polymer film
on the wafer and the particle layer's own index are not part of it (Physics and Methods 9.10, 15.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Optional

import numpy as np

R_E_NM = 2.8179403262e-6
"""Classical electron radius (nm)."""
AVOGADRO = 6.02214076e23
HC_KEV_NM = 1.239841984
ENERGY_RANGE_KEV = (4.8946, 30.0)

ATOMIC_WEIGHT = {"O": 15.999, "Na": 22.98977, "Si": 28.0855, "Ca": 40.078, "Au": 196.96657}

_TABLES = {  # E (keV), f1, f2
    "O": """
    4.8946 8.11427 0.0953398  5.21903 8.104 0.0835779  5.56497 8.09445 0.0732117  5.93383 8.08562 0.0640857
    6.32715 8.07746 0.0560545  6.74654 8.06994 0.0489916  7.19372 8.06304 0.0427875  7.67054 8.0567 0.0373402
    8.17898 8.05091 0.0325612  8.72111 8.04561 0.0283707  9.29917 8.04077 0.0247005  9.91555 8.03637 0.0214883
    10.5728 8.03236 0.0186788  11.2736 8.02872 0.016224  12.0208 8.02541 0.0140812  12.8176 8.02242 0.0122124
    13.6672 8.01971 0.0105844  14.5731 8.01726 0.00916766  15.5391 8.01504 0.00793628  16.5691 8.01304 0.00686735
    17.6674 8.01124 0.00594073  18.8384 8.00962 0.00513869  20.0871 8.00816 0.00444652  21.4185 8.00686 0.00384804
    22.8382 8.00569 0.003331  24.352 8.00464 0.00288407  25.9662 8.0037 0.00249755  27.6873 8.00286 0.00216311
    29.5225 8.00212 0.00187363  30 8.00194 0.00180754""",
    "Na": """
    4.8946 11.2677 0.32691  5.21903 11.2481 0.288568  5.56497 11.2292 0.254539  5.93383 11.2111 0.224381
    6.32715 11.1938 0.19767  6.74654 11.1776 0.174028  7.19372 11.1622 0.15312  7.67054 11.1479 0.134644
    8.17898 11.1345 0.11833  8.72111 11.1221 0.103929  9.29917 11.1105 0.0912268  9.91555 11.0999 0.0800322
    10.5728 11.09 0.0701711  11.2736 11.081 0.0614898  12.0208 11.0726 0.0538517  12.8176 11.065 0.0471356
    13.6672 11.058 0.0412335  14.5731 11.0516 0.0360501  15.5391 11.0458 0.0315006  16.5691 11.0405 0.0275099
    17.6674 11.0357 0.0240116  18.8384 11.0313 0.020947  20.0871 11.0273 0.0182679  21.4185 11.0237 0.0159221
    22.8382 11.0205 0.0138729  24.352 11.0175 0.012083  25.9662 11.0149 0.0105198  27.6873 11.0125 0.00915492
    29.5225 11.0103 0.00796342  30 11.0098 0.00769003""",
    "Si": """
    4.90557 14.3972 0.823285  5.23073 14.3839 0.731621  5.57744 14.3678 0.649655  5.94713 14.3497 0.576388
    6.34133 14.3303 0.511039  6.74917 14.3109 0.454332  7.19653 14.2907 0.402264  7.67354 14.2706 0.35594
    8.05194 14.2558 0.324599  8.44899 14.2414 0.295924  9.00902 14.2227 0.26146  9.60617 14.2047 0.230876
    10.2234 14.1882 0.204512  10.9011 14.1721 0.18039  11.6236 14.1569 0.159022  12.3941 14.1427 0.140104
    13.2156 14.1294 0.123362  14.0916 14.117 0.108553  15.0256 14.1055 0.0954592  16.0216 14.0949 0.0838856
    17.0836 14.0851 0.0736596  18.2159 14.076 0.0646272  19.4233 14.0677 0.0566514  20.7108 14.06 0.0496136
    22.0835 14.053 0.0434289  23.5473 14.0465 0.0379973  25.1081 14.0406 0.0332288  26.7724 14.0352 0.029044
    28.5469 14.0303 0.025373  30 14.0269 0.0228461""",
    "Ca": """
    4.8946 19.3243 3.06169  5.21903 19.668 2.7571  5.56497 19.9052 2.47878  5.93383 20.0726 2.22539
    6.32715 20.1908 1.99527  6.74654 20.2734 1.78686  7.19372 20.3293 1.5985  7.67054 20.3651 1.42857
    8.17898 20.3857 1.27556  8.72111 20.3948 1.13797  9.29917 20.3951 1.01443  9.91555 20.3888 0.90364
    10.5728 20.3778 0.804344  11.2736 20.3633 0.715455  12.0208 20.3463 0.635945  12.8176 20.3278 0.564874
    13.6672 20.3083 0.501383  14.5731 20.2884 0.444694  15.5391 20.2684 0.394101  16.5691 20.2487 0.348963
    17.6674 20.2294 0.308705  18.8384 20.2107 0.272806  20.0871 20.1926 0.240866  21.4185 20.1754 0.212394
    22.8382 20.159 0.187139  24.352 20.1434 0.164761  25.9662 20.1288 0.144951  27.6873 20.115 0.127429
    29.5225 20.1026 0.111945  30 20.0981 0.108365""",
    "Au": """
    4.8946 74.4302 16.5884  5.21903 74.7789 15.1636  5.56497 75.1809 13.844  5.93383 75.3935 12.5542
    6.32715 75.491 11.3838  6.74654 75.4719 10.2725  7.19372 75.3633 9.26191  7.67054 75.1756 8.34582
    8.04879 74.9851 7.71688  8.44569 74.7514 7.13442  9.0055 74.3618 6.42578  9.60242 73.858 5.78882
    10.2389 73.1719 5.2172  10.9176 72.1127 4.70552  11.0942 71.7263 4.58628  11.2736 71.2436 4.47034
    11.4559 70.5981 4.35762  11.6412 69.6151 4.24804  11.8295 67.4461 4.14153  11.9186 54.2731 4.09265
    11.9188 54.2736 10.2075  12.0208 67.8226 10.0588  12.2153 69.8923 9.78487  12.4128 70.816 9.51839
    12.6136 71.3557 9.25916  12.8176 71.6745 9.00701  13.025 71.8212 8.76171  13.2356 71.7884 8.52311
    13.4497 71.4804 8.29101  13.6672 70.2234 8.06522  13.7335 63.7471 7.99837  13.7337 63.7474 11.1563
    13.8883 71.2206 10.9501  14.1129 72.1012 10.6611  14.3412 71.3028 10.3797  14.3527 69.3284 10.3658
    14.3529 69.3299 11.9146  14.5731 73.244 11.651  14.8089 74.0231 11.3795  15.0484 74.5877 11.1126
    16.0458 76.0552 10.0905  17.1094 76.9499 9.14202  18.2435 77.5519 8.2666  19.4527 77.967 7.46228
    20.7421 78.2514 6.72616  22.117 78.4403 6.05473  23.583 78.5576 5.4441  25.1462 78.6204 4.89018
    26.8129 78.6418 4.38882  28.5902 78.6317 3.93592  30 78.5768 3.62565""",
}


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
CUSTOM = "custom"
"""A substrate given by δ and β (at the cut's wavelength)."""


@lru_cache(maxsize=None)
def _table(element: str) -> np.ndarray:
    numbers = np.array(_TABLES[element].split(), float).reshape(-1, 3)
    numbers.setflags(write=False)
    return numbers


def scattering_factors(element: str, energy_kev: float) -> tuple[float, float]:
    """(f₁, f₂) of an element of the table at ``energy_kev`` (module docstring)."""
    lo, hi = ENERGY_RANGE_KEV
    if not lo <= energy_kev <= hi:
        raise ValueError(f"The table of substrates covers {lo:.1f}–{hi:.0f} keV (here {energy_kev:.3g} keV): "
                         "type δ and β of the substrate")
    table = _table(element)
    log_e = np.log(table[:, 0])
    x = math.log(energy_kev)
    f1 = float(np.interp(x, log_e, table[:, 1]))
    f2 = float(np.exp(np.interp(x, log_e, np.log(table[:, 2]))))
    return f1, f2


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


@dataclass(frozen=True)
class Substrate:
    """The substrate under the particles: a material of the table, or δ and β typed (``CUSTOM``; at the cut's λ)."""
    material: str = CUSTOM
    delta: float = 0.0
    beta: float = 0.0

    def __post_init__(self):
        if self.material != CUSTOM and self.material not in MATERIALS:
            raise ValueError(f"Unknown substrate {self.material!r}")
        if self.material == CUSTOM and not (self.delta > 0 and self.beta >= 0):
            raise ValueError("A substrate needs δ > 0 and β ≥ 0")

    def optical(self, wavelength_nm: float) -> tuple[float, float]:
        """(δ, β) at ``wavelength_nm``."""
        if self.material == CUSTOM:
            return float(self.delta), float(self.beta)
        return optical_constants(self.material, wavelength_nm)

    def critical_angle_deg(self, wavelength_nm: float) -> float:
        """√(2δ) in degrees: the exit angle of the Yoneda row."""
        return math.degrees(math.sqrt(2.0 * self.optical(wavelength_nm)[0]))

    @property
    def label(self) -> str:
        return MATERIALS[self.material].label if self.material != CUSTOM else f"δ = {self.delta:.4g}, β = {self.beta:.3g}"

    def to_dict(self) -> dict:
        return {"material": self.material} if self.material != CUSTOM else {
            "material": CUSTOM, "delta": float(self.delta), "beta": float(self.beta)}

    @classmethod
    def from_dict(cls, raw: Optional[Mapping]) -> Optional["Substrate"]:
        if not raw:
            return None
        material = str(raw.get("material") or CUSTOM)
        if material == CUSTOM:
            return cls(CUSTOM, float(raw.get("delta", 0.0)), float(raw.get("beta", 0.0)))
        return cls(material)


__all__ = ["CUSTOM", "ENERGY_RANGE_KEV", "MATERIALS", "Material", "Substrate", "optical_constants",
           "scattering_factors"]
