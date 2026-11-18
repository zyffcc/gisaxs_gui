"""Where a horizontal GISAXS cut is taken, and the scattering paths a sphere's amplitude sums over.

Four modes (Physics and Methods, section 9.10):

* ``eia``: the effective interface approximation, the form factor at q_z = 0; needs nothing about the substrate;
* ``born``: the Born term at the vacuum q_z of the cut, Q = k_i + k_f (fit1d's *Cut qz*);
* ``two_path``: the direct and the exit-reflected path with a free effective reflection coefficient r = |r| e^{i arg r}
  (Fitting's *Spheres in grazing incidence*):  A = f(√(q²+q₊²)R) e^{−iq₊R} + r f(√(q²+q₋²)R) e^{−iq₋R},
  q₊ = k_i + k_f, q₋ = k_i − k_f;
* ``dwba``: the four terms of a particle on a substrate of refractive index 1 − δ + iβ (Renaud, Lazzari & Leroy 2009;
  fit1d's DWBA, ``tools/benchmarks/yoneda_truth``):  A = F(q, k_f+k_i) + r_i F(q, k_f−k_i) + r_f F(q, k_i−k_f)
  + r_i r_f F(q, −(k_f+k_i)), F(q, Q) = f(√(q²+Q²)R) e^{−iQR} for a sphere whose base is on the substrate.

k_{i,f} = (2π/λ) sin α_{i,f}. Upright cylinders are the same in every mode (the Q-dependent factor of a cylinder of
fixed height is common to all its radii, so a cut has the in-plane shape exactly). Random cylinders are
orientation-averaged at |q| = √(q² + Q²) with Q = 0 in ``eia`` and ``two_path`` (as Fitting before 0.3) and
Q = k_i + k_f in ``born`` and ``dwba`` (the Born term, as fit1d).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

MODES = ("eia", "born", "two_path", "dwba")


@dataclass(frozen=True)
class Geometry:
    mode: str = "eia"
    wavelength_nm: float = 0.1
    alpha_i_deg: float = 0.0
    alpha_f_deg: Optional[float] = 0.0
    """``None`` in ``dwba``: the critical angle √(2δ) of the substrate (the Yoneda row)."""
    refl_amp: float = 0.0
    refl_phase: float = 0.0
    delta: float = 0.0
    beta: float = 0.0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Unknown geometry {self.mode!r} (one of {', '.join(MODES)})")
        if self.mode != "eia" and not self.wavelength_nm > 0:
            raise ValueError("A grazing geometry needs a wavelength > 0")
        if self.mode == "dwba" and not self.delta > 0:
            raise ValueError("The DWBA needs the substrate's delta > 0")

    @property
    def k(self) -> float:
        return 2.0 * math.pi / self.wavelength_nm

    @property
    def exit_angle_deg(self) -> float:
        if self.alpha_f_deg is None:
            return math.degrees(math.sqrt(2.0 * self.delta))
        return float(self.alpha_f_deg)

    @property
    def k_i(self) -> float:
        return self.k * math.sin(math.radians(self.alpha_i_deg))

    @property
    def k_f(self) -> float:
        return self.k * math.sin(math.radians(self.exit_angle_deg))

    @property
    def qz_born(self) -> float:
        """The vacuum q_z of the cut, k_i + k_f (0 in ``eia``)."""
        return 0.0 if self.mode == "eia" else self.k_i + self.k_f

    def fresnel(self, alpha_deg: float) -> complex:
        """Reflection coefficient of the bare substrate at the grazing angle ``alpha_deg``."""
        a = math.radians(alpha_deg)
        kz0 = self.k * math.sin(a)
        ks = self.k * complex(math.sin(a) ** 2 - 2.0 * self.delta, 2.0 * self.beta) ** 0.5
        return (kz0 - ks) / (kz0 + ks)

    def sphere_terms(self) -> tuple[tuple[complex, float], ...]:
        """(coefficient, Q) of every path of a sphere's amplitude."""
        if self.mode == "eia":
            return ((1.0 + 0.0j, 0.0),)
        k_i, k_f = self.k_i, self.k_f
        if self.mode == "born":
            return ((1.0 + 0.0j, k_i + k_f),)
        if self.mode == "two_path":
            if self.refl_amp == 0:
                return ((1.0 + 0.0j, k_i + k_f),)  # the Born term alone (and the tables of a single path)
            r = self.refl_amp * complex(math.cos(self.refl_phase), math.sin(self.refl_phase))
            return ((1.0 + 0.0j, k_i + k_f), (r, k_i - k_f))
        r_i, r_f = self.fresnel(self.alpha_i_deg), self.fresnel(self.exit_angle_deg)
        return ((1.0 + 0.0j, k_f + k_i), (r_i, k_f - k_i), (r_f, k_i - k_f), (r_i * r_f, -(k_f + k_i)))

    def random_cylinder_qz(self) -> float:
        return self.qz_born if self.mode in ("born", "dwba") else 0.0

    @classmethod
    def born_at(cls, qz: float) -> "Geometry":
        """The Born term at a given vacuum q_z (any unit of q): λ = 2π (k = 1), α_i = α_f = asin(q_z/2)."""
        if not 0.0 <= qz <= 2.0:
            raise ValueError("born_at needs 0 <= q_z <= 2 k")
        if qz == 0.0:
            return cls()
        angle = math.degrees(math.asin(qz / 2.0))
        return cls("born", 2.0 * math.pi, angle, angle)


def from_angles(alpha_i_deg: float, alpha_f_deg: float, wavelength_nm: float, refl_amp: float = 0.0,
                refl_phase: float = 0.0) -> Geometry:
    """Fitting's grazing globals: the effective interface without angles, else the two paths (Born when |r| = 0)."""
    if alpha_i_deg == 0 and alpha_f_deg == 0 and refl_amp == 0:
        return Geometry()
    return Geometry("two_path", wavelength_nm, alpha_i_deg, alpha_f_deg, refl_amp, refl_phase)


__all__ = ["MODES", "Geometry", "from_angles"]
