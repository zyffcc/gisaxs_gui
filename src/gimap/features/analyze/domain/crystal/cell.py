"""A unit cell, its reciprocal lattice and the scattering vector of a reflection (q in Å⁻¹).

The Cartesian frame of the cell is the usual one (International Tables Vol. B, 1.1): **a** along x, **b** in the
x–y plane, **c** completing a right-handed set. The columns of ``direct_matrix`` are **a**, **b**, **c** (Å);
``reciprocal_matrix`` is ``2π (A⁻¹)ᵀ``, so the scattering vector of the reflection ``hkl`` is ``q = B · hkl`` and
``d = 2π / |q|``. The orientation of a fibre texture is given in the same frame: the normal of the plane ``(hkl)``
is along ``B · hkl``, the crystal axis ``[uvw]`` along ``A · uvw``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UnitCell:
    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    """Lengths in Å, angles in degrees."""

    def __post_init__(self) -> None:
        for name in ("a", "b", "c"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"The cell length {name} must be positive, got {value!r}.")
        for name in ("alpha", "beta", "gamma"):
            value = float(getattr(self, name))
            if not 0.0 < value < 180.0:
                raise ValueError(f"The cell angle {name} must lie between 0° and 180°, got {value!r}.")
        if not self.volume > 0.0:
            raise ValueError("These cell angles do not make a cell (zero or imaginary volume).")

    @property
    def volume(self) -> float:
        ca, cb, cg = (math.cos(math.radians(angle)) for angle in (self.alpha, self.beta, self.gamma))
        root = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
        return self.a * self.b * self.c * math.sqrt(root) if root > 0.0 else 0.0

    def as_tuple(self) -> tuple[float, float, float, float, float, float]:
        return (self.a, self.b, self.c, self.alpha, self.beta, self.gamma)

    def direct_matrix(self) -> np.ndarray:
        """3 × 3, columns **a**, **b**, **c** in Å."""
        ca, cb, cg = (math.cos(math.radians(angle)) for angle in (self.alpha, self.beta, self.gamma))
        sg = math.sin(math.radians(self.gamma))
        cy = (ca - cb * cg) / sg
        cz = math.sqrt(max(1.0 - cb * cb - cy * cy, 0.0))
        return np.array([
            [self.a, self.b * cg, self.c * cb],
            [0.0, self.b * sg, self.c * cy],
            [0.0, 0.0, self.c * cz],
        ])

    def reciprocal_matrix(self) -> np.ndarray:
        """3 × 3, ``2π (A⁻¹)ᵀ`` (Å⁻¹): ``q = B · hkl``."""
        return 2.0 * math.pi * np.linalg.inv(self.direct_matrix()).T

    def q_vectors(self, hkl) -> np.ndarray:
        """Cartesian q (Å⁻¹) of each row ``(h, k, l)``."""
        return np.atleast_2d(np.asarray(hkl, dtype=float)) @ self.reciprocal_matrix().T

    def q_of(self, hkl) -> np.ndarray:
        """|q| (Å⁻¹) of each row ``(h, k, l)``."""
        return np.linalg.norm(self.q_vectors(hkl), axis=1)

    def d_of(self, hkl) -> np.ndarray:
        return 2.0 * math.pi / self.q_of(hkl)

    def plane_normal(self, hkl) -> np.ndarray:
        """Unit vector along the normal of the plane ``(hkl)`` (the reciprocal vector)."""
        vector = self.reciprocal_matrix() @ np.asarray(hkl, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("The plane (000) has no normal.")
        return vector / norm

    def axis_direction(self, uvw) -> np.ndarray:
        """Unit vector along the crystal axis ``[uvw]`` (a direct-lattice vector)."""
        vector = self.direct_matrix() @ np.asarray(uvw, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            raise ValueError("The axis [000] has no direction.")
        return vector / norm

    def index_limits(self, q_max: float) -> tuple[int, int, int]:
        """The largest |h|, |k|, |l| a reflection with |q| ≤ ``q_max`` can have.

        Since **a** · **a*** = 2π and **a** · **b*** = **a** · **c*** = 0, h = (q · **a**) / 2π, so
        |h| ≤ q_max |**a**| / 2π (and so on for k and l), for any cell angles."""
        lengths = np.linalg.norm(self.direct_matrix(), axis=0)
        return tuple(int(math.floor(q_max * length / (2.0 * math.pi) + 1e-9)) for length in lengths)


__all__ = ["UnitCell"]
