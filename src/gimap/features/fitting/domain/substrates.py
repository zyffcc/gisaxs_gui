"""Optical constants and serialization of the substrate used in the four-term DWBA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Optional

from src.gimap.shared.material_optics import (
    ATOMIC_WEIGHT, AVOGADRO, ENERGY_RANGE_KEV, HC_KEV_NM, MATERIALS, R_E_NM, Material,
    optical_constants, scattering_factors,
)

CUSTOM = "custom"
"""A substrate given by δ and β (at the cut's wavelength)."""


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
