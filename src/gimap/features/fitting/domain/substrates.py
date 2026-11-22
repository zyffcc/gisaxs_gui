"""Substrate and optional films beneath a particle in the horizontal four-term DWBA (Physics and Methods 9.17)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Optional

from src.gimap.shared.material_optics import (
    ATOMIC_WEIGHT, AVOGADRO, ENERGY_RANGE_KEV, HC_KEV_NM, MATERIALS, R_E_NM, Material,
    optical_constants, scattering_factors,
)
from src.gimap.shared.numbers import format_number

from .cut_model.geometry import Geometry
from .cut_model.reflectivity import Layer

CUSTOM = "custom"
"""A substrate given by δ and β (at the cut's wavelength)."""


@dataclass(frozen=True)
class Substrate:
    """Semi-infinite material or typed constants at the cut's λ, optionally with layers in top-to-bottom order."""
    material: str = CUSTOM
    delta: float = 0.0
    beta: float = 0.0
    layers: tuple[Layer, ...] = ()

    def __post_init__(self):
        if self.material != CUSTOM and self.material not in MATERIALS:
            raise ValueError(f"Unknown substrate {self.material!r}")
        if self.material == CUSTOM and not (math.isfinite(self.delta) and self.delta > 0
                                            and math.isfinite(self.beta) and self.beta >= 0):
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
        def constants(delta, beta):
            return f"δ = {format_number(delta, digits=4)}, β = {format_number(beta)}"

        base = MATERIALS[self.material].label if self.material != CUSTOM else constants(self.delta, self.beta)
        layers = [f"{format_number(layer.thickness_nm, digits=6)} nm ({constants(layer.delta, layer.beta)})"
                  for layer in self.layers]
        return " / ".join([*layers, base])

    def geometry(self, wavelength_nm: float, alpha_i_deg: float, alpha_f_deg: float) -> Geometry:
        delta, beta = self.optical(wavelength_nm)
        return Geometry("dwba", wavelength_nm, alpha_i_deg, alpha_f_deg, delta=delta, beta=beta,
                        layers=self.layers, substrate_label=self.label)

    def to_dict(self) -> dict:
        result = {"material": self.material} if self.material != CUSTOM else {
            "material": CUSTOM, "delta": float(self.delta), "beta": float(self.beta)}
        if self.layers:
            result["layers"] = [asdict(layer) for layer in self.layers]
        return result

    @classmethod
    def from_dict(cls, raw: Optional[Mapping]) -> Optional["Substrate"]:
        if not raw:
            return None
        material = str(raw.get("material") or CUSTOM)
        layers = tuple(Layer(float(layer["thickness_nm"]), float(layer["delta"]), float(layer["beta"]))
                       for layer in raw.get("layers", ()))
        if material == CUSTOM:
            return cls(CUSTOM, float(raw.get("delta", 0.0)), float(raw.get("beta", 0.0)), layers)
        return cls(material, layers=layers)


__all__ = ["CUSTOM", "ENERGY_RANGE_KEV", "MATERIALS", "Layer", "Material", "Substrate", "optical_constants",
           "scattering_factors"]
