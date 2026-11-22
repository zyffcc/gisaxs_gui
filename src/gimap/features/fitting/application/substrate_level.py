"""The level at which Model treats the spheres of a horizontal cut (Physics and Methods 9.10, 9.17), and the substrates
a person can choose for the four-term DWBA.

Four levels, from what Model holds: αi = αf = 0, the effective interface (q_z = 0); the angles with |r| = 0, the Born
term at the cut's q_z; the angles with |r| > 0, the direct and the exit-reflected path with a free effective r; the
angles with a known substrate, the four-term DWBA with the substrate's Fresnel coefficients (r is then not used).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

from ..domain.fit_model import FitModel, grazing
from ..domain.substrates import CUSTOM, ENERGY_RANGE_KEV, MATERIALS, Substrate

LEVELS = {
    "eia": "Effective interface (q_z = 0)",
    "born": "Born term at the cut's q_z",
    "two_path": "Two paths, free r",
    "dwba": "Four-term DWBA, substrate known",
}
SUBSTRATE_CHOICES = (("", "Unknown"),) + tuple((key, m.label) for key, m in MATERIALS.items()) + (
    (CUSTOM, "Typed δ, β"),)
"""The substrate combo of Model: none (the levels without it), a material of the table, or δ and β typed."""


@dataclass(frozen=True)
class Level:
    key: str
    substrate: Optional[Substrate] = None
    delta: Optional[float] = None
    beta: Optional[float] = None
    error: str = ""
    """Why a substrate chosen cannot be used (a wavelength outside the table)."""

    @property
    def label(self) -> str:
        return LEVELS[self.key]


def level(model: FitModel) -> Level:
    """The level of the model's spheres (module docstring)."""
    try:
        geometry = grazing(model)
    except ValueError as exc:  # the table does not cover the wavelength
        return Level("eia", model.substrate, error=str(exc))
    if geometry is None:
        return Level("eia", model.substrate)
    if geometry.mode == "dwba":
        return Level("dwba", model.substrate, geometry.delta, geometry.beta)
    if geometry.mode == "two_path" and geometry.refl_amp == 0:
        return Level("born", model.substrate)
    return Level(geometry.mode, model.substrate)


def with_substrate(model: FitModel, material: str, delta: float = 0.0, beta: float = 0.0) -> FitModel:
    """The model with a substrate of the table (key), ``custom`` δ and β, or none (``""``)."""
    if not material:
        return replace(model, substrate=None)
    if material == CUSTOM:
        return replace(model, substrate=Substrate(CUSTOM, float(delta), float(beta)))
    return replace(model, substrate=Substrate(material))


__all__ = ["CUSTOM", "ENERGY_RANGE_KEV", "LEVELS", "Level", "MATERIALS", "SUBSTRATE_CHOICES", "Substrate", "level",
           "with_substrate"]
