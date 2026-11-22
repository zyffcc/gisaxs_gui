"""Reflection amplitudes at sharp, parallel interfaces, for a particle above the top surface."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.gimap.shared.material_optics import normal_wavevector_ratio


@dataclass(frozen=True)
class Layer:
    """A finite, homogeneous layer, with optical constants at the cut's wavelength."""
    thickness_nm: float
    delta: float
    beta: float

    def __post_init__(self):
        if not (math.isfinite(self.thickness_nm) and self.thickness_nm > 0):
            raise ValueError("A layer needs a finite thickness > 0 nm")
        if not (math.isfinite(self.delta) and self.delta >= 0 and math.isfinite(self.beta) and self.beta >= 0):
            raise ValueError("A layer needs finite δ ≥ 0 and β ≥ 0")


def parratt(k: float, angle_deg: float, delta: float, beta: float, layers: tuple[Layer, ...]) -> complex:
    """Complex reflection at the surface, with layers listed from the surface down to the semi-infinite substrate.

    Propagate the field E and its slope E'/i up through each layer, the admittance form of Parratt recursion.
    Their common round-trip phase cancels from the surface amplitude. The chosen sign keeps it bounded in a
    thick, absorbing film.
    """
    sine = math.sin(math.radians(angle_deg))

    def kz(d, b):
        return k * normal_wavevector_ratio(sine, d, b)

    field, slope = 1., kz(delta, beta)
    for layer in reversed(layers):
        wave = kz(layer.delta, layer.beta)
        if wave == 0:
            # At the exact critical angle the field in this layer is linear in depth.
            field -= 1j * slope * layer.thickness_nm
        else:
            odd = -0.5 * complex(np.expm1(2j * wave * layer.thickness_nm))
            even = 1 - odd
            field, slope = even * field + odd * slope / wave, even * slope + odd * wave * field
        scale = max(abs(field), abs(slope))
        field, slope = field / scale, slope / scale
    vacuum = k * sine
    return (vacuum * field - slope) / (vacuum * field + slope)
