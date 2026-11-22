"""Fixed optical and height inputs of a horizontal cut on a known substrate."""

from __future__ import annotations

import math

from ..cut_model import model, shapes
from ..substrates import Substrate


def resolve(substrate, grazing):
    """Resolve the optical constants at the measured wavelength and require two measured angles."""
    if not isinstance(substrate, Substrate):
        if substrate and substrate.get("material", "custom") == "custom" and not {"delta", "beta"} <= substrate.keys():
            raise ValueError("Give both δ and β of the substrate at the cut's wavelength")
        substrate = Substrate.from_dict(substrate)
    if substrate is None:
        raise ValueError("A four-term DWBA cut needs a substrate")
    grazing = grazing or {}
    wavelength, incident, exit_angle = (float(grazing.get(key) or 0.0)
                                        for key in ("wavelength", "alpha_i", "alpha_f"))
    if not (math.isfinite(wavelength) and wavelength > 0 and 0 < incident < 90 and 0 < exit_angle < 90):
        raise ValueError("A four-term DWBA cut needs a wavelength and both positive grazing angles")
    return substrate, substrate.geometry(wavelength, incident, exit_angle)


def heights_by_slot(heights):
    """Known upright-cylinder height distributions, numbered by increasing radius from 1."""
    values = []
    for number, pair in (heights or {}).items():
        height, spread = map(float, pair)
        if not (int(number) >= 1 and math.isfinite(height) and height > 0 and math.isfinite(spread) and spread >= 0):
            raise ValueError("An upright-cylinder height distribution needs h > 0 and a nonnegative relative spread")
        values.append((int(number), height, spread))
    return tuple(values)


def upright_da_terms(q, radius, spread, distance, distance_spread, law, geometry, axial_fraction):
    """An independent height distribution changes the coherent part of DA as in the manual fit."""
    form, coherent = model.size_averages("vertical_cylinder", q, radius=radius, radius_spread=spread, law=law,
                                         coherent=True, geometry=geometry)
    return model.couple(form, coherent * axial_fraction, q, distance, distance_spread, "da")


def height_fractions(data, types, law, coupling):
    fractions = {}
    if coupling == "da" and data.upright_heights:
        for slot, (family, structured) in enumerate(types):
            height = data.upright_height(slot)
            if family == "vertical_cylinder" and structured and height and height[1] > 0:
                fractions[slot] = shapes.upright_height_coherence(law, *height, data.geometry())
    return fractions
