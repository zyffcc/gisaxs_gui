"""Fixed optical and height inputs of a horizontal cut on a known substrate."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from .. import fit_links
from ..cut_model import model, shapes, sizes
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


@dataclass(frozen=True)
class KnownHeight:
    height: float
    spread: float
    law: str | None = None
    coupling: str | None = None

    def settings(self, law, coupling):
        return self.law or law, self.coupling or coupling


def heights_by_slot(heights):
    """Known upright-cylinder height distributions, numbered by increasing radius from 1."""
    values = []
    for number, record in (heights or {}).items():
        if isinstance(record, dict):
            height, spread = float(record["h"]), float(record["sigma_h"])
            law, coupling = record["law"], record["coupling"]
            if law not in sizes.LAWS or coupling not in model.COUPLINGS:
                raise ValueError("Unknown size distribution or coupling for a supplied upright-cylinder height")
        else:  # Earlier prediction records used the run's size law and coupling for all populations.
            height, spread = map(float, record)
            law = coupling = None
        if not (int(number) >= 1 and math.isfinite(height) and height > 0 and math.isfinite(spread) and spread >= 0):
            raise ValueError("An upright-cylinder height distribution needs h > 0 and a nonnegative relative spread")
        values.append((int(number), KnownHeight(height, spread, law, coupling)))
    return tuple(values)


def upright_da_terms(q, radius, spread, distance, distance_spread, law, geometry, axial_fraction):
    """An independent height distribution changes the coherent part of DA as in the manual fit."""
    form, coherent = model.size_averages("vertical_cylinder", q, radius=radius, radius_spread=spread, law=law,
                                         coherent=True, geometry=geometry)
    return model.couple(form, coherent * axial_fraction, q, distance, distance_spread, "da")


def height_fractions(data, types, law, coupling):
    fractions = {}
    if data.upright_heights:
        for slot, (family, structured) in enumerate(types):
            height = data.upright_height(slot)
            if family == "vertical_cylinder" and height is not None and structured and height.spread > 0:
                size_law, method = height.settings(law, coupling)
                if method == "da":
                    fractions[slot] = shapes.upright_height_coherence(size_law, height.height, height.spread,
                                                                      data.geometry())
    return fractions


def ordered_model(model):
    """Refine in the increasing-radius order used by the supplied heights and by the posterior prior."""
    if len(model.components) <= 1 or not any(c.heights for c in model.components):
        return model
    # Keep each R below the next R's upper limit and, with contact, below D_max / 2.
    # linked_fit.limits retains 1e-9 of a constrained interval at its upper end.
    # Reserve this interval so that even opposite unit-box corners satisfy both inequalities.
    next_cap = math.inf
    components = []
    for component in reversed(model.components):
        radius = component.params["R"]
        if component.structure:
            distance = component.params["D"]
            next_cap = min(next_cap, (distance.upper - (distance.upper - distance.lower) * 1e-9) / 2)
        upper = min(radius.upper, next_cap)
        if upper <= radius.lower:
            raise ValueError("The radius bounds have no ordered solution for the supplied cylinder heights")
        bounded = replace(radius, upper=upper, value=min(radius.value, upper))
        components.append(replace(component, params={**component.params, "R": bounded}))
        next_cap = (upper - (upper - radius.lower) * 1e-9) / _RANK_MARGIN
    model = replace(model, components=tuple(reversed(components)))
    order = [fit_links.Inequality(f"R_{n}", ">=", f"{_RANK_MARGIN}*R_{n-1}") for n in range(2, len(components) + 1)]
    # The engine normally sets D's contact limit before processing inequalities. Reapply it at the ordered R.
    contact = [fit_links.Inequality(f"D_{n}", ">=", f"2*R_{n}")
               for n, c in enumerate(model.components, 1) if c.structure]
    return fit_links.with_inequalities(model, order + contact)


# exp(log(R)) at the edge of a logarithmic fit interval can differ from R by a few ulps.
_RANK_MARGIN = 1.0 + 1e-12
