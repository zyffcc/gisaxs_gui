"""Hard parameter contracts shared by fitting methods and proposal conversion."""
from __future__ import annotations

import math

from .cut_model.shapes import RADIUS_POWER
from .cut_model.sizes import size_report
from .fit_model import INFO, FitModel


TAIL_RANGES = frozenset({"res_amplitude", "res_width", "res_exponent", "diffuse_amplitude", "diffuse_width",
                         "diffuse_hurst"})
"""The low-q terms (beam tail, diffuse term): their ranges and whether they are fixed are set by GIMaP (``beam_tail``,
Predict parameters), and kept with any composition (globals)."""


def _explicit_limit(owner, key, parameter, *, include_tail=False):
    defaults = INFO[key]
    if owner == "globals" and key in TAIL_RANGES and not include_tail:
        return False
    changed_range = (parameter.lower, parameter.upper) != (defaults.lower, defaults.upper)
    return changed_range or not parameter.free and (defaults.free or parameter.value != defaults.value)


def explicit_limits(model: FitModel) -> bool:
    """Defaults do not count as user constraints when exploring different compositions (nor do the low-q terms,
    which GIMaP sets from the curve and a proposal of any composition keeps)."""
    return any(_explicit_limit(owner, key, p) for (owner, key), p in model.parameters())


def constrain_proposal(proposal: FitModel, base: FitModel) -> FitModel:
    """A proposal is only a start: retain fixed values and all limits of matching populations."""
    same = [c.family for c in proposal.components] == [c.family for c in base.components]
    if not same and explicit_limits(base):
        raise ValueError("This composition cannot preserve Model's constraints. Use the particles in Model.")
    for path, parameter in base.parameters():
        if path[0] != "globals" and not same:
            continue
        if proposal.predict_level is not None and not _explicit_limit(*path, parameter, include_tail=True):
            # A saved Predict model carries the prior bounds, scale settings and resolution that produced its curve.
            # Only an explicit constraint in the receiving Model overrides those saved settings.
            continue
        try:
            value = proposal.get(path).value
        except (KeyError, IndexError):
            continue
        value = min(max(value, parameter.lower), parameter.upper) if parameter.free else parameter.value
        proposal = proposal.with_parameter(path, value=value, free=parameter.free,
                                           lower=parameter.lower, upper=parameter.upper)
    if same and proposal.predict_level is None:
        from dataclasses import replace
        proposal = replace(proposal, components=tuple(replace(c, structure=b.structure)
                           for c, b in zip(proposal.components, base.components)))
    return proposal


def manual_chain_start(proposal: FitModel, base: FitModel) -> FitModel:
    """Seed a manual series fit with the prediction, preserving its constraints, laws and point settings."""
    def order(item):
        return item.family, item.structure, item.value("R")

    populations = sorted(enumerate(base.components), key=lambda pair: order(pair[1]))
    proposed = sorted(proposal.components, key=order)
    values = {}
    for (index, component), candidate in zip(populations, proposed):
        if (component.family, component.structure) != (candidate.family, candidate.structure):
            continue
        params = {key: candidate.value(key) for key in candidate.keys() if key in component.keys()}
        if (component.law != candidate.law and
                {component.law, candidate.law} != {"gaussian", "number_gauss"}):
            # A proposal supplies moments, while the fit may ask for a median and the SD of ln(size).
            # If the number moments do not exist, keep the manual population's own starting sizes.
            for size, spread, power in (("R", "sigma_R", RADIUS_POWER.get(component.family, 6)),
                                        ("h", "sigma_h", 2)):
                if size not in params:
                    continue
                moments = size_report(candidate.law, params.pop(size), params.pop(spread), power)
                moment = moments["intensity" if component.law == "intensity_gauss" else "number"]
                if moment is not None:
                    centre, width = moment
                    if component.law == "lognormal_median":
                        centre /= math.sqrt(1 + width**2)
                        width = math.sqrt(math.log1p(width**2))
                    params[size], params[spread] = centre, width
        values.update({(index, key): value for key, value in params.items()})
    values.update({("globals", key): p.value for key, p in proposal.globals.items()})
    return base.with_values({path: min(p.upper, max(p.lower, values[path])) for path, p in base.parameters()
                             if p.free and p.tie is None and path in values})
