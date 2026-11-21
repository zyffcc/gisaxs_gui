"""Hard parameter contracts shared by fitting methods and proposal conversion."""
from __future__ import annotations

from .fit_model import INFO, FitModel


TAIL_RANGES = frozenset({"res_amplitude", "res_width", "res_exponent"})
"""The beam tail's ranges: set from the curve by GIMaP (``beam_tail``), and kept with any composition (globals)."""


def explicit_limits(model: FitModel) -> bool:
    """Defaults do not count as user constraints when exploring different compositions (nor do the beam tail's
    ranges, which GIMaP sets from the curve and a proposal of any composition keeps)."""
    return any(((p.lower, p.upper) != (INFO[key].lower, INFO[key].upper) and not (owner == "globals" and key in TAIL_RANGES))
               or (not p.free and (INFO[key].free or p.value != INFO[key].value))
               for (owner, key), p in model.parameters())


def constrain_proposal(proposal: FitModel, base: FitModel) -> FitModel:
    """A proposal is only a start: retain fixed values and all limits of matching populations."""
    same = [c.family for c in proposal.components] == [c.family for c in base.components]
    if not same and explicit_limits(base):
        raise ValueError("This composition cannot preserve Model's constraints. Use the particles in Model.")
    for path, parameter in base.parameters():
        if path[0] != "globals" and not same:
            continue
        try:
            value = proposal.get(path).value
        except (KeyError, IndexError):
            continue
        value = min(max(value, parameter.lower), parameter.upper) if parameter.free else parameter.value
        proposal = proposal.with_parameter(path, value=value, free=parameter.free,
                                           lower=parameter.lower, upper=parameter.upper)
    if same:
        from dataclasses import replace
        proposal = replace(proposal, components=tuple(replace(c, structure=b.structure)
                           for c, b in zip(proposal.components, base.components)))
    return proposal
