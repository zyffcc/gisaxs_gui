"""Compare the occupied, symmetry-expanded sites with the CIF sum formula."""

import math
import re

from .scattering import ATOMIC_NUMBER
from .symmetry import expand_sites

SITE_MISMATCH = "Expanded sites differ from the CIF sum formula × Z. Check the site model before using its reflection intensities."
RATIO_MISMATCH = "Expanded site ratios differ from the CIF sum formula. Z was not given."
INVALID_Z = "The CIF formula-unit count Z must be positive. Check Z before using the bulk formula."
_ELEMENT = re.compile(r"([A-Z][a-z]?)\s*(\d+(?:\.\d+)?|\.\d+)?\s*")


def _formula_counts(formula):
    counts = {}
    remaining = formula.strip()
    while remaining:
        match = _ELEMENT.match(remaining)
        if not match or match[1] not in ATOMIC_NUMBER:
            return None
        amount = float(match[2] or 1.)
        if amount <= 0. or not math.isfinite(amount):
            return None
        counts[match[1]] = counts.get(match[1], 0.) + amount
        remaining = remaining[match.end():]
    return counts or None


def phase_composition(phase):
    """Counts per cell from each source. Without Z only elemental ratios can be checked."""
    sites = {}
    if phase.atoms:
        positions = expand_sites([atom.xyz for atom in phase.atoms], phase.operations)
        for atom, images in zip(phase.atoms, positions):
            sites[atom.element] = sites.get(atom.element, 0.) + atom.occupancy * len(images)
    formula = _formula_counts(phase.formula_sum)
    z = phase.formula_units_z
    bulk = None
    if formula and z is not None and math.isfinite(z) and z > 0:
        bulk = {el: n*z for el, n in formula.items()}
    warning = None
    status = "unavailable"
    if z is not None and (z <= 0 or not math.isfinite(z)):
        status, warning = "invalid Z", INVALID_Z
    elif sites and formula:
        scale = z if bulk else sum(sites.values()) / sum(formula.values())
        matches = all(math.isclose(sites.get(el, 0.), scale * formula.get(el, 0.), abs_tol=1e-4, rel_tol=1e-3)
                      for el in sites.keys() | formula.keys())
        if matches:
            status = "consistent" if bulk else "consistent ratios"
        else:
            status = "mismatch"
            warning = SITE_MISMATCH if bulk else RATIO_MISMATCH
    return {"site_atoms_per_cell": sites, "formula_atoms_per_cell": bulk,
            "formula_sum": phase.formula_sum, "formula_units_z": z,
            "composition_check": status, "composition_warning": warning}
