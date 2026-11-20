"""Starts for least squares at the degeneracies of a cut of several populations (numpy).

A population's paracrystal peak fixes its D to a fraction of a per cent, while its form factor fixes R only loosely:
another R at the same D, or the structures of two populations exchanged, often describe the curve as well, and the
network seldom proposes them (Physics and Methods 9.13). From a solution of a composition of two or three
populations, ``starts`` gives every population's R × ``FACTORS`` at its D (a random cylinder's h with it) and every
pair's paracrystal (D, σnn/D) exchanged, where the composition's order of radii and D ≥ 2R still hold.
"""

from __future__ import annotations

import numpy as np

from . import encoding as E

FACTORS = (0.5, 2.0)


def starts(values, composition: int) -> list[np.ndarray]:
    """Physical values (DIMS,) of the moved starts of the solution ``values`` of ``composition``."""
    values = np.asarray(values, float)
    types = E.composition_types(composition)
    k = len(types)
    if k < 2:
        return []
    out = []

    def keeps_order(v) -> bool:
        radii = np.array([v[s * E.PER_SLOT] for s in range(k)])
        apart = all(not structure or v[s * E.PER_SLOT + 4] >= 2.0 * v[s * E.PER_SLOT]
                    for s, (_family, structure) in enumerate(types))
        return bool(np.all(np.diff(radii) > 0) and apart)

    for slot, (family, _structure) in enumerate(types):
        j = slot * E.PER_SLOT
        for factor in FACTORS:
            v = values.copy()
            v[j] *= factor
            if family == "cylinder":
                v[j + 2] *= factor
            if keeps_order(v):
                out.append(v)
    for a in range(k):
        for b in range(a + 1, k):
            if not (types[a][1] and types[b][1]):
                continue
            v = values.copy()
            i, j = a * E.PER_SLOT, b * E.PER_SLOT
            v[[i + 4, i + 5]], v[[j + 4, j + 5]] = values[[j + 4, j + 5]], values[[i + 4, i + 5]]
            if keeps_order(v):
                out.append(v)
    return out


__all__ = ["FACTORS", "starts"]
