"""Vectorised grids for the (R, D) alignment: the same numbers as research.fit1d.gridsearch.Aligner, faster.

gridsearch.Aligner builds the form-factor grid with one table lookup per radius and the structure-factor grid
with one call per distance (64 + 96 calls per gain map, about 40 % of the alignment time on K = 2-3 curves,
profile of 2026-10-06). The table lookup and the Hosemann structure factor are elementwise numpy functions, so
one call with a column vector of radii / distances gives the whole grid; each element goes through the same
operations (on a platform whose numpy uses the same code for scalars and arrays the result is bit-identical,
elsewhere equal to rounding). gridsearch.py itself stays untouched: its source is part of the chi2_ref cache
keys (research.bench.reference), and the reference search keeps using the plain Aligner.

Spheres with the DWBA model (research.fast_inference.tabfit.TabPhysics.sphere_dwba) take one radius at a
time: the base class loop is used for them.
"""
from __future__ import annotations

import numpy as np

from research.fit1d.gridsearch import Aligner


def structure_grid(ph, distances, w_rel):
    """ph.structure for every distance: [n_d, n_q]."""
    return ph.structure(np.asarray(distances, float)[:, None], w_rel)


class FastAligner(Aligner):
    def ff(self, j, typ, sigma_r, g):
        ph = self.screen.ph
        if typ not in (1, 3) or (typ == 1 and getattr(ph, 'dwba', None) is not None):
            return super().ff(j, typ, sigma_r, g)
        key = (j, typ, round(float(sigma_r), 9))
        if key not in self._ff:
            if len(self._ff) > 64:
                self._ff.clear()
            self._ff[key] = ph.combine(typ, ph.radial(typ, np.asarray(g['r'], float)[:, None], sigma_r), None)*self._w[None, :]
        return self._ff[key]

    def sf(self, j, w_rel, g, d_current):
        if not self.screen.sf_on or not g['sf_free']:
            return super().sf(j, w_rel, g, d_current)
        key = (j, round(float(w_rel), 9))
        if key not in self._sf:
            if len(self._sf) > 64:
                self._sf.clear()
            self._sf[key] = structure_grid(self.screen.ph, g['d'], w_rel)
        return self._sf[key]
