"""Block-coordinate grid search over (R, D) of one population at a time (global peak / oscillation alignment).

Why (DEV studies 2026-10-01, results/m2_*_study.json): LM from a network head misses the optimum when a
population's R is off by more than ~20 % or its D by more than the paracrystal peak width; the chi2 landscape
has many local minima along R (form-factor oscillations) and D (structure-factor peaks), while the relative
widths have wide basins. R and D of ONE population can be searched exhaustively at almost no cost, because
the model column of population j factorises as F(q; R, sigma_R) * S(q; D, w_rel) and every other column is
fixed. With O the other (weighted) columns and c = F_R * S_D,

    gain(R, D) = (c_perp . y)^2 / (c_perp . c_perp),   c_perp = c - O (O^T O)^-1 O^T c,

is the chi2 reduction of adding c (unconstrained amplitudes); all inner products of an nR x nD grid are a few
matrix products. The screening runs on a rebinned copy of the curve (<= SCREEN_POINTS points); the best grid
points are then checked with the real non-negative solve on the full curve, so a step is only accepted when
the true chi2 falls.

  align(u)    Gauss-Seidel sweeps of the single-population search (largest amplitude first)
  pursuit(u)  greedy construction without the network: start from background + low-q term, add the
              populations of the composition one at a time (full grid, a few widths, every remaining type)

Constraints: grids are generated through the fit's codec (decode at u = 0..1), so constrained boxes and fixed
coordinates (Setup.fixed) are respected and fixed values are never touched. Spheres and standing cylinders
only: random cylinders (two size coordinates) are left as they are.
"""
from __future__ import annotations

import numpy as np

N_R, N_D, TOP = 64, 96, 4
SCREEN_POINTS = 384
SIGMA_GRID = (0.08, 0.2, 0.35)
W_GRID = (0.08, 0.18, 0.35)


class Aligner:
    def __init__(self, fit, screen_fit=None, fixed=(), n_r=N_R, n_d=N_D, top=TOP):
        self.fit, self.screen = fit, screen_fit if screen_fit is not None else fit
        self.fixed = {int(c) for c in fixed}
        self.n_r, self.n_d, self.top = int(n_r), int(n_d), int(top)
        self.codec = fit.codec
        self._grids, self._ff, self._sf = {}, {}, {}
        s = self.screen
        self._w = 1.0/s.sigma
        background = getattr(s, 'background', None)
        self._y = (s.y if background is None else s.y-background)*self._w
        self._skip = {4} if background is not None else set()

    # ------------------------------------------------------------------ helpers
    def chi2(self, u, active=None):
        m = self.fit.matrix_from(*self.fit.full(u))
        if active is not None:
            for k in range(4):
                if k not in active:
                    m[:, k] = 0.0
        armed = getattr(self.fit, '_armed', None)
        if armed is not None:
            self.fit._armed = False                    # do not log grid evaluations in the LM trace
        r = self.fit.solve(m)[2]
        return float(r@r)

    def grids(self, u, j):
        """R grid (with its u values) and D grid of slot j through the codec; cached per slot."""
        if j in self._grids:
            return self._grids[j]
        ur = np.array([u[7*j]]) if 7*j in self.fixed else np.linspace(0.0, 1.0, self.n_r)
        probe = np.repeat(np.asarray(u, float)[None, :], len(ur), 0)
        probe[:, 7*j] = ur
        lo, hi = probe.copy(), probe.copy()
        lo[:, 7*j+4], hi[:, 7*j+4] = 0.0, 1.0
        dec = [self.codec.decode(row) for row in probe]
        r = np.array([d['R'][j] for d in dec])
        d0 = np.array([self.codec.decode(row)['D'][j] for row in lo])
        d1 = np.array([self.codec.decode(row)['D'][j] for row in hi])
        sf_free = self.fit.sf_on and 7*j+4 not in self.fixed and np.all(d1 > d0*(1+1e-9))
        if sf_free:
            d = np.exp(np.linspace(np.log(d0.min()), np.log(d1.max()), self.n_d))
        else:
            d = None
        self._grids[j] = dict(ur=ur, r=r, d0=d0, d1=d1, d=d, sf_free=sf_free)
        return self._grids[j]

    def ff(self, j, typ, sigma_r, g):
        key = (j, typ, round(float(sigma_r), 9))
        if key not in self._ff:
            if len(self._ff) > 64:
                self._ff.clear()
            ph = self.screen.ph
            self._ff[key] = np.array([ph.combine(typ, ph.radial(typ, r, sigma_r), None) for r in g['r']])*self._w[None, :]
        return self._ff[key]

    def sf(self, j, w_rel, g, d_current):
        if not self.screen.sf_on:
            return np.ones((1, len(self.screen.q)))
        if not g['sf_free']:
            return self.screen.ph.structure(d_current, w_rel)[None, :]
        key = (j, round(float(w_rel), 9))
        if key not in self._sf:
            if len(self._sf) > 64:
                self._sf.clear()
            self._sf[key] = np.array([self.screen.ph.structure(d, w_rel) for d in g['d']])
        return self._sf[key]

    def gain(self, m, j, ff, sf, g):
        """Gain table [R, D] of column ff_R * sf_D for slot j against the non-zero other columns of m."""
        keep = [k for k in range(6) if k != j and k not in self._skip and np.any(m[:, k] != 0)]
        cy = ff@(sf*self._y[None, :]).T
        cc = (ff*ff)@(sf*sf).T
        if keep:
            others = m[:, keep]*self._w[:, None]
            others = others/np.linalg.norm(others, axis=0)
            gram_inv = np.linalg.inv(others.T@others+1e-10*np.eye(len(keep)))
            b_o = others.T@self._y
            oc = np.stack([ff@(sf*others[:, k][None, :]).T for k in range(len(keep))], -1)
            proj = oc@gram_inv
            cy = cy-proj@b_o
            cc = cc-np.einsum('rdp,rdp->rd', proj, oc)
        cc = np.maximum(cc, 1e-300)
        valid = np.ones(cy.shape, bool)
        if g['sf_free']:
            valid = (g['d'][None, :] >= g['d0'][:, None]*(1-1e-12)) & (g['d'][None, :] <= g['d1'][:, None]*(1+1e-12))
        return np.where((cy > 0) & valid, cy*cy/cc, -np.inf)

    def place(self, u, j, g, ir, idx):
        """u with slot j moved to grid point (ir, idx); only the R and D coordinates change."""
        u2 = np.array(u, float)
        if 7*j not in self.fixed:
            u2[7*j] = g['ur'][ir]
        if g['sf_free']:
            u2[7*j+4] = float(np.clip(np.log(g['d'][idx]/g['d0'][ir])/np.log(g['d1'][ir]/g['d0'][ir]), 0.0, 1.0))
        return u2

    # ------------------------------------------------------------------ single population
    def search_population(self, u, j, current):
        typ = int(self.fit.types[j])
        if typ not in (1, 3) or (7*j in self.fixed and (not self.fit.sf_on or 7*j+4 in self.fixed)):
            return u, current
        p = self.codec.decode(u)
        g = self.grids(u, j)
        m = self.screen.matrix_from(*self.screen.full(u))
        gain = self.gain(m, j, self.ff(j, typ, p['sigma_R'][j], g), self.sf(j, p['sigma_D'][j], g, p['D'][j]), g)
        best_u, best = u, current
        for flat in np.argsort(gain, axis=None)[::-1][:self.top]:
            ir, idx = np.unravel_index(flat, gain.shape)
            if not np.isfinite(gain[ir, idx]):
                break
            u2 = self.place(u, j, g, ir, idx)
            c = self.chi2(u2)
            if c < best:
                best_u, best = u2, c
        return best_u, best

    def align(self, u, sweeps=2):
        """Returns (u, chi2, improved)."""
        u = np.array(u, float)
        start = chi2 = self.chi2(u)
        slots = [j for j in range(4) if self.fit.types[j] in (1, 3)]
        for _ in range(int(sweeps)):
            coef = self.fit.solve(self.fit.matrix_from(*self.fit.full(u)))[0]
            if getattr(self.fit, '_armed', None) is not None:
                self.fit._armed = False
            changed = False
            for j in sorted(slots, key=lambda k: -coef[k]):
                u2, c = self.search_population(u, j, chi2)
                if c < chi2*(1-1e-9):
                    u, chi2, changed = u2, c, True
            if not changed:
                break
        return u, chi2, chi2 < start

    # ------------------------------------------------------------------ greedy construction
    def _width_u(self, u, index, target):
        """u value of a linear width coordinate (sigma_R at 7j+1, w_rel at 7j+5) for a target value."""
        key = 'sigma_R' if index % 7 == 1 else 'sigma_D'
        j = index//7
        a, b = np.array(u, float), np.array(u, float)
        a[index], b[index] = 0.0, 1.0
        v0, v1 = self.codec.decode(a)[key][j], self.codec.decode(b)[key][j]
        if not v1 > v0:
            return u[index], v0
        x = float(np.clip((target-v0)/(v1-v0), 0.0, 1.0))
        return x, v0+x*(v1-v0)

    def pursuit(self, u_template, sigma_grid=SIGMA_GRID, w_grid=W_GRID, top=3):
        """Greedy construction (see module docstring); random-cylinder slots keep the template values."""
        u = np.array(u_template, float)
        types = self.fit.types
        slots = [j for j in range(4) if types[j] in (1, 3)]
        placed = [j for j in range(4) if types[j] == 2]
        while True:
            free = [j for j in slots if j not in placed]
            if not free:
                break
            m = self.screen.matrix_from(*self.screen.full(u))
            for k in range(4):
                if k not in placed:
                    m[:, k] = 0.0
            best = None
            for typ in sorted({int(types[j]) for j in free}):
                j = next(k for k in free if types[k] == typ)
                g = self.grids(u, j)
                p = self.codec.decode(u)
                sigmas = [(u[7*j+1], p['sigma_R'][j])] if 7*j+1 in self.fixed else [self._width_u(u, 7*j+1, t) for t in sigma_grid]
                if not self.fit.sf_on or 7*j+5 in self.fixed:
                    widths = [(u[7*j+5], p['sigma_D'][j])]
                else:
                    widths = [self._width_u(u, 7*j+5, t) for t in w_grid]
                for us, sr in sigmas:
                    ff = self.ff(j, typ, sr, g)
                    for uw, wr in widths:
                        gain = self.gain(m, j, ff, self.sf(j, wr, g, p['D'][j]), g)
                        for flat in np.argsort(gain, axis=None)[::-1][:top]:
                            ir, idx = np.unravel_index(flat, gain.shape)
                            if not np.isfinite(gain[ir, idx]):
                                break
                            u2 = self.place(u, j, g, ir, idx)
                            u2[7*j+1], u2[7*j+5] = us, uw
                            c = self.chi2(u2, placed+[j])
                            if best is None or c < best[0]:
                                best = (c, j, u2)
            if best is None:
                break
            _, j, u = best
            placed.append(j)
        return u


def screen_curve(q, y, s, max_points=SCREEN_POINTS):
    """Rebinned copy of the curve for the screening (the refinement always uses every point)."""
    from research.conditional_fit.pipeline import network_view
    if len(q) <= max_points:
        return q, y, s
    return network_view(q, y, s, max_points=max_points)
