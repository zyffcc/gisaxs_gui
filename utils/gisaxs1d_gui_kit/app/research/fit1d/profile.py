"""Conditional (R, D) profile of one population: are there other (R, D) regions with an equivalent chi2?

After a fit, the (R, D) gain map of gridsearch.Aligner (every other nonlinear parameter fixed, amplitudes
re-solved) is the chi2 profile of that population over its whole allowed range at almost no cost. Regions of
the map whose chi2 comes within a band of the solution's own cell and that lie more than 10 % away in R or D
are candidate ALTERNATIVE solutions (typical: the D harmonics D/2, 2D of a paracrystal peak, a second
form-factor solution for R, or simply a wide flat valley around the solution). The best cell of each such region is verified
with the true non-negative solve on the full curve and, when polish > 0, refined by a short LM run; what
ends within the PLAN chi2 tolerance of the fit is returned as an alternative (and a refined alternative that
is BETTER than the fit simply becomes the new best solution of the run).

The profile is conditional: alternatives that need a simultaneous change of another population or of the
widths are not seen (research.fit1d.ensemble compares complete solutions of several starts for that), with
one exception that is searched explicitly: EXCHANGES of two active populations (exchange_candidates). On DEV
(2026-10-06, 281 curves) the largest class of undetected equivalent solutions was a structure-factor peak
assigned to another population than in the truth (the two D values exchanged), followed by exchanged sizes
of a sphere and a cylinder population; one population alone cannot move there, because the peak it leaves
stays unexplained.

Exchanged starts are far from their optimum (a few hundred to a few thousand chi2 units above the fit on DEV)
and still end at or BELOW the fit after an aligned LM start, so they are refined as complete starts and are
also a search move: on DEV row 78 the fast fit went from chi2 2912 to 1143 through one exchange.

Cost control: every start is first scored with one non-negative solve (its true chi2); the profile starts
are refined in order of that chi2 up to `budget` LM runs (a start already within the tolerance counts as an
alternative even when the budget is used up), the exchanges up to `exchange` aligned starts.
Spheres and standing cylinders only, like the alignment.
"""
from __future__ import annotations

import numpy as np

from research.fit1d.ensemble import REL_TOL, tolerance
from research.fit1d.fastgrid import structure_grid

BAND = 4.0            # candidate regions: screening chi2 within BAND x tolerance of the solution's own cell
MAX_REGIONS = 3       # regions verified per population (best gain first)
N_FINE_D = 600        # points of the fine 1D scan over D (steps of ~0.5 % over the factor-20 range of the codec)


def _own_cell(g, r, d):
    ir = int(np.argmin(np.abs(np.log(g['r']/r))))
    idx = int(np.argmin(np.abs(np.log(g['d']/d)))) if g['sf_free'] else 0
    return ir, idx


def candidates(aligner, u, j, tol, band=BAND, max_regions=MAX_REGIONS):
    """[(u_alt, gain deficit)] best cells of the alternative regions of slot j (see module docstring)."""
    from scipy.ndimage import label
    fit = aligner.fit
    typ = int(fit.types[j])
    if typ not in (1, 3):
        return []
    p = aligner.codec.decode(u)
    g = aligner.grids(u, j)
    if len(g['r']) < 2 and not g['sf_free']:
        return []
    m = aligner.screen.matrix_from(*aligner.screen.full(u))
    if getattr(aligner.screen, '_armed', None) is not None:
        aligner.screen._armed = False
    gain = aligner.gain(m, j, aligner.ff(j, typ, p['sigma_R'][j], g), aligner.sf(j, p['sigma_D'][j], g, p['D'][j]), g)
    r0, d0 = float(p['R'][j]), float(p['D'][j])
    ir, idx = _own_cell(g, r0, d0)
    window = gain[max(ir-1, 0):ir+2, max(idx-1, 0):idx+2]
    own = float(np.max(window[np.isfinite(window)])) if np.any(np.isfinite(window)) else -np.inf
    if not np.isfinite(own):
        return []
    far_r = np.abs(g['r'][:, None]/r0-1) > REL_TOL
    far_d = (np.abs(g['d'][None, :]/d0-1) > REL_TOL) if g['sf_free'] else np.zeros((1, gain.shape[1]), bool)
    # every cell more than 10 % away in R or D whose screening chi2 is within the band: grouped into connected
    # regions AFTER the solution's own neighbourhood is cut out, so that a wide flat valley around the solution
    # (many shallow minima: the local 1-sigma of one of them says nothing) yields candidates on both sides,
    # exactly like a separate harmonic does
    sel = np.isfinite(gain) & (gain >= own-band*tol) & (far_r | far_d)
    regions, n = label(sel, structure=np.ones((3, 3), int))
    found = []
    for lab in range(1, n+1):
        flat = int(np.argmax(np.where(regions == lab, gain, -np.inf)))
        a, b = np.unravel_index(flat, gain.shape)
        found.append((float(own-gain[a, b]), int(a), int(b)))
    found.sort()
    return [(aligner.place(u, j, g, a, b), deficit) for deficit, a, b in found[:max_regions]]


def fine_d_candidates(aligner, u, j, tol, band=BAND, max_peaks=MAX_REGIONS, n_fine=N_FINE_D):
    """[(u_alt, gain deficit)] local chi2 minima along D alone (R and everything else at the solution), on a
    fine logarithmic D grid over the whole allowed range. The (R, D) map of candidates() has steps of ~12 % in
    D; at high counts and a narrow paracrystal (small w) the chi2 valleys in D are only ~1 % wide, so the
    harmonics of D (D/2, 2D, ...) fall between its cells (DEV row 44, 2026-10-06)."""
    fit = aligner.fit
    typ = int(fit.types[j])
    if typ not in (1, 3) or not fit.sf_on or 7*j+4 in aligner.fixed:
        return []
    p = aligner.codec.decode(u)
    r0, d0 = float(p['R'][j]), float(p['D'][j])
    lo, hi = np.array(u, float), np.array(u, float)
    lo[7*j+4], hi[7*j+4] = 0.0, 1.0
    d_lo, d_hi = float(aligner.codec.decode(lo)['D'][j]), float(aligner.codec.decode(hi)['D'][j])
    if not d_hi > d_lo*(1+1e-9):
        return []
    grid = np.unique(np.r_[np.exp(np.linspace(np.log(d_lo), np.log(d_hi), n_fine)), d0])
    own = int(np.argmin(np.abs(grid-d0)))
    g = dict(ur=np.array([u[7*j]]), r=np.array([r0]), d0=np.array([d_lo]), d1=np.array([d_hi]), d=grid, sf_free=True)
    screen = aligner.screen
    m = screen.matrix_from(*screen.full(u))
    if getattr(screen, '_armed', None) is not None:
        screen._armed = False
    ff = (screen.ph.combine(typ, screen.ph.radial(typ, r0, p['sigma_R'][j]), None)*aligner._w)[None, :]
    sf = structure_grid(screen.ph, grid, p['sigma_D'][j])
    gain = aligner.gain(m, j, ff, sf, g)[0]
    if not np.isfinite(gain[own]):
        return []
    ok = np.isfinite(gain) & (gain >= gain[own]-band*tol) & (np.abs(grid/d0-1) > REL_TOL)
    padded = np.r_[-np.inf, np.where(np.isfinite(gain), gain, -np.inf), -np.inf]
    peak = (padded[1:-1] >= padded[:-2]) & (padded[1:-1] >= padded[2:])
    found = sorted((float(gain[own]-gain[k]), int(k)) for k in np.flatnonzero(ok & peak))
    return [(aligner.place(u, j, g, 0, k), deficit) for deficit, k in found[:max_peaks]]


def _set_log(aligner, u, j, key, index, target):
    """u with the log-uniform coordinate `index` (R or D of slot j) set to `target` nm; None when the
    coordinate is fixed or the target lies outside its allowed range."""
    if index in aligner.fixed:
        return None
    lo, hi = np.array(u, float), np.array(u, float)
    lo[index], hi[index] = 0.0, 1.0
    v0, v1 = float(aligner.codec.decode(lo)[key][j]), float(aligner.codec.decode(hi)[key][j])
    if not v1 > v0*(1+1e-9) or not v0*(1-1e-6) <= target <= v1*(1+1e-6):
        return None
    out = np.array(u, float)
    out[index] = float(np.clip(np.log(target/v0)/np.log(v1/v0), 0.0, 1.0))
    return out


def _exchanged(aligner, u, p, i, j, swap_r, swap_d):
    """u with slots i and j exchanging R (and sigma_R) and / or D (and w_rel), in physical units; None when
    a value does not fit into the other slot's allowed range or a needed coordinate is fixed."""
    sf_on = bool(aligner.fit.sf_on)
    out = np.array(u, float)
    for a, b in ((i, j), (j, i)):                              # slot a takes the values of slot b
        if swap_r:
            out = _set_log(aligner, out, a, 'R', 7*a, float(p['R'][b]))
            if out is None:
                return None
            if 7*a+1 not in aligner.fixed:
                out[7*a+1] = aligner._width_u(out, 7*a+1, float(p['sigma_R'][b]))[0]
        if sf_on:                                              # D is coded relative to R: set it again after R moved
            target = float(p['D'][b] if swap_d else p['D'][a])
            if 7*a+4 in aligner.fixed:                         # a fixed D must already have the wanted value
                if abs(float(aligner.codec.decode(out)['D'][a])/target-1) > 1e-9:
                    return None
            else:
                out = _set_log(aligner, out, a, 'D', 7*a+4, target)
                if out is None:
                    return None
            if swap_d and 7*a+5 not in aligner.fixed:
                out[7*a+5] = aligner._width_u(out, 7*a+5, float(p['sigma_D'][b]))[0]
    return out


def exchange_candidates(aligner, u, coef):
    """[(label, u_alt)] starts in which two active populations exchange roles (see module docstring):
    same type: the two D (the other exchange is a relabelling); different types: D, R, or both."""
    fit = aligner.fit
    p = aligner.codec.decode(u)
    slots = [j for j in range(4) if fit.types[j] in (1, 3) and coef[j] > 0]
    out = []
    for n, i in enumerate(slots):
        for j in slots[n+1:]:
            kinds = [(False, True)] if fit.types[i] == fit.types[j] else [(False, True), (True, False), (True, True)]
            for swap_r, swap_d in kinds:
                if swap_d and not fit.sf_on:
                    continue
                u_alt = _exchanged(aligner, u, p, i, j, swap_r, swap_d)
                if u_alt is not None and np.max(np.abs(u_alt-np.asarray(u, float))) > 1e-9:
                    out.append((f"exchange{i}{j}{'R' if swap_r else ''}{'D' if swap_d else ''}", u_alt))
    return out


def _score(fit, starts):
    """[(true chi2, label, u, coef)] of the starts (one non-negative solve each), best first."""
    out = []
    for label, u_alt in starts:
        fit.trace = []
        c, _, residual = fit.solve(fit.matrix_from(*fit.full(u_alt)))
        out.append((float(residual@residual), label, u_alt, c))
    return sorted(out, key=lambda item: item[0])


def alternatives(run, polish=0, band=BAND, max_regions=MAX_REGIONS, budget=None, exchange=None, exchange_steps=16):
    """Alternative solutions of the best solution of a CompositionRun; adds them to the run as stage 'P'
    candidates and returns their number.

    polish          LM evaluations spent on a profile start (0: no refinement, its true chi2 decides)
    budget          profile starts refined (None: all), best true chi2 first; a start that is within the
                    tolerance without refinement counts as an alternative in any case
    exchange        population exchanges refined (None: all, 0: none), best true chi2 first; each one is a
                    complete aligned start (align -> LM -> align -> LM, controller.aligned_lm) of
                    exchange_steps LM evaluations, because an exchanged start is far from its optimum. An
                    exchange that is not refined still counts when its true chi2 is within the tolerance
                    as it stands (one non-negative solve)"""
    from research.fit1d.controller import aligned_lm, lm_run
    best = run.best
    if best.get('cov_u') is None:
        return 0
    aligner, fit = run.aligner, run.fit
    u = np.asarray(best['u'], float)
    coef = np.asarray(best['coef'], float)
    tol = tolerance(best['chi2'], run.n)
    starts = []
    for j in range(4):
        if run.setup.types[j] not in (1, 3) or not coef[j] > 0:
            continue
        found = candidates(aligner, u, j, tol, band, max_regions)+fine_d_candidates(aligner, u, j, tol, band, max_regions)
        starts += [(f'profile{j}', u_alt) for u_alt, _ in found]
    added = refined = 0
    for chi2, label, u_alt, c in _score(fit, starts):
        if polish and (budget is None or refined < budget):
            r = lm_run(fit, run.setup, u_alt, polish)
            refined += 1
        elif chi2-best['chi2'] <= tol:
            r = dict(u=u_alt, coef=c, chi2=chi2, cov_u=None, coords=[], status=None, nfev=1, evaluations=1, falling=False)
        else:
            continue
        if r['chi2']-best['chi2'] <= tol:
            run._add('P', label, r)
            added += 1
    for n, (chi2, label, u_alt, c) in enumerate(_score(fit, exchange_candidates(aligner, u, coef))):
        if exchange is None or n < exchange:
            r = aligned_lm(fit, run.setup, aligner, u_alt, exchange_steps)
        elif chi2-best['chi2'] <= tol:              # not refined: an exchange that is equivalent as it stands
            r = dict(u=u_alt, coef=c, chi2=chi2, cov_u=None, coords=[], status=None, nfev=1, evaluations=1, falling=False)
        else:
            continue
        if r['chi2']-best['chi2'] <= tol:
            run._add('P', label, r)
            added += 1
    return added
