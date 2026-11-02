"""Ambiguity of the reportable parameters, judged from the chi2-equivalent solutions a fit has found.

Why (DEV, 2026-10-01, bench v3): on broad synthetic data most R and D values have other solutions with an
equivalent chi2 (D harmonics, other decompositions into populations); the local 1-sigma of one minimum says
nothing about them, so a fit can be confidently wrong. A reportable parameter (R_nm, D_nm of an active
population of the best solution) is called AMBIGUOUS here when another finished solution of the same fit
with chi2 within the PLAN tolerance max(25, 0.02 N) max(1, chi2_best / N) of the best

  * has that parameter more than REL_TOL (10 %) away, populations matched per type by the assignment that
    minimises sum |dlnR| + |dlnD| (all permutations, <= 4 populations per type), or
  * has a different number of active populations of that type (the solutions decompose the curve differently).

This is the definition bench v3 uses with the truth (research.bench.metrics), applied to the solutions the
product itself has: the refined starts (S1/S2), the robust starts (S3/S4) and the conditional (R, D) profile
alternatives (stage P, research.fit1d.profile). More starts find more alternatives: 'unambiguous' only means
that none of the solutions that were tried disagrees. assess() also returns the range of every parameter
over the equivalent solutions.
"""
from __future__ import annotations

import itertools

import numpy as np

REL_TOL = 0.10
DCHI2 = (25.0, 0.02)           # PLAN same-basin chi2 tolerance: max(25, 0.02 N) max(1, chi2/N)
FINISHED_STAGES = ('S3', 'S4', 'P')


def tolerance(chi2_best, n):
    return max(DCHI2[0], DCHI2[1]*n)*max(1.0, chi2_best/max(n, 1))


def describe(setup, cand):
    """[{slot, type, active, R, D}] of one solution (D None without a structure factor)."""
    p = setup.codec.decode(np.asarray(cand['u'], float))
    coef = np.asarray(cand['coef'], float)
    return [dict(slot=j, type=int(setup.types[j]), active=bool(coef[j] > 0), R=float(p['R'][j]),
                 D=float(p['D'][j]) if setup.sf_on else None) for j in range(4) if setup.types[j]]


def _cost(a, b):
    c = abs(np.log(a['R']/b['R']))
    if a['D'] is not None and b['D'] is not None:
        c += abs(np.log(a['D']/b['D']))
    return c


def match(best, other):
    """{slot of best: matched population of other, or None when the active counts of its type differ}."""
    out = {}
    for typ in sorted({p['type'] for p in best}):
        la = [p for p in best if p['type'] == typ and p['active']]
        lb = [p for p in other if p['type'] == typ and p['active']]
        if len(la) != len(lb):
            out.update({p['slot']: None for p in la})
            continue
        if not la:
            continue
        perm = min(itertools.permutations(range(len(lb))), key=lambda order: sum(_cost(a, lb[k]) for a, k in zip(la, order)))
        out.update({a['slot']: lb[k] for a, k in zip(la, perm)})
    return out


def equivalent(run):
    """Finished solutions other than the best whose chi2 lies within the tolerance of the best."""
    best = run.best
    tol = tolerance(best['chi2'], run.n)
    pool = list(run.tracks)+[c for c in run.candidates if c['stage'] in FINISHED_STAGES]
    seen, out = {id(best)}, []
    for cand in pool:
        if id(cand) in seen:
            continue
        seen.add(id(cand))
        if cand['chi2']-best['chi2'] <= tol:
            out.append(cand)
    return out


def assess(run, labels):
    """{'popN.R_nm' | 'popN.D_nm': dict(value, min, max, n_solutions, n_disagree, activity_mismatch, ambiguous)}
    for the active populations of the best solution; labels: {slot: 'popN'}. Fixed parameters are left out."""
    setup, best = run.setup, run.best
    mine = describe(setup, best)
    others = [(c, describe(setup, c)) for c in equivalent(run)]
    codec = setup.codec
    out = {}
    for pop in mine:
        if not pop['active']:
            continue
        for key, field in (('R', 'R_nm'), ('D', 'D_nm')):
            if pop[key] is None or (hasattr(codec, 'is_fixed') and codec.is_fixed(pop['slot'], key)):
                continue
            out[f"{labels[pop['slot']]}.{field}"] = dict(value=pop[key], min=pop[key], max=pop[key], n_solutions=len(others),
                                                          n_disagree=0, activity_mismatch=0, ambiguous=False)
    for cand, pops in others:
        pairs = match(mine, pops)
        for pop in mine:
            if not pop['active']:
                continue
            partner = pairs.get(pop['slot'])
            for key, field in (('R', 'R_nm'), ('D', 'D_nm')):
                item = out.get(f"{labels[pop['slot']]}.{field}")
                if item is None:
                    continue
                if partner is None:
                    item['activity_mismatch'] += 1
                    item['ambiguous'] = True
                    continue
                value = partner[key]
                item['min'], item['max'] = min(item['min'], value), max(item['max'], value)
                if abs(value/pop[key]-1) > REL_TOL:
                    item['n_disagree'] += 1
                    item['ambiguous'] = True
    return out
