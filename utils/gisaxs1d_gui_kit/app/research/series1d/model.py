"""Model of a series (composition, physics switches, constraints) and the per-frame fit objects.

ModelSpec builds, for every frame, exactly the Setup that research.fit1d.api.fit1d builds for the same
curve (per-curve CurveCodec(q_min, q_max) wrapped by ConstrainedCodec, physics v2 = TabFit number /
hosemann), so a u vector of a fit1d result is valid in the tracker's fit and vice versa.
Conventions: q in nm^-1, lengths in nm, number-weighted size distributions (sigma = relative width),
Hosemann paracrystal with w = w_rel * D; spheres / random cylinders on a Yoneda cut see
|q| = sqrt(q_par^2 + qz^2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from research.cond_v2.codec import CurveCodec
from research.fit1d import constraints as cons
from research.fit1d import schema
from research.fit1d.api import parse_compositions
from research.fit1d.controller import BOUND_TOL, Setup, powerlaw_box, scaled_covariance
from research.multishape_v1.physics import COMBOS

SIZE_FIELDS = (0, 2, 4)                # slot offsets of R, h, D (the size coordinates of a population)
TOLERANCE = (25.0, 0.02)               # PLAN same-basin chi2 tolerance max(25, 0.02 N) max(1, chi2/N)
UNMATCHED_COST = 1e3                   # identity matching: an 'on' population paired with an 'off' one


def tolerance(n, chi2):
    """Chi2 difference below which two solutions of one curve count as equivalent (PLAN_zh.md §1)."""
    return max(TOLERANCE[0], TOLERANCE[1]*n)*max(1.0, chi2/max(n, 1))


@dataclass
class ModelSpec:
    composition: str
    combo: int
    mode: str
    sf_on: bool
    res_on: bool
    lowq: str
    lowq_nu: float | None
    structure_factor: bool | None
    constraints: dict
    qz: float | None                 # None: the vacuum qz of the cut (per frame); SAXS: 0
    sphere_model: str
    seed: int = 0
    model_dir: str | None = None
    auto: bool = False               # candidate list ('auto...'): only for independent curves (no tracking)
    _norm: dict = field(default_factory=dict, repr=False)
    _rules: dict = field(default_factory=dict, repr=False)
    _nu: tuple | None = field(default=None, repr=False)
    _pl: tuple = field(default=(False, None, {}, {}), repr=False)
    _background: float | None = field(default=None, repr=False)
    _codec: dict | None = field(default=None, repr=False)

    @classmethod
    def from_recipe(cls, recipe, independent=False):
        from research.series1d.recipe import RecipeError
        m = recipe['model']
        gisaxs = m['mode'] == 'gisaxs'
        try:
            combos, auto = parse_compositions(m['composition'])
        except ValueError as exc:
            raise RecipeError(f'recipe: model.composition: {exc}') from None
        if auto and not independent:
            # a tracked series needs fixed population slots; candidates are a single-curve question
            raise RecipeError(
                f"recipe: model.composition: {m['composition']!r} asks for candidate compositions; a tracked "
                "series needs one explicit composition (e.g. 'sphere,vc'). Run 'python3 -m research.fit1d' on "
                'a representative frame to see the candidates; a single curve does not identify the '
                "composition (K >= 2). 'auto...' is accepted only for a folder of independent curves.")
        sf_on = gisaxs if m['structure_factor'] is None else bool(m['structure_factor'])
        lowq = m['lowq'] if m['lowq'] is not None else ('lorentz' if gisaxs else 'none')
        res_on = lowq != 'none'
        spec = cls(composition=m['composition'], combo=int(combos[0]), mode=m['mode'], sf_on=sf_on, res_on=res_on,
                   lowq=lowq, lowq_nu=m['lowq_nu'], structure_factor=m['structure_factor'],
                   constraints=m['constraints'], qz=(m['qz'] if gisaxs else 0.0), sphere_model=m['sphere_model'],
                   seed=int(recipe['fitting']['seed']), model_dir=m.get('model_dir'), auto=bool(auto))
        try:
            spec._norm = cons.normalise(_tuples(m['constraints']))
            spec._nu = cons.nu_rule(lowq, m['lowq_nu'], spec._norm) if res_on else None
            if 'nu' in spec._norm['globals'] and lowq != 'lorentz':
                raise cons.ConstraintError("'nu' applies only to lowq='lorentz'")
            pl_rule = spec._norm['globals'].get('powerlaw_n')
            if pl_rule is not None and lowq != 'powerlaw':
                raise cons.ConstraintError("'powerlaw_n' applies only to lowq='powerlaw'")
            spec._pl = powerlaw_box(pl_rule)
            if 'background' in spec._norm['globals']:
                spec._background = spec._norm['globals']['background'][1]
            if not auto:
                rules, unused = cons.slot_rules(spec._norm, spec.types, sf_on)
                if unused:
                    raise cons.ConstraintError(f'{", ".join(unused)}: composition {m["composition"]} has only '
                                               f'{int(np.sum(spec.types > 0))} population(s)')
                spec._rules = rules
                # feasibility of the constraint set on a typical q window (errors do not depend on the window
                # except for the smallest resolvable size, which the per-frame codec reports again)
                cons.ConstrainedCodec(CurveCodec(1e-3, 5.0), spec.types, rules,
                                      nu=spec._nu if lowq == 'lorentz' else None)
        except cons.ConstraintError as exc:
            raise RecipeError(f'recipe: model.constraints: {exc}') from None
        return spec

    # ------------------------------------------------------------------ composition
    @property
    def types(self):
        return COMBOS[self.combo]

    @property
    def slots(self):
        return [j for j in range(4) if self.types[j]]

    @property
    def labels(self):
        """{slot j: 'popN'} in the fit1d population order (type code, then slot)."""
        return {j: label for label, j, _ in cons.population_labels(self.types)}

    def slot_class(self, j):
        """Populations may exchange identities only inside one class: same type and same constraints."""
        rules = self._rules.get(j, {})
        return (int(self.types[j]), tuple(sorted((k, tuple(v)) for k, v in rules.items())))

    def slot_coords(self, j):
        """Nonlinear u indices of slot j that exist for its type (R, sigma_R[, h, sigma_h][, D, w])."""
        offs = [0, 1]+([2, 3] if self.types[j] == 2 else [])+([4, 5] if self.sf_on else [])
        return [7*j+k for k in offs]

    def needs_qz(self):
        return self.mode == 'gisaxs' and bool(np.any(np.isin(self.types, (1, 2))))

    # ------------------------------------------------------------------ per-frame objects
    def codec(self, q_min, q_max):
        """ConstrainedCodec of a q window (one object per window: 'is' identifies equal codecs)."""
        key = (float(q_min), float(q_max))
        if self._codec is None:
            self._codec = {}
        if key not in self._codec:
            if len(self._codec) > 8:
                self._codec.clear()
            self._codec[key] = cons.ConstrainedCodec(CurveCodec(*key), self.types, self._rules,
                                                     nu=self._nu if self.lowq == 'lorentz' else None)
        return self._codec[key]

    def setup(self, q, y, s, qz, frozen=()):
        """controller.Setup of one frame; frozen = extra fixed u indices (populations in state 'off')."""
        codec = self.codec(q[0], q[-1])
        pl_fixed, u31, lower, upper = self._pl
        fixed = list(codec.fixed_indices)
        powerlaw = self.lowq == 'powerlaw'
        if powerlaw and pl_fixed:
            fixed.append(31)
        fixed = tuple(sorted(set(fixed) | {int(c) for c in frozen}))
        return Setup(q, y, s, self.combo, float(qz), self.sf_on, self.res_on, self.lowq, codec, fixed,
                     dict(lower) if powerlaw else {}, dict(upper) if powerlaw else {}, self._background,
                     u31 if powerlaw else None)

    def fit1d_kwargs(self, qz, policy, dwba=None):
        return dict(composition=self.composition, mode=self.mode, qz=float(qz) if self.mode == 'gisaxs' else 0.0,
                    structure_factor=self.structure_factor, lowq=self.lowq, lowq_nu=self.lowq_nu,
                    constraints=_tuples(self.constraints), policy=policy, seed=self.seed, workers=1,
                    model_dir=self.model_dir, dwba=dwba)

    def convert(self, u, codec_from, codec_to):
        """u of one codec expressed in another through physical parameters (same object: unchanged copy)."""
        u = np.array(u, float)
        if codec_from is codec_to or (codec_from.q_min == codec_to.q_min and codec_from.q_max == codec_to.q_max):
            return u
        out = codec_to.encode(codec_from.decode(u))
        if self.lowq == 'powerlaw':
            out[31] = u[31]                      # the power-law exponent coordinate does not depend on the codec
        return out


def _tuples(constraints):
    """JSON lists back to (lo, hi) tuples (fit1d accepts both; tuples keep its error messages exact)."""
    out = {}
    for k, v in (constraints or {}).items():
        if isinstance(v, dict):
            out[k] = {kk: (tuple(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}
        else:
            out[k] = tuple(v) if isinstance(v, list) else v
    return out


# ---------------------------------------------------------------------------- solution helpers
def k_params(setup, coef):
    """Free nonlinear coordinates + positive linear coefficients (fit1d CompositionRun.k_params)."""
    coef = np.asarray(coef, float)
    linear = int(np.sum(coef[:4] > 0))+int(coef[5] > 0 and setup.res_on)
    if setup.background is None:
        linear += int(coef[4] > 0)
    return len(setup.free_coords())+linear


def at_bound(setup, u, coef):
    """[(u index, 'lower'|'upper')] of free coordinates of contributing terms on an edge of their box."""
    out = []
    for c in setup.free_coords():
        if (c >= 28 and not coef[5] > 0) or (c < 28 and not coef[c//7] > 0):
            continue
        lo, hi = setup.lower.get(c, 0.0), setup.upper.get(c, 1.0)
        x = (u[c]-lo)/max(hi-lo, 1e-300)
        if x < BOUND_TOL:
            out.append((int(c), 'lower'))
        elif x > 1-BOUND_TOL:
            out.append((int(c), 'upper'))
    return out


def coordinate_name(index, spec):
    if index >= 28:
        return {30: 'sigma_res', 31: 'powerlaw_n' if spec.lowq == 'powerlaw' else 'nu'}.get(index, str(index))
    field_ = ('R_nm', 'sigma_R_rel', 'h_nm', 'sigma_h_rel', 'D_nm', 'w_rel', 'A')[index % 7]
    return f'{spec.labels.get(index//7, index//7)}.{field_}'


def populations(spec, setup, r):
    """fit1d population dicts (values + 1-sigma) of an lm_run result, in slot order."""
    chi2 = float(r['chi2'])
    dof = max(len(setup.y)-k_params(setup, r['coef']), 1)
    cov = scaled_covariance(r.get('cov_u'), chi2, dof)
    return schema.populations(setup.codec, setup.types, np.asarray(r['u'], float), np.asarray(r['coef'], float),
                              r['coords'] if cov is not None else [], cov, setup.sf_on)


def low_q(spec, setup, u, coef):
    if not spec.res_on:
        return dict(form='none', C=None)
    active = bool(coef[5] > 0)
    if spec.lowq == 'powerlaw':
        from research.conditional_fit.fastfit import powerlaw_exponent
        return dict(form='powerlaw', C=float(coef[5]), powerlaw_n=float(powerlaw_exponent(u)) if active else None)
    p = setup.codec.decode(np.asarray(u, float))
    return dict(form='lorentz', C=float(coef[5]), sigma_res_nm_inv=float(p['sigma_res']) if active else None,
                nu=float(p['nu_res']))


def permute_result(r, perm):
    """LM result with its population slots relabelled: perm[target slot] = source slot.
    u blocks and amplitudes move, the coordinate list of the covariance is renamed (order unchanged)."""
    if all(t == s for t, s in perm.items()):
        return r
    u, coef = np.array(r['u'], float), np.array(r['coef'], float)
    u2, c2 = u.copy(), coef.copy()
    target_of = {}
    for target, source in perm.items():
        u2[7*target:7*target+7] = u[7*source:7*source+7]
        c2[target] = coef[source]
        target_of[source] = target
    coords = [7*target_of.get(c//7, c//7)+c % 7 if c < 28 else c for c in r['coords']]
    return dict(r, u=u2, coef=c2, coords=coords)


def identity_points(spec, codec, u, coef=None):
    """{slot: (ln R, ln D or 0, has_values)} of a solution; has_values False for amplitude 0 (coef given)."""
    p = codec.decode(np.asarray(u, float))
    out = {}
    for j in spec.slots:
        has = True if coef is None else bool(coef[j] > 0)
        out[j] = (float(np.log(p['R'][j])), float(np.log(p['D'][j])) if spec.sf_on else 0.0, has)
    return out


def match_identities(spec, refs, news):
    """Hungarian matching of the populations of a new solution to reference populations.

    refs, news: lists (one entry per compared frame) of identity_points dicts; the cost of giving reference
    slot a the new slot b is the sum over the frames of (d ln R)^2 + (d ln D)^2 (D only with the structure
    factor); pairing a population that has values with one that has none (amplitude 0) costs UNMATCHED_COST,
    two populations without values cost nothing. Only slots of one class (same type and
    same constraints) can be exchanged, and the identity is kept unless another assignment is strictly
    better. Returns perm {reference slot: new slot}."""
    from scipy.optimize import linear_sum_assignment
    perm = {j: j for j in spec.slots}
    classes = {}
    for j in spec.slots:
        classes.setdefault(spec.slot_class(j), []).append(j)
    for members in classes.values():
        if len(members) < 2:
            continue
        cost = np.zeros((len(members), len(members)))
        for ref, new in zip(refs, news):
            for a, ja in enumerate(members):
                for b, jb in enumerate(members):
                    ra, rb = ref[ja], new[jb]
                    if ra[2] and rb[2]:
                        cost[a, b] += (ra[0]-rb[0])**2+(ra[1]-rb[1])**2
                    elif ra[2] != rb[2]:
                        cost[a, b] += UNMATCHED_COST
        if not cost.any():
            continue
        rows, cols = linear_sum_assignment(cost)
        if cost[rows, cols].sum() < np.trace(cost)-1e-12:
            for a, b in zip(rows, cols):
                perm[members[a]] = members[b]
    return perm


def canonical_order(spec, codec, u, coef):
    """perm {target slot: source slot} that sorts the populations of one class by R (ascending), populations
    with amplitude 0 last: the slot order of the first frame of a series."""
    p = codec.decode(np.asarray(u, float))
    perm = {j: j for j in spec.slots}
    classes = {}
    for j in spec.slots:
        classes.setdefault(spec.slot_class(j), []).append(j)
    for members in classes.values():
        ordered = sorted(members, key=lambda j: (not coef[j] > 0, float(p['R'][j]), j))
        for target, source in zip(members, ordered):
            perm[target] = source
    return perm
