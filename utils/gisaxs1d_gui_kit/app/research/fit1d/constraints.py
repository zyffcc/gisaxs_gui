"""Parameter constraints in physical units (nm, nm^-1, relative widths) for fit1d.

User syntax (Python):
    constraints = {
        'pop1': {'R_nm': 2.5},                 # fix: a number
        'pop2': {'D_nm': (5, 8)},              # range: (lo, hi); lo == hi is a fix
        'all':  {'w_rel': (0.05, 0.3)},        # every population where the key applies
        'nu': 2.0,                             # low-q Lorentzian exponent (lowq='lorentz')
        'powerlaw_n': (2, 4),                  # low-q power-law exponent (lowq='powerlaw'), within [1, 4]
        'background': 0.0,                     # constant background B fixed (value only; 0 = no background)
    }
CLI syntax: --fix pop1.R_nm=2.5  --range pop2.D_nm=5:8  --fix nu=2  --fix background=0.

Population keys: R_nm, sigma_R_rel, D_nm, w_rel (all shapes), h_nm, sigma_h_rel (random cylinders only).
D_nm and w_rel need the structure factor (GISAXS default; SAXS only with structure_factor=True).
Populations 'pop1', 'pop2', ... are numbered in the canonical order of the composition: by type code
(1 sphere, 2 random cylinder, 3 vertical cylinder), then by slot, as in multishape_v1.physics.COMBOS
(e.g. 'vc,sphere,vc' -> pop1 sphere, pop2 vc, pop3 vc). In auto mode popN applies to every candidate
composition that has an N-th population. Not supported: ranges on the background, constraints on
amplitudes, on the low-q width sigma_res or on the low-q amplitude C, and a fitted qz.

Implementation: ConstrainedCodec wraps the per-curve research.cond_v2.codec.CurveCodec and exposes the
same decode/encode/lo_hi/name interface, so FastFit/TabFit accept it unchanged.
  * a range re-maps that coordinate u in [0, 1] onto its own box (log for R, h and D; linear for
    relative widths and nu); decoded values are clipped into the box, so results are always within it;
  * a fix gives the quantity zero width: decode returns the given float itself (bitwise unchanged)
    and the u index is passed to FastFit.run as fixed (fixed_indices);
  * a constrained D is encoded absolutely in nm: D = L (D_hi/L)^u with L = max(D_lo, 2.002 R), so
    D stays in [D_lo, D_hi] exactly while R is free and D >= 2.002 R (no overlap) always holds; the R
    box of that population is narrowed to R <= D_hi/2.002 (reported as a derived bound). Constraint
    sets that cannot satisfy D >= 2.002 R raise ConstraintError with an explanation.
Unconstrained coordinates decode bitwise identically to the base CurveCodec.
"""
from __future__ import annotations

import numpy as np

from research.cond_v2.codec import D_MAX_RATIO, NU, W_REL, WIDTH
from research.conditional_fit.fastfit import POWERLAW_N

TYPE_NAMES = {1: 'sphere', 2: 'random_cylinder', 3: 'vertical_cylinder'}
# user key -> (slot offset, decode field, mapping)
SLOT_PARAMS = {'R_nm': (0, 'R', 'log'), 'sigma_R_rel': (1, 'sigma_R', 'lin'), 'h_nm': (2, 'h', 'log'),
               'sigma_h_rel': (3, 'sigma_h', 'lin'), 'D_nm': (4, 'D', 'dist'), 'w_rel': (5, 'sigma_D', 'lin')}
FIELD_TO_KEY = {v[1]: k for k, v in SLOT_PARAMS.items()}
GLOBAL_PARAMS = ('nu', 'powerlaw_n', 'background')
# accepted physical limits of user values (table range for relative widths, FastFit range for n)
LIMITS = {'R_nm': (1e-3, 1e6), 'h_nm': (1e-3, 1e6), 'D_nm': (1e-3, 1e7), 'sigma_R_rel': (0.02, 0.6),
          'sigma_h_rel': (0.02, 0.6), 'w_rel': (1e-3, 1.0), 'nu': (0.5, 12.0), 'powerlaw_n': POWERLAW_N,
          'background': (0.0, np.inf)}
NONOVERLAP = 2.002        # D >= 2.002 R, as in the base codec


class ConstraintError(ValueError):
    """Invalid or infeasible constraint set."""


# ---------------------------------------------------------------- parsing
def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ConstraintError(f'not a number: {text!r}') from None


def normalise_value(value, key='value'):
    """('fixed', v) or ('range', lo, hi) from a number, a (lo, hi) pair, 'lo:hi' or {'fixed'|'min','max'}."""
    if isinstance(value, dict):
        if 'fixed' in value:
            return normalise_value(value['fixed'], key)
        return normalise_value((value.get('min'), value.get('max')), key)
    if isinstance(value, str):
        value = tuple(value.split(':')) if ':' in value else value
    if isinstance(value, (tuple, list, np.ndarray)):
        if len(value) != 2:
            raise ConstraintError(f'{key}: a range needs exactly two values (lo, hi), got {value!r}')
        lo, hi = (_number(v) for v in value)
        if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
            raise ConstraintError(f'{key}: invalid range ({lo}, {hi}); need finite lo <= hi')
        return ('fixed', lo) if lo == hi else ('range', lo, hi)
    v = _number(value)
    if not np.isfinite(v):
        raise ConstraintError(f'{key}: value must be finite, got {v}')
    return ('fixed', v)


def _check_limits(key, rule):
    lo, hi = LIMITS[key]
    values = rule[1:]
    if any(v < lo or v > hi for v in values):
        raise ConstraintError(f'{key}={values}: allowed values are within [{lo}, {hi}]')


def normalise(constraints):
    """Canonical form {'populations': {'all'|int N: {key: rule}}, 'globals': {key: rule}} (rules as tuples)."""
    out = dict(populations={}, globals={})
    if not constraints:
        return out
    if not isinstance(constraints, dict):
        raise ConstraintError('constraints must be a dict, e.g. {"pop1": {"R_nm": 2.5}}')
    for name, body in constraints.items():
        name = str(name).strip()
        if name in GLOBAL_PARAMS:
            rule = normalise_value(body, name)
            _check_limits(name, rule)
            if name == 'background' and rule[0] != 'fixed':
                raise ConstraintError('background: only a fixed value is supported (e.g. 0), not a range')
            out['globals'][name] = rule
            continue
        if name == 'all':
            target = 'all'
        elif name.startswith('pop') and name[3:].isdigit() and int(name[3:]) >= 1:
            target = int(name[3:])
        else:
            raise ConstraintError(f'unknown constraint target {name!r}: use pop1, pop2, ..., all, '
                                  f'or one of {GLOBAL_PARAMS}')
        if not isinstance(body, dict):
            raise ConstraintError(f'{name}: expected a dict of parameters, e.g. {{"R_nm": 2.5}}')
        rules = {}
        for key, value in body.items():
            if key not in SLOT_PARAMS:
                raise ConstraintError(f'{name}.{key}: unknown parameter; use one of {tuple(SLOT_PARAMS)}')
            rule = normalise_value(value, f'{name}.{key}')
            _check_limits(key, rule)
            rules[key] = rule
        out['populations'][target] = rules
    return out


def parse_cli(fixes=(), ranges=()):
    """Constraint dict from CLI strings: fixes 'pop1.R_nm=2.5', 'nu=2'; ranges 'pop2.D_nm=5:8'."""
    out = {}
    for text, is_range in [(t, False) for t in fixes or ()]+[(t, True) for t in ranges or ()]:
        if '=' not in text:
            raise ConstraintError(f'{text!r}: expected name=value (e.g. pop1.R_nm=2.5 or pop2.D_nm=5:8)')
        name, value = (s.strip() for s in text.split('=', 1))
        if is_range:
            if ':' not in value:
                raise ConstraintError(f'--range {text!r}: expected lo:hi')
            value = tuple(value.split(':', 1))
        elif ':' in value:
            raise ConstraintError(f'--fix {text!r}: a fix takes one value (use --range for lo:hi)')
        if '.' in name:
            target, key = name.split('.', 1)
            out.setdefault(target, {})[key] = value
        else:
            out[name] = value
    return out


def rule_json(rule):
    return {'fixed': rule[1]} if rule[0] == 'fixed' else {'range': [rule[1], rule[2]]}


def population_labels(types):
    """[(label 'popN', slot j, type code)] in canonical order (COMBOS rows are already sorted by type)."""
    out = []
    for j, typ in enumerate(types):
        if int(typ):
            out.append((f'pop{len(out)+1}', j, int(typ)))
    return out


def slot_rules(normalised, types, sf_on):
    """({slot j: {key: rule}}, [unused popN labels]) for one composition; explicit popN rules override 'all'.

    popN with N larger than the number of populations is returned as unused (an error for a single
    composition, skipped for that candidate in auto mode)."""
    pops = population_labels(types)
    rules = {}
    specific = {k: v for k, v in normalised['populations'].items() if k != 'all'}
    unused = [f'pop{n}' for n in sorted(specific) if n > len(pops)]
    for index, (label, j, typ) in enumerate(pops, 1):
        merged = {}
        for key, rule in normalised['populations'].get('all', {}).items():
            if key in ('h_nm', 'sigma_h_rel') and typ != 2:
                continue
            merged[key] = rule
        for key, rule in specific.get(index, {}).items():
            if key in ('h_nm', 'sigma_h_rel') and typ != 2:
                raise ConstraintError(f'{label}.{key}: height parameters exist only for random cylinders '
                                      f'({label} is a {TYPE_NAMES[typ]})')
            merged[key] = rule
        if not sf_on and any(k in merged for k in ('D_nm', 'w_rel')):
            raise ConstraintError(f'{label}: D_nm / w_rel constraints need the structure factor '
                                  '(structure_factor=True)')
        if merged:
            rules[j] = merged
    return rules, unused


# ---------------------------------------------------------------- codec
class ConstrainedCodec:
    """CurveCodec with per-population fixes/ranges in physical units (see module docstring)."""

    def __init__(self, base, types, rules=None, nu=None, name=None):
        """base: CurveCodec; types: the 4 type codes; rules: {slot j: {key: rule}}; nu: rule or None (free)."""
        self.base = base
        self.types = tuple(int(t) for t in types)
        lo, hi = base.lo_hi
        self._lo, self._span = lo, hi-lo
        self.name = name or f'{base.name}+constraints'
        self.q_min, self.q_max = base.q_min, base.q_max
        self.rules = {int(j): dict(r) for j, r in (rules or {}).items()}
        self.nu = nu
        self.derived = {}                  # {slot j: {'R_nm': (lo, hi)}} bounds implied by a D constraint
        self._plan = {}                    # {slot j: [(field, kind, a, b, mapping), ...]} R first, D last
        size_lo, size_hi = base.size_range
        labels = {j: label for label, j, _ in population_labels(self.types)}
        for j, rules_j in self.rules.items():
            steps = []
            r_rule = rules_j.get('R_nm')
            r_lo, r_hi = (r_rule[1], r_rule[1]) if r_rule and r_rule[0] == 'fixed' else \
                ((r_rule[1], r_rule[2]) if r_rule else (size_lo, size_hi))
            d_rule = rules_j.get('D_nm')
            if d_rule is not None:
                d_hi = d_rule[-1]
                limit = d_hi/NONOVERLAP
                if r_lo > limit*(1+1e-12):
                    what = f'R_nm = {r_lo:g}' if r_rule and r_rule[0] == 'fixed' else f'R_nm >= {r_lo:g}'
                    raise ConstraintError(
                        f'{labels[j]}: infeasible constraints: D_nm <= {d_hi:g} needs R_nm <= {limit:g} '
                        f'(non-overlap D >= 2.002 R), but {what}' +
                        ('' if r_rule else f' (smallest size this q range resolves: 0.5/q_max = {size_lo:g} nm)'))
                if r_hi > limit:
                    r_hi = limit
                    if r_rule is None or r_rule[0] == 'range':
                        self.derived.setdefault(j, {})['R_nm'] = (r_lo, r_hi)
            if r_rule is not None and r_rule[0] == 'fixed':
                steps.append(('R', 'fixed', r_rule[1], r_rule[1], 'log'))
            elif r_rule is not None or j in self.derived:
                steps.append(('R', 'range', r_lo, r_hi, 'log'))
            for key in ('sigma_R_rel', 'h_nm', 'sigma_h_rel', 'w_rel'):
                if key in rules_j:
                    rule = rules_j[key]
                    a, b = (rule[1], rule[1]) if rule[0] == 'fixed' else (rule[1], rule[2])
                    steps.append((SLOT_PARAMS[key][1], rule[0], a, b, SLOT_PARAMS[key][2]))
            if d_rule is not None:
                a, b = (d_rule[1], d_rule[1]) if d_rule[0] == 'fixed' else (d_rule[1], d_rule[2])
                steps.append(('D', d_rule[0], a, b, 'dist'))
            self._plan[j] = steps

    # -- interface used by FastFit / TabFit ---------------------------------
    @property
    def lo_hi(self):
        """Box of the base codec (valid for unconstrained coordinates only)."""
        return self._lo.copy(), self._lo+self._span

    @property
    def size_range(self):
        return self.base.size_range

    @property
    def fixed_indices(self):
        out = [7*j+SLOT_PARAMS[FIELD_TO_KEY[f]][0] for j, steps in self._plan.items()
               for f, kind, *_ in steps if kind == 'fixed']
        if self.nu is not None and self.nu[0] == 'fixed':
            out.append(31)
        return sorted(out)

    def decode(self, u):
        u = np.asarray(u, float)
        z = self._lo+u*self._span                                   # == CurveCodec.decode arithmetic
        slot = z[..., :28].reshape(*z.shape[:-1], 4, 7)
        r = np.exp(slot[..., 0])
        distance = 2.002*r*D_MAX_RATIO**slot[..., 4]
        ones = np.ones(z.shape[:-1])
        p = dict(R=r, sigma_R=slot[..., 1], h=np.exp(slot[..., 2]), sigma_h=slot[..., 3], D=distance,
                 sigma_D=slot[..., 5], A=np.ones_like(r), B=ones, C=ones,
                 sigma_res=np.exp(z[..., 30]), nu_res=z[..., 31])
        for j, steps in self._plan.items():
            r_changed = False
            for field, kind, a, b, mapping in steps:
                k = SLOT_PARAMS[FIELD_TO_KEY[field]][0]
                x = u[..., 7*j+k]
                if kind == 'fixed':
                    value = a
                elif mapping == 'log':
                    value = np.clip(np.exp(np.log(a)+x*(np.log(b)-np.log(a))), a, b)
                elif mapping == 'lin':
                    value = np.clip(a+x*(b-a), a, b)
                else:                                                # absolute distance in nm
                    lower = np.minimum(np.maximum(a, NONOVERLAP*p['R'][..., j]), b)
                    value = np.minimum(np.maximum(lower*(b/lower)**x, lower), b)
                if field == 'R':
                    p['R'] = p['R'].copy()
                    r_changed = True
                elif field == 'D':
                    p['D'] = p['D'].copy()
                elif field in ('sigma_R', 'sigma_h', 'sigma_D', 'h'):
                    p[field] = np.array(p[field], copy=True)
                p[field][..., j] = value
            if r_changed and not any(s[0] == 'D' for s in steps):
                p['D'] = p['D'].copy()
                p['D'][..., j] = 2.002*p['R'][..., j]*D_MAX_RATIO**slot[..., j, 4]
        if self.nu is not None:
            if self.nu[0] == 'fixed':
                p['nu_res'] = np.full(z.shape[:-1], self.nu[1])
            else:
                a, b = self.nu[1], self.nu[2]
                p['nu_res'] = np.clip(a+u[..., 31]*(b-a), a, b)
        return p

    def encode(self, p):
        """u of physical parameters p (dict as returned by decode), projected into the constrained box."""
        p = {k: np.array(v, float, copy=True) for k, v in p.items()}
        for j, steps in self._plan.items():
            for field, kind, a, b, mapping in steps:
                if field == 'D':
                    lower = np.maximum(a, NONOVERLAP*p['R'][..., j])
                    p['D'][..., j] = a if kind == 'fixed' else np.minimum(np.maximum(p['D'][..., j], lower), b)
                else:
                    p[field][..., j] = a if kind == 'fixed' else np.clip(p[field][..., j], a, b)
        if self.nu is not None:
            p['nu_res'] = np.full(np.shape(p['nu_res']), self.nu[1]) if self.nu[0] == 'fixed' else \
                np.clip(p['nu_res'], self.nu[1], self.nu[2])
        u = self.base.encode(p)
        for j, steps in self._plan.items():
            for field, kind, a, b, mapping in steps:
                k = SLOT_PARAMS[FIELD_TO_KEY[field]][0]
                v = p[field][..., j]
                if kind == 'fixed':
                    frac = 0.0
                elif mapping == 'log':
                    frac = (np.log(v)-np.log(a))/(np.log(b)-np.log(a)) if b > a else 0.0
                elif mapping == 'lin':
                    frac = (v-a)/(b-a) if b > a else 0.0
                else:
                    lower = np.minimum(np.maximum(a, NONOVERLAP*p['R'][..., j]), b)
                    span = np.log(b/lower)
                    frac = np.where(span > 0, np.log(np.maximum(v, lower)/lower)/np.where(span > 0, span, 1.0), 0.0)
                u[..., 7*j+k] = frac
        if self.nu is not None:
            u[..., 31] = 0.0 if self.nu[0] == 'fixed' or self.nu[2] <= self.nu[1] else \
                (p['nu_res']-self.nu[1])/(self.nu[2]-self.nu[1])
        return np.clip(u, 0.0, 1.0)

    # -- reporting -------------------------------------------------------------
    def bounds(self, j, field, p=None):
        """Physical (lo, hi) box of one quantity of slot j; D bounds depend on R (pass decoded p)."""
        for f, kind, a, b, mapping in self._plan.get(j, ()):
            if f == field:
                if field == 'D':
                    r = p['R'][j] if p is not None else np.nan
                    return (a, b) if kind == 'fixed' else (max(a, NONOVERLAP*r), b)
                return (a, b)
        if field in ('R', 'h'):
            return self.base.size_range
        if field in ('sigma_R', 'sigma_h'):
            return WIDTH
        if field == 'sigma_D':
            return W_REL
        if field == 'D':
            r = p['R'][j] if p is not None else np.nan
            return (2.002*r, 2.002*r*D_MAX_RATIO)
        raise KeyError(field)

    def is_fixed(self, j, field):
        return any(f == field and kind == 'fixed' for f, kind, *_ in self._plan.get(j, ()))

    def describe(self, j):
        """JSON-friendly constraint summary of slot j ({} when unconstrained)."""
        out = {key: rule_json(rule) for key, rule in self.rules.get(j, {}).items()}
        for key, (a, b) in self.derived.get(j, {}).items():
            out[f'{key}_derived_range'] = [a, b]
        return out


def nu_rule(lowq, lowq_nu, normalised):
    """Rule for the Lorentzian exponent: a 'nu' constraint wins over lowq_nu; lowq_nu None = free in NU."""
    if lowq != 'lorentz':
        if 'nu' in normalised['globals']:
            raise ConstraintError("'nu' applies only to lowq='lorentz'")
        return None
    if 'nu' in normalised['globals']:
        return normalised['globals']['nu']
    if lowq_nu is None:
        return ('range', NU[0], NU[1])
    v = float(lowq_nu)
    if not LIMITS['nu'][0] <= v <= LIMITS['nu'][1]:
        raise ConstraintError(f'lowq_nu={v}: allowed values are within {LIMITS["nu"]}')
    return ('fixed', v)
