"""fit1d: fastest accurate fit of a 1D GISAXS Yoneda cut (or SAXS curve) for a given composition.

    from research.fit1d import fit1d, to_csv, plot
    result = fit1d(q, I, sigma, composition='sphere,vc,vc', mode='gisaxs', qz=0.5)
    result['results'][0]['populations']      # physical parameters with 1-sigma errors

Network proposals (cond_v2, 8 heads, NNLS) -> a few LM steps -> multi-start robust search only when
state signals say the fast result is unreliable (research.fit1d.controller). Physics v2 only (see
schema.CONVENTIONS). Composition: 'vc,vc', 'sphere,vc', ... or 'auto:vc:3', 'auto:sphere:3', 'auto'
(candidate compositions ranked by BIC; those within delta_bic of the lowest are marked
indistinguishable: a single curve does not determine the composition for K >= 2).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from research.cond_v2.codec import CurveCodec
from research.cond_v2.model import cond_vector, features
from research.conditional_fit.fastfit import powerlaw_exponent
from research.conditional_fit.pipeline import _clean_inputs, network_view
from research.conditional_fit.tool import parse_composition
from research.fit1d import constraints as cons
from research.fit1d import ensemble
from research.fit1d import flags as fl
from research.fit1d import schema
from research.fit1d.controller import (CompositionRun, Setup, effective_workers, powerlaw_box, propagate,
                                       run_jobs, scaled_covariance)
from research.multishape_v1.physics import COMBOS

DEFAULT_MODEL_DIR = schema.WORK_DIR/'models'/'cond_v2_a'
POLICIES = ('fast', 'auto', 'robust', 'preview')
PROFILE_POLISH = {'fast': 8, 'auto': 12, 'robust': 16}     # LM evaluations per profile alternative (stage P)
PROFILE_BUDGET = {'fast': 3, 'auto': 8, 'robust': 24}      # stage P starts refined per round (best true chi2 first)
# exchanges refined per round, LM evaluations each. Measured for fast on 281 DEV curves (2026-10-06): one refined
# exchange against none = chi2_ok 78.3 against 76.9 %, silent failures 21.4 against 24.2 %, median time 320 against
# 295 ms (p95 821 against 672 ms); exchanges that are equivalent as they stand are reported in any case
PROFILE_EXCHANGE = {'fast': (1, 16), 'auto': (4, 16), 'robust': (None, 24)}
MODES = ('gisaxs', 'saxs')
LOWQ = ('lorentz', 'powerlaw', 'none')
_FITTERS: dict = {}
_THREADS = {}


def get_fitter(model_dir=None):
    """Process-wide FastFitterV2 per model directory (weights loaded once)."""
    path = Path(model_dir or DEFAULT_MODEL_DIR).resolve()
    key = str(path)
    if key not in _FITTERS:
        if not (path/'weights_numpy.npz').exists():
            raise FileNotFoundError(f'fit1d: no weights_numpy.npz in {path} (cond_v2 model directory expected)')
        from research.cond_v2.inference import FastFitterV2
        _FITTERS[key] = FastFitterV2(key)
    return _FITTERS[key]


def single_thread():
    """Context manager limiting BLAS to one thread (multi-threaded BLAS makes these small LM problems ~8x slower
    and is a source of run-to-run differences)."""
    if 'controller' not in _THREADS:
        try:
            from threadpoolctl import ThreadpoolController
            _THREADS['controller'] = ThreadpoolController()
        except ImportError:          # pragma: no cover
            _THREADS['controller'] = None
    controller = _THREADS['controller']
    if controller is None:          # pragma: no cover
        from contextlib import nullcontext
        return nullcontext()
    return controller.limit(limits=1)


def composition_text(combo):
    return ','.join(schema.SHORT[int(t)] for t in COMBOS[int(combo)] if t)


def network_heads(fitter, q, y, s, combos, qz, flag):
    """cond_v2 heads u[len(combos), 8, 32] in CurveCodec(q.min(), q.max()) (same inputs as FastFitterV2.propose)."""
    vq, vy, vs = network_view(q, y, s)
    pts, m = features(vq[None], vy[None], vs[None], np.ones((1, len(vq)), bool), q_max=[q.max()])
    cond = cond_vector([qz], [flag], [q.min()], [q.max()])
    combos = np.asarray(combos, int)
    return fitter.net(np.repeat(pts, len(combos), 0), np.repeat(m, len(combos), 0), combos,
                      np.repeat(cond, len(combos), 0))


def _reportable(run, cand):
    """R_nm and D_nm (+1-sigma; inf = unconstrained, None = not available) per population of one candidate,
    for the heads_disagree signal and the alternative-solution check."""
    setup = run.setup
    codec, types = setup.codec, setup.types
    slots = [j for j in range(4) if types[j]]
    cov = scaled_covariance(cand['cov_u'], cand['chi2'], run.dof(cand))

    def func(u):
        p = codec.decode(u)
        vals = [p['R'][j] for j in slots]
        if setup.sf_on:
            vals += [p['D'][j] for j in slots]
        return np.array(vals, float)

    values, errors = propagate(func, np.asarray(cand['u'], float), cand['coords'], cov)
    labels = {j: label for label, j, _ in cons.population_labels(types)}
    err = lambda e: None if np.isnan(e) else float(e)          # inf = unconstrained direction (kept)
    out = []
    for i, j in enumerate(slots):
        item = dict(pop=labels[j], type=int(types[j]), amplitude=float(cand['coef'][j]), R_nm=float(values[i]),
                    R_nm_err=err(errors[i]))
        if setup.sf_on:
            k = len(slots)+i
            item.update(D_nm=float(values[k]), D_nm_err=err(errors[k]))
        if codec.is_fixed(j, 'R'):
            item['R_nm_err'] = None
        if setup.sf_on and codec.is_fixed(j, 'D'):
            item['D_nm_err'] = None
        out.append(item)
    return out


def _coordinate_name(index, setup):
    if index >= 28:
        return {30: 'sigma_res', 31: 'powerlaw_n' if setup.lowq == 'powerlaw' else 'nu'}.get(index, str(index))
    labels = {j: label for label, j, _ in cons.population_labels(setup.types)}
    field = ('R_nm', 'sigma_R_rel', 'h_nm', 'sigma_h_rel', 'D_nm', 'w_rel', 'A')[index % 7]
    return f'{labels.get(index//7, index//7)}.{field}'


def _alternatives_summary(alternatives):
    """{'popN.key': value, alternative_min/max, n_alternatives, max_n_sigma, min_delta_chi2} per parameter."""
    out = {}
    for a in alternatives:
        s = out.setdefault(a['parameter'], dict(value=a['value'], alternative_min=a['alternative'],
                                                alternative_max=a['alternative'], n_alternatives=0,
                                                max_n_sigma=0.0, min_delta_chi2=a['delta_chi2']))
        s['alternative_min'] = min(s['alternative_min'], a['alternative'])
        s['alternative_max'] = max(s['alternative_max'], a['alternative'])
        s['n_alternatives'] += 1
        s['max_n_sigma'] = max(s['max_n_sigma'], a['n_sigma'])
        s['min_delta_chi2'] = min(s['min_delta_chi2'], a['delta_chi2'])
    return out


def _low_q_block(setup, fit, u, coef, p, powerlaw_rule):
    """low_q entry of a result. With C = 0 the term is absent: its fitted shape values are not constrained
    by the data and are reported as null (the optimiser's last values go to 'unconstrained_values')."""
    if not setup.res_on:
        return dict(form='none', C=None)
    active = bool(coef[5] > 0)
    if setup.lowq == 'powerlaw':
        fixed = 31 in setup.fixed
        n_exp = powerlaw_rule[1] if powerlaw_rule is not None and powerlaw_rule[0] == 'fixed' else powerlaw_exponent(u)
        box = [powerlaw_exponent(np.r_[np.zeros(31), setup.lower.get(31, 0.0)]),
               powerlaw_exponent(np.r_[np.zeros(31), setup.upper.get(31, 1.0)])]
        low_q = dict(form='C*(max(q,q_min)/q_min)^-n', C=float(coef[5]),
                     powerlaw_n=float(n_exp) if active or fixed else None, q_min_nm_inv=fit.q_ref,
                     powerlaw_n_fixed=bool(fixed), powerlaw_n_bounds=box, inactive=not active)
        if not active and not fixed:
            low_q['unconstrained_values'] = dict(powerlaw_n=float(n_exp))
    else:
        nu_fixed = setup.codec.nu is not None and setup.codec.nu[0] == 'fixed'
        low_q = dict(form='C/(1+(q/sigma_res)^nu)', C=float(coef[5]),
                     sigma_res_nm_inv=float(p['sigma_res']) if active else None,
                     nu=float(p['nu_res']) if active or nu_fixed else None, nu_fixed=bool(nu_fixed),
                     inactive=not active)
        if not active:
            low_q['unconstrained_values'] = dict(sigma_res_nm_inv=float(p['sigma_res']), nu=float(p['nu_res']))
    if not active:
        low_q['note'] = 'C = 0: the low-q term does not contribute; its shape values are not constrained (null)'
    return low_q


def _composition_result(run, settings, ood):
    """Schema entry of one composition from its best candidate."""
    setup, fit = run.setup, run.fit
    cand = run.best
    u = np.asarray(cand['u'], float)
    coef = np.asarray(cand['coef'], float)
    m = fit.matrix_from(*fit.full(u))
    fit._armed = False
    prediction = m@coef
    residual = (prediction-setup.y)/setup.s
    chi2 = float(residual@residual)
    n = len(setup.y)
    k = run.k_params(cand)
    dof = max(n-k, 1)
    columns = m*coef[None, :]
    total_sum = float(prediction.sum())
    cov = scaled_covariance(cand['cov_u'], chi2, dof)
    has_errors = cov is not None
    pops = schema.populations(setup.codec, setup.types, u, coef, cand['coords'] if has_errors else [], cov,
                              setup.sf_on, components_sum=columns.sum(0), total_sum=total_sum)
    components = {}
    for pop in pops:
        components[f"{pop['pop']}_{pop['type']}"] = columns[:, pop['slot']]
    components['background_B'] = columns[:, 4]
    if setup.res_on:
        components['low_q_term'] = columns[:, 5]
    p = setup.codec.decode(u)
    low_q = _low_q_block(setup, fit, u, coef, p, settings['_powerlaw_rule'])
    # bounds of terms that do not contribute (amplitude 0) carry no information: not listed
    at_bound = [f'{_coordinate_name(c, setup)}@{side}' for c, side in run.at_bound(cand)
                if (c >= 28 and coef[5] > 0) or (c < 28 and setup.types[c//7] and coef[c//7] > 0)]
    alternatives = run.alternatives(lambda c: _reportable(run, c)) if has_errors else []
    alt_summary = _alternatives_summary(alternatives)
    preview = settings['_policy'] == 'preview'
    # ambiguity over the chi2-equivalent solutions of this fit (10 % rule, research.fit1d.ensemble); not after the
    # DWBA stage (the other solutions were found with the Born model: their chi2 is not comparable)
    labels = {j: label for label, j, _ in cons.population_labels(setup.types)}
    ambiguity = ensemble.assess(run, labels) if (has_errors and run.born is None) else {}
    ambiguous = [name for name, item in ambiguity.items() if item['ambiguous']]
    for pop in pops:
        for key in ('R_nm', 'D_nm'):
            item = ambiguity.get(f"{pop['pop']}.{key}")
            pop[f'{key}_range'] = None if item is None else [item['min'], item['max']]
    weak, weak_detail = fl.weakly_constrained(pops, at_bound, has_uncertainties=not preview,
                                              multimodal=set(alt_summary) | set(ambiguous))
    quality = fl.quality(chi2, n, residual, prediction, setup.s, settings['_stratum'])
    flags = dict(at_bound=at_bound, at_bound_reportable=fl.reportable_only([n.split('@')[0] for n in at_bound]),
                 weakly_constrained=weak, weakly_constrained_reportable=fl.reportable_only(weak),
                 weakly_constrained_detail=weak_detail, alternatives=alt_summary, ambiguity=ambiguity,
                 ambiguous=ambiguous,
                 inactive_populations=[pop['pop'] for pop in pops if pop['inactive']],
                 uncertainties='not available (preview: pure network proposal)' if preview else
                 ('gauss-newton covariance' if has_errors else 'not available (no free coordinate)'),
                 quality=quality['flag'], quality_detail=quality,
                 sigma_assumed=settings['_sigma_assumed'], fallback='S3' in run.stages_run, ood=ood.get('ood'))
    born_reference = None
    evaluations = run.evaluations()
    if run.born is not None:
        b = run.born
        pb = b['setup'].codec.decode(b['u'])
        born_reference = dict(chi2=b['chi2'], chi2_per_point=b['chi2']/n, stage=b['stage'], source=b['source'],
                              populations=[dict(pop=pop['pop'], type=pop['type'], R_nm=float(pb['R'][pop['slot']]),
                                                D_nm=float(pb['D'][pop['slot']]) if setup.sf_on else None,
                                                amplitude=float(b['coef'][pop['slot']])) for pop in pops])
        merged = dict(b['evaluations']['by_stage'], **evaluations['by_stage'])
        evaluations = dict(total={k: sum(v[k] for v in merged.values()) for k in evaluations['total']}, by_stage=merged)
    flags['raised'] = fl.raised(flags)
    return dict(composition=[schema.TYPE_NAMES[int(t)] for t in setup.types if t],
                composition_text=composition_text(setup.combo), combo=int(setup.combo), chi2=chi2,
                chi2_per_point=chi2/n, n_points=n, dof=dof, k_params=k, bic=chi2+k*np.log(n),
                populations=pops, background_B=float(coef[4]), background_fixed=setup.background is not None,
                low_q=low_q, flags=flags, stop_stage=run.stop_stage, stages_run=list(run.stages_run),
                signals=dict(run.signals, fired=list(run.fired)), stage_seconds=dict(run.stage_seconds),
                evaluations=evaluations, winner=dict(stage=cand['stage'], source=cand['source']),
                sphere_model='dwba4' if run.born is not None else 'born', born_reference=born_reference,
                u=u, u_base=setup.codec.base.encode(p), coef=coef,
                curves=dict(model=prediction, components=components))


def parse_compositions(composition):
    """(combo indices, auto?) from a composition string, a combo index or a list of combo indices.

    Raises ValueError with the accepted syntax for anything else."""
    syntax = ("use a comma list of 1..4 shapes out of sphere|rc|vc (e.g. 'sphere,vc,vc'), or "
              "'auto[:shapes][:Kmax]' (e.g. 'auto:vc:3'), or a combo index 0..%d" % (len(COMBOS)-1))
    if isinstance(composition, str):
        text = composition.strip().lower()
        try:
            combos = parse_composition(text)
        except (KeyError, IndexError, ValueError) as exc:
            detail = f'unknown shape {exc.args[0]!r}' if isinstance(exc, KeyError) and exc.args else str(exc)
            raise ValueError(f'composition {composition!r} not understood ({detail}): {syntax}') from None
        auto = text.startswith('auto')
    else:
        try:
            items = [composition] if isinstance(composition, (int, np.integer)) and not isinstance(composition, bool) \
                else list(composition)
            combos = [int(c) for c in items]
        except (TypeError, ValueError):
            raise ValueError(f'composition {composition!r} not understood: {syntax}') from None
        if any(isinstance(c, bool) for c in items) or any(not 0 <= c < len(COMBOS) for c in combos):
            raise ValueError(f'composition {composition!r}: combo indices must be within 0..{len(COMBOS)-1}')
        auto = len(combos) > 1
    if not combos:
        raise ValueError(f'composition {composition!r} gives no candidate compositions: {syntax}')
    return [int(c) for c in combos], auto


def fit1d(q, intensity, sigma=None, *, composition, mode='gisaxs', qz=0.0, structure_factor=None, lowq=None,
          lowq_nu=2.0, constraints=None, policy='auto', heads=2, max_steps=40, first_steps=8, workers=1,
          model_dir=None, delta_bic=10.0, seed=0, source=None, dwba=None, align=True, pursuit=True, profile=True,
          sigma_floor=0.0):
    """Fit one curve; returns a JSON-serialisable dict (schema 'gisaxs1d_result_v1', see research.fit1d.schema).

    q (nm^-1), intensity, sigma (None -> Poisson sqrt(max(I, 1)), flag sigma_assumed).
    sigma_floor: relative systematic error added in quadrature, sigma_eff^2 = sigma^2 + (sigma_floor * I)^2
    (0 = off). Counting statistics alone understate the error of bright curves (flat field, background,
    model imperfections of a few per cent); the fit then chases those systematics. chi2, the uncertainties and
    the flags refer to sigma_eff; data.sigma in the result is sigma_eff.
    mode 'gisaxs' | 'saxs'. qz: out-of-plane momentum (nm^-1) for spheres/random cylinders in GISAXS mode
    (the caller decides; the physical default is the vacuum qz at the Yoneda row). structure_factor:
    None -> on for gisaxs, off for saxs. lowq: None -> 'lorentz' (gisaxs) / 'none' (saxs); 'lorentz'
    C/(1+(q/sigma_res)^nu) with nu fixed at lowq_nu (None = free in [1, 6]); 'powerlaw' C (q/q_min)^-n,
    n in [1, 4]. constraints: see research.fit1d.constraints. policy 'preview' (S0), 'fast' (S0-S2),
    'auto' (S3 on state signals), 'robust' (always S3). heads: heads refined in S1. workers > 1: spawn
    pool for S3/S4. source: optional dict with input information (file name, reader meta) stored in 'input'.
    dwba: None (Born: spheres at one qz) or the geometry of the cut for the four-term DWBA sphere model
    (particles on a substrate): a research.fast_inference.tabfit.YonedaDWBA or a dict with wavelength_nm,
    alpha_i_deg, optionally alpha_f_deg (exit angle of the cut; default: the critical angle of the substrate,
    where a Yoneda cut sits) and either substrate_delta[, substrate_beta] or substrate='Si'. GISAXS only.
    align: (R, D) grid alignment of every start before/after its first LM chunk (research.fit1d.gridsearch);
    pursuit: one extra start built greedily without the network (K >= 2). Both default on.
    profile: after the search, look for other (R, D) regions of every population with an equivalent chi2
    (research.fit1d.profile, stage P) and refine them briefly; together with the other finished starts they
    decide which parameters are ambiguous (research.fit1d.ensemble: flags.ambiguity, weakly_constrained
    reason 'multimodal', R_nm_range / D_nm_range of the populations). Default on (needs align).
    The Born stages run first (fast); stage S5 then refits every composition that contains spheres in DWBA and
    the result reports the DWBA solution (born_reference keeps the Born optimum). qz is then the vacuum qz
    of that geometry.
    """
    t_start = time.perf_counter()
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if policy not in POLICIES:
        raise ValueError(f'policy must be one of {POLICIES}')
    if lowq is not None and lowq not in LOWQ:
        raise ValueError(f'lowq must be one of {LOWQ} or None')
    gisaxs = mode == 'gisaxs'
    if qz is not None and not np.isfinite(float(qz)):
        raise ValueError(f'qz must be a finite number (nm^-1), got {qz!r}')
    if not gisaxs and qz not in (None, 0, 0.0):
        raise ValueError('SAXS mode has no qz (it must be 0); use mode="gisaxs" for Yoneda cuts')
    dwba = make_dwba(dwba)
    if dwba is not None:
        if not gisaxs:
            raise ValueError('dwba applies to mode="gisaxs" only')
        qz = dwba.qz_vacuum
    if int(max_steps) < 1 or int(first_steps) < 1 or int(heads) < 1:
        raise ValueError('max_steps, first_steps and heads must be >= 1')
    if not float(delta_bic) >= 0:
        raise ValueError('delta_bic must be >= 0')
    if not (np.isfinite(sigma_floor) and 0.0 <= float(sigma_floor) < 1.0):
        raise ValueError('sigma_floor must be in [0, 1) (a relative error, e.g. 0.03)')
    with single_thread():
        result = _fit(q, intensity, sigma, composition=composition, gisaxs=gisaxs, mode=mode,
                      qz=float(qz or 0.0) if gisaxs else 0.0, structure_factor=structure_factor, lowq=lowq,
                      lowq_nu=lowq_nu, constraints=constraints, policy=policy, heads=int(heads),
                      max_steps=int(max_steps), first_steps=int(first_steps), workers=int(workers or 1),
                      model_dir=model_dir, delta_bic=float(delta_bic), seed=int(seed), source=source, t_start=t_start,
                      dwba=dwba, align=bool(align), pursuit=bool(pursuit), profile=bool(profile),
                      sigma_floor=float(sigma_floor))
    return result


def make_dwba(spec):
    """None, a YonedaDWBA, or a dict (wavelength_nm, alpha_i_deg, alpha_f_deg, substrate_delta[, substrate_beta]
    | substrate='Si') -> YonedaDWBA or None."""
    if spec is None:
        return None
    from research.fast_inference.tabfit import YonedaDWBA, si_index
    if isinstance(spec, YonedaDWBA):
        return spec
    if not isinstance(spec, dict):
        raise ValueError('dwba must be None, a YonedaDWBA or a dict')
    missing = [k for k in ('wavelength_nm', 'alpha_i_deg') if spec.get(k) is None]
    if missing:
        raise ValueError(f'dwba: missing {", ".join(missing)}')
    wavelength = float(spec['wavelength_nm'])
    if spec.get('substrate_delta') is not None:
        delta, beta = float(spec['substrate_delta']), float(spec.get('substrate_beta') or 0.0)
    elif str(spec.get('substrate', 'Si')).lower() == 'si':
        delta, beta = si_index(wavelength)
    else:
        raise ValueError("dwba: give substrate_delta (and substrate_beta) or substrate='Si'")
    # A Yoneda cut sits at the critical angle of the substrate: that is the default exit angle when the cut's
    # own alpha_f is not given (a measured alpha_f from the cut geometry is better: R is sensitive to it).
    alpha_f = spec.get('alpha_f_deg')
    alpha_f = float(np.degrees(np.sqrt(2*delta))) if alpha_f is None else float(alpha_f)
    values = (wavelength, float(spec['alpha_i_deg']), alpha_f, delta, beta)
    if not all(np.isfinite(v) for v in values) or wavelength <= 0 or values[1] <= 0 or values[2] <= 0:
        raise ValueError(f'dwba: wavelength and angles must be finite and positive, got {values[:3]}')
    return YonedaDWBA(*values)


def _fit(q, intensity, sigma, *, composition, gisaxs, mode, qz, structure_factor, lowq, lowq_nu, constraints,
         policy, heads, max_steps, first_steps, workers, model_dir, delta_bic, seed, source, t_start, dwba=None,
         align=True, pursuit=True, profile=True, sigma_floor=0.0):
    q_raw = np.asarray(q, float).ravel()
    y_raw = np.asarray(intensity, float).ravel()
    if q_raw.shape != y_raw.shape:
        raise ValueError(f'q and intensity differ in length ({len(q_raw)} vs {len(y_raw)})')
    sigma_assumed = sigma is None
    s_raw = np.sqrt(np.maximum(np.where(np.isfinite(y_raw), y_raw, 1.0), 1.0)) if sigma_assumed else \
        np.asarray(sigma, float).ravel()
    if s_raw.shape != y_raw.shape:
        raise ValueError('sigma and intensity differ in length')
    feats = fl.input_features(q_raw, y_raw, s_raw)
    ood = fl.ood_check(feats)
    qc, yc, sc = _clean_inputs(q_raw, y_raw, s_raw)          # features / ood above describe the sigma that was given
    if sigma_floor > 0:
        sc = np.sqrt(sc*sc+(sigma_floor*np.abs(yc))**2)
    if len(qc) < 10:
        raise ValueError(f'only {len(qc)} valid points (finite, q > 0, sigma > 0); need at least 10')
    sf_on = gisaxs if structure_factor is None else bool(structure_factor)
    lowq_eff = lowq if lowq is not None else ('lorentz' if gisaxs else 'none')
    res_on = lowq_eff != 'none'
    net_flag = bool(gisaxs or sf_on or res_on)

    combos, auto = parse_compositions(composition)             # usage errors first (cheap, no files needed)
    norm = cons.normalise(constraints)
    schema.check_tables()
    model_dir = Path(model_dir or DEFAULT_MODEL_DIR).resolve()
    fitter = get_fitter(model_dir)

    nu = cons.nu_rule(lowq_eff, lowq_nu, norm) if res_on else None
    if 'nu' in norm['globals'] and lowq_eff != 'lorentz':
        raise cons.ConstraintError("'nu' applies only to lowq='lorentz'")
    pl_rule = norm['globals'].get('powerlaw_n')
    if pl_rule is not None and lowq_eff != 'powerlaw':
        raise cons.ConstraintError("'powerlaw_n' applies only to lowq='powerlaw'")
    background = norm['globals']['background'][1] if 'background' in norm['globals'] else None
    base = CurveCodec(float(qc.min()), float(qc.max()))
    pl_fixed, u31, lower, upper = powerlaw_box(pl_rule)

    setups, skipped = [], {}
    for combo in combos:
        types = COMBOS[int(combo)]
        rules, unused = cons.slot_rules(norm, types, sf_on)
        if unused and not auto:
            raise cons.ConstraintError(f'{", ".join(unused)}: composition {composition_text(combo)} has only '
                                       f'{int(np.sum(types > 0))} population(s)')
        if unused:
            skipped[composition_text(combo)] = unused
        codec = cons.ConstrainedCodec(base, types, rules, nu=nu if lowq_eff == 'lorentz' else None)
        fixed = list(codec.fixed_indices)
        if lowq_eff == 'powerlaw' and pl_fixed:
            fixed.append(31)
        setups.append(Setup(qc, yc, sc, int(combo), qz, sf_on, res_on, lowq_eff, codec, tuple(sorted(set(fixed))),
                            dict(lower) if lowq_eff == 'powerlaw' else {},
                            dict(upper) if lowq_eff == 'powerlaw' else {}, background,
                            u31 if lowq_eff == 'powerlaw' else None))

    # a curve cannot determine more parameters than it has points
    too_many = {}
    for setup in setups:
        k_max = len(setup.free_coords())+int(np.sum(setup.types > 0))+int(background is None)+int(res_on)
        if len(qc) <= k_max:
            too_many[composition_text(setup.combo)] = k_max
    if too_many and (not auto or len(too_many) == len(setups)):
        worst = min(too_many.values())
        raise ValueError(f'only {len(qc)} valid points for a model with {worst} free parameters '
                         f'({", ".join(too_many)}): use fewer populations, fix parameters, or give more points')
    if too_many:
        setups = [s for s in setups if composition_text(s.combo) not in too_many]
        combos = [s.combo for s in setups]

    t0 = time.perf_counter()
    u_net = network_heads(fitter, qc, yc, sc, combos, qz, net_flag)
    network_seconds = time.perf_counter()-t0
    runs = [CompositionRun(setup, u_net[i], n_heads=heads, first_steps=first_steps, max_steps=max_steps, seed=seed,
                           align=align, pursuit=pursuit)
            for i, setup in enumerate(setups)]
    for run in runs:
        run.stage0(network_seconds/len(runs))
    if policy != 'preview':
        for run in runs:
            run.stage12()
            run.compute_signals(lambda cand, run=run: _reportable(run, cand))

    # S3/S4 selection
    # 'robust': every candidate composition (otherwise the BIC ranking would compare robust with fast fits);
    # 'auto': the lowest-BIC candidate and those within delta_bic of it, when a state signal fired
    selected = []
    if policy == 'robust':
        selected = list(runs)
    elif policy == 'auto':
        bics = [run.bic(run.best) for run in runs]
        top = min(bics)
        selected = [run for run, b in zip(runs, bics) if b-top <= delta_bic and run.fired]
    tables_dir = schema.tables_dir()
    if selected:
        serial = {run.setup.key: run.fit for run in runs}
        for stage in ('S3', 'S4'):
            batches = [(run, *(run.robust_jobs() if stage == 'S3' else run.dscan_jobs())) for run in selected]
            batches = [b for b in batches if b[1]]
            if not batches:
                continue
            if workers > 1:
                t0 = time.perf_counter()
                results = run_jobs([job for _, jobs, _ in batches for job in jobs], workers, serial, tables_dir)
                wall = time.perf_counter()-t0
                offset = 0
                for run, jobs, labels in batches:
                    run.absorb(stage, labels, results[offset:offset+len(jobs)], wall)
                    offset += len(jobs)
            else:
                for run, jobs, labels in batches:
                    t0 = time.perf_counter()
                    results = run_jobs(jobs, 1, serial, tables_dir)
                    run.absorb(stage, labels, results, time.perf_counter()-t0)

    if profile and policy != 'preview':
        # stage P on the compositions that will be reported as candidates (within delta_bic of the lowest BIC)
        bics = [run.bic(run.best) for run in runs]
        for run, b in zip(runs, bics):
            if b-min(bics) <= delta_bic:
                run.profile_stage(polish=PROFILE_POLISH[policy], budget=PROFILE_BUDGET[policy],
                                  exchange=PROFILE_EXCHANGE[policy][0], exchange_steps=PROFILE_EXCHANGE[policy][1])

    if dwba is not None:
        for run in runs:
            if np.any(run.setup.types == 1):                 # only spheres change; vc is exact, rc keeps qz
                run.dwba_refine(dwba)

    settings = dict(mode=mode, composition_request=composition if isinstance(composition, str) else list(combos),
                    compositions=[composition_text(c) for c in combos], qz_nm_inv=qz, structure_factor=sf_on,
                    lowq=lowq_eff, lowq_nu=(nu[1] if nu is not None and nu[0] == 'fixed' else None),
                    lowq_nu_rule=cons.rule_json(nu) if nu is not None else None,
                    constraints=dict(populations={('all' if k == 'all' else f'pop{k}'): {kk: cons.rule_json(r) for kk, r in v.items()}
                                                  for k, v in norm['populations'].items()},
                                     globals={k: cons.rule_json(r) for k, r in norm['globals'].items()},
                                     skipped_in_auto=skipped),
                    policy=policy, heads=heads, first_steps=first_steps, max_steps=max_steps, workers=workers,
                    workers_effective=effective_workers(workers), delta_bic=delta_bic, seed=seed,
                    network_gisaxs_flag=net_flag, skipped_too_few_points=too_many,
                    align=align, pursuit=pursuit, profile=profile, sigma_floor=sigma_floor,
                    sphere_model='dwba4 (stage S5)' if dwba is not None else 'born',
                    dwba=dwba.as_dict() if dwba is not None else None,
                    _sigma_assumed=sigma_assumed, _powerlaw_rule=pl_rule, _policy=policy,
                    _stratum=feats[fl.FEATURE_NAMES.index(fl.STRATUM_FEATURE)])
    results = [_composition_result(run, settings, ood) for run in runs]
    results.sort(key=lambda r: (r['bic'], r['combo']))
    top = results[0]['bic']
    for i, r in enumerate(results, 1):
        r['rank'] = i
        r['delta_bic'] = r['bic']-top
        r['in_candidate_set'] = bool(r['delta_bic'] <= delta_bic)
    candidate_set = [r['composition_text'] for r in results if r['in_candidate_set']]
    for r in results:
        # indistinguishable: within delta_bic of the lowest BIC together with at least one other composition
        r['indistinguishable'] = bool(r['in_candidate_set'] and len(candidate_set) > 1)
    settings = {k: v for k, v in settings.items() if not k.startswith('_')}
    notes = []
    if auto:
        if len(candidate_set) > 1:
            notes.append(f'{len(results)} candidate compositions ranked by BIC; {len(candidate_set)} of them '
                         f'({", ".join(candidate_set)}) lie within delta BIC <= {delta_bic:g} and are '
                         'indistinguishable on this curve; choose among them from sample knowledge.')
        else:
            notes.append(f'{len(results)} candidate compositions ranked by BIC; only {candidate_set[0]} lies within '
                         f'delta BIC <= {delta_bic:g}. This is a statement about the listed candidates and this '
                         'curve only (other shapes or more populations were not tried); prefer sample knowledge.')
    else:
        notes.append('Composition supplied by the user; the fit does not test it against alternatives.')
    if gisaxs and qz == 0.0 and any(t in (1, 2) for c in combos for t in COMBOS[int(c)]):
        notes.append('qz = 0 with spheres/random cylinders in GISAXS mode: the physical default is the vacuum qz '
                     'at the Yoneda row (PROJECT_LOG decision 2026-10-01); qz = 0 only on request.')
    if sigma_assumed:
        notes.append('No sigma given: Poisson sqrt(max(I, 1)) assumed; chi2 values have no absolute scale.')
    if sigma_floor > 0:
        notes.append(f'sigma_floor = {sigma_floor:g}: sigma_eff^2 = sigma^2 + ({sigma_floor:g} I)^2; chi2, uncertainties and flags '
                     'refer to sigma_eff (data.sigma).')
    if ood.get('ood'):
        notes.append('Input features lie outside the DEV distribution (ood flag); treat the fit with caution.')
    if policy == 'preview':
        notes.append('Preview: network proposal with NNLS amplitudes only (no LM refinement); no uncertainties, '
                     'so weakly_constrained is not evaluated.')
    if too_many:
        notes.append(f'Candidates skipped because the curve has only {len(qc)} points for their parameter count: '
                     f'{", ".join(too_many)}.')
    if workers > 1 and effective_workers(workers) == 1:
        notes.append('workers > 1 was requested, but the fit ran serially: either this is a child process (no '
                     'nested pool; protect the calling script with "if __name__ == \'__main__\':") or the main '
                     'script is not a file (piped to "python -"), which spawn workers cannot re-import.')
    for r in results:
        if r['flags']['inactive_populations']:
            notes.append(f"{r['composition_text']}: {', '.join(r['flags']['inactive_populations'])} has amplitude 0 "
                         'and does not contribute to the model; its parameters carry no information.')
        if r['flags']['alternatives']:
            notes.append(f"{r['composition_text']}: other solutions of this fit with an equivalent chi2 have different "
                         f"values of {', '.join(r['flags']['alternatives'])} (flags.alternatives); the quoted 1-sigma "
                         'describes one local minimum only.')
    result = dict(schema=schema.SCHEMA, fit1d_version=schema.VERSION, conventions=schema.CONVENTIONS,
                  population_order=schema.POPULATION_ORDER,
                  input=dict(n_points_given=int(len(q_raw)), n_points_used=int(len(qc)),
                             n_points_dropped=int(len(q_raw)-len(qc)), q_min_nm_inv=float(qc.min()),
                             q_max_nm_inv=float(qc.max()), sigma_assumed=sigma_assumed,
                             features=dict(zip(fl.FEATURE_NAMES, feats)), ood=ood, source=source or {}),
                  settings=settings, provenance=schema.provenance(model_dir), candidate_set=candidate_set,
                  results=results,
                  data=dict(q=qc, intensity=yc, sigma=sc), notes=notes,
                  timing=dict(total_seconds=time.perf_counter()-t_start, network_seconds=network_seconds,
                              created_unix=time.time()))
    return schema.jsonable(result)
