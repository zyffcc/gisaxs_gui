"""The result a GUI gets: a flat, JSON-serialisable dict built from a research.fit1d result.

schema 'gisaxs1d_gui_result_v1'
  summary       chi2_per_point, n_points, seconds, policy, composition, mode, qz_nm_inv, model, stop_stage,
                flags (raised flag names), signals (state signals that fired), status ('ok' | 'check'),
                messages (short English sentences: what to look at), notes (the notes of fit1d)
  populations   one dict per population: name, shape, R_nm, R_nm_err, R_nm_range, sigma_R_rel, D_nm, D_nm_err,
                D_nm_range, w_rel, h_nm, amplitude, intensity_fraction, inactive, status {parameter: status}
  table         display rows: group, parameter, key, value, error, unit, range, status, text
                status: ok | ambiguous (another solution fits equally well) | weak (not pinned down by the
                curve) | at_bound | fixed | inactive (population with zero amplitude) | derived
  background, low_q
  curves        q_nm_inv, q (in the unit of the input), q_unit, intensity, sigma, model, residual
                ((intensity - model) / sigma) and components {name: array}
  candidates    ranked compositions of an 'auto:...' request (rank, composition, chi2_per_point, delta_bic,
                indistinguishable)
  gimap_candidates   the same solutions as rows of the GIMaP candidate table (see gimap_candidate)
  conventions   units and physics of the numbers (from fit1d)
  raw           the complete fit1d result (only with include_raw=True)
"""
from __future__ import annotations

import math

SCHEMA = 'gisaxs1d_gui_result_v1'
SHAPE_TITLES = {'sphere': 'sphere', 'vertical_cylinder': 'vertical cylinder', 'random_cylinder': 'random cylinder'}
GIMAP_SHAPES = {'sphere': 'sphere', 'vertical_cylinder': 'vertical_cylinder', 'random_cylinder': 'cylinder'}
# (key, label, unit, error key, range key)
POPULATION_ROWS = (('R_nm', 'R', 'nm', 'R_nm_err', 'R_nm_range'), ('sigma_R_rel', 'sigma_R / R', '', 'sigma_R_rel_err', None),
                   ('D_nm', 'D', 'nm', 'D_nm_err', 'D_nm_range'), ('w_rel', 'w / D', '', 'w_rel_err', None),
                   ('h_nm', 'h', 'nm', 'h_nm_err', None), ('sigma_h_rel', 'sigma_h / h', '', 'sigma_h_rel_err', None))


def _finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def format_value(value, error=None):
    """'2.60 ± 0.03' with the error to two significant digits; '2.596' without a usable error."""
    if not _finite(value):
        return ''
    if _finite(error) and error > 0:
        digits = max(0, min(8, 1-int(math.floor(math.log10(error)))))
        return f'{value:.{digits}f} ± {error:.{digits}f}'
    return f'{value:.4g}'


def _status(pop, key, flags):
    name = f"{pop['pop']}.{key}"
    if pop.get('inactive'):
        return 'inactive'
    if key in (pop.get('fixed') or []):
        return 'fixed'
    if name in (flags.get('ambiguous') or []) or 'multimodal' in (flags.get('weakly_constrained_detail') or {}).get(name, []):
        return 'ambiguous'
    if any(str(item).split('@')[0] == name for item in (flags.get('at_bound') or [])):
        return 'at_bound'
    if name in (flags.get('weakly_constrained') or []):
        return 'weak'
    return 'ok'


def population_rows(result):
    """(populations, table rows) of one fit1d composition result."""
    flags = result.get('flags') or {}
    populations, table = [], []
    for pop in result.get('populations') or []:
        shape = pop.get('type')
        item = dict(name=pop['pop'], shape=shape, inactive=bool(pop.get('inactive')), amplitude=pop.get('amplitude'),
                    intensity_fraction=pop.get('intensity_fraction'), status={})
        group = f"{pop['pop']} ({SHAPE_TITLES.get(shape, shape)})"
        for key, label, unit, err_key, range_key in POPULATION_ROWS:
            value = pop.get(key)
            if value is None:
                continue
            error = pop.get(err_key)
            status = _status(pop, key, flags)
            item[key] = value
            item[err_key] = error
            if range_key:
                item[range_key] = pop.get(range_key)
            item['status'][key] = status
            table.append(dict(group=group, parameter=label, key=f"{pop['pop']}.{key}", value=value,
                              error=error if _finite(error) else None, unit=unit,
                              range=pop.get(range_key) if range_key else None, status=status, text=format_value(value, error)))
        fraction = pop.get('intensity_fraction')
        if _finite(fraction):
            table.append(dict(group=group, parameter='intensity share', key=f"{pop['pop']}.intensity_fraction", value=fraction,
                              error=None, unit='', range=None, status='inactive' if pop.get('inactive') else 'derived',
                              text=f'{100*fraction:.1f} %'))
        populations.append(item)
    low = result.get('low_q') or {}
    if _finite(result.get('background_B')):
        table.append(dict(group='global', parameter='background B', key='background_B', value=result['background_B'], error=None,
                          unit='counts', range=None, status='fixed' if result.get('background_fixed') else 'derived',
                          text=format_value(result['background_B'])))
    for key, label, unit in (('C', 'low-q amplitude C', 'counts'), ('sigma_res_nm_inv', 'low-q width', 'nm^-1'), ('nu', 'low-q exponent nu', ''),
                             ('n', 'low-q power n', '')):
        if _finite(low.get(key)):
            fixed = bool(low.get(f'{key}_fixed'))
            table.append(dict(group='global', parameter=label, key=f'low_q.{key}', value=low[key], error=None, unit=unit, range=None,
                              status='fixed' if fixed else ('inactive' if low.get('inactive') else 'derived'), text=format_value(low[key])))
    return populations, table


def messages(result, fit):
    """Short English sentences for the status line of a GUI (the panel translates them by exact match)."""
    flags = result.get('flags') or {}
    out = []
    if flags.get('quality'):
        out.append('The model does not describe the curve within the error bars (quality flag).')
    if flags.get('ood'):
        out.append('The curve is unlike the curves the tool was tested on (out-of-distribution flag).')
    if flags.get('ambiguous'):
        out.append('Another solution fits equally well: ' + ', '.join(flags['ambiguous']) + '.')
    weak = [n for n in (flags.get('weakly_constrained_reportable') or []) if n not in (flags.get('ambiguous') or [])]
    if weak:
        out.append('Not pinned down by this curve: ' + ', '.join(weak) + '.')
    if flags.get('inactive_populations'):
        out.append('Populations without contribution: ' + ', '.join(flags['inactive_populations']) + '.')
    if flags.get('sigma_assumed'):
        out.append('No error bars were given: Poisson errors were assumed.')
    if result.get('indistinguishable'):
        out.append('Several compositions fit equally well; the curve alone does not decide between them.')
    return out


def gimap_candidate(result, rank=1):
    """One solution as a row of the GIMaP candidate table (keys of fitting/domain/candidates.py).

    The numbers keep the conventions of this tool (physics v2): R and sigma_R of the number-weighted distribution,
    sigma_D = w_rel (Hosemann spread w / D). GIMaP's own forward model (intensity-weighted sizes, damping
    exp(-pi q^2 (sigma_D D)^2), no qz) draws a different curve from the same numbers: plot result['curves'] for
    this row, and treat a transfer into the manual model as a starting point only (legacy_sigma_D is the
    spread that gives the same paracrystal damping there: w_rel / sqrt(2 pi))."""
    components = []
    for pop in result.get('populations') or []:
        params = {}
        for src, dst in (('R_nm', 'R'), ('sigma_R_rel', 'sigma_R'), ('h_nm', 'h'), ('sigma_h_rel', 'sigma_h'), ('D_nm', 'D'),
                         ('w_rel', 'sigma_D')):
            if _finite(pop.get(src)):
                params[dst] = float(pop[src])
        if 'sigma_D' in params:
            params['legacy_sigma_D'] = params['sigma_D']/math.sqrt(2*math.pi)
        components.append(dict(type=GIMAP_SHAPES.get(pop.get('type'), pop.get('type')), weight=float(pop.get('amplitude') or 0.0),
                               params=params, name=pop.get('pop'), inactive=bool(pop.get('inactive'))))
    low = result.get('low_q') or {}
    global_params = dict(background=float(result.get('background_B') or 0.0), k=1.0)
    if _finite(low.get('C')):
        global_params['int_res'] = float(low['C'])
    if _finite(low.get('sigma_res_nm_inv')):
        global_params['sigma_res'] = float(low['sigma_res_nm_inv'])
    if _finite(low.get('nu')):
        global_params['nu_res'] = float(low['nu'])
    n = max(int(result.get('n_points') or 1), 1)
    return dict(rank=int(rank), source='gisaxs1d_kit', composition=result.get('composition_text'), components=components,
                global_params=global_params, best_chi2_weighted=float(result.get('chi2') or 0.0),
                chi2_per_point=float(result.get('chi2_per_point') or 0.0), best_log_rmse=None, n_points=n,
                flags=list((result.get('flags') or {}).get('raised') or []), conventions='fit1d physics v2 (see gisaxs1d_kit.results.gimap_candidate)')


def _log_rmse(intensity, model):
    pairs = [(a, b) for a, b in zip(intensity, model) if a > 0 and b > 0]
    if not pairs:
        return None
    return math.sqrt(sum((math.log10(a)-math.log10(b))**2 for a, b in pairs)/len(pairs))


def gui_result(fit, *, model_name, q_unit='nm^-1', q_factor=1.0, seconds=None, include_raw=False, kit_version=''):
    """fit: the dict returned by research.fit1d.fit1d. q_factor: nm^-1 per input unit (10 for A^-1)."""
    best = fit['results'][0]
    data = fit.get('data') or {}
    q = [float(v) for v in data.get('q') or []]
    intensity = [float(v) for v in data.get('intensity') or []]
    sigma = [float(v) for v in data.get('sigma') or []]
    model = [float(v) for v in (best.get('curves') or {}).get('model') or []]
    populations, table = population_rows(best)
    flags = best.get('flags') or {}
    settings = fit.get('settings') or {}
    text = messages(best, fit)
    rows = []
    for i, r in enumerate(fit['results'], 1):
        row = gimap_candidate(r, i)
        row['best_log_rmse'] = _log_rmse(intensity, (r.get('curves') or {}).get('model') or [])
        rows.append(row)
    out = dict(
        schema=SCHEMA, ok=True, kit_version=kit_version, fit1d_version=fit.get('fit1d_version'),
        summary=dict(chi2_per_point=best.get('chi2_per_point'), chi2=best.get('chi2'), n_points=best.get('n_points'), dof=best.get('dof'),
                     seconds=seconds if seconds is not None else (fit.get('timing') or {}).get('total_seconds'),
                     network_seconds=(fit.get('timing') or {}).get('network_seconds'), policy=settings.get('policy'),
                     composition=best.get('composition_text'), mode=settings.get('mode'), qz_nm_inv=settings.get('qz_nm_inv'),
                     sigma_floor=settings.get('sigma_floor'), model=model_name, stop_stage=best.get('stop_stage'),
                     stages=list(best.get('stages_run') or []), flags=list(flags.get('raised') or []),
                     signals=list((best.get('signals') or {}).get('fired') or []),
                     status='check' if (flags.get('quality') or flags.get('ood') or flags.get('weakly_constrained_reportable')
                                        or best.get('indistinguishable')) else 'ok',
                     messages=text, notes=list(fit.get('notes') or []), sphere_model=best.get('sphere_model')),
        populations=populations, table=table, background=best.get('background_B'), low_q=best.get('low_q'),
        curves=dict(q_nm_inv=q, q=[v/q_factor for v in q], q_unit=q_unit, intensity=intensity, sigma=sigma, model=model,
                    residual=[(a-b)/s if s > 0 else 0.0 for a, b, s in zip(intensity, model, sigma)],
                    components={k: [float(x) for x in v] for k, v in ((best.get('curves') or {}).get('components') or {}).items()}),
        candidates=[dict(rank=r.get('rank'), composition=r.get('composition_text'), chi2_per_point=r.get('chi2_per_point'),
                         bic=r.get('bic'), delta_bic=r.get('delta_bic'), indistinguishable=bool(r.get('indistinguishable')),
                         flags=list((r.get('flags') or {}).get('raised') or [])) for r in fit['results']],
        gimap_candidates=rows, conventions=fit.get('conventions'), settings=settings)
    if include_raw:
        out['raw'] = fit
    return out


def error_result(exc, kit_version=''):
    import traceback
    return dict(schema=SCHEMA, ok=False, kit_version=kit_version,
                error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
