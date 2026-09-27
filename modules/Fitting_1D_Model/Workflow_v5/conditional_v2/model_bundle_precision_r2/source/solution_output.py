"""NumPy-only, versioned output contract and permutation-invariant deduplication.

This module never changes neural candidates or averages fitted parameters.
"""
from functools import lru_cache
from itertools import permutations
import numpy as np
from TrainSetBuild import schema

OUTPUT_VERSION = 'v5-output-r1'
PARAMETER_UNITS = {'R': 'nm', 'sigma_R': '1', 'h': 'nm', 'sigma_h': '1',
                   'D': 'nm', 'sigma_D': '1'}
GLOBAL_UNITS = {'rho_BG': '1', 'sigma_Res': 'nm^-1', 'nu_Res': '1', 'rho_Res': '1'}
WIDTH_DEFINITIONS = {'sigma_R': 'standard deviation of radius / R',
                     'sigma_h': 'standard deviation of height / h',
                     'sigma_D': 'standard deviation of spacing / D'}


def denormalize(values, names, ranges):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    for k, name in enumerate(names):
        spec = ranges[name]
        x = np.clip(values[..., k], 0., 1.)
        result[..., k] = (np.exp(np.log(spec.low) + x*np.log(spec.high/spec.low))
                         if spec.transform == 'log' else spec.low+x*(spec.high-spec.low))
    return result


def active_weights(cand, i, j, combos):
    active = combos[int(cand['combos'][i, j])] > 0
    z = np.asarray(cand['weights'][i, j], dtype=np.float64)
    w = np.zeros(4, dtype=np.float64)
    z = z[active]
    w[active] = np.exp(z-z.max())
    return w/w.sum()


def component_state(cand, i, j, combos, collapse=True):
    """Ignore inactive fields; collapse only EXACTLY identical same-type components."""
    types = combos[int(cand['combos'][i, j])]
    weights = active_weights(cand, i, j, combos)
    components = []
    for slot in np.flatnonzero(types):
        typ = int(types[slot]); d = bool(cand['d'][i, j, slot] > 0)
        mask = np.array([True, True, typ == 2, typ == 2, d, d])
        params = np.where(mask, cand['params'][i, j, slot], 0.).astype(np.float64)
        match = next((c for c in components if collapse and c['type'] == typ
                      and c['d'] == d and np.array_equal(c['params'], params)), None)
        if match is None:
            components.append({'type': typ, 'd': d, 'params': params,
                               'weight': weights[slot], 'slots': [int(slot)]})
        else:
            match['weight'] += weights[slot]; match['slots'].append(int(slot))
    components.sort(key=lambda c:(c['type'], c['d'], *c['params'], c['weight']))
    res = bool(cand['res'][i, j] > 0)
    glob = np.where([True, res, res, res], cand['globals'][i, j], 0.).astype(np.float64)
    return {'components': components, 'res': res, 'globals': glob,
            'original_k': int((types > 0).sum()),
            'signature': (res, tuple((c['type'], c['d']) for c in components))}


def canonical_vector(cand, i, j, combos):
    """Compatibility representation; selection uses minimum-cost slot matching below."""
    s = component_state(cand, i, j, combos, collapse=False)
    p = np.zeros((4, 6)); w = np.zeros(4); d = np.zeros(4)
    for slot, c in enumerate(s['components']):
        p[slot] = c['params']; w[slot] = c['weight']; d[slot] = c['d']
    return np.concatenate((p.ravel(), w, d, s['globals'], [float(s['res'])]))


@lru_cache(maxsize=4)
def assignments(n):
    return np.array(list(permutations(range(n))), dtype=int)


def state_distance(a, b):
    """37-dimensional padded RMS, minimized over permutations of matching type/gate.

    Retains the old 0.03 scale. Discrete active gates must match; inactive gates
    and parameters are absent. This tolerance is a display rule, not probability.
    """
    if a['signature'] != b['signature']:
        return float('inf')
    cost = float(np.sum((a['globals']-b['globals'])**2))
    for key in sorted(set(a['signature'][1])):
        aa = np.array([np.r_[c['params'], c['weight']] for c in a['components']
                       if (c['type'], c['d']) == key])
        bb = np.array([np.r_[c['params'], c['weight']] for c in b['components']
                       if (c['type'], c['d']) == key])
        pairwise = np.sum((aa[:, None, :]-bb[None, :, :])**2, axis=-1)
        cost += float(np.min(pairwise[np.arange(len(aa))[None, :], assignments(len(aa))].sum(1)))
    return float(np.sqrt(cost/37.))


def select_candidates(cand, i, combos, errors, mask, distance=.03, limit=None, eligible=None):
    """Lowest observed-error representative, stable ties, no parameter modification.

    Cross-K suppression additionally requires essentially identical reduced state
    and a maximum pointwise log-curve difference <= 1e-5 on valid input q points.
    Merely similar curves from different physical components are retained.
    """
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError('distance must be finite and positive')
    if limit is not None and (not isinstance(limit, (int, np.integer)) or limit < 1):
        raise ValueError('limit must be a positive integer or None')
    errors = np.asarray(errors)
    valid_q = np.asarray(mask, dtype=bool)
    if not valid_q.any() or not np.isfinite(errors).all():
        raise ValueError('Valid q points and finite candidate errors are required')
    states = {}; buckets = {}; kept = []; duplicate_of = {}
    for j in np.argsort(errors, kind='stable'):
        j = int(j)
        if eligible is not None and not eligible[j]:
            continue
        state = component_state(cand, i, j, combos)
        duplicate = None
        for k in buckets.get(state['signature'], []):
            other = states[k]
            delta = state_distance(state, other)
            if state['original_k'] == other['original_k']:
                same = delta < distance
            else:
                # 1e-8 RMS only accommodates floating-point softmax weight sums.
                same = delta <= 1e-8 and np.max(np.abs(
                    np.log(np.maximum(cand['curves'][i, j, valid_q], 1e-30))-
                    np.log(np.maximum(cand['curves'][i, k, valid_q], 1e-30)))) <= 1e-5
            if same:
                duplicate = k; break
        if duplicate is not None:
            duplicate_of[j] = duplicate
            continue
        states[j] = state; buckets.setdefault(state['signature'], []).append(j); kept.append(j)
        if limit is not None and len(kept) >= limit:
            break
    return kept, duplicate_of


def summarize(data, cand, combos, max_solutions=8, threshold=.05, distance=.03):
    if not isinstance(max_solutions, (int, np.integer)) or max_solutions < 1:
        raise ValueError('max_solutions must be a positive integer')
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('threshold must be finite and positive')
    curves = np.asarray(cand['curves'])
    residual = np.log(np.maximum(curves, 1e-30))-np.log(np.maximum(data['observed'][:, None], 1e-30))
    errors = np.sqrt(np.sum(residual**2*data['mask'][:, None], axis=-1)/data['mask'].sum(-1)[:, None])
    physical = denormalize(cand['params'], schema.PARAM_NAMES, schema.V5_PARAM_NORM_RANGES)
    global_names = schema.V5_GLOBAL_TARGET_NAMES[:4]
    glob = denormalize(cand['globals'], global_names, schema.V5_GLOBAL_NORM_RANGES)
    rows = []
    for i in range(len(errors)):
        all_chosen, duplicate_of = select_candidates(cand, i, combos, errors[i], data['mask'][i], distance)
        solutions = []
        for j in all_chosen[:max_solutions]:
            types = combos[int(cand['combos'][i, j])]; w = active_weights(cand, i, j, combos)
            components = []
            for slot in np.flatnonzero(types):
                t = int(types[slot])
                p = {name:float(physical[i, j, slot, k]) for k, name in enumerate(schema.PARAM_NAMES)}
                if t != 2:
                    p['h'] = p['sigma_h'] = None
                if cand['d'][i, j, slot] <= 0:
                    p['D'] = p['sigma_D'] = None
                absolute = {sigma:None if p[length] is None else p[length]*p[sigma]
                            for sigma, length in [('sigma_R','R'),('sigma_h','h'),('sigma_D','D')]}
                components.append({'type_id':t, 'type':['unused','sphere','random_cylinder','vertical_cylinder'][t],
                                   'weight':float(w[slot]), 'parameters':p, 'absolute_widths_nm':absolute})
            gp = {name:float(glob[i, j, k]) for k, name in enumerate(global_names)}
            if cand['res'][i, j] <= 0:
                for name in global_names[1:]:
                    gp[name] = None
            state = component_state(cand, i, j, combos)
            solutions.append({'candidate_index':j, 'observed_logrmse':float(errors[i, j]),
                'passes_fit_threshold':bool(errors[i, j] < threshold),
                'all_component_weights_ge_005':bool(np.all(w[types > 0] >= .05)),
                'effective_component_count':len(state['components']),
                'identical_component_groups': [{'original_component_indices':c['slots'], 'total_weight':float(c['weight'])}
                                               for c in state['components'] if len(c['slots']) > 1],
                'components':components, 'global_parameters':gp})
        rows.append({'curve_index':i, 'best_observed_logrmse':float(errors[i].min()),
                     'raw_candidate_count':len(errors[i]), 'unique_candidate_count':len(all_chosen),
                     'duplicate_of':{str(j):k for j,k in duplicate_of.items()}, 'solutions':solutions})
    return {'output_schema_version':OUTPUT_VERSION,
        'fit_reference':'observed curve, natural logarithm, all native valid q points', 'fit_threshold':threshold,
        'probability_status':'No calibrated posterior probabilities; candidates ranked by forward fit',
        'units':'q in nm^-1; lengths in nm; sigma_R/h/D in parameters are dimensionless fractions',
        'unit_contract':{'inputs':{'q':'nm^-1','observed':'V5 relative intensity (k=1)',
            'sigma':'standard deviation in the same V5 intensity units as observed'},
            'component_parameters':PARAMETER_UNITS, 'component_parameter_definitions':WIDTH_DEFINITIONS,
            'absolute_widths_nm':'Derived length standard deviations in nm; not alternative forward inputs',
            'weight':'dimensionless normalized model mixture weight; not posterior, mass or volume fraction',
            'global_parameters':GLOBAL_UNITS,
            'global_definitions':{'rho_BG':'BG = rho_BG * median(P) on valid input q',
                'rho_Res':'intRes = rho_Res * max(P[first 5 valid q]) / max(g[first 5 valid q])',
                'sigma_Res':'q scale in g(q) = 1/(1+(q/sigma_Res)^nu_Res)',
                'nu_Res':'dimensionless exponent in g(q)'},
            'forward':'I = P + intRes*g + BG; P = sum(w_i*F_i*S_i), sum(w_i)=1, F_i(0)=1; no automatic unit or intensity scaling'},
        'deduplication':{'version':OUTPUT_VERSION,'distance_threshold':distance,
            'distance':'minimum type/gate-preserving assignment RMS over 37 padded coordinates',
            'identical_components':'exact active normalized parameters and same type/gate; weights are summed for comparison only',
            'cross_K':'reduced-state RMS <=1e-8 AND maximum valid-q log-curve difference <=1e-5',
            'representative':'lowest observed error; ties retain lowest raw candidate index',
            'claim':'display deduplication, not proof of complete or independent posterior modes'},
        'parameter_identifiability':'Inactive parameters are null; good fit does not imply unique or true parameters',
        'seconds':float(cand['seconds']) if 'seconds' in cand else None,
        'timing_scope':'neural prediction only, excludes output processing; null when reprocessing saved candidates',
        'curves':rows}
