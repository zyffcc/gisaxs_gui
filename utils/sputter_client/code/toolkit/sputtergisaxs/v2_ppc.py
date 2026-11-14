'Private v2 individual-draw predictive statistics; never a release receipt.'
import math
import numpy as np
from .features import rate_model, rate_jac
from .prediction import summarize_profiles
from .resolvable import PEAK_WINDOW
from .stage5_measurement import counts_profiles, describe, Q

GATES = {'q_star': .05, 'FWHM': .10, 'highq_ratio': .15,
         'valley_ratio': .20, 'Yoneda': .01}
STAT_SOURCES = ('conditional_poisson_expected_fisher', 'poisson_count_bootstrap')


def _finite(x):
    return (not isinstance(x, (bool, np.bool_)) and isinstance(x, (int, float, np.number))
            and np.ndim(x) == 0 and np.isfinite(x))


def poisson_fit_se(q, counts, den, fit):
    'Conditional expected-Fisher SE, without deviance/dof inflation.'
    if fit.get('status') != 'resolved_empirical_feature':
        return None
    q, y, n = (np.asarray(v, float) for v in (q, counts, den))
    if q.shape != y.shape or q.shape != n.shape:
        raise ValueError('Peak profile coordinates differ')
    p = np.asarray(fit['parameters'], float)
    if p.shape != (6,) or not np.isfinite(p).all():
        return None
    lo, hi = fit['window']; background = fit['background']
    good = (q >= lo) & (q <= hi) & (n > 0) & np.isfinite(y) & (y >= 0)
    rate = rate_model(q[good], p, background, True)
    if not np.all(rate > 0):
        return None
    J = rate_jac(q[good], p, background, True) * np.sqrt(n[good] / rate)[:, None]
    scale = np.linalg.norm(J, axis=0)
    if not np.isfinite(J).all() or np.any(scale <= 0):
        return None
    normalized = J / scale
    H = normalized.T @ normalized
    eigen = np.linalg.eigvalsh(H)
    if eigen[0] <= 1e-11 * eigen[-1]:
        return None
    covariance = np.linalg.inv(H) / scale[:, None] / scale[None, :]
    if not np.isfinite(covariance).all() or np.any(np.diag(covariance) < 0):
        return None
    se = np.sqrt(np.diag(covariance))
    return {'q_star': float(se[4]),
            'FWHM': float(2 * np.sqrt(2 * np.log(2)) * se[5]),
            'source': STAT_SOURCES[0], 'residual_inflation': False,
            'scope': 'Fitted peak/background conditional Poisson information only'}


def extract(Y, N, *, AF_band=None, estimate_stat_se=False):
    'Extract one count replica; default band is selected from observation.'
    Y, N = np.asarray(Y, float), np.asarray(N, float)
    if (not np.isfinite(Y).all() or not np.isfinite(N).all() or
            np.any(Y < 0) or np.any(N < 0) or np.any((N == 0) & (Y != 0))):
        raise ValueError('Nonnegative supported photon counts required')
    hy, hn, vy, vn, _, _ = counts_profiles(Y, N)
    fits, bands, selected, ridge = summarize_profiles(hy, hn, vy, vn, PEAK_WINDOW)
    desc = describe(hy, hn, vy, vn)
    if AF_band is None:
        AF_band = selected['AF_band'] if selected is not None else None
    if AF_band is not None and (isinstance(AF_band, (bool, np.bool_)) or
                               not isinstance(AF_band, (int, np.integer)) or AF_band not in range(4)):
        raise ValueError('AF_band must be an original observed band 0..3')
    if AF_band is not None:
        AF_band = int(AF_band)
    band = bands[AF_band] if AF_band is not None else None
    bilateral = band is not None and band['support'] == 'bilateral'
    values = {k: None for k in GATES}; se = dict(values); source = dict(values)
    reasons = {k: 'unresolved_or_missing_support' for k in GATES}
    if bilateral:
        values.update(q_star=band['q_star'], FWHM=band['FWHM'])
        reasons.update(q_star=None, FWHM=None)
        if estimate_stat_se:
            errors = [poisson_fit_se(Q, hy[2*AF_band+s], hn[2*AF_band+s],
                                    fits[2*AF_band+s]) for s in range(2)]
            if all(e is not None for e in errors):
                for key in ('q_star', 'FWHM'):


                    se[key] = math.hypot(errors[0][key], errors[1][key]) / 2
                    source[key] = STAT_SOURCES[0]
        den = hn[2*AF_band:2*AF_band+2].sum(axis=0)
        rr = np.divide(hy[2*AF_band:2*AF_band+2].sum(axis=0), den,
                       out=np.zeros(len(Q)), where=den > 0)
        below = (Q >= max(.05, .35*band['q_star'])) & (Q <= .75*band['q_star']) & (den > 0)
        pk = int(np.argmin(abs(Q-band['q_star'])))
        local = np.flatnonzero(below & (np.arange(len(Q)) > 0) & (np.arange(len(Q)) < len(Q)-1))
        local = [i for i in local if rr[i] <= rr[i-1] and rr[i] < rr[i+1]
                 and den[i-1] > 0 and den[i+1] > 0]
        if local and rr[pk] > 0:
            i = min(local, key=lambda i: rr[i])
            error = np.sqrt(rr[i]/den[i] + rr[pk]/den[pk])
            if rr[pk]-rr[i] > 2*error:
                values['valley_ratio'] = float(rr[i]/rr[pk]); reasons['valley_ratio'] = None
    if AF_band is not None:
        index = 9+AF_band
        if desc['features'][20+index] == 1:
            value = float(np.exp(desc['features'][index]))
            if np.isfinite(value) and value > 0:
                values['highq_ratio'] = value; reasons['highq_ratio'] = None
    if ridge.get('ridge_status') == 'empirical_local_maximum':
        values['Yoneda'] = ridge['alpha_f_observed_ridge']; reasons['Yoneda'] = None
    return dict(values=values, stat_se=se, stat_se_source=source,
                feature_missing_reasons=reasons, AF_band=AF_band,
                support=band['support'] if band else 'unresolved',
                bands_support=[b['support'] for b in bands],
                shared_features=desc['features'].tolist(), forward_failed=False,
                structural_release=False)


def _q(sorted_values, probability):

    return float(sorted_values[max(0, min(len(sorted_values)-1,
                                        int(math.ceil(probability*len(sorted_values)))-1))])


def _quantile_bounds(values, missing):
    values = np.sort(np.asarray(values, float))
    low = np.r_[np.full(missing, -np.inf), values]
    high = np.r_[values, np.full(missing, np.inf)]
    return {'lower_quantile_bounds': [_q(low, .025), _q(high, .025)],
            'upper_quantile_bounds': [_q(low, .975), _q(high, .975)],
            'median_bounds': [float(np.median(low)), float(np.median(high))]}


def evaluate(observed, draws, *, expected_draws):
    'Partial-identification PPC from every declared original draw.'
    draws = list(draws)
    if (isinstance(expected_draws, bool) or not isinstance(expected_draws, int) or
            expected_draws < 2 or len(draws) != expected_draws):
        raise ValueError('Every declared original draw must be represented')
    for draw in draws:
        if not isinstance(draw.get('forward_failed'), (bool, np.bool_)):
            raise ValueError('Each draw requires explicit forward_failed state')
        if draw.get('AF_band') != observed.get('AF_band'):
            raise ValueError('Predictive features must use the original observed AF band')
    failures = sum(bool(d['forward_failed']) for d in draws)
    outside = [d.get('outside_range') for d in draws]
    known_outside = [x for x in outside if isinstance(x, (bool, np.bool_))]
    output = dict(schema='effective_v2_PPC_statistics_1', original_draws=expected_draws,
                  forward_failed_draws=failures,
                  AF_band=int(observed['AF_band']) if observed.get('AF_band') is not None else None,
                  outside_range_probability=sum(known_outside)/expected_draws if len(known_outside)==expected_draws else None,
                  outside_range_unknown_draws=expected_draws-len(known_outside),
                  quantile_rule='empirical_inverse_CDF; conventional_even_sample_median',
                  missing_rule='all_completion_bounds_no_drop_no_replacement',
                  structural_release=False, features={})
    for key, base in GATES.items():
        actual = observed['values'].get(key)
        known = _finite(actual) and actual > 0
        result = dict(observed=float(actual) if known else None, applicable=bool(known),
                      base_gate=base, passed=False, missing_draws=0,
                      original_draws=expected_draws, gate_units='degree' if key == 'Yoneda' else 'relative')
        se = observed.get('stat_se', {}).get(key)
        src = observed.get('stat_se_source', {}).get(key)
        if se is not None and (not _finite(se) or se < 0 or src not in STAT_SOURCES):
            raise ValueError('Only declared count-statistical SE may widen a gate')
        result['stat_se'] = float(se) if se is not None else None
        result['stat_se_source'] = src if se is not None else None
        noise = 2*se/(1. if key == 'Yoneda' else actual) if se is not None and known else 0.
        result['effective_gate'] = max(base, noise)
        values = []
        for draw in draws:
            value = draw.get('values', {}).get(key)
            if _finite(value) and value > 0 and not draw['forward_failed']:
                values.append(float(value))
        missing = expected_draws-len(values); result['missing_draws'] = missing
        result['resolved_draws'] = len(values)
        missing_reasons = {}
        for draw in draws:
            value = draw.get('values', {}).get(key)
            if not _finite(value) or value <= 0 or draw['forward_failed']:
                reason = ('forward_failure' if draw['forward_failed'] else
                          draw.get('feature_missing_reasons', {}).get(key) or 'unresolved_feature')
                missing_reasons[reason] = missing_reasons.get(reason, 0)+1
        result['missing_draw_reasons'] = missing_reasons
        result['resolved_fraction'] = len(values)/expected_draws
        bounds = _quantile_bounds(values, missing)
        lower, upper, med = (bounds[k] for k in ('lower_quantile_bounds', 'upper_quantile_bounds', 'median_bounds'))
        envelope, core = [lower[0], upper[1]], [lower[1], upper[0]]
        for name, pair in dict(interval_envelope_95=envelope, guaranteed_core_95=core, median_bounds=med).items():
            result[name] = [float(x) if np.isfinite(x) else None for x in pair]
            result[name+'_unbounded'] = [bool(not np.isfinite(x)) for x in pair]
        inclusion = bool(known and core[0] <= actual <= core[1])
        error = (max(abs(x-actual)/(1. if key == 'Yoneda' else actual) for x in med)
                 if known and np.isfinite(med).all() else None)
        result.update(inclusion95_all_completions=inclusion, worst_median_error=error,
                      passed=bool(not failures and inclusion and error is not None and
                                  error <= result['effective_gate']+1e-12),
                      reason='forward_failure' if failures else 'unmeasurable_observation' if not known
                      else 'undefined_median' if error is None else 'outside_predictive_95_core' if not inclusion
                      else 'median_feature_mismatch' if error > result['effective_gate']+1e-12 else 'passed_feature_check')
        output['features'][key] = result
    output['D_feature_check_pass'] = bool(observed.get('support') == 'bilateral' and output['features']['q_star']['passed'])
    output['diameter_feature_check_pass'] = bool(output['D_feature_check_pass'] and
                                               output['features']['FWHM']['passed'] and
                                               output['features']['highq_ratio']['passed'])
    applicable = [v for v in output['features'].values() if v['applicable']]
    output['all_measurable_feature_checks_pass'] = bool(applicable and all(v['passed'] for v in applicable))
    return output
