"""Experimental per-candidate stopping; never drops component or parameter heads.

Only measured intensity and supplied sigma enter stopping decisions. A pass is
an empirical synthetic-domain diagnostic, not a posterior or a clean-error bound.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from gpu_precision import forward, KEYS
from flow_codec import encode, decode, COMBOS
from diagnose_v5 import add_curves
from noise_quality import quality_metrics
from solution_output_r2 import fits

POLICIES = {
    'full': dict(initial_only=False, scale=None),
    'skip_strict': dict(initial_only=True, scale=1.),
    'early_strict': dict(initial_only=False, scale=1.),
    'early_guard': dict(initial_only=False, scale=.5),
}


class QualityReached(Exception):
    pass


def refine_adaptive(data, cand, policy='early_guard', max_nfev=80):
    if policy not in POLICIES:
        raise ValueError(f'Unknown policy: {policy}')
    setting = POLICIES[policy]
    rule = dict(cand['output_policy']['rules']['strict'])
    if setting['scale'] is not None:
        rule['rms'] *= setting['scale']
        rule['local'] *= setting['scale']
    n, h = cand['combos'].shape
    wl = cand['weights'].reshape(-1, 4)
    w = np.exp(wl-wl.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    encoded = encode(COMBOS[cand['combos'].reshape(-1)],
                     cand['params'].reshape(-1, 4, 6), w,
                     cand['globals'].reshape(-1, 4),
                     cand['d'].reshape(-1, 4), cand['res'].reshape(-1))
    final = encoded['theta'].copy(); records = []
    for i in range(n):
        qq = np.asarray(data['q'][i:i+1], 'float32')
        mm = np.asarray(data['mask'][i:i+1], 'float32'); valid = mm[0] > 0
        target = np.log(np.maximum(data['observed'][i, valid], 1e-30)).astype('float64')
        for j in range(h):
            flat = i*h+j; c = encoded['combo'][flat]; g = encoded['gate'][flat]
            columns = np.flatnonzero(encoded['active_mask'][flat])
            template = encoded['theta'][flat].copy()
            tick = time.perf_counter(); calls = [0, 0]; checks = [0]
            best = [np.inf, template.copy()]; initial_error = [None]
            reason = 'solver_finished'; status = None

            def curves(values):
                count = len(values)
                return forward(values, np.full(count, c, 'int32'),
                               np.full(count, g, 'int32'), np.repeat(qq, count, 0),
                               np.repeat(mm, count, 0)).numpy()

            def residual(v):
                x = template.copy(); x[columns] = v; y = curves(x[None])
                r = np.log(np.maximum(y[0, valid], 1e-30)).astype('float64')-target
                loss = float(np.mean(r*r)); calls[0] += 1
                if initial_error[0] is None:
                    initial_error[0] = float(np.sqrt(loss))
                improved = loss < best[0]
                if improved:
                    best[:] = [loss, x.copy()]
                check_now = setting['scale'] is not None and (
                    calls[0] == 1 or (improved and not setting['initial_only']))
                if check_now:
                    checks[0] += 1
                    qm = quality_metrics(y, data['observed'][i], data['sigma'][i], mm[0])
                    if fits({k: float(v[0]) for k, v in qm.items()}, rule):
                        raise QualityReached()
                return r/np.sqrt(len(target))

            def jacobian(v):
                x = template.copy(); x[columns] = v; eps = .001
                batch = np.repeat(x[None], 2*len(columns), 0)
                batch[np.arange(len(columns)), columns] += eps
                batch[len(columns)+np.arange(len(columns)), columns] -= eps
                y = np.log(np.maximum(curves(batch)[:, valid], 1e-30)).astype('float64')
                calls[1] += 1
                return ((y[:len(columns)]-y[len(columns):])/(2*eps)).T/np.sqrt(len(target))

            try:
                residual(template[columns])
                limits = np.where(np.abs(template) < 15., 15., np.abs(template)+1.)
                result = least_squares(residual, template[columns].astype('float64'),
                                       jac=jacobian, bounds=(-limits[columns], limits[columns]),
                                       method='trf', x_scale='jac', ftol=1e-6,
                                       xtol=1e-5, gtol=1e-6, max_nfev=int(max_nfev))
                status = int(result.status)
            except QualityReached:
                reason = 'initial_quality_pass' if calls[0] == 1 else 'quality_early_stop'
            final[flat] = best[1]
            records.append(dict(row=i, head=j, policy=policy, stop_reason=reason,
                                initial_observed_error=initial_error[0],
                                final_observed_error=float(np.sqrt(best[0])),
                                forward_calls=calls[0], jacobian_calls=calls[1],
                                quality_checks=checks[0], status=status,
                                seconds=time.perf_counter()-tick))
    dec = decode(final, encoded['combo'], encoded['gate'])
    out = {k: v.reshape(n, h, *v.shape[1:]) for k, v in dec.items()}
    out = add_curves(data, out)
    for key in ('classifier_nlp', 'output_policy', 'components_condition'):
        if key in cand: out[key] = cand[key]
    out['theta'] = final.reshape(n, h, 31)
    out['gate_pattern'] = encoded['gate'].reshape(n, h)
    return out, records
