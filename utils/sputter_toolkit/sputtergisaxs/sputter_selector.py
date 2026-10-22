"""Learned choice among the polymer network's explanations of each window (protocol sputter_selector_v1).

A conditional logit P(c) = softmax(beta . x_c) over the 4 mixture components
of every window (branches merged, weights summed). Features: component index, relative network weight, top-weight flag,
agreement of D with the direct 2pi/q*, temporal support of R and D from the
candidates of the three previous and three following windows, and the
candidate's ln R relative to its window. Coefficients live in a JSON model;
inference needs NumPy only. Trained and scored on simulated growth sequences
(scripts/train_selector_v1.py); film (TS) windows keep the observation rule.

Example:
    from sputtergisaxs.sputter_selector import selector_table
    selector_table("OUT/binned", "OUT", metal="Ag", polymer_expert="experts/polymer", ts_expert="experts/TS",
                   model="configs/sputter_selector_v1.json")
"""
from __future__ import annotations
from pathlib import Path
import csv, json, math
import numpy as np

FEATURES = ('logw_rel', 'is_top', 'lnD_over_direct', 'lnD_over_direct_sq', 'R_support_past', 'R_support_future',
            'D_support_past', 'D_support_future', 'lnR_rel_window', 'component_1', 'component_2', 'component_3',
            'log_coverage_own', 'log_coverage_direct', 'R_support_past_far', 'R_support_future_far')
KERNEL_SD = 0.15
NEIGHBOURS = 3
FAR = (4, 10)   # second, longer temporal scale (windows)


def collapse_components(cands):
    """Merge the branches of each mixture component: summed weight, values of its heaviest branch.

    The branches of one component give near-identical centres; keeping them
    separately splits the selector's probability between duplicates.
    """
    groups = {}
    for c in cands:
        k = int(c['mode'].split(':')[1]); groups.setdefault(k, []).append(c)
    out = []
    for k in sorted(groups):
        g = groups[k]; best = max(g, key=lambda c: c['logw'])
        lw = np.array([c['logw'] for c in g]); total = float(lw.max() + np.log(np.exp(lw - lw.max()).sum()))
        out.append(dict(best, logw=total, mode=f'k{k}', component=k, branches=len(g)))
    return out


def _arrays(window):
    c = window['candidates']
    lw = np.array([x['logw'] for x in c]); w = np.exp(lw - lw.max()); w /= w.sum()
    lnR = np.log([max(x['values'].get('R_number_mean_nm', np.nan), 1e-6) for x in c])
    lnD = np.log([max(x['values'].get('D_nm', np.nan), 1e-6) for x in c])
    sig = np.array([x['values'].get('radius_sigma_log', 0.) or 0. for x in c])
    return lw, w, lnR, lnD, sig


def _log_coverage(lnR, sig, lnD):
    """log projected coverage of hemisphere bases on a hexagonal arrangement (number-mean R, lognormal sigma)."""
    return math.log(math.pi * 2 / math.sqrt(3)) + 2 * lnR + sig ** 2 - 2 * lnD


def _support(values, others):
    """log of the mean over neighbour windows of the weight-averaged Gaussian kernel to their candidates."""
    if not others:
        return None
    acc = np.zeros(len(values))
    for w2, v2 in others:
        acc += (w2[None, :] * np.exp(-.5 * ((values[:, None] - v2[None, :]) / KERNEL_SD) ** 2)).sum(1)
    return np.log(acc / len(others) + 1e-6)


def window_features(windows):
    """windows: time-ordered [{candidates (polymer), D_corr, peak}] -> list of (K x F) feature arrays."""
    arr = [_arrays(w) for w in windows]
    out = []
    for t, w in enumerate(windows):
        lw, wt, lnR, lnD, sig = arr[t]
        past = [(arr[t - o][1], arr[t - o]) for o in range(1, NEIGHBOURS + 1) if t - o >= 0]
        future = [(arr[t + o][1], arr[t + o]) for o in range(1, NEIGHBOURS + 1) if t + o < len(windows)]
        Rp = _support(lnR, [(a[1], a[2]) for _, a in past]); Rf = _support(lnR, [(a[1], a[2]) for _, a in future])
        Dp = _support(lnD, [(a[1], a[3]) for _, a in past]); Df = _support(lnD, [(a[1], a[3]) for _, a in future])
        far_p = [(arr[t - o][1], arr[t - o][2]) for o in range(FAR[0], FAR[1] + 1) if t - o >= 0]
        far_f = [(arr[t + o][1], arr[t + o][2]) for o in range(FAR[0], FAR[1] + 1) if t + o < len(windows)]
        Rpf = _support(lnR, far_p); Rff = _support(lnR, far_f)
        Rpf = Rpf if Rpf is not None else (Rff if Rff is not None else np.zeros(len(lw)))
        Rff = Rff if Rff is not None else Rpf
        cov_own = _log_coverage(lnR, sig, lnD)
        cov_dir = _log_coverage(lnR, sig, math.log(w['D_corr'])) if (w.get('peak') and w.get('D_corr')) else np.zeros(len(lw))
        Rp = Rp if Rp is not None else (Rf if Rf is not None else np.zeros(len(lw)))
        Rf = Rf if Rf is not None else Rp
        Dp = Dp if Dp is not None else (Df if Df is not None else np.zeros(len(lw)))
        Df = Df if Df is not None else Dp
        if w.get('peak') and w.get('D_corr'):
            dd = lnD - math.log(w['D_corr'])
        else:
            dd = np.zeros(len(lw))
        comp = np.array([c.get('component', 0) for c in w['candidates']])
        X = np.column_stack([lw - lw.max(), (lw == lw.max()).astype(float), dd, dd ** 2, Rp, Rf, Dp, Df,
                             lnR - (wt * lnR).sum(), comp == 1, comp == 2, comp == 3,
                             cov_own, cov_dir, Rpf, Rff]).astype(float)
        out.append(np.nan_to_num(X, nan=0., posinf=0., neginf=-20.))
    return out


def fit(groups, labels, l2=1e-2):
    """Conditional logit by L-BFGS on standardized features; groups: list of (K x F) arrays."""
    from scipy.optimize import minimize
    allX = np.concatenate(groups); mean = allX.mean(0); std = allX.std(0); std[std < 1e-9] = 1.
    Z = [(g - mean) / std for g in groups]

    def nll(beta):
        total, grad = 0., np.zeros_like(beta)
        for z, y in zip(Z, labels):
            s = z @ beta; s -= s.max(); p = np.exp(s); p /= p.sum()
            total -= math.log(max(p[y], 1e-300)); grad -= z[y] - p @ z
        n = len(Z)
        return total / n + .5 * l2 * beta @ beta, grad / n + l2 * beta
    res = minimize(nll, np.zeros(allX.shape[1]), jac=True, method='L-BFGS-B')
    return dict(version='sputter_selector_v1', features=list(FEATURES), mean=mean.tolist(), std=std.tolist(),
                beta=res.x.tolist(), l2=l2, train_nll=float(res.fun), converged=bool(res.success), groups=len(groups))


def probabilities(model, X):
    z = (X - np.asarray(model['mean'])) / np.asarray(model['std'])
    s = z @ np.asarray(model['beta']); s -= s.max(); p = np.exp(s)
    return p / p.sum()


def choose(model, windows):
    feats = window_features(windows)
    out = []
    for X in feats:
        p = probabilities(model, X); i = int(np.argmax(p)); out.append((i, float(p[i])))
    return out


def load_model(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# ------------------------------------------------------------------ real scans --
def selector_rows(binned, *, metal, polymer_expert, ts_expert, model, device='cpu'):
    from .neural_refine_infer import NRExpert
    from .sputter_sequence import load_windows, mixture_candidates
    from .sputter_struct import reliability
    model = load_model(model) if not isinstance(model, dict) else model
    windows, encoded = load_windows(binned)
    poly = NRExpert(Path(polymer_expert) / 'model/best.pt', Path(polymer_expert) / 'policy.json', device=device)
    film = NRExpert(Path(ts_expert) / 'model/best.pt', Path(ts_expert) / 'policy.json', device=device)
    pc, tc = mixture_candidates(poly, encoded), mixture_candidates(film, encoded)
    for w, c in zip(windows, pc):
        w['candidates'] = collapse_components(c)
    picks = choose(model, windows)
    rel = reliability()['parameters']; model_metal = 'Ag' if metal == 'Cu' else metal
    stats = model.get('test_reliability', {}); band = rel.get('D_over_direct', {}).get('band_90', [0.587, 1.75])
    rows = []
    for w, (i, p), tcands in zip(windows, picks, tc):
        c = w['candidates'][i]; v = c['values']; top = max(w['candidates'], key=lambda x: x['logw'])
        film_window = w['rule_module'] == 'TS_layer'
        row = dict(frame=w['key'], time_s=w['time_s'], stage=w['stage'], module='TS_layer' if film_window else 'polymer_embed',
                   feature_status=w['feature_status'], D_corr_nm=w['D_corr'], selector_choice=c['mode'],
                   selector_probability=p, selector_low_confidence=p < .5,
                   network_top_R_nm=top['values'].get('R_number_mean_nm'), network_top_D_nm=top['values'].get('D_nm'),
                   metal=metal)
        if film_window:
            ts = max(tcands, key=lambda x: x['logw'])['values']
            row.update(R_nm=None, R_grade=None, R_reason='late TS window (observation rule): island R/D not defined',
                       D_nm=None, D_grade=None, D_reason='late TS window', L_TS_nm=ts.get('spectral_domain_nm'),
                       TS_metal_fraction_conditional=ts.get('metal_fraction'), TS_film_thickness_nm_conditional=ts.get('film_thickness_nm'))
        else:
            for name, key in (('R', 'R_number_mean_nm'), ('D', 'D_nm')):
                value = v.get(key)
                if name == 'D' and not w['peak']:
                    row.update(D_nm=None, D_grade=None, D_reason='no resolved peak: D not reported', D_direct_check=None)
                    continue
                s = stats.get(name, {}).get(model_metal) or stats.get(name, {}).get('all') or {}
                err = s.get('typical_rel_scatter')
                row.update({f'{name}_nm': value, f'{name}_source': 'selector', f'{name}_grade': 'L', f'{name}_reason': '',
                            f'{name}_typical_rel_error': err,
                            f'{name}_lo_nm': value / (1 + err) if value and err is not None else None,
                            f'{name}_hi_nm': value * (1 + err) if value and err is not None else None,
                            f'{name}_P90_rel_error': s.get('P90_rel_error'), f'{name}_order_accuracy': s.get('order_accuracy_1p2x')})
            if row.get('D_nm') and w['D_corr']:
                ratio = row['D_nm'] / w['D_corr']
                row.update(D_over_direct=ratio, D_direct_check='consistent' if band[0] <= ratio <= band[1]
                           else 'outside simulated 90% range of D/(2pi/q*)')
            row.update(sigma_log=v.get('radius_sigma_log'), burial_nm_conditional=v.get('burial_base_depth_nm'),
                       thickness_nm_conditional=v.get('polymer_thickness_nm'), spacing_disorder=None, L_TS_nm=None,
                       TS_metal_fraction_conditional=None, TS_film_thickness_nm_conditional=None)
        row['note'] = 'learned selector (sputter_selector_v1); typical errors from its independent simulated TEST'
        rows.append(row)
    return rows


def selector_table(binned, out, *, metal, polymer_expert, ts_expert, model, device='cpu'):
    out = Path(out)
    rows = selector_rows(binned, metal=metal, polymer_expert=polymer_expert, ts_expert=ts_expert, model=model, device=device)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    path = out / 'structure_selected.csv'
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    from .sputter_struct import summary, plot_structure
    s = summary(rows, metal)
    m = load_model(model) if not isinstance(model, dict) else model
    s['selector'] = dict(protocol='sputter_selector_v1', model_sha_hint=m.get('sha256'), test=m.get('test_summary'),
                         low_confidence=sum(bool(r.get('selector_low_confidence')) for r in rows if r['module'] != 'TS_layer'),
                         changed_from_network_top=sum(r.get('R_nm') is not None and r.get('network_top_R_nm') is not None
                                                      and abs(r['R_nm'] / r['network_top_R_nm'] - 1) > 1e-6 for r in rows))
    (out / 'structure_selected_summary.json').write_text(json.dumps(s, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    plot_structure(rows, out / 'structure_selected_trend.png', title=Path(out).name + ' (learned selector)')
    return path
