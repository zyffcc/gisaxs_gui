"""Build research/fit1d/ood_stats.json from DEV only (data/dev_v1.npz).

    python3 -m research.fit1d.make_ood_stats [--dev ../data/dev_v1.npz] [--workers 9] [--no-quality]
                                             [--cache FILE.json] [--from-cache]

ood: the five input features of flags.input_features on every DEV curve (native arrays, padding
  removed, the curve's own sigma); per-feature mean and scale (standard deviation, floored at 0.02 so
  a nearly constant feature cannot dominate), the inverse of their correlation matrix (ridge 1e-6), and
  the 99.5th percentile of the squared Mahalanobis distance over DEV as the threshold.
quality: reference fits = product physics (TabFit number/hosemann in CurveCodec(q_min, q_max), qz and
  mode from DEV, Lorentz nu fixed at 2 in GISAXS mode) started from the DEV truth and refined for 40 LM
  function evaluations. Thresholds use only the model-matched curves (SAXS, and GISAXS whose true nu is 2:
  the product fixes nu = 2, so a curve with another nu is a model mismatch, not a reference), split into
  N_STRATA equal-count strata of the input feature log10_median_relative_sigma (the S/N of the curve: DEV
  carries 0-5 % smooth systematic errors that are not in sigma, so the expected chi2/pt grows with S/N).
  Per stratum: chi2_per_point_threshold = 99th percentile of the reference chi2/pt; runs_z_threshold =
  1st percentile of the runs-test z (flags.runs_test on the points chosen by flags.runs_selection); the
  runs test fails only when p < 1e-3 and z is below this threshold. 'fallback' holds the same two numbers
  over all matched curves (used when the feature is not finite). rates_at_reference lists the resulting
  flag rates at the reference fits (matched, unmatched, by mode), i.e. the in-distribution false-flag rate.
--cache stores/reads the per-curve reference results (row, chi2/pt, runs z, seconds), so the thresholds
can be re-derived (--from-cache) without refitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from research.fit1d import flags as fl

DEFAULT_DEV = Path(__file__).resolve().parents[3]/'data'/'dev_v1.npz'
SCALE_FLOOR = 0.02
OOD_PERCENTILE = 99.5
QUALITY_PERCENTILE = 99.0
N_STRATA = 8
STEPS = 40


def dev_arrays(path):
    with np.load(path) as f:
        return {k: f[k] for k in ('native_q', 'native_observed', 'native_sigma', 'native_mask', 'u', 'combo', 'qz',
                                  'gisaxs', 'q_min', 'q_max', 'peak', 'systematic')}


def curve(d, i):
    m = d['native_mask'][i]
    return d['native_q'][i][m], d['native_observed'][i][m], d['native_sigma'][i][m]


def _init():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def reference_fit(args):
    """chi2/pt and runs-test statistics of the truth-started product fit of one DEV curve."""
    row, q, y, s, u, combo, qz, gisaxs, q_min, q_max = args
    from research.cond_v2.codec import CurveCodec
    from research.conditional_fit.pipeline import _clean_inputs
    from research.fit1d.constraints import ConstrainedCodec
    from research.fit1d.controller import TracedFit
    from research.multishape_v1.physics import COMBOS
    q, y, s = _clean_inputs(q, y, s)
    types = COMBOS[int(combo)]
    base = CurveCodec(float(q.min()), float(q.max()))
    codec = ConstrainedCodec(base, types, {}, nu=('fixed', 2.0) if gisaxs else None)
    fit = TracedFit(q, y, s, types, qz=float(qz) if gisaxs else 0.0, sf_on=bool(gisaxs), res_on=bool(gisaxs),
                    codec=codec, lowq='lorentz')
    u0 = codec.encode(CurveCodec(float(q_min), float(q_max)).decode(np.asarray(u, float)))
    t0 = time.perf_counter()
    r = fit.run(u0, fixed=codec.fixed_indices, max_nfev=STEPS)
    coef, prediction, residual = fit.solve(fit.matrix_from(*fit.full(r['u'])))
    runs = fl.runs_test(residual, fl.runs_selection(prediction, s))
    runs_all = fl.runs_test(residual)
    return dict(row=int(row), chi2_per_point=float(residual@residual)/len(q), runs_z=runs['z'], runs_p=runs['p'],
                runs_points_used=runs['n_used'], runs_all_p=runs_all['p'], n_points=int(len(q)),
                seconds=time.perf_counter()-t0)


def ood_stats(d):
    n = len(d['combo'])
    feats = np.array([fl.input_features(*curve(d, i)) for i in range(n)])
    ok = np.all(np.isfinite(feats), axis=1)
    x = feats[ok]
    mean = x.mean(0)
    scale = np.maximum(x.std(0), SCALE_FLOOR)
    z = (x-mean)/scale
    corr = np.corrcoef(z, rowvar=False)+1e-6*np.eye(z.shape[1])
    inv = np.linalg.inv(corr)
    d2 = np.einsum('ij,jk,ik->i', z, inv, z)
    return dict(feature_names=list(fl.FEATURE_NAMES), mean=mean.tolist(), scale=scale.tolist(), scale_floor=SCALE_FLOOR,
                inverse_correlation=inv.tolist(), threshold_d2=float(np.percentile(d2, OOD_PERCENTILE)),
                percentile=OOD_PERCENTILE, n_curves=int(ok.sum()), n_nonfinite=int((~ok).sum()),
                dev_d2_percentiles={str(p): float(np.percentile(d2, p)) for p in (50, 90, 99, 99.5, 100)},
                feature_min=x.min(0).tolist(), feature_max=x.max(0).tolist()), feats


def reference_fits(d, workers, cache=None, log=print):
    """Per-curve reference results (list ordered by row); resumes from / writes to the cache file."""
    n = len(d['combo'])
    done = {}
    if cache is not None and Path(cache).exists():
        done = {int(r['row']): r for r in json.loads(Path(cache).read_text(encoding='utf-8'))}
    jobs = [(i, *curve(d, i), d['u'][i], int(d['combo'][i]), float(d['qz'][i]), bool(d['gisaxs'][i]),
             float(d['q_min'][i]), float(d['q_max'][i])) for i in range(n) if i not in done]

    def save():
        if cache is not None:
            Path(cache).write_text(json.dumps([done[k] for k in sorted(done)]), encoding='utf-8')
    if jobs:
        started = time.perf_counter()
        if workers > 1:
            from multiprocessing import get_context
            with get_context('spawn').Pool(workers, initializer=_init) as pool:
                for k, r in enumerate(pool.imap_unordered(reference_fit, jobs, chunksize=1), 1):
                    done[r['row']] = r
                    if k % 100 == 0 or k == len(jobs):
                        save()
                        log(f'reference fits {k}/{len(jobs)} ({time.perf_counter()-started:.0f} s)', flush=True)
        else:
            _init()
            for job in jobs:
                r = reference_fit(job)
                done[r['row']] = r
        save()
    return [done[i] for i in range(n)]


def true_nu(d):
    from research.cond_v2.codec import CurveCodec
    return np.array([float(CurveCodec(float(a), float(b)).decode(np.asarray(u, float))['nu_res'])
                     for a, b, u in zip(d['q_min'], d['q_max'], d['u'])])


def quality_stats(d, records, feats):
    """Stratified thresholds from the model-matched reference fits (see module docstring)."""
    chi = np.array([r['chi2_per_point'] for r in records])
    z = np.array([np.nan if r['runs_z'] is None else r['runs_z'] for r in records])
    p_all = np.array([np.nan if r['runs_all_p'] is None else r['runs_all_p'] for r in records])
    g = d['gisaxs'].astype(bool)
    matched = ~g | np.isclose(true_nu(d), 2.0, atol=1e-6)
    x = feats[:, list(fl.FEATURE_NAMES).index(fl.STRATUM_FEATURE)]
    usable = matched & np.isfinite(x) & np.isfinite(chi)
    edges = np.percentile(x[usable], np.linspace(0, 100, N_STRATA+1)[1:-1])
    stratum = np.searchsorted(edges, x, side='right')

    def thresholds(sel):
        zz = z[sel & np.isfinite(z)]
        return (float(np.percentile(chi[sel], QUALITY_PERCENTILE)),
                float(min(np.percentile(zz, 100-QUALITY_PERCENTILE), fl.RUNS_Z_AT_P)) if len(zz) else fl.RUNS_Z_AT_P)
    chi_thr, z_thr, n_ref = [], [], []
    for i in range(N_STRATA):
        sel = usable & (stratum == i)
        a, b = thresholds(sel)
        chi_thr.append(a)
        z_thr.append(b)
        n_ref.append(int(sel.sum()))
    fallback = thresholds(usable)
    chi_of = np.array(chi_thr)[stratum]
    z_of = np.array(z_thr)[stratum]
    high = chi > chi_of
    runs_fail = np.nan_to_num(z, nan=0.0) < z_of        # z_of <= z(p = 1e-3), so p < 1e-3 is implied
    old_runs_fail = np.nan_to_num(p_all, nan=1.0) < fl.RUNS_P_THRESHOLD
    peak = d['peak']

    def rates(sel):
        if not sel.any():
            return dict(n=0)
        return dict(n=int(sel.sum()), chi2_high=float(np.mean(high[sel])), runs_fail=float(np.mean(runs_fail[sel])),
                    flag=float(np.mean(high[sel] | runs_fail[sel])),
                    runs_test_applicable=float(np.mean(np.isfinite(z[sel]))),
                    runs_fail_fixed_p_selected_points=float(np.mean(np.nan_to_num(z[sel], nan=0.0) < fl.RUNS_Z_AT_P)),
                    runs_fail_fixed_p_all_points=float(np.mean(old_runs_fail[sel])),
                    chi2_per_point_median=float(np.median(chi[sel])))
    return dict(version=2, stratify_by=fl.STRATUM_FEATURE, edges=[float(e) for e in edges],
                chi2_per_point_threshold=chi_thr, runs_z_threshold=z_thr, n_reference=n_ref,
                fallback=dict(chi2_per_point_threshold=fallback[0], runs_z_threshold=fallback[1]),
                percentile=QUALITY_PERCENTILE, runs_p_threshold=fl.RUNS_P_THRESHOLD,
                runs_selection=dict(min_expected=fl.RUNS_MIN_EXPECTED, smooth_window=fl.RUNS_SMOOTH_WINDOW,
                                    min_points=fl.RUNS_MIN_POINTS),
                reference=('product physics started from the DEV truth, 40 LM function evaluations, nu fixed at 2 '
                           '(GISAXS), qz and mode from DEV; thresholds from the model-matched curves only '
                           '(SAXS, and GISAXS with true nu = 2)'),
                n_matched=int(usable.sum()), n_unmatched=int((~matched).sum()),
                dev_chi2_per_point_percentiles_matched={str(q): float(np.percentile(chi[usable], q))
                                                        for q in (50, 90, 95, 99, 100)},
                rates_at_reference=dict(all=rates(np.ones(len(chi), bool)), matched=rates(matched),
                                        unmatched_nu_not_2=rates(~matched), gisaxs_matched=rates(matched & g),
                                        saxs=rates(~g), matched_peak_below_1e3=rates(matched & (peak < 1e3)),
                                        matched_peak_1e3_1e4=rates(matched & (peak >= 1e3) & (peak < 1e4)),
                                        matched_peak_1e4_1e5=rates(matched & (peak >= 1e4) & (peak < 1e5)),
                                        matched_peak_above_1e5=rates(matched & (peak >= 1e5))),
                seconds_total=float(sum(r['seconds'] for r in records)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dev', type=Path, default=DEFAULT_DEV)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--no-quality', action='store_true')
    p.add_argument('--cache', type=Path, default=None, help='per-curve reference results (JSON), resumable')
    p.add_argument('--from-cache', action='store_true', help='derive the thresholds from --cache without fitting')
    p.add_argument('--out', type=Path, default=fl.STATS_PATH)
    a = p.parse_args(argv)
    d = dev_arrays(a.dev)
    ood, feats = ood_stats(d)
    stats = dict(version=2, built='make_ood_stats.py', dev_file=a.dev.name,
                 dev_sha256=hashlib.sha256(a.dev.read_bytes()).hexdigest(), n_dev=int(len(d['combo'])), ood=ood)
    if not a.no_quality:
        if a.from_cache:
            records = json.loads(a.cache.read_text(encoding='utf-8'))
            if len(records) != len(d['combo']):
                raise SystemExit(f'{a.cache}: {len(records)} records for {len(d["combo"])} DEV curves; run without --from-cache')
        else:
            records = reference_fits(d, a.workers, a.cache)
        stats['quality'] = quality_stats(d, records, feats)
    elif a.out.exists():
        old = json.loads(a.out.read_text(encoding='utf-8'))
        if 'quality' in old:
            stats['quality'] = old['quality']
    a.out.write_text(json.dumps(stats, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in stats.items() if k not in ('ood',)}, indent=1)[:6000])
    print('ood threshold d2', stats['ood']['threshold_d2'], stats['ood']['dev_d2_percentiles'])


if __name__ == '__main__':
    main()
