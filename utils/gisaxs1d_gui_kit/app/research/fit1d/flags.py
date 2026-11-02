"""Result flags of fit1d (PLAN_zh.md §1 flag set {at_bound, weakly_constrained, quality, ood, fallback}).

  at_bound            free nonlinear coordinates of active terms within BOUND_TOL (controller) of an edge
                      of their box (list of names such as 'pop2.D_nm@lower'); any entry raises the flag.
                      at_bound_reportable lists the R_nm/D_nm among them (information only)
  weakly_constrained  list of parameters 'popN.key' that the curve does not pin down, with the reasons in
                      weakly_constrained_detail:
                        relative_error  relative 1-sigma > 50 %
                        no_error        no finite 1-sigma (direction not constrained by the data)
                        at_bound        the coordinate sits at a bound
                        inactive        the population has NNLS amplitude 0 (it does not contribute)
                        multimodal      another LM solution of this fit with an equivalent chi2 (PLAN same-basin
                                        tolerance) has this R_nm/D_nm differing by > 5 sigma (see alternatives;
                                        threshold tuned on DEV, controller.MULTIMODAL_SIGMA)
                      weakly_constrained_reportable is the subset of reportable parameters (R_nm, D_nm); only
                      this subset counts in flags['raised'] (sigma_R, w and h are qualitative outputs, PLAN §1).
                      A preview (policy 'preview') has no uncertainties: the list is empty there.
  quality             chi2/pt above the DEV threshold of the curve's S/N stratum, or the residual signs fail the
                      Wald-Wolfowitz runs test (too few runs): one-sided p < 1e-3 and z below the DEV envelope
                      of the stratum. The runs test uses only points with (model / smoothed sigma)^2 >= 10
                      (>= 10 expected counts for Poisson data; selected by the model, not by the data, because
                      at low counts the residual signs of an exact model are not exchangeable) and needs >= 20
                      such points.
  sigma_assumed       no sigma was given; Poisson sqrt(max(I, 1)) was used (chi2 then has no absolute scale)
  fallback            the robust stage S3 ran for this composition
  ood                 Mahalanobis distance of the input-feature vector (input_features) from the DEV
                      feature distribution above its DEV 99.5th percentile
Thresholds come only from DEV (data/dev_v1.npz) and are stored in ood_stats.json by make_ood_stats.py.
The S/N stratum is the bin of the input feature log10_median_relative_sigma; per stratum the chi2/pt
threshold is the 99th percentile and the runs threshold the 1st percentile of z (never above the z of
p = 1e-3) over the model-matched DEV reference fits (see make_ood_stats.py).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

STATS_PATH = Path(__file__).with_name('ood_stats.json')
FEATURE_NAMES = ('log10_q_range', 'log10_n_points', 'log10_dynamic_range', 'log10_median_relative_sigma',
                 'fraction_invalid_or_nonpositive')
STRATUM_FEATURE = 'log10_median_relative_sigma'
RUNS_P_THRESHOLD = 1e-3
RUNS_Z_AT_P = -3.090232306167813          # z with one-sided p = 1e-3
RUNS_MIN_EXPECTED = 10.0                  # (model / smoothed sigma)^2 of the points entering the runs test
RUNS_SMOOTH_WINDOW = 21                   # running-median window (points) of the sigma used for that selection
RUNS_MIN_POINTS = 20
WEAK_RELATIVE = 0.5
FLAG_SET = ('at_bound', 'weakly_constrained', 'quality', 'ood', 'fallback')
REPORTABLE = ('R_nm', 'D_nm')
POPULATION_KEYS = ('R_nm', 'sigma_R_rel', 'D_nm', 'w_rel', 'h_nm', 'sigma_h_rel')


def input_features(q, y, sigma):
    """Five scale-free descriptors of a raw input curve (before cleaning), in FEATURE_NAMES order.

    q_range = q_max/q_min of the valid points; dynamic range = 99.5th/0.5th percentile of positive I;
    relative sigma = median sigma/I over positive I; the last one counts non-finite, q <= 0, sigma <= 0
    and I <= 0 points among all given points (zero counts included)."""
    q, y, sigma = (np.asarray(v, float).ravel() for v in (q, y, sigma))
    n_raw = max(len(q), 1)
    valid = np.isfinite(q) & np.isfinite(y) & np.isfinite(sigma) & (q > 0) & (sigma > 0)
    good = valid & (y > 0)
    if good.sum() < 3:
        return [float('nan')]*len(FEATURE_NAMES)
    qv = q[valid]
    yp = y[good]
    lo, hi = np.percentile(yp, [0.5, 99.5])
    return [float(np.log10(qv.max()/qv.min())), float(np.log10(valid.sum())),
            float(np.log10(hi/max(lo, 1e-300))), float(np.log10(np.median(sigma[good]/yp))),
            float(1.0-good.sum()/n_raw)]


@lru_cache(maxsize=1)
def load_stats():
    """DEV statistics (dict) or None when ood_stats.json is missing."""
    if not STATS_PATH.exists():
        return None
    return json.loads(STATS_PATH.read_text(encoding='utf-8'))


def ood_check(features):
    """{'ood': bool|None, 'distance2', 'threshold', 'z': per-feature standardized values, ...}."""
    stats = load_stats()
    if stats is None or 'ood' not in stats:
        return dict(ood=None, available=False, note='ood_stats.json missing: run python3 -m research.fit1d.make_ood_stats')
    s = stats['ood']
    x = np.asarray(features, float)
    if not np.all(np.isfinite(x)):
        return dict(ood=True, available=True, distance2=None, threshold=s['threshold_d2'],
                    note='features not finite (fewer than 3 positive points)')
    mean, scale = np.asarray(s['mean']), np.asarray(s['scale'])
    z = (x-mean)/scale
    d2 = float(z@np.asarray(s['inverse_correlation'])@z)
    return dict(ood=bool(d2 > s['threshold_d2']), available=True, distance2=d2, threshold=s['threshold_d2'],
                z={k: float(v) for k, v in zip(FEATURE_NAMES, z)})


def runs_selection(prediction, sigma):
    """Points entering the runs test: (model / running-median sigma)^2 >= RUNS_MIN_EXPECTED.

    Selected by the model and a smoothed sigma, never by the data: with a data-derived sigma (Poisson
    sqrt(max(I, 1))) a selection on I/sigma correlates with the noise and biases the sign sequence."""
    from scipy.ndimage import median_filter
    sigma = np.asarray(sigma, float)
    smooth = median_filter(sigma, size=min(RUNS_SMOOTH_WINDOW, max(len(sigma), 1)), mode='nearest')
    return (np.asarray(prediction, float)/np.maximum(smooth, 1e-300))**2 >= RUNS_MIN_EXPECTED


def runs_test(residual, select=None):
    """One-sided Wald-Wolfowitz runs test on residual signs (points in q order): p of this few runs or fewer.

    select: boolean mask of the points to use (runs_selection); fewer than RUNS_MIN_POINTS selected points
    -> not applicable (z and p None)."""
    r = np.asarray(residual, float)
    if select is not None:
        r = r[np.asarray(select, bool)]
        if len(r) < RUNS_MIN_POINTS:
            return dict(runs=None, expected=None, z=None, p=None, n_used=int(len(r)))
    signs = np.sign(r[r != 0])
    n1, n2 = int(np.sum(signs > 0)), int(np.sum(signs < 0))
    n = n1+n2
    if n1 == 0 or n2 == 0 or n < 3:
        return dict(runs=int(n > 0), expected=None, z=None, p=None, n_used=n)
    runs = int(1+np.sum(signs[1:] != signs[:-1]))
    mu = 2.0*n1*n2/n+1
    var = 2.0*n1*n2*(2.0*n1*n2-n)/(n*n*(n-1))
    if var <= 0:
        return dict(runs=runs, expected=mu, z=None, p=None, n_used=n)
    z = (runs-mu)/math.sqrt(var)
    return dict(runs=runs, expected=float(mu), z=float(z), p=float(0.5*math.erfc(-z/math.sqrt(2))), n_used=n)


def quality_thresholds(stratum_value):
    """DEV thresholds of the S/N stratum of one curve: dict(chi2_per_point, runs_z, stratum, available)."""
    stats = load_stats()
    q = None if stats is None else stats.get('quality')
    if not q:
        return dict(chi2_per_point=None, runs_z=RUNS_Z_AT_P, stratum=None, available=False)
    if 'edges' not in q:                                   # version 1 file: one global chi2/pt threshold
        return dict(chi2_per_point=q.get('chi2_per_point_threshold'), runs_z=RUNS_Z_AT_P, stratum=None, available=True)
    if stratum_value is None or not math.isfinite(stratum_value):
        fb = q['fallback']
        return dict(chi2_per_point=fb['chi2_per_point_threshold'], runs_z=min(fb['runs_z_threshold'], RUNS_Z_AT_P),
                    stratum=None, available=True)
    i = int(np.searchsorted(np.asarray(q['edges'], float), stratum_value, side='right'))
    return dict(chi2_per_point=q['chi2_per_point_threshold'][i], runs_z=min(q['runs_z_threshold'][i], RUNS_Z_AT_P),
                stratum=i, available=True)


def quality(chi2, n_points, residual, prediction=None, sigma=None, stratum_value=None):
    """quality flag (see module docstring). residual in q order; prediction and sigma select the runs-test
    points (all points when they are not given); stratum_value = input feature log10_median_relative_sigma."""
    thr = quality_thresholds(stratum_value)
    per_point = chi2/max(n_points, 1)
    select = runs_selection(prediction, sigma) if prediction is not None and sigma is not None else None
    runs = runs_test(residual, select)
    high = bool(thr['chi2_per_point'] is not None and per_point > thr['chi2_per_point'])
    runs_fail = bool(runs['p'] is not None and runs['p'] < RUNS_P_THRESHOLD and runs['z'] < thr['runs_z'])
    return dict(flag=bool(high or runs_fail), chi2_per_point=per_point, chi2_per_point_threshold=thr['chi2_per_point'],
                chi2_high=high, runs_p=runs['p'], runs_z=runs['z'], runs_z_threshold=thr['runs_z'], runs=runs['runs'],
                runs_expected=runs['expected'], runs_points_used=runs['n_used'], runs_fail=runs_fail,
                snr_stratum=thr['stratum'], thresholds_available=thr['available'])


def weakly_constrained(populations, at_bound_names, has_uncertainties=True, multimodal=()):
    """(names, detail): parameters 'popN.key' not pinned down by the curve and the reasons (module docstring).

    has_uncertainties False (preview: no covariance) -> ([], {}): no statement is made.
    multimodal: names flagged by the alternative-solution check (controller.CompositionRun.alternatives)."""
    if not has_uncertainties:
        return [], {}
    out, detail = [], {}
    bound_keys = {name.split('@')[0] for name in at_bound_names}
    multimodal = set(multimodal)
    for pop in populations:
        for key in POPULATION_KEYS:
            value = pop.get(key)
            if value is None or key in pop.get('fixed', []):
                continue
            name = f"{pop['pop']}.{key}"
            reasons = []
            if pop.get('inactive'):
                reasons.append('inactive')
            else:
                err = pop.get(f'{key}_err')
                if err is None or not np.isfinite(err) or err <= 0:
                    reasons.append('no_error')
                elif value != 0 and abs(err/value) > WEAK_RELATIVE:
                    reasons.append('relative_error')
                if name in bound_keys:
                    reasons.append('at_bound')
                if name in multimodal:
                    reasons.append('multimodal')
            if reasons:
                out.append(name)
                detail[name] = reasons
    return out, detail


def reportable_only(names):
    """Subset of 'popN.key' names that are reportable parameters (R_nm, D_nm; PLAN_zh.md §1)."""
    return [n for n in names if n.split('.', 1)[-1] in REPORTABLE]


def raised(flags):
    """Names of the PLAN flag set that are set. weakly_constrained counts only through its reportable
    subset (R_nm, D_nm); sigma_assumed is informational and not counted."""
    out = []
    for name in FLAG_SET:
        value = flags.get('weakly_constrained_reportable') if name == 'weakly_constrained' else flags.get(name)
        if isinstance(value, list) and value or value is True:
            out.append(name)
    return out
