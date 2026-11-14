'Internal public-observation composition with one complete eight-side fit.'
from __future__ import annotations

import math

import numpy as np

from .prediction import summarize_profiles
from .resolvable import PEAK_WINDOW, coverage, visible_for_regression
from .stage5_measurement import Q, counts_profiles
from .v2_observation import COUNT_CONTRACT, CountFrame
from .v2_ppc import GATES, STAT_SOURCES, poisson_fit_se


ORIGIN_SHA256 = {
    'v2_ppc.py': '09dd7030406fb634b1aeae76338b4d8f67bd3d7ff7f082c701fc1bc533b1d3a3',
    'stage5_measurement.py': '02bfae5179fd2678bd3e5a36c13fe50822fa082e052b7a2163d847954b1c2012',
    'v2_observation.py': '546638aa5047107d649d1619376d3279ad502d9086752162c7dea7d02d16d0a5',
    'prediction.py': '754057ad3bf8ef0312d1b88a94cb8acb538a1d18b9594274bee290c87645f489',
}


def empirical_summary(public):
    'Return unchanged observed fields and all-eight-side plotting fits.'
    if not isinstance(public, CountFrame):
        raise ValueError('Explicit pixel-frame observation required')
    Y, N = np.asarray(public.Y, float), np.asarray(public.P, float)
    if (not np.isfinite(Y).all() or not np.isfinite(N).all() or
            np.any(Y < 0) or np.any(N < 0) or np.any((N == 0) & (Y != 0))):
        raise ValueError('Nonnegative supported photon counts required')
    hy, hn, vy, vn, picture, scale = counts_profiles(Y, N)
    fits, bands, selected, ridge = summarize_profiles(hy, hn, vy, vn, PEAK_WINDOW)


    cov = coverage(hn)
    feature = np.zeros(20, np.float32); known = np.zeros(20, np.float32)
    for b, band in enumerate(bands):
        if band['q_star'] is not None:
            feature[2*b] = np.log(band['q_star']); feature[2*b+1] = np.log(band['FWHM']/band['q_star']); known[2*b:2*b+2] = 1
    y = ridge.get('alpha_f_observed_ridge')
    if y is not None and ridge.get('ridge_status') == 'empirical_local_maximum':
        feature[8] = y; known[8] = 1
    for b in range(4):
        r = np.divide(hy[2*b:2*b+2].sum(axis=0), hn[2*b:2*b+2].sum(axis=0),
                      out=np.zeros(len(Q)), where=hn[2*b:2*b+2].sum(axis=0) > 0)
        a = (Q >= .9) & (Q < 1.2) & (r > 0); z = (Q >= 1.4) & (Q < 1.7) & (r > 0)
        if a.sum() >= 3 and z.sum() >= 3:
            feature[9+b] = np.log(np.median(r[z])/np.median(r[a])); known[9+b] = 1
    visible = visible_for_regression(selected, cov)
    desc = dict(features=np.r_[feature, known].astype(np.float32), visible_peak=visible,
                coverage=cov, q_star=selected['q_star'] if selected else None,
                width=selected['FWHM'] if selected else None, Yoneda=y,
                support=selected['support'] if selected else 'unresolved')


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
        if bool(public.photon_counts_known):
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
    observed = dict(values=values, stat_se=se, stat_se_source=source,
                    feature_missing_reasons=reasons, AF_band=AF_band,
                    support=band['support'] if band else 'unresolved',
                    bands_support=[b['support'] for b in bands],
                    shared_features=desc['features'].tolist(), forward_failed=False,
                    structural_release=False)
    observed.update(count_support_contract=COUNT_CONTRACT,
                    photon_count_statistics_known=bool(public.photon_counts_known))
    return dict(observed=observed, hy=hy, hn=hn, vy=vy, vn=vn,
                picture=picture, scale=scale, fits=fits, bands=bands,
                selected=selected, ridge=ridge, coverage=cov, descriptor=desc)
