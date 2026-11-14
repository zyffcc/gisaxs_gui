'Evaluate an already fixed AF band for finite-library/medoid previews.'
import math

import numpy as np

from .prediction import summarize_profiles
from .resolvable import PEAK_WINDOW
from .stage5_measurement import Q, counts_profiles
from .v2_observation import COUNT_CONTRACT, CountFrame
from .v2_ppc import GATES, STAT_SOURCES, poisson_fit_se

SCHEMA = 'original_fixed_AF_candidate_observation_1'
SOURCE_FILES = (
    'sputtergisaxs/scenario_fixed_af_observation.py',
    'sputtergisaxs/scenario_candidates.py',
    'sputtergisaxs/stage5_measurement.py',
    'sputtergisaxs/prediction.py',
    'sputtergisaxs/features.py',
    'sputtergisaxs/resolvable.py',
    'sputtergisaxs/v2_observation.py',
    'sputtergisaxs/v2_ppc.py',
)


def _shared_features(hy, hn, band, ridge, index):
    'Original describe arithmetic with uncomputed peak slots explicit.'
    features = np.zeros(20, np.float32)
    known = np.zeros(20, np.float32)
    computed = np.ones(40, dtype=bool)
    for other in range(4):
        if other != index:
            computed[2*other:2*other+2] = False
            computed[20+2*other:20+2*other+2] = False
    if band['q_star'] is not None:
        features[2*index] = np.log(band['q_star'])
        features[2*index+1] = np.log(band['FWHM']/band['q_star'])
        known[2*index:2*index+2] = 1
    y = ridge.get('alpha_f_observed_ridge')
    if y is not None and ridge.get('ridge_status') == 'empirical_local_maximum':
        features[8] = y
        known[8] = 1
    for other in range(4):
        denominator = hn[2*other:2*other+2].sum(axis=0)
        rate = np.divide(hy[2*other:2*other+2].sum(axis=0), denominator,
                         out=np.zeros(len(Q)), where=denominator > 0)
        low = (Q >= .9) & (Q < 1.2) & (rate > 0)
        high = (Q >= 1.4) & (Q < 1.7) & (rate > 0)
        if low.sum() >= 3 and high.sum() >= 3:
            features[9+other] = np.log(np.median(rate[high])/np.median(rate[low]))
            known[9+other] = 1
    return np.r_[features, known].astype(np.float32), computed


def observe_candidate_frame(frame, *, AF_band=None, estimate_stat_se=True):
    'Original candidate-observer fields for one caller-fixed comparison.'
    if AF_band is None:
        from .scenario_candidates import observe_candidate_frame as original
        return original(frame, AF_band=None, estimate_stat_se=estimate_stat_se)
    if not isinstance(frame, CountFrame):
        raise ValueError('Explicit pixel-frame observation required')
    if (isinstance(AF_band, (bool, np.bool_))
            or not isinstance(AF_band, (int, np.integer))
            or AF_band not in range(4)):
        raise ValueError('AF_band must be an original observed band 0..3')
    b = int(AF_band)
    hy, hn, vy, vn, _, _ = counts_profiles(frame.Y, frame.P)


    pair_y, pair_n = hy[2*b:2*b+2], hn[2*b:2*b+2]
    fits, local_bands, _, ridge = summarize_profiles(pair_y, pair_n, vy, vn, PEAK_WINDOW)
    band = dict(local_bands[0], AF_band=b)


    from .features import AF_BANDS
    band['af_range_deg'] = list(AF_BANDS[b])
    shared, computed = _shared_features(hy, hn, band, ridge, b)

    values = {key: None for key in GATES}
    se = dict(values)
    source = dict(values)
    reasons = {key: 'unresolved_or_missing_support' for key in GATES}
    if band['support'] == 'bilateral':
        values.update(q_star=band['q_star'], FWHM=band['FWHM'])
        reasons.update(q_star=None, FWHM=None)
        if estimate_stat_se and frame.photon_counts_known:
            errors = [poisson_fit_se(Q, pair_y[side], pair_n[side], fits[side])
                      for side in range(2)]
            if all(error is not None for error in errors):
                for key in ('q_star', 'FWHM'):
                    se[key] = math.hypot(errors[0][key], errors[1][key])/2
                    source[key] = STAT_SOURCES[0]
        denominator = pair_n.sum(axis=0)
        rate = np.divide(pair_y.sum(axis=0), denominator,
                         out=np.zeros(len(Q)), where=denominator > 0)
        below = ((Q >= max(.05, .35*band['q_star']))
                 & (Q <= .75*band['q_star']) & (denominator > 0))
        peak = int(np.argmin(abs(Q-band['q_star'])))
        local = np.flatnonzero(below & (np.arange(len(Q)) > 0)
                              & (np.arange(len(Q)) < len(Q)-1))
        local = [i for i in local if rate[i] <= rate[i-1] and rate[i] < rate[i+1]
                 and denominator[i-1] > 0 and denominator[i+1] > 0]
        if local and rate[peak] > 0:
            i = min(local, key=lambda i: rate[i])
            error = np.sqrt(rate[i]/denominator[i] + rate[peak]/denominator[peak])
            if rate[peak]-rate[i] > 2*error:
                values['valley_ratio'] = float(rate[i]/rate[peak])
                reasons['valley_ratio'] = None
    index = 9+b
    if shared[20+index] == 1:
        value = float(np.exp(shared[index]))
        if np.isfinite(value) and value > 0:
            values['highq_ratio'] = value
            reasons['highq_ratio'] = None
    if ridge.get('ridge_status') == 'empirical_local_maximum':
        values['Yoneda'] = ridge['alpha_f_observed_ridge']
        reasons['Yoneda'] = None

    def usable(fit):
        return (fit.get('status') in ('resolved_empirical_feature', 'unresolved_or_boundary_feature')
                and fit.get('optimizer_success') is True and fit.get('gap_near_peak') is not True)

    resolved = sum(fit.get('status') == 'resolved_empirical_feature' for fit in fits)
    no_peak = bool(all(usable(fit) for fit in fits) and resolved == 0)


    if band['support'] == 'one_side':
        values['q_star'] = band['q_star']
        values['FWHM'] = None
        reasons['q_star'] = None
        reasons['FWHM'] = '单侧峰；不合成双侧峰宽'
    bands_support = [None]*4
    bands_support[b] = band['support']
    evaluated = [False]*4
    evaluated[b] = True
    return dict(values=values, stat_se=se, stat_se_source=source,
                feature_missing_reasons=reasons, AF_band=b,
                support=band['support'], bands_support=bands_support,
                shared_features=shared.tolist(), forward_failed=False,
                structural_release=False, count_support_contract=COUNT_CONTRACT,
                photon_count_statistics_known=bool(frame.photon_counts_known),
                peak_measurability_known=bool(resolved or no_peak),
                no_peak_with_sufficient_support=no_peak,
                comparison_band_source='caller_fixed_AF_band',
                shared_features_computed_mask=computed.tolist(), bands_evaluated=evaluated,
                execution_scope=dict(schema=SCHEMA, fitted_sides=2, repeated_peak_fits=False,
                    numeric_equivalence_scope='selected AF five features/SE/support/known peak classification',
                    other_band_peak_support='not_evaluated; None and unknown feature flags, not evidence of absence',
                    complete_network_descriptor=False, physical_model_changed=False))


observe_candidate_frame_fixed_af = observe_candidate_frame
