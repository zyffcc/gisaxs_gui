'Fixed-band high-q and discrete peak-support checks for original PPC draws.'
from collections import Counter
from copy import deepcopy
from numbers import Integral

import numpy as np

from .scenario_candidates import observe_candidate_frame, _peak_kind
from .scenario_fixed_af_observation import observe_candidate_frame as fixed_observer
from .v2_observation import CountFrame
from .v2_ppc import evaluate

SCHEMA = 'original_draw_fixed_AF_peak_support_PPC_1'
CLASSES = ('bilateral', 'one_side', 'conflicting_bilateral', 'no_resolved_peak')
MATCH_LOWER_GATE = .5
CLASS_MASS_LOWER_GATE = .025
SOURCE_FILES = ('sputtergisaxs/scenario_peak_support_ppc.py',
    'sputtergisaxs/scenario_candidates.py', 'sputtergisaxs/scenario_fixed_af_observation.py',
    'sputtergisaxs/v2_observation.py', 'sputtergisaxs/v2_ppc.py')


def _band(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value not in range(4):
        raise ValueError('Original AF band must be None or a physical band index0--3')
    return int(value)


def comparison_observed(frame, observed=None):
    'Freeze a measured comparison band without replacing network features.'
    if not isinstance(frame, CountFrame):
        raise ValueError('An actual original CountFrame is required')
    if observed is None:
        descriptor = observe_candidate_frame(frame, estimate_stat_se=frame.photon_counts_known)
    elif isinstance(observed, dict) and isinstance(observed.get('values'), dict):
        descriptor = deepcopy(observed)
    else:
        raise ValueError('Actual observed feature descriptor required')
    original_band = _band(descriptor.get('AF_band'))
    chosen = original_band
    source = descriptor.get('comparison_band_source', 'provided_original_observed_band')
    if chosen is None:
        shared = np.asarray(descriptor.get('shared_features', []), float)
        if shared.shape == (40,):
            known = shared[29:33]
            if not np.isin(known, (0., 1.)).all():
                raise ValueError('Original shared high-q flags must be binary')
            available = [b for b in range(4) if known[b] == 1 and np.isfinite(shared[9+b])]
            chosen = available[0] if available else None
        source = 'first_original_known_highq_band' if chosen is not None else 'no_supported_comparison_band'
    classification_already_computed = any(key in descriptor for key in (
        'peak_measurability_known', 'no_peak_with_sufficient_support', 'peak_classification_status'))
    if chosen is not None and (original_band is None or not classification_already_computed):


        descriptor = fixed_observer(frame, AF_band=chosen, estimate_stat_se=frame.photon_counts_known)
    if chosen is None:


        descriptor.update(AF_band=None, support='unresolved', bands_support=[],
            peak_measurability_known=False, no_peak_with_sufficient_support=False,
            peak_classification_status='unknown_no_fixed_comparison_band')
    elif descriptor.get('support') != 'bilateral':


        keys = ('FWHM', 'valley_ratio') if descriptor.get('support') == 'one_side' else ('q_star', 'FWHM', 'valley_ratio')
        for key in keys:
            descriptor['values'][key] = None
            descriptor.setdefault('feature_missing_reasons', {})[key] = 'standard_PPC_requires_bilateral_peak_support'
    descriptor.update(AF_band=chosen, comparison_band_source=source,
        peak_support_comparison_scope='only the frozen original observed AF band',
        comparison_original_AF_band=original_band,
        comparison_band_changed=chosen != original_band,
        comparison_q_feature_definition='measured q for bilateral/one_side; FWHM/valley bilateral only; consumer must match',
        q_availability_expanded_from_old_bilateral_only_PPC=True,
        network_inputs_modified=False, structural_release=False)
    descriptor['comparison_peak_kind'] = _peak_kind(descriptor)
    return descriptor


def evaluate_peak_support(observed, originalreplicas, expected_draws):
    'Conservative empirical class-mass bounds from every original replica.'
    replicas = list(originalreplicas)
    if (isinstance(expected_draws, (bool, np.bool_)) or not isinstance(expected_draws, Integral)
            or expected_draws < 2 or len(replicas) != expected_draws):
        raise ValueError('Every declared original draw must remain in the support check')
    expected_draws = int(expected_draws)
    if not isinstance(observed, dict):raise ValueError('Original observed descriptor required')
    observed_band = _band(observed.get('AF_band'))
    kind = _peak_kind(observed) if observed_band is not None else 'unknown'
    counts = Counter(); failure_records = []
    for position, replica in enumerate(replicas):
        if not isinstance(replica, dict) or not isinstance(replica.get('forward_failed'), (bool, np.bool_)):
            raise ValueError('Each original replica needs an explicit forward_failed state')
        if _band(replica.get('AF_band')) != observed_band:
            raise ValueError('Every original replica must use the fixed observed AF band')
        failed = bool(replica['forward_failed'])
        label = 'unknown' if failed or observed_band is None else _peak_kind(replica)
        if label not in CLASSES + ('unknown',):raise ValueError('Unknown original peak-support classification')
        counts[label] += 1
        if failed:
            failure_records.append(dict(original_position=position, original_draw_index=replica.get('index'),
                reason=replica.get('failure_reason') or replica.get('reason') or 'forward_failure'))
    unknown = counts['unknown']; failures = len(failure_records)
    bounds = {label:[counts[label]/expected_draws, (counts[label]+unknown)/expected_draws] for label in CLASSES}
    applicable = kind in CLASSES
    matching = counts[kind] if applicable else 0
    lower = matching/expected_draws if applicable else None
    upper = (matching+unknown)/expected_draws if applicable else None
    majority = bool(applicable and lower >= MATCH_LOWER_GATE)
    class_mass = bool(applicable and lower >= CLASS_MASS_LOWER_GATE)
    passed = bool(applicable and not failures and majority and class_mass)
    return dict(schema=SCHEMA, original_draws=expected_draws, observed_AF_band=observed_band,
        observed_peak_kind=kind, applicable=applicable,
        forward_failed_draws=failures, forward_failure_records=failure_records,
        successfully_computed_original_draws=expected_draws-failures,
        known_classified_draws=expected_draws-unknown, unknown_classified_draws=unknown,
        unknown_nonfailed_draws=unknown-failures,
        class_counts={label:int(counts[label]) for label in CLASSES + ('unknown',)},
        class_probability_all_completion_bounds=bounds,
        matching_class_draws=matching if applicable else None,
        matching_probability_all_completion_bounds=[lower,upper] if applicable else None,
        matching_probability_lower_gate=MATCH_LOWER_GATE,
        observed_class_mass_lower_gate=CLASS_MASS_LOWER_GATE,
        binary_median_match_all_completions=majority,
        observed_class_empirical_predictive_support_lower_pass=class_mass,
        threshold025_is_implied_by_majority_threshold=True,
        passed=passed,
        reason='unknown_observed_peak_support' if not applicable else 'original_forward_failure' if failures
            else 'insufficient_guaranteed_matching_mass' if not majority
            else 'insufficient_guaranteed_observed_class_mass' if not class_mass else 'passed_peak_support_check',
        bounds_definition='empirical fraction identification bounds; each unknown may be any class; no drop/no replacement',
        median_definition='binary same-class indicator threshold>=.5; no ordering of physical classes',
        categorical95_interval_calibrated=False,
        unknown_observation_is_not_peak_absence=True,
        calibrated=False, OOD_qualified=False, structural_release=False)


def enhance_ppc(observed, replicas, expected_draws):
    'Keep the original numerical PPC fields and add discrete support checks.'
    original = list(replicas)
    output = evaluate(observed, original, expected_draws=expected_draws)
    support = evaluate_peak_support(observed, original, expected_draws)
    output['peak_support_check'] = support


    output['all_measurable_and_peak_support_checks_pass'] = bool(
        output['all_measurable_feature_checks_pass'] and
        (not support['applicable'] or support['passed']))
    output['enhanced_all_measurable_checks_pass'] = output['all_measurable_and_peak_support_checks_pass']
    output['enhanced_D_feature_check_pass'] = bool(output['D_feature_check_pass'] and support['passed'])
    output['enhanced_diameter_feature_check_pass'] = bool(output['diameter_feature_check_pass'] and support['passed'])
    output['peak_support_check_complete'] = bool(support['applicable'])
    output['enhancement_policy'] = dict(numerical_gate_operator='unchanged v2_ppc.evaluate',
        numerical_feature_values_changed=False, original_draws_replaced=False,
        unknown_support='incomplete comparison; never treated as confirmed absence',
        acceptance_performed=False, structural_release=False)
    return output
