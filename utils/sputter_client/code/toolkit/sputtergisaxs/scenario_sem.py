'Optional SEM comparisons of original four-scenario conditional draws.'
from collections.abc import Mapping
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr


SCENARIOS = ('S1', 'S2', 'S3', 'S4')
ALL_METAL_DEFINITIONS = (
    'number_mean_isolated_metal_max_lateral_diameter',
    'number_mean_isolated_metal_equivalent_circle_diameter',
)
MAIN_POPULATION_DEFINITIONS = (
    'number_mean_isolated_main_metal_population_max_lateral_diameter',
    'number_mean_isolated_main_metal_population_equivalent_circle_diameter',
)
DIAMETER_DEFINITIONS = ALL_METAL_DEFINITIONS + MAIN_POPULATION_DEFINITIONS
RANKING_RULE = 'ascending abs(log(conditional_population_mean_median / SEM_mean)); ties share rank'


def _plain(value):
    'Preserve invalid supplied numbers as text, rather than inventing values.'
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _number(value, *, zero=False, unit_interval=False):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if (not math.isfinite(result) or (result < 0 if zero else result <= 0)
            or unit_interval and result > 1):
        return None
    return result


def _count(value):
    return int(value) if (isinstance(value, (int, np.integer))
                         and not isinstance(value, (bool, np.bool_)) and value >= 0) else None


def _population_mean(physical, scene):
    'Main-population mean of the original homothetic [-8,8] size law.'
    if not isinstance(physical, Mapping) or physical.get('scenario') != scene:
        raise ValueError('original physical scenario missing or changed')
    particle = physical.get('particle')
    if not isinstance(particle, Mapping):
        raise ValueError('original joint particle parameters missing')
    a = _number(particle.get('base_radius_nm'))
    h = _number(particle.get('height_nm'))
    sigma = _number(particle.get('sigma_log'))
    median = _number(physical.get('diameter_nm'))
    shape = particle.get('shape')
    from .scenario_mainline_chart import SCHEMA as MAINLINE_SCHEMA
    upper=.65 if physical.get('schema')==MAINLINE_SCHEMA and 'd' in physical.get('modifiers',[]) else .45
    if (a is None or h is None or sigma is None or median is None
            or not .03 <= sigma <= upper or shape not in ('cap', 'hemiellipsoid')
            or not .1-1e-12 <= h/a <= 2+1e-12):
        raise ValueError('original positive size/shape/spread parameters invalid')
    footprint = 2*((a*a+h*h)/(2*h) if shape == 'cap' and h > a else a)
    if not math.isclose(median, footprint, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('original primitive diameter disagrees with joint geometry')
    factor = math.exp(.5*sigma*sigma)*(ndtr(8-sigma)-ndtr(-8-sigma))/(ndtr(8)-ndtr(-8))
    result = median*float(factor)
    if not math.isfinite(result) or result <= 0:
        raise ValueError('original population-mean diameter nonfinite')
    return result


def _draw_values(part, scene, requested):
    declared = _count(part.get('original_draws'))
    expected = requested if requested is not None else declared
    records = part.get('draws')
    records = records if isinstance(records, (list, tuple)) else []
    errors = []
    if declared is None or expected is None or expected <= 0:
        errors.append('missing_positive_original_draw_denominator')
    if declared != expected:
        errors.append('scenario_denominator_differs_from_original_request')
    if expected is not None and len(records) != expected:
        errors.append('original_record_count_differs')
    values = {}; seen = set(); failures = []; second_population = []
    for record in records:
        index = _count(record.get('index')) if isinstance(record, Mapping) else None
        reason = None
        if index is None or expected is None or index >= expected:
            reason = 'invalid_original_draw_index'
        elif index in seen:
            reason = 'duplicate_original_draw_index'
        else:
            seen.add(index)
            if record.get('decode_failed') is not False:
                reason = 'original_draw_decode_failed'
            else:
                try:
                    physical = record.get('physical')
                    if scene == 'S4':
                        if not isinstance(physical, Mapping) or physical.get('scenario') != scene:
                            raise ValueError('original physical scenario missing or changed')


                        values[index] = None
                    else:
                        values[index] = _population_mean(physical, scene)
                        if 'small_clusters' in physical:
                            second_population.append(index)
                except (ValueError, TypeError, OverflowError) as error:
                    reason = str(error)
        if reason is not None:
            failures.append(dict(index=index, reason=reason))
    complete = (not errors and not failures and expected is not None
                and set(values) == set(range(expected)))
    meta = dict(original_draws=expected, declared_scenario_draws=declared,
        records_received=len(records), resolved_original_draws=len(values),
        failed_original_draws=max(0, expected-len(values)) if expected is not None else None,
        original_identity_errors=errors, failures=failures,
        complete_original_draws=complete, intervals_from_successful_subset=False,
        second_population_original_draws=len(second_population),
        second_population_original_draw_indices=second_population,
        diameter_proxy_population_scope='main_population_only',
        second_population_number_weights_used=False)
    return (np.array([values[i] for i in range(expected)], float)
            if complete and scene != 'S4' else None), meta


def _interval(values):
    if values is None:
        return dict(median=None, interval68=None, interval95=None, posterior_sd=None)
    q = np.quantile(values, [.025, .16, .5, .84, .975])
    return dict(median=float(q[2]), interval68=q[[1, 3]].tolist(),
        interval95=q[[0, 4]].tolist(), posterior_sd=float(np.std(values, ddof=1)) if len(values) > 1 else None)


def compare_sem(sampled, sem):
    'Retain all S, their conditional proxies, and fixed diagnostic ranking.'
    if not isinstance(sampled, Mapping) or not isinstance(sampled.get('scenarios'), Mapping):
        raise TypeError('Original ScenarioRuntime sampled scenarios required')
    provided = sem is not None
    notes = []
    if sem is None:
        sem = {}
    elif not isinstance(sem, Mapping):
        notes.append('invalid_SEM_mapping_kept_unknown')
        sem = {'unparsed_supplied_value': _plain(sem)}
    mean_raw = sem.get('mean_diameter_nm'); se_raw = sem.get('mean_diameter_se_nm')
    mean = _number(mean_raw); se = _number(se_raw, zero=True)
    definition = sem.get('diameter_definition', sem.get('definition'))
    matched = isinstance(definition, str) and definition in DIAMETER_DEFINITIONS
    if mean_raw is not None and mean is None:
        notes.append('invalid_SEM_mean_diameter_kept_unknown')
    if se_raw is not None and se is None:
        notes.append('invalid_SEM_standard_error_kept_unknown; no_error_imputed')
    nn_raw = sem.get('nearest_neighbor_nm', sem.get('NN_nm'))
    coverage_raw = sem.get('coverage_fraction', sem.get('coverage'))
    observation = dict(mean_diameter_nm=mean, mean_diameter_se_nm=se,
        diameter_definition=_plain(definition), matched_diameter_definition=matched,
        definition_match_requires_scenario_population_scope=True,
        measurement_error_imputed=False, supplied_statistics=_plain(sem),
        nearest_neighbor=dict(supplied_value=_plain(nn_raw), value_nm=_number(nn_raw),
            comparable=False, reason='D_corr is a scattering scale; radial correlation step/hex-equivalent density length is not a certified 2D SEM nearest-neighbour mean'),
        coverage=dict(supplied_value=_plain(coverage_raw),
            value_fraction=_number(coverage_raw, zero=True, unit_interval=True),
            comparable=False, reason='No certified visible SEM union coverage: S1/S2 lack centre geometry; S3 footprint occupation and S4 planar phase fraction have different definitions'))
    requested = _count(sampled.get('original_draws_per_applicable_scenario'))
    output = {}; candidates = []
    for scene in SCENARIOS:
        part = sampled['scenarios'].get(scene)
        present = isinstance(part, Mapping)
        applicable = part.get('applicable') if present else None
        one = dict(applicable=applicable, status='not_evaluated',
            mean_diameter_proxy_nm=_interval(None), comparison=None,
            rank=None, ranking_score=None, rankable=False,
            calibrated=False, structural_release=False,
            nearest_neighbor_comparable=False, coverage_comparable=False)
        output[scene] = one
        if not present:
            one.update(status='missing_original_scenario', missing_reason='original scenario entry absent')
            continue
        if applicable is not True:
            one.update(status='not_applicable' if applicable is False else 'invalid_applicability',
                missing_reason=part.get('not_applicable_reason') or 'original applicability unavailable',
                original_draws=_count(part.get('original_draws')))
            continue
        if scene == 'S4':
            _, meta = _draw_values(part, scene, requested)
            one.update(meta)
            one.update(status='island_diameter_not_applicable',
                original_draws=_count(part.get('original_draws')),
                missing_reason='effective two-phase S4 has no metal island/primitive diameter')
            continue
        values, meta = _draw_values(part, scene, requested)
        one.update(meta)
        one['mean_diameter_proxy_nm'] = _interval(values)
        one['proxy_definition'] = ('conditional effective-column number-mean maximum footprint diameter; not certified SEM metal particles'
            if scene == 'S3' else 'conditional main-metal-population number-mean maximum footprint diameter; excludes any second cluster population and host/template particles')
        one['assumptions'] = ['homothetic normal-coordinate-truncated lognormal marks [-8,8]',
            'number weighting, resolved individual primitives, circular projected footprints',
            'visibility, segmentation and projection overlap are not modeled']
        if values is None:
            one.update(status='incomplete_original_draws', missing_reason='No interval/rank from a retained subset of original draws')
            continue
        one['status'] = 'conditional_proxy_only'
        if mean is None:
            one['missing_reason'] = 'SEM mean diameter not supplied or invalid'
            continue
        median = one['mean_diameter_proxy_nm']['median']
        population_matched = (not meta['second_population_original_draws']
                              or matched and definition in MAIN_POPULATION_DEFINITIONS)
        one['SEM_population_scope_matched'] = bool(population_matched)
        difference = values-mean
        relative = values/mean-1
        comparison = dict(difference_nm=_interval(difference),
            relative_difference=_interval(relative),
            SEM_mean_in_conditional_interval95=bool(np.quantile(values, .025) <= mean <= np.quantile(values, .975)),
            standardized_difference=None, standardized_difference_definition=None,
            matched_definition=bool(matched and population_matched and scene != 'S3'), comparison_is_conditional=True,
            calibrated=False, structural_release=False)
        if se is not None and len(values) > 1:
            denominator = math.hypot(float(np.std(values, ddof=1)), se)
            if denominator > 0:
                comparison.update(standardized_difference=float((median-mean)/denominator),
                    standardized_difference_definition='(posterior median minus SEM mean)/sqrt(posterior population-mean variance + supplied SEM standard-error squared); uncalibrated diagnostic')
        one['comparison'] = comparison
        if scene == 'S3':
            one['missing_reason'] = 'effective column footprint is not a certified SEM metal particle; excluded from ranking'
        elif not population_matched:
            one['missing_reason'] = ('Original posterior includes a second metal-cluster population; '
                'this proxy only represents the main population, not the all-metal SEM mean. '
                'Ranking requires an explicitly segmented main-population SEM definition.')
        elif not matched:
            one['missing_reason'] = 'SEM diameter definition/object/weighting unmatched; proxy shown, excluded from ranking'
        else:
            one.update(status='definition_matched_conditional_diagnostic', rankable=True,
                ranking_score=float(abs(math.log(median/mean))), missing_reason=None)
            candidates.append(scene)
    order = sorted(candidates, key=lambda s: (output[s]['ranking_score'], s))
    last_score = None; last_rank = None
    for position, scene in enumerate(order, 1):
        score = output[scene]['ranking_score']
        rank = last_rank if last_score is not None and score == last_score else position
        output[scene]['rank'] = rank; last_score = score; last_rank = rank
    return dict(schema='four_scenario_SEM_comparison_1',
        status='optional_SEM_conditional_diagnostic' if provided else 'not_provided',
        observations=observation, scenarios=output, ranked_scenarios=order,
        ranking_rule=RANKING_RULE, ranking_definition_matched_only=True,
        ranking_is_model_evidence=False, scenario_probabilities_unchanged=True,
        SEM_used_for_posterior=False, original_draws_changed=False,
        nuisance_resampled=False, scenarios_pruned=False,
        source_receipt=_plain(sampled.get('receipt')), original_seed=sampled.get('original_seed'),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        notes=notes, calibrated=False, structural_release=False)
