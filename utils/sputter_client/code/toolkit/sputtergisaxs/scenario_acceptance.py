'Pure C5 aggregation and time evidence from actually quantity-qualified D.'
import copy
import numpy as np


SCHEMA = 'four_scenario_C5_acceptance_1'
CENTRE_SCENARIOS = ('S1', 'S2')
POLICIES = ('none', 'host')
FEATURES = ('q_star', 'FWHM', 'Yoneda')


def _finite(value):
    return (isinstance(value, (int, float, np.number))
            and not isinstance(value, (bool, np.bool_)) and bool(np.isfinite(value)))


def _result(status, reason=None, **values):
    return dict(status=status, passed=True if status == 'passed' else
                False if status == 'failed' else None,
                reason=reason, **values)


def _and(items):
    statuses = [item['status'] for item in items]
    if 'failed' in statuses:
        return 'failed'
    if not statuses or any(s not in ('passed', 'not_applicable') for s in statuses):
        return 'pending'
    return 'passed'


def _interval(value):
    try:
        values = list(value)
        if len(values) == 2 and all(_finite(v) for v in values) and values[0] <= values[1]:
            return [float(v) for v in values]
    except (TypeError, ValueError):
        pass
    return None


def _time(frame):
    raw = frame.get('t_header_s', frame.get('time_s'))
    if _finite(raw):
        return float(raw)
    if isinstance(raw, (list, tuple, np.ndarray)) and len(raw) == 1 and _finite(raw[0]):
        return float(raw[0])
    return None


def _observed(frame):
    observed = frame.get('observed') or {}
    values = observed.get('values', observed)
    q = values.get('q_star', values.get('q_star_nm_inv'))
    D = values.get('D_corr_nm')
    if not (_finite(D) and D > 0):
        D = float(2*np.pi/q) if _finite(q) and q > 0 else None
    intervals = observed.get('interval95', {})
    interval = _interval(intervals.get('D_corr_nm')) if isinstance(intervals, dict) else None
    scope = 'original reported Dcorr interval' if interval else None
    qse = observed.get('stat_se', {}).get('q_star')
    if interval is None and _finite(q) and q > 0:
        qi = _interval(intervals.get('q_star')) if isinstance(intervals, dict) else None
        if qi and qi[0] > 0:
            interval = [float(2*np.pi/qi[1]), float(2*np.pi/qi[0])]
            scope = 'inverse of original reported q_star interval'
        elif _finite(qse) and qse >= 0 and q-1.96*qse > 0:
            interval = [float(2*np.pi/(q+1.96*qse)), float(2*np.pi/(q-1.96*qse))]
            scope = 'conditional normal approximation q_star +/-1.96 statistical SE; geometry/systematics excluded'
    return dict(q_star_nm_inv=float(q) if _finite(q) and q > 0 else None,
                D_corr_nm=float(D) if _finite(D) and D > 0 else None,
                q_star_stat_se=float(qse) if _finite(qse) and qse >= 0 else None,
                D_corr_interval95=interval, interval_scope=scope)


def _sign(value):
    return 1 if value > 0 else -1 if value < 0 else 0


def _slope(points, name):
    t = np.asarray([p['time_s'] for p in points], float)
    x = np.asarray([p[name] for p in points], float)
    if len(t) < 2 or np.ptp(t) <= 0:
        return None
    if np.ptp(x) == 0:
        return 0.
    centred = t-t.mean()
    return float(np.dot(centred, x-x.mean())/np.dot(centred, centred))


def _interval_direction(a, b):
    if a is None or b is None:
        return None
    return 1 if b[0] > a[1] else -1 if b[1] < a[0] else None


def qualified_D_time(frames, *, quantity_gate, expected_original_windows=223):
    'No qualification logic duplicated: the caller owns the original D gate.'
    frames = list(frames or []); reasons = []
    keys = [f.get('key') for f in frames]
    complete = bool(len(frames) == expected_original_windows and
                    all(isinstance(k, str) and k for k in keys) and len(set(keys)) == len(keys))
    if not complete:
        reasons.append('complete unique original %d-window evidence is missing' % expected_original_windows)
    if not callable(quantity_gate):
        reasons.append('original quantity-gate callback is missing')
    scans = {}; required = []
    names = sorted({f.get('scan') for f in frames if isinstance(f.get('scan'), str)})
    if len(names) != len({f.get('scan') for f in frames}):
        reasons.append('original scan identity is missing')
    for scan in names:
        original = [f for f in frames if f.get('scan') == scan]
        one_scan = dict(original_scan_windows=len(original), policies={})
        for policy in POLICIES:
            one_scan['policies'][policy] = {}
            for scene in CENTRE_SCENARIOS:
                original_rows = []; points = []; qualified_count = 0; applicable_count = 0
                for frame in original:
                    p = (frame.get('policies') or {}).get(policy, {})
                    credibility = p.get('credibility') or {}
                    part = (p.get('scenarios') or {}).get(scene, {})
                    physical_na = (part.get('physical_applicable') is False or
                                   (scene == 'S2' and frame.get('host_category') == 'Si'))
                    binding = dict(key=frame.get('key'), policy=policy, scenario=scene,
                        original_seed=p.get('original_seed', credibility.get('original_seed', frame.get('original_posterior_seed'))),
                        frame_identity=copy.deepcopy(p.get('frame_identity', credibility.get('frame_identity'))),
                        joint_sha256=copy.deepcopy(p.get('joint_sha256', credibility.get('joint_sha256'))))
                    identity_ready = bool(isinstance(binding['original_seed'], (int, np.integer))
                        and not isinstance(binding['original_seed'], (bool, np.bool_))
                        and binding['original_seed'] >= 0 and binding['frame_identity']
                        and binding['joint_sha256'])
                    item = None; gate_error = None
                    if not physical_na:
                        applicable_count += 1
                        if callable(quantity_gate):
                            try:
                                item = quantity_gate(frame, policy, scene)
                            except Exception as error:
                                gate_error = type(error).__name__+': '+str(error)
                    if not isinstance(item, dict):
                        item = {}
                    qualified = bool(not physical_na and item.get('data_qualified') is True)
                    qualified_count += int(qualified)
                    t = _time(frame); observation = _observed(frame)
                    D = item.get('median'); Di = _interval(item.get('interval95'))
                    ready = bool(qualified and identity_ready and t is not None and _finite(D) and D > 0
                                 and observation['D_corr_nm'] is not None)
                    missing = []
                    if not physical_na:
                        if gate_error: missing.append(gate_error)
                        if not qualified: missing.append(item.get('reason') or 'D is not quantity-qualified by original gates')
                        if qualified and not identity_ready: missing.append('original frame/seed/joint identity missing')
                        if qualified and t is None: missing.append('original scalar header time missing')
                        if qualified and not (_finite(D) and D > 0): missing.append('qualified D median missing')
                        if qualified and observation['D_corr_nm'] is None: missing.append('observed Dcorr missing')
                    row = dict(original_binding=binding, physically_applicable=not physical_na,
                        data_qualified=qualified, time_s=t, D_nm=float(D) if _finite(D) else None,
                        D_interval95=Di, observed=observation, used_for_direction=ready,
                        missing_reasons=missing)
                    original_rows.append(row)
                    if ready:
                        points.append(dict(key=frame['key'], time_s=t, D_nm=float(D),
                            D_interval95=Di, D_corr_nm=observation['D_corr_nm'],
                            D_corr_interval95=observation['D_corr_interval95']))
                points.sort(key=lambda r:(r['time_s'], r['key']))
                adjacent = []
                for a,b in zip(points, points[1:]):
                    d = b['D_nm']-a['D_nm']; corr = b['D_corr_nm']-a['D_corr_nm']
                    di = _interval_direction(a['D_interval95'], b['D_interval95'])
                    ci = _interval_direction(a['D_corr_interval95'], b['D_corr_interval95'])
                    adjacent.append(dict(first_key=a['key'], last_key=b['key'], delta_time_s=b['time_s']-a['time_s'],
                        delta_D_nm=d, delta_Dcorr_nm=corr,
                        point_directions_agree=_sign(d) == _sign(corr) if d and corr else None,
                        D_direction_from_marginal_interval_separation=di,
                        Dcorr_direction_from_marginal_interval_separation=ci,
                        interval_supported_directions_agree=di == ci if di is not None and ci is not None else None,
                        interval_separation_is_not_a_joint_difference_95_interval=True,
                        required_for_C5=False))
                ds = _slope(points, 'D_nm'); cs = _slope(points, 'D_corr_nm')
                if not applicable_count:
                    status, why = 'not_applicable', 'S2 centre-density D is not applicable on known Si'
                elif not qualified_count:
                    status, why = 'pending', 'no actually quantity-qualified centre-D window'
                elif len(points) != qualified_count:
                    status, why = 'pending', 'some original qualified windows lack time/feature/pool identity'
                elif ds is None or cs is None:
                    status, why = 'pending', 'fewer than two qualified original windows or no distinct time'
                elif ds == 0 or cs == 0:
                    status, why = 'pending', 'constant qualified D or observed Dcorr; growth direction is not measured'
                else:
                    status, why = ('passed', 'overall point-value slopes have the same sign') if _sign(ds) == _sign(cs) else ('failed', 'overall qualified D and observed Dcorr slopes have opposite signs')
                summary = _result(status, why, original_scan_windows=len(original),
                    physically_applicable_windows=applicable_count,
                    original_quantity_qualified_windows=qualified_count,
                    qualified_observation_pairs=len(points), qualified_adjacent_pairs=len(adjacent),
                    missing_qualified_windows=qualified_count-len(points),
                    D_slope_nm_per_s=ds, Dcorr_slope_nm_per_s=cs,
                    original_rows=original_rows, adjacent_changes=adjacent,
                    criterion='same sign of overall qualified D and Dcorr OLS slopes; no significance or per-step monotonicity threshold',
                    uncertainty_caveat='point-value direction only; broad intervals and shared geometry/time systematics may leave significance unresolved',
                    required_for_C5=qualified_count > 0)
                one_scan['policies'][policy][scene] = summary
                if qualified_count:
                    required.append(summary)
        scans[scan] = one_scan
    status = _and(required) if required else 'not_applicable'
    if not required and not reasons:
        why='no centre-D quantity was released; time comparison is conditional on released D, not an additional requirement for S4-only feature results'
    else:
        why='; '.join(reasons) if reasons else None
    if reasons and status != 'failed':
        status = 'pending'
    return _result(status, why,
        original_window_denominator=expected_original_windows, provided_original_windows=len(frames),
        complete_original_windows=complete, scans=scans,
        S3='excluded: effective-column correlation repeat is not metal centre-density spacing',
        S4='excluded: no island D', correlations_used_as_acceptance_threshold=False,
        adjacent_monotonicity_required=False, temporal_significance_required=False)


def s4_feature_calibration(rows, *, expected_families):
    'Recompute clean-mean coverage at the observed band, not an intrinsic TS score.'
    if (not isinstance(expected_families, (int, np.integer))
            or isinstance(expected_families, (bool, np.bool_)) or expected_families < 1):
        return _result('pending', 'original expected S4 family denominator missing')
    if not isinstance(rows, (list, tuple)) or len(rows) != expected_families or not all(isinstance(r, dict) for r in rows):
        return _result('pending', 'complete original S4 family/latent-feature records missing',
            original_S4_family_denominator=int(expected_families),
            provided_original_families=len(rows) if isinstance(rows, (list,tuple)) else 0)
    selected = [r for r in rows if r.get('eligible_visible') is not False]
    denominator = len(selected); unknown = sum(r.get('eligible_visible') is not True for r in selected)
    features = {}
    for name in FEATURES:
        hits = {68:0, 95:0}; failed = 0
        applicable = [r for r in selected
            if (r.get('quantities') or {}).get(name,{}).get('truth_applicable') is not False]
        feature_denominator = len(applicable)
        known_na = denominator-feature_denominator
        for row in applicable:
            draws = row.get('original_draws')
            item = (row.get('quantities') or {}).get(name, {})
            truth = item.get('truth'); intervals = {level:_interval(item.get('interval'+str(level))) for level in (68,95)}
            valid = bool(row.get('eligible_visible') is True
                and isinstance(draws, (int,np.integer)) and not isinstance(draws, (bool,np.bool_)) and draws >= 2
                and row.get('generation_failed') is not True
                and item.get('valid_complete_original_joint') is True
                and item.get('original_draws', draws) == draws
                and _finite(truth) and truth > 0 and all(intervals.values()))
            if not valid:
                failed += 1
                continue
            for level in (68,95):
                interval = intervals[level]
                hits[level] += int(interval[0] <= truth <= interval[1])
        cov = {level:hits[level]/feature_denominator if feature_denominator else None for level in (68,95)}
        if not denominator:
            status, why = 'pending', 'no original visible S4 truth rows'
        elif not feature_denominator:
            status, why = 'not_applicable', 'feature is known unmeasurable in all original visible clean truth rows'
        else:
            passed = all(abs(cov[level]-nominal) <= .05+1e-12 for level,nominal in ((68,.68),(95,.95)))
            status, why = ('passed', 'failure-inclusive original visible-row interval coverage passed') if passed else ('failed', 'original visible-row interval coverage outside nominal +/-0.05')
        features[name] = _result(status, why, original_visible_row_denominator=denominator,
            original_applicable_feature_denominator=feature_denominator,
            known_not_applicable_truth_rows=known_na,
            valid_complete_original_rows=feature_denominator-failed, failed_or_missing_original_rows=failed,
            unknown_visible_eligibility_rows=unknown,
            coverage68=cov[68], coverage95=cov[95], coverage68_hits=hits[68], coverage95_hits=hits[95],
            failed_rows_in_denominator=True)
    return _result(_and(list(features.values())), None,
        original_S4_family_denominator=int(expected_families), original_visible_row_denominator=denominator,
        unknown_visible_eligibility_rows=unknown, features=features,
        truth_target='latent_mean_response_features', noisy_replica_PPC_is_separate=True,
        visibility_subset='original_clean_mean_bilateral_support_at_observed_exit_angle_band',
        island_R2_required=False, unknown_fraction_threshold_added=False,
        merged_phase_is_conditional_hypothesis=True, connected_topology_certified=False)


def _simulation(simulation):
    result = {}
    for scene in ('S1', 'S2', 'S3'):
        result[scene] = {}
        for quantity in ('D_nm', 'diameter_nm'):
            stat = (simulation or {}).get(scene, {}).get(quantity, {})
            numeric = bool(_finite(stat.get('rows')) and stat['rows'] > 0)
            metrics = all(_finite(stat.get(k)) for k in ('R2_log', 'coverage68', 'coverage95'))
            if not numeric:
                status, why = 'pending', 'own original independent TEST quantity statistics missing'
            elif stat.get('invalid_original_joint_rows', 0) > 0:
                status, why = 'failed', 'failed original joint rows prevent quantity calibration'
            elif not metrics:
                status, why = 'pending', 'own original recovery/coverage statistics incomplete'
            else:
                numerical = (stat['R2_log'] >= .85 and abs(stat['coverage68']-.68) <= .05+1e-12
                             and abs(stat['coverage95']-.95) <= .05+1e-12)
                passed = bool(numerical and stat.get('calibrated') is True)
                status, why = ('passed', 'own original TEST recovery/coverage passed') if passed else ('failed', 'own original TEST recovery/coverage failed')
            result[scene][quantity] = _result(status, why, original_statistics=copy.deepcopy(stat),
                quantity_definition='effective-column correlation repeat / footprint' if scene == 'S3' else 'centre-density D / primitive diameter')
    return result


def _cohorts(cohorts, required):
    result = {}
    for name in required:
        cohort = (cohorts or {}).get(name, {})
        denominator = cohort.get('original_peak_denominator'); fraction = cohort.get('any_scenario_PPC_fraction')
        ready = bool(_finite(denominator) and denominator > 0 and _finite(fraction) and 0 <= fraction <= 1)
        core = _result('passed' if ready and fraction >= .60 else 'failed' if ready else 'pending',
            'original conservative any-scenario PPC fraction' if ready else 'original cohort/PPC denominator missing',
            original_peak_denominator=denominator, any_scenario_PPC_fraction=fraction)
        optional = {}
        scenes = ('S1',) if name.endswith('/Si') else ('S1', 'S2', 'S3')
        for scene in scenes:
            optional[scene] = {}
            for quantity in ('D_nm', 'diameter_nm'):
                item = cohort.get('optional_consistency', {}).get(scene, {}).get(quantity, {})
                n = item.get('original_known_host_pairs'); value = item.get('fraction')
                ready = bool(_finite(n) and n > 0 and _finite(value) and 0 <= value <= 1)
                unknown_host = name.endswith('/unknown') and n == 0
                optional[scene][quantity] = _result('not_applicable' if unknown_host else 'passed' if ready and value >= .80 else 'failed' if ready else 'pending',
                    'no known host value exists for this unknown-host cohort' if unknown_host else
                    'original given/unknown host consistency fraction' if ready else 'original optional-input paired evidence missing',
                    original_known_host_pairs=n, fraction=value)
        result[name] = dict(core_real=core, optional=optional,
            status=_and([core]+[v for scene in optional.values() for v in scene.values()]))
    return result


def c5_acceptance(simulation, real_cohorts, frame_quantity_evidence, *, quantity_gate,
                  s4_observed=None, required_cohorts=None,
                  complete_original_evidence=False, expected_original_windows=223,
                  generation_evidence=None):
    'Combine verified actual evidence; do not release a frame or select a model.'
    frames = list(frame_quantity_evidence or [])
    if required_cohorts is None:
        required_cohorts = sorted({f['metal']+'/'+f['host_category'] for f in frames
            if isinstance(f.get('metal'), str) and isinstance(f.get('host_category'), str)})
    else:
        required_cohorts = list(required_cohorts)
    sim = _simulation(simulation); real = _cohorts(real_cohorts, required_cohorts)
    s4 = (s4_feature_calibration(s4_observed.get('rows'), expected_families=s4_observed.get('expected_families'))
          if isinstance(s4_observed, dict) else
          _result('pending', 'actual original S4 latent mean response rows and expected family count missing'))
    temporal = qualified_D_time(frames, quantity_gate=quantity_gate,
                                expected_original_windows=expected_original_windows)
    original = _result('passed' if complete_original_evidence is True else 'pending',
        None if complete_original_evidence is True else 'caller has not verified complete original source-bound simulation/real TEST evidence')


    failed_ids = (generation_evidence or {}).get('failed_family_ids')
    expected = (generation_evidence or {}).get('original_family_denominator')
    generation_ready = bool(isinstance(failed_ids, list)
        and isinstance(expected, int) and not isinstance(expected, bool) and expected > 0
        and all(isinstance(fid, int) and not isinstance(fid, bool) for fid in failed_ids)
        and failed_ids == sorted(set(failed_ids)) and len(failed_ids) <= expected)
    generation = _result('failed' if generation_ready and failed_ids else
        'passed' if generation_ready else 'pending',
        'original generation failure prevents complete C5 acceptance' if generation_ready and failed_ids else
        None if generation_ready else 'verified original-family generation evidence missing',
        original_family_denominator=expected, failed_family_ids=copy.deepcopy(failed_ids),
        failed_original_families=len(failed_ids) if generation_ready else None,
        optional_policy_duplicates_removed=True)
    checks = [original, generation, s4, temporal]
    checks.extend(v for scene in sim.values() for v in scene.values())
    checks.extend(_result(v['status']) for v in real.values())
    if not required_cohorts:
        checks.append(_result('pending', 'original material/substrate cohorts missing'))
    status = _and(checks)
    reasons = [item.get('reason') or item['status'] for item in checks
               if item['status'] not in ('passed', 'not_applicable')]
    return dict(schema=SCHEMA, status=status, C5_complete=status == 'passed',
        original_evidence=original, original_generation_gate=generation,
        simulation=sim, S4_observed_features=s4,
        real_cohorts=real, required_cohorts=required_cohorts,
        qualified_centre_D_time=temporal, blocking_reasons=reasons,
        model_selection_using_TEST=False, structural_release=False,
        topology_certified=False, new_thresholds_added=False)
