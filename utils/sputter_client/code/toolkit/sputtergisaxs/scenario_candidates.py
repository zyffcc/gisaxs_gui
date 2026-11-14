'Representative conditional structures from actual decoded joint samples.'
from contextlib import ExitStack
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .scenario_chart import SCENARIO_NAMES
from .scenario_credibility import D_semantics
from .scenario_runtime import _array_sha, _frame_identity, _inside, _json, _sha, _source_sha
from .v2_ppc import GATES, STAT_SOURCES

SCHEMA = 'conditional_original_joint_representative_candidates_1'
OBVIOUS_CONTRADICTION_MULTIPLIER = 3.
LABELS = {'S1':'表面独立金属岛', 'S2':'沿宿主排列的金属帽',
          'S3':'部分渗入的有效金属柱', 'S4':'条带等效网络条件近似'}
DISTINGUISH = {
    'S1':'用SEM的帽尺寸及中心统计、随时间的峰位移动区分；D是条件密度长度，SEM近邻统计需单独对应。',
    'S2':'比较有明确来源的沉积前宿主峰和生长峰；峰可追随宿主，金属帽及金属密度长度仍是条件结构。',
    'S3':'结合截面或深度测量区分露出与埋入；D是横向相关重复长度，足迹是有效柱截面。',
    'S4':'用连通网络图像与尺度统计区分；条带等效宽度／缝隙需要额外几何假设，TS谱本身不决定拓扑。'}


def _finite(value, *, nonnegative=False):
    good = (not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.number))
            and np.ndim(value) == 0 and np.isfinite(value))
    return bool(good and (value >= 0 if nonnegative else value > 0))


def _structure(physical, scene):
    if physical.get('scenario') != scene:
        raise ValueError('Original decoded draw changed scenario')
    if scene == 'S4':
        ts = physical['ts']; domain = float(ts['domain_nm']); fraction = float(ts['fraction'])
        values = dict(network_spacing_nm=domain, metal_band_width_nm=fraction*domain,
            gap_width_nm=(1-fraction)*domain, height_nm=float(ts['thickness_nm']), coverage_fraction=fraction,
            model_metal_volume_per_projected_area_nm=fraction*float(ts['thickness_nm']))
        note = '网络带宽／缝隙采用一维周期条带等效：带宽=相分数×谱域尺度；高度为有效层厚度。'
        extra = dict(definition='stripe_equivalent_TS_conditional_approximation',
            connected_topology_proven=False, stripe_geometry_is_not_TS_ground_truth=True,
            surface_union_coverage_proven=False)
    else:
        particle = physical['particle']; particle = particle if isinstance(particle, dict) else vars(particle)
        D = float(physical['D_nm']); diameter = float(physical['diameter_nm'])
        signed_gap = D-diameter
        if scene == 'S3':
            values = dict(D_nm=D, footprint_diameter_nm=diameter,
                exposed_height_nm=float(physical['exposed_span_nm']),
                burial_depth_nm=float(physical['burial_span_nm']), gap_nm=max(signed_gap, 0.),
                coverage_fraction=float(physical['coverage_fraction']))
            steps = physical['steps']
            values['model_metal_volume_per_projected_area_nm'] = float(
                np.dot(np.diff(steps['edges_nm']), steps['metal_fractions']))
            note = '露出／埋入高度是有效柱支撑范围；间隙由相关重复长度减中位足迹近似。'
            extra = dict(definition='effective_column_correlation_and_footprint',
                centre_distance_inferred=False, union_coverage_proven=False,
                signed_repeat_minus_footprint_nm=signed_gap, conditional_overlap_or_contact=signed_gap <= 0)
        else:
            sigma = float(particle['sigma_log']); density = 2/(np.sqrt(3)*D*D)
            occupation = density*np.pi*(diameter/2)**2*np.exp(2*sigma*sigma)
            values = dict(D_nm=D, diameter_nm=diameter, height_nm=float(particle['height_nm']),
                          gap_nm=max(signed_gap, 0.), coverage_fraction=float(occupation))
            a = float(particle['base_radius_nm']); h = float(particle['height_nm'])
            volume = np.pi*h*(3*a*a+h*h)/6 if particle['shape']=='cap' else 2*np.pi*a*a*h/3
            values['model_metal_volume_per_projected_area_nm'] = float(density*volume*np.exp(4.5*sigma*sigma))
            note = 'D按金属原语密度换算；间隙为中位足迹近似，覆盖率为平均足迹面积和，重叠时不等于面积并集。'
            note += ' 金属体积／投影面积是该条件模型的体积预算，不是由未知通量强度标定的总沉积量。'
            if scene == 'S2':
                note += ' 金属帽置于宿主表面；宿主峰位置可区别于金属帽密度间距。'
            extra = dict(definition='conditional_primitive_density_length_and_median_footprint',
                density_nm_minus2=float(density), mean_footprint_area_factor=float(np.exp(2*sigma*sigma)),
                signed_density_length_minus_diameter_nm=signed_gap,
                conditional_overlap_or_contact=signed_gap <= 0,
                footprint_area_sum_exceeds_one=bool(occupation > 1), union_coverage_proven=False,
                host_peak_may_differ_from_metal_D=scene == 'S2')
            if physical.get('small_clusters'):
                note += ' 本行尺寸、覆盖面积和及模型体积仅指主粒群；新成核小团簇另作为散射群，未并入这些量。'
                extra.update(main_population_only=True,small_population_omitted=True,
                    additional_population_parameters=dict(physical['small_clusters']))
    if any(not _finite(v, nonnegative=('gap' in k or 'burial' in k or 'exposed' in k)) for k, v in values.items()):
        raise ValueError('Every displayed conditional structure quantity must be finite and physical')
    return values, note, extra


def _partition(values, original_indices, maximum):
    'Deterministic posterior quantization, not a claim of multimodal truth.'
    names = tuple(values[0]); x = np.array([[v[k] for k in names] for v in values], float)
    for j, name in enumerate(names):
        x[:,j] = np.log1p(x[:,j]) if 'gap' in name or 'burial' in name or 'exposed' in name else np.log(x[:,j])
    scale = np.std(x, axis=0); scale[scale < 1e-10] = 1.
    x = (x-np.median(x, axis=0))/scale
    distance = cdist(x, x, metric='sqeuclidean')
    medoids = [int(np.argmin(distance.sum(axis=1)))]; k = min(maximum, len(values))
    while len(medoids) < k:
        nearest = distance[:,medoids].min(axis=1); nearest[medoids] = -1
        next_index = int(np.argmax(nearest))
        if nearest[next_index] < 1e-10:
            break
        medoids.append(next_index)
    for _ in range(12):
        labels = np.argmin(distance[:,medoids], axis=1); updated = []
        for j, old in enumerate(medoids):
            members = np.flatnonzero(labels == j)
            updated.append(int(members[np.argmin(distance[np.ix_(members, members)].sum(axis=1))]) if len(members) else old)
        if updated == medoids:
            break
        medoids = updated
    labels = np.argmin(distance[:,medoids], axis=1)
    return [(m, np.flatnonzero(labels == j).tolist()) for j, m in enumerate(medoids)]


def _peak_kind(descriptor):
    if not isinstance(descriptor, dict):
        return 'unknown'
    support = descriptor.get('support')
    bands = descriptor.get('bands_support', []) if descriptor.get('AF_band') is None else []
    if support == 'bilateral' or 'bilateral' in bands:
        return 'bilateral'
    if support == 'one_side' or 'one_side' in bands:
        return 'one_side'
    if support == 'bilateral_conflict' or 'bilateral_conflict' in bands:
        return 'conflicting_bilateral'


    known_absence = (descriptor.get('no_peak_with_sufficient_support') is True
        or descriptor.get('peak_classification_status') == 'no_peak_with_sufficient_support'
        or (descriptor.get('peak_measurability_known') is True
            and descriptor.get('peak_measurability') == 'no_resolved_peak'))
    return 'no_resolved_peak' if known_absence else 'unknown'


def _matching(observed, predicted):
    results = {}; ratios = []; contradictions = 0; checked = 0
    pending = predicted is None or bool(predicted.get('forward_failed', False))
    for name, base in GATES.items():
        actual = observed.get('values', {}).get(name); value = predicted.get('values', {}).get(name) if predicted else None
        known = _finite(actual); resolved = _finite(value) and not pending
        reason = observed.get('feature_missing_reasons', {}).get(name) or '该观测特征未提取'
        gate = base; se = observed.get('stat_se', {}).get(name); src = observed.get('stat_se_source', {}).get(name)
        if se is not None:
            if not _finite(se, nonnegative=True) or src not in STAT_SOURCES:
                raise ValueError('Only original declared count-statistical SE can widen a representative feature gate')
            if known:
                gate = max(base, 2*float(se)/(1. if name == 'Yoneda' else float(actual)))
        error = abs(float(value)-float(actual))/(1. if name == 'Yoneda' else float(actual)) if known and resolved else 0.
        if known and resolved:
            checked += 1; ratios.append(error/gate); contradictions += error > OBVIOUS_CONTRADICTION_MULTIPLIER*gate
        results[name] = dict(observed=float(actual) if known else '缺测：'+str(reason),
            representative=float(value) if resolved else '尚待模拟核对' if pending else
                '缺测：'+str(predicted.get('feature_missing_reasons', {}).get(name) or '代表图案无此可测特征'),
            status='匹配' if known and resolved and error <= gate else '有差异' if known and resolved else
                '观测缺测' if not known else '尚待模拟核对' if pending else '代表图案缺测',
            comparable=bool(known and resolved), effective_gate=float(gate),
            error=float(error) if known and resolved else '无可比较数值',
            gate_units='degree' if name == 'Yoneda' else 'relative', calibration_score=False)
        if known and not resolved and not pending:
            contradictions += 1; ratios.append(OBVIOUS_CONTRADICTION_MULTIPLIER+1.)
    actual_kind = _peak_kind(observed); predicted_kind = _peak_kind(predicted) if not pending else 'unknown'
    support_known = actual_kind != 'unknown' and predicted_kind != 'unknown'
    support_match = bool(support_known and actual_kind == predicted_kind)
    if support_known and not support_match:
        contradictions += 1; ratios.append(OBVIOUS_CONTRADICTION_MULTIPLIER+1.)
    observed_known = sum(_finite(observed.get('values', {}).get(k)) for k in GATES)
    all_comparable = checked == observed_known
    fits = bool(not pending and all_comparable and (checked > 0 or support_known)
                and all(r <= 1+1e-12 for r in ratios) and (support_match or not support_known))
    return dict(results, features=results, peak_measurability=dict(observed=actual_kind, representative=predicted_kind,
        comparable=support_known, matching=support_match, approximation='代表图案可测性；未估计检测概率'),
        observed_numeric_features=observed_known, compared_numeric_features=checked,
        all_observed_numeric_features_comparable=all_comparable, all_five_features_comparable=checked == len(GATES),
        matches_measured_features=fits, obvious_contradiction=bool(not pending and contradictions),
        rank_score=float(np.mean(ratios)) if ratios else 0. if support_match else '尚待模拟核对',
        comparison_kind='代表抽样特征匹配；不等于完整后验预测检验', calibration_score=False)


def _prediction_maps(sampled, predictive):
    result = {s:{} for s in SCENARIO_NAMES}
    if predictive is None:
        return result
    for scene, part in predictive.get('scenarios', {}).items():
        if scene not in result:
            raise ValueError('Unknown provided predictive scenario')
        joint = sampled['scenarios'][scene].get('original_joint')
        pins = part.get('original_joint_sha256')
        if pins is not None and (joint is None or pins != {k:_array_sha(v) for k, v in joint.items()}):
            raise ValueError('Provided representative features belong to a different original joint pool')
        original_indices = {int(d['index']) for d in sampled['scenarios'][scene]['draws']}
        for draw in part.get('draws', []):
            index = int(draw['index'])
            if index not in original_indices or index in result[scene]:
                raise ValueError('Provided predictive original draw index differs or duplicates')
            result[scene][index] = draw
    return result


def observe_candidate_frame(frame, *, AF_band=None, estimate_stat_se=True):
    'Keep a comparison band even when neither side has a resolved peak.'
    from .stage5_measurement import counts_profiles
    from .prediction import summarize_profiles
    from .resolvable import PEAK_WINDOW
    from .v2_observation import ppc_features
    hy,hn,vy,vn,_,_ = counts_profiles(frame.Y, frame.P)
    fits,bands,selected,_ = summarize_profiles(hy,hn,vy,vn,PEAK_WINDOW)
    def usable(f):
        return (f.get('status') in ('resolved_empirical_feature','unresolved_or_boundary_feature')
            and f.get('optimizer_success') is True and f.get('gap_near_peak') is not True)
    b = AF_band
    if b is None:
        b = selected['AF_band'] if selected else next(
            (i for i in range(4) if all(usable(f) for f in fits[2*i:2*i+2])),None)
    observed = ppc_features(frame,AF_band=b,estimate_stat_se=estimate_stat_se)
    pair = fits[2*b:2*b+2] if b is not None else []
    resolved = sum(f.get('status')=='resolved_empirical_feature' for f in pair)
    no_peak = bool(pair and all(usable(f) for f in pair) and resolved==0)
    observed.update(peak_measurability_known=bool(resolved or no_peak),
        no_peak_with_sufficient_support=no_peak,
        comparison_band_source='caller_fixed_AF_band' if AF_band is not None else 'observed_selected_peak_band' if selected else
            'first_two_supported_nongapped_fits' if b is not None else 'no_supported_comparison_band')
    fixed = bands[b] if b is not None else None
    if fixed and fixed['support']=='one_side':
        observed['values']['q_star'] = fixed['q_star']
        observed['values']['FWHM'] = None
        observed['feature_missing_reasons']['q_star'] = None
        observed['feature_missing_reasons']['FWHM'] = '单侧峰；不合成双侧峰宽'
    return observed


def build_candidates(sampled, observed, predictive=None, max_per_scenario=3):
    'Choose actual medoid draws, conditional intervals, feature/source evidence.'
    if (not isinstance(max_per_scenario, Integral) or isinstance(max_per_scenario, bool)
            or not 1 <= max_per_scenario <= 3):
        raise ValueError('Choose one to three actual representatives per scenario')
    if not isinstance(observed, dict) or not isinstance(observed.get('values'), dict):
        raise ValueError('Actual observed feature descriptor required, including explicit missing values')
    prediction = _prediction_maps(sampled, predictive); scenes = {}; diagnostics = {}
    original_total = 0; failed_total = 0
    for scene in SCENARIO_NAMES:
        part = sampled['scenarios'][scene]; draws = part['draws']
        indices = [int(d['index']) for d in draws]
        if len(set(indices)) != len(indices) or indices != sorted(indices):
            raise ValueError('Original joint draw indices must remain unique and ordered')
        declared = int(part.get('original_draws', len(draws)))
        if declared != len(draws):
            raise ValueError('Original draw denominator changed before representative selection')
        if not part['applicable']:
            scenes[scene] = dict(status='不适用', reason=part.get('not_applicable_reason') or '该宿主类别不采用此情景', candidates=[])
            diagnostics[scene] = dict(original_draws=declared, not_applicable=True)
            continue
        valid = []; geometries = []; notes = []; extras = []; failures = []
        for draw in draws:
            try:
                if draw.get('decode_failed') or draw.get('physical') is None:
                    raise ValueError(draw.get('reason') or 'Original complete joint draw did not decode')
                values, note, extra = _structure(draw['physical'], scene)
                valid.append(draw); geometries.append(values); notes.append(note); extras.append(extra)
            except (ValueError, TypeError, KeyError, OverflowError, FloatingPointError) as error:
                failures.append(dict(original_draw=int(draw['index']), reason=str(error)))
        original_total += declared; failed_total += len(failures)
        diagnostics[scene] = dict(original_draws=declared, successful_geometry_draws=len(valid),
            failed_original_draws=len(failures), failure_records=failures,
            original_joint_sha256={k:_array_sha(v) for k, v in part['original_joint'].items()}
                if part.get('original_joint') is not None else '调用者未附联合向量；保留原抽样编号',
            successful_draws_are_not_full_pool_calibration=True)
        if not valid:
            scenes[scene] = dict(status='needs_prior_candidates', reason='原联合抽样均未得到有限结构；需要调用者提供实际条件先验或库样本', candidates=[])
            continue
        candidates = []
        for cluster_id, (medoid, members) in enumerate(_partition(geometries, [d['index'] for d in valid], int(max_per_scenario))):
            representative = valid[medoid]; index = int(representative['index']); values = geometries[medoid]
            intervals = {}
            for name in values:
                q = np.quantile([geometries[m][name] for m in members], [.025, .16, .5, .84, .975])
                intervals[name] = dict(value=float(values[name]), representative=float(values[name]), conditional_cluster_median=float(q[2]),
                    interval68=q[[1,3]].tolist(), interval95=q[[0,4]].tolist(),
                    interval_definition='该条件聚类分位区间；代表取实际medoid，可能位于某量的边缘；不是独立验证覆盖率')
            match = _matching(observed, prediction[scene].get(index))
            candidates.append(dict(candidate_id=scene+'_C'+str(cluster_id+1), scenario=scene,
                scenario_label=LABELS[scene], original_draw_index=index, representative_index=index, representative_is_original_medoid=True,
                quantities=intervals, geometry_conditions=extras[medoid], conditions=notes[medoid],
                D_semantics=D_semantics(scene), distinguish=DISTINGUISH[scene], matching=match,
                source=representative.get('candidate_source', part.get('candidate_source', sampled.get('candidate_source', '同一原联合条件样本中的代表结构'))),
                candidate_source=representative.get('candidate_source', part.get('candidate_source', sampled.get('candidate_source', '同一原联合条件样本中的代表结构'))),
                cluster=dict(original_draw_indices=[int(valid[m]['index']) for m in members],
                    original_successful_members=len(members), fraction_of_original_pool=len(members)/declared,
                    quantization='deterministic geometry k-medoids; groups are not proven distinct physical modes'),
                conditional_candidate=True, calibration_score=False))
        any_fit = any(c['matching']['matches_measured_features'] for c in candidates)
        displayed = candidates
        def order(c):
            m = c['matching']; score = m['rank_score']
            return (not m['matches_measured_features'], score if isinstance(score, (int, float)) else float('inf'),
                    -c['cluster']['original_successful_members'], c['original_draw_index'])
        displayed.sort(key=order)
        scenes[scene] = dict(status='conditional_candidates', candidates=displayed,
            matching_summary='存在与已测特征相合的代表' if any_fit else '保留最接近的条件代表；逐项匹配程度见表',
            all_matching_pending=all(not isinstance(c['matching']['rank_score'], (int, float)) for c in displayed),
            scenario_probability_is_reference_only=True)
        diagnostics[scene]['hidden_obviously_conflicting_representative_indices'] = []


    global_fit = any(c['matching']['matches_measured_features']
        for group in scenes.values() for c in group['candidates'])
    for scene in list(scenes):
        group = scenes[scene]
        if group['status'] != 'conditional_candidates':
            continue
        retain_nearest = (group['candidates'][0]['original_draw_index']
            if global_fit and group['candidates'] and
            all(c['matching']['obvious_contradiction'] for c in group['candidates']) else None)
        hidden = [c['original_draw_index'] for c in group['candidates']
                  if global_fit and c['matching']['obvious_contradiction']
                  and c['original_draw_index'] != retain_nearest]
        diagnostics[scene]['hidden_obviously_conflicting_representative_indices'] = hidden
        diagnostics[scene]['hidden_only_when_any_scenario_has_a_matching_representative'] = bool(global_fit)
        diagnostics[scene]['nearest_conflicting_medoid_retained_for_scenario_comparison'] = retain_nearest
        if hidden:
            group['candidates'] = [c for c in group['candidates'] if c['original_draw_index'] not in hidden]
            group['all_matching_pending'] = all(not isinstance(c['matching']['rank_score'], (int, float))
                for c in group['candidates'])
    observation = {k:float(observed['values'][k]) if _finite(observed['values'].get(k)) else
        '缺测：'+str(observed.get('feature_missing_reasons', {}).get(k) or '该观测特征未提取') for k in GATES}
    kind = _peak_kind(observed)
    return dict(schema=SCHEMA, scenarios=scenes, observation=observation, observed_peak_kind=kind,
        observed_input_source='观测选定的双侧峰特征' if kind == 'bilateral' else
            '单侧信号；双侧峰位／峰宽缺测，不补造数值' if kind == 'one_side' else
            '图像可比较特征与显式缺测；缺失几何信息的候选依赖模型条件先验',
        input_sources=sampled.get('input_sources', {'posterior_input_mode':sampled.get('receipt', {}).get('input_mode', '调用者提供的实际条件样本'),
            'reference_frame':'调用者未提供已登记参考帧；不把最早帧视为零金属',
            'temporal_continuation':'调用者未提供相邻帧延续证据'}),
        reference_scenario_probabilities=sampled.get('reference_scenario_probabilities', '调用者未提供情景概率'),
        developer=dict(original_applicable_draws=original_total, failed_original_draws=failed_total,
            scenarios=diagnostics, no_draw_replacement=True, structural_release=False,
            CAL_TEST_gates_changed=False, posterior_predictive_validation_performed=False),
        status='needs_prior_candidates' if original_total == failed_total else 'conditional_candidates')


def preview_medoid_features(sampled, candidates, full, public, *, runtime=None, bank_plan=None,
        feature_certificate=None, host_shapes=None, host_metadata=None, harddisk=None, backend='cpu', observed=None):
    'Only original medoids: bound forward, scalar gain, clean-mean descriptors.'
    if runtime is not None:
        from .scenario_frozen_worker import FrozenManifestRuntime
        if isinstance(runtime, FrozenManifestRuntime):
            return runtime.preview_medoid_features(sampled, candidates, full, public,
                bank_plan=bank_plan, feature_certificate=feature_certificate,
                backend=backend, observed=observed)
        from .scenario_mainline_posterior import MainlineRuntime
        if isinstance(runtime,MainlineRuntime):
            from .scenario_mainline_preview import preview_medoid_features as mainline_preview
            return mainline_preview(sampled,candidates,full,public,runtime=runtime,
                bank_plan=bank_plan,feature_certificate=feature_certificate,
                host_shapes=host_shapes,host_metadata=host_metadata,harddisk=harddisk,
                backend=backend,observed=observed)
    from .scenario_forward import QUICK, mean_forward, harddisk_table
    from .v2_observation import CountFrame, ppc_features
    identity = _frame_identity(full, public)
    observed = (observe_candidate_frame(public,estimate_stat_se=identity['photon_statistics_known'])
                if observed is None else observed)
    source = dict(kind='actual_parameter_host_conditional_prior', certified=False)
    asset_error = None
    with ExitStack() as stack:
        hostarr = {} if host_shapes is None else host_shapes
        disk = harddisk
        if runtime is not None:
            try:
                if sampled.get('receipt') != runtime.receipt or sampled.get('frame_identity') != identity:
                    raise ValueError('Original sampled runtime/full-public observation differs')
                if bank_plan is None or feature_certificate is None:
                    raise ValueError('Actual original bank plan and feature certificate required')
                plan_path = Path(bank_plan).resolve(); plan = _json(plan_path); bank = runtime.declaration['bank']
                cert_path = Path(feature_certificate).resolve(); cert = _json(cert_path)
                if (_sha(plan_path) != bank['plan_sha256'] or plan['source_pins'] != bank['source_pins']
                        or _sha(cert_path) != bank['certificate_sha256'] or cert.get('passed') is not True
                        or cert.get('TEST_decoded') is not False or cert.get('quick') != QUICK
                        or any(cert.get('source_pins', {}).get(k) != v for k, v in bank['source_pins'].items())):
                    raise ValueError('Original source-bound forward plan/certificate differs')
                for relative, digest in bank['source_pins'].items():
                    if (_source_sha(_inside(runtime.producer_package, relative)) != digest
                            or _source_sha(_inside(runtime.package, relative)) != digest):
                        raise ValueError('Original representative forward source changed: '+relative)
                def asset(name, supplied):
                    item = plan['bindings'][name]
                    path = _inside(plan_path.parent, item['path']) if supplied is None else Path(supplied).resolve()
                    if _sha(path) != item['sha256']:
                        raise ValueError('Original representative TRAIN asset changed: '+name)
                    return path
                hp = asset('host_shapes.npz', host_shapes); hm = asset('host_shapes.json', host_metadata)
                dp = asset('harddisk.npz', harddisk)
                if tuple(_json(hm)['rows']) != runtime.templates:
                    raise ValueError('Original TRAIN host-template metadata differs')
                hostarr = stack.enter_context(np.load(hp, allow_pickle=False))
                disk = harddisk_table(stack.enter_context(np.load(dp, allow_pickle=False)))
                source = dict(kind='actual_trained_original_SOURCE_and_TRAIN_assets',
                    plan_sha256=_sha(plan_path), certificate_sha256=_sha(cert_path),
                    host_shapes_sha256=_sha(hp), harddisk_sha256=_sha(dp), certified=False)
            except (ValueError, TypeError, KeyError, OSError) as error:
                asset_error = str(error)
        if not identity['photon_statistics_known']:
            asset_error = '原观测计数单位未确立，不能拟合Poisson标量强度'
        output = {}
        for scene, group in candidates['scenarios'].items():
            part = sampled['scenarios'][scene]; records = {int(d['index']):d for d in part['draws']}; descriptors = []
            for candidate in group['candidates']:
                index = candidate['original_draw_index']; record = records[index]
                try:
                    if asset_error is not None:
                        raise ValueError(asset_error)
                    if record.get('decode_failed') or record.get('physical') is None:
                        raise ValueError('Original representative joint draw has no physical structure')
                    physical = record['physical']
                    if physical['scenario'] != scene:
                        raise ValueError('Original representative scenario changed')
                    if physical.get('host_template_key') and (hostarr is None or physical['host_template_key']+'__shape' not in hostarr):
                        raise ValueError('实际条件抽样的宿主谱资产未提供')
                    if physical.get('structure_factor') == 'harddisk' and disk is None:
                        raise ValueError('该原抽样需要实际harddisk表；尚待模拟核对')
                    rate, _, receipt = mean_forward(physical, full.P, hostarr, recipe=QUICK, backend=backend, disk=disk)
                    raw_mu = np.asarray(rate, float)*public.P
                    if (not np.isfinite(raw_mu).all() or np.any(raw_mu < 0)
                            or np.any(raw_mu[public.P == 0] != 0) or float(raw_mu.sum()) <= 0):
                        raise ValueError('Actual representative mean is not finite positive supported counts')
                    gain = float(public.Y.sum())/float(raw_mu.sum()); mu = raw_mu*gain
                    from .scenario_fixed_af_observation import observe_candidate_frame_fixed_af
                    descriptor = observe_candidate_frame_fixed_af(CountFrame(mu, public.P, False),
                        AF_band=observed['AF_band'],estimate_stat_se=False)
                    descriptor.update(index=index, fitted_scalar_gain=gain, representative_clean_mean=True,
                        original_joint_modified=False, count_support_applications=1,
                        raw_expected_photons_sha256=_array_sha(raw_mu), profiled_mean_photons_sha256=_array_sha(mu),
                        forward_receipt=receipt, posterior_predictive_replica=False)
                except (ValueError, TypeError, KeyError, OverflowError, FloatingPointError, RuntimeError, MemoryError, OSError) as error:
                    descriptor = dict(index=index, forward_failed=True, AF_band=observed['AF_band'],
                        values={k:None for k in GATES}, feature_missing_reasons={k:'尚待模拟核对：'+str(error) for k in GATES},
                        failure_reason=str(error), representative_clean_mean=True, posterior_predictive_replica=False)
                descriptors.append(descriptor)
            output[scene] = dict(draws=descriptors,
                original_joint_sha256={k:_array_sha(v) for k, v in part['original_joint'].items()}
                    if part.get('original_joint') is not None else None,
                original_medoid_indices=[c['original_draw_index'] for c in group['candidates']])
    return dict(scenarios=output, match_kind='profiled_clean_mean_original_medoids', source=source,
        frame_identity=identity, mean_image_is_not_posterior_average=True,
        full_posterior_predictive_check=False, new_photon_replica_drawn=False,
        original_physical_records_changed=False, structural_release=False)
