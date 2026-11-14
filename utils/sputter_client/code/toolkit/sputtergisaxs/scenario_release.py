'Four-scenario quantity gates from actual evidence and original samples.'
from datetime import datetime, timezone
import copy
import numpy as np

from .scenario_chart import SCENARIO_NAMES
from .scenario_runtime import _array_sha
from .scenario_credibility import D_semantics, D_SEMANTIC_VERSION
from .v2_ppc import GATES

SCHEMA = 'four_scenario_quantity_release_1'
QUANTITIES = ('D_nm', 'diameter_nm', 'height_nm', 'gap_nm', 'coverage_fraction',
              'depth_span_nm', 'exposed_span_nm', 'burial_span_nm')
NOT_BEFORE = datetime(2026, 10, 17, 22, tzinfo=timezone.utc)
POLICY = dict(log_R2_minimum=.85, coverage_absolute_tolerance=.05,
    real_any_scenario_PPC_fraction=.60, optional_consistency_fraction=.80,
    variant_relative_spread=.15, outside_probability_maximum=.5,
    feature_gates=GATES, quantity_specific_calibration=True,
    height_point_estimate_released=False, scenarios_pruned=False,
    optimiser_refit_required=False, failed_draws_removed=False,
    D_semantic_version=D_SEMANTIC_VERSION, S3_correlation_D_released_as_centre_spacing=False)


def finite(value):
    return isinstance(value, (int,float,np.number)) and not isinstance(value,(bool,np.bool_)) and bool(np.isfinite(value))


def joint_identity(sampled):
    return {s:{k:_array_sha(v) for k,v in sampled['scenarios'][s]['original_joint'].items()}
            for s in SCENARIO_NAMES}


def pending_qualification(reason='Actual independent new TEST evidence is not supplied'):
    return dict(schema=SCHEMA, status='pending', policy=copy.deepcopy(POLICY),
        simulation={s:{q:dict(calibrated=False, reason=reason) for q in QUANTITIES}
                    for s in SCENARIO_NAMES}, real_cohorts={}, evidence_binding=None,
        complete_original_TEST_evidence=False, C5_complete=False)


def _same_model(receipt, expected):
    return all(receipt.get(k)==expected.get(k) for k in
               ('weights_sha256','state_sha256','plan_sha256','input_mode','selected_epoch',
                'execution_source_pins','architecture','architecture_spec'))


def _statistics(rows, quantity):
    'Original-row arithmetic; a failed row prevents certification.'
    selected=[r for r in rows if r['eligible_in_range_visible_peak'] is not False]
    items=[r['quantities'].get(quantity) for r in selected]
    n=len(items)
    valid=bool(n and all(i is not None and i.get('valid_complete_original_joint') is True
                        and finite(i.get('truth')) and i['truth']>0
                        and finite(i.get('median')) and i['median']>0 for i in items))
    out=dict(rows=n, invalid_original_joint_rows=sum(i is None or i.get('valid_complete_original_joint') is not True for i in items),
             R2_log=None, coverage68=None, coverage95=None, calibrated=False)
    if not valid:return out
    truth=np.log([i['truth'] for i in items]); pred=np.log([i['median'] for i in items])
    den=float(np.sum((truth-truth.mean())**2))
    out['R2_log']=float(1-np.sum((truth-pred)**2)/den) if den>0 else None
    for level in (68,95):out['coverage'+str(level)]=float(np.mean([i['coverage'+str(level)+'_hit'] is True for i in items]))
    coverage_ok=all(abs(out['coverage'+str(level)]-nominal)<=.05+1e-12 for level,nominal in ((68,.68),(95,.95)))
    recovery_ok=quantity not in ('D_nm','diameter_nm') or (finite(out['R2_log']) and out['R2_log']>=.85)
    out['calibrated']=bool(coverage_ok and recovery_ok)
    return out


def qualify_evidence(runtime_receipt, selection=None, test_plan=None, simulated=None, real=None):
    'Validate original new TEST identities and recompute quantity calibration.'
    if all(x is None for x in (selection,test_plan,simulated,real)):return pending_qualification()
    if datetime.now(timezone.utc)<NOT_BEFORE:
        raise RuntimeError('Independent TEST evidence is locked until 2026-10-18 Europe/Berlin')
    if any(x is None for x in (selection,test_plan,simulated,real)):
        return pending_qualification('Frozen selection, independent simulation TEST and complete real TEST evidence are all required')
    if (selection.get('schema')!='four_scenario_frozen_selection_1'
            or selection.get('status')!='frozen' or selection.get('TEST_inspected') is not False
            or selection.get('frozen_before_any_TEST') is not True
            or selection['full_bank_plan_sha256']!=runtime_receipt['plan_sha256']):
        raise ValueError('Original pre-TEST model selection required')
    matches=[m for m in selection['models'] if m['sha256']==runtime_receipt['weights_sha256'] and m['mode']==runtime_receipt['input_mode']]
    if len(matches)!=1:raise ValueError('Exactly one original frozen entry must bind this actual model')
    model=matches[0]
    if (test_plan.get('schema')!='four_scenario_new_TEST_bank_1'
            or test_plan.get('namespace')!=5000000 or test_plan.get('root_seed')!=94627581307
            or test_plan['full_bank_plan_sha256']!=runtime_receipt['plan_sha256']):
        raise ValueError('Independent original new TEST plan required')
    n=test_plan['per_scenario']; expected=[int(5000000+s*100000+10*k+9) for s in range(4) for k in range(n)]
    identity=simulated['identity']; records=simulated['records']; policies=('none','all','declared')
    if (simulated.get('status')!='completed_new_TEST_evaluation'
            or not _same_model(identity['runtime'],runtime_receipt)
            or identity['selection_sha256']!=test_plan['selection_sha256']
            or simulated['completed_original_TEST_family_ids']!=expected
            or len(records)!=len(expected)*3 or identity['draws']!=selection['posterior_draws']
            or identity.get('model_source',{}).get('original_producer_source_pins')!=model.get('producer_source',{}).get('source_pins')
            or identity.get('frozen_execution_source_pins')!=selection['source_pins']
            or identity['seed']!=selection['evaluation_seed']):
        raise ValueError('Complete original frozen-model simulated TEST evidence required')
    for j,fid in enumerate(expected):
        for k,policy in enumerate(policies):
            r=records[j*3+k]
            if (r['family_id']!=fid or r['optional_policy']!=policy
                    or r['scenario']!=SCENARIO_NAMES[j//n]
                    or r['original_draws']!=identity['draws']):
                raise ValueError('Original TEST row/scenario/policy/draw denominator differs')
    out=pending_qualification(); out.update(status='actual_evidence_computed',complete_original_TEST_evidence=True,
        selected_model_name=model['name'], original_TEST_families=len(expected),
        selected_before_TEST=True, evidence_binding=dict(selection_sha256=test_plan['selection_sha256'],
            weights_sha256=runtime_receipt['weights_sha256'],state_sha256=runtime_receipt['state_sha256'],
            full_bank_plan_sha256=runtime_receipt['plan_sha256']))
    for scene in SCENARIO_NAMES:
        rows=[r for r in records if r['optional_policy']=='none' and r['scenario']==scene]
        failed_generation=any(r.get('generation_failed') is True for r in rows)
        for quantity in QUANTITIES:
            if scene=='S4' or quantity=='gap_nm' or (scene=='S3' and quantity=='height_nm'):
                out['simulation'][scene][quantity]=dict(calibrated=False,reason='Quantity is not defined by this scenario or has no validated original geometry label')
                continue
            stat=_statistics(rows,quantity);stat['calibrated'] &= not failed_generation
            stat['reason']='own_original_TEST_recovery_and_coverage_passed' if stat['calibrated'] else 'own_original_TEST_quantity_gate_failed_or_missing'
            stat['height_is_interval_only']=quantity=='height_nm';out['simulation'][scene][quantity]=stat
    ri=real['identity']
    if (real.get('schema')!='four_scenario_held_TEST_real_PPC_1_summary'
            or real.get('status')!='completed_all_original_TEST_windows'
            or real.get('complete_original_windows') is not True
            or real.get('original_TEST_window_denominator')!=223
            or real['selection_sha256']!=test_plan['selection_sha256']
            or not _same_model(ri['runtime'],runtime_receipt) or real['model_name']!=model['name']
            or ri.get('model_source',{}).get('original_producer_source_pins')!=model.get('producer_source',{}).get('source_pins')
            or ri.get('frozen_execution_source_pins')!=selection['source_pins']
            or ri.get('OOD_run_sha256')!=model.get('OOD_run',{}).get('sha256')
            or ri['original_seed']!=selection['real_PPC']['seed']
            or ri['original_draws_per_applicable_scenario']!=selection['real_PPC']['draws']
            or ri['policies']!=['none','host'] or selection['real_PPC']['gates']!=GATES
            or ri['policy'].get('posterior_images_averaged') is not False
            or ri['policy'].get('nuisance_resampled') is not False):
        raise ValueError('Complete original frozen-model 223-window real TEST evidence required')
    numeric=real['numeric_PPC_diagnostics']
    if numeric.get('complete') is not True or numeric.get('original_window_denominator')!=223:
        raise ValueError('Original real TEST numeric denominator differs')
    for name,cohort in real['conservative_TEST_cohorts'].items():
        c=cohort['policies']['none'];den=c['original_conservative_peak_denominator']
        any_pass=c['at_least_one_PPC_consistent'];fraction=any_pass/den if den else None
        if not 0<=any_pass<=den or fraction!=c['at_least_one_fraction']:raise ValueError('Original real PPC cohort denominator differs')
        paired=numeric['cohorts'][name]['paired_optional_diagnostic'];qs={}
        for scene in SCENARIO_NAMES:
            qs[scene]={}
            p=paired.get(scene,{})
            for q in QUANTITIES:
                item=p.get('quantities',{}).get(q);kn=p.get('known_host_denominator',0)
                ok=bool(item and kn and 0<=item['known_host_pass']<=kn and item['known_host_fraction']==item['known_host_pass']/kn and item['known_host_fraction']>=.8)
                qs[scene][q]=dict(passed=ok,original_known_host_pairs=kn,
                    fraction=item.get('known_host_fraction') if item else None,
                    reason='original_optional_pair_gate_passed' if ok else 'own_optional_pair_gate_missing_or_failed')
        out['real_cohorts'][name]=dict(original_peak_denominator=den,any_scenario_PPC_fraction=fraction,
            target_passed=bool(den and fraction>=.60),optional_consistency=qs)
    from .scenario_acceptance import c5_acceptance
    generation_evidence=dict(original_family_denominator=len(expected),
        failed_family_ids=sorted({r['family_id'] for r in records
                                if r.get('generation_failed') is True}))
    s4_rows=[dict(r.get('S4_latent_features',dict(eligible_visible=None,
        original_draws=r['original_draws'],quantities={})),
        original_family_id=r['family_id'],generation_failed=r.get('generation_failed') is True,
        failure_reason=r.get('failure_reason')) for r in records
        if r['scenario']=='S4' and r['optional_policy']=='none']
    acceptance=c5_acceptance(out['simulation'],out['real_cohorts'],
        real.get('frame_quantity_evidence'),
        quantity_gate=lambda frame,policy,scene:qualified_D_from_evidence(
            frame,policy,scene,out,runtime_receipt),
        s4_observed=dict(rows=s4_rows,expected_families=n),
        complete_original_evidence=True, generation_evidence=generation_evidence)
    out.update(C5_complete=acceptance['C5_complete'],acceptance=acceptance)
    return out


def _valid_interval(item):
    try:
        values=np.asarray([*item['interval95'][:1],*item['interval68'][:1],item['median'],
                           *item['interval68'][1:],*item['interval95'][1:]],float)
        return bool(values.shape==(5,) and np.isfinite(values).all() and np.all(np.diff(values)>=0) and values[0]>=0)
    except (TypeError,KeyError,ValueError):return False


def _common_reason(scene,applicable,qualification,cohort,oid,ood_ready,
                   ppc_bound,consistent,range_info,variants):
    variant_ok=bool(variants.get('diagnostic_pass') is True
                    and variants.get('all_expected_classes_resolved'))
    range_ok=bool(range_info.get('domain_out_of_range') is False
                  and finite(range_info.get('outside_probability'))
                  and range_info['outside_probability']<=.5
                  and range_info.get('unresolved_original_draws')==0)
    return ('not_applicable' if not applicable else 'S4_has_no_island_quantities' if scene=='S4'
        else 'simulation_TEST_pending' if not qualification.get('complete_original_TEST_evidence')
        else 'real_TEST_scope_not_evaluated' if not cohort
        else 'OOD_score_unavailable_or_input_regime_unmatched' if not ood_ready
        else 'out_of_distribution' if oid['OOD'] else 'original_PPC_missing_or_unbound' if not ppc_bound
        else 'scenario_inconsistent_with_measured_features' if not consistent
        else 'outside_or_unknown_measurable_range' if not range_ok
        else 'model_variant_gate_missing_or_failed' if not variant_ok else None)


def _quantity_reason(scene,quantity,common,calibration,cohort,check,research):
    return ('effective_column_correlation_repeat_not_metal_centre_spacing' if scene=='S3' and quantity=='D_nm' else common) or (
        'own_quantity_calibration_missing_or_failed' if not calibration.get('calibrated')
        else 'own_optional_consistency_missing_or_failed' if quantity in ('D_nm','diameter_nm')
            and not cohort.get('optional_consistency',{}).get(scene,{}).get(quantity,{}).get('passed')
        else 'diameter_width_or_highq_feature_gate_failed' if quantity=='diameter_nm' and not check.get('diameter_feature_check_pass')
        else 'D_feature_gate_failed' if quantity=='D_nm' and not check.get('D_feature_check_pass')
        else 'invalid_or_incomplete_original_quantity_interval' if not _valid_interval(research) else None)


def _input_calibration(calibration,oid):
    'OOD for an optional context is distinct from its parameter calibration.'
    regime=oid.get('input_regime')
    if isinstance(regime,str) and regime.startswith('mask_') and regime not in ('mask_0000','mask_1000'):
        return calibration.get('optional_presence_calibration',{}).get(regime,
            dict(calibrated=False,reason='independent_quantity_calibration_for_exact_optional_presence_pending'))
    return calibration


def qualified_D_from_evidence(frame,policy,scene,qualification,runtime_receipt):
    'Reuse the frame release gates on sealed held-PPC pool evidence.'
    part=(frame.get('policies') or {}).get(policy,{})
    assessment=part.get('credibility') or {}
    binding=part.get('PPC_binding') or {}
    source=assessment.get('scenarios',{}).get(scene,{})
    original=part.get('scenarios',{}).get(scene,{})
    research=original.get('conditional_quantities',{}).get('D_nm',{})
    expected=dict(runtime_receipt=runtime_receipt,frame_identity=part.get('frame_identity'),
        joint_sha256=part.get('joint_sha256'),original_seed=frame.get('original_posterior_seed'),
        original_draws_per_applicable_scenario=part.get('original_draws_per_applicable_scenario'))
    bound=bool(binding==expected and expected['frame_identity'] is not None
        and expected['joint_sha256'] is not None
        and assessment.get('runtime_receipt')==runtime_receipt
        and assessment.get('frame_identity')==expected['frame_identity']
        and assessment.get('joint_sha256')==expected['joint_sha256']
        and assessment.get('original_seed')==expected['original_seed'])
    oid=assessment.get('OOD',{})
    ready=bool(oid.get('threshold_available') and oid.get('input_regime_matched')
        and oid.get('score_bound') and isinstance(oid.get('OOD'),(bool,np.bool_)))
    host=part.get('host_category_input')
    cohort=qualification.get('real_cohorts',{}).get(frame['metal']+'/'+host if host in ('Si','oxide_particles','polymer') else 'all',{})
    check=original.get('PPC') or {}
    common=_common_reason(scene,original.get('physical_applicable',False),qualification,
        cohort,oid,ready,bound,bool(check.get('all_measurable_feature_checks_pass')),
        source.get('range',{}),source.get('variants',{}))
    reason=_quantity_reason(scene,'D_nm',common,_input_calibration(qualification['simulation'][scene]['D_nm'],oid),
                            cohort,check,research)
    return dict(data_qualified=reason is None,reason=reason,
        median=research.get('median') if reason is None else None,
        interval68=copy.deepcopy(research.get('interval68')) if reason is None else None,
        interval95=copy.deepcopy(research.get('interval95')) if reason is None else None)


def _s4_feature_release(sampled, ppc, qualification, cohort, oid, ood_ready,
                       ppc_bound, consistent):
    'Independent mean-feature qualification, never S4 island qualification.'
    from .scenario_cached_ppc import S4_FEATURES, clean_feature_summary
    from .scenario_runtime import _source_sha
    from pathlib import Path
    part=sampled['scenarios']['S4']; original=part['original_joint']
    predictive=(ppc or {}).get('scenarios',{}).get('S4',{})
    latent=predictive.get('latent_feature_draws',[]); replicas=predictive.get('draws',[])
    n=part['original_draws'];band=sampled['observed']['AF_band']
    helper_sha=_source_sha(Path(__file__).parent/'scenario_cached_ppc.py')
    expected=dict(runtime_receipt=sampled['receipt'],frame_identity=sampled['frame_identity'],
        joint_sha256={k:_array_sha(v) for k,v in original.items()},
        original_seed=sampled['original_seed'],original_draws=n,AF_band=band,
        extraction_source_sha256=helper_sha)
    bound=bool(ppc_bound and predictive.get('latent_response_binding')==expected
        and (ppc or {}).get('cached_PPC_source_sha256')==helper_sha
        and len(latent)==len(replicas)==n
        and all(a.get('index')==b.get('index')==i and a.get('AF_band')==band
            and a.get('photon_count_statistics_known') is False and a.get('Poisson_draw') is False
            and a.get('expected_photons_sha256')==b.get('expected_photons_sha256')
            for i,(a,b) in enumerate(zip(latent,replicas))))
    summaries=clean_feature_summary(latent,n) if bound else {}
    calibrated_features=qualification.get('acceptance',{}).get('S4_observed_features',{}).get('features',{})


    regime=oid.get('input_regime');none_regime=regime in ('mandatory_only','mask_0000')


    visible_scope=bool(band in range(4) and sampled['observed'].get('support')=='bilateral'
        and len(latent)==n and all(d.get('support')=='bilateral' and not d.get('forward_failed') for d in latent))
    common=('not_applicable' if not part['applicable']
        else 'simulation_TEST_pending' if not qualification.get('complete_original_TEST_evidence')
        else 'real_TEST_scope_not_evaluated' if not cohort
        else 'OOD_score_unavailable_or_input_regime_unmatched' if not ood_ready
        else 'out_of_distribution' if oid['OOD']
        else 'original_mean_feature_pool_missing_or_unbound' if not bound
        else 'outside_original_visible_S4_feature_calibration_scope' if not visible_scope
        else 'scenario_inconsistent_with_measured_features' if not consistent else None)
    conditional={};released={}
    check=predictive.get('PPC') or {}
    for key in S4_FEATURES:
        research=copy.deepcopy(summaries.get(key,dict(median=None,interval68=None,interval95=None,
            original_draws=n,valid_complete_original_joint=False,
            target='latent_mean_response_features',Poisson_draw=False)))
        calibration=calibrated_features.get(key,{})
        calibrated=bool(none_regime and calibration.get('status')=='passed')
        why=common or ('own_feature_calibration_missing_or_failed' if not calibrated
            else 'own_measured_feature_PPC_gate_failed' if not check.get('features',{}).get(key,{}).get('passed')
            else 'invalid_or_incomplete_original_mean_feature_interval'
                if not research.get('valid_complete_original_joint') or not _valid_interval(research) else None)
        passed=why is None
        research.update(calibrated=calibrated,data_qualified=passed,structural_release=False,
            calibration_scope='own_independent_simulation_TEST_mean_feature_none_optional',
            calibration_visibility_scope='clean_bilateral_support_at_original_observed_exit_angle_band',
            current_frame_within_calibration_visibility_scope=visible_scope,
            reason=why or 'passed_own_conditional_mean_feature_gates')
        conditional[key]=research
        released[key]=dict(status='released' if passed else 'withheld',reason=research['reason'],
            median=research['median'] if passed else None,
            interval68=copy.deepcopy(research['interval68']) if passed else None,
            interval95=copy.deepcopy(research['interval95']) if passed else None,
            calibrated=calibrated,data_qualified=passed,structural_release=False,
            target='latent_mean_response_features',units='degree' if key=='Yoneda' else 'nm^-1')
    return conditional,released


def release_frame(sampled, ppc, credibility, qualification, *, host_category=None):
    'Retain all S; no evidence or original-pool failure silently grants release.'
    if (credibility.get('runtime_receipt')!=sampled['receipt']
            or credibility.get('frame_identity')!=sampled['frame_identity']
            or credibility.get('joint_sha256')!=joint_identity(sampled)
            or credibility.get('original_seed')!=sampled['original_seed']):
        raise ValueError('Release credibility must bind exactly the original runtime/frame/joint pool')
    ppc_bound=bool(ppc and ppc.get('bundle_PPC_binding')==dict(runtime_receipt=sampled['receipt'],
        frame_identity=sampled['frame_identity'],joint_sha256=joint_identity(sampled),
        original_seed=sampled['original_seed'],original_draws_per_applicable_scenario=sampled['original_draws_per_applicable_scenario']))
    oid=credibility['OOD'];ood_ready=bool(oid.get('threshold_available') and oid.get('input_regime_matched') and oid.get('score_bound') and isinstance(oid.get('OOD'),(bool,np.bool_)))
    known=host_category in ('Si','oxide_particles','polymer')
    cohort_name=sampled['geometry']['metal']+'/'+host_category if known else 'all'
    cohort=qualification.get('real_cohorts',{}).get(cohort_name,{})
    output={};any_released=False;any_features_released=False
    for scene in SCENARIO_NAMES:
        part=sampled['scenarios'][scene];assessment=credibility['scenarios'][scene]
        check=(ppc or {}).get('scenarios',{}).get(scene,{}).get('PPC')
        consistent=bool(ppc_bound and check and check.get('all_measurable_feature_checks_pass'))
        rng=assessment['range'];variants=assessment['variants']


        common=_common_reason(scene,part['applicable'],qualification,cohort,oid,ood_ready,
                              ppc_bound,consistent,rng,variants)
        quantities={};points={q:None for q in QUANTITIES}
        conditional=copy.deepcopy(part['conditional_quantities'])
        for q in QUANTITIES:
            research=part['conditional_quantities'].get(q,{})
            calibration=_input_calibration(qualification['simulation'][scene][q],oid)
            why=_quantity_reason(scene,q,common,calibration,cohort,check,research)
            passed=why is None
            item=dict(status='released_interval' if passed and q=='height_nm' else 'released' if passed else 'withheld',
                reason='passed_own_quantity_gates' if passed else why,median=research.get('median') if passed and q!='height_nm' else None,
                interval68=copy.deepcopy(research.get('interval68')) if passed else None,
                interval95=copy.deepcopy(research.get('interval95')) if passed else None,
                calibrated=bool(calibration.get('calibrated')),data_qualified=passed,structural_release=passed)
            quantities[q]=item;points[q]=item['median'];any_released |= passed
            if q in conditional:


                conditional[q].update(calibrated=item['calibrated'],
                    data_qualified=passed,structural_release=passed,
                    calibration_reason=calibration.get('reason'),
                    calibration_scope='own_independent_simulation_TEST_quantity')
                conditional[q]['original_runtime_reason']=research.get('reason')
                if item['calibrated']:
                    conditional[q]['reason']=('calibrated conditional interval; measured-frame quantity gates passed'
                        if passed else 'calibrated conditional interval; measured-frame quantity withheld: '+str(why))
        semantics=D_semantics(scene,part['draws'])
        quantities['D_nm'].update(definition=semantics['D_definition'],
            raw_definitions=semantics['raw_D_definitions'],scope=semantics['scope'])
        output[scene]=dict(applicable=part['applicable'],not_applicable_reason=part['not_applicable_reason'],
            consistency='not_applicable' if not part['applicable'] else 'with_data_consistent' if consistent else 'with_data_inconsistent' if ppc_bound else 'not_evaluated',
            conditional_quantities=conditional,quantity_release=quantities,
            calibrated=any(v['calibrated'] for v in quantities.values()),
            calibrated_quantities=[q for q,v in quantities.items() if v['calibrated']],
            calibration_scope='quantity_specific; not a whole-scenario certificate',
            public_quantities=points,OOD=copy.deepcopy(oid),range=copy.deepcopy(rng),variants=copy.deepcopy(variants),
            PPC=copy.deepcopy(check),merged_phase_assumed=scene=='S4',connected_topology_verified=False,
            D_semantics=semantics,
            height_scope='depth support, not island height' if scene=='S3' else 'not applicable' if scene=='S4' else 'conditional island height',
            structural_release=any(v['structural_release'] for v in quantities.values()))
        output[scene].update(real_cohort_acceptance_passed=cohort.get('target_passed',False),
            delivery_scope='accepted_tested_cohort' if cohort.get('target_passed')
                else 'conditional_partial_delivery_cohort_target_not_passed' if cohort
                else 'TEST_scope_not_evaluated',
            C5_acceptance_passed=qualification.get('C5_complete',False),
            whole_v2_acceptance_passed=None,
            whole_v2_acceptance_scope='Final C6 deployment qualification is recorded separately from scientific C5 evidence')
        if scene=='S4':
            conditional_features,feature_release=_s4_feature_release(sampled,ppc,qualification,
                cohort,oid,ood_ready,ppc_bound,consistent)
            available=any(v['data_qualified'] for v in feature_release.values())
            any_features_released |= available
            output[scene].update(conditional_features=conditional_features,
                feature_release=feature_release,conditional_feature_release=available,
                calibrated_features=[k for k,v in feature_release.items() if v['calibrated']],
                calibrated=any(v['calibrated'] for v in feature_release.values()),
                calibration_scope='S4 mean features only; no island quantities or connectivity qualification')
    return dict(schema=SCHEMA,scenarios=output,all_applicable_scenarios_retained=True,
        highest_probability_scenario_selected=False,OOD_score=oid.get('score'),OOD_flag=oid.get('OOD'),
        independent_OOD_validation_performed=oid.get('independent_OOD_validation_performed',False),
        calibrated=any(v['calibrated'] for v in output.values()),
        calibrated_quantities={s:v['calibrated_quantities'] for s,v in output.items()},
        calibrated_features={s:v.get('calibrated_features',[]) for s,v in output.items()},
        calibration_scope='quantity_or_feature_specific; independent simulation calibration differs from measured-frame release',
        reliability='qualified_quantities_available' if any_released
            else 'qualified_conditional_features_available' if any_features_released else 'unreliable',
        conditional_feature_release=any_features_released,structural_release=any_released,
        evidence_binding=copy.deepcopy(qualification.get('evidence_binding')),policy=copy.deepcopy(POLICY))
