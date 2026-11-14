'Observation-conditioned routing for staged GISAXS fits.'
from __future__ import annotations
import numpy as np

DEFAULT_POLICY=dict(q_feature_window_nm_inv=[.05,1.75],q_bin_width_nm_inv=.025,
    history_points=3,Yoneda_platform_range_deg=.01,q_motion_min_relative=.05,
    q_branch_max_upward_factor=1.5,q_branch_max_downward_factor=2.,
    exit_extrapolation_max_previous_gaps=3.,jump_factor_D_R=1.5)
STAGES={0:'pre_nucleation_candidate',1:'separate_islands_candidate',
        2:'coalescence_candidate',3:'continuous_film_candidate'}

def finite(value):
    return value is not None and np.isfinite(value)

def classify_sequence(items,policy=None):
    'Route all supplied windows in input time order without fitting a peak.'
    p={**DEFAULT_POLICY,**(policy or {})};n=int(p['history_points'])
    if n<3:raise ValueError('At least3 history observations required')
    times=[x.get('metadata',{}).get('time_s') for x in items]
    known=True
    qmin=float(p['q_feature_window_nm_inv'][0]);dq=float(p['q_bin_width_nm_inv'])
    rows=[];track=[];branch=0;ever_visible=False;previous_stage=None;exit_history=False
    for i,item in enumerate(items):
        was_known=known;known=bool(known and finite(times[i]))
        if known and i and times[i]<=times[i-1]:raise ValueError('Strictly increasing time/input order required')
        if was_known and not known:


            for point in track:point['t']=float(point['index'])
        ti=float(times[i]) if known else float(i)
        o=item['observations'];status=o.get('feature_status','no_resolved_peak')
        q=o.get('q_star_nm_inv');visible=status in ('bilateral','one_side') and finite(q) and q>0
        evidence=[];conflicts=[];new_branch=False
        if visible:
            ever_visible=True


            exit_history=False
            if track:
                ratio=q/track[-1]['q']
                band_changed=o.get('feature_AF_band')!=track[-1]['AF_band']
                if ratio>p['q_branch_max_upward_factor'] or ratio<1/p['q_branch_max_downward_factor'] or band_changed:
                    conflicts.append('new_feature_branch_not_silently_substituted_for_previous_peak')
                    if band_changed:conflicts.append('empirical_peak_selected_from_different_AF_band')
                    branch+=1;track=[];new_branch=True;exit_history=False
            track.append(dict(index=i,t=ti,q=float(q),status=status,
                fwhm=o.get('FWHM_nm_inv'),q_se=o.get('q_star_fit_se_nm_inv'),AF_band=o.get('feature_AF_band')))
        recent=track[-n:];moving=False;slope=None;motion_threshold=None
        if len(recent)>=n:
            qs=np.array([v['q'] for v in recent]);ts=np.array([v['t'] for v in recent])
            slope=float(np.polyfit(ts-ts[0],qs,1)[0])
            ses=[recent[0]['q_se'],recent[-1]['q_se']]
            se_threshold=2*float(np.hypot(*ses)) if all(finite(x) for x in ses) else 0.
            motion_threshold=max(dq,p['q_motion_min_relative']*recent[0]['q'],se_threshold)
            moving=bool(slope<0 and qs[0]-qs[-1]>motion_threshold)
            if moving:evidence.append('same_feature_branch_moves_to_lower_q')
        ys=[items[j]['observations'].get('Yoneda_deg') for j in range(max(0,i-n+1),i+1)]
        platform=len(ys)==n and all(finite(y) for y in ys) and np.ptp(ys)<=p['Yoneda_platform_range_deg']
        if platform:evidence.append('last3_observed_ridge_positions_within_declared_resolution')
        extrapolated_q=None;exit_candidate=False;low_flank_near_edge=False
        current_history=visible or bool(known and len(track)>1 and
            ti-track[-1]['t']<=p['exit_extrapolation_max_previous_gaps']*(track[-1]['t']-track[-2]['t']))
        if not visible and moving and len(recent)>=n and finite(recent[-1]['fwhm']):
            last=recent[-1];gap=last['t']-recent[-2]['t'];elapsed=float(ti-last['t'])
            low_flank_near_edge=last['q']-last['fwhm']/2<=qmin+dq
            if known and gap>0 and elapsed<=p['exit_extrapolation_max_previous_gaps']*gap:
                extrapolated_q=float(last['q']+slope*elapsed)
                exit_candidate=bool(low_flank_near_edge and extrapolated_q<=qmin)
                if exit_candidate:
                    exit_history=True;evidence.append('tracked_peak_low_flank_clipped_then_conditional_centre_extrapolation_exits')
        if exit_history and not visible:evidence.append('previous_conditional_exit_branch_still_not_observed')
        stage=None;reason='insufficient_or_conflicting_stage_evidence'
        if visible and status=='bilateral' and not new_branch:
            stage=1;reason='bilateral_empirical_feature';evidence.append('measured_bilateral_feature')
        elif not visible and exit_history and platform:
            stage=3;reason='ridge_platform_and_conditional_lowq_peak_exit'
        elif visible and status=='one_side' and moving and not new_branch:
            stage=2;reason='one_side_feature_with_recorded_lowq_motion'
        elif not visible and previous_stage==2 and moving and current_history:
            stage=2;reason='moving_branch_temporarily_unresolved_not_yet_platform_exit'
            evidence.append('stage2_continuation_is_conditional_not_new_spacing_measurement')
        elif not visible and not ever_visible and platform:
            stage=0;reason='no_feature_and_stationary_ridge_no_observed_peak_history'
            conflicts.append('does_not_establish_zero_metal_or_true_pre_nucleation')
        if not known:
            conflicts.append('header_time_missing_trends_use_input_order_not_growth_rate')
        if status=='one_side':conflicts.append('one_side_visibility_does_not_imply_peak_below_qmin')
        if stage==3:conflicts.append('conditional_extrapolated_exit_and_platform_do_not_prove_continuous_film')
        if o.get('Yoneda_status')=='observed_ridge_support_gap':conflicts.append('ridge_has_support_gap')
        routes={0:'features_only',1:'island_D_R',2:'highq_R_only',3:'film_or_network_conditional'}
        route=routes[stage] if stage is not None else 'unresolved_features_and_conditional_highq_only'

        lower=None;lower_reason='no_justified_spacing_lower_bound'
        if stage==2 and track and moving:
            last=track[-1];margin=max(dq,2*last['q_se']) if finite(last['q_se']) else dq
            lower=float(2*np.pi/(last['q']+margin))
            lower_reason='conditional_Dcorr_lower_if_same_peak_continues_nonincreasing;one_qbin_margin_not_calibrated95'
        elif stage==3:
            lower=float(2*np.pi/qmin)
            lower_reason='conditional_Dcorr_lower_if_extrapolated_peak_really_below_qmin;not_measured_bound'
        row=dict(key=item['key'],time_s=times[i],input_index=i,time_known=known,
            stage_id=stage,stage_label=STAGES.get(stage,'stage_unresolved'),
            stage_scope='observation-conditioned candidate, not certified morphology/connectivity',
            stage_reason=reason,stage_evidence=evidence,stage_conflicts=conflicts,
            feature_status=status,feature_branch_id=branch,branch_q_slope_nm_inv_s=slope if known else None,
            q_motion_threshold_nm_inv=motion_threshold,ridge_platform=bool(platform),
            peak_exit_status='conditional_extrapolation_only' if exit_history else 'not_established',
            extrapolated_q_nm_inv=extrapolated_q,last_peak_low_flank_near_q_edge=bool(low_flank_near_edge),
            route=route,D_fit_enabled=stage==1,R_fit_module='island' if stage==1 else 'highq_or_change' if stage in (2,None) else 'none',
            D_model_lower_nm=None,D_model_lower_reason='Feature2pi/q is not generally the paracrystal/nearest-neighbour D',
            Dcorr_conditional_lower_nm=lower,Dcorr_lower_reason=lower_reason,
            checks_passed=['original_observations_retained','declared_stage_policy','no_new_physics'],
            checks_not_passed=['stage_morphology_truth','new_module_recovery','new_independent_TEST'])
        rows.append(row);previous_stage=stage
    return dict(schema='v3_staged_routing_1',original_windows=len(items),time_known=known,
        policy=p,rows=rows,scope='routing only; no new fit, parameter recovery or film certification')
