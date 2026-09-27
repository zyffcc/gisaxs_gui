"""Two levels: component hypotheses, then all numerically distinct parameter heads."""
import numpy as np
from noise_quality import quality_metrics, combo_order
from solution_output import summarize as physical_summary, component_state, state_distance
from solution_output_r2 import fits
from component_prior import TYPE_NAMES


def select_parameter_heads(cand, i, combos, indices, errors, mask):
    """Do not merge distinct parameters just because their forward curves agree."""
    states={};kept=[];duplicates={};valid=np.asarray(mask,bool)
    for j in sorted(map(int,indices),key=lambda j:(float(errors[j]),j)):
        s=component_state(cand,i,j,combos);duplicate=None
        for k in kept:
            # Both physical equivalence and forward equivalence are required.
            if state_distance(s,states[k])<=1e-7:
                delta=np.max(np.abs(np.log(np.maximum(cand['curves'][i,j,valid],1e-30))-
                                    np.log(np.maximum(cand['curves'][i,k,valid],1e-30))))
                if delta<=1e-5:duplicate=k;break
        if duplicate is None:kept.append(j);states[j]=s
        else:duplicates[j]=duplicate
    return kept,duplicates


def summarize(data,cand,combos,max_solutions=8,threshold=.03,distance=.03,max_parameters=None):
    if not isinstance(max_solutions,(int,np.integer)) or not 1<=max_solutions<=len(combos):
        raise ValueError('max_solutions counts distinct component combinations')
    if max_parameters is not None and (not isinstance(max_parameters,(int,np.integer)) or max_parameters<1):
        raise ValueError('max_parameters must be a positive integer or None for all available heads')
    if threshold not in (.03,.05):raise ValueError('Validated signal-quality targets are .03 and .05')
    policy=cand['output_policy'];condition=cand.get('components_condition');target='strict' if threshold==.03 else 'good'
    if condition is not None and not np.all(cand['combos']==condition['combination_id']):
        raise ValueError('A conditional prediction contains candidates from another component combination')
    rows=[];base=None
    for i in range(len(data['q'])):
        qm=quality_metrics(cand['curves'][i],data['observed'][i],data['sigma'][i],data['mask'][i])
        order=combo_order(cand['combos'][i],qm['observed_error'],cand['classifier_nlp'][i],policy['beta'])
        chosen=order[:max_solutions];best=int(qm['observed_error'].argmin())
        def materialize(j):
            nonlocal base
            one={k:np.asarray(cand[k])[i:i+1,j:j+1] for k in ('params','weights','globals','d','res','combos','curves')}
            sub={k:np.asarray(v)[i:i+1] for k,v in data.items()}
            physical=physical_summary(sub,one,combos,1,threshold,distance)
            if base is None:base=physical
            solution=physical['curves'][0]['solutions'][0];metrics={k:float(v[j]) for k,v in qm.items()}
            flags={name:fits(metrics,rule) for name,rule in policy['rules'].items()}
            status=('strict_match' if flags['strict'] else 'good_match' if flags['good'] else
                    'noise_limited' if metrics['band32_noise']>.03 else 'not_confirmed')
            group_indices=np.flatnonzero(cand['combos'][i]==cand['combos'][i,j])
            solution.update(candidate_index=int(j),parameter_head_index=int(np.flatnonzero(group_indices==j)[0]),
                combination_id=int(cand['combos'][i,j]),passes_fit_threshold=flags[target],fit_quality=status,
                fit_target_clean_logrmse=threshold,quality_metrics=metrics,parameter_probability=None)
            return solution
        groups=[]
        for rank,representative in enumerate(chosen,1):
            c=int(cand['combos'][i,representative]);indices=np.flatnonzero(cand['combos'][i]==c)
            kept,duplicates=select_parameter_heads(cand,i,combos,indices,qm['observed_error'],data['mask'][i])
            displayed=kept if max_parameters is None else kept[:max_parameters]
            types=[int(t) for t in combos[c] if t>0]
            groups.append(dict(combination_id=c,combination_rank=None if condition else rank,
                component_types=types,component_names=[TYPE_NAMES[t] for t in types],
                combination_ranking_score=None if condition else float(qm['observed_error'][representative]**2+policy['beta']*cand['classifier_nlp'][i,representative]),
                best_observed_logrmse=float(qm['observed_error'][representative]),
                raw_parameter_head_count=len(indices),distinct_parameter_candidate_count=len(kept),
                returned_parameter_count=len(displayed),parameter_solutions_truncated=len(displayed)<len(kept),
                duplicate_heads={str(j):int(k) for j,k in duplicates.items()},
                parameter_solutions=[materialize(j) for j in displayed],
                complete_parameter_space_coverage=False))
        size=policy['coverage_calibration']['rank_quantile']
        coverage=({'applicable':False,'reason':'Complete component multiset supplied by user; classification is bypassed','guarantee':False}
            if condition else dict(applicable=True,empirical_target=.9,calibrated_set_size=size,
                enough_candidates_for_calibrated_size=len(chosen)>=size,per_curve_probability=None,guarantee=False))
        rows.append(dict(curve_index=i,raw_candidate_count=len(qm['observed_error']),distinct_available_combinations=len(order),
            returned_distinct_combinations=len(groups),combination_candidates=groups,fit_best_reference=materialize(best),
            best_observed_logrmse=float(qm['observed_error'][best]),coverage=coverage))
    base.update(output_schema_version='v5-output-r3',curves=rows,prediction_mode='conditional_parameters' if condition else 'components_and_parameters',
        components_condition=condition,fit_threshold=threshold,
        fit_reference='Observed curve with absolute sigma; clean quality target is inferred empirically',
        fit_threshold_definition='Same noise-aware rules as r2, not a fixed observed-logRMSE cutoff',quality_rules=policy['rules'],
        probability_status='Neither classification scores nor parameter heads are calibrated posterior probabilities',
        deduplication=dict(scope='within each component multiset only',parameter_state_RMS_tolerance=1e-7,max_logcurve_difference=1e-5,
            slot_permutations='same-type/gate minimum matching; inactive parameters ignored',
            claim='Only numerical physical equivalence; curve similarity alone NEVER removes distinct parameters'),
        candidate_coverage='All available distinct neural heads by default; finite candidates, not all posterior modes or all continuous solutions',
        parameter_ranking='Observed forward error within each component multiset; no probability from head counts',
        seconds=float(cand['seconds']) if 'seconds' in cand else None,
        calibration_limits='r2 quality thresholds reused. All-head quality reliability must be assessed separately from r2 representative-only statistics. No true independent parameter-mode coverage benchmark yet.')
    return base
