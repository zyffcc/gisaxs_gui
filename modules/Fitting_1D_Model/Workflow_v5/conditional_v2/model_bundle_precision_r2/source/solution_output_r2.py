"""Distinct physical-combination candidates with empirically checked quality flags."""
import numpy as np
from noise_quality import quality_metrics, combo_order
from solution_output import summarize as physical_summary


def fits(metrics, rule):
    return bool(rule is not None and metrics['band32_excess'] <= rule['rms']
        and metrics['band128_max_excess'] <= rule['local']
        and metrics['band32_noise'] <= rule['max_noise'])


def summarize(data, cand, combos, max_solutions=8, threshold=.03, distance=.03):
    if not isinstance(max_solutions,(int,np.integer)) or not 1<=max_solutions<=len(combos):
        raise ValueError('max_solutions must be an integer between 1 and the number of combinations')
    if threshold not in (.03,.05):
        raise ValueError('The validated clean-signal quality targets are 0.03 and 0.05')
    if 'classifier_nlp' not in cand or 'output_policy' not in cand:
        raise ValueError('r2 output requires classifier scores and the frozen output policy')
    policy=cand['output_policy'];rows=[];base=None
    target='strict' if threshold==.03 else 'good'
    for i in range(len(data['q'])):
        qm=quality_metrics(cand['curves'][i],data['observed'][i],data['sigma'][i],data['mask'][i])
        order=combo_order(cand['combos'][i],qm['observed_error'],cand['classifier_nlp'][i],policy['beta'])
        chosen=order[:max_solutions];best=int(np.argmin(qm['observed_error']))
        def materialize(j):
            nonlocal base
            one={k:np.asarray(cand[k])[i:i+1,j:j+1] for k in ('params','weights','globals','d','res','combos','curves')}
            sub={k:np.asarray(v)[i:i+1] for k,v in data.items()}
            physical=physical_summary(sub,one,combos,1,threshold,distance)
            if base is None:base=physical
            solution=physical['curves'][0]['solutions'][0]
            metrics={k:float(v[j]) for k,v in qm.items()}
            flags={name:fits(metrics,rule) for name,rule in policy['rules'].items()}
            status=('strict_match' if flags['strict'] else 'good_match' if flags['good']
                    else 'noise_limited' if metrics['band32_noise']>.03 else 'not_confirmed')
            solution.update(candidate_index=int(j),combination_id=int(cand['combos'][i,j]),
                ranking_score=float(qm['observed_error'][j]**2+policy['beta']*cand['classifier_nlp'][i,j]),
                passes_fit_threshold=flags[target],fit_quality=status,
                fit_target_clean_logrmse=threshold,quality_metrics=metrics,
                quality_interpretation='Empirical synthetic-validation flag, not a measurement of unknown clean error or proof of correct components')
            return solution
        size=policy['coverage_calibration']['rank_quantile']
        rows.append(dict(curve_index=i,raw_candidate_count=len(qm['observed_error']),
            distinct_available_combinations=len(order),returned_distinct_combinations=len(chosen),
            solutions=[materialize(int(j)) for j in chosen],fit_best_reference=materialize(best),
            best_observed_logrmse=float(qm['observed_error'][best]),
            coverage=dict(empirical_target=.9,calibrated_set_size=size,
                enough_candidates_for_calibrated_size=len(chosen)>=size,
                per_curve_probability=None,guarantee=False,
                note='Fixed rank size calibrated on synthetic validation. Fewer candidates remove the nominal coverage target; distribution shift can invalidate it.'),
            selection_note='One lowest-observed-error parameter solution per original combination; no rejection by fit quality or component weight. Fit-best reference is separate.'))
    base.update(output_schema_version='v5-output-r2',curves=rows,
        fit_reference='Observed residuals and supplied absolute sigma; clean target is inferred empirically, not observed',
        fit_threshold=threshold,fit_threshold_definition='Empirical noise-aware rule targeting clean natural-log RMSE; not an observed-RMSE cutoff',
        quality_rules=policy['rules'],
        probability_status='Classifier log scores are uncalibrated ranking features; no posterior probabilities',
        deduplication=dict(version='r2-combination-groups',claim='Exactly one representative per original type multiset; equivalent parameter decompositions can remain across different multisets'),
        ranking=dict(beta=policy['beta'],score='observed_logrmse^2 + beta * classifier_negative_log_softmax',
            fit_best_reference='Always preserved separately; may repeat one displayed candidate'),
        seconds=float(cand['seconds']) if 'seconds' in cand else None,
        calibration_limits=policy['limitations'])
    return base
