"""Optional fast conditional prediction, gated by the bundle's validation record."""
import argparse,json,sys,time
from pathlib import Path


def main(a):
    sys.path.insert(0,str(a.bundle/'source'))
    import numpy as np
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu,True)
    from predict_1d import Predictor,preprocess,summarize
    from component_prior import resolve_components
    from flow_codec import COMBOS
    from gpu_precision import initial_candidates,KEYS
    from adaptive_precision import refine_adaptive
    config=json.loads((a.bundle/'ADAPTIVE_CONFIRMATION.json').read_text())
    if not config['eligible_for_optional_fast_mode']:
        raise ValueError('This bundle has no validated optional fast mode')
    selected=config['policy']
    raw=np.load(a.input,allow_pickle=False)
    data=preprocess(raw['q'],raw['observed'],raw['sigma'],raw['mask'] if 'mask' in raw else None)
    c,types=resolve_components(a.components,COMBOS)
    condition=dict(combination_id=c,component_types=types,
                   semantics='Complete supplied multiset including repeated types')
    predictor=Predictor(a.bundle,data);start=time.perf_counter()
    cand=initial_candidates(predictor,data,np.full(len(data['q']),c,'int32'))
    cand['components_condition']=condition
    out,records=refine_adaptive(data,cand,selected);out['seconds']=time.perf_counter()-start
    result=summarize(data,out,max_solutions=12,threshold=a.threshold,max_parameters=None)
    result.update(output_schema_version='v5-precision-adaptive-r1',seconds=out['seconds'],
                  inference_method='Six original neural candidates after20learned passes; observed/sigma early stopping during trust-region numerical refinement',
                  adaptive_policy=selected,
                  stopping_target='Fixed conservative strict-match rule; --threshold controls reporting, not the frozen stopping policy',
                  validated_scope='Supplied complete component multiset, synthetic V5 curves; no real-data or exhaustive posterior guarantee',
                  calibration_limits='Noise-aware flags are empirical; adaptive-specific counts are in ADAPTIVE_CONFIRMATION.json. Supplied sigma must be justified for real measurements.',
                  probability_status='Uncalibrated finite candidates, not posterior probabilities')
    result['adaptive_counts']={reason:sum(r['stop_reason']==reason for r in records)
                               for reason in ('initial_quality_pass','quality_early_stop','solver_finished')}
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'solutions.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (a.output/'solver_records.json').write_text(json.dumps(records,indent=2))
    np.savez_compressed(a.output/'forward_candidates.npz',**data,
                        **{k:out[k] for k in (*KEYS,'curves','classifier_nlp','theta','gate_pattern')})
    print(json.dumps(dict(curves=len(data['q']),seconds=out['seconds'],policy=selected,output=str(a.output))))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bundle','input','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--components',nargs='+',required=True)
    p.add_argument('--threshold',type=float,choices=[.03,.05],default=.05)
    main(p.parse_args())
