"""Measured curve -> component candidates -> distinct fitted parameter candidates."""
import argparse,json,sys,time
from pathlib import Path
def main(a):
    sys.path.insert(0,str(a.bundle/'source'))
    import numpy as np
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for gpu in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(gpu,True)
    from predict_1d import Predictor,preprocess,summarize
    from component_prior import resolve_components
    from flow_codec import COMBOS
    from gpu_precision import initial_candidates,refine_candidates,KEYS
    raw=np.load(a.input,allow_pickle=False);data=preprocess(raw['q'],raw['observed'],raw['sigma'],raw['mask'] if 'mask' in raw else None)
    predictor=Predictor(a.bundle,data);condition=None;combo_ids=None
    if a.components:
        c,types=resolve_components(a.components,COMBOS);combo_ids=np.full(len(data['q']),c,'int32');condition=dict(combination_id=c,component_types=types,semantics='Complete supplied multiset including repeated types')
    start=time.perf_counter();cand=initial_candidates(predictor,data,combo_ids,12);cand['components_condition']=condition;out,records=refine_candidates(data,cand,80);out['seconds']=time.perf_counter()-start
    result=summarize(data,out,max_solutions=12,threshold=a.threshold,max_parameters=None)
    result.update(output_schema_version='v5-precision-r1',inference_method='Original r3 six neural starts after20neural passes, then up to80trust-region evaluations per start; this includes numerical refinement',component_selection='Original classifier top12, all retained; conditional mode bypasses classification',probability_status='Candidate scores are not calibrated posterior probabilities; finite parameter candidates do not exhaust all solutions',seconds=out['seconds'])
    # With12 selected classes, ordering the displayed classes does not drop any.
    for row in result['curves']:row['coverage']=dict(applicable=condition is None,guarantee=False,reason='Precision-r1 empirical component coverage is documented separately; no per-curve guarantee')
    result['calibration_limits']='Noise-aware flags inherited from r3; empirical reliability for precision-r1 must be read from its own validation report.'
    a.output.mkdir(parents=True,exist_ok=False);(a.output/'solutions.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');(a.output/'solver_records.json').write_text(json.dumps(records,indent=2))
    np.savez_compressed(a.output/'forward_candidates.npz',**data,**{k:out[k] for k in (*KEYS,'curves','classifier_nlp','theta','gate_pattern')});print(json.dumps(dict(curves=len(data['q']),seconds=out['seconds'],output=str(a.output))))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bundle','input','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--components',nargs='+');p.add_argument('--threshold',type=float,choices=[.03,.05],default=.05);main(p.parse_args())
