"""Run from the copied bundle, with no workspace import path."""
import json,time,sys,argparse,importlib.metadata
from pathlib import Path
import scipy.special,scipy.optimize
import numpy as np
from conditional_fast_predict_v2 import ConditionalFastPredictor
import scaled_condition_common,conditional_fast_predict_v2,active_component_physics,numpy_candidate_forward,solution_output
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);args=ap.parse_args()
    assert scaled_condition_common.R==HERE
    locations={m.__name__:str(Path(m.__file__).resolve()) for m in (scaled_condition_common,conditional_fast_predict_v2,active_component_physics,numpy_candidate_forward,solution_output)}
    assert all(Path(v).is_relative_to(HERE) for v in locations.values()),locations
    d=dict(np.load(HERE/'example_input.npz'));condition=json.loads((HERE/'example_conditions.json').read_text());ref=dict(np.load(HERE/'example_reference_candidates.npz'))
    model=ConditionalFastPredictor();reports=[]
    for numerical in (False,True):
        tick=time.perf_counter();result,c=model.predict(d['q'],d['observed'],d['sigma'],condition['components'],condition['sigma_res'],condition['nu_res'],mask=d['mask'],numerical=numerical);wall=time.perf_counter()-tick
        n=30 if numerical else 18;assert c['combos'].shape==(1,n)
        valid=c['mask'][0]>0;a=c['curves'][0][:,valid];b=ref['curves'][0,:n][:,valid]
        diff=float(np.max(np.sqrt(np.mean(np.log(a/b)**2,1))));assert diff<(1e-3 if numerical else 1e-4),diff
        expected=float(np.sqrt(np.mean(np.log(a/c['observed'][0,valid])**2,1)).min())
        assert abs(result['curves'][0]['best_observed_logrmse']-expected)<1e-12
        json.dumps(result,allow_nan=False)
        reports.append(dict(numerical=numerical,wall_seconds=wall,curve_max_difference=diff))
    packages={}
    for name in ('tensorflow','tensorflow-intel','numpy','scipy','h5py'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    evidence=dict(passed=True,scope='One TRAIN example, actual standalone copied package, both modes. No project imports or fresh test.',module_locations=locations,python=sys.version,packages=packages,reports=reports)
    args.result.write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence))
