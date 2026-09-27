"""Run from the bundle directory with PYTHONPATH restricted to bundle/source."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import tensorflow as tf

def main(a):
    root=a.bundle.resolve()
    import predict_1d, benchmark_model, correction_model, physics_v5, input_representation
    for module in [predict_1d,benchmark_model,correction_model,physics_v5,input_representation]:
        assert Path(module.__file__).resolve().is_relative_to(root/'source'),module.__file__
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device,True)
    raw=np.load(root/'example_input.npz',allow_pickle=False)
    data=predict_1d.preprocess(raw['q'],raw['observed'],raw['sigma'],raw['mask'])
    def forbidden(*args,**kwargs): raise AssertionError('Inference must not use GradientTape')
    tf.GradientTape=forbidden
    model=predict_1d.Predictor(root,data)
    candidate=model.predict(data)
    result=predict_1d.summarize(data,candidate)
    assert np.isfinite(candidate['curves']).all()
    for row in result['curves']:
        assert row['solutions'] and np.isfinite(row['best_observed_logrmse'])
    alternatives=[v for v in model.config.get('validated_budgets',[]) if
        (v['topk'],v['passes'])!=(model.config['topk'],model.config['passes'])]
    override_checked=None
    if alternatives:
        budget=min(alternatives,key=lambda v:v['topk']*v['passes'])
        alternate=predict_1d.Predictor(root,data,**budget)
        prediction=alternate.predict(data)
        assert np.isfinite(prediction['curves']).all()
        assert prediction['curves'].shape[1]==budget['topk']*alternate.base.hypotheses
        override_checked=budget
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'solutions.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (a.output/'VERIFIED.json').write_text(json.dumps({'verified':True,'no_inference_gradient_tape':True,
        'bundle_source_imports':True,'curves':len(data['q']),'seconds':candidate['seconds'],
        'validated_budget_override_checked':override_checked,
        'python':sys.version,'tensorflow':tf.__version__},indent=2))
    print('INDEPENDENT_BUNDLE_VERIFIED',len(data['q']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);main(p.parse_args())
