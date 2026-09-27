"""Export exact Dense/swish inference weights; check TF parity before shipping."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[name]='1'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.blue_curve_distillation import OUT,build_model,forward


def main():
    bundle=ROOT/'modules/Fitting_1D_Model/Workflow_v5/stable_blue_rc_v1'
    model=build_model();model.load_weights(str(bundle/'curve_best.weights.h5'))
    weights={}
    for i,layer in enumerate(model.layers):
        w,b=layer.get_weights();weights[f'kernel_{i}']=w;weights[f'bias_{i}']=b
    dataset=np.load(OUT/'dataset.npz');start=int(dataset['ntrain'])+int(dataset['nval'])
    scale=np.load(bundle/'feature_scaling.npz')
    x=(dataset['x'][start:]-scale['mean'])/scale['scale']
    expected=model(x,training=False).numpy();actual=x.copy()
    for i in range(4):
        actual=actual@weights[f'kernel_{i}']+weights[f'bias_{i}']
        actual=actual*expit(actual) if i<3 else expit(actual)
    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=2e-6)
    q=np.linspace(.001,4.3,500);curve_differences=[]
    for i in np.linspace(0,len(x)-1,16,dtype=int):
        a,b=forward(q,actual[i]),forward(q,expected[i])
        curve_differences.append(float(np.sqrt(np.mean(np.log(a/b)**2))))
    assert max(curve_differences)<1e-5
    np.savez_compressed(bundle/'network.npz',**weights)
    manifest=json.loads((bundle/'MANIFEST.json').read_text())
    manifest['inference_backend']='numpy_dense_swish_v1'
    manifest['files']['network.npz']=hashlib.sha256((bundle/'network.npz').read_bytes()).hexdigest()
    (bundle/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    report=dict(rows=len(x),max_latent_absolute_difference=float(abs(actual-expected).max()),
                audited_curves=len(curve_differences),max_curve_logrmse_difference=max(curve_differences),
                passed=True,note='Exact architecture/weights conversion, no retraining or test-based model selection.')
    out=ROOT/'validation/stable_blue_20260922';out.mkdir(parents=True,exist_ok=True)
    (out/'numpy_conversion.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
