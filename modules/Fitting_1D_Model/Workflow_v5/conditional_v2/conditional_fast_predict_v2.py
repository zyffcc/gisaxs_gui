"""Experimental reusable known-component prediction with optional fixed four-step GN."""
import os,time,argparse,json,hashlib
from pathlib import Path
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','8');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import scipy.special,scipy.optimize
from scaled_condition_common import R,B,KEYS,tf_setup,build_batch,subset,raw_update,decode_update
import numpy as np
tf=tf_setup()
from predict_1d import preprocess
from portable_predictor import PortablePredictor
from portable_corrector_weights import load_corrector_exact
from conditional_resolution import ConditionalCorrector,physical_conditions,predict,constrain_candidates,physics,feedback
from correction_model import features
from diagnose_v5 import add_curves
from feasible_distance_corrector import FeasibleDistanceCorrector
from flow_codec import encode,decode,COMBOS
from component_prior import resolve_components
from active_autodiff_gn import correct
from numpy_candidate_forward import direct_forward
from solution_output import summarize

class BoundedCorrector(FeasibleDistanceCorrector):
    def call(self,b,training=False):return decode_update(b,tf.tanh(raw_update(self,b,training)))

def canonical(c,cv=None):
    n,h=c['combos'].shape;w=c['weights'].reshape(-1,4);w=np.exp(w-w.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
    e=encode(COMBOS[c['combos'].ravel()],c['params'].reshape(-1,4,6),w,c['globals'].reshape(-1,4),c['d'].reshape(-1,4),c['res'].ravel())
    out={k:v.reshape(n,h,*v.shape[1:]) for k,v in decode(e['theta'],e['combo'],e['gate']).items()}
    if cv is not None:out['globals'][:,:,1:3]=cv[:,None]
    return out

class ConditionalFastPredictor:
    """Keep this object alive to reuse loaded weights and compiled physics kernels."""
    def __init__(self):
        manifest=json.loads((R/'conditional_fast_manifest_v2.json').read_text())
        for name,h in manifest['files'].items():
            assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h,name
        self.manifest=manifest;self.proposal=None;self.condition=None;self.update=None
        self.gn_warmed=False

    def predict(self,q,observed,sigma,components,sigma_res,nu_res,*,numerical=True,mask=None):
        """One curve, known full component multiset. Physical q and sigma_res in nm^-1."""
        data=preprocess(q,observed,sigma,mask)
        if len(data['q'])!=1:raise ValueError('This public entry accepts one curve at a time')
        cid,types=resolve_components(components,COMBOS);values,condition_mask=physical_conditions(sigma_res,nu_res)
        if not np.all(condition_mask==1):raise ValueError('Both sigma_res and nu_res must be supplied for this validated scope')
        cv=values[None];cm=condition_mask[None]
        # Internal feature builder requires these keys; clean is never consumed at inference.
        data['globals']=np.zeros((1,4),'float32');data['globals'][:,1:3]=cv
        data['clean']=np.ones_like(data['observed'])
        measured={k:data[k] for k in ('q','mask','observed','sigma','curve','context')}
        init_tick=time.perf_counter()
        if self.proposal is None:self.proposal=PortablePredictor(B,measured)
        loading_seconds=time.perf_counter()-init_tick
        from gpu_precision import initial_candidates
        tick=time.perf_counter();initial=canonical(initial_candidates(self.proposal,measured,np.array([cid],'int32')))
        initial_seconds=time.perf_counter()-tick
        if self.condition is None:
            start=time.perf_counter();warm=features(data,add_curves(data,constrain_candidates(initial,cv,cm)))
            warm.update(condition_values=np.repeat(cv,6,0),condition_mask=np.repeat(cm,6,0))
            self.condition=ConditionalCorrector(True);self.condition({k:tf.constant(v[:1]) for k,v in warm.items()})
            load_corrector_exact(self.condition,R/self.manifest['condition_weights']);loading_seconds+=time.perf_counter()-start
        tick=time.perf_counter();anchor=canonical(predict(self.condition,data,initial,cv,cm,4),cv)
        b=build_batch(data,anchor);condition_seconds=time.perf_counter()-tick
        if self.update is None:
            start=time.perf_counter();self.update=BoundedCorrector(True);self.update(b)
            load_corrector_exact(self.update,R/self.manifest['update_weights']);loading_seconds+=time.perf_counter()-start
        tick=time.perf_counter();banks=[anchor];stage=['anchor']*6
        for step in range(1,5):
            pred=self.update(b,training=False);y=physics(b,*pred)
            if step in (1,4):
                c={k:anchor[k].copy() for k in ('combos','d','res')}
                c.update({k:v.numpy()[None] for k,v in zip(('params','weights','globals'),pred)})
                banks.append(c);stage.extend([f'neural_{step}']*6)
            if step<4:b.update(feedback(b,*pred,y))
        neural_seconds=time.perf_counter()-tick;numerical_meta=None;numerical_total=0.
        if numerical:
            start=time.perf_counter();snapshots,numerical_meta=correct(data,banks[-1],0,steps=4,epsilon=.002,dampings=(.001,.01,.1,1.),chunk=16,warmup=not self.gn_warmed)
            self.gn_warmed=True;numerical_total=time.perf_counter()-start
            for step in (1,4):banks.append(snapshots[step]);stage.extend([f'numerical_{step}']*6)
        candidates={k:np.concatenate([c[k] for c in banks],1) for k in KEYS}
        assert np.all(candidates['combos']==cid)
        assert np.array_equal(candidates['globals'][:,:,1:3],np.repeat(cv[:,None],len(stage),1))
        start=time.perf_counter();valid=data['mask'][0]>0;nheads=len(stage);curves=np.ones((1,nheads,1000),'float64')
        for head in range(nheads):curves[0,head,valid]=direct_forward(candidates,0,head,data['q'][0,valid].astype('float64'))
        candidates['curves']=curves;report_data=dict(data);report_data['observed']=data['observed'].astype('float64');result=summarize(report_data,candidates,COMBOS,max_solutions=nheads)
        # The benchmark .05 threshold is against clean synthetic truth, unavailable on real data.
        result.pop('fit_threshold',None)
        for row in result['curves']:
            for solution in row['solutions']:
                solution.pop('passes_fit_threshold',None);head=solution['candidate_index'];solution['candidate_stage']=stage[head]
                solution['sigma_normalized_residual_rms']=float(np.sqrt(np.mean(((curves[0,head,valid]-data['observed'][0,valid])/data['sigma'][0,valid])**2)))
        processing_seconds=time.perf_counter()-start
        result.update(inference_method='Neural candidates plus fixed four-step Gauss-Newton with automatic derivatives and active-shape physics' if numerical else 'Fixed neural prediction only',
            numerical_correction=bool(numerical),fixed_parameters=dict(sigma_Res=float(sigma_res),nu_Res=float(nu_res)),
            components_condition=dict(type_ids=types,semantics='Complete supplied multiset; repeated types allowed'),
            fit_assessment='Observed residual metrics only. No calibrated pass/fail or posterior probability; synthetic clean<0.05 validation is a separate measure.',
            validation_status=self.manifest['status'],validation_scope='Complete components and both resolution values supplied; no blind-composition or all-modes guarantee',
            timing=dict(loading_seconds=loading_seconds,initial20_seconds=initial_seconds,condition4_and_features_seconds=condition_seconds,neural4_seconds=neural_seconds,numerical_total_seconds=numerical_total,numerical_kernel_seconds=None if numerical_meta is None else numerical_meta['seconds'],forward_and_output_seconds=processing_seconds),
            timing_scope='Actual local/remote device; first calls include compilation. Reuse the object for warm calls. Reported total includes model loading; later calls reuse it. Numerical kernel excludes explicit warmup, but new shapes can still trigger compilation; numerical total includes explicit warmup.',
            devices=[v.name for v in tf.config.list_physical_devices()],candidate_stages=stage)
        result['seconds']=loading_seconds+initial_seconds+condition_seconds+neural_seconds+numerical_total+processing_seconds
        artifact={k:data[k] for k in ('q','observed','sigma','mask')};artifact.update(candidates,candidate_stage=np.array(stage))
        return result,artifact

def main():
    p=argparse.ArgumentParser(description='Known-components conditional prediction; q and sigma_res in nm^-1. No automatic unit/intensity scaling.')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--components',nargs='+',required=True);p.add_argument('--sigma-res',type=float,required=True);p.add_argument('--nu-res',type=float,required=True)
    p.add_argument('--mode',choices=['neural','four-step'],required=True);p.add_argument('--relative-noise',type=float)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Output directory already exists')
    if a.input.suffix.lower()=='.npz':
        with np.load(a.input,allow_pickle=False) as raw:q=raw['q'];observed=raw['observed'];sigma=raw['sigma'];mask=raw['mask'] if 'mask' in raw else None
    else:
        raw=np.loadtxt(a.input,comments='#',ndmin=2)
        if raw.shape[1] not in (2,3):raise ValueError('Text input needs q,intensity[,sigma] columns')
        q,observed=raw[:,0],raw[:,1];mask=None
        if raw.shape[1]==3:sigma=raw[:,2]
        elif a.relative_noise is not None and np.isfinite(a.relative_noise) and a.relative_noise>0:sigma=a.relative_noise*observed
        else:raise ValueError('Two-column input needs an explicit positive --relative-noise assumption')
    result,artifact=ConditionalFastPredictor().predict(q,observed,sigma,a.components,a.sigma_res,a.nu_res,numerical=a.mode=='four-step',mask=mask)
    if a.relative_noise is not None:result['user_relative_noise_assumption']=a.relative_noise
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'solutions.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    np.savez_compressed(a.output/'forward_candidates.npz',**artifact)
    print(json.dumps(dict(output=str(a.output),best_observed_logrmse=result['curves'][0]['best_observed_logrmse'],unique_candidates=result['curves'][0]['unique_candidate_count'],seconds=result['seconds'],numerical_correction=result['numerical_correction'])))
if __name__=='__main__':main()
