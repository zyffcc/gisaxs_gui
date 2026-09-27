"""Experimental signed-observation adapter around the unchanged conditional V2.

Fit measured nodes, rank before deduplication using signed residuals, and render
on an independent grid while retaining the original forward normalization.
The input intensity normalizer and measurement uncertainties are explicit.
"""
import scipy.special,scipy.optimize
import numpy as np
from conditional_fast_predict_v2 import ConditionalFastPredictor,COMBOS
from solution_output import select_candidates,summarize
from referenced_forward64 import referenced_forward,reference_coefficients

def summarize_signed(candidates,observed,sigma,normalizer,render_q,max_solutions=8):
    c=candidates;observed=np.asarray(observed,'float64');sigma=np.asarray(sigma,'float64');render_q=np.asarray(render_q,'float64')
    if c['q'].shape[0]!=1:raise ValueError('One measured curve per call')
    valid=c['mask'][0]>0;reference_q=c['q'][0,valid].astype('float64')
    if observed.shape!=reference_q.shape or sigma.shape!=observed.shape:raise ValueError('Observed/sigma must match native valid nodes')
    if not np.isfinite(normalizer) or normalizer<=0:raise ValueError('Explicit positive intensity normalizer required')
    if not np.isfinite(observed).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):raise ValueError('Finite observations and positive sigma required')
    if render_q.ndim!=1 or not np.isfinite(render_q).all() or np.any(render_q<=0) or np.any(np.diff(render_q)<=0):raise ValueError('Render q must be positive, finite and increasing')
    # Permit the small rounding difference between float32 backend q and user q.
    if render_q[0]<reference_q[0]*(1-1e-6) or render_q[-1]>reference_q[-1]*(1+1e-6):raise ValueError('Rendering outside measured q range is not supported')
    if not isinstance(max_solutions,int) or max_solutions<1:raise ValueError('Positive max_solutions required')
    y=c['curves'][0][:,valid]*normalizer;residual=(y-observed)/sigma
    scores=np.mean(np.where(abs(residual)<=2,.5*residual**2,2*(abs(residual)-1)),1)
    chosen,duplicate_of=select_candidates(c,0,COMBOS,scores,c['mask'][0],.03)
    positive=observed>0;solutions=[];rendered=[]
    for head in chosen[:max_solutions]:
        head=int(head)
        single={k:c[k][:,head:head+1] for k in ('params','weights','globals','d','res','combos','curves')}
        # Reuse physical unit conversion only; each singleton has no cross-head dedup.
        detail=summarize(c,single,COMBOS,max_solutions=1)['curves'][0]['solutions'][0]
        detail['candidate_index']=head;detail.pop('passes_fit_threshold',None)
        detail['input_proxy_logrmse']=detail.pop('observed_logrmse')
        detail.update(candidate_stage=str(c['candidate_stage'][head]),signed_huber_delta2=float(scores[head]),
            signed_weighted_rms=float(np.sqrt(np.mean(residual[head]**2))),
            positive_observation_logrmse=float(np.sqrt(np.mean(np.log(y[head,positive]/observed[positive])**2))) if positive.any() else None,
            reference_coefficients_normalized_intensity=reference_coefficients(c,0,head,reference_q))
        solutions.append(detail);rendered.append(referenced_forward(c,0,head,render_q,reference_q)*normalizer)
    result=dict(status='Experimental real-observation adapter; no new generalization validation',
        ranking='Mean Huber(delta=2) of (prediction-original signed observation)/supplied sigma, computed BEFORE deduplication',
        reference='Forward normalization fixed at measured q; 500 display points do not become observations',
        probability_status='No calibrated posterior or goodness-of-fit probability; depends on supplied sigma',
        normalizer=float(normalizer),raw_candidate_count=len(scores),unique_candidate_count=len(chosen),
        duplicate_of={str(int(k)):int(v) for k,v in duplicate_of.items()},solutions=solutions)
    artifact=dict(reference_q=reference_q,observed=observed,sigma=sigma,render_q=render_q,
        rendered_curves=np.asarray(rendered),selected_raw_heads=np.array([s['candidate_index'] for s in solutions]),raw_scores=scores)
    return result,artifact

class NativeObservationPredictor:
    def __init__(self):self.model=ConditionalFastPredictor()

    def predict(self,q,observed,sigma,components,sigma_res,nu_res,*,normalizer,numerical=False,render_points=500,max_solutions=8):
        q=np.asarray(q,'float64');observed=np.asarray(observed,'float64');sigma=np.asarray(sigma,'float64')
        if q.ndim!=1 or not 8<=len(q)<=1000 or np.any(np.diff(q)<=0) or np.any(q<=0) or not np.isfinite(q).all():raise ValueError('Supply8–1000 strictly increasing positive measured q, one side at a time')
        if observed.shape!=q.shape or sigma.shape!=q.shape or not np.isfinite(observed).all() or not np.isfinite(sigma).all() or np.any(sigma<=0):raise ValueError('Finite measured intensity and positive sigma must match q')
        if not np.isfinite(normalizer) or normalizer<=0:raise ValueError('Explicit positive intensity normalizer required')
        if not isinstance(render_points,int) or render_points<2:raise ValueError('render_points must be an integer >=2')
        proxy=np.maximum(observed,.1*sigma)
        original,c=self.model.predict(q,proxy/normalizer,sigma/normalizer,components,sigma_res,nu_res,numerical=numerical)
        result,artifact=summarize_signed(c,observed,sigma,normalizer,np.linspace(q[0],q[-1],render_points),max_solutions)
        result.update(fixed_parameters=original['fixed_parameters'],components_condition=original['components_condition'],
            units=original['units'],unit_contract=original['unit_contract'],numerical_correction=bool(numerical),
            input_proxy='max(original signed observation,0.1*sigma) used only by frozen positive-input backend; scores retain original signed observations',
            backend_seconds=original['seconds'],timing_scope='Backend time excludes adapter scoring/rendering; not a new timing benchmark')
        result['unit_contract']['inputs']['observed']='Caller intensity divided by explicit normalizer for backend; plotted output restored to caller intensity'
        result['unit_contract']['inputs']['sigma']='Caller intensity uncertainty; divided by same explicit normalizer for backend'
        return result,artifact,c
