"""Fast finite-difference trust-region correction of measured-curve candidates."""
import time
import numpy as np
import tensorflow as tf
from scipy.optimize import least_squares
from flow_codec import encode,decode,COMBOS
from flow_physics_codec import forward_theta
from diagnose_v5 import proposals,add_curves
from input_representation import apply
from validate_candidates import retain_observed_best
KEYS=('params','weights','globals','d','res','combos')

@tf.function(reduce_retracing=True)
def forward(x,c,g,q,m):return forward_theta(x,c,g,q,m)

def initial_candidates(predictor,data,combo_ids=None,top_combinations=12):
    """combo_ids is a complete supplied prior per curve; gates always predicted."""
    if combo_ids is None:cand=proposals(predictor.base,data,topk=int(top_combinations))
    else:
        combo_ids=np.asarray(combo_ids,dtype='int32');assert combo_ids.shape==(len(data['q']),)
        cand=proposals(predictor.base,{**data,'combo':combo_ids},truth_combo=True)
    running=cand
    for _ in range(20):
        running=apply(predictor.corrector,data,running,'log512',compiled=True);cand=retain_observed_best(data,cand,running)
    logits=predictor.base.classifier(predictor.base.encode({k:tf.constant(data[k]) for k in ('curve','context')},training=False));cand['classifier_nlp']=np.take_along_axis(-tf.nn.log_softmax(logits).numpy(),cand['combos'],axis=1)
    cand['output_policy']=predictor.output_policy;return cand

def refine_candidates(data,cand,max_nfev=80):
    """Only observed/q/mask enter optimization; all supplied starts are retained."""
    n,h=cand['combos'].shape;wl=cand['weights'].reshape(-1,4);w=np.exp(wl-wl.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
    e=encode(COMBOS[cand['combos'].reshape(-1)],cand['params'].reshape(-1,4,6),w,cand['globals'].reshape(-1,4),cand['d'].reshape(-1,4),cand['res'].reshape(-1));final=e['theta'].copy();records=[]
    for i in range(n):
        qq=np.asarray(data['q'][i:i+1],'float32');mm=np.asarray(data['mask'][i:i+1],'float32');valid=mm[0]>0;target=np.log(np.maximum(data['observed'][i,valid],1e-30)).astype('float64')
        for j in range(h):
            flat=i*h+j;c=e['combo'][flat];g=e['gate'][flat];columns=np.flatnonzero(e['active_mask'][flat]);template=e['theta'][flat].copy();start=time.perf_counter();calls=[0,0];best=[np.inf,template.copy()]
            def curves(values):
                count=len(values);return forward(values,np.full(count,c,'int32'),np.full(count,g,'int32'),np.repeat(qq,count,0),np.repeat(mm,count,0)).numpy()
            def residual(v):
                x=template.copy();x[columns]=v;y=curves(x[None]);r=np.log(np.maximum(y[0,valid],1e-30)).astype('float64')-target;loss=float(np.mean(r*r));calls[0]+=1
                if loss<best[0]:best[:]=[loss,x.copy()]
                return r/np.sqrt(len(target))
            def jacobian(v):
                x=template.copy();x[columns]=v;eps=.001;batch=np.repeat(x[None],2*len(columns),0);batch[np.arange(len(columns)),columns]+=eps;batch[len(columns)+np.arange(len(columns)),columns]-=eps
                y=np.log(np.maximum(curves(batch)[:,valid],1e-30)).astype('float64');calls[1]+=1;return ((y[:len(columns)]-y[len(columns):])/(2*eps)).T/np.sqrt(len(target))
            initial=residual(template[columns]);limits=np.where(np.abs(template)<15.,15.,np.abs(template)+1.)
            result=least_squares(residual,template[columns].astype('float64'),jac=jacobian,bounds=(-limits[columns],limits[columns]),method='trf',x_scale='jac',ftol=1e-6,xtol=1e-5,gtol=1e-6,max_nfev=int(max_nfev))
            final[flat]=best[1];records.append(dict(row=i,head=j,initial_observed_error=float(np.linalg.norm(initial)),final_observed_error=float(np.sqrt(best[0])),forward_calls=calls[0],jacobian_calls=calls[1],status=int(result.status),seconds=time.perf_counter()-start))
    dec=decode(final,e['combo'],e['gate']);out={k:v.reshape(n,h,*v.shape[1:]) for k,v in dec.items()};out=add_curves(data,out)
    for key in ('classifier_nlp','output_policy','components_condition'):
        if key in cand:out[key]=cand[key]
    out['theta']=final.reshape(n,h,31);out['gate_pattern']=e['gate'].reshape(n,h);return out,records
