"""Validation-only localization of topology, gates, parameters and ranking errors."""
import os,json,time,argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
from benchmark_model import COMBOS,load_arrays
from method_evaluation import load_model,compiled_forward
from evaluate_benchmark import project,logrmse,stats
from correction_model import multi_metrics

R=Path('/data/dust/user/zhaiyufe/MaxwellRuns/network-benchmark-20260914-v5')

def setup():
    assert os.environ.get('SLURM_JOB_ID'),'Use a worker'
    devices=tf.config.list_physical_devices('GPU');assert devices
    for d in devices:tf.config.experimental.set_memory_growth(d,True)
    tf.keras.utils.set_random_seed(20260930)

def subset(split='val',n=128,offset=0):
    raw=load_arrays(R/'cache'/split)
    types=raw['types'];limit=min(offset+2500,len(types)) if split=='val' else len(types)
    assert offset>=0 and n%4==0
    rng=np.random.default_rng(20260930);ks=(types[offset:limit]>0).sum(1)
    ids=np.concatenate([rng.permutation(np.flatnonzero(ks==k))[:n//4] for k in (1,2,3,4)])+offset
    assert len(ids)==n,'Not enough validation records for each K'
    return {k:np.asarray(v[ids]) for k,v in raw.items()},ids

def proposals(model,data,topk=4,truth_combo=False,truth_gates=False):
    z=model.encode({k:tf.convert_to_tensor(data[k]) for k in ('curve','context')})
    probabilities=tf.nn.softmax(model.classifier(z)).numpy()
    combos=data['combo'][:,None] if truth_combo else np.argsort(-probabilities,axis=1)[:,:topk]
    n,k=combos.shape;h=model.hypotheses
    out={key:v.numpy() for key,v in model.decode(tf.repeat(z,k,axis=0),combos.ravel()).items()}
    cand={'params':project(out['params']).reshape(n,k*h,4,6),
        'weights':out['weight_logits'].reshape(n,k*h,4),'globals':out['globals'].reshape(n,k*h,4),
        'd':out['d_logits'].reshape(n,k*h,4),'res':out['resolution_logits'].reshape(n,k*h),
        'combos':np.repeat(combos,h,axis=1),'probabilities':probabilities}
    if truth_gates:
        assert truth_combo
        cand['d']=np.repeat((data['d']*2-1)[:,None],k*h,axis=1)
        cand['res']=np.repeat((data['resolution']*2-1)[:,None],k*h,axis=1)
    return add_curves(data,cand)

def add_curves(data,cand,batch=32):
    n,h=cand['combos'].shape;curves=[]
    types=COMBOS[cand['combos']].reshape(-1,4)
    vals={k:cand[k].reshape((-1,)+cand[k].shape[2:]).astype('float32') for k in ('params','weights','globals','d','res')}
    for start in range(0,n*h,batch):
        sl=slice(start,start+batch);ids=np.arange(start,min(start+batch,n*h))//h
        c=compiled_forward(data['q'][ids],data['mask'][ids],types[sl],*[vals[k][sl] for k in ('params','weights','globals','d','res')])
        curves.append(c.numpy())
    cand['curves']=np.concatenate(curves).reshape(n,h,1000)
    assert np.isfinite(cand['curves']).all()
    return cand

def main():
    setup();data,ids=subset();base=load_model(R/'results/base_physics',data)
    result={'split':'validation only','source_indices':ids.tolist(),'stages':{}}
    start=time.monotonic()
    for name,kwargs in [('top4',{}),('top8',{'topk':8}),('oracle_combo',{'truth_combo':True}),('oracle_combo_gates',{'truth_combo':True,'truth_gates':True})]:
        cand=proposals(base,data,**kwargs);metric,detail=multi_metrics(data,cand)
        metric['combination_coverage']=float(np.mean(np.any(cand['combos']==data['combo'][:,None],axis=1)))
        metric['selection_regret_median']=float(np.median(detail['clean_error'][np.arange(len(ids)),detail['selected_head']]-detail['clean_error'].min(1)))
        result['stages'][name]=metric
        if name=='top4':
            np.savez_compressed(R/'results/validation_initial.npz',source_indices=ids,**{k:v for k,v in cand.items() if isinstance(v,np.ndarray)})
        print('DIAGNOSTIC',name,json.dumps({'selected':metric['selected_clean'],'oracle':metric['oracle_best_clean'],'coverage':metric['combination_coverage']}),flush=True)
    truth={'params':data['params'][:,None],'weights':np.log(np.maximum(data['weights'],1e-10))[:,None],
        'globals':data['globals'][:,None],'d':(data['d']*2-1)[:,None],'res':(data['resolution']*2-1)[:,None],'combos':data['combo'][:,None]}
    result['truth_forward']=stats(logrmse(add_curves(data,truth)['curves'][:,0],data['clean'],data['mask']))
    result['seconds']=time.monotonic()-start
    (R/'results/diagnosis.json').write_text(json.dumps(result,indent=2))
    print('DIAGNOSIS_COMPLETE',flush=True)

if __name__=='__main__':main()
