"""Choose a deployable fixed neural budget on validation only, with fit-monotone retention."""
import argparse,json,time
import numpy as np
import tensorflow as tf
from correction_model import Corrector,features,corrected,multi_metrics
from method_evaluation import load_model
from evaluate_benchmark import logrmse,stats
from diagnose_v5 import R,setup,subset,proposals
from input_representation import prepare,apply

def retain_observed_best(data,old,new):
    a=logrmse(old['curves'],data['observed'][:,None],data['mask'][:,None])
    b=logrmse(new['curves'],data['observed'][:,None],data['mask'][:,None])
    take=b<a
    result=dict(new)
    for k in ('params','weights','globals','d','res','combos','curves'):
        shape=take.shape+(1,)*(new[k].ndim-2)
        result[k]=np.where(take.reshape(shape),new[k],old[k])
    return result

def main(a):
    setup();data,ids=subset(n=a.count,offset=a.validation_offset);base=load_model(R/'results/base_physics',data)
    result={'split':'validation only','validation_offset':a.validation_offset,'indices':ids.tolist(),'methods':{},
        'noise_reference_clean_vs_observed':stats(logrmse(data['clean'],data['observed'],data['mask']))}
    for topk in a.topks:
        initial=proposals(base,data,topk=topk)
        for name in a.models:
            cfg=json.loads((R/'results'/name/'config.json').read_text());representation=cfg.get('representation','log512')
            model=Corrector(True);prep=prepare({k:v[:1] for k,v in data.items()},
                {k:v[:1] for k,v in initial.items() if isinstance(v,np.ndarray)},representation)
            model({k:tf.convert_to_tensor(v) for k,v in prep.items()})
            model.load_weights(str(R/'results'/name/'best.weights.h5'))
            # Warm once on training curves to separate tracing from measured validation runtime.
            from benchmark_model import load_arrays
            raw=load_arrays(R/'cache/train');warm={k:np.asarray(v[:1]) for k,v in raw.items()}
            apply(model,warm,proposals(base,warm,topk=topk),representation,compiled=a.compiled)
            running=initial;best=initial;start=time.monotonic()
            for repeat in range(1,max(a.passes)+1):
                running=apply(model,data,running,representation,compiled=a.compiled);best=retain_observed_best(data,best,running)
                if repeat not in a.passes:continue
                elapsed=time.monotonic()-start
                for keep,cand in [('last',running),('best_observed',best)]:
                    if keep not in a.retentions:continue
                    metric,detail=multi_metrics(data,cand)
                    key=f'{name}|top{topk}|pass{repeat}|{keep}'
                    result['methods'][key]={'model':name,'topk':topk,'passes':repeat,'retention':keep,'representation':representation,'compiled_inference':a.compiled,
                        'seconds_per_curve_excluding_base':elapsed/len(ids),'metrics':metric}
                    result['methods'][key]['per_curve_selected_clean']=detail['clean_error'][np.arange(len(ids)),detail['selected_head']].tolist()
                    m=metric['selected_clean']
                    result['methods'][key]['selection_score']=-m['fraction_lt_0.05']+.1*m['median']+.03*m['p90']
            print('VALIDATED',name,topk,flush=True)
    winner=min(result['methods'],key=lambda key:result['methods'][key]['selection_score'])
    result['winner']=winner
    result['criterion']='-validation fraction<.05 + .1median + .03P90; test unused'
    (R/'results'/a.output).write_text(json.dumps(result,indent=2))
    print('VALIDATION_WINNER',winner,json.dumps(result['methods'][winner]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--count',type=int,default=128)
    p.add_argument('--output',default='candidate_validation.json');p.add_argument('--compiled',action='store_true')
    p.add_argument('--validation-offset',type=int,default=0)
    p.add_argument('--topks',type=int,nargs='+',default=[4,8]);p.add_argument('--passes',type=int,nargs='+',default=[1,3,6,10])
    p.add_argument('--retentions',nargs='+',choices=['last','best_observed'],default=['last','best_observed']);main(p.parse_args())
