"""Portable prediction entry: candidates from fixed neural passes, no optimizer."""
import argparse,json,time
from pathlib import Path
import numpy as np
import tensorflow as tf
import physics_v5
from TrainSetBuild import schema
from benchmark_model import ProposalModel,COMBOS
from correction_model import Corrector,features,corrected
from diagnose_v5 import proposals
from validate_candidates import retain_observed_best
from evaluate_benchmark import logrmse
from input_representation import prepare,apply
from component_prior import resolve_components

def preprocess(q,observed,sigma,mask=None):
    q,observed,sigma=[np.atleast_2d(np.asarray(x,dtype='float32')) for x in (q,observed,sigma)]
    assert q.shape==observed.shape==sigma.shape
    mask=q>0 if mask is None else np.atleast_2d(mask).astype(bool)
    assert mask.shape==q.shape and q.shape[1]<=1000
    n=len(q);data={k:np.zeros((n,1000),dtype='float32') for k in ('q','observed','sigma','mask')}
    curves=[];contexts=[]
    for i in range(n):
        use=mask[i];count=int(use.sum());assert 8<=count<=1000
        native=q[i,use];obs=observed[i,use];sig=sigma[i,use]
        assert np.isfinite(native).all() and np.all(np.diff(native)>0) and np.all(native>0)
        assert np.isfinite(obs).all() and np.all(obs>0) and np.isfinite(sig).all() and np.all(sig>0)
        for key,value in [('q',native),('observed',obs),('sigma',sig)]:data[key][i,:count]=value
        data['mask'][i,:count]=1
        lq=np.log(native);grid=np.linspace(lq[0],lq[-1],256)
        p99=max(float(np.quantile(obs,.99,method='nearest')),1e-30)
        li=np.log(obs)-np.log(p99);ls=np.log(sig)-np.log(p99)
        curves.append(np.stack([(grid-np.log(9e-5))/np.log(6/9e-5),np.interp(grid,lq,li)/10,np.interp(grid,lq,ls)/10],-1))
        contexts.append([lq[0]/10,lq[-1]/10,np.log(p99)/10,count/1000,np.median(li)/10])
    data['curve']=np.asarray(curves,dtype='float32');data['context']=np.asarray(contexts,dtype='float32')
    return data

class Predictor:
    def __init__(self,bundle,data,topk=None,passes=None):
        self.bundle=Path(bundle);self.config=json.loads((self.bundle/'model.json').read_text())
        self.output_policy=json.loads((self.bundle/'output_policy.json').read_text())
        if topk is not None or passes is not None:
            default={k:self.config[k] for k in ('topk','passes')}
            requested={'topk':default['topk'] if topk is None else topk,
                'passes':default['passes'] if passes is None else passes}
            allowed=self.config.get('validated_budgets',[default])
            if requested not in allowed:raise ValueError(f'Budget must be one of the validation-supported settings: {allowed}')
            self.config['default_budget']=default;self.config.update(requested)
        directory=self.bundle/self.config['base_dir'];bc=json.loads((directory/'config.json').read_text())
        self.base=ProposalModel(bc['architecture'],bc['hypotheses'])
        self.base({'curve':data['curve'][:1],'context':data['context'][:1],'combo':np.zeros(1,'int32')})
        self.base.load_weights(str(directory/'best.weights.h5'))
        self.corrector=None
        if self.config['passes']:
            self.corrector=Corrector(True)
            warm={k:v[:1] for k,v in data.items()};cand=proposals(self.base,warm,topk=self.config['topk'])
            self.corrector({k:tf.convert_to_tensor(v[:1]) for k,v in prepare(warm,cand,self.config.get('representation','log512')).items()})
            self.corrector.load_weights(str(self.bundle/self.config['corrector_dir']/'best.weights.h5'))
    def predict(self,data,components=None):
        start=time.monotonic();condition=None
        if components is None:
            cand=proposals(self.base,data,topk=self.config['topk'])
        else:
            combo,types=resolve_components(components,COMBOS)
            conditional_data={**data,'combo':np.full(len(data['q']),combo,dtype='int32')}
            cand=proposals(self.base,conditional_data,truth_combo=True)
            condition={'combination_id':combo,'component_types':types,'semantics':'complete multiset, including repeated types'}
        best=cand
        for _ in range(self.config['passes']):
            cand=apply(self.corrector,data,cand,self.config.get('representation','log512'),compiled=self.config.get('compiled_inference',False))
            best=retain_observed_best(data,best,cand)
        result=best if self.config.get('retain_best',True) else cand
        logits=self.base.classifier(self.base.encode({k:tf.convert_to_tensor(data[k]) for k in ('curve','context')}))
        result['classifier_nlp']=np.take_along_axis(-tf.nn.log_softmax(logits).numpy(),result['combos'],axis=1)
        result['output_policy']=self.output_policy
        result['components_condition']=condition
        result['seconds']=time.monotonic()-start
        return result

def canonical_vector(cand,i,j):
    from solution_output import canonical_vector as canonical
    return canonical(cand,i,j,COMBOS)


def summarize(data,cand,max_solutions=8,threshold=.03,distance=.03,max_parameters=None):
    from solution_output_r3 import summarize as summarize_output
    return summarize_output(data,cand,COMBOS,max_solutions,threshold,distance,max_parameters)


def main(a):
    for d in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(d,True)
    raw=np.load(a.input,allow_pickle=False)
    data=preprocess(raw['q'],raw['observed'],raw['sigma'],raw['mask'] if 'mask' in raw else None)
    predictor=Predictor(a.bundle,data,topk=a.topk,passes=a.passes);cand=predictor.predict(data,components=a.components)
    a.output.mkdir(parents=True,exist_ok=False)
    result=summarize(data,cand,a.max_solutions,a.threshold,max_parameters=a.max_parameters);result['model']=predictor.config
    (a.output/'solutions.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    np.savez_compressed(a.output/'forward_candidates.npz',**data,**{k:cand[k] for k in ('params','weights','globals','d','res','combos','curves','classifier_nlp')})
    print(json.dumps({'curves':len(data['q']),'seconds':cand['seconds'],'output':str(a.output)}))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('bundle','input','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--max-solutions','--max-combinations',dest='max_solutions',type=int,default=8);p.add_argument('--threshold',type=float,choices=[.03,.05],default=.03)
    p.add_argument('--topk',type=int);p.add_argument('--passes',type=int)
    p.add_argument('--components',nargs='+',help='Exact complete multiset: sphere/random_cylinder/vertical_cylinder or 1/2/3')
    p.add_argument('--max-parameters',type=int,help='Per-combination display cap; default keeps all available distinct heads')
    main(p.parse_args())
