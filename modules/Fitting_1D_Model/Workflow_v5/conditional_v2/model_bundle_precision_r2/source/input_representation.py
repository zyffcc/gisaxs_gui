"""Optional native-q residual input; preserve fine peaks and explicit padding mask."""
import numpy as np
import tensorflow as tf
from correction_model import features,corrected
from benchmark_model import COMBOS

def prepare(data,cand,representation='log512'):
    if representation=='log512':return features(data,cand)
    assert representation=='native1000'
    n,h=cand['combos'].shape;ids=np.repeat(np.arange(n),h)
    b={k:cand[k].reshape((-1,)+cand[k].shape[2:]).astype('float32') for k in ('params','weights','globals','d','res')}
    b['types']=COMBOS[cand['combos']].reshape(-1,4)
    b['context']=data['context'][ids].astype('float32')
    mask=data['mask'][ids].astype('float32');q=data['q'][ids]
    obs=np.log(np.maximum(data['observed'][ids],1e-30));sig=np.log(np.maximum(data['sigma'][ids],1e-30))
    pred=np.log(np.maximum(cand['curves'].reshape(n*h,1000),1e-30))
    scales=np.array([np.log(max(np.percentile(data['observed'][i,data['mask'][i]>.5],99),1e-30)) for i in range(n)])[ids,None]
    channels=[(np.log(np.maximum(q,1e-30))-np.log(9e-5))/np.log(6/9e-5),
        (obs-scales)/10,(sig-scales)/10,(pred-scales)/10,np.clip(pred-obs,-5,5),mask]
    b['x']=(np.stack(channels,-1)*mask[...,None]).astype('float32')
    return b

def apply(model,data,cand,representation='log512',compiled=False):
    network=model
    if compiled:
        if not hasattr(model,'_v5_inference_graph'):
            graph=tf.function(lambda b:model(b,training=False),reduce_retracing=True)
            object.__setattr__(model,'_v5_inference_graph',graph)
        network=lambda b,training=False:model._v5_inference_graph(b)
    return corrected(network,data,cand,prepared=prepare(data,cand,representation))
