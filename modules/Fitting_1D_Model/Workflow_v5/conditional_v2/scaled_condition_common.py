"""Shared helpers for the bounded128-train/32-development expansion."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;B=R/'model_bundle_precision_r2';O=R/'results/scaled_condition128_r1';D=R/'results/local_teacher512_r1'
sys.path.insert(0,str(B/'source'))
KEYS=('params','weights','globals','d','res','combos')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def pack(p):return dict(np.load(p))
def subset(d,ix):return {k:v[ix] for k,v in d.items()}
def verify():
    p=json.loads((O/'PROTOCOL.json').read_text())
    for category,base in [('sources',R),('inputs',R),('production',B)]:
        for n,h in p[category].items():assert sha(base/n)==h,(category,n)
    return p
def tf_setup():
    import tensorflow as tf
    tf.config.experimental.enable_tensor_float_32_execution(False)
    for dev in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(dev,True)
    return tf
def build_batch(d,s,ix=None):
    import tensorflow as tf
    from conditional_resolution import static_features,feedback,physics
    from flow_codec import COMBOS
    n,h=s['combos'].shape
    ds={k:np.repeat(v,h,0) for k,v in d.items()}
    ss={k:v.reshape(n*h,*v.shape[2:]) for k,v in s.items() if k in KEYS}
    if ix is not None:ds=subset(ds,ix);ss=subset(ss,ix)
    b={k:tf.constant(ds[k]) for k in ('q','mask','clean','context')}
    b.update({k:tf.constant(v) for k,v in static_features(ds).items()})
    b.update({k:tf.constant(ss[k]) for k in ('params','weights','globals','d','res')})
    b.update(types=tf.constant(COMBOS[ss['combos']]),condition_values=tf.constant(ds['globals'][:,1:3]),condition_mask=tf.ones((len(ds['q']),2)))
    b.update(feedback(b,b['params'],b['weights'],b['globals'],physics(b,b['params'],b['weights'],b['globals'])))
    return b
def raw_update(model,b,training=False):
    import tensorflow as tf
    x=b['x']
    for layer in model.blocks:x=layer(x,training=training)
    active=tf.cast(b['types']>0,tf.float32);wp=tf.nn.softmax(b['weights']+(1-active)*-1e4)
    cm=b['condition_mask'];cv=tf.where(cm>0,b['condition_values'],tf.zeros_like(b['condition_values']))
    state=tf.concat([tf.reshape(tf.one_hot(b['types'],4)[:,:,1:],[-1,12]),tf.reshape(b['params'],[-1,24]),wp,b['globals'],tf.cast(b['d']>0,tf.float32),tf.cast(b['res'][:,None]>0,tf.float32),b['context']],-1)
    return model.final(model.hidden(tf.concat([model.flat(x),state,cm,cv],-1),training=training))
def decode_update(b,dx):
    import tensorflow as tf
    from feasible_distance_corrector import feasible_parameters,logit
    from conditional_resolution import project_globals
    return feasible_parameters(b['params'],tf.reshape(dx[:,:24],[-1,4,6])),b['weights']+dx[:,24:28],project_globals(tf.sigmoid(logit(b['globals'])+dx[:,28:]),b['condition_values'],b['condition_mask'])
def update_targets(b,t):
    from feasible_distance_corrector import distance_floor,logit as decoder_logit
    def logit(v):
        v=np.clip(np.asarray(v,'float64'),1e-7,1-1e-7);return np.log(v)-np.log1p(-v)
    def floor(v):return np.maximum(0,np.log(2*np.exp(v*np.log(100))*1.001/3)/np.log(500/3))
    p=t['params'].reshape(-1,4,6).astype('float64');g=t['globals'].reshape(-1,4);w=t['weights'].reshape(-1,4);n=len(p)
    delta=logit(p)-decoder_logit(b['params']).numpy().astype('float64')
    lo=distance_floor(b['params'][...,0]);u0=(b['params'][...,4]-lo)/(1-lo)
    lt=floor(p[...,0]);ut=(p[...,4]-lt)/(1-lt)
    delta[...,4]=logit(ut)-decoder_logit(u0).numpy().astype('float64')
    target=np.concatenate([delta.reshape(n,24),w-b['weights'].numpy(),logit(g)-decoder_logit(b['globals']).numpy().astype('float64')],1).astype('float32')
    typ=b['types'].numpy();act=typ>0;dp=(b['d'].numpy()>0)&act;pm=np.stack([act,act,typ==2,typ==2,dp,dp],-1)
    mask=np.concatenate([pm.reshape(n,24),act&((act.sum(1)>1)[:,None]),np.tile([1,0,0,1],(n,1))],1).astype('float32')
    return target*mask,mask
