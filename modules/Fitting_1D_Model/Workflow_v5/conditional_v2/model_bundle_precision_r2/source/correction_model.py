"""One shared feed-forward correction applied to every frozen proposal head."""
import time
import numpy as np
import tensorflow as tf
from benchmark_model import COMBOS
from fast_refinement import project_tf, forward
from method_evaluation import compiled_forward
from evaluate_benchmark import logrmse, stats

def features(data, cand):
    n,h = cand['combos'].shape
    ids = np.repeat(np.arange(n),h)
    typ = COMBOS[cand['combos']].reshape(-1,4)
    b = {k:cand[k].reshape((-1,)+cand[k].shape[2:]).astype('float32') for k in ('params','weights','globals','d','res')}
    b['types'] = typ
    b['context'] = data['context'][ids].astype('float32')
    x = np.empty((n*h,512,5),dtype='float32')
    for i in range(n):
        ok=data['mask'][i]>.5
        q=np.log(data['q'][i,ok]); grid=np.linspace(q[0],q[-1],512)
        obs=np.log(np.maximum(data['observed'][i,ok],1e-30))
        sig=np.log(np.maximum(data['sigma'][i,ok],1e-30))
        scale=np.log(max(np.percentile(data['observed'][i,ok],99),1e-30))
        common=np.stack([(grid-np.log(9e-5))/np.log(6/9e-5),
            (np.interp(grid,q,obs)-scale)/10,(np.interp(grid,q,sig)-scale)/10],axis=-1)
        for j in range(h):
            pred=np.interp(grid,q,np.log(np.maximum(cand['curves'][i,j,ok],1e-30)))
            x[i*h+j,:,:3]=common
            x[i*h+j,:,3]=(pred-scale)/10
            x[i*h+j,:,4]=np.clip(pred-np.interp(grid,q,obs),-5,5)
    b['x']=x
    return b

class Corrector(tf.keras.Model):
    def __init__(self, feedback=True):
        super().__init__()
        self.feedback=feedback
        self.blocks=[]
        for w in (32,64,96,128):
            self.blocks.append(tf.keras.Sequential([
                tf.keras.layers.Conv1D(w,7,strides=2,padding='same'),
                tf.keras.layers.LayerNormalization(),tf.keras.layers.Activation('swish'),
                tf.keras.layers.Conv1D(w,3,padding='same',activation='swish')]))
        self.flat=tf.keras.layers.Flatten()
        self.hidden=tf.keras.Sequential([tf.keras.layers.Dense(512,activation='swish'),
            tf.keras.layers.Dense(256,activation='swish')])
        self.final=tf.keras.layers.Dense(32,kernel_initializer='zeros',bias_initializer='zeros')

    def call(self,b,training=False):
        x=b['x'] if self.feedback else tf.concat([b['x'][...,:3],tf.zeros_like(b['x'][...,3:])],-1)
        for layer in self.blocks: x=layer(x,training=training)
        active=tf.cast(b['types']>0,tf.float32)
        wp=tf.nn.softmax(b['weights']+(1-active)*-1e4)
        state=tf.concat([tf.reshape(tf.one_hot(b['types'],4)[:,:,1:],[-1,12]),
            tf.reshape(b['params'],[-1,24]),wp,b['globals'],tf.cast(b['d']>0,tf.float32),
            tf.cast(b['res'][:,None]>0,tf.float32),b['context']],-1)
        dx=self.final(self.hidden(tf.concat([self.flat(x),state],-1),training=training))
        def logit(p):
            p=tf.clip_by_value(p,1e-5,1-1e-5)
            return tf.math.log(p)-tf.math.log1p(-p)
        p=project_tf(tf.sigmoid(logit(b['params'])+tf.reshape(dx[:,:24],[-1,4,6])))
        w=b['weights']+dx[:,24:28]
        g=tf.sigmoid(logit(b['globals'])+dx[:,28:32])
        return p,w,g

def losses(model,b,training=True):
    p,w,g=model(b,training=training)
    active=tf.cast(b['types']>0,tf.float32)
    dp=tf.cast(b['d']>0,tf.float32)
    hp=tf.cast(b['types']==2,tf.float32)
    pm=tf.stack([active,active,hp,hp,dp*active,dp*active],-1)
    pe=tf.reduce_sum((p-b['target_params'])**2*pm,[1,2])/tf.maximum(tf.reduce_sum(pm,[1,2]),1.)
    rp=tf.cast(b['res']>0,tf.float32)
    gm=tf.stack([tf.ones_like(rp),rp,rp,rp],-1)
    ge=tf.reduce_sum((g-b['target_globals'])**2*gm,1)/tf.reduce_sum(gm,1)
    wp=tf.nn.softmax(w+(1-active)*-1e4)
    wt=tf.nn.softmax(b['target_weights']+(1-active)*-1e4)
    we=tf.reduce_sum((wp-wt)**2,1)
    curve=forward(b['q'],b['mask'],b['types'],p,w,g,
        tf.where(b['d']>0,30.,-30.),tf.cast(b['res']>0,tf.float32))
    err=tf.math.log(tf.maximum(curve,1e-30))-tf.math.log(tf.maximum(b['clean'],1e-30))
    # Every training pair receives native-grid curve supervision; robust only for very bad starts.
    hub=tf.where(tf.abs(err)<1.,err**2,2*tf.abs(err)-1.)
    ce=tf.reduce_sum(hub*b['mask'],1)/tf.reduce_sum(b['mask'],1)
    quality=tf.exp(-b['teacher_error']/.1)
    cost=100*ce+10*pe+4*ge+2*we
    total=tf.reduce_sum(quality*cost)/tf.reduce_sum(quality)
    return total,tf.reduce_mean(ce),tf.reduce_mean(pe)

def corrected(model,data,cand,prepared=None):
    start=time.monotonic()
    b=features(data,cand) if prepared is None else prepared
    n,h=cand['combos'].shape
    output={k:[] for k in ('params','weights','globals','curves')}
    for i in range(0,n*h,32):
        sl=slice(i,i+32)
        batch={k:tf.convert_to_tensor(v[sl]) for k,v in b.items()}
        p,w,g=model(batch,training=False)
        ids=np.arange(i,min(i+32,n*h))//h
        curve=compiled_forward(data['q'][ids],data['mask'][ids],batch['types'],p,w,g,batch['d'],batch['res'])
        for k,v in zip(output,(p,w,g,curve)): output[k].append(v.numpy())
    result={k:np.concatenate(v).reshape((n,h)+v[0].shape[1:]) for k,v in output.items()}
    result.update({k:cand[k] for k in ('d','res','combos')})
    result['correction_seconds']=time.monotonic()-start
    return result

def multi_metrics(data,cand):
    ec=logrmse(cand['curves'],data['clean'][:,None],data['mask'][:,None])
    eo=logrmse(cand['curves'],data['observed'][:,None],data['mask'][:,None])
    best=eo.argmin(1); rows=np.arange(len(best))
    types=COMBOS[cand['combos']]
    logits=cand['weights']+(types==0)*-1e4
    weights=np.exp(logits-logits.max(-1,keepdims=True)); weights/=weights.sum(-1,keepdims=True)
    substantial=np.all((weights>=.05)|(types==0),axis=-1)
    result={'selected_clean':stats(ec[rows,best]),'selected_observed':stats(eo[rows,best]),
        'oracle_best_clean':stats(ec.min(1)), 'all_heads_clean':stats(ec.ravel()),
        'selected_true_combo':float(np.mean(cand['combos'][rows,best]==data['combo']))}
    for threshold in (.05,.1):
        valid=ec<threshold
        distinct=np.array([len(set(cand['combos'][i,valid[i]].tolist())) for i in rows])
        meaningful=np.array([len(set(cand['combos'][i,valid[i]&substantial[i]].tolist())) for i in rows])
        result[str(threshold)]={'at_least_one':float(np.mean(valid.any(1))),
            'at_least_two_heads':float(np.mean(valid.sum(1)>=2)),
            'at_least_two_topologies':float(np.mean(distinct>=2)),
            'at_least_two_topologies_weights_ge_005':float(np.mean(meaningful>=2)),
            'mean_valid_topologies':float(distinct.mean())}
    result['by_K']={str(k):stats(ec[rows,best][(data['types']>0).sum(1)==k])
        for k in range(1,5) if np.any((data['types']>0).sum(1)==k)}
    return result,{'clean_error':ec,'observed_error':eo,'selected_head':best,'active_weights':weights}
