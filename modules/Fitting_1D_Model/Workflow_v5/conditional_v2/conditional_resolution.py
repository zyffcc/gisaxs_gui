"""Optional exact sigma_Res/nu_Res conditions for a pretrained neural corrector.

Numerical optimization is never used here. Known values are normalized V5 globals
1 and2; supplying either implies the resolution term is enabled.
"""
import numpy as np
import tensorflow as tf
from correction_model import Corrector,features
from fast_refinement import project_tf,forward
from benchmark_model import COMBOS
from diagnose_v5 import add_curves
from validate_candidates import retain_observed_best

KEYS=('params','weights','globals','d','res','combos')

def physical_conditions(sigma_res=None,nu_res=None):
    values=np.zeros(2,'float32');mask=np.zeros(2,'float32')
    for j,v in enumerate((sigma_res,nu_res)):
        if v is None:continue
        lo,hi=(.007,.013) if j==0 else (5.,10.)
        if not np.isfinite(v) or not lo<=v<=hi:raise ValueError(f'Condition{j} outside trained range [{lo},{hi}]')
        values[j]=np.log(v/lo)/np.log(hi/lo) if j==0 else (v-lo)/(hi-lo)
        mask[j]=1
    return values,mask

def project_globals(g,values,mask):
    # tf.where gives exactly zero output gradient for every fixed coordinate.
    middle=tf.where(mask>0,values,g[:,1:3])
    return tf.concat([g[:,:1],middle,g[:,3:]],1)

class ConditionalCorrector(Corrector):
    def call(self,b,training=False):
        x=b['x']
        for layer in self.blocks:x=layer(x,training=training)
        active=tf.cast(b['types']>0,tf.float32)
        wp=tf.nn.softmax(b['weights']+(1-active)*-1e4)
        cm=tf.cast(b['condition_mask'],tf.float32)
        cv=tf.where(cm>0,b['condition_values'],tf.zeros_like(b['condition_values']))
        state=tf.concat([tf.reshape(tf.one_hot(b['types'],4)[:,:,1:],[-1,12]),tf.reshape(b['params'],[-1,24]),wp,
            b['globals'],tf.cast(b['d']>0,tf.float32),tf.cast(b['res'][:,None]>0,tf.float32),b['context']],-1)
        dx=self.final(self.hidden(tf.concat([self.flat(x),state,cm,cv],-1),training=training))
        def logit(v):
            v=tf.clip_by_value(v,1e-5,1-1e-5);return tf.math.log(v)-tf.math.log1p(-v)
        p=project_tf(tf.sigmoid(logit(b['params'])+tf.reshape(dx[:,:24],[-1,4,6])))
        w=b['weights']+dx[:,24:28]
        g=project_globals(tf.sigmoid(logit(b['globals'])+dx[:,28:32]),b['condition_values'],cm)
        return p,w,g

def transfer(original,warm):
    model=ConditionalCorrector(True);model(warm,training=False)
    for dst,src in zip(model.blocks,original.blocks):dst.set_weights(src.get_weights())
    kernel,bias=original.hidden.layers[0].get_weights()
    model.hidden.layers[0].set_weights([np.concatenate([kernel,np.zeros((4,kernel.shape[1]),kernel.dtype)],0),bias])
    model.hidden.layers[1].set_weights(original.hidden.layers[1].get_weights())
    model.final.set_weights(original.final.get_weights())
    return model

def constrain_candidates(cand,values,mask):
    out={k:np.array(cand[k],copy=True) for k in KEYS}
    for j in range(2):out['globals'][:,:,j+1]=np.where(mask[:,j,None]>0,values[:,j,None],out['globals'][:,:,j+1])
    out['res']=np.where(np.any(mask>0,1)[:,None],1.,out['res']).astype('float32')
    return out

def predict(model,data,start,values,mask,passes=4):
    n,h=start['combos'].shape
    values=np.asarray(values,'float32');mask=np.asarray(mask,'float32')
    if values.shape!=(n,2) or mask.shape!=(n,2):raise ValueError('Conditions must be [curves,2]')
    if not np.isfinite(values).all() or not np.isin(mask,[0,1]).all():raise ValueError('Invalid conditions')
    if np.any(((values<0)|(values>1))&(mask>0)):raise ValueError('Known normalized values must be in [0,1]')
    running=add_curves(data,constrain_candidates(start,values,mask));best=running
    if not hasattr(model,'_conditional_graph'):
        object.__setattr__(model,'_conditional_graph',tf.function(lambda b:model(b,training=False),reduce_retracing=True))
    for _ in range(passes):
        b=features(data,running);b['condition_values']=np.repeat(values,h,0);b['condition_mask']=np.repeat(mask,h,0)
        outputs=[[],[],[]]
        for off in range(0,n*h,16):
            out=model._conditional_graph({k:tf.constant(v[off:off+16]) for k,v in b.items()})
            for bucket,v in zip(outputs,out):bucket.append(v.numpy())
        next_c={k:running[k] for k in ('combos','d','res')}
        for k,bucket in zip(('params','weights','globals'),outputs):
            arr=np.concatenate(bucket);next_c[k]=arr.reshape(n,h,*arr.shape[1:])
        running=add_curves(data,constrain_candidates(next_c,values,mask));best=retain_observed_best(data,best,running)
    for j in range(2):
        known=mask[:,j]>0
        assert np.array_equal(best['globals'][known,:,j+1],np.broadcast_to(values[known,j,None],(known.sum(),h)))
    return best

def static_features(data):
    fixed=[];left=[];right=[];frac=[];scales=[];obsgrid=[]
    for i in range(len(data['q'])):
        use=data['mask'][i]>.5;q=np.log(data['q'][i,use]);grid=np.linspace(q[0],q[-1],512)
        obs=np.log(np.maximum(data['observed'][i,use],1e-30));sig=np.log(np.maximum(data['sigma'][i,use],1e-30))
        scale=np.log(max(np.percentile(data['observed'][i,use],99),1e-30));og=np.interp(grid,q,obs)
        hi=np.clip(np.searchsorted(q,grid,side='right'),1,len(q)-1);lo=hi-1
        fixed.append(np.stack([(grid-np.log(9e-5))/np.log(6/9e-5),(og-scale)/10,(np.interp(grid,q,sig)-scale)/10],-1))
        left.append(lo);right.append(hi);frac.append((grid-q[lo])/(q[hi]-q[lo]));scales.append([scale]);obsgrid.append(og)
    return dict(fixed_x=np.asarray(fixed,'float32'),left=np.asarray(left,'int32'),right=np.asarray(right,'int32'),
        fraction=np.asarray(frac,'float32'),scale=np.asarray(scales,'float32'),obsgrid=np.asarray(obsgrid,'float32'))

def physics(b,p,w,g):
    return forward(b['q'],b['mask'],b['types'],p,w,g,tf.where(b['d']>0,30.,-30.),tf.cast(b['res']>0,tf.float32))

def feedback(b,p,w,g,curve):
    lp=tf.math.log(tf.maximum(curve,1e-30));lo=tf.gather(lp,b['left'],batch_dims=1);hi=tf.gather(lp,b['right'],batch_dims=1)
    grid=lo+(hi-lo)*b['fraction']
    x=tf.concat([b['fixed_x'],((grid-b['scale'])/10)[...,None],tf.clip_by_value(grid-b['obsgrid'],-5.,5.)[...,None]],-1)
    return dict(x=x,params=p,weights=w,globals=g,types=b['types'],d=b['d'],res=b['res'],context=b['context'],
        condition_values=b['condition_values'],condition_mask=b['condition_mask'])

def curve_losses(b,curve):
    err=tf.math.log(tf.maximum(curve,1e-30))-tf.math.log(tf.maximum(b['clean'],1e-30))
    hub=tf.where(tf.abs(err)<.5,err*err,tf.abs(err)-.25)
    return tf.reduce_sum(hub*b['mask'],1)/tf.reduce_sum(b['mask'],1)
