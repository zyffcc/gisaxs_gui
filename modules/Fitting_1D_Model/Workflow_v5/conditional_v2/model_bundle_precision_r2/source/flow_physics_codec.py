"""Differentiable equivalent of the validated conditional parameter decoder."""
import numpy as np
import tensorflow as tf
from flow_codec import COMBOS,decode

def decode_tf(theta,combo,gate):
    x=tf.cast(theta,tf.float32);combo=tf.cast(combo,tf.int32);gate=tf.cast(gate,tf.int32)
    typ=tf.gather(tf.constant(COMBOS),combo)
    sigmoid=lambda v:tf.sigmoid(tf.clip_by_value(v,-30.,30.))
    p=tf.reshape(sigmoid(x[:,:24]),[-1,4,6]);radius=tf.exp(p[:,:,0]*np.log(100.))
    lower=tf.maximum(tf.constant(np.log(3.),tf.float32),tf.math.log(2*radius*1.001))
    logd=lower+p[:,:,4]*(np.log(500.)-lower)
    dn=(logd-np.log(3.))/np.log(500./3.)
    p=tf.concat([p[:,:,:4],dn[:,:,None],p[:,:,5:]],2)
    k=tf.reduce_sum(tf.cast(typ>0,tf.int32),1);slots=tf.range(4)[None,:]
    free=tf.concat([x[:,24:27],tf.zeros_like(x[:,:1])],1)
    w=tf.where(slots<(k-1)[:,None],free,tf.where(slots==(k-1)[:,None],0.,-30.))
    d=tf.where(tf.bitwise.bitwise_and(gate[:,None],tf.constant([1,2,4,8]))>0,1.,-1.)
    res=tf.where(tf.bitwise.bitwise_and(gate,16)>0,1.,-1.)
    return dict(params=p,weights=w,globals=sigmoid(x[:,27:]),d=d,res=res,combos=combo,types=typ)

def forward_theta(theta,combo,gate,q,mask):
    from method_evaluation import compiled_forward
    v=decode_tf(theta,combo,gate)
    return compiled_forward(q,mask,v['types'],*[v[k] for k in ('params','weights','globals','d','res')])

def verify_decoder(data):
    from evaluate_benchmark import logrmse
    x=data['theta'];c=data['combo'];g=data['gate'];reference=decode(x,c,g);native=decode_tf(x,c,g)
    delta={k:float(np.max(np.abs(reference[k]-native[k].numpy()))) for k in reference}
    assert max(delta.values())<3e-6,delta
    curves=[]
    for off in range(0,len(x),16):
        sl=slice(off,off+16);curves.append(forward_theta(x[sl],c[sl],g[sl],data['q'][sl],data['mask'][sl]).numpy())
    errors=logrmse(np.concatenate(curves),data['clean'],data['mask']);assert errors.max()<.002,errors.max()
    # A directional derivative through all continuous coordinates, using three
    # single-component geometries; verify against central finite differences.
    ids=np.array([np.flatnonzero((data['types'][:,0]==t)&((data['types']>0).sum(1)==1))[0] for t in (1,2,3)])
    rng=np.random.default_rng(202609160);active=data['active_mask'][ids]
    point=x[ids]+(.08*rng.normal(size=(3,31))*active).astype('float32')
    direction=(rng.normal(size=(3,31))*active).astype('float32');direction/=np.linalg.norm(direction,axis=1,keepdims=True)
    @tf.function
    def loss(v):
        curve=forward_theta(v,c[ids],g[ids],data['q'][ids],data['mask'][ids])
        residual=tf.math.log(tf.maximum(curve,1e-30))-tf.math.log(tf.maximum(data['clean'][ids],1e-30))
        return tf.reduce_mean(tf.reduce_sum(residual**2*data['mask'][ids],1)/data['mask'][ids].sum(1))
    point=tf.constant(point)
    with tf.GradientTape() as tape:tape.watch(point);value=loss(point)
    gradient=tape.gradient(value,point);tf.debugging.assert_all_finite(gradient,'decoder gradient')
    exact=float(tf.reduce_sum(gradient*direction));h=.002
    finite=float((loss(point+h*direction)-loss(point-h*direction))/(2*h))
    relative=abs(exact-finite)/max(abs(exact),abs(finite),1e-3)
    assert relative<.1,(exact,finite,relative)
    return dict(tensor_vs_numpy_max_errors=delta,truth_forward_count=len(x),truth_forward_max_logrmse=float(errors.max()),
        autodiff_directional_derivative=exact,finite_difference_derivative=finite,relative_difference=relative,
        note='training-time differentiability check; no inference optimization introduced')
