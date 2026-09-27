"""Same frozen forward quadrature, evaluating only the supplied active shape types."""
import tensorflow as tf
import physics_v5
from fast_refinement import project_tf

def physics(b,p,w,g):
    q=b['q'];types=b['types'];params=physics_v5.denormalize_component_params(project_tf(p))
    shape=tf.stack([tf.shape(q)[0],4,tf.shape(q)[1]])
    form=tf.zeros(shape,tf.float32)
    for typ in (1,2,3):
        ix=tf.cast(tf.where(types==typ),tf.int32)
        selected=tf.gather_nd(params,ix);qq=tf.gather(q,ix[:,0])[:,None,:]
        r,sr,h,sh,d,sd=tf.unstack(selected,axis=-1)
        def evaluate():
            if typ==1:value=physics_v5.sphere_form_factor(qq,r[:,None],(r*sr)[:,None])
            elif typ==2:value=physics_v5.random_cylinder_form_factor(qq,r[:,None],(r*sr)[:,None],h[:,None],(h*sh)[:,None])
            else:value=physics_v5.vertical_cylinder_form_factor(qq,r[:,None],sr[:,None])
            return value[:,0,:]
        values=tf.cond(tf.shape(ix)[0]>0,evaluate,lambda:tf.zeros([0,tf.shape(q)[1]],tf.float32))
        form+=tf.scatter_nd(ix,values,shape)
    d=params[...,4];sd=params[...,5]
    probability=tf.sigmoid(tf.where(b['d']>0,30.,-30.))
    form*=physics_v5.structure_factor(q[:,None,:],d,d*sd,probability)
    active=tf.cast(types>0,tf.float32);weights=tf.nn.softmax(w+(1-active)*-1e4,axis=-1)*active
    particle=tf.reduce_sum(weights[:,:,None]*form,axis=1)
    return physics_v5._v5_add_resolution_background(q,particle,tf.pad(g,[[0,0],[0,1]]),tf.cast(b['res']>0,tf.float32),b['mask']>.5)
