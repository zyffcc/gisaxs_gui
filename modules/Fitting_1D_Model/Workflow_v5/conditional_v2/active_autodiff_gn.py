"""Fixed-budget Gauss-Newton with forward-mode autodiff Jacobian. Numerical correction."""
import numpy as np
import tensorflow as tf
from train_teacher_path_control import encode_state,decode_state
from active_component_physics import physics

@tf.function(reduce_retracing=True)
def log_forward(z,b):
    return tf.math.log(tf.maximum(physics(b,*decode_state(b,z)),1e-30))

@tf.function(reduce_retracing=True)
def directional(z,tangent,b):
    with tf.autodiff.ForwardAccumulator(z,tangent) as accumulator:
        y=log_forward(z,b)
    return accumulator.jvp(y)


def context(d,c,row):
    from flow_codec import COMBOS
    n=c['combos'].shape[1]
    b={k:tf.constant(np.repeat(d[k][[row]],n,0),tf.float32) for k in ('q','mask')}
    b.update({k:tf.constant(c[k][row],tf.float32) for k in ('params','weights','globals','d','res')})
    b['types']=tf.constant(COMBOS[c['combos'][row]],tf.int32)
    b.update(condition_values=tf.constant(np.repeat(d['globals'][[row],1:3],n,0),tf.float32),condition_mask=tf.ones((n,2)))
    return b

def correct(d,c,row,steps=4,epsilon=.002,dampings=(.001,.01,.1,1.),chunk=16,warmup=False):
    import time
    b=context(d,c,row);z=encode_state(b).numpy();n=len(z);columns=b['types'].numpy();active=columns>0;distance=(c['d'][row]>0)&active
    pm=np.stack([active,active,columns==2,columns==2,distance,distance],-1).reshape(n,24)
    free=np.concatenate([pm,active&(active.sum(1,keepdims=True)>1),np.tile([True,False,False,True],(n,1))],1)
    hids,colids=np.where(free);valid=d['mask'][row]>0;target=np.log(np.maximum(d['observed'][row,valid],1e-30)).astype('float64')
    total_forward_points=0;calls=0;derivative_points=0;derivative_calls=0
    def evaluate(values,ids):
        nonlocal total_forward_points,calls
        results=[]
        for off in range(0,len(values),chunk):
            jj=ids[off:off+chunk];bb={k:tf.gather(v,jj) for k,v in b.items() if k not in ('params','weights','globals')}
            results.append(log_forward(tf.constant(values[off:off+chunk],tf.float32),bb).numpy()[:,valid]);calls+=1
        total_forward_points+=len(values)
        return np.concatenate(results).astype('float64')
    def derivatives(values,ids,columns):
        nonlocal derivative_points,derivative_calls
        results=[]
        # Smaller derivative batches bound temporary JVP memory; physics evaluation remains unchanged.
        for off in range(0,len(values),8):
            jj=ids[off:off+8];bb={k:tf.gather(v,jj) for k,v in b.items() if k not in ('params','weights','globals')}
            tangent=tf.one_hot(columns[off:off+8],32,dtype=tf.float32)
            results.append(directional(tf.constant(values[off:off+8],tf.float32),tangent,bb).numpy()[:,valid]);derivative_calls+=1
        derivative_points+=len(values)
        return np.concatenate(results).astype('float64')
    if warmup:
        evaluate(np.repeat(z[:1],chunk,0),np.zeros(chunk,'int32'));evaluate(z,np.arange(n))
        derivatives(np.repeat(z[:1],8,0),np.zeros(8,'int32'),np.zeros(8,'int32'))
        total_forward_points=0;calls=0;derivative_points=0;derivative_calls=0
    tick=time.perf_counter();y=evaluate(z,np.arange(n));scores=np.mean((y-target)**2,1)
    snapshots={};history=[]
    for step in range(1,steps+1):
        values=derivatives(z[hids],hids,colids)
        jac=np.zeros((n,len(target),32),'float64');jac[hids,:,colids]=values
        assert np.isfinite(jac).all()
        residual=y-target
        gram=np.einsum('hqi,hqj->hij',jac,jac)/len(target)
        rhs=np.einsum('hqi,hq->hi',jac,residual)/len(target)
        diagonal=np.diagonal(gram,axis1=1,axis2=2).copy();scale=np.maximum(diagonal,1e-6)
        updates=[]
        for damping in dampings:
            matrix=gram.copy();matrix[:,np.arange(32),np.arange(32)]+=damping*scale+1e-8
            delta=-np.linalg.solve(matrix,rhs[...,None])[...,0]
            delta*=free;delta/=np.maximum(1,np.max(abs(delta),1,keepdims=True))
            assert np.isfinite(delta).all();updates.append(delta)
        trials=np.stack([z+u for u in updates],1).astype('float32')
        trial_y=evaluate(trials.reshape(-1,32),np.repeat(np.arange(n),len(dampings))).reshape(n,len(dampings),-1)
        trial_scores=np.mean((trial_y-target)**2,2);winner=trial_scores.argmin(1);best=trial_scores[np.arange(n),winner]
        accept=best<scores;z=np.where(accept[:,None],trials[np.arange(n),winner],z)
        y=np.where(accept[:,None],trial_y[np.arange(n),winner],y);new_scores=np.minimum(best,scores)
        assert np.all(new_scores<=scores+1e-12)
        history.append(dict(step=step,accepted=int(accept.sum()),selected_dampings=[float(dampings[j]) if ok else None for j,ok in zip(winner,accept)],observed_rmse=np.sqrt(new_scores).tolist()))
        scores=new_scores
        pred=decode_state(b,tf.constant(z))
        snapshots[step]={k:v.numpy()[None] for k,v in zip(('params','weights','globals'),pred)}
        snapshots[step].update({k:c[k][row:row+1].copy() for k in ('combos','d','res')})
    seconds=time.perf_counter()-tick
    return snapshots,dict(seconds=seconds,forward_head_evaluations=total_forward_points,forward_batches=calls,jvp_head_directions=derivative_points,jvp_batches=derivative_calls,jacobian_method="forward_mode_autodiff",history=history,initial_projection='Encode uses current feasible decoder1e-5 boundary clipping; original candidate retained separately')
