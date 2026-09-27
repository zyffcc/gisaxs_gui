"""Local paired perturbations, identity targets and matched curve-loss control."""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','8')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import scipy.special,scipy.optimize
from scaled_condition_common import *
import argparse,time
from numpy_candidate_forward import direct_forward

OUT=R/'results/teacher_path_control_r1'

def verify_local():
    p=json.loads((OUT/'PROTOCOL.json').read_text())
    for n,h in p['sources'].items():assert sha(R/n)==h,n
    verify();return p

def encode_state(b):
    import tensorflow as tf
    from feasible_distance_corrector import logit,distance_floor
    p=b['params'];lo=distance_floor(p[...,0]);u=(p[...,4]-lo)/(1-lo)
    z=logit(p);z=tf.concat([z[...,:4],logit(u)[...,None],z[...,5:]],-1)
    return tf.concat([tf.reshape(z,[-1,24]),b['weights'],logit(b['globals'])],-1)

def decode_state(b,z):
    import tensorflow as tf
    from feasible_distance_corrector import distance_floor
    from conditional_resolution import project_globals
    p=tf.sigmoid(tf.reshape(z[:,:24],[-1,4,6]));lo=distance_floor(p[...,0]);dist=lo+(1-lo)*p[...,4]
    p=tf.concat([p[...,:4],dist[...,None],p[...,5:]],-1)
    return p,z[:,24:28],project_globals(tf.sigmoid(z[:,28:]),b['condition_values'],b['condition_mask'])

def make_bank(d,s,t):
    """TRAIN only: finite good own-head teachers, preserving source/head identities."""
    import tensorflow as tf
    from flow_codec import COMBOS
    from conditional_resolution import static_features,feedback,physics
    from predict_1d import preprocess
    ids,heads=np.where(t['clean_error']<.05);dd=subset(d,ids);ss={k:t[k][ids,heads][:,None] for k in KEYS}
    types=COMBOS[ss['combos'][:,0]]
    b={k:tf.constant(dd[k]) for k in ('q','mask','context')}
    b.update({k:tf.constant(ss[k][:,0]) for k in ('params','weights','globals','d','res')})
    b.update(types=tf.constant(types),condition_values=tf.constant(dd['globals'][:,1:3]),condition_mask=tf.ones((len(ids),2)))
    z=encode_state(b);bounded=tf.concat([tf.clip_by_value(z[:,:24],-8.,8.),z[:,24:28],tf.clip_by_value(z[:,28:],-8.,8.)],-1)
    vals=decode_state(b,bounded)
    for k,v in zip(('params','weights','globals'),vals):b[k]=v
    center=encode_state(b)
    # Float64 target generation: target is a valid curve at this interior center,
    # not a claim that clipping preserves the original fitted teacher exactly.
    c={k:ss[k] for k in ('combos','d','res')}
    for k,v in zip(('params','weights','globals'),vals):c[k]=v.numpy()[:,None]
    clean=np.ones_like(dd['clean'])
    for i in range(len(ids)):
        use=dd['mask'][i]>0;clean[i,use]=direct_forward(c,i,0,dd['q'][i,use].astype('float64'))
    relative_observed=dd['observed']/np.maximum(dd['clean'],1e-30)
    relative_sigma=dd['sigma']/np.maximum(dd['clean'],1e-30)
    dd['clean']=clean;dd['observed']=np.maximum(clean*relative_observed,1e-30);dd['sigma']=np.maximum(clean*relative_sigma,1e-30)
    dd['context']=preprocess(dd['q'],dd['observed'],dd['sigma'],dd['mask'])['context']
    b.update(context=tf.constant(dd['context']),clean=tf.constant(clean))
    b.update({k:tf.constant(v) for k,v in static_features(dd).items()})
    original=dict(b)
    for k in ('params','weights','globals'):original[k]=tf.constant(s[k][ids,heads])
    anchor=encode_state(original)
    act=types>0;dp=(ss['d'][:,0]>0)&act
    mask=np.concatenate([np.stack([act,act,types==2,types==2,dp,dp],-1).reshape(-1,24),act&((act.sum(1)>1)[:,None]),np.tile([1,0,0,1],(len(ids),1))],1).astype('float32')
    b.update(center=center,anchor=anchor,update_mask=tf.constant(mask),relative_observed=tf.constant(relative_observed),relative_sigma=tf.constant(relative_sigma))
    # Small native-q parity audit before any gradient update.
    probe=np.unique(np.linspace(0,len(ids)-1,min(16,len(ids)),dtype=int));bb={k:tf.gather(v,probe) for k,v in b.items()}
    y=physics(bb,bb['params'],bb['weights'],bb['globals']).numpy();diff=[]
    for j,i in enumerate(probe):
        use=dd['mask'][i]>0;diff.append(float(np.sqrt(np.mean(np.log(y[j,use]/clean[i,use])**2))))
    assert max(diff)<1e-4,max(diff)
    meta=dict(rows=ids.tolist(),heads=heads.tolist(),source_indices=d['source_indices'][ids].tolist(),n=len(ids),forward_parity_max=max(diff),
        center_physics='Interior feasible latent coordinates clipped to[-8,8]; globals sigma/nu remain exact. New clean curve recomputed at center; observed/clean and sigma/clean ratios inherited from TRAIN parent only.')
    return b,meta

def augmented(bank,ix,noise,kind,progress):
    import tensorflow as tf
    from conditional_resolution import feedback,physics
    b={k:tf.gather(v,ix) for k,v in bank.items()}
    # Independent columns of the sampled noise vary the true center and start.
    cz=b['center']+noise[:,32:]*b['update_mask']
    cz=tf.concat([tf.clip_by_value(cz[:,:24],-8.,8.),cz[:,24:28],tf.clip_by_value(cz[:,28:],-8.,8.)],-1)
    cp=decode_state(b,cz)
    for key,value in zip(('params','weights','globals'),cp):b[key]=value
    b['center']=encode_state(b)
    clean=physics(b,*cp);b['clean']=clean
    observed=tf.maximum(clean*b['relative_observed'],1e-30)
    sigma=tf.maximum(clean*b['relative_sigma'],1e-30)
    logobs=tf.math.log(observed);logsig=tf.math.log(sigma)
    counts=tf.cast(tf.reduce_sum(b['mask'],1),tf.int32)
    ordered=tf.sort(tf.where(b['mask']>0,observed,tf.fill(tf.shape(observed),tf.constant(float('inf'),tf.float32))),axis=1)
    def at(index):return tf.gather(ordered,index,batch_dims=1)
    qindex=tf.cast(counts-1,tf.float32)*.99
    lo=tf.cast(tf.floor(qindex),tf.int32);hi=tf.cast(tf.math.ceil(qindex),tf.int32)
    # static_features uses linearly interpolated p99; preprocess context uses nearest.
    scale=tf.math.log(at(lo)+(at(hi)-at(lo))*(qindex-tf.floor(qindex)))
    nearest=tf.math.log(at(tf.cast(tf.round(qindex),tf.int32)))
    median=.5*(tf.math.log(at((counts-1)//2))+tf.math.log(at(counts//2)))
    def interp(y):
        low=tf.gather(y,b['left'],batch_dims=1);high=tf.gather(y,b['right'],batch_dims=1)
        return low+(high-low)*b['fraction']
    og=interp(logobs);sg=interp(logsig)
    b['fixed_x']=tf.stack([b['fixed_x'][...,0],(og-scale[:,None])/10,(sg-scale[:,None])/10],-1)
    b['scale']=scale[:,None];b['obsgrid']=og
    b['context']=tf.stack([b['context'][:,0],b['context'][:,1],nearest/10,b['context'][:,3],(median-nearest)/10],-1)
    b['observed']=observed;b['sigma']=sigma
    perturb=noise[:,:32]*b['update_mask']
    bridge=tf.clip_by_value(b['anchor']-b['center'],-12.,12.)*progress[:,None]*b['update_mask']
    shift=tf.where((kind==0)[:,None],tf.zeros_like(perturb),tf.where((kind==3)[:,None],bridge,perturb))
    z=b['center']+shift;z=tf.concat([tf.clip_by_value(z[:,:24],-9.,9.),z[:,24:28],tf.clip_by_value(z[:,28:],-9.,9.)],-1)
    vals=decode_state(b,z);b.update(feedback(b,*vals,physics(b,*vals)))
    target=tf.clip_by_value(b['center']-encode_state(b),-1.,1.)*b['update_mask']
    return b,target


def main():
    p=verify_local();ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    assert a.smoke or os.environ.get('SLURM_JOB_ID'),'Allocated worker required'
    assert not (OUT/'COMPLETE.json').exists()
    tf=tf_setup()
    from feasible_distance_corrector import FeasibleDistanceCorrector
    from portable_corrector_weights import load_corrector_exact,save_corrector_exact
    from train_scaled_condition import combine_teachers
    from train_generalization_controls import evaluate
    from conditional_resolution import feedback,physics,curve_losses
    class LocalCorrector(FeasibleDistanceCorrector):
        def call(self,b,training=False):return decode_update(b,tf.tanh(raw_update(self,b,training)))
    d=pack(O/'train_inputs.npz');s=pack(O/'train_starts.npz')
    if a.smoke:
        ix=json.loads((O/'SOLVER_SMOKE.json').read_text())['rows'];d=subset(d,ix);s=subset(s,ix);t=pack(O/'solver_smoke.npz')
    else:t=combine_teachers('train',len(d['q']))
    tick=time.time();bank,meta=make_bank(d,s,t)
    dump(OUT/('SMOKE_BANK.json' if a.smoke else 'BANK.json'),meta)
    if not a.smoke:np.savez_compressed(OUT/'centers.npz',center=bank['center'].numpy(),mask=bank['update_mask'].numpy(),source_indices=meta['source_indices'],rows=meta['rows'],heads=meta['heads'])
    # Real observation/features and actual initial candidate from the same TRAIN source/head.
    flat=np.array(meta['rows'])*6+np.array(meta['heads'])
    original_parts=[build_batch(d,s,flat[off:off+16]) for off in range(0,len(flat),16)]
    original={key:tf.concat([part[key] for part in original_parts],0) for key in original_parts[0]}
    del original_parts
    if a.smoke:
        reference=build_batch(d,s,flat)
        differences={key:float(tf.reduce_max(tf.abs(tf.cast(original[key],tf.float32)-tf.cast(value,tf.float32)))) for key,value in reference.items()}
        assert max(differences.values())<1e-5,differences
        dump(OUT/'CHUNK_PARITY.json',differences)
    original.update(update_mask=bank['update_mask'])
    teacher_state=dict(original)
    for key in ('params','weights','globals'):
        teacher_state[key]=tf.constant(t[key][meta['rows'],meta['heads']])
    original['center']=encode_state(teacher_state)
    initial_state=encode_state(original)
    active=original['types']>0
    dw=original['center'][:,24:28]-initial_state[:,24:28]
    middle=.5*(tf.reduce_max(tf.where(active,dw,tf.fill(tf.shape(dw),tf.constant(-1e30))),1)+tf.reduce_min(tf.where(active,dw,tf.fill(tf.shape(dw),tf.constant(1e30))),1))
    original['center']=tf.concat([original['center'][:,:24],original['center'][:,24:28]-middle[:,None],original['center'][:,28:]],1)
    # A common softmax offset must leave the teacher's physical mixture unchanged.
    assert np.max(abs(tf.nn.softmax(teacher_state['weights']+tf.cast(~active,tf.float32)*-1e4).numpy()-tf.nn.softmax(original['center'][:,24:28]+tf.cast(~active,tf.float32)*-1e4).numpy()))<1e-6

    assert np.array_equal(original['globals'].numpy()[:,1:3],original['condition_values'].numpy())
    # No clean values enter model features. Compare to original build_batch exactly.
    assert np.array_equal(original['context'].numpy(),d['context'][meta['rows']])
    dump(OUT/('SMOKE_ORIGINAL_BANK.json' if a.smoke else 'ORIGINAL_BANK.json'),dict(n=len(flat),source_indices=meta['source_indices'],rows=meta['rows'],heads=meta['heads'],fixed_conditions_exact=True,original_context_exact=True,starting_state='Original conditioned6 anchors, with original observed/context/static features; no synthetic target intensity substitution.'))
    ks=(bank['types'].numpy()>0).sum(1);groups=[np.flatnonzero(ks==k) for k in range(1,5)]
    assert all(len(g)>0 for g in groups)
    tf.keras.utils.set_random_seed(p['seed']);rng=np.random.default_rng(p['seed'])
    model=LocalCorrector(True);warm={k:v[:8] for k,v in original.items()}
    warm.update(feedback(warm,warm['params'],warm['weights'],warm['globals'],physics(warm,warm['params'],warm['weights'],warm['globals'])))
    model(warm);load_corrector_exact(model,R/p['initial_weights'])
    changed=dict(warm);changed['clean']=warm['clean']*10;changed['center']=warm['center']+3
    assert all(np.array_equal(x.numpy(),y.numpy()) for x,y in zip(model(warm),model(changed))),'Teacher or clean leaked into model features'
    hist=[];opt=tf.keras.optimizers.Adam(2e-5,clipnorm=5.);opt.build(model.trainable_variables)
    def build_step(curve_weight,rollout=1,teacher_forcing=False,parameter_weight=1.):
        @tf.function(reduce_retracing=True)
        def step(ix,noise,kind,progress):
            bb={key:tf.gather(value,ix) for key,value in original.items()}
            m=bb['update_mask'];pes=[];ces=[]
            # Stop gradients through rollout states: supervise actual reached states,
            # without requiring backpropagation through a long physical trajectory.
            with tf.GradientTape() as tape:
                for depth in range(rollout):
                    target=tf.stop_gradient(tf.clip_by_value(bb['center']-encode_state(bb),-1.,1.)*m)
                    dx=tf.tanh(raw_update(model,bb,True))
                    pes.append(tf.reduce_mean(tf.reduce_sum((dx-target)**2*m,1)/tf.reduce_sum(m,1)))
                    pred=decode_update(bb,dx)
                    y=physics(bb,*pred) if curve_weight or depth+1<rollout else None
                    ces.append(tf.reduce_mean(curve_losses(bb,y)) if curve_weight else tf.constant(0.,tf.float32))
                    if depth+1<rollout:
                        # Same additional forward call in both branches; runtime still reported.
                        path_pred=decode_update(bb,target) if teacher_forcing else pred
                        path_y=physics(bb,*path_pred)
                        nxt=feedback(bb,*path_pred,path_y)
                        bb=dict(bb);bb.update({key:tf.stop_gradient(value) for key,value in nxt.items()})
                pe=tf.add_n(pes)/rollout;ce=tf.add_n(ces)/rollout
                loss=parameter_weight*pe+curve_weight*ce
            grad=tape.gradient(loss,model.trainable_variables);tf.debugging.assert_all_finite(loss,'loss')
            for g in grad:tf.debugging.assert_all_finite(g,'gradient')
            norm=tf.linalg.global_norm(grad);opt.apply_gradients(zip(grad,model.trainable_variables));return loss,pe,ce,norm
        return step
    def samples(random):
        ix=np.concatenate([random.choice(g,2,replace=True) for g in groups]).astype('int32')
        kind=random.integers(0,4,8,dtype='int32');scale=random.choice([.1,.3,.7],8)
        noise=np.clip(random.normal(size=(8,64))*scale[:,None],-.9,.9).astype('float32')
        return tuple(tf.constant(x) for x in (ix,noise,kind,random.uniform(0,1,8).astype('float32')))
    if a.smoke:
        from predict_1d import preprocess
        from conditional_resolution import static_features
        probe,_=augmented(bank,*samples(np.random.default_rng(p['seed']+20)))
        dd={k:probe[k].numpy() for k in ('q','mask','observed','sigma')}
        reference=preprocess(dd['q'],dd['observed'],dd['sigma'],dd['mask'])
        static=static_features(dd)
        parity=dict(context=float(np.max(abs(reference['context']-probe['context'].numpy()))),fixed_x=float(np.max(abs(static['fixed_x']-probe['fixed_x'].numpy()))))
        assert max(parity.values())<1e-5,parity
        dump(OUT/'ONLINE_PREPROCESS_PARITY.json',parity)
    warm_steps=2 if a.smoke else p['warmup'];step=build_step(0.)
    for it in range(1,warm_steps+1):
        vals=step(*samples(rng));hist.append([it]+[float(x) for x in vals])
        if it%128==0:
            print('WARM',it,float(vals[0]),flush=True);dump(OUT/'STATUS.json',dict(stage='warmup',update=it))
    shared=[x.copy() for x in model.get_weights()];shared_optimizer=[x.numpy().copy() for x in opt.variables()]
    initial_hash=hashlib.sha256(b''.join(x.tobytes() for x in shared)).hexdigest();result={}
    optimizer_hash=hashlib.sha256(b''.join(x.tobytes() for x in shared_optimizer)).hexdigest();first_sample_hash=None
    for arm,teacher_forcing in [('self_rollout',False),('teacher_path',True)]:
        parameter_weight=1.
        rollout=4
        cw=p['curve_weight']
        model.set_weights(shared)
        for var,value in zip(opt.variables(),shared_optimizer):var.assign(value)
        opt.learning_rate.assign(2e-5)
        assert all(np.array_equal(x,y) for x,y in zip(model.get_weights(),shared))
        assert all(np.array_equal(x.numpy(),y) for x,y in zip(opt.variables(),shared_optimizer))
        rng=np.random.default_rng(p['seed']+1);step=build_step(cw,rollout,teacher_forcing,parameter_weight);armhist=[];arm_tick=time.time()
        sample_hash=hashlib.sha256();counts=np.zeros(len(meta['rows']),'int32');kinds=np.zeros(4,'int32')
        steps=2 if a.smoke else p['branch_updates']
        for it in range(1,steps+1):
            if it==p['lr_decay_at']:opt.learning_rate.assign(5e-6)
            args=samples(rng)
            for x in args:sample_hash.update(x.numpy().tobytes())
            counts+=np.bincount(args[0].numpy(),minlength=len(counts));kinds+=np.bincount(args[2].numpy(),minlength=4)
            vals=step(*args);armhist.append([it]+[float(x) for x in vals])
            if it%128==0:
                print('UPDATE',arm,it,float(vals[0]),flush=True);dump(OUT/'STATUS.json',dict(stage='training',arm=arm,update=it))
        if first_sample_hash is None:first_sample_hash=sample_hash.hexdigest()
        assert first_sample_hash==sample_hash.hexdigest()
        if a.smoke:
            import tempfile
            from pathlib import Path
            with tempfile.TemporaryDirectory() as td:
                checkpoint=Path(td)/'smoke.weights.h5'
                save_corrector_exact(model,checkpoint,B/'weights/corrector/best.weights.h5')
                fresh=LocalCorrector(True);fresh(warm);load_corrector_exact(fresh,checkpoint)
                assert max(float(tf.reduce_max(abs(x-y))) for x,y in zip(model(warm),fresh(warm)))<1e-6
            out=model(warm);assert all(np.isfinite(x.numpy()).all() for x in out)
            assert np.array_equal(out[2].numpy()[:,1:3],warm['condition_values'].numpy())
            result[arm]=dict(final_losses=[float(x) for x in vals],sample_sha256=sample_hash.hexdigest());continue
        dest=OUT/arm;dest.mkdir(exist_ok=True);metrics={}
        training_seconds=time.time()-arm_tick
        save_corrector_exact(model,dest/'final.weights.h5',B/'weights/corrector/best.weights.h5')
        np.savez_compressed(dest/'sampling.npz',counts=counts,kinds=kinds)
        dump(dest/'HISTORY.json',armhist)
        dump(dest/'TRAINED.json',dict(training_seconds=training_seconds,initial_weights_sha256=initial_hash,initial_optimizer_sha256=optimizer_hash,sample_sha256=sample_hash.hexdigest(),weights_sha256=sha(dest/'final.weights.h5')))
        # Evaluation uses original observed data and original anchors, not teacher-centered input.
        for split in ('train','development'):
            dd=pack(O/f'{split}_inputs.npz');ss=pack(O/f'{split}_starts.npz');te=t if split=='train' else combine_teachers(split,len(dd['q']))
            parts=[build_batch(subset(dd,slice(i,i+4)),subset(ss,slice(i,i+4))) for i in range(0,len(dd['q']),4)]
            bb={k:tf.concat([x[k] for x in parts],0) for k in parts[0]}
            for passes in range(1,5):
                outs=[[],[],[]];newparts=[]
                for off in range(0,len(dd['q'])*6,16):
                    sub={k:v[off:off+16] for k,v in bb.items()};pred=model(sub,training=False)
                    for bucket,v in zip(outs,pred):bucket.append(v.numpy())
                    if passes<4:
                        sub.update(feedback(sub,*pred,physics(sub,*pred)));newparts.append(sub)
                if passes in (1,4):
                    name=f'{split}_pass{passes}';metrics[name]=evaluate(dd,ss,[np.concatenate(v) for v in outs],te,dest/f'{name}.npz')
                    print('EVAL',arm,name,json.dumps(metrics[name]['selected']),flush=True)
                if passes<4:bb={k:tf.concat([x[k] for x in newparts],0) for k in newparts[0]}
        fresh=LocalCorrector(True);fresh(warm);load_corrector_exact(fresh,dest/'final.weights.h5')
        reload=max(float(tf.reduce_max(abs(x-y))) for x,y in zip(model(warm),fresh(warm)));assert reload<1e-6
        np.savez_compressed(dest/'sampling.npz',counts=counts,kinds=kinds)
        dump(dest/'HISTORY.json',armhist);result[arm]=dict(training_seconds=training_seconds,rollout=rollout,optimizer_updates=steps,supervised_states=steps*8*rollout,metrics=metrics,initial_weights_sha256=initial_hash,initial_optimizer_sha256=optimizer_hash,sample_sha256=sample_hash.hexdigest(),reload_max_difference=reload,weights_sha256=sha(dest/'final.weights.h5'))
        dump(dest/'COMPLETE.json',result[arm])
    verify_local();dump(OUT/('SMOKE.json' if a.smoke else 'COMPLETE.json'),dict(arms=result,warmup_history=hist,seconds=time.time()-tick,gpu=[str(x) for x in tf.config.list_physical_devices('GPU')],job=os.environ.get('SLURM_JOB_ID'),production_unchanged=True))
    print('ONPOLICY_COMPLETE',flush=True)

if __name__=='__main__':main()
