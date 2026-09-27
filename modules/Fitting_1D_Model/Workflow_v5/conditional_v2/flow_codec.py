"""Active, bounded coordinates for a conditional parameter-density pilot."""
import itertools
import numpy as np
COMBOS=np.array([c+(0,)*(4-k) for k in range(1,5) for c in itertools.combinations_with_replacement((1,2,3),k)],'int32')
LOOKUP={tuple(c):i for i,c in enumerate(COMBOS)}
BITS=(1<<np.arange(5)).astype('int32')

def masks(combos,gates):
    typ=COMBOS[np.asarray(combos)];active=typ>0;d=(np.asarray(gates)[:,None]&BITS[:4])>0;r=(np.asarray(gates)&16)>0
    pm=np.stack([active,active,typ==2,typ==2,active&d,active&d],-1).reshape(-1,24)
    wm=np.arange(3)[None,:]<(active.sum(1)-1)[:,None]
    gm=np.stack([np.ones(len(typ),bool),r,r,r],1)
    return np.concatenate([pm,wm,gm],1).astype('float32')

def encode(types,params,weights,globals_,d,res):
    t=np.asarray(types).copy();p=np.asarray(params).copy();w=np.asarray(weights).copy();dd=np.asarray(d).copy()
    for i in range(len(t)):
        order=sorted(range(4),key=lambda j:(int(t[i,j]) if t[i,j]>0 else 9,float(p[i,j,0]),j))
        t[i]=t[i,order];p[i]=p[i,order];w[i]=w[i,order];dd[i]=dd[i,order]
    c=np.array([LOOKUP[tuple(v)] for v in t],'int32');dd=(dd>0)&(t>0);rr=np.asarray(res)>0
    gates=(dd*BITS[:4]).sum(1).astype('int32')+rr.astype('int32')*16
    x=p.astype('float64').copy();rad=np.exp(x[:,:,0]*np.log(100.))
    lower=np.maximum(np.log(3.),np.log(2*rad*1.001));logd=np.log(3.)+x[:,:,4]*np.log(500/3)
    x[:,:,4]=(logd-lower)/(np.log(500.)-lower)
    def logit(v):
        v=np.clip(v,1e-5,1-1e-5);return np.log(v)-np.log1p(-v)
    theta=np.zeros((len(t),31));theta[:,:24]=logit(x).reshape(-1,24);theta[:,27:]=logit(np.asarray(globals_)[:,:4])
    for i,k in enumerate((t>0).sum(1)):
        if k>1:theta[i,24:24+k-1]=np.log(np.maximum(w[i,:k-1],1e-12)/max(w[i,k-1],1e-12))
    mask=masks(c,gates);theta*=mask
    return dict(combo=c,theta=theta.astype('float32'),active_mask=mask,gate=gates,types=t.astype('int32'),
        params=p.astype('float32'),weights=w.astype('float32'),globals=np.asarray(globals_)[:,:4].astype('float32'),d=dd.astype('float32'),res=rr.astype('float32'))

def decode(theta,combos,gates):
    x=np.asarray(theta,dtype='float64');c=np.asarray(combos,dtype='int32');gates=np.asarray(gates,dtype='int32');typ=COMBOS[c]
    def sigmoid(v):return 1/(1+np.exp(-np.clip(v,-30,30)))
    p=sigmoid(x[:,:24]).reshape(-1,4,6);rad=np.exp(p[:,:,0]*np.log(100.))
    lower=np.maximum(np.log(3.),np.log(2*rad*1.001));logd=lower+p[:,:,4]*(np.log(500.)-lower)
    p[:,:,4]=(logd-np.log(3.))/np.log(500/3)
    w=np.full((len(x),4),-30.,'float64')
    for i,k in enumerate((typ>0).sum(1)):
        w[i,k-1]=0.
        if k>1:w[i,:k-1]=x[i,24:24+k-1]
    d=np.where((gates[:,None]&BITS[:4])>0,1.,-1.);res=np.where((gates&16)>0,1.,-1.)
    return dict(params=p.astype('float32'),weights=w.astype('float32'),globals=sigmoid(x[:,27:]).astype('float32'),d=d.astype('float32'),res=res.astype('float32'),combos=c)
