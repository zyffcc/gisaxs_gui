'Joint photon realization for q images and native empirical descriptors.'
import numpy as np
from .tool_prior import physical
from .tool_qinput import AF_EDGES,Q_EDGES,GRID_SHAPE,angles,tensor
from .features import AF_BANDS,V_EDGES
from .cell_response import adaptive_nodes,rule,response
from .forward import _stack,simulate
from .forward_v06_gpu import ssca_cuda
from .stage5_gpu import complex_amplitudes_cuda
from .prediction import summarize_profiles
from .resolvable import PEAK_WINDOW,coverage,visible_for_regression

NATIVE_EDGES=np.unique(np.round(np.r_[np.arange(.02,1.50001,.01),AF_EDGES,V_EDGES,np.array(AF_BANDS).ravel()],10))
NATIVE_AF=(NATIVE_EDGES[:-1]+NATIVE_EDGES[1:])/2
Q=(Q_EDGES[:-1]+Q_EDGES[1:])/2

class NativeOperator:
    def __init__(self,x,material,shape,metal,E,ai,order=4,*,q_order=None):
        self.x=np.asarray(x,float).copy();self.material=material;self.shape=shape;self.metal=metal;self.E=E;self.ai=ai
        s,p=physical(x,material,shape,metal,E,ai);s._measurement_af_shift=x[19];s._measurement_qscale=x[18]
        nd,_,_,_,_=_stack(s,p.state(),448)
        ac,aw,self.qc,self.qw,meta=adaptive_nodes(s,p,nd,order if q_order is None else q_order)
        da=float(AF_EDGES[1]-AF_EDGES[0]);minimum=.03*da;shift=x[19]
        edges=np.unique(np.r_[AF_EDGES,NATIVE_EDGES,V_EDGES,np.array(AF_BANDS).ravel()])


        extra=[-shift];extra.extend((edges[:,None]+[-2*minimum,2*minimum]).ravel())
        for nn in nd:
            delta=max(float(np.real(nn)),0.)
            if delta<=0:continue
            critical=np.sqrt(2*delta);width=max(abs(float(np.imag(nn)))/max(critical,1e-12),1e-6)
            extra.extend(np.degrees(critical+np.array([-4,-1,0,1,4])*width)-shift)

        self.ac,self.aw=rule(edges,order,4*da,extra,meta['AF_max_segment_deg'])
        self.ang=angles(E,ai,x[18],x[19],q_centres=self.qc,af_centres=self.ac)
        self.sample=s;self.particle=p
        from .multilayer import amplitudes
        k=2*np.pi/(1239.841984/E);nd,thick,sigma,rough,iz=_stack(s,p.state(),448)
        aa=amplitudes(k,np.maximum(np.radians(self.ac+x[19]),1e-6),nd,thick,sigma)
        self.kzf_bound=float(np.max(abs(aa['kz'][iz])))
        self.base=simulate(s,None,self.ang)['total'];self.base=np.where(self.ang['alpha_f']>0,self.base,0.)
        self.rule_meta={**meta,'AF_nodes':len(self.ac),'AF_order':order,'q_order':order if q_order is None else q_order,'native_AF_edges':len(NATIVE_EDGES),'same_q_cuts':True}
    def evaluate(self,x):
        out=self.evaluate_many([x]);return out[0]
    def evaluate_many(self,rows):
        from .tool_operator_v11 import AdaptiveOperator
        rows=np.asarray(rows,float)
        for x in rows:
            if not np.allclose(x[list(AdaptiveOperator.FIXED)],self.x[list(AdaptiveOperator.FIXED)],rtol=0,atol=1e-12):raise ValueError('Changed structural field family')
        q=self.ang['qpar'];qt=self.ang['q_total'];flat=q.ravel();values=np.zeros((len(rows),len(flat)))


        for lo in range(0,len(flat),16384):
            hi=min(lo+16384,len(flat));ang={k:np.asarray(v).ravel()[lo:hi] for k,v in self.ang.items()}
            F,R,weights,pref=complex_amplitudes_cuda(self.sample,self.particle.state(),ang,448,192,kzf_bound=self.kzf_bound)
            for j,x in enumerate(rows):values[j,lo:hi]=pref*ssca_cuda(F,R,weights,flat[lo:hi],x[0],x[4]*x[0],x[5])
            del F
        if not np.isfinite(values).all() or values.min()<-1e-7*max(float(abs(values).max()),1e-30):raise ValueError('Nonfinite/negative SSCA')
        return [self._project(np.maximum(value,0).reshape(q.shape),x) for value,x in zip(values,rows)]
    def _project(self,Ip,x):
        q=self.ang['qpar'];qt=self.ang['q_total']
        Wa,ea=response(NATIVE_EDGES,self.ac,self.aw,x[16]*(AF_EDGES[1]-AF_EDGES[0]))
        Wc,ec=response(AF_EDGES,self.ac,self.aw,x[16]*(AF_EDGES[1]-AF_EDGES[0]))
        Wq,eq=response(Q_EDGES,self.qc,self.qw,x[17]*(Q_EDGES[1]-Q_EDGES[0]))
        def project(v,W=Wa):return W@np.asarray(v).reshape(len(self.ac),len(self.qc))@Wq.T
        particle=project(Ip);base=project(self.base)
        ref=max(float(np.quantile(project(Ip,Wc),.8)),1e-30)
        bg=x[14]*ref+x[12]*ref/(1+(q*x[13])**2)**2+x[15]*ref*np.exp(-.5*((qt-x[21])/x[22])**2)
        background=project(bg);mean=particle+base+background;pre=base+background
        flux=x[20]/max(float(np.quantile(project(Ip+self.base+bg,Wc),.8)),1e-30)
        return mean*flux,pre*flux,{'flux':float(flux),'rule':{**self.rule_meta,'raw_constant_response_error_nativeAF':ea,'raw_constant_response_error_modelAF':ec,'raw_constant_response_error_q':eq}}

def residual_rates(rate,residual,amplitude,tau0=.11):
    'Transport residual in original log1p(tau0*rate) units before photons.'
    rate=np.asarray(rate,float);e=np.asarray(residual,float)*float(amplitude)
    if e.shape!=rate.shape:raise ValueError('Residual q/nativeAF coordinates differ')
    log=np.log1p(tau0*np.maximum(rate,0))+e
    return np.maximum(np.expm1(np.maximum(log,0))/tau0,0),float(np.mean(log<0))

def counts_profiles(Y,N):
    'Both image and all empirical features use these same angular counts.'
    Y=np.asarray(Y,float);N=np.asarray(N,float)
    expected=(2,len(NATIVE_AF),len(Q))
    if Y.shape!=expected or N.shape!=expected:raise ValueError('Expected two nativeAF/q photon sides')
    hy=[];hn=[]
    for lo,hi in AF_BANDS:
        sel=(NATIVE_AF>=lo)&(NATIVE_AF<hi)
        for side in range(2):hy.append(Y[side,sel].sum(axis=0));hn.append(N[side,sel].sum(axis=0))
    qs=(Q>=.15)&(Q<1.2);vi=np.searchsorted(V_EDGES,NATIVE_AF,side='right')-1;ok=(vi>=0)&(vi<len(V_EDGES)-1)
    vy=np.bincount(vi[ok],weights=Y[:,:,qs].sum(axis=(0,2))[ok],minlength=len(V_EDGES)-1)
    vn=np.bincount(vi[ok],weights=N[:,:,qs].sum(axis=(0,2))[ok],minlength=len(V_EDGES)-1)
    ci=np.searchsorted(AF_EDGES,NATIVE_AF,side='right')-1;cy=np.zeros(GRID_SHAPE);cn=np.zeros(GRID_SHAPE)
    np.add.at(cy,ci,Y.sum(axis=0));np.add.at(cn,ci,N.sum(axis=0))
    rates=np.divide(cy,cn,out=np.zeros_like(cy),where=cn>0);image,scale=tensor(rates,cn>0)
    return np.array(hy),np.array(hn),vy,vn,image,scale

def describe(Y,N,VY,VN):
    fits,bands,selected,vr=summarize_profiles(Y,N,VY,VN,PEAK_WINDOW);cov=coverage(N)
    feature=np.zeros(20,np.float32);known=np.zeros(20,np.float32)
    for b,band in enumerate(bands):
        if band['q_star'] is not None:
            feature[2*b]=np.log(band['q_star']);feature[2*b+1]=np.log(band['FWHM']/band['q_star']);known[2*b:2*b+2]=1
    y=vr.get('alpha_f_observed_ridge')
    if y is not None and vr.get('ridge_status')=='empirical_local_maximum':feature[8]=y;known[8]=1

    for b in range(4):
        r=np.divide(Y[2*b:2*b+2].sum(axis=0),N[2*b:2*b+2].sum(axis=0),out=np.zeros(len(Q)),where=N[2*b:2*b+2].sum(axis=0)>0)
        a=(Q>=.9)&(Q<1.2)&(r>0);z=(Q>=1.4)&(Q<1.7)&(r>0)
        if a.sum()>=3 and z.sum()>=3:feature[9+b]=np.log(np.median(r[z])/np.median(r[a]));known[9+b]=1
    visible=visible_for_regression(selected,cov)
    return {'features':np.r_[feature,known].astype(np.float32),'visible_peak':visible,'coverage':cov,
            'q_star':selected['q_star'] if selected else None,'width':selected['FWHM'] if selected else None,'Yoneda':y,
            'support':selected['support'] if selected else 'unresolved'}

def observe(mean,pre,E,rng,residual=None):
    sdd=float(np.exp(rng.uniform(np.log(800),np.log(5000))));pixel=float(rng.choice([.075,.1,.172,.2]));exposure=float(rng.choice([.022,.05,.1,.22,.55]))
    k=2*np.pi/(1239.841984/E);expected=.025/k*np.radians(.01)/(pixel/sdd)**2
    expected_atomic=expected*np.diff(NATIVE_EDGES)/.01
    mu=np.broadcast_to(expected_atomic[None,:,None],(2,len(NATIVE_AF),len(Q)))
    N=(np.floor(mu)+(rng.random(mu.shape)<mu-np.floor(mu))).astype(np.int32)
    valid=rng.random(N.shape)>.01
    for side in range(2):
        if rng.random()<.5:
            i=int(rng.integers(2,len(Q)-2));valid[side,:,i:i+2]=False
        if rng.random()<.25:
            i=int(rng.integers(0,len(NATIVE_AF)-3));valid[side,i:i+3]=False
    N*=valid;clip=0.;amp=0.
    if residual is not None:
        amp=float(rng.uniform(0.,1.));mean,clip=residual_rates(mean,residual,amp)
        pre,_=residual_rates(pre,residual,float(rng.uniform(0.,1.)))
    Y=rng.poisson(mean[None,:,:]*N*exposure);RY=rng.poisson(pre[None,:,:]*N*exposure)
    hn=N.astype(float)*exposure
    hy,hden,vy,vn,image,scale=counts_profiles(Y,hn);py,pn,pvy,pvn,preim,prescale=counts_profiles(RY,hn)

    rate=np.expm1(preim[0])*prescale;pt=np.stack([np.log1p(rate/scale)*preim[1],preim[1]]).astype(np.float32)
    desc=describe(hy,hden,vy,vn)
    for b in range(4):
        a=np.sum(hy[2*b:2*b+2]);z=np.sum(py[2*b:2*b+2])
        if a>0 and z>0:desc['features'][13+b]=np.log(a/z);desc['features'][33+b]=1
    measurement=np.array([np.log(max(scale,1e-6)),np.log(max(float(np.median(hn[hn>0])) if np.any(hn>0) else 0.,1e-6))],np.float32)
    return {'image':image,'pre':pt,'features':desc['features'],'measurement':measurement,'visible_peak':desc['visible_peak'],'Y':hy.astype(np.float32),'N':hden.astype(np.float32),'VY':vy.astype(np.float32),'VN':vn.astype(np.float32),
            'meta':{'instrument':{'sdd_mm':sdd,'pixel_mm':pixel,'exposure_s':exposure,'expected_pixel_multiplicity_per_01deg_cell_per_side':expected,'representation':'Jacobian virtual angular histogram with stochastic integer multiplicities; not exact physical Pilatus gap map','atomic_AF_edges':'union native.01/modelAF/band boundaries; integercounts neverfractionallysplit'},'residual_amplitude':amp,'residual_negative_rate_clipped_fraction':clip,'support':desc['support'],'q_star':desc['q_star'],'width':desc['width'],'Yoneda':desc['Yoneda']}}
