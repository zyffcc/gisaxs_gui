'Exact unique-AF optical/phase reuse; inherited quadrature is unchanged.'
import numpy as np
from . import optics
from .forward import _stack,simulate
from .multilayer import amplitudes,fields_at_bottom
from .formfactor import cap_profile,hemiellipsoid_profile
from .forward_v06_gpu import gl
from .bessel_cuda import j1
from .tool_operator_v11 import AdaptiveOperator

def complex_amplitudes_cuda(sample,state,ang,nbins=448,nz=192,chunk_pixels=8192,kzf_bound=None):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    device='cuda';af=np.asarray(ang['alpha_f']).ravel();q=np.asarray(ang['qpar']).ravel()
    unique,rev=np.unique(af,return_inverse=True)
    k=2*np.pi/optics.wavelength_nm(sample.energy_ev);nd,thick,sig,rough,iz=_stack(sample,state,nbins)
    if iz is None:raise ValueError('No particle zone')
    inc=amplitudes(k,np.array([np.radians(sample.alpha_i_deg)]),nd,thick,sig)
    ex=amplitudes(k,np.maximum(unique,1e-6),nd,thick,sig)
    Di,Ui=fields_at_bottom(inc,iz);Df,Uf=fields_at_bottom(ex,iz)
    Di,Ui=complex(Di[0]),complex(Ui[0]);kzi=complex(inc['kz'][iz][0]);kzf=ex['kz'][iz]
    R,H,p=state.size_bins(nbins);profile=cap_profile if state.shape=='cap' else hemiellipsoid_profile
    qq=torch.as_tensor(q,device=device,dtype=torch.float64);fz=torch.as_tensor(kzf,device=device,dtype=torch.complex128)
    df=torch.as_tensor(Df,device=device,dtype=torch.complex128);uf=torch.as_tensor(Uf,device=device,dtype=torch.complex128)
    ids=torch.as_tensor(rev,device=device,dtype=torch.int64);F=torch.empty(len(R),len(q),device=device,dtype=torch.complex128)
    with torch.no_grad():
        for j,(a,h) in enumerate(zip(R,H)):
            bound=np.max(abs(kzf)) if kzf_bound is None else float(kzf_bound)
            n=max(nz,int(np.ceil(.75*(abs(kzi)+bound)*h))+12);t,w=gl(n);rad=profile(a,h,t)
            zz=torch.as_tensor(t*h,device=device,dtype=torch.float64);rr=torch.as_tensor(rad,device=device,dtype=torch.float64)
            ww=torch.as_tensor(w*h*np.pi*rad*rad,device=device,dtype=torch.float64)
            ei=torch.exp(-1j*kzi*zz);eiu=torch.exp(1j*kzi*zz)
            ef=torch.exp(-1j*zz[:,None]*fz[None,:]);efu=torch.exp(1j*zz[:,None]*fz[None,:])

            e1=ef*ei[:,None];e2=ef*eiu[:,None];e3=efu*ei[:,None];e4=efu*eiu[:,None]
            for lo in range(0,len(q),chunk_pixels):
                hi=min(lo+chunk_pixels,len(q));ii=ids[lo:hi]
                x=rr[:,None]*qq[None,lo:hi];jinc=torch.where(abs(x)<1e-7,1-x*x/8,2*j1(x)/x);A=ww[:,None]*jinc
                F[j,lo:hi]=Di*df[ii]*torch.sum(A*e1[:,ii],dim=0)+Ui*df[ii]*torch.sum(A*e2[:,ii],dim=0)+Di*uf[ii]*torch.sum(A*e3[:,ii],dim=0)+Ui*uf[ii]*torch.sum(A*e4[:,ii],dim=0)
        F[:,torch.as_tensor(af<=0,device=device)]=0
    mnd=optics.nd(sample.metal,energy_ev=sample.energy_ev);dn2=-2*(mnd-nd[iz]).real+2j*(mnd-nd[iz]).imag
    return F,R,p,(k*k/(4*np.pi))**2*abs(dn2)**2*state.n_islands

class Stage5Operator(AdaptiveOperator):
    def __init__(self,x,material,shape,metal,E,ai,order=4,nbins=448,nz=192,backend='cuda'):
        if backend!='cuda':
            super().__init__(x,material,shape,metal,E,ai,order,nbins,nz,backend);return
        from .tool_prior import physical
        from .tool_qinput import angles
        from .cell_response import adaptive_nodes
        self.x=np.asarray(x,float).copy();self.material=material;self.shape=shape;self.metal=metal;self.E=E;self.ai=ai;self.order=order;self.nbins=nbins;self.nz=nz;self.backend=backend
        s,p=physical(x,material,shape,metal,E,ai);self.sample=s;self.particle=p
        s._measurement_af_shift=x[19];s._measurement_qscale=x[18]
        nd,_,_,_,_=_stack(s,p.state(),nbins);self.ac,self.aw,self.qc,self.qw,self.rule_meta=adaptive_nodes(s,p,nd,order)
        self.ang=angles(E,ai,x[18],x[19],q_centres=self.qc,af_centres=self.ac)
        self.F,self.R,self.weights,self.pref=complex_amplitudes_cuda(s,p.state(),self.ang,nbins,nz)
        self.base=simulate(s,None,self.ang)['total'];self.base=np.where(self.ang['alpha_f']>0,self.base,0.)
