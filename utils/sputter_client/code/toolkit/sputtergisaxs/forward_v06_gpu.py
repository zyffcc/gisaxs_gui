'Optional float64/complex128 CUDA evaluator of the same v0.6 operator.'
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre
from . import optics
from .forward import _stack
from .multilayer import amplitudes,fields_at_bottom
from .formfactor import cap_profile,hemiellipsoid_profile
from .bessel_cuda import j1

@lru_cache(maxsize=512)
def gl(n):
    t,w=roots_legendre(n);return (t+1)/2,w/2

def complex_amplitudes_cuda(sample,state,ang,nbins=448,nz=192,chunk_pixels=8192):
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for this optional evaluator')
    device='cuda';af=np.asarray(ang['alpha_f']).ravel();q=np.asarray(ang['qpar']).ravel();k=2*np.pi/optics.wavelength_nm(sample.energy_ev);nd,thick,sig,rough,iz=_stack(sample,state,nbins)
    if iz is None:raise ValueError('No particle zone')
    inc=amplitudes(k,np.array([np.radians(sample.alpha_i_deg)]),nd,thick,sig);ex=amplitudes(k,np.maximum(af,1e-6),nd,thick,sig);Di,Ui=fields_at_bottom(inc,iz);Df,Uf=fields_at_bottom(ex,iz);Di,Ui=complex(Di[0]),complex(Ui[0]);kzi=complex(inc['kz'][iz][0]);kzf=ex['kz'][iz];R,H,p=state.size_bins(nbins);profile=cap_profile if state.shape=='cap' else hemiellipsoid_profile
    qq=torch.as_tensor(q,device=device,dtype=torch.float64);fz=torch.as_tensor(kzf,device=device,dtype=torch.complex128);df=torch.as_tensor(Df,device=device,dtype=torch.complex128);uf=torch.as_tensor(Uf,device=device,dtype=torch.complex128);F=torch.empty(len(R),len(q),device=device,dtype=torch.complex128)
    with torch.no_grad():
        for j,(a,h) in enumerate(zip(R,H)):
            n=max(nz,int(np.ceil(.75*(abs(kzi)+np.max(abs(kzf)))*h))+12);t,w=gl(n);rad=profile(a,h,t);zz=torch.as_tensor(t*h,device=device,dtype=torch.float64);rr=torch.as_tensor(rad,device=device,dtype=torch.float64);ww=torch.as_tensor(w*h*np.pi*rad*rad,device=device,dtype=torch.float64);ei=torch.exp(-1j*kzi*zz);eiu=torch.exp(1j*kzi*zz)
            for lo in range(0,len(q),chunk_pixels):
                hi=min(lo+chunk_pixels,len(q));x=rr[:,None]*qq[None,lo:hi];jinc=torch.where(abs(x)<1e-7,1-x*x/8,2*j1(x)/x);A=ww[:,None]*jinc;ef=torch.exp(-1j*zz[:,None]*fz[None,lo:hi]);efu=torch.exp(1j*zz[:,None]*fz[None,lo:hi]);F[j,lo:hi]=Di*df[lo:hi]*torch.sum(A*ef*ei[:,None],dim=0)+Ui*df[lo:hi]*torch.sum(A*ef*eiu[:,None],dim=0)+Di*uf[lo:hi]*torch.sum(A*efu*ei[:,None],dim=0)+Ui*uf[lo:hi]*torch.sum(A*efu*eiu[:,None],dim=0)
        F[:,torch.as_tensor(af<=0,device=device)]=0
    mnd=optics.nd(sample.metal,energy_ev=sample.energy_ev);dn2=-2*(mnd-nd[iz]).real+2j*(mnd-nd[iz]).imag
    return F,R,p,(k*k/(4*np.pi))**2*abs(dn2)**2*state.n_islands

def ssca_cuda(F,R,p,q,D,omega,kappa,chunk_pixels=8192):
    import torch
    with torch.no_grad():
        weights=torch.as_tensor(p,device=F.device,dtype=torch.float64);dr=torch.as_tensor(R-float(p@R),device=F.device,dtype=torch.float64);qq=torch.as_tensor(q,device=F.device,dtype=torch.float64);out=torch.empty(len(q),device=F.device,dtype=torch.float64)
        for lo in range(0,len(q),chunk_pixels):
            hi=min(lo+chunk_pixels,len(q));qc=qq[lo:hi];fc=F[:,lo:hi];phase=torch.exp(1j*kappa*dr[:,None]*qc[None,:]);M=torch.sum(weights[:,None]*abs(fc)**2,dim=0);p2=torch.sum(weights[:,None]*phase**2,dim=0);fp=torch.sum(weights[:,None]*fc*phase,dim=0);fm=torch.sum(weights[:,None]*fc*torch.conj(phase),dim=0);p0=torch.exp(1j*qc*D-.5*(qc*omega)**2-D/1e4);out[lo:hi]=M+2*torch.real(p0*fp*torch.conj(fm)/(1-p0*p2))
        return out.cpu().numpy()
