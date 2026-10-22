"""v0.6 explicit complex SSCA on a layered optical reference.

Independent from the historical DA sampler; labels describe the actual
particle geometry. Interlayer optical variables remain nuisance parameters.
This additive particle/film model has no coalescence or percolation topology.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict,replace
import numpy as np
from scipy.special import ndtri,ndtr
from scipy.ndimage import gaussian_filter
from . import optics
from .forward import SampleParams,MorphState,simulate,_stack
from .multilayer import amplitudes,fields_at_bottom
from .formfactor import cap_profile,hemiellipsoid_profile,_gauss_legendre,jinc,volume

@dataclass
class IntegratedState(MorphState):
    def size_bins(self,nbins=31):
        # Fixed normal-coordinate domain [-8,8], composite Gauss-Legendre.
        # Even at sigma=.45 the sixth-moment tail omitted is <6e-8.
        edges=np.array([-8.,-5.,-3.,-1.,1.,3.,5.,8.]);n=max(4,int(np.ceil(nbins/7)));t,w=_gauss_legendre(n)
        z=np.concatenate([lo+(hi-lo)*t for lo,hi in zip(edges[:-1],edges[1:])]);pr=np.concatenate([(hi-lo)*w for lo,hi in zip(edges[:-1],edges[1:])])*np.exp(-.5*z*z)/np.sqrt(2*np.pi);pr/=pr.sum();a=self.a_med*np.exp(self.s_log*z)
        return a,self.aspect*a,pr

@dataclass
class SSCAParams:
    D_nm: float=12.
    base_radius_nm: float=2.5
    height_nm: float=1.625
    sigma_log: float=.25
    omega_rel: float=.2
    kappa: float=1.
    shape: str='cap'
    film_thickness_nm: float=0.
    average_material: bool=False
    def state(self):
        if min(self.D_nm,self.base_radius_nm,self.height_nm,self.omega_rel)<=0:raise ValueError('Nonpositive physical scale')
        if self.shape not in ('cap','hemiellipsoid') or not (0<=self.kappa<=1) or not (0.03<=self.sigma_log<=.45):raise ValueError('Outside declared v0.6 shape/spread/coupling domain')
        aspect=self.height_nm/self.base_radius_nm
        if not .1<=aspect<=2:raise ValueError('Outside declared v0.6 h/a domain')
        return IntegratedState(n_islands=2/(np.sqrt(3)*self.D_nm**2),a_med=self.base_radius_nm,aspect=aspect,s_log=self.sigma_log,omega_rel=self.omega_rel,shape=self.shape,film_thickness=self.film_thickness_nm,use_avg_material=self.average_material)
    def diameter(self):
        a,h=self.base_radius_nm,self.height_nm
        return 2*((a*a+h*h)/(2*h) if self.shape=='cap' and h>a else a)

def complex_amplitudes(sample,state,ang,nbins=224,nz=128):
    af=np.asarray(ang['alpha_f']).ravel();q=np.asarray(ang['qpar']).ravel();k=2*np.pi/optics.wavelength_nm(sample.energy_ev)
    nd,thick,sig,rough,iz=_stack(sample,state,nbins)
    if iz is None:raise ValueError('Particle complex amplitudes require island zone')
    inc=amplitudes(k,np.array([np.radians(sample.alpha_i_deg)]),nd,thick,sig);ex=amplitudes(k,np.maximum(af,1e-6),nd,thick,sig)
    Di,Ui=fields_at_bottom(inc,iz);Df,Uf=fields_at_bottom(ex,iz);kzi=inc['kz'][iz][0];kzf=ex['kz'][iz]
    sizes,heights,weights=state.size_bins(nbins);profile=cap_profile if state.shape=='cap' else hemiellipsoid_profile;F=[]
    for a,h in zip(sizes,heights):
        t,w=_gauss_legendre(max(nz,int(np.ceil(.75*(abs(kzi)+np.max(abs(kzf)))*h))+12))
        rad=profile(a,h,t);z=t*h;A=(w*h*np.pi*rad*rad)[:,None]*jinc(np.outer(rad,q));Ef=np.exp(-1j*np.outer(z,kzf));ei=np.exp(-1j*kzi*z)
        # Separate subtractions/exponents avoid overflow from reciprocal arrays.
        Efup=np.exp(1j*np.outer(z,kzf));eiup=np.exp(1j*kzi*z)
        f=Di*Df*np.einsum('kp,kp,k->p',A,Ef,ei)+Ui*Df*np.einsum('kp,kp,k->p',A,Ef,eiup)+Di*Uf*np.einsum('kp,kp,k->p',A,Efup,ei)+Ui*Uf*np.einsum('kp,kp,k->p',A,Efup,eiup)
        f[af<=0]=0;F.append(f)
    mnd=optics.nd(sample.metal,energy_ev=sample.energy_ev);dn2=-2*(mnd-nd[iz]).real+2j*(mnd-nd[iz]).imag
    return np.array(F),sizes,weights,(k*k/(4*np.pi))**2*abs(dn2)**2*state.n_islands

def ssca_intensity(F,radii,weights,q,D,omega,kappa):
    M=np.einsum('j,jk->k',weights,abs(F)**2);dr=radii-float(weights@radii);phase=np.exp(1j*kappa*dr[:,None]*q[None,:]);p0=np.exp(1j*q*D-.5*(q*omega)**2-D/1e4)
    p2=weights@(phase**2);fp=np.einsum('j,jk->k',weights,F*phase);fm=np.einsum('j,jk->k',weights,F/phase)
    return M+2*np.real(p0*fp*np.conj(fm)/(1-p0*p2))

def simulate_v06(sample,particle,ang,*,nbins=224,nz=128,resolution_bins=0.,background=None,continuous_film_nm=0.):
    """Physical components first; optional phenomenological background and PSF.

    Returns flags for approximate projected overlap/filling. A positive total
    is not proof of independent islands or four-stage topology.
    """
    if particle is None:
        state=MorphState(n_islands=0.,film_thickness=continuous_film_nm) if continuous_film_nm>0 else None
        base=simulate(sample,state,ang,resolution_bins=0.);Ip=np.zeros_like(base['total']);flags={'island_parameters_defined':False,'topological_stage':'continuous_film_simulation' if continuous_film_nm>0 else 'undefined'}
    else:
        state=particle.state();F,R,p,pref=complex_amplitudes(sample,state,ang,nbins,nz);raw=ssca_intensity(F,R,p,np.asarray(ang['qpar']).ravel(),particle.D_nm,particle.omega_rel*particle.D_nm,particle.kappa)
        if not np.all(np.isfinite(raw)) or raw.min()<-1e-7*max(abs(raw).max(),1e-30):raise ValueError('SSCA intensity numerical/physical positivity failure')
        Ip=(pref*np.maximum(raw,0)).reshape(np.shape(ang['alpha_f']))
        # Bare particle-zone is transparent; removing it does not alter interface
        # intensity. For averaged material retain the identical stack via old base.
        bare=state if particle.average_material else replace(state,n_islands=0.)
        base=simulate(sample,bare,ang,nbins=nbins,nz=nz,resolution_bins=0.)
        rad=R*((1+state.aspect**2)/(2*state.aspect) if state.shape=='cap' and state.aspect>1 else 1.)
        meanR=float(p@rad);area=state.n_islands*np.pi*float(p@(rad*rad));heightmax=float((state.aspect*R).max());mass=float(state.n_islands*np.sum(p*volume(R,state.aspect*R,state.shape))+state.film_thickness*state.film_density)
        flags={'island_parameters_defined':True,'D_definition':'radialGaussianconditionalmeanstep with densitymapped2/(sqrt3D²),notcertified2DNN','diameter_definition':'medianparticlemaximumlateraldiameter','projected_area_sum':area,'Gaussian_mean_pair_overlap_probability':float(ndtr((2*meanR-particle.D_nm)/(particle.omega_rel*particle.D_nm))),'mean_zone_volume_fraction':mass/max(heightmax,1e-30),'average_layer_clamped':bool(particle.average_material and mass/heightmax>=.95),'topological_stage':'not_generated','coalescence_percolation_groundtruth':False}
    rough=np.maximum(base['roughness'],0);total=Ip+rough
    if background:
        q=np.asarray(ang['qpar']);qt=np.asarray(ang.get('q_total',np.sqrt(q*q+np.asarray(ang['qz'])**2)))
        total=total+background.get('flat',0.)+background.get('db_amp',0.)/(1+(q*background.get('db_xi_nm',3.))**2)**2+background.get('ring_amp',0.)*np.exp(-.5*((qt-background.get('ring_q_nm_inv',2.1))/background.get('ring_sigma',.06))**2)
    if np.max(np.atleast_1d(resolution_bins))>0 and total.ndim==2:total=gaussian_filter(total,resolution_bins,mode='nearest')
    return {'particles':Ip,'roughness':rough,'total':total,'flags':flags,'sample':sample.to_dict(),'particle':asdict(particle) if particle else None,'background':dict(background or {}),'resolution_bins':np.atleast_1d(resolution_bins).tolist(),'continuous_film_nm':continuous_film_nm}

def draw_model(rng,energy_ev=12000.,alpha_i_deg=.4):
    """Independent v0.6 prior; no legacy shape-specific mass/stage labels."""
    ll=lambda lo,hi:float(np.exp(rng.uniform(np.log(lo),np.log(hi))))
    kind=str(rng.choice(['none','dense_oxide','porous_oxide','polymer']));sub='Si' if kind=='none' else str(rng.choice(['SiO2','TiO2'])) if kind=='dense_oxide' else 'TiO2' if kind=='porous_oxide' else str(rng.choice(['PS','PMMA']))
    density=1. if kind=='none' else ll(.6,1.8) if kind=='dense_oxide' else ll(.08,.95) if kind=='porous_oxide' else ll(.5,1.5)
    sample=SampleParams(metal=str(rng.choice(['Au','Ag'])),substrate=sub,layer_thickness=0. if kind=='none' else ll(.5,200),layer_density=density,sigma_top=ll(.02,2),sigma_si=ll(.02,.6),xi_top=ll(1,100),xi_si=ll(3,100),pore_xi=ll(.5,20),pore_gain=ll(.03,20),alpha_i_deg=alpha_i_deg,energy_ev=energy_ev)
    # Radius >8 nm has no full-domain integration receipt yet. Keep this a
    # declared prior restriction, not an undocumented numerical rejection.
    D=ll(3,100);a=ll(.4,min(8,D*.55));aspect=ll(.1,2.)
    particle=SSCAParams(D_nm=D,base_radius_nm=a,height_nm=a*aspect,sigma_log=float(rng.uniform(.03,.45)),omega_rel=ll(.06,.6),kappa=float(rng.uniform(0,1)),shape=str(rng.choice(['cap','hemiellipsoid'])))
    return sample,particle,{'layer_kind':kind,'regime_label':'unvalidated_independent_particle_family','coalescence_percolation_generated':False}
