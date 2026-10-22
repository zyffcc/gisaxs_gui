"""Exact separable flat-surface particle DWBA, transparent island reference.

Research acceleration of v0.6 SSCA, not a new calibrated inverse model. The
fields are referenced to the substrate top (z UP); a transparent particle-zone
phase common to all radius marks cancels in SSCA. Mean metal optical feedback,
films, infiltration, rough 3D hosts and coalescence are outside this operator.
"""
import numpy as np
from .optics import nd,wavelength_nm
from .forward import _stack
from .forward_v06 import ssca_intensity
from .multilayer import amplitudes
from .formfactor import cap_profile,hemiellipsoid_profile,_gauss_legendre,jinc

def complex_surface(sample,particle,q_nm_inv,alpha_f_deg,nbins=28,nz=32):
    state=particle.state()
    if state.use_avg_material or state.film_thickness or state.infil_mass:raise ValueError('Separable surface operator needs a transparent island reference without film/infiltration')
    q=np.atleast_1d(q_nm_inv);af=np.atleast_1d(alpha_f_deg)
    if not np.all(np.isfinite(q)) or not np.all(np.isfinite(af)) or np.any(q<0) or np.any(af<=0):raise ValueError('Unsupported q/exit angles')
    k=2*np.pi/wavelength_nm(sample.energy_ev);ns,th,sig,_,_=_stack(sample,None,nbins)
    inc=amplitudes(k,np.radians([sample.alpha_i_deg]),ns,th,sig);rec=amplitudes(k,np.radians(af),ns,th,sig)
    ki=inc['kz'][0][0];kf=rec['kz'][0];Di=inc['T'][0][0];Ui=inc['R'][0][0];Df=rec['T'][0];Uf=rec['R'][0]
    R,H,w=state.size_bins(nbins);profile=cap_profile if state.shape=='cap' else hemiellipsoid_profile
    F=np.empty((len(R),len(af),len(q)),complex)
    for j,(a,h) in enumerate(zip(R,H)):
        t,weights=_gauss_legendre(max(nz,int(np.ceil(.75*(abs(ki)+np.max(abs(kf)))*h))+12));z=t*h;r=profile(a,h,t)
        B=(weights*h*np.pi*r*r)[:,None]*jinc(r[:,None]*q)
        Ei=Di*np.exp(-1j*ki*z)+Ui*np.exp(1j*ki*z)
        Ef=Df[:,None]*np.exp(-1j*kf[:,None]*z)+Uf[:,None]*np.exp(1j*kf[:,None]*z)
        F[j]=(Ef*Ei[None,:])@B
    metal=nd(sample.metal,energy_ev=sample.energy_ev);chi=-2*metal.real+2j*metal.imag
    pref=(k*k/(4*np.pi))**2*abs(chi)**2*state.n_islands
    if not np.all(np.isfinite(F)):raise ValueError('Nonfinite surface amplitudes')
    return F,R,w,pref

def intensity(sample,particle,q_nm_inv,alpha_f_deg,nbins=28,nz=32):
    F,R,w,pref=complex_surface(sample,particle,q_nm_inv,alpha_f_deg,nbins,nz)
    out=pref*ssca_grid(F,R,w,np.asarray(q_nm_inv),particle.D_nm,particle.omega_rel*particle.D_nm,particle.kappa)
    if not np.all(np.isfinite(out)) or out.min()<-1e-7*max(float(abs(out).max()),1e-30):raise ValueError('SSCA surface covariance positivity failure')
    return np.maximum(out,0).reshape(F.shape[1:])

def ssca_grid(F,radii,weights,q,D,omega,kappa):
    """Same v0.6 SSCA on an AF x q grid without duplicating radius phases.

    Raw covariance is returned, before any positivity tolerance/roundoff clip.
    D is the original conditional radial step, not a measured 2D neighbour.
    """
    F=np.asarray(F);q=np.atleast_1d(q);r=np.asarray(radii);w=np.asarray(weights)
    if F.ndim!=3 or F.shape[0]!=len(r) or F.shape[2]!=len(q) or w.shape!=r.shape:raise ValueError('SSCA Cartesian dimensions disagree')
    M=np.einsum('r,raq->aq',w,abs(F)**2);phase=np.exp(1j*kappa*(r-float(w@r))[:,None]*q[None,:])
    p2=w@(phase**2);p0=np.exp(1j*q*D-.5*(q*omega)**2-D/1e4)
    # Loop over marks to preserve F/phase and bound peak temporary memory.
    fp=np.zeros(F.shape[1:],complex);fm=np.zeros_like(fp)
    for j in range(len(r)):
        fp+=w[j]*F[j]*phase[j];fm+=w[j]*F[j]/phase[j]
    return M+2*np.real(p0[None,:]*fp*np.conj(fm)/(1-p0*p2)[None,:])
