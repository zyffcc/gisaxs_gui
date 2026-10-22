"""Own-occupancy graded optical reference for additive surface islands.

Research candidate, not a certified inverse or a coalesced geometry. Mean metal
occupancy and the complex particle amplitudes use the same radius marks. The
uniform lateral mean has zero non-DC transform, so the scattering contrast is
the full metal contrast. No second subtraction of the optical reference.
"""
import numpy as np
from .optics import nd,wavelength_nm
from .forward import _stack
from .multilayer import amplitudes
from .formfactor import cap_profile,hemiellipsoid_profile,_gauss_legendre,jinc,volume
from .surface_separable import ssca_grid

class OccupancyError(ValueError):
    def __init__(self,fraction):
        self.fraction=float(fraction);super().__init__('Additive mean occupancy exceeds unity: '+str(fraction))

def primitive_volume_below(a,h,z,shape):
    z=np.minimum(np.maximum(np.asarray(z),0),h)
    if shape=='cap':
        center=h-(a*a+h*h)/(2*h)
        return np.pi*(a*a*z+center*z*z-z**3/3)
    return np.pi*a*a*(z-z**3/(3*h*h))

def mean_stack(particle,nbins=28,ndepth=64):
    """Exact cell means, upward edges; no clipping/renormalising material.

    Low-probability tall marks use a logarithmic outer mesh, still included in
    both optics and scattering. This mesh needs independent depth refinement.
    """
    state=particle.state()
    if state.film_thickness or state.infil_mass:raise ValueError('Surface-only graded islands required')
    R,H,w=state.size_bins(nbins);top=float(H.max());dense=min(top,4*particle.height_nm)
    edges=np.unique(np.r_[np.linspace(0,dense,ndepth+1),np.geomspace(dense,top,max(2,ndepth//4+1)),H])
    widths=np.diff(edges);cell_volume=np.zeros(len(widths));candidates=list(edges)
    # Within each interval r(z)^2 is quadratic. Check its exact stationary
    # point as well as every mark-height boundary, not merely cell averages.
    for lo,hi in zip(edges[:-1],edges[1:]):
        live=H>(lo+hi)/2
        if state.shape=='cap':
            b=H-(R*R+H*H)/(2*H);den=float(w[live].sum())
            peak=float(w[live]@b[live]/den) if den else -1
        else:peak=0
        if lo<peak<hi:candidates.append(peak)
    zz=np.array(candidates);point=np.zeros(len(zz))
    profile=cap_profile if state.shape=='cap' else hemiellipsoid_profile
    for a,h,ww in zip(R,H,w):
        lo=np.minimum(edges[:-1],h);hi=np.minimum(edges[1:],h);delta=hi-lo
        # Two positive Gauss nodes integrate the quadratic area EXACTLY.
        # Factored tip coordinates avoid subtracting large antiderivatives
        # in an arbitrarily thin interval after a finite-difference height move.
        t2,w2=_gauss_legendre(2);z=lo[:,None]+delta[:,None]*t2;s=(h-hi[:,None])+delta[:,None]*(1-t2)
        area=s*(a*a/h+z) if state.shape=='cap' else a*a*s*(h+z)/(h*h)
        dv=np.pi*delta*(area@w2)
        cell_volume+=ww*dv
        inside=zz<h;point[inside]+=ww*np.pi*profile(a,h,zz[inside]/h)**2
    phi_max=float(state.n_islands*point.max())
    if not np.isfinite(phi_max) or phi_max>1:raise OccupancyError(phi_max)
    phi=state.n_islands*cell_volume/widths
    if np.any(phi<0) or not np.isfinite(phi).all():raise ValueError('Invalid exact cell mean')
    return {'edges_up_nm':edges,'phi_bottom_up':phi,'max_fraction':phi_max,'R':R,'H':H,'w':w,'n':state.n_islands,'shape':state.shape,'mass_integral_nm':float(phi@widths),'primitive_mass_nm':float(state.n_islands*np.sum(w*volume(R,H,state.shape)))}

def complex_graded(sample,particle,q_nm_inv,alpha_f_deg,nbins=28,ndepth=64,order=4,mean_scale=1.):
    q=np.atleast_1d(q_nm_inv);af=np.atleast_1d(alpha_f_deg)
    if not np.isfinite(q).all() or not np.isfinite(af).all() or np.any(q<0) or np.any(af<=0) or not 0<=mean_scale<=1:raise ValueError('Finite positive exit angles/q and mean scale required')
    stack=mean_stack(particle,nbins,ndepth);edges=stack['edges_up_nm'];dz=np.diff(edges);phi=stack['phi_bottom_up'];k=2*np.pi/wavelength_nm(sample.energy_ev);m=nd(sample.metal,energy_ev=sample.energy_ev)
    base,th,sig,_,_=_stack(sample,None,nbins)
    means=(mean_scale*m*phi)[::-1];nds=[0j]+means.tolist()+base[1:];thick=dz[::-1].tolist()+th;sigma=[0.]* (1+len(phi))+sig[1:]
    inc=amplitudes(k,np.radians([sample.alpha_i_deg]),nds,thick,sigma);rec=amplitudes(k,np.radians(af),nds,thick,sigma)
    R,H,w=stack['R'],stack['H'],stack['w'];profile=cap_profile if stack['shape']=='cap' else hemiellipsoid_profile
    F=np.zeros((len(R),len(af),len(q)),complex);t,weights=_gauss_legendre(order)
    for mark,(a,h) in enumerate(zip(R,H)):
        live=np.flatnonzero(edges[:-1]<h);lo=edges[live];hi=np.minimum(edges[live+1],h);z=lo[:,None]+(hi-lo)[:,None]*t
        rad=profile(a,h,z/h);B=(weights[None,:]*(hi-lo)[:,None]*np.pi*rad*rad).ravel()[:,None]*jinc(rad.ravel()[:,None]*q)
        # Convert upward z to positive-down distance within its own cell.
        layer=1+len(phi)-1-live;distance=(edges[live+1,None]-z).ravel();indices=np.repeat(layer,order)
        ki=np.array([inc['kz'][i][0] for i in indices]);Di=np.array([inc['T'][i][0] for i in indices]);Ui=np.array([inc['R'][i][0] for i in indices]);Ei=(Di+Ui)*np.cos(ki*distance)+1j*ki*(Di-Ui)*distance*np.sinc(ki*distance/np.pi)
        ko=np.array([rec['kz'][i] for i in indices]).T;Do=np.array([rec['T'][i] for i in indices]).T;Uo=np.array([rec['R'][i] for i in indices]).T
        Ef=(Do+Uo)*np.cos(ko*distance)+1j*ko*(Do-Uo)*distance*np.sinc(ko*distance/np.pi)
        F[mark]=(Ef*Ei[None,:])@B
    chi=-2*m.real+2j*m.imag;pref=(k*k/(4*np.pi))**2*abs(chi)**2*stack['n']
    if not np.isfinite(F).all():raise ValueError('Nonfinite graded amplitudes')
    return F,R,w,pref,stack

def intensity(sample,particle,q_nm_inv,alpha_f_deg,nbins=28,ndepth=64,order=4,mean_scale=1.):
    F,R,w,pref,stack=complex_graded(sample,particle,q_nm_inv,alpha_f_deg,nbins,ndepth,order,mean_scale)
    power=ssca_grid(F,R,w,np.atleast_1d(q_nm_inv),particle.D_nm,particle.D_nm*particle.omega_rel,particle.kappa)
    if not np.isfinite(power).all() or power.min()<-1e-7*max(float(abs(power).max()),1e-30):raise ValueError('Graded SSCA covariance positivity failure')
    return pref*np.maximum(power,0),stack
