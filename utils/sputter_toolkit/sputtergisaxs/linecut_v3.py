"""Conditional horizontal GISAXS line fits, independent of frozen v2 networks.

Uses original nonoverlapping photon histograms, not the resized network image.
R is the NUMBER mean of a lognormal radius distribution. D is the Gaussian
1D-paracrystal distance, not automatically 2*pi/q_peak or a 2D density.
Cylinder: common unknown height (separable vertical factor); hemisphere: h=R.
Reference: Lazzari, IsGISAXS manual, sections1.2--1.5. See V3_METHOD.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
from scipy.special import j0,j1
from .features import dev_residual, fit_peak, ridge, AF_BANDS, V_EDGES
from .formfactor import jinc
from .interference import radial_paracrystal
from .tool_qinput import Q, Q_EDGES, AF_EDGES

# Exact existing native bin definition, without importing the GPU generator.
NATIVE_EDGES=np.unique(np.round(np.r_[np.arange(.02,1.50001,.01),AF_EDGES,
                                     V_EDGES,np.array(AF_BANDS).ravel()],10))
NATIVE_AF=(NATIVE_EDGES[:-1]+NATIVE_EDGES[1:])/2


@lru_cache(None)
def quadrature(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


@lru_cache(None)
def size_rule(n):
    # Same composite normal-domain rule as IntegratedState, not equal-probability
    # bins. Sparse GH nodes undersample high-q oscillations for broad/large R.
    edges=np.array([-8.,-5.,-3.,-1.,1.,3.,5.,8.])
    u,v=quadrature(max(4,int(np.ceil(n/7))))
    z=np.concatenate([lo+(hi-lo)*u for lo,hi in zip(edges[:-1],edges[1:])])
    w=np.concatenate([(hi-lo)*v for lo,hi in zip(edges[:-1],edges[1:])])
    w*=np.exp(-.5*z*z)/np.sqrt(2*np.pi)
    return z,w/w.sum()


@dataclass
class Linecut:
    counts: np.ndarray             # side, alpha_f, q: raw photons, NOT rates
    exposure_pixels: np.ndarray    # sum(valid pixels * exposure seconds)
    af_deg: np.ndarray
    q_edges: np.ndarray
    energy_ev: float
    alpha_i_deg: float
    photon_counts: bool = True
    source: str = ''

    def __post_init__(self):
        self.counts = np.asarray(self.counts, float)
        self.exposure_pixels = np.asarray(self.exposure_pixels, float)
        self.af_deg = np.asarray(self.af_deg, float)
        self.q_edges = np.asarray(self.q_edges, float)
        if (self.counts.shape != self.exposure_pixels.shape or
                self.counts.shape != (2, len(self.af_deg), len(self.q_edges)-1)):
            raise ValueError('Expected two sides, exit-angle cells, q cells')
        if (not np.isfinite(self.counts).all() or np.any(self.counts < 0) or
                not np.isfinite(self.exposure_pixels).all() or np.any(self.exposure_pixels < 0)):
            raise ValueError('Nonnegative finite original counts/support required')
        if min(self.energy_ev, self.alpha_i_deg) <= 0 or not np.all(np.diff(self.q_edges)>0):
            raise ValueError('Positive geometry and increasing q edges required')
        if self.photon_counts and np.any(self.counts != np.rint(self.counts)):
            raise ValueError('Poisson inference requires integer original photons')

    @property
    def q(self):
        return (self.q_edges[:-1]+self.q_edges[1:])/2

    @property
    def Y(self):
        return self.counts.sum(axis=1)

    @property
    def E(self):
        return self.exposure_pixels.sum(axis=1)


def from_atomic(Y, E, af_band, energy_ev, alpha_i_deg, *, photon_counts=True, source=''):
    """Use complete native cells in one declared, scan-fixed alpha_f band.

    The actual boundary is outward-rounded to native cells, and returned.
    No splitting/reuse of one photon and no synthetic reconstruction of pixels.
    """
    Y, E = np.asarray(Y), np.asarray(E)
    if Y.shape != (2, len(NATIVE_AF), len(Q)) or E.shape != Y.shape:
        raise ValueError('Original native angular histogram required')
    lo, hi = map(float, af_band)
    if not 0 <= lo < hi <= 1.5:
        raise ValueError('Exit-angle band must be within0--1.5 degrees')
    keep = (NATIVE_EDGES[:-1] < hi-1e-9) & (NATIVE_EDGES[1:] > lo+1e-9)
    cut = Linecut(Y[:, keep], E[:, keep], NATIVE_AF[keep], Q_EDGES,
                  energy_ev, alpha_i_deg, photon_counts, source)
    ids = np.where(keep)[0]
    return cut, [float(NATIVE_EDGES[ids[0]]), float(NATIVE_EDGES[ids[-1]+1])]


def direct_features(Y, E):
    """Existing empirical peak/ridge plus an explicitly conditional high-q slope."""
    from .prediction import summarize_profiles
    from .resolvable import PEAK_WINDOW
    Y,E=np.asarray(Y,float),np.asarray(E,float)
    hy=[];hn=[]
    for lo,hi in AF_BANDS:
        sel=(NATIVE_AF>=lo)&(NATIVE_AF<hi)
        for side in range(2):
            hy.append(Y[side,sel].sum(axis=0));hn.append(E[side,sel].sum(axis=0))
    qs=(Q>=.15)&(Q<1.2)
    vi=np.searchsorted(V_EDGES,NATIVE_AF,side='right')-1
    ok=(vi>=0)&(vi<len(V_EDGES)-1)
    vy=np.bincount(vi[ok],weights=Y[:,:,qs].sum(axis=(0,2))[ok],minlength=len(V_EDGES)-1)
    vn=np.bincount(vi[ok],weights=E[:,:,qs].sum(axis=(0,2))[ok],minlength=len(V_EDGES)-1)
    fits, bands, selected, vr = summarize_profiles(hy,hn,vy,vn,PEAK_WINDOW)
    y, e = np.asarray(Y).sum(axis=(0,1)), np.asarray(E).sum(axis=(0,1))
    rate = np.divide(y,e,out=np.zeros_like(y,dtype=float),where=e>0)
    good = (Q>=1.0)&(Q<=1.75)&(y>0)&(e>0)
    slope = None
    if good.sum()>=8:
        slope = float(np.polyfit(np.log(Q[good]),np.log(rate[good]),1,
                                 w=np.sqrt(y[good]))[0])
    return dict(D_corr_nm=2*np.pi/selected['q_star'] if selected else None,
                q_star_nm_inv=selected['q_star'] if selected else None,
                FWHM_nm_inv=selected['FWHM'] if selected else None,
                feature_status=selected['support'] if selected else 'no_resolved_peak',
                feature_AF_band=selected['AF_band'] if selected else None,
                feature_AF_range_deg=selected['af_range_deg'] if selected else None,
                feature_source='existing broad-AF empirical estimator, distinct from narrow fitted linecut',
                Yoneda_deg=vr.get('alpha_f_observed_ridge'),
                Yoneda_status=vr['ridge_status'], highq_log_slope=slope,
                highq_status='available' if slope is not None else 'insufficient_positive_supported_bins',
                highq_reason='Poisson-weighted log-slope on total intensity, not a metal-only size' if slope is not None else 'At least8positive-count bins with nonzero exposure in1.0--1.75nm^-1 are required',
                highq_positive_supported_bins=int(good.sum()),
                highq_domain_nm_inv=[1.,1.75],
                highq_scope='apparent total-intensity slope, not a particle radius',
                peak_attribution='total intensity, host/metal not uniquely separated')


class Model:
    """Band/support integrated effective surface model.

    Four above-surface DWBA terms for a single effective interface. The
    critical angle is an INPUT nuisance, not a recovered interlayer density.
    Cylinder uses common unit height, so height is absorbed in amplitude.
    Hemisphere uses h=a. DA averages amplitudes; LMA_common_D sums domain
    intensities with common D. Neither is called size--D coupled SSCA.
    """
    def __init__(self, cut, shape='cylinder', approximation='DA',
                 critical_angle_deg=.16, sigma_q_nm_inv=0., nsize=448, nz=48,
                 cylinder_tangent=True, hemisphere_tangent=True,
                 side_response='shared'):
        if shape not in ('cylinder','hemisphere') or approximation not in ('DA','LMA_common_D'):
            raise ValueError('Unknown shape/decoupling approximation')
        if not 0<=critical_angle_deg<1 or sigma_q_nm_inv<0:
            raise ValueError('Invalid effective optical/resolution input')
        if side_response not in ('shared','independent'):
            raise ValueError('side_response must be shared or independent')
        self.cut,self.shape,self.approximation=cut,shape,approximation
        self.critical_angle_deg=float(critical_angle_deg)
        self.sigma_q_nm_inv=float(sigma_q_nm_inv)
        self.nsize,self.nz=nsize,nz
        self.cylinder_tangent=bool(cylinder_tangent and np.ptp(cut.af_deg)<=.12)
        self.hemisphere_tangent=bool(hemisphere_tangent and np.ptp(cut.af_deg)<=.12)
        self.side_response=side_response
        # Integrate finite q cells. AF cells retain measured per-q pixel support.
        u,w=quadrature(3)
        q = cut.q_edges[:-1,None]+np.diff(cut.q_edges)[:,None]*u
        self.qw=w
        if sigma_q_nm_inv:
            z,gw=np.polynomial.hermite.hermgauss(5)
            q=np.maximum(q[:,:,None]+np.sqrt(2)*sigma_q_nm_inv*z,1e-6)
            self.qw=(w[:,None]*gw[None,:]/np.sqrt(np.pi)).ravel()
            q=q.reshape(len(cut.q),-1)
        self.qy=q
        k=2*np.pi/(1239.841984/cut.energy_ev)
        ai=np.radians(cut.alpha_i_deg); af=np.radians(cut.af_deg)
        phi=np.arcsin(np.clip(q[None,:,:]/(k*np.cos(af)[:,None,None]),-1,1))
        qx=k*(np.cos(af)[:,None,None]*np.cos(phi)-np.cos(ai))
        self.qpar=np.sqrt(qx*qx+q[None,:,:]**2)
        ki=k*np.sin(ai); kf=k*np.sin(af)
        ac=np.radians(critical_angle_deg)
        # Tiny absorption regularises a critical point, explicitly fixed.
        delta=.5*ac**2; beta=max(delta*.02,1e-12)
        def refl(a):
            kz=k*np.sin(a);inside=k*np.sqrt(np.sin(a)**2-2*delta+2j*beta)
            return (kz-inside)/(kz+inside)
        ri=refl(ai); rf=refl(af)
        # Same downward/upward phase convention as forward.py (not the
        # conjugated form factor paired with unchanged complex reflectivity).
        self.kz=-np.stack([ki+kf,kf-ki,ki-kf,-ki-kf],axis=0)
        self.coeff=np.stack([np.ones_like(rf),np.full_like(rf,ri),rf,ri*rf])
        if shape=='cylinder':
            phase=np.exp(.5j*self.kz)*np.sinc(self.kz/(2*np.pi))
            self.vertical=np.sum(self.coeff*phase,axis=0)

    @lru_cache(maxsize=4)
    def _moments(self,R,sigma_log):
        # Numerical derivatives of A/B/C/p, D and omega must NOT rebuild the
        # expensive shape quadrature. These moments depend only on R/sigma
        # for this immutable geometry/model. Bounded cache, no approximation.
        z,w=size_rule(self.nsize)
        radii=R*np.exp(sigma_log*z-.5*sigma_log**2)
        if self.shape=='cylinder' and self.cylinder_tangent:
            # Within a narrow AF band q_parallel changes very little at fixed
            # q_y. Taylor the AMPLITUDE in this tiny delta, not in q_y itself;
            # keep first/second moments and all vertical/support factors.
            # The full original evaluation remains an explicit reference.
            qr=(self.qpar.min(axis=0)+self.qpar.max(axis=0))/2
            x=radii[:,None,None]*qr[None,:,:]
            deriv=np.empty_like(x);small=abs(x)<1e-4
            deriv[small]=-x[small]/4+x[small]**3/48
            deriv[~small]=2*j0(x[~small])/x[~small]-4*j1(x[~small])/x[~small]**2
            f0=(radii[:,None,None]/R)**2*jinc(x)
            f1=(radii[:,None,None]/R)**2*radii[:,None,None]*deriv
            dq=self.qpar-qr[None,:,:]
            P=(np.einsum('s,sfq->fq',w,f0*f0)[None,:,:]
               +2*dq*np.einsum('s,sfq->fq',w,f0*f1)[None,:,:]
               +dq*dq*np.einsum('s,sfq->fq',w,f1*f1)[None,:,:])
            M=(np.einsum('s,sfq->fq',w,f0)[None,:,:]
               +dq*np.einsum('s,sfq->fq',w,f1)[None,:,:])
            P=P*abs(self.vertical[:,None,None])**2
            M=M*self.vertical[:,None,None]
        elif self.shape=='cylinder':
            F=(radii[:,None,None,None]/R)**2*jinc(
                radii[:,None,None,None]*self.qpar[None,:,:,:])
            F=F*self.vertical[None,:,None,None]
        elif self.hemisphere_tangent:
            t,v=quadrature(self.nz)
            r=radii[:,None]*np.sqrt(1-t[None,:]**2)
            A=v[None,:]*radii[:,None]*np.pi*r*r/R**3
            qr=(self.qpar.min(axis=0)+self.qpar.max(axis=0))/2
            x=r[:,:,None,None]*qr[None,None,:,:]
            deriv=np.empty_like(x);small=abs(x)<1e-4
            deriv[small]=-x[small]/4+x[small]**3/48
            deriv[~small]=2*j0(x[~small])/x[~small]-4*j1(x[~small])/x[~small]**2
            lat0=(A[:,:,None,None]*jinc(x)).reshape(len(radii),len(t),-1)
            lat1=(A[:,:,None,None]*r[:,:,None,None]*deriv).reshape(lat0.shape)
            zpos=radii[:,None]*t[None,:]
            field=np.sum(self.coeff[None,None,:,:]*np.exp(
                1j*zpos[:,:,None,None]*self.kz[None,None,:,:]),axis=2)
            f0=np.matmul(field.transpose(0,2,1),lat0).reshape((len(radii),)+self.qpar.shape)
            f1=np.matmul(field.transpose(0,2,1),lat1).reshape(f0.shape)
            F=f0+(self.qpar-qr[None,:,:])[None,:,:,:]*f1
        else:
            t,v=quadrature(self.nz)
            r=radii[:,None]*np.sqrt(1-t[None,:]**2)
            A=(v[None,:]*radii[:,None]*np.pi*r*r/R**3)
            lat=jinc(r[:,:,None,None,None]*self.qpar[None,None,:,:,:])
            zpos=radii[:,None]*t[None,:]
            field=np.sum(self.coeff[None,None,:,:]*np.exp(
                1j*zpos[:,:,None,None]*self.kz[None,None,:,:]),axis=2)
            F=np.einsum('sz,szafq,sza->safq',A,lat,field,optimize=True)
        if self.shape!='cylinder' or not self.cylinder_tangent:
            P=np.einsum('s,safq->afq',w,abs(F)**2)
            M=np.einsum('s,safq->afq',w,F)
        return P,M

    def signal(self,D,R,sigma_log,omega_rel,*,correlated=True):
        P,M=self._moments(float(R),float(sigma_log))
        S=radial_paracrystal(self.qpar,D,omega_rel*D) if correlated else 1.
        if self.approximation=='DA':
            intensity=P+abs(M)**2*(S-1)
        else:
            intensity=P*S
        intensity=np.maximum(intensity,0)
        return np.einsum('afq,q->af',intensity,self.qw)

    def atomic_rates(self,p):
        D,R=np.exp(p[:2]);sig,omega=p[2:4]
        signal=self.signal(D,R,sig,omega)
        power=p[7]
        amplitudes=np.exp(np.asarray(p)[[4,5,6]])[None,:]
        if self.side_response=='independent':
            amplitudes=np.vstack([amplitudes,np.exp(np.asarray(p)[8:11])])
        A,B,C=amplitudes.T
        tail=np.einsum('fq,q->f',(self.qy/.5)**(-power),self.qw)
        return A[:,None,None]*signal[None,:,:]+B[:,None,None]+C[:,None,None]*tail[None,None,:]

    def predict(self,p):
        return np.sum(self.atomic_rates(p)*self.cut.exposure_pixels,axis=1)


def fit(cut, *, shape='cylinder', approximation='DA', critical_angle_deg=.16,
        q_range=(.15,1.75), sigma_q_nm_inv=0., starts=6, max_nfev=250,
        initial=None,side_response='shared'):
    """Fit two original sides jointly; no temporal smoothing or v2 point override.

    Returned errors are uninflated conditional local errors. Model mismatch
    stays in the residual; it is NOT added to the photon variance.
    """
    wall=time.perf_counter();cpu=time.process_time()
    if not cut.photon_counts:
        raise ValueError('Statistical fitting requires known original photons; corrected images need propagated variance')
    model=Model(cut,shape,approximation,critical_angle_deg,sigma_q_nm_inv,side_response=side_response)
    y,e=cut.Y,cut.E
    good=(e>0)&(cut.q>=q_range[0])[None,:]&(cut.q<=q_range[1])[None,:]
    if good.sum()<24 or y[good].sum()<100:
        return dict(status='insufficient_linecut_support',bins=int(good.sum()),
                    shape=shape,approximation=approximation)
    features=[fit_peak(cut.q,ys,es,lo=q_range[0],hi=q_range[1]) for ys,es in zip(y,e)]
    resolved=[f for f in features if f['status']=='resolved_empirical_feature']
    qstars=[f['q_star'] for f in resolved]
    D0=2*np.pi/float(np.median(qstars)) if qstars else 10.
    rates=np.divide(y,e,out=np.zeros_like(y),where=e>0)
    scale=max(float(np.quantile(rates[good],.75)),1e-6)
    lower=np.array([np.log(1.5),np.log(.25),.03,.035,np.log(scale)-18,
                    np.log(scale)-18,np.log(scale)-18,.05])
    upper=np.array([np.log(120.),np.log(35.),.45,.85,np.log(scale)+12,
                    np.log(scale)+8,np.log(scale)+8,5.])
    if side_response=='independent':
        lower=np.r_[lower,lower[4:7]];upper=np.r_[upper,upper[4:7]]
    def fun(p):return dev_residual(y[good],model.predict(p)[good])
    seeds=[]
    if initial is not None:
        initial=np.asarray(initial,float)
        if side_response=='independent' and len(initial)==8:initial=np.r_[initial,initial[4:7]]
        seeds.append(initial)
    for i in range(starts):
        d=D0*(.78,1.,1.25)[i%3]
        r=d*(.16,.27,.39)[(i//3)%3]
        seeds.append(np.array([np.log(d),np.log(max(r,.26)),.16 if i%2 else .32,
                               .16 if i%2 else .30,np.log(scale),
                               np.log(scale*.02),np.log(scale*.04),2.]))
        if side_response=='independent':seeds[-1]=np.r_[seeds[-1],seeds[-1][4:7]]
    solutions=[]
    for x in seeds:
        sol=least_squares(fun,np.clip(x,lower+1e-7,upper-1e-7),bounds=(lower,upper),
                          x_scale='jac',max_nfev=max_nfev,ftol=2e-7,xtol=2e-7,gtol=2e-6)
        solutions.append(sol)
    best=min(solutions,key=lambda f:float(f.fun@f.fun))
    p=best.x;deviance=float(best.fun@best.fun);dof=int(good.sum()-len(p))
    _,singular,vh=np.linalg.svd(best.jac,full_matrices=False)
    cov=np.linalg.pinv(best.jac.T@best.jac,rcond=1e-12)
    se=np.sqrt(np.maximum(np.diag(cov),0));D,R=np.exp(p[:2])
    predicted=model.predict(p)
    local_low=np.maximum(p-1.96*se,lower);local_high=np.minimum(p+1.96*se,upper)
    # Bounds are not evidence that a weak parameter has a narrow interval.
    bound=(p-lower < .002*(upper-lower)) | (upper-p < .002*(upper-lower))
    maxrank=max(singular[0],1e-30)
    rank=int(np.sum(singular>maxrank*1e-6))
    weak_weight=np.sum(vh[rank:]**2,axis=0) if rank<len(p) else np.zeros(len(p))
    for i in (0,1):
        if weak_weight[i]>.01 or not best.success:
            local_low[i],local_high[i]=lower[i],upper[i]
    result=dict(status='conditional_linecut_fit',shape=shape,approximation=approximation,
        D_nm=float(D),R_number_mean_nm=float(R),diameter_nm=float(2*R),
        side_response=side_response,side_response_scope='shared structural D/R/dispersity/disorder; independent A/B/C is a response/background condition, not proved instrument correction',
        radius_sigma_log=float(p[2]),radius_CV=float(np.sqrt(np.expm1(p[2]**2))),
        distance_disorder_rel=float(p[3]),
        mean_diameter_over_D=float(2*R/D),conditional_gap_mean_nm=float(D-2*R),
        nominal_hex_surface_footprint=float(2*np.pi*R**2*np.exp(p[2]**2)/(np.sqrt(3)*D**2)),
        geometry_scope='footprint/gap assume full-occupancy hexagonal density mapping; overlap is not measured connectivity',
        morphology_interpretation='overlapping domains/network alternative required' if 2*R>=D else 'isolated-island conditional interpretation; overlap not excluded by disorder',
        D_local95_nm=np.exp([local_low[0],local_high[0]]).tolist(),
        R_local95_nm=np.exp([local_low[1],local_high[1]]).tolist(),
        D_local_se_nm=float(D*se[0]),R_local_se_nm=float(R*se[1]),
        error_scope='uninflated conditional Hessian only; not calibrated posterior/systematic interval',
        h_scope='hemisphere h=radius assumption' if shape=='hemisphere' else 'vertical response assumes common cylinder h=1nm in finite AF band; height not measured',
        deviance=deviance,deviance_per_dof=deviance/max(dof,1),dof=dof,bins=int(good.sum()),
        asymptotic_gof_p=float(chi2.sf(deviance,max(dof,1))),
        statistical_compatibility='asymptotic_only_requires_bootstrap',
        AIC=deviance+2*len(p),optimizer_success=bool(best.success),
        optimizer_nfev=int(sum(s.nfev for s in solutions)),local_jacobian_rank=rank,
        local_covariance_logDR=float(cov[0,1]),parameter_boundary_flags=bound.tolist(),
        weak_direction_weights=weak_weight.tolist(),
        local_range_sources={name:('prior_bounds_due_to_incomplete_optimizer' if not best.success else 'prior_bounds_due_to_weak_jacobian' if weak_weight[i]>.01 else 'conditional_local_log_curvature_1.96SE') for i,name in enumerate(('D','R'))},
        peak_support=('bilateral' if np.ptp(qstars)/np.mean(qstars)<=.10 else 'bilateral_conflict') if len(resolved)==2 else 'one_side' if len(resolved)==1 else 'no_resolved_peak',
        linecut_peak_features=features,parameters=p.tolist(),parameter_se=se.tolist(),
        predicted_counts=predicted.tolist(),observed_counts=y.tolist(),exposure_pixels=e.tolist(),
        q_nm_inv=cut.q.tolist(),fit_mask=good.tolist(),
        critical_angle_input_deg=critical_angle_deg,q_range_nm_inv=list(q_range),
        sigma_q_input_nm_inv=sigma_q_nm_inv,starts=len(seeds),max_nfev=max_nfev,
        cylinder_AF_amplitude_tangent=model.cylinder_tangent if shape=='cylinder' else False,
        hemisphere_AF_amplitude_tangent=model.hemisphere_tangent if shape=='hemisphere' else False,
        wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu,
        physical_caution='effective surface linecut model; multilayer/rough-host response and host peaks are conditional')
    return result


def bootstrap_gof(cut,result,*,replicates=99,seed=391731,max_nfev=100,
                  checkpoint=None,resume=None):
    """Parametric refit GOF, seed declared by caller before observing results.

    Uses SAME fitting method/starts to avoid optimistic data-vs-bootstrap
    optimisation asymmetry. This is conditional model GOF, not structural truth.
    """
    rng=np.random.default_rng(seed);d=[];valid=[]
    if resume:
        if resume['seed']!=seed or resume['replicates']!=replicates:
            raise ValueError('Bootstrap identity changed')
        d=list(resume['refit_deviances']);valid=list(resume['refit_valid'])
        rng.bit_generator.state=resume['rng_state']
    mu=np.asarray(result['predicted_counts'])
    model=Model(cut,result['shape'],result['approximation'],
                result['critical_angle_input_deg'],result['sigma_q_input_nm_inv'],
                side_response=result.get('side_response','shared'))
    # Need atomic, not side-summed, means for exact same support integration.
    atom_mu=model.atomic_rates(result['parameters'])*cut.exposure_pixels
    for _ in range(len(d),replicates):
        simulated=Linecut(rng.poisson(atom_mu),cut.exposure_pixels,cut.af_deg,
                          cut.q_edges,cut.energy_ev,cut.alpha_i_deg,True,'parametric_bootstrap')
        b=fit(simulated,shape=result['shape'],approximation=result['approximation'],
              critical_angle_deg=result['critical_angle_input_deg'],
              q_range=result['q_range_nm_inv'],sigma_q_nm_inv=result['sigma_q_input_nm_inv'],
              starts=result['starts'],max_nfev=max_nfev,
              side_response=result.get('side_response','shared'))
        ok=bool(b.get('optimizer_success') and np.isfinite(b.get('deviance',float('inf'))))
        d.append(float(b['deviance']) if ok else None);valid.append(ok)
        if checkpoint:
            checkpoint(dict(seed=seed,replicates=replicates,refit_deviances=d,
                            refit_valid=valid,rng_state=rng.bit_generator.state))
    failures=sum(not v for v in valid)
    exceed=sum(ok and value>=result['deviance'] for value,ok in zip(d,valid))
    low=(1+exceed)/(replicates+1);high=(1+exceed+failures)/(replicates+1)
    if replicates<39:
        state='insufficient_replicates_for_acceptance'
    elif failures>.2*replicates:
        state='undetermined_many_failed_refits'
    else:
        state='compatible' if low>=.05 else 'incompatible' if high<.05 else 'undetermined_failed_refits'
    return dict(replicates=replicates,seed=seed,refit_deviances=d,p_value=low,
                p_value_bounds_for_failed_refits=[low,high],p_value_scope='lower bound if refits failed; failed draws never inflate compatibility',
                gof_state=state,compatible_at_5pct=state=='compatible',failed_refits=failures,
                interpretation='conditional Poisson linecut fit GOF; not D/R ground-truth recovery')


def qcm_geometry(D,R,sigma_log,equivalent_thickness_nm,*,area_factor=1.,
                 packing='hexagonal_full_occupancy'):
    """Mass closure for a common-height cylinder under explicit 2D packing.

    No inference of number density from 1D D without the stated assumption.
    area_factor stretches available area; all mass retained, no embedded/film
    metal and one population. Overlap fraction is not Boolean union coverage.
    """
    if min(D,R,area_factor)<=0 or equivalent_thickness_nm<0:
        raise ValueError('Nonnegative thickness and positive sizes/area required')
    if packing!='hexagonal_full_occupancy':raise ValueError('Unsupported density condition')
    n=2/(np.sqrt(3)*D**2)
    mean_r2=R*R*np.exp(sigma_log**2)
    nominal_area=n*np.pi*mean_r2
    h=equivalent_thickness_nm/(area_factor*nominal_area)
    return dict(height_nm=float(h),height_over_mean_radius=float(h/R),
                nominal_surface_footprint_fraction=float(nominal_area),gap_mean_nm=float(D-2*R),
                n_per_surface_nm2=float(n),area_factor=area_factor,
                volume_model='equal-height cylinders, number-mean lognormal radius',
                conditions='hexagonal density mapping; full occupancy; all QCM metal captured in one population; no buried metal or continuous base film',
                overlap_warning=bool(nominal_area>1 or 2*R>D))


def analyze_frame(image,geometry,alpha_i_deg,metal,*,af_band=(.14,.24),
                  exposure_s=None,mask=None,photon_counts=None,
                  shapes=('cylinder',),approximations=('DA','LMA_common_D'),
                  critical_angle_deg=.16,qcm_rate_nm_s=None,t0_s=None,time_s=None,**fit_options):
    """Single-frame public v3 API; does not load a neural network.

    `image` is a detector array or CBF/TIFF/EDF path; geometry is the same
    dictionary/PONI accepted by v2. Unknown exposure is absorbed into scale,
    not invented in seconds. Corrected-intensity data retain features but
    cannot silently acquire photon statistics for an error-calibrated fit.
    """
    from pathlib import Path
    from .detector_image_input import read_detector_image
    from .scenario_predict import detector_valid_mask
    from .stage5_input import AtomicMap
    header={};source='supplied_array';started=time.perf_counter()
    if isinstance(image,(str,Path)):
        path=Path(image);source=str(path);image,header=read_detector_image(path)
        if photon_counts is None:photon_counts=path.suffix.lower()=='.cbf'
    image=np.asarray(image,float)
    if str(metal).lower() not in ('au','ag'):raise ValueError('Use explicit Au or Ag; Cu is not mapped')
    exposure=exposure_s if exposure_s is not None else header.get('exposure_s')
    if exposure is not None and (not np.isfinite(exposure) or exposure<=0):
        raise ValueError('Explicit/header exposure must be positive')
    amap=AtomicMap(geometry,image.shape,alpha_i_deg,exclude_rings=((2.1,.13),))
    valid,mask_meta=detector_valid_mask(geometry,mask,shape=image.shape)
    Y,E=amap.project(image,1. if exposure is None else exposure,valid)
    obs=direct_features(Y,E)
    intensity_kind='counts' if photon_counts else 'supplied_intensity'
    out=dict(observations=obs,source=source,metal=metal,geometry=amap.meta,mask=mask_meta,
             intensity_unit=intensity_kind+('_per_pixel_second' if exposure is not None else '_per_pixel_unknown_exposure'),
             exposure_s=exposure,variants=[])
    from .sequence_v3 import attach_qcm,conditional_interpretations
    out['interpretations']=conditional_interpretations(obs)
    if not photon_counts:
        out.update(fit_status='variance_required_for_corrected_intensity',
                   fit_reason='Original photon counts or propagated variance required; features preserved',
                   qcm_status='no fitted size for QCM geometry',wall_seconds=time.perf_counter()-started)
        return out
    cut,actual=from_atomic(Y,E,af_band,amap.meta['energy_ev'],alpha_i_deg,source=source)
    out['af_band_actual']=actual
    out['variants']=[fit(cut,shape=s,approximation=ap,critical_angle_deg=critical_angle_deg,
                         **fit_options) for s in shapes for ap in approximations]
    out['qcm_status']=attach_qcm(out['variants'],qcm_rate_nm_s,t0_s,time_s)
    valid=[v for v in out['variants'] if v.get('status')=='conditional_linecut_fit']
    out['interpretations']=conditional_interpretations(obs,min(valid,key=lambda v:v['AIC']) if valid else None)
    out.update(fit_status='conditional_linecut_fit' if any(v['status']=='conditional_linecut_fit' for v in out['variants']) else 'insufficient_linecut_support',wall_seconds=time.perf_counter()-started)
    return out


def analyze_scan(images,geometry,alpha_i_deg,metal,*,times_s=None,**frame_options):
    """Stream the same frame API over a directory/list, without loading a NN.

    AF band/options remain fixed for the scan. Header times are relative to
    first valid supplied header, NOT deposition onset. Explicit times_s must
    use the same clock as a supplied QCM t0. Yields one complete result/record
    at a time, including unknown-variance frames, so long scans stay bounded.
    """
    from pathlib import Path
    from datetime import datetime
    from .detector_image_input import read_detector_image
    if 'time_s' in frame_options:raise ValueError('For scans use times_s, not one broadcast frame time')
    if isinstance(images,(str,Path)):
        p=Path(images)
        paths=sorted(x for x in p.iterdir() if x.suffix.lower() in ('.cbf','.tif','.tiff','.edf')) if p.is_dir() else [p]
    else:paths=[Path(x) for x in images]
    if not paths:raise ValueError('No supported input frames')
    if times_s is not None and (len(times_s)!=len(paths) or not np.isfinite(times_s).all()):
        raise ValueError('Explicit times_s must have one finite supplied-clock value per frame')
    options=dict(frame_options);declared_photons=options.pop('photon_counts',None)
    declared_exposure=options.pop('exposure_s',None);first_time=None
    for i,path in enumerate(paths):
        image,header=read_detector_image(path);t=None
        if times_s is not None:t=float(times_s[i])
        elif header.get('timestamp'):
            try:current=datetime.fromisoformat(str(header['timestamp']).replace('Z','+00:00'))
            except ValueError:current=None
            if current is not None:
                if first_time is None:first_time=current
                t=(current-first_time).total_seconds()
        photons=path.suffix.lower()=='.cbf' if declared_photons is None else declared_photons
        exposure=declared_exposure if declared_exposure is not None else header.get('exposure_s')
        result=analyze_frame(image,geometry,alpha_i_deg,metal,exposure_s=exposure,
                             photon_counts=photons,time_s=t,**options)
        result.update(source=str(path),frame_index=i,time_s=t,
            time_origin='explicit supplied clock, not inferred shutter' if times_s is not None else 'first valid supplied header, not shutter; missing timestamps retained')
        yield result
