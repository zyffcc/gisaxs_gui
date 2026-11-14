'Stage3 conditional finite-layer TS fit using the frozen S4 forward.'
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit
from .linecut_v3 import Model
from .effective_host import HostOptics
from .scenario_ts import TSParams,build_context,metal_intensity
from .features import dev_residual

class NetworkModel:
    def __init__(self,cut,metal='Ag',reference=False):
        self.cut=cut;self.metal=metal;self.reference=reference;self.grid=Model(cut,'cylinder','DA',.16)
        self.bands=[(.08,.14),(.14,.24),(.24,.40)]
        self.band_masks=[(cut.af_deg>=lo)&(cut.af_deg<hi) for lo,hi in self.bands]
        if not all(m.any() for m in self.band_masks):raise ValueError('All three disjoint AF bands required for stage3 optical fit')
        self.Y=np.stack([cut.counts[:,m].sum(axis=1) for m in self.band_masks],axis=1)
        self.E=np.stack([cut.exposure_pixels[:,m].sum(axis=1) for m in self.band_masks],axis=1)
    def physical(self,p):return TSParams(domain_nm=float(np.exp(p[0])),width_rel=float(p[1]),fraction=float(expit(p[2])),thickness_nm=float(np.exp(p[3])),depth_corr_nm=float(np.exp(p[4])))
    def predict(self,p):
        params=self.physical(p);host=HostOptics(roughness_nm=float(p[5]))
        context=build_context(host,params,self.metal,self.cut.energy_ev,self.cut.alpha_i_deg)
        qp=self.grid.qpar.reshape(len(self.cut.af_deg),-1)
        intensity=metal_intensity(context,self.cut.af_deg,qp,order=8,reference=self.reference).reshape(self.grid.qpar.shape)
        signal=np.einsum('afq,q->af',intensity,self.grid.qw)
        amp=np.exp(np.array([p[6:9],p[10:13]]));A,B,C=amp.T
        tail=np.einsum('fq,q->f',(self.grid.qy/.5)**(-p[9]),self.grid.qw)
        rates=A[:,None,None]*signal[None,:,:]+B[:,None,None]+C[:,None,None]*tail[None,None,:]
        predicted=np.stack([(rates[:,m]*self.cut.exposure_pixels[:,m]).sum(axis=1) for m in self.band_masks],axis=1)
        return predicted,rates

def fit_network(cut,*,metal='Ag',initial=None,max_nfev=200):
    start=time.perf_counter()
    if not cut.photon_counts:raise ValueError('Original counts required')
    model=NetworkModel(cut,metal);y,e=model.Y,model.E;good=(e>0)&((cut.q>=.05)&(cut.q<=1.75))[None,None,:]
    if any(good[:,band].sum()<8 for band in range(3)):return dict(status='missing_stage3_band_support',module='finite_layer_TS_frozen_forward',reason='Each declared AF band needs actual exposed fitting q cells; no three-band optics claim from empty support')
    if good.sum()<26 or y[good].sum()<100:return dict(status='insufficient_network_support')
    scale=max(float(np.quantile(np.divide(y,e,out=np.zeros_like(y),where=e>0)[good],.75)),1e-6)

    prototype=np.array([np.log(30.),.8,0.,np.log(10.),np.log(5.),.5,0.,-40.,-40.,1.,0.,-40.,-40.])
    signal=model.predict(prototype)[0];s=max(float(np.median((signal/e.clip(1e-300))[good])),1e-20);al=np.log(scale/s)
    lower=np.array([np.log(3.),.03,np.log(.01/.99),np.log(.2),np.log(.2),0.,al-20,np.log(scale)-18,np.log(scale)-18,.05,al-20,np.log(scale)-18,np.log(scale)-18])
    upper=np.array([np.log(200.),1.35,np.log(.99/.01),np.log(60.),np.log(120.),5.,al+15,np.log(scale)+8,np.log(scale)+8,5.,al+15,np.log(scale)+8,np.log(scale)+8])
    def fun(p):return dev_residual(y[good],model.predict(p)[0][good])
    seeds=[]
    if initial is not None:seeds.append(np.asarray(initial,float))
    for h in [3.,15.,45.]:seeds.append(np.array([np.log(50.),.8,0.,np.log(h),np.log(5.),.5,al,np.log(scale*.01),np.log(scale*.01),1.,al,np.log(scale*.01),np.log(scale*.01)]))
    sols=[least_squares(fun,np.clip(p,lower+1e-7,upper-1e-7),bounds=(lower,upper),x_scale='jac',max_nfev=max_nfev,ftol=2e-7,xtol=2e-7,gtol=2e-6) for p in seeds]
    continuity=None
    if initial is not None and any(abs(s.x[j]-initial[j])>np.log(1.5) for s in sols for j in [0,3]):
        lo,hi=lower.copy(),upper.copy()
        for j in [0,3]:lo[j]=max(lo[j],initial[j]-np.log(1.5));hi[j]=min(hi[j],initial[j]+np.log(1.5))
        continuity=least_squares(fun,np.clip(initial,lo+1e-7,hi-1e-7),bounds=(lo,hi),x_scale='jac',max_nfev=max_nfev);sols.append(continuity)
    sol=min(sols,key=lambda s:s.fun@s.fun);p=sol.x;dev=float(sol.fun@sol.fun);dof=int(good.sum()-13)
    _,sv,vh=np.linalg.svd(sol.jac,full_matrices=False);rank=int(np.sum(sv>max(sv[0],1e-30)*1e-6));weak=np.sum(vh[rank:]**2,axis=0) if rank<13 else np.zeros(13)
    cov=np.linalg.pinv(sol.jac.T@sol.jac,rcond=1e-12);se=np.sqrt(np.maximum(np.diag(cov),0));lo=np.maximum(p-1.96*se,lower);hi=np.minimum(p+1.96*se,upper);bad=(weak>.01)|(not sol.success);lo[bad]=lower[bad];hi[bad]=upper[bad]
    phys=model.physical(p);mu,rates=model.predict(p)
    def lateral_xi(D,w):
        ss=(2*np.pi/D)**2;bb=ss*w*np.sqrt(1-w*w/4)
        return float(1/np.sqrt(ss+1j*bb).imag)
    xis=[lateral_xi(float(np.exp(dd)),float(ww)) for dd in [lo[0],hi[0]] for ww in [lo[1],hi[1]]]
    return dict(status='conditional_stage3_TS',module='finite_layer_TS_frozen_forward',stage_id=3,
        spectral_domain_nm=phys.domain_nm,spectral_FWHM_relative=phys.width_rel,effective_thickness_nm=phys.thickness_nm,metal_fraction_condition=phys.fraction,host_network_interface_rms_nm=float(p[5]),depth_correlation_nm=phys.depth_corr_nm,
        lateral_decay_correlation_nm=lateral_xi(phys.domain_nm,phys.width_rel),lateral_decay_local_box_nm=[min(xis),max(xis)],lateral_correlation_definition='TS pole-envelope 1/Im(sqrt(q0^2+i*b)); spectral peak period is separate',
        lamellar_conditional_metal_band_nm=phys.domain_nm*phys.fraction,lamellar_conditional_gap_nm=phys.domain_nm*(1-phys.fraction),
        gap_scope='Only the declared alternating-lamella surrogate with this volume fraction; TS spectrum has no unique generic gap or proven connected topology',
        roughness_scope='Fitted host/network buried-interface rms only; top free surface rms is not measured by this parameter',
        optical_scope='Si/1.5nm native oxide and frozen S4 mean-metal layer; fixed uncalibrated reference and side gains, not absolute mass',
        domain_local95_nm=np.exp([lo[0],hi[0]]).tolist(),thickness_local95_nm=np.exp([lo[3],hi[3]]).tolist(),buried_rms_local95_nm=[float(lo[5]),float(hi[5])],
        range_scope='Uncalibrated conditional local-curvature/weak-direction implementation bounds; thickness domain remains .2--60nm, no silent expansion',
        parameters=p.tolist(),predecessor_parameters=(continuity.x if continuity is not None and continuity.success else p).tolist(),propagated_branch_id='continuity_conditioned' if continuity is not None and continuity.success else f'start_{next(i for i,s in enumerate(sols) if s is sol)}',
        solver_candidates=[dict(branch_id='continuity_conditioned' if s is continuity else f'start_{i}',parameters=s.x.tolist(),deviance=float(s.fun@s.fun),deviance_per_dof=float(s.fun@s.fun)/max(dof,1),optimizer_success=bool(s.success),predicted_counts=model.predict(s.x)[0].tolist()) for i,s in enumerate(sols)],
        deviance=dev,dof=dof,deviance_per_dof=dev/max(dof,1),approximate_one=bool(sol.success and abs(dev-dof)<=np.sqrt(2*dof)),optimizer_success=bool(sol.success),optimizer_nfev=int(sum(s.nfev for s in sols)),local_jacobian_rank=rank,parameter_boundary_flags=((p-lower<.002*(upper-lower))|(upper-p<.002*(upper-lower))).tolist(),
        observed_counts=y.tolist(),exposure_pixels=e.tolist(),predicted_counts=mu.tolist(),predicted_atomic_rates=rates.tolist(),AF_bands_deg=model.bands,q_range_nm_inv=[.05,1.75],q_nm_inv=cut.q.tolist(),fit_mask=good.tolist(),
        checks_not_passed=['newAgAuTEST','formalbootstrap','connected_topology','unique_gap','free_surface_roughness'],wall_seconds=time.perf_counter()-start)
