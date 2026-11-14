'Conditional partly embedded hemispheres in a sharp polymer/oxide/Si field.'
from __future__ import annotations
from functools import lru_cache
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
from scipy.special import roots_legendre
from .linecut_v3 import Model,size_rule,quadrature
from .effective_host import HostOptics,field_at,field_arrays,host_base,chi
from .optics import nd
from .formfactor import jinc
from .interference import radial_paracrystal
from .features import dev_residual

@lru_cache(maxsize=64)
def depth_rule(order):


    x,w=roots_legendre(order)
    return (x+1)/2,w/2

def fixed_host_context(host,E,ai,zmax=600.):
    host.validate()
    if host.transition_nm or min(host.thickness_nm,host.oxide_nm)<=0:raise ValueError('Sharp finite polymer and oxide required')
    edges=np.array([-host.thickness_nm-host.oxide_nm,-host.thickness_nm,0.,zmax])
    values=[0j,0j,host.decrement,nd('SiO2',energy_ev=E),nd('Si',energy_ev=E)]
    return dict(host=host,E=E,ai=ai,edges=edges,pre_nds=values,post_nds=list(values),thick=np.diff(edges)[::-1].tolist(),sigma=[0.,0.,host.roughness_nm,host.buried_roughness_nm,0.])

class EmbeddedKernel:
    def __init__(self,cut,thickness_nm,metal,delta12,beta12,*,nsize=224,nz=64,host_material=None):
        self.cut=cut;self.thickness=thickness_nm;self.metal=metal;self.delta12=delta12;self.beta12=beta12;self.nsize=nsize;self.nz=nz
        self.host_material=host_material
        self.geometry=Model(cut,'hemisphere','DA',.16)
        self.qpar=self.geometry.qpar;self.qw=self.geometry.qw
    @lru_cache(maxsize=3)
    def context(self,density_factor):
        factor=(12000./self.cut.energy_ev)**2
        optical=nd(self.host_material,energy_ev=self.cut.energy_ev) if self.host_material else complex(self.delta12,self.beta12)*factor
        host=HostOptics(decrement=optical*density_factor,thickness_nm=self.thickness,oxide_nm=1.5)
        ctx=fixed_host_context(host,self.cut.energy_ev,self.cut.alpha_i_deg);field_arrays(ctx,self.cut.af_deg,post=False)
        return ctx
    def mark_amplitudes(self,radii,b,ctx,*,nz=None):
        q=self.qpar.reshape(len(self.cut.af_deg),-1);out=[]
        k=2*np.pi/(1239.841984/self.cut.energy_ev)
        phase_per_nm=float(q.max())+k*(np.sin(np.radians(self.cut.alpha_i_deg))+np.sin(np.radians(self.cut.af_deg.max())))
        contrast_metal=nd(self.metal,energy_ev=self.cut.energy_ev)
        for radius in radii:


            order=max(nz or self.nz,int(np.ceil(.75*phase_per_nm*radius))+12)
            order=16*int(np.ceil(order/16))
            t,w=depth_rule(order)
            cuts=np.unique(np.clip([-b,0.,radius-b],-b,radius-b));F=np.zeros_like(q,dtype=complex)
            for lo,hi in zip(cuts[:-1],cuts[1:]):
                if hi<=lo:continue
                z=lo+(hi-lo)*t;rr=radius*np.sqrt(np.maximum(1-((z+b)/radius)**2,0))
                weight=(hi-lo)*w*np.pi*rr**2*chi(contrast_metal-host_base(ctx['host'],self.cut.energy_ev,z))
                fields=field_at(ctx,self.cut.af_deg,z,post=False)
                lat=jinc(rr[None,:,None]*q[:,None,:])
                F+=np.einsum('az,z,azq->aq',fields,weight,lat,optimize=True)
            out.append(F)
        return np.asarray(out).reshape((len(radii),)+self.qpar.shape)
    @lru_cache(maxsize=4)
    def moments(self,R,sigma,b,density):
        z,w=size_rule(self.nsize);r=R*np.exp(sigma*z-.5*sigma**2)
        F=self.mark_amplitudes(r,b,self.context(float(density)))/R**3
        return np.einsum('s,safq->afq',w,abs(F)**2),np.einsum('s,safq->afq',w,F)
    def rates(self,p,approximation):
        D,R=np.exp(p[:2]);sig,omega=p[2:4];b=p[11]*min(R,self.thickness);density=np.exp(p[12])
        P,M=self.moments(float(R),float(sig),float(b),float(density));S=radial_paracrystal(self.qpar,D,omega*D)
        intensity=P+abs(M)**2*(S-1) if approximation=='DA' else P*S
        signal=np.einsum('afq,q->af',np.maximum(intensity,0),self.qw)
        amp=np.exp(np.array([p[4:7],p[8:11]]));A,B,C=amp.T;tail=np.einsum('fq,q->f',(self.geometry.qy/.5)**(-p[7]),self.qw)
        return A[:,None,None]*signal[None,:,:]+B[:,None,None]+C[:,None,None]*tail[None,None,:]
    def predict(self,p,approximation):return (self.rates(p,approximation)*self.cut.exposure_pixels).sum(axis=1)

def fit_embedded(cut,baseline,*,metal='Ag',polymer='PS',thickness_nm=50.,approximation='DA',initial=None,max_nfev=150):
    begun=time.perf_counter()
    if not cut.photon_counts:raise ValueError('Original photon counts required, no invented corrected-intensity variance')
    if metal not in ('Au','Ag') or approximation not in ('DA','LMA_common_D'):raise ValueError('Declared Au/Ag and DA/LMA condition required')
    if polymer=='PS':
        optical=nd('PS',energy_ev=12000.);delta12,beta12=optical.real,optical.imag;density_range=(.5,1.5)
    elif polymer=='PSBMA':delta12,beta12=2.e-6,2.e-8;density_range=(.4,2.25)
    else:raise ValueError('Declare PS or unknown-optical PSBMA condition')
    kernel=EmbeddedKernel(cut,thickness_nm,metal,delta12,beta12,host_material='PS' if polymer=='PS' else None)
    y,e=cut.Y,cut.E;good=(e>0)&((cut.q>=.15)&(cut.q<=1.75))[None,:]
    if good.sum()<26 or y[good].sum()<100:return dict(status='insufficient_original_polymer_support',module='polymer_fixedhost_embed',good_cells=int(good.sum()))
    scale=max(float(np.quantile(np.divide(y,e,out=np.zeros_like(y),where=e>0)[good],.75)),1e-6)


    signal_scale=max(abs(chi(nd(metal,energy_ev=cut.energy_ev)))**2,1e-18)
    al=np.log(scale/signal_scale)
    lower=np.array([np.log(1.5),np.log(.25),.03,.035,al-18,np.log(scale)-18,np.log(scale)-18,.05,al-18,np.log(scale)-18,np.log(scale)-18,0.,np.log(density_range[0])])
    upper=np.array([np.log(120.),np.log(15.),.45,.85,al+12,np.log(scale)+8,np.log(scale)+8,5.,al+12,np.log(scale)+8,np.log(scale)+8,1.,np.log(density_range[1])])
    old=np.array(baseline['parameters'],float)
    if old.shape!=(11,):raise ValueError('Original independent-side hemisphere seed required')
    old=old.copy();old[[4,8]]-=np.log(signal_scale)
    seeds=[]
    if initial is not None:seeds.append(np.asarray(initial,float))
    for burial in [0.,.35,.75]:seeds.append(np.r_[old,burial,0.])
    def residual(p):return dev_residual(y[good],kernel.predict(p,approximation)[good])
    solutions=[least_squares(residual,np.clip(seed,lower+1e-7,upper-1e-7),bounds=(lower,upper),x_scale='jac',max_nfev=max_nfev,ftol=2e-7,xtol=2e-7,gtol=2e-6) for seed in seeds]
    jump=bool(initial is not None and any(abs(s.x[j]-initial[j])>np.log(1.5) for s in solutions for j in [0,1]))
    continuity=None
    if jump:
        tl=lower.copy();tu=upper.copy()
        for j in [0,1]:tl[j]=max(tl[j],initial[j]-np.log(1.5));tu[j]=min(tu[j],initial[j]+np.log(1.5))
        continuity=least_squares(residual,np.clip(initial,tl+1e-7,tu-1e-7),bounds=(tl,tu),x_scale='jac',max_nfev=max_nfev,ftol=2e-7,xtol=2e-7,gtol=2e-6)
        solutions.append(continuity)
    sol=min(solutions,key=lambda s:s.fun@s.fun);p=sol.x;dev=float(sol.fun@sol.fun);dof=int(good.sum()-len(p))
    _,sv,vh=np.linalg.svd(sol.jac,full_matrices=False);rank=int(np.sum(sv>max(sv[0],1e-30)*1e-6));weak=np.sum(vh[rank:]**2,axis=0) if rank<len(p) else np.zeros(len(p))
    cov=np.linalg.pinv(sol.jac.T@sol.jac,rcond=1e-12);se=np.sqrt(np.maximum(np.diag(cov),0));lo=np.maximum(p-1.96*se,lower);hi=np.minimum(p+1.96*se,upper)
    bad=(weak>.01)|(not sol.success);lo[bad]=lower[bad];hi[bad]=upper[bad]
    D,R=np.exp(p[:2]);b=p[11]*min(R,thickness_nm);z,w=size_rule(kernel.nsize);r=R*np.exp(p[2]*z-.5*p[2]**2)
    return dict(status='conditional_embedded_hemisphere_fit',module='polymer_fixedhost_embed',polymer=polymer,metal=metal,approximation=approximation,
        stage_id=1,D_nm=float(D),R_number_mean_nm=float(R),radius_sigma_log=float(p[2]),burial_nm=float(b),exposed_height_number_mean_nm=float(w@np.maximum(r-b,0)),surface_section_radius_number_mean_nm=float(w@np.sqrt(np.maximum(r*r-b*b,0))),
        thickness_condition_nm=thickness_nm,host_delta12_condition=float(delta12*np.exp(p[12])),host_beta12_condition=float(beta12*np.exp(p[12])),host_source='Declared conditional optical family; no transferable prestate measurement. PS uses table then density scaling; PSBMA optics unknown.',
        reference_scope='Fixed-host dilute DWBA, metal mean density omitted; not frozen v2 full mean field',
        D_local95_nm=np.exp([lo[0],hi[0]]).tolist(),R_local95_nm=np.exp([lo[1],hi[1]]).tolist(),range_scope='Uncalibrated conditional local curvature/weak-direction declared bounds; host thickness held at declared condition',
        parameters=p.tolist(),jump_triggered=jump,
        solver_candidates=[dict(branch_id='continuity_conditioned' if s is continuity else 'previous_start' if initial is not None and i==0 else f'independent_start_{i}',D_nm=float(np.exp(s.x[0])),R_number_mean_nm=float(np.exp(s.x[1])),burial_nm=float(s.x[11]*min(np.exp(s.x[1]),thickness_nm)),parameters=s.x.tolist(),deviance=float(s.fun@s.fun),deviance_per_dof=float(s.fun@s.fun)/max(dof,1),deviance_minus_best=float(s.fun@s.fun)-dev,optimizer_success=bool(s.success),predicted_counts=kernel.predict(s.x,approximation).tolist(),range_scope='No calibrated interval from an individual start; full local range belongs to selected fit only') for i,s in enumerate(solutions)],
        predecessor_parameters=(continuity.x if continuity is not None and continuity.success else p).tolist(),
        deviance=dev,dof=dof,deviance_per_dof=dev/max(dof,1),asymptotic_gof_p=float(chi2.sf(dev,max(dof,1))),approximate_one=bool(sol.success and abs(dev-dof)<=np.sqrt(2*dof)),
        optimizer_success=bool(sol.success),optimizer_nfev=int(sum(s.nfev for s in solutions)),local_jacobian_rank=rank,parameter_boundary_flags=((p-lower<.002*(upper-lower))|(upper-p<.002*(upper-lower))).tolist(),
        observed_counts=y.tolist(),exposure_pixels=e.tolist(),predicted_counts=kernel.predict(p,approximation).tolist(),q_nm_inv=cut.q.tolist(),fit_mask=good.tolist(),
        checks_not_passed=['transferable_prestate_optics','formalbootstrap','newAgAuTEST','global_range_calibration'],wall_seconds=time.perf_counter()-begun)
