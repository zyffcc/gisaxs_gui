'Si two-population conditional linecut model on the existing surface field.'
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2
from .linecut_v3 import Model
from .features import dev_residual

class BimodalModel:
    def __init__(self,cut,shape='cylinder',approximation='DA',critical_angle_deg=.16,stage=1):
        if stage not in (1,2):raise ValueError('Two-population island/self module only stages1/2')
        self.base=Model(cut,shape,approximation,critical_angle_deg,side_response='independent')
        self.cut=cut;self.stage=stage;self.shape=shape;self.approximation=approximation
        self.radius_normalization_power=4 if shape=='cylinder' else 6

    def atomic_rates(self,p,components=False):
        D,R=np.exp(p[:2]);Rs=np.exp(p[11]);ratio=np.exp(p[12])
        main=self.base.signal(D,R,p[2],p[3],correlated=self.stage==1)
        small=self.base.signal(1.,Rs,.15,1.,correlated=False)
        amplitudes=np.exp(np.array([p[4:7],p[8:11]]));A,B,C=amplitudes.T


        weight=ratio*(Rs/R)**self.radius_normalization_power
        big=A[:,None,None]*main[None,:,:]
        little=A[:,None,None]*weight*small[None,:,:]
        tail=np.einsum('fq,q->f',(self.base.qy/.5)**(-p[7]),self.base.qw)
        background=B[:,None,None]+C[:,None,None]*tail[None,None,:]
        if components:return dict(main=big,secondary=little,background=np.broadcast_to(background,big.shape))
        return big+little+background

    def predict(self,p):return np.sum(self.atomic_rates(p)*self.cut.exposure_pixels,axis=1)

def fit_bimodal(cut,baseline,*,stage=1,initial=None,starts=3,max_nfev=250,temporal_factor=None):
    start=time.perf_counter();shape=baseline['shape'];approx=baseline.get('approximation','DA')
    if approx=='uncorrelated_self_DA':approx='DA'
    ac=baseline.get('critical_angle_input_deg',.16);model=BimodalModel(cut,shape,approx,ac,stage)
    qrange=(.15,1.75) if stage==1 else (.8,1.75)
    y,e=cut.Y,cut.E;good=(e>0)&((cut.q>=qrange[0])&(cut.q<=qrange[1]))[None,:]
    if good.sum()<24 or y[good].sum()<100:return dict(status='insufficient_bimodal_support')
    rate=np.divide(y,e,out=np.zeros_like(y),where=e>0);scale=max(float(np.quantile(rate[good],.75)),1e-6)
    lower=np.array([np.log(1.5),np.log(.25),.03,.035,np.log(scale)-18,np.log(scale)-18,np.log(scale)-18,.05,np.log(scale)-18,np.log(scale)-18,np.log(scale)-18,np.log(.5),-12.])
    upper=np.array([np.log(120.),np.log(35.),.45,.85,np.log(scale)+12,np.log(scale)+8,np.log(scale)+8,5.,np.log(scale)+12,np.log(scale)+8,np.log(scale)+8,np.log(1.5),16.])
    active=np.arange(13) if stage==1 else np.array([1,2,4,5,6,7,8,9,10,11,12])
    prior=np.array(baseline['parameters'],float)
    if len(prior)==9:prior=np.r_[np.log(10.),prior[0],prior[1],.2,prior[2:5],prior[8],prior[5:8]]
    if len(prior)!=11:raise ValueError('Expected original independent-side main or highq baseline')
    seeds=[]
    if initial is not None:seeds.append(np.array(initial,float))
    for i in range(starts):seeds.append(np.r_[prior,np.log((.6,1.,1.4)[i%3]),np.log((5.,50.,500.)[i%3])])
    if not seeds:raise ValueError('At least one seed')
    if temporal_factor is not None:
        if initial is None or temporal_factor<=1:raise ValueError('Temporal box requires prior parameters/factor')
        constrained=[0,1] if stage==1 else [1]
        for j in constrained:lower[j]=max(lower[j],initial[j]-np.log(temporal_factor));upper[j]=min(upper[j],initial[j]+np.log(temporal_factor))
    solutions=[]
    for seed in seeds:
        seed=np.clip(seed,lower+1e-7,upper-1e-7)
        def expand(x):
            full=seed.copy();full[active]=x;return full
        def fun(x):return dev_residual(y[good],model.predict(expand(x))[good])
        fit=least_squares(fun,seed[active],bounds=(lower[active],upper[active]),x_scale='jac',max_nfev=max_nfev,ftol=2e-7,xtol=2e-7,gtol=2e-6)
        solutions.append((fit,expand(fit.x)))
    sol,p=min(solutions,key=lambda x:x[0].fun@x[0].fun);dev=float(sol.fun@sol.fun);dof=int(good.sum()-len(active))
    cov=np.linalg.pinv(sol.jac.T@sol.jac,rcond=1e-12);se_active=np.sqrt(np.maximum(np.diag(cov),0));se=np.zeros(13);se[active]=se_active
    _,sv,vh=np.linalg.svd(sol.jac,full_matrices=False);rank=int(np.sum(sv>max(sv[0],1e-30)*1e-6));wa=np.sum(vh[rank:]**2,axis=0) if rank<len(active) else np.zeros(len(active));weak=np.zeros(13);weak[active]=wa
    lo=np.maximum(p-1.96*se,lower);hi=np.minimum(p+1.96*se,upper)
    for j in active:
        if weak[j]>.01 or not sol.success:lo[j],hi[j]=lower[j],upper[j]
    components={name:np.sum(value*cut.exposure_pixels,axis=1).tolist() for name,value in model.atomic_rates(p,True).items()}
    D,R,Rs=np.exp(p[[0,1,11]]);nratio=np.exp(p[12])
    return dict(status='conditional_bimodal_fit',module='Si_bimodal',stage_id=stage,shape=shape,approximation=approx,
        D_nm=float(D) if stage==1 else None,D_reason='Correlated main-population1D step' if stage==1 else 'No D parameter optimized in stage2',
        R_number_mean_nm=float(R),R_small_number_mean_nm=float(Rs),radius_sigma_log=float(p[2]),small_radius_sigma_log_condition=.15,
        small_to_main_number_ratio=float(nratio),small_number_fraction=float(nratio/(1+nratio)),secondary_larger_than_main=bool(Rs>R),
        main_mean_diameter_exceeds_correlated_step=bool(stage==1 and 2*R>D),packing_scope='Observed bilateral-peak stage does not certify disjoint particles; free-amplitude two-population fit is not an absolute occupancy measurement',
        ratio_scope='Same-shape conditional number ratio after undoing reference-radius amplitude normalization; shared side gain; no interpopulation cross term/absolute number density',
        D_local95_nm=np.exp([lo[0],hi[0]]).tolist() if stage==1 else None,R_local95_nm=np.exp([lo[1],hi[1]]).tolist(),R_small_local95_nm=np.exp([lo[11],hi[11]]).tolist(),
        range_scope='Uncalibrated local curvature/weak-direction bound, conditional on model and explicit temporal box',temporal_factor_condition=temporal_factor,
        deviance=dev,dof=dof,deviance_per_dof=dev/max(dof,1),AIC=dev+2*len(active),asymptotic_gof_p=float(chi2.sf(dev,max(dof,1))),optimizer_success=bool(sol.success),optimizer_nfev=int(sum(x[0].nfev for x in solutions)),
        parameters=p.tolist(),parameter_se=se.tolist(),active_parameter_indices=active.tolist(),parameter_boundary_flags=(((p-lower<.002*(upper-lower))|(upper-p<.002*(upper-lower)))&np.isin(np.arange(13),active)).tolist(),local_jacobian_rank=rank,weak_direction_weights=weak.tolist(),
        observed_counts=y.tolist(),exposure_pixels=e.tolist(),predicted_counts=model.predict(p).tolist(),predicted_components=components,q_nm_inv=cut.q.tolist(),fit_mask=good.tolist(),q_range_nm_inv=list(qrange),critical_angle_input_deg=ac,
        checks_not_passed=['newAgAuTEST','formalbootstrap','absolute_number_ratio_truth'],wall_seconds=time.perf_counter()-start)

def fit_bimodal_branches(cut,baseline,previous=None,*,stage=1,jump_factor=1.5,max_nfev=250):
    cold=fit_bimodal(cut,baseline,stage=stage,starts=3,max_nfev=max_nfev)
    if cold['status']!='conditional_bimodal_fit':return dict(status=cold['status'],branches=[],predecessor_for_next=None)
    cold.update(branch_id='independent_bimodal',initialization='registered_three_starts');branches=[cold]
    if previous is None:return dict(status='first_applicable_bimodal',branches=branches,predecessor_for_next=cold['branch_id'])
    warm=fit_bimodal(cut,baseline,stage=stage,initial=previous['parameters'],starts=0,max_nfev=max_nfev);warm.update(branch_id='previous_frame_bimodal',initialization='previous_frame_solution');branches.append(warm)
    keys=['R_number_mean_nm']+(['D_nm'] if stage==1 else [])
    changed=lambda r:any(previous.get(k) and abs(np.log(r[k]/previous[k]))>np.log(jump_factor) for k in keys)
    triggered=changed(cold) or changed(warm)
    if triggered:
        cont=fit_bimodal(cut,baseline,stage=stage,initial=previous['parameters'],starts=0,max_nfev=max_nfev,temporal_factor=jump_factor);cont.update(branch_id='continuity_conditioned_bimodal',initialization='previous_frame_plus_explicit_box');branches.append(cont);next_id=cont['branch_id']
    else:next_id=warm['branch_id']
    for r in branches:r['deviance_minus_best_branch']=r['deviance']-min(x['deviance'] for x in branches)
    return dict(status='explicit_bimodal_branches',branches=branches,predecessor_for_next=next_id,jump_triggered=triggered)
