'Native positive-count NR mode refinement, separate from frozen tools.'
from __future__ import annotations
import copy,time
from functools import lru_cache
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import brentq
from scipy.stats import chi2
from .neural_refine_chart import NAMES,INDEX,decode
from .neural_refine_forward import rates
from .neural_refine_measurement import response,empirical_background
from .neural_refine_input import NATIVE_AF,Q,stripe_indices,refinement_partition,project_partition_rates,extract
from .neural_refine_likelihood import residual,compare_floors
from .features import AF_BANDS


def selected_partition(Y,E,qstar,policy):
    'Select disjoint photon groups, including the frozen support policy.'
    use_v=bool(policy.get('vertical',False)) and qstar is not None
    stripe=qstar if use_v else None
    y,e,kinds=refinement_partition(Y,E,stripe);ix,known=stripe_indices(stripe)
    qkeep=np.ones(len(Q),bool)
    if known:qkeep[ix]=False
    n=8*len(Q)+2*len(NATIVE_AF);common=np.zeros(n,bool);common[policy['common_signal_indices']]=True
    allow=[];groups=[];coordinates=[]
    for band in range(4):
        mask=common[band*2*len(Q):(band+1)*2*len(Q)].reshape(2,len(Q))[:,qkeep]
        mask&=band in policy['horizontal']
        if policy.get('highq_only'):mask&=Q[qkeep][None,:]>=.8
        allow.append(mask.ravel())
        groups.extend([f'H{band}_side{s}' for s in range(2) for _ in range(int(qkeep.sum()))])
        coordinates.extend(np.tile(Q[qkeep],2).tolist())
    if known:
        allow.append(common[8*len(Q):]&use_v)
        groups.extend([f'V_side{s}' for s in range(2) for _ in NATIVE_AF])
        coordinates.extend(np.tile(NATIVE_AF,2).tolist())
    keep=np.concatenate(allow)&(e>0)
    return dict(y=y,e=e,kinds=kinds,groups=np.asarray(groups),coordinates=np.asarray(coordinates),keep=keep,partition_qstar=stripe,
        observed_qstar=qstar,scope='retained input-policy counts only; no duplicated photons')


def _profile_cut_scales(y,mu,groups,eps):
    'Remove the same between-cut amplitude information as per_cut inputs.'
    output=mu.copy();factors={}
    for key in np.unique(groups):
        ix=groups==key;a=y[ix];b=np.maximum(mu[ix],1e-300)
        if a.sum()==0:gain=0.
        elif eps==0:gain=float(a.sum()/b.sum())
        else:
            r=1/eps**2
            def derivative(g):return float(np.sum(r*(a-g*b)/(r+g*b)))
            upper=max(float(a.sum()/b.sum()),1.)
            while derivative(upper)>0:upper*=2
            gain=float(brentq(derivative,0.,upper,xtol=1e-12,rtol=1e-12))
        output[ix]*=gain;factors[str(key)]=gain
    return output,factors


def _bounds(prototype,active,D_window):

    b={'D_nm':(2.,120.),'R_number_mean_nm':(.35,12.),'radius_sigma_log':(.03,.45),
       'spacing_disorder':(.04,.8),'height_nm':(.25,2.),
       'R_small_number_mean_nm':(.5,1.5),'secondary_number_ratio':(.1,500.),
       'burial_base_depth_nm':(-13.,13.),
       'polymer_thickness_nm':(20.,80.),'polymer_density_factor':(.5,1.8),
       'spectral_domain_nm':(3.,120.),'width_rel':(.08,1.3),'metal_fraction':(.05,.95),
       'film_thickness_nm':(.4,60.),'depth_corr_nm':(.3,50.),
       'buried_interface_rms_nm':(.05,3.),'critical_angle_deg':(.10,.27)}
    if D_window is not None:
        lo,hi=map(float,D_window)
        if not 0<lo<hi:raise ValueError('Frozen DEV physical D window required')
        b['D_nm']=(max(2.,lo),min(120.,hi))
    low=[];high=[]
    for name in np.array(NAMES)[active]:
        lo,hi=b[name]
        if not lo<hi:raise ValueError('Observed D window does not intersect declared model domain')
        if name=='burial_base_depth_nm':pass
        elif name=='metal_fraction':lo,hi=np.log(lo/(1-lo)),np.log(hi/(1-hi))
        else:lo,hi=np.log(lo),np.log(hi)
        low.append(lo);high.append(hi)
    return np.asarray(low),np.asarray(high)


def _to_internal(z,active,boundary_notes):
    v=np.asarray(z,float).copy()
    if active[INDEX['height_nm']]:v[INDEX['height_nm']]-=v[INDEX['R_number_mean_nm']]
    if active[INDEX['burial_base_depth_nm']]:
        b=np.expm1(v[INDEX['burial_base_depth_nm']]);R=np.exp(v[INDEX['R_number_mean_nm']]);H=np.exp(v[INDEX['polymer_thickness_nm']])
        f=b/min(2.5*R,H)
        if not 0<=f<=1:raise ValueError('Network burial proposal outside the declared generator fraction domain')
        if f in (0.,1.):
            boundary_notes.append(dict(coordinate='burial_fraction',network_value=float(f),numerical_start='finite logit +/-13; original surface/bottom limit retained'))
            v[INDEX['burial_base_depth_nm']]=-13. if f==0 else 13.
        else:v[INDEX['burial_base_depth_nm']]=np.log(f/(1-f))
    return v[active]


def _from_internal(x,z,active):
    v=np.asarray(z,float).copy();v[active]=x
    if active[INDEX['height_nm']]:v[INDEX['height_nm']]+=v[INDEX['R_number_mean_nm']]
    if active[INDEX['burial_base_depth_nm']]:
        f=1/(1+np.exp(-v[INDEX['burial_base_depth_nm']]))
        v[INDEX['burial_base_depth_nm']]=np.log1p(f*min(2.5*np.exp(v[INDEX['R_number_mean_nm']]),np.exp(v[INDEX['polymer_thickness_nm']])))
    return v


def _unit_signal(d):
    d=copy.deepcopy(d);p=np.asarray(d['p'],float)
    if d['module']=='TS_layer':p[[6,10]]=0.;p[[7,8,11,12]]=-700.
    else:p[[4,8]]=0.;p[[5,6,9,10]]=-700.
    d['p']=p.tolist();return d


def refine_modes(Y,E,proposals,*,energy_ev,alpha_i_deg,metal,input_policy,
                 reference=None,D_window=None,eps=.05,max_nfev=80,nsize=224,nz=64):
    'Refine each supplied physical mode, retaining all failure/source records.'
    if metal not in ('Ag','Au') or max_nfev<1 or eps<0:raise ValueError('Declared element/settings required')
    item=extract(Y,E,energy_ev,alpha_i_deg,metal=metal);obs=item['observations']
    part=selected_partition(Y,E,obs.get('q_star_nm_inv'),input_policy);keep=part['keep']
    if keep.sum()<12:raise ValueError('Fewer than12 original supported retained count groups')
    y=part['y'][keep];e=np.asarray(E,float);observed=np.divide(Y,e,out=np.zeros_like(e),where=e>0)
    scale=max(float(np.quantile(observed[e>0],.9)),1e-6)
    if reference is not None:
        ro=reference['observations'];preserve=reference.get('stationary_host_peak_preserved',False) or '_pet_' in reference.get('stem',reference['source']).lower()
        background=empirical_background(reference['Y'],reference['E'],None if preserve else ro.get('q_star_nm_inv'),None if preserve else ro.get('FWHM_nm_inv'))
        b90=max(float(np.quantile(background[e>0],.9)),1e-30);background=background/b90
        bg_source=reference['source']+'; conditional empirical reference, not pure host'+('; stationary host peak preserved' if preserve else '')
    else:
        background=np.broadcast_to((Q[None,None,:]/.5)**-2,e.shape).copy()
        background/=max(float(np.quantile(background[e>0],.9)),1e-30)
        bg_source='parameter power-law background; no supplied prestate'
    outputs=[]
    for proposal in proposals:
        started=time.perf_counter();z=np.asarray(proposal['chart_z'],float);active=np.asarray(proposal['active'],bool)
        d=copy.deepcopy(proposal['prototype']);d.update(energy_ev=float(energy_ev),alpha_i_deg=float(alpha_i_deg),metal=metal)
        result=dict(mode_id=proposal['mode_id'],branch=proposal.get('branch'),probability=proposal.get('probability'),
            network_chart_z=z.tolist(),active=active.tolist(),network_intervals=proposal.get('intervals'),
            model_error_floor=eps,background_source=bg_source,structural_validation='pending_new_end_to_end_simulation_TEST',
            refined_interval_scope='not a calibrated posterior; network intervals retained separately')
        try:
            initial=decode(d,z,active)
            if not initial['valid']:raise ValueError('Invalid original network physical proposal: '+str(initial['reasons']))
            boundary_notes=[];x0=_to_internal(z,active,boundary_notes);lo,hi=_bounds(d,active,D_window)
            if np.any(x0<lo) or np.any(x0>hi):raise ValueError('Network proposal outside frozen refinement bounds; not silently clipped')
            for k in np.flatnonzero((x0==lo)|(x0==hi)):
                boundary_notes.append(dict(coordinate=str(np.array(NAMES)[active][k]),network_internal_value=float(x0[k]),
                    numerical_start='inside boundary by1e-8 of declared span; original legal boundary retained'))
                x0[k]=lo[k]+1e-8*(hi[k]-lo[k]) if x0[k]==lo[k] else hi[k]-1e-8*(hi[k]-lo[k])
            result['numerical_boundary_start_notes']=boundary_notes
            structural=len(x0);numerical_start=decode(d,_from_internal(x0,z,active),active)
            if not numerical_start['valid']:raise ValueError('Numerical boundary start invalid: '+str(numerical_start['reasons']))
            unit=_unit_signal(numerical_start['parameters'])
            raw0=rates(unit,nsize=nsize,nz=nz);q90=max(float(np.quantile(raw0[e>0],.9)),1e-100)
            initial_key=tuple(x0)
            @lru_cache(maxsize=4)
            def structural_signal(key):
                if key==initial_key:return raw0/q90
                zz=_from_internal(np.asarray(key),z,active);decoded=decode(d,zz,active)
                if not decoded['valid']:raise ValueError('Coupled physical chart invalid: '+str(decoded['reasons']))
                return rates(_unit_signal(decoded['parameters']),nsize=nsize,nz=nz)/q90


            per_cut=input_policy['normalization']=='per_cut'
            nuisance0=np.r_[np.log([.1,.01] if per_cut else [scale,scale,scale*.1,scale*.01]),.015,.008,0.]
            count=2 if per_cut else 4;base_scale=1. if per_cut else scale
            lower=np.r_[lo,np.log([base_scale*1e-8]*count),.002,.002,-.3]
            upper=np.r_[hi,np.log([base_scale*1e6]*count),.035,.025,.3]
            def predict(x):
                zz=_from_internal(x[:structural],z,active);decoded=decode(d,zz,active)
                if not decoded['valid']:raise ValueError('Coupled physical chart invalid: '+str(decoded['reasons']))
                raw=structural_signal(tuple(x[:structural]))
                nx=x[structural:];side=1. if per_cut else np.exp(nx[:2])[:,None,None];offset=0 if per_cut else 2
                mean=response(raw*side+np.exp(nx[offset])*background+np.exp(nx[offset+1]),
                    nx[offset+2],nx[offset+3],tilt=nx[offset+4])
                mu=project_partition_rates(mean,e,part['partition_qstar'])[keep];gains={}
                if input_policy['normalization']=='per_cut':mu,gains=_profile_cut_scales(y,mu,part['groups'][keep],eps)
                return mu,mean,zz,decoded['parameters'],gains
            nuisance_fit=least_squares(lambda nx:residual(y,predict(np.r_[x0,nx])[0],eps),nuisance0,
                bounds=(lower[structural:],upper[structural:]),max_nfev=20,x_scale='jac',ftol=2e-6,xtol=2e-6,gtol=1e-5)
            x=np.r_[x0,nuisance_fit.x];network_mu,_,_,_,_=predict(x)
            sol=least_squares(lambda v:residual(y,predict(v)[0],eps),x,bounds=(lower,upper),
                max_nfev=max_nfev,x_scale='jac',ftol=2e-6,xtol=2e-6,gtol=1e-5)
            mu,mean,zz,physical,gains=predict(sol.x);parameter_count=len(sol.x)+len(gains)
            scores=compare_floors(y,mu,parameter_count)
            for score in scores:
                score['asymptotic_tail_probability']=float(chi2.sf(score['deviance'],score['dof']))
                score['compatible_at_5pct_asymptotic']=bool(score['asymptotic_tail_probability']>=.05)
                score['compatibility_scope']='uncalibrated asymptotic model-level reference; end-to-end TEST required'
            result.update(status='refined' if sol.success else 'refinement_iteration_limit',refined_chart_z=zz.tolist(),
                parameters=physical,optimizer_success=bool(sol.success),optimizer_nfev=int(sol.nfev),nuisance=sol.x[structural:].tolist(),
                refined_boundary_flags=((sol.x-lower<.002*(upper-lower))|(upper-sol.x<.002*(upper-lower))).tolist(),
                network_initial_scores=compare_floors(y,network_mu,parameter_count),refined_scores=scores,
                network_score_scope='fixed network structural proposal with separately fitted measurement nuisance; not structural recovery proof',
                network_nuisance=nuisance_fit.x.tolist(),network_nuisance_nfev=int(nuisance_fit.nfev),
                profiled_per_cut_gains=gains,profile_scope='each retained side/band freely scaled; no restored AF amplitude information' if gains else 'global relative AF response retained',
                observed_partition_counts=y.tolist(),network_partition_counts=network_mu.tolist(),refined_partition_counts=mu.tolist(),
                selected_partition_kind=part['kinds'][keep].tolist(),selected_partition_group=part['groups'][keep].tolist(),
                selected_partition_coordinate=part['coordinates'][keep].tolist(),selected_partition_exposure=part['e'][keep].tolist(),
                coordinate_units='H:q_y nm^-1; V:alpha_f deg',input_policy=input_policy,
                predicted_cuts_unprofiled_native=extract(mean*e,e,energy_ev,alpha_i_deg,metal=metal,integer=False),
                curve_scope='selected partition curves include fitted per-cut gains; unprofiled native cuts are distinct, not a common gain-corrected 2D image',
                native_forward_replay=dict(unit_signal_parameters=_unit_signal(physical),fixed_initial_physics_q90=q90),
                local_parameter_count=parameter_count,native_orders=dict(nsize=nsize,nz=nz))
        except (ValueError,FloatingPointError,OverflowError) as exc:result.update(status='preserved_network_proposal_refinement_not_completed',reason=str(exc))
        result['wall_seconds']=time.perf_counter()-started;outputs.append(result)
    return dict(observations=obs,modes=outputs,partition_scope=part['scope'],
        discarded_cuts_scope='predicted for comparison, never silently restored to refinement objective',
        interpretation='conditional multimodal proposals; fit agreement and structure recovery reported separately')
