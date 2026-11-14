'Budgeted polymer development refinement; unchanged native forward kernel.'
from __future__ import annotations
import copy, time, signal
import numpy as np
from scipy.optimize import least_squares
from .neural_refine_input import Q,NATIVE_AF,project_partition_rates
from .features import AF_BANDS,fit_peak
from .neural_refine_chart import decode
from .neural_refine_infer import physical_values
from .neural_refine_refine import selected_partition,_to_internal,_from_internal,_bounds,_unit_signal
from .neural_refine_bounded import background_contract,profile_components,scores
from .neural_refine_forward import rates
from .neural_refine_measurement import response
from .neural_refine_likelihood import residual


def policy():
    ids=[]
    for b,lo,hi in [(2,.8,1.2),(3,.25,.8)]:
        ids.extend((b*2*len(Q)+np.flatnonzero(np.tile((Q>=lo)&(Q<hi),2))).tolist())
    return dict(horizontal=[2,3],vertical=False,highq_only=False,common_signal_indices=ids)


def quick_refine(Y,E,proposal,observations,reference,*,energy_ev,alpha_i_deg,metal,
                 actual_metal=None,seconds=45.,max_nfev=6):
    started=time.monotonic();best={};result=dict(status='network_only',reason='not_started',
        recipe='polymer_dev_v0_fixed_retained_regions_peak_features_bounded_background',
        feature_scope='q/FWHM working 0.025/0.05 nm^-1 scales; correlated descriptor error not calibrated',
        compatibility_scope='5% model-error count-only asymptotic score; excludes feature penalties; not structural validation',
        PSF_q_nm_inv=.015,AF_blur_deg=.008,native_orders=dict(nsize=224,nz=64),
        refined_intervals='Not calibrated; retain network component conditional ranges',Dcorr_truncation=False)
    def expired(*unused):raise TimeoutError('per_frame_refinement_budget')
    old_handler=None
    if hasattr(signal,'setitimer'):
        old_handler=signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,max(.01,seconds))
    try:
        part=selected_partition(Y,E,None,policy());keep=part['keep'].copy()


        band=observations.get('feature_AF_range_deg');has_peak=observations.get('q_star_nm_inv') is not None and band is not None
        if has_peak:
            for b,(lo,hi) in enumerate(AF_BANDS):
                if max(lo,band[0])<min(hi,band[1]):keep &= ~np.char.startswith(part['groups'],f'H{b}_')
        result['feature_band_removed_from_count_likelihood']=band if has_peak else 'No resolved peak; no feature penalty'
        if keep.sum()<12:raise ValueError('Too few disjoint supported retained bins')
        y,e=part['y'][keep],part['e'][keep];groups,q=part['groups'][keep],part['coordinates'][keep]
        rp=selected_partition(reference['Y'],reference['E'],None,policy())
        bg,flat,contract,_=background_contract(y,e,rp['y'][keep],rp['e'][keep],q,groups,observations,reference['observations'])
        result['background_contract']=contract
        result['flat_scope']='Bounded flat fluorescence allowed; Cu absorption/optics remain Ag approximation' if actual_metal=='Cu' else 'Bounded flat detector/background component'
        z=np.asarray(proposal['chart_z']);active=np.asarray(proposal['active'],bool);prototype=copy.deepcopy(proposal['prototype'])
        prototype.update(energy_ev=energy_ev,alpha_i_deg=alpha_i_deg,metal=metal)
        if not decode(prototype,z,active)['valid']:raise ValueError('Original component centre outside decoder domain')
        notes=[];x0=_to_internal(z,active,notes);lo,hi=_bounds(prototype,active,None)
        if np.any(x0<lo) or np.any(x0>hi):raise ValueError('Original centre outside physical generator domain; not clipped/redrawn')
        x0=np.clip(x0,lo+1e-8*(hi-lo),hi-1e-8*(hi-lo))
        def prediction(x):
            if time.monotonic()-started>seconds:expired()
            chart=_from_internal(x,z,active);decoded=decode(prototype,chart,active)
            if not decoded['valid']:raise ValueError('Invalid coupled chart')
            raw=response(rates(_unit_signal(decoded['parameters']),nsize=224,nz=64),.015,.008)
            if actual_metal=='Cu':raw=raw*.89**2
            s=project_partition_rates(raw,E,None)[keep]
            mu,metal_counts,nuisance=profile_components(y,s,bg,flat,groups,.05)
            feature_res=[];feature_pred={}
            if has_peak:
                af=(NATIVE_AF>=band[0])&(NATIVE_AF<band[1]);ey=E[:,af].sum((0,1))
                rate=np.divide((raw[:,af]*E[:,af]).sum((0,1)),ey,out=np.zeros(len(Q)),where=ey>0)


                signal_y=rate/max(float(np.max(rate)),1e-100)*1e5
                peak=fit_peak(Q,signal_y,np.where(ey>0,1.,0.),lo=.05,hi=1.75)
                qp=peak.get('q_star');fw=peak.get('fwhm')
                if qp is None or fw is None:feature_res=[10.,10.];feature_pred=dict(status='No resolved predicted metal peak')
                else:
                    feature_res=[(qp-observations['q_star_nm_inv'])/.025,(fw-observations['FWHM_nm_inv'])/.05]
                    feature_pred=dict(q_star_nm_inv=float(qp),FWHM_nm_inv=float(fw))
            rr=np.r_[residual(y,mu,.05),feature_res];loss=float(rr@rr)
            if not best or loss<best['loss']:
                best.update(loss=loss,chart=chart,mu=mu,metal_counts=metal_counts,nuisance=nuisance,feature_prediction=feature_pred)
            return rr
        sol=least_squares(prediction,x0,bounds=(lo,hi),max_nfev=max_nfev,x_scale='jac',ftol=2e-5,xtol=2e-5,gtol=1e-4)
        result.update(status='refined' if sol.success else 'refinement_iteration_limit',reason='Converged within fixed budget' if sol.success else 'Fixed6-evaluation iteration limit',optimizer_nfev=int(sol.nfev))
    except (TimeoutError,ValueError,FloatingPointError,OverflowError) as exc:
        result.update(status='budget_limited_partial_refinement' if best else 'network_only',reason=str(exc))
    finally:
        if old_handler is not None:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
    if best:
        values=physical_values(best['chart'],active);values['height_nm']=values['R_number_mean_nm']
        result.update(values=values,height_scope='Hemisphere h=R condition; not independently fitted',
            model_scores=scores(y,best['mu'],len(x0)+3*len(best['nuisance'])),
            q=q.tolist(),group=groups.tolist(),observed_counts=y.tolist(),predicted_counts=best['mu'].tolist(),
            feature_prediction=best['feature_prediction'],objective=float(best['loss']),nuisance=best['nuisance'])
    result['wall_seconds']=time.monotonic()-started
    return result
