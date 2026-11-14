'Conditional v3 sequence indicators and QCM geometry, no new fitting.'
import numpy as np


def conditional_interpretations(observations,fit=None,*,change=False):
    'Interpretation alternatives, not extra fitted models or measured topology.'
    if change:
        return [dict(scenario='scaled_persistent_reference_plus_additional_population',
            quantity='R is an additional-population form-factor radius under the declared shape/response, not total metal radius',
            distinguish='same-position pre-deposition reference, QCM and end-state SEM/AFM'),
            dict(scenario='reference_response_or_preexisting_population_changed',
            quantity='Part of the signed change may be optical reweighting, rearrangement or host-metal interference; apparent R is not uniquely assigned to new metal',
            distinguish='compare reference-gain/shape/AF conditions; track separate Yoneda branches; independent microscopy')]
    supported=bool(fit and fit.get('peak_support') in ('bilateral','one_side'))
    return [dict(scenario='metal_islands',quantity='D is a conditional1D correlation step; R is a conditional number-mean island radius' if supported else 'No measured island spacing without an aligned peak; a fitted R remains a shape/background condition',
        distinguish='SEM isolated islands and lateral size statistics; compare fitted variants and QCM mass closure'),
        dict(scenario='host_template_or_mixed_host_metal_peak',quantity='Observed q* may describe host ordering rather than metal nearest neighbours; fitted form-factor scale can be a contrast-decoration scale',
        distinguish='Peak already in pre-deposition frame or fixed through growth supports template attribution; compare same-position reference'),
        dict(scenario='coalesced_or_connected_domains',quantity='Island D/R can be effective correlation/form-factor scales; this fit does not measure strand width or establish connected topology',
        distinguish='SEM connectivity, transport or independent percolation evidence;2R/D crossing1 alone is insufficient')]


def attach_qcm(variants,rate_nm_s,t0_s,time_s):
    if rate_nm_s is None or t0_s is None:return 'rate_or_t0_missing; no absolute thickness inferred'
    if time_s is None:return 'time_missing; QCM inputs supplied but not applied'
    if not np.isfinite([rate_nm_s,t0_s,time_s]).all() or rate_nm_s<=0:
        raise ValueError('Positive finite QCM rate and finite same-clock time/t0 required')
    from .linecut_v3 import qcm_geometry
    delta=float(rate_nm_s*max(time_s-t0_s,0.));applied=0
    for fit in variants:
        if fit.get('status')!='conditional_linecut_fit' or fit['shape']!='cylinder':continue
        D,R,sig=fit['D_nm'],fit['R_number_mean_nm'],fit['radius_sigma_log']
        d_lo,dh=fit['D_local95_nm'];rl,rh=fit['R_local95_nm']
        se=fit['parameter_se'][2]
        sl,sh=max(.03,sig-1.96*se),min(.45,sig+1.96*se)
        if not fit['optimizer_success'] or fit.get('weak_direction_weights',[0,0,0])[2]>.01:sl,sh=.03,.45
        fp_low=2*np.pi*rl*rl*np.exp(sl*sl)/(np.sqrt(3)*dh*dh)
        fp_high=2*np.pi*rh*rh*np.exp(sh*sh)/(np.sqrt(3)*d_lo*d_lo)
        geometry=[]
        for f in (1.,1.5,2.):
            g=qcm_geometry(D,R,sig,delta,area_factor=f)
            hbounds=[delta/(f*fp_high),delta/(f*fp_low)]
            g.update(equivalent_thickness_nm=delta,rate_input_nm_s=rate_nm_s,t0_input_s=t0_s,time_input_s=time_s,
                height_conditional_box_nm=hbounds,aspect_conditional_box=[hbounds[0]/rh,hbounds[1]/rl],
                footprint_conditional_box=[float(fp_low),float(fp_high)],gap_conditional_box_nm=[d_lo-2*rh,dh-2*rl],
                range_scope='envelope of conditional marginal D/R/sigma curvature boxes at fixed supplied rate/t0; not a calibrated joint95 interval or QCM systematic uncertainty',
                spacing_scope=fit['peak_support'],
                applicability='formal volume-equivalent only; isolated-island condition conflicts with overlap' if g['overlap_warning'] else 'conditional isolated-population mass closure, not independently measured height')
            geometry.append(g)
        fit['qcm_conditions']=geometry;applied+=1
    return 'conditional_QCM_applied_to_cylinder_variants' if applied else 'QCM_inputs_known_but_no_fitted_cylinder_variant; hemisphere_height_not_inferred'


def stage_candidates(items,best):
    'Predetermined descriptive rules; all supplied windows retained.'
    known=all(d['metadata'].get('time_s') is not None for d in items)
    t=np.array([d['metadata']['time_s'] for d in items],float) if known else np.arange(len(items),dtype=float)
    unit='header_seconds_from_supplied_origin_not_shutter' if known else 'input_frame_index_time_unknown'
    events=[]
    detected=[i for i,d in enumerate(items) if d['observations'].get('feature_status') in ('bilateral','one_side')]
    if detected:
        i=detected[0];events.append(dict(kind='correlations_first_observed',coordinate=float(t[i]),unit=unit,
            left_censored=i==0,source_key=items[i]['key'],
            interpretation='nucleation/correlation visibility bound, not metal-arrival or nucleation time; broad-band feature may be a host peak',
            distinguish='compare pre-deposition spectrum and SEM; earliest supplied frame may already contain metal'))
    def eligible(i):
        b=best[i]
        return bool(b and b.get('status')=='conditional_linecut_fit' and b['optimizer_success'] and
                    b.get('peak_support') in ('bilateral','one_side') and b.get('asymptotic_gof_p',0)>=.05 and
                    all(b.get('local_range_sources',{}).get(k)=='conditional_local_log_curvature_1.96SE' for k in ('D','R')))
    for i in range(2,len(items)):
        ids=(i-2,i-1,i)
        if not all(eligible(j) for j in ids) or not np.all(np.diff(t[list(ids)])>0):continue
        fits=[best[j] for j in ids]
        if len({(b['shape'],b['approximation']) for b in fits})!=1:continue
        R=np.array([b['R_number_mean_nm'] for b in fits]);D=np.array([b['D_nm'] for b in fits])
        change_se=np.hypot(fits[0]['R_local_se_nm'],fits[-1]['R_local_se_nm'])
        if np.all(np.diff(R)>0) and np.all(np.diff(D)>=0) and np.all(2*R<D) and R[-1]-R[0]>2*change_se:
            events.append(dict(kind='island_growth_compatible_interval',interval=[float(t[i-2]),float(t[i])],unit=unit,
                interpretation='conditional island growth, compatible with capture/diffusion; mechanism not identified; preliminary curve-GOF gate only',
                distinguish='check QCM mass balance, shape variants and SEM; local curvature errors omit systematic model error'))
            break
    for i in range(1,len(items)):
        if not (eligible(i-1) and eligible(i)) or t[i]<=t[i-1]:continue
        a,b=best[i-1],best[i]
        if (a['shape'],a['approximation'])!=(b['shape'],b['approximation']):continue
        if 2*a['R_number_mean_nm']/a['D_nm']<1<=2*b['R_number_mean_nm']/b['D_nm']:
            events.append(dict(kind='crowding_or_island_model_breakdown_candidate',interval=[float(t[i-1]),float(t[i])],unit=unit,
                interpretation='mean-diameter/D crosses1 under the fitted geometry: possible coalescence OR failure of isolated-island mapping, not measured percolation',
                distinguish='SEM connectivity/electrical conductance; broad size distributions and host peaks can mimic this indicator'))
            break
    ridge=np.array([d['observations'].get('Yoneda_deg') if d['observations'].get('Yoneda_deg') is not None else np.nan for d in items])
    good=np.isfinite(ridge[:-1])&np.isfinite(ridge[1:])&(np.diff(t)>0)
    if good.any():
        steps=np.where(good,abs(np.diff(ridge)),np.nan);i=int(np.nanargmax(steps))
        if steps[i]>=.03:
            events.append(dict(kind='dominant_ridge_change_candidate',interval=[float(t[i]),float(t[i+1])],unit=unit,change_deg=float(ridge[i+1]-ridge[i]),
                interpretation='optical-loading/ridge-switch observation, not a unique density or phase transition; may accompany wetting/coalescence',
                distinguish='track separate ridge branches, compare reference and QCM; no absolute metal amount from ridge alone'))
    return dict(schema='v3_conditional_stage_indicators_1',original_windows=len(items),unit=unit,events=events,
        percolation='not established: fitted size/spacing or optical changes do not establish connected topology',
        missing_stages='No forced nucleation/diffusion/coalescence/percolation date when supplied observations/conditions do not support it')
