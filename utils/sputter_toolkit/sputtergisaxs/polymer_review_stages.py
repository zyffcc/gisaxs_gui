"""Preregistered observation-history stages and periodicity for development r1.

No connectivity truth, zero-metal reference or sample-motion cause is inferred.
All thresholds are working observation rules recorded before this rerun.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from scipy.stats import spearmanr
from scipy.signal import lombscargle, lfilter


def number(v):
    try:
        x=float(v)
        return x if np.isfinite(x) else np.nan
    except (TypeError,ValueError):return np.nan


def stages(rows):
    t=np.array([r['time_s'] for r in rows],float)
    if not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Ordered header times required')
    obs=[r['observations'] for r in rows]
    y=np.array([number(o.get('Yoneda_deg')) for o in obs]);rate=np.array([number(o.get('intensity_rate')) for o in obs])
    se=np.array([number(o.get('intensity_rate_SE')) for o in obs])
    initial=(t-t[0]<=2)&np.isfinite(y);base_y=float(np.median(y[initial])) if initial.any() else np.nan
    initial_rate=(t-t[0]<=2)&np.isfinite(rate);rr=rate[initial_rate]
    low=(min(float(np.median(rr)+max(3*1.4826*np.median(abs(rr-np.median(rr))),3*np.nanmedian(se[initial_rate]))),
             float(np.nanquantile(rate,.25))) if len(rr) else np.nan)
    # Trailing smoothing: no future ridge information in a row's history.
    smooth=np.array([np.nanmedian(y[(t<=ti)&(t>=ti-5)]) if np.isfinite(y[(t<=ti)&(t>=ti-5)]).any() else np.nan for ti in t])
    result=[];tracks={};moving_seen=False;last_visible=None;ever_moved=False;last_band=None
    for i,(r,o) in enumerate(zip(rows,obs)):
        ti=t[i];q=number(o.get('q_star_nm_inv'));visible=np.isfinite(q) and o['feature_status'] in ('bilateral','one_side')
        yi=smooth[i];shift=yi-base_y;ever_moved=ever_moved or bool(np.isfinite(shift) and abs(shift)>=.02)
        use=(t<=ti)&(t>=ti-10)&np.isfinite(smooth)
        plateau=False;yslope=None
        if use.sum()>=5 and np.ptp(t[use])>=8:
            yslope=float(np.polyfit(t[use]-ti,smooth[use],1)[0])
            plateau=bool(np.ptp(np.quantile(smooth[use],[.1,.9]))<=.02 and abs(yslope)<=.001)
        moving=False;qrho=None;qslope=None
        if visible:
            band=o['feature_AF_band']
            if last_band is not None and band!=last_band:
                # Do not inherit lower-q motion from a different AF branch.
                moving_seen=False;tracks[band]=[]
            last_band=band;track=tracks.setdefault(band,[])
            if track and (q/track[-1][1]>1.5 or q/track[-1][1]<.5):track.clear();moving_seen=False
            track.append((ti,q));recent=[x for x in track if x[0]>=ti-10]
            if len(recent)>=5 and recent[-1][0]-recent[0][0]>=2.5:
                tt,qq=np.asarray(recent).T
                if np.ptp(qq)>0:
                    qrho=float(spearmanr(tt,qq).statistic);qslope=float(np.polyfit(tt-tt[0],qq,1)[0])
                    drop=float(np.median(qq[:max(2,len(qq)//3)])-np.median(qq[-max(2,len(qq)//3):]))
                    moving=bool(qrho<=-.6 and qslope<0 and drop>=max(.025,.05*qq[0]))
            moving_seen=moving_seen or moving;last_visible=ti
        lost=not visible and last_visible is not None and ti-last_visible>=1
        late=bool(lost and moving_seen and ever_moved and plateau)
        low_count=bool(np.isfinite(rate[i]) and rate[i]<=low)
        unmoved=bool(np.isfinite(shift) and abs(shift)<=.01 and not ever_moved)
        if late:
            stage='coalescence_or_film_candidate';reason='Peak tracked to lower q then lost; shifted Yoneda now plateau; connectivity not proven'
        elif visible and o['feature_status']=='one_side' and moving:
            stage='coalescence_candidate';reason='One-side peak with same-band lower-q motion'
        elif visible:
            stage='separate_islands_candidate';reason='Resolved peak; island interpretation conditional on host contribution'
        elif low_count and unmoved and not moving_seen:
            stage='pre_nucleation_candidate';reason='Low relative count rate and no observed Yoneda shift; not proven zero metal'
        else:
            stage='stage_unresolved';reason='Requested motion/loss/shift/plateau or low-count evidence incomplete'
        result.append(dict(stage=stage,stage_reason=reason,Yoneda_shift_deg=number(o.get('Yoneda_deg'))-base_y,
            Yoneda_smoothed_shift_deg=shift,ridge_platform=plateau,ridge_slope_deg_s=yslope,
            ever_Yoneda_moved=ever_moved,peak_lowq_motion=moving,peak_lowq_motion_history=moving_seen,
            q_history_spearman=qrho,q_history_slope=qslope,low_count=low_count,
            intensity_low_threshold=low,reference_Yoneda_deg=base_y,
            stage_scope='Observation-conditioned development candidate; no measured topology'))
    return result,dict(counts=dict(Counter(r['stage'] for r in result)),reference_Yoneda_deg=base_y,
        low_count_threshold=low,reference='Earliest observed0-2s; not independently zero-metal',
        no_model_or_stage_truth=True)


def detrend_local(t,x,width=60.):
    t=np.asarray(t,float);x=np.asarray(x,float)
    lo=np.searchsorted(t,t-width/2);hi=np.searchsorted(t,t+width/2,side='right');n=hi-lo
    def sums(a):
        cs=np.r_[0.,np.cumsum(a)];return cs[hi]-cs[lo]
    st,sx,stt,stx=sums(t),sums(x),sums(t*t),sums(t*x)
    slope=np.divide(stx-st*sx/n,stt-st*st/n,out=np.zeros_like(t),where=(stt-st*st/n)>0)
    return x-(sx/n+slope*(t-st/n))


def periodicity(rows,seed=202610211621):
    """AR1 max-frequency surrogate check, shared peak and split stability.

    A pass establishes repeatable scattering modulation only. Irregular/gappy
    sampling is rejected rather than silently interpolated into periodicity.
    """
    t=np.array([r['time_s'] for r in rows]);use=t>=5;t=t[use]
    keys=['Yoneda_deg','highq_log_slope'];values={k:np.array([number(r['observations'].get(k)) for r in rows])[use] for k in keys}
    keep=np.isfinite(t)
    for x in values.values():keep &= np.isfinite(x)
    t=t[keep];values={k:x[keep] for k,x in values.items()}
    if len(t)<100 or np.ptp(t)<100:return dict(confirmed=False,reason='Fewer than5 target20s cycles or100 supported bins',supported_bins=len(t))
    dt=float(np.median(np.diff(t)))
    if np.max(np.diff(t))>1.6*dt:return dict(confirmed=False,reason='Sampling gap; no interpolation for periodicity gate',supported_bins=len(t))
    freq=np.linspace(1/40,1/10,301);omega=2*np.pi*freq;tt=t-t[0];rng=np.random.default_rng(seed);records={}
    for k,x in values.items():
        z=detrend_local(tt,x);z-=z.mean();power=lombscargle(tt,z,omega,normalize=True);j=int(np.argmax(power));period=float(1/freq[j])
        if np.std(z)<1e-12 or not np.isfinite(power).all():
            return dict(confirmed=False,reason='No finite nonconstant detrended signal in '+k,supported_bins=len(t))
        rho=float(np.clip(np.corrcoef(z[:-1],z[1:])[0,1],-.95,.95)) if np.std(z)>0 else 0.
        maxima=[]
        for b in range(399):
            noise=rng.normal(size=len(z)+200);sur=lfilter([1.],[1.,-rho],noise)[200:]
            # The null receives the same detrending as the observed feature.
            sur=detrend_local(tt,sur);sur-=sur.mean();maxima.append(float(lombscargle(tt,sur,omega,normalize=True).max()))
        p=(1+sum(v>=power[j] for v in maxima))/400
        halves=[]
        for h in (tt<=np.median(tt),tt>np.median(tt)):
            hz=detrend_local(tt[h],x[h]);hz-=hz.mean();hp=lombscargle(tt[h],hz,omega,normalize=True)
            halves.append(dict(period_s=float(1/freq[np.argmax(hp)]),cycles=float(np.ptp(tt[h])/period)))
        records[k]=dict(period_s=period,max_frequency_corrected_AR1_p=float(p),rho=rho,peak_normalized_power=float(power[j]),
            halves=halves,stable=bool(all(h['cycles']>=3 and abs(h['period_s']/period-1)<=.15 for h in halves)))
    periods=[r['period_s'] for r in records.values()]
    passed=all(15<=r['period_s']<=25 and r['max_frequency_corrected_AR1_p']<=.01 and r['stable'] for r in records.values()) and max(periods)/min(periods)<=1.1
    return dict(confirmed=bool(passed),period_s=float(np.mean(periods)) if passed else None,features=records,
        supported_bins=len(t),seed=seed,surrogates=399,search_period_s=[10,40],
        interpretation='Repeatable scattering modulation only; sample movement needs position/encoder log',
        reason='Shared stable20s-band periodicity passed registered gate' if passed else 'Shared period/significance/split stability not all established')
