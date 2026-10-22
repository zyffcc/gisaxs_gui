"""Development review feature branch; frozen feature/training modules unchanged.

Same empirical power-law+Gaussian/Poisson evidence thresholds as the delivered
v0, with explicit low-q starts and actual support-edge checks. No boundary
extrapolation is promoted to a measured peak.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from .features import rate_model,rate_jac,dev_residual,AF_BANDS,V_EDGES,ridge
from .linecut_v3 import NATIVE_AF,direct_features
from .tool_qinput import Q,Q_EDGES

def fit_review_peak(q,counts,den,lo=.05,hi=1.75,background='power'):
    q,counts,den=map(lambda x:np.asarray(x,float),(q,counts,den))
    good=(q>=lo)&(q<=hi)&(den>0)&np.isfinite(counts)&(counts>=0)
    if good.sum()<12 or counts[good].sum()<100:return dict(status='insufficient_support_or_counts',q_star=None,fwhm=None,D_corr=None)
    x,y,n=q[good],counts[good],den[good];rates=y/n;scale=max(float(np.quantile(rates,.8)),1e-6)
    def solve(initial,peak):
        lower=[0.,0.,0.];upper=[1e3,1e3,5.]
        if peak:lower+=[0.,lo+.01,.02];upper+=[1e3,hi-.01,.6]
        def fun(p):return dev_residual(y,rate_model(x,p,background,peak)*scale*n)
        def jac(p):
            mu=np.maximum(rate_model(x,p,background,peak)*scale*n,1e-12);res=dev_residual(y,mu)
            factor=np.divide(1-y/mu,res,out=np.ones_like(mu)/np.sqrt(np.maximum(mu,1e-12)),where=abs(res)>1e-5)
            return rate_jac(x,p,background,peak)*(factor*scale*n)[:,None]
        return least_squares(fun,np.clip(initial,np.array(lower)+1e-9,np.array(upper)-1e-9),jac=jac,bounds=(lower,upper),x_scale='jac',max_nfev=350,ftol=1e-9,xtol=1e-9,gtol=1e-7)
    null=solve([max(float(np.median(rates[-4:]))/scale,.001),.2,1.],False)
    excess=rates-rate_model(x,null.x,background)*scale;peaks=find_peaks(excess)[0];peaks=peaks[(x[peaks]>lo+.0125)&(x[peaks]<hi-.06)]
    centre=float(x[peaks[np.argmax(excess[peaks])]]) if len(peaks) else float(np.clip(x[np.argmax(excess)],lo+.07,hi-.07))
    starts=[centre,float(np.clip(.57,lo+.07,hi-.07))]
    # Explicit low-q starts; thresholds and full-flank resolution test retained.
    starts += [float(x[k]) for k in peaks if x[k]<.25]
    starts += [float(np.clip(v,lo+.0125,hi-.07)) for v in (.0875,.1375)]
    fits=[solve([*null.x,max(float(excess.max())/scale,.05),start,width],True)
          for start in sorted(set(starts)) for width in (.03,.07,.16)]
    f=min(fits,key=lambda f:float(f.fun@f.fun));p=f.x;dev=float(f.fun@f.fun);dev0=float(null.fun@null.fun)
    cov=np.linalg.pinv(f.jac.T@f.jac,rcond=1e-11)*max(1.,dev/max(len(x)-6,1));se=np.sqrt(np.maximum(np.diag(cov),0.))
    q0=float(p[4]);sig=float(p[5]);width=float(2*np.sqrt(2*np.log(2))*sig);bracket=int(np.searchsorted(x,q0));gaps=np.diff(x[(x>q0-sig)&(x<q0+sig)])
    gap=bool(bracket==0 or bracket==len(x) or x[bracket]-x[bracket-1]>.051 or (len(gaps)>0 and gaps.max()>.051))
    edge=bool(q0-width/2<=max(lo,x[0]-.0125) or q0+width/2>=min(hi,x[-1]+.0125) or sig<=.0205 or sig>=.59)
    aic=dev0-dev-6;resolved=bool(f.success and aic>10 and p[3]>2*se[3] and not edge and not gap)
    return dict(status='resolved_empirical_feature' if resolved else 'unresolved_or_boundary_feature',q_star=q0,q_star_fit_se=float(se[4]),fwhm=width,fwhm_fit_se=float(2*np.sqrt(2*np.log(2))*se[5]),D_corr=float(2*np.pi/q0),D_corr_fit_se=float(2*np.pi*se[4]/q0**2),width_over_qstar=width/q0,peak_amplitude=float(p[3]*scale),peak_amplitude_fit_se=float(se[3]*scale),delta_AIC=float(aic),deviance=dev,null_deviance=dev0,optimizer_success=bool(f.success),edge_or_width_bound=edge,gap_near_peak=gap,parameters=(p*np.array([scale,scale,1,scale,1,1])).tolist(),background=background,window=[lo,hi],statistical_error='Residual-scaled conditional covariance, not complete systematic CI')

def features_review(Y,E):
    Y,E=np.asarray(Y,float),np.asarray(E,float)
    fits=[];bands=[];lower=[]
    for b,(lo,hi) in enumerate(AF_BANDS):
        af=(NATIVE_AF>=lo)&(NATIVE_AF<hi);pair=[]
        for side in range(2):
            y=Y[side,af].sum(0);e=E[side,af].sum(0);support=np.flatnonzero(e>0)
            low=float(Q_EDGES[support[0]]) if len(support) else None
            fit=fit_review_peak(Q,y,e,lo=max(.05,low)) if low is not None else dict(status='no_support')
            pair.append(fit);fits.append(dict(AF_band=b,side=side,lower_q_edge=low,**fit));lower.append(low)
        valid=[f for f in pair if f['status']=='resolved_empirical_feature'];qs=[f['q_star'] for f in valid]
        agree=len(valid)==2 and np.ptp(qs)/np.mean(qs)<=.1
        bands.append(dict(AF_band=b,af_range_deg=list(AF_BANDS[b]),
            support='bilateral' if agree else 'one_side' if len(valid)==1 else 'bilateral_conflict' if len(valid)==2 else 'unresolved',
            q_star=float(np.mean(qs)) if agree or len(valid)==1 else None,
            FWHM=float(np.mean([f['fwhm'] for f in valid])) if agree or len(valid)==1 else None,
            q_se=float(max(f['q_star_fit_se'] for f in valid)) if agree or len(valid)==1 else None))
    candidates=[b for b in bands if b['support']=='bilateral'] or [b for b in bands if b['support']=='one_side']
    selected=candidates[0] if candidates else None
    # Ridge uses lowest unmasked support too; preserve historical .15 cutoff
    # separately for a like-for-like comparison of Yoneda extraction.
    def ridge_range(qlo):
        qs=(Q>=qlo)&(Q<1.2);vi=np.searchsorted(V_EDGES,NATIVE_AF,side='right')-1;ok=(vi>=0)&(vi<len(V_EDGES)-1)
        vy=np.bincount(vi[ok],weights=Y[:,:,qs].sum((0,2))[ok],minlength=len(V_EDGES)-1)
        ve=np.bincount(vi[ok],weights=E[:,:,qs].sum((0,2))[ok],minlength=len(V_EDGES)-1)
        return ridge(vy,ve)
    rr=ridge_range(.05);oldridge=ridge_range(.15)
    y,e=Y.sum((0,1)),E.sum((0,1));good=(Q>=1)&(Q<=1.75)&(y>0)&(e>0)
    slope=float(np.polyfit(np.log(Q[good]),np.log(y[good]/e[good]),1,w=np.sqrt(y[good]))[0]) if good.sum()>=8 else None
    return dict(q_star_nm_inv=selected['q_star'] if selected else None,
        D_corr_nm=2*np.pi/selected['q_star'] if selected else None,FWHM_nm_inv=selected['FWHM'] if selected else None,
        q_star_fit_se_nm_inv=selected['q_se'] if selected else None,
        feature_status=selected['support'] if selected else 'no_resolved_peak',feature_AF_band=selected['AF_band'] if selected else None,
        feature_AF_range_deg=selected['af_range_deg'] if selected else None,
        Yoneda_deg=rr.get('alpha_f_observed_ridge'),Yoneda_status=rr['ridge_status'],Yoneda_old_q015_deg=oldridge.get('alpha_f_observed_ridge'),
        highq_log_slope=slope,highq_reason='Total scattering apparent log slope,1.0-1.75nm^-1; not metal-only R',
        q_support_lower_by_band_side=lower,q_support_lower_nm_inv=min((v for v in lower if v is not None),default=None),
        lowq_boundary_candidates=[f for f in fits if f.get('delta_AIC',0)>10 and f.get('edge_or_width_bound') and f.get('q_star',2)<.25],
        bands=bands,peak_fits=fits,peak_attribution='Total scattering; host/metal not uniquely separated',
        intensity_rate=float(Y.sum()/E.sum()) if E.sum()>0 else None,
        intensity_rate_SE=float(np.sqrt(Y.sum())/E.sum()) if E.sum()>0 else None)
