'Observed reference-relative scattering, not morphology or absolute metal mass.'
import numpy as np
from .features import Q_EDGES

def _fit_linear(x,y,var):
    w=1/np.maximum(var,1e-20);X=np.column_stack([x,np.ones(len(x))]);normal=X.T@(w[:,None]*X);cov=np.linalg.pinv(normal);b=cov@(X.T@(w*y))
    if b[0]<0:b=np.array([0.,float(np.sum(w*y)/w.sum())])
    loss=float(np.sum(w*(X@b-y)**2));return b,loss,cov

def fit_excess_length(q,difference,variance):
    q,y,v=map(lambda x:np.asarray(x,float),(q,difference,variance));ok=(q>=.05)&(q<.3)&np.isfinite(y)&np.isfinite(v)&(v>=0)
    result={'xi_nm':None,'xi_lo95_nm':None,'xi_hi95_nm':None,'status':'insufficient_lowq_support','interpretation':'Conditional Debye-Bueche change length; not pore/particle size or D'}
    if ok.sum()<8:return result
    x,y,v=q[ok],y[ok],v[ok];v=np.maximum(v,1e-12)
    if y.sum()<=0:result['status']='no_positive_lowq_excess';return result
    grid=np.geomspace(1.,100.,241);fits=[_fit_linear(1/(1+(x*xi)**2)**2,y,v) for xi in grid];loss=np.array([f[1] for f in fits]);j=int(np.argmin(loss));xi=float(grid[j]);b,chi,cov=fits[j];w=1/v;null=float(np.sum(w*(y-np.sum(w*y)/w.sum())**2));inflate=max(1.,chi/max(len(y)-3,1));se=float(np.sqrt(max(cov[0,0],0)*inflate));delta=(null-chi)/inflate-4;profile=grid[loss<=chi+3.84*inflate]
    result.update(delta_AIC=float(delta),amplitude=float(b[0]),amplitude_fit_se=se,profile_diagnostic_nm=[float(profile[0]),float(profile[-1])],q_range_nm_inv=[float(x.min()),float(x.max())])
    knee=x.min()*xi<1<x.max()*xi;interior=j not in (0,len(grid)-1) and profile[0]>grid[0] and profile[-1]<grid[-1]
    if delta<=6 or b[0]<=2*se:result['status']='excess_turnover_not_supported'
    elif not knee:result['status']='turnover_outside_measured_lowq'
    elif not interior:result['status']='length_profile_unbounded'
    else:result.update(status='conditional_length_resolved',xi_nm=xi,xi_lo95_nm=float(profile[0]),xi_hi95_nm=float(profile[-1]))
    return result

def reference_changes(q,Y,N,Yref,Nref):
    q=np.asarray(q,float);Y,N,Yref,Nref=map(lambda x:np.asarray(x,float),(Y,N,Yref,Nref));valid=(N>0)&(Nref>0);a=np.divide(Y,N,out=np.full_like(Y,np.nan),where=valid);b=np.divide(Yref,Nref,out=np.full_like(Y,np.nan),where=valid);va=np.divide(Y,N*N,out=np.full_like(Y,np.nan),where=valid);vb=np.divide(Yref,Nref*Nref,out=np.full_like(Y,np.nan),where=valid);difference=a-b;ratio=np.divide(a,b,out=np.full_like(Y,np.nan),where=valid&(b>0));ratio_var=np.divide(va,b*b,out=np.full_like(Y,np.nan),where=valid&(b>0))+np.divide(a*a*vb,b**4,out=np.full_like(Y,np.nan),where=valid&(b>0));summaries=[];dq=np.diff(Q_EDGES)
    for view in range(len(Y)):
        hi=(q>=1.3)&(q<1.7)&valid[view];low=(q<.3)&valid[view];A=float(np.sum(Y[view,hi])/np.sum(N[view,hi])) if hi.any() else None;B=float(np.sum(Yref[view,hi])/np.sum(Nref[view,hi])) if hi.any() else None;R=A/B if A is not None and B is not None and B>0 else None
        SE=None
        if R is not None:SE=float(np.sqrt(np.sum(Y[view,hi])/np.sum(N[view,hi])**2/B**2+A*A*np.sum(Yref[view,hi])/np.sum(Nref[view,hi])**2/B**4))
        flat=None;rr=ratio[view,hi];xx=q[hi];good=np.isfinite(rr)
        if good.sum()>=6 and R and R>0:flat=float(abs(np.polyfit(xx[good],rr[good],1)[0])*.4/R)
        integral=float(np.sum(difference[view,low]*dq[low])) if low.any() else None;error=float(np.sqrt(np.sum((va[view,low]+vb[view,low])*dq[low]**2))) if low.any() else None
        summaries.append({'view':view,'AF_band':view//2+1,'side':'-' if view%2==0 else '+','highq_band_ratio':R,'highq_ratio_conditional_se':SE,'ratio_platform_relative_slope':flat,'highq_platform_ratio':R if flat is not None and flat<=.30 else None,'highq_platform_status':'ratio_platform' if flat is not None and flat<=.30 else 'nonflat_or_insufficient_highq_ratio','lowq_signed_excess_integral':integral,'lowq_integral_conditional_se':error,'length':fit_excess_length(q,difference[view],va[view]+vb[view])})
    sides=[]
    for band in range(4):
        left,right=summaries[2*band:2*band+2];r1,r2=left['highq_band_ratio'],right['highq_band_ratio'];e1,e2=left['highq_ratio_conditional_se'],right['highq_ratio_conditional_se'];sides.append({'AF_band':band+1,'ratio_side_relative_difference':abs(r1-r2)/max(abs((r1+r2)/2),1e-12) if r1 is not None and r2 is not None else None,'ratio_side_conditional_z':abs(r1-r2)/max(np.hypot(e1,e2),1e-12) if e1 is not None and e2 is not None else None,'lowq_excess_left':left['lowq_signed_excess_integral'],'lowq_excess_right':right['lowq_signed_excess_integral']})
    ratios=[s['highq_band_ratio'] for s in summaries if s['highq_band_ratio'] is not None];lengths=[s['length']['xi_nm'] for s in summaries if s['length']['xi_nm'] is not None]
    return {'q':q,'difference':difference,'difference_se':np.sqrt(va+vb),'ratio':ratio,'ratio_se':np.sqrt(ratio_var),'current_rate':a,'reference_rate':b,'valid':valid,'summaries':summaries,'side_consistency':sides,'highq_ratio_view_range':[min(ratios),max(ratios)] if ratios else None,'length_view_range_nm':[min(lengths),max(lengths)] if lengths else None,'interpretation':'Observed exposure-normalized change; unknown I0/attenuation/motion phase; not isolated metal scattering','conditional_error':'Independent Poisson counts; excludes flux/reference/systematic variation','views_are_not_independent_replicates':True}

def no_peak_reason(fits):
    if all(f.get('status')=='insufficient_support_or_counts' for f in fits):return 'insufficient_counts_or_q_support'
    if any(f.get('gap_near_peak') for f in fits):return 'candidate_feature_intersects_missing_q'
    if any(f.get('delta_AIC',-np.inf)>10 and f.get('edge_or_width_bound') for f in fits):return 'candidate_feature_not_bracketed_or_width_boundary'
    return 'no_supported_internal_peak_against_monotonic_background'
