"""Empirical intensity features; shared v1 API, not physical morphology.

The peak/background estimator preserves growth_features.py support gates.
Analytic derivatives accelerate the same Poisson-deviance objective.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import xlogy

AF_BANDS=((.08,.30),(.30,.60),(.60,1.00),(1.00,1.50))
Q_EDGES=np.arange(.05,1.80001,.025)
V_EDGES=np.arange(.02,.80001,.01)

def rate_model(q,p,background='power',peak=False):
    b0,b1,slope=p[:3]
    base=b0+b1*((q/.5)**(-slope) if background=='power' else np.exp(-slope*(q-.5)))
    if peak:
        amp,centre,sigma=p[3:];base+=amp*np.exp(-.5*((q-centre)/sigma)**2)
    return base

def dev_residual(y,mu):
    mu=np.maximum(mu,1e-12);d=2*(xlogy(y,y/mu)-y+mu)
    return np.sign(mu-y)*np.sqrt(np.maximum(d,0.))

def rate_jac(q,p,background,peak):
    log=np.log(q/.5) if background=='power' else q-.5
    b=np.exp(-p[2]*log);columns=[np.ones_like(q),b,-p[1]*b*log]
    if peak:
        g=np.exp(-.5*((q-p[4])/p[5])**2)
        columns.extend([g,p[3]*g*(q-p[4])/p[5]**2,p[3]*g*(q-p[4])**2/p[5]**3])
    return np.array(columns).T

def fit_peak(q,counts,den,lo=.15,hi=1.30,background='power'):
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
    excess=rates-rate_model(x,null.x,background)*scale;peaks=find_peaks(excess)[0];peaks=peaks[(x[peaks]>lo+.06)&(x[peaks]<hi-.06)]
    centre=float(x[peaks[np.argmax(excess[peaks])]]) if len(peaks) else float(np.clip(x[np.argmax(excess)],lo+.07,hi-.07))
    fits=[solve([*null.x,max(float(excess.max())/scale,.05),start,width],True) for start in (centre,float(np.clip(.57,lo+.07,hi-.07))) for width in (.07,.16)]
    f=min(fits,key=lambda f:float(f.fun@f.fun));p=f.x;dev=float(f.fun@f.fun);dev0=float(null.fun@null.fun)
    cov=np.linalg.pinv(f.jac.T@f.jac,rcond=1e-11)*max(1.,dev/max(len(x)-6,1));se=np.sqrt(np.maximum(np.diag(cov),0.))
    q0=float(p[4]);sig=float(p[5]);width=float(2*np.sqrt(2*np.log(2))*sig);bracket=int(np.searchsorted(x,q0));gaps=np.diff(x[(x>q0-sig)&(x<q0+sig)])
    gap=bool(bracket==0 or bracket==len(x) or x[bracket]-x[bracket-1]>.051 or (len(gaps)>0 and gaps.max()>.051))
    edge=bool(q0-width/2<=lo or q0+width/2>=hi or sig<=.0205 or sig>=.59)
    aic=dev0-dev-6;resolved=bool(f.success and aic>10 and p[3]>2*se[3] and not edge and not gap)
    return dict(status='resolved_empirical_feature' if resolved else 'unresolved_or_boundary_feature',q_star=q0,q_star_fit_se=float(se[4]),fwhm=width,fwhm_fit_se=float(2*np.sqrt(2*np.log(2))*se[5]),D_corr=float(2*np.pi/q0),D_corr_fit_se=float(2*np.pi*se[4]/q0**2),width_over_qstar=width/q0,peak_amplitude=float(p[3]*scale),peak_amplitude_fit_se=float(se[3]*scale),delta_AIC=float(aic),deviance=dev,null_deviance=dev0,optimizer_success=bool(f.success),edge_or_width_bound=edge,gap_near_peak=gap,parameters=(p*np.array([scale,scale,1,scale,1,1])).tolist(),background=background,window=[lo,hi],statistical_error='Residual-scaled conditional covariance, not complete systematic CI')

def ridge(vy,vn):
    vy,vn=np.asarray(vy,float),np.asarray(vn,float)
    x=(V_EDGES[:-1]+V_EDGES[1:])/2;rate=np.divide(vy,vn,out=np.zeros_like(vy),where=vn>0)
    indices=[i for i in find_peaks(rate)[0] if .08<=x[i]<=.55 and vn[i]>0]
    if not indices:return dict(alpha_f_observed_ridge=None,ridge_status='no_interior_ridge',ridge_critical_angle_or_density=False)
    i=max(indices,key=lambda i:rate[i]);local=(abs(x-x[i])<.07)&(vn>0);base=np.quantile(rate[local],.2);w=np.maximum(rate[local]-base,0.)
    centre=float(x[local]@w/w.sum()) if w.sum()>0 else float(x[i]);bracket=int(np.searchsorted(x,centre))
    gap=bool(bracket==0 or bracket==len(x) or vn[bracket]==0 or vn[max(bracket-1,0)]==0)
    return dict(alpha_f_observed_ridge=centre,alpha_f_native_max=float(x[i]),ridge_resolution_deg=.01,ridge_status='observed_ridge_support_gap' if gap else 'empirical_local_maximum',ridge_critical_angle_or_density=False,native_candidate_peaks=len(indices))
