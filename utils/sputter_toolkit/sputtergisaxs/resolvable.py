"""Observable q coverage and predeclared stage-five evaluation strata.

Coverage of 2*pi/q is a search domain, not proof of structural or NN
identifiability. Masks and interrupted intervals are retained without filling.
"""
import numpy as np
from .features import Q_EDGES

PEAK_WINDOW=(.05,1.75)
CENTRE_WINDOW=(.15,1.75)

def _intervals(keep):
    ix=np.flatnonzero(keep)
    if not len(ix):return []
    groups=np.split(ix,np.flatnonzero(np.diff(ix)>1)+1)
    return [[max(float(Q_EDGES[g[0]]),CENTRE_WINDOW[0]),
             min(float(Q_EDGES[g[-1]+1]),CENTRE_WINDOW[1])] for g in groups]

def coverage(N):
    """Actual exposure-weighted pixel support, eight AF/side views."""
    N=np.asarray(N,float)
    if N.shape!=(8,len(Q_EDGES)-1):raise ValueError('Expected eight side profiles')
    q=(Q_EDGES[:-1]+Q_EDGES[1:])/2
    valid=np.isfinite(N)&(N>0)&(q[None,:]>=CENTRE_WINDOW[0])&(q[None,:]<CENTRE_WINDOW[1])
    bands=[]
    for b in range(4):
        joint=valid[2*b]&valid[2*b+1];intervals=_intervals(joint)
        bands.append({'AF_band':b,'q_intervals_nm_inv':intervals,
                      'D_corr_intervals_nm':[[float(2*np.pi/z),float(2*np.pi/a)] for a,z in intervals],
                      'bilateral_bins':int(joint.sum()),
                      'one_side_bins':int((valid[2*b]^valid[2*b+1]).sum()),
                      'q_support_indices':np.flatnonzero(joint).tolist()})
    union=np.any(valid[::2]&valid[1::2],axis=0)
    intervals=_intervals(union)
    ds=[[float(2*np.pi/z),float(2*np.pi/a)] for a,z in intervals]
    return {'definition':'Observable D_corr search coverage =2pi/|qy|; structural D needs independent recovery and PPC',
            'peak_fit_window_nm_inv':list(PEAK_WINDOW),'peak_centre_window_nm_inv':list(CENTRE_WINDOW),
            'q_intervals_nm_inv':intervals,'D_corr_intervals_nm':ds,
            'D_search_min_nm':min((d[0] for d in ds),default=None),
            'D_search_max_nm':max((d[1] for d in ds),default=None),
            'bands':bands,'holes_interpolated':False}

def in_range(D,intervals):
    values=np.asarray(D,float);result=np.zeros(values.shape,bool)
    for lo,hi in intervals:result|=(values>=lo)&(values<=hi)&np.isfinite(values)
    return result

def visible_for_regression(selected,cov):
    """Observed feature label, frozen before any posterior/OOD prediction."""
    if not selected or selected.get('support')!='bilateral':return False
    q=selected.get('q_star');b=selected.get('AF_band')
    if q is None or b not in range(4) or not CENTRE_WINDOW[0]<=q<CENTRE_WINDOW[1]:return False
    return bool(in_range(2*np.pi/q,cov['bands'][b]['D_corr_intervals_nm']))

def strata(truth_D,visible_peak,intervals):
    inside=in_range(truth_D,intervals);visible=np.asarray(visible_peak,bool)
    if inside.shape!=visible.shape:raise ValueError('Truth and visible-peak labels differ')
    return {'in_range_visible_peak':inside&visible,'in_range_no_visible_peak':inside&~visible,
            'out_of_range':~inside,'all':np.ones(inside.shape,bool)}

def regression_metrics(truth,draws,family_id,mask):
    """No success/OOD-dependent filtering. Nonfinite rows count as failures."""
    truth=np.asarray(truth,float);draws=np.asarray(draws,float)
    mask=np.asarray(mask,bool);family_id=np.asarray(family_id)
    if truth.ndim!=1 or draws.ndim!=2 or family_id.shape!=truth.shape or draws.shape[0]!=len(truth) or mask.shape!=(len(truth),):raise ValueError('Metric row identity differs')
    t=truth[mask];v=draws[mask];n=len(t)
    result={'rows':n,'independent_families':int(len(np.unique(family_id[mask]))),
            'invalid_prediction_rows':int((~(np.isfinite(v).all(axis=1)&(v>0).all(axis=1))).sum()),
            'R2_nm':None,'R2_log':None,'coverage68':None,'coverage95':None}
    if not n or result['invalid_prediction_rows'] or not np.isfinite(t).all() or not (t>0).all():return result
    p=np.median(v,axis=1)
    for key,a,b in [('R2_nm',t,p),('R2_log',np.log(t),np.log(p))]:
        denom=float(np.sum((a-a.mean())**2))
        result[key]=float(1-np.sum((a-b)**2)/denom) if denom>0 else None
    for level,limits in [(68,(.16,.84)),(95,(.025,.975))]:
        lo,hi=np.quantile(v,limits,axis=1)
        result['coverage'+str(level)]=float(np.mean((t>=lo)&(t<=hi)))
    return result
