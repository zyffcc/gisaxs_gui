"""v1 candidate q_y/alpha_f tensor contract, independent of detector pixels.

Rows alpha_f (degrees), columns |q_y| (nm^-1). Valid zeros are preserved;
missing cells have a separate mask. This is a canonical physical projection,
not a resized detector image. Model release still requires C5/C6 acceptance.
"""
import numpy as np
Q_EDGES=np.arange(.05,1.80001,.025)
AF_EDGES=np.linspace(.02,1.5,33)
Q=(Q_EDGES[:-1]+Q_EDGES[1:])/2
AF=(AF_EDGES[:-1]+AF_EDGES[1:])/2
GRID_SHAPE=(len(AF),len(Q))

def angles(energy_ev,alpha_i_deg,qscale=1.,af_shift_deg=0.,*,q_centres=None,af_centres=None):
    qs=Q if q_centres is None else np.asarray(q_centres);afs=AF if af_centres is None else np.asarray(af_centres)
    k=2*np.pi/(1239.841984/energy_ev);ai=np.radians(alpha_i_deg);af,qy=np.meshgrid(np.radians(afs+af_shift_deg),qs*qscale,indexing='ij');phi=np.arcsin(qy/(k*np.cos(af)));qx=k*(np.cos(af)*np.cos(phi)-np.cos(ai));qz=k*(np.sin(af)+np.sin(ai))
    return {'alpha_f':af,'qy':qy,'qx':qx,'qz':qz,'qpar':np.sqrt(qx*qx+qy*qy),'q_total':np.sqrt(qx*qx+qy*qy+qz*qz)}

def tensor(rate,valid):
    rate=np.asarray(rate,float);valid=np.asarray(valid,bool)
    if rate.shape!=GRID_SHAPE or valid.shape!=rate.shape:raise ValueError('Expected canonical q-space grid '+str(GRID_SHAPE))
    scale=max(float(np.quantile(rate[valid],.8)),1e-6) if valid.any() else 1.
    return np.stack([np.log1p(np.maximum(rate,0)/scale)*valid,valid.astype(float)]).astype(np.float32),float(scale)

def project(qy,af_deg,image,exposure_s=1.,mask=None):
    qy,af,image=map(np.asarray,(qy,af_deg,image));valid=np.isfinite(image)&(image>=0)
    if mask is not None:valid&=np.asarray(mask,bool)
    na,nq=GRID_SHAPE;qi=np.searchsorted(Q_EDGES,abs(qy),side='right')-1;ai=np.searchsorted(AF_EDGES,af,side='right')-1;valid&=(qi>=0)&(qi<nq)&(ai>=0)&(ai<na);labels=ai[valid]*nq+qi[valid];Y=np.bincount(labels,weights=image[valid],minlength=na*nq).reshape(GRID_SHAPE);N=np.bincount(labels,minlength=na*nq).reshape(GRID_SHAPE)*exposure_s
    rate=np.divide(Y,N,out=np.zeros_like(Y),where=N>0);t,s=tensor(rate,N>0)
    return {'tensor':t,'Y':Y,'N':N,'scale':s}
