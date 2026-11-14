'New stage-five prior. Does not change old artifact target transforms.'
import numpy as np
from . import tool_prior
from .resolvable import CENTRE_WINDOW

D_RANGE=(float(2*np.pi/CENTRE_WINDOW[1]),float(2*np.pi/CENTRE_WINDOW[0]))
OUT_DOMAIN=(3.,100.)
OUT_FRACTION=.20
DOMAIN={'version':'stage5-A-1','D_search_nm':list(D_RANGE),'D_out_nm':list(OUT_DOMAIN),
        'out_fraction':OUT_FRACTION,'radius_nm':'loguniform .4..min(8,.55D)',
        'aspect':[.1,2.],'sigma_log':[.03,.45],'physics':tool_prior.DOMAIN,
        'regression_subset':'in_range_and_observed_bilateral_peak','stage_output':False}

def draw(seed,D_range=D_RANGE):
    x,m,s,metal,E,ai,rate,rng=tool_prior.draw(seed)
    lo,hi=map(float,D_range)
    if not OUT_DOMAIN[0]<lo<hi<OUT_DOMAIN[1]:raise ValueError('Requested range outside declared simulator domain')
    outside=rng.random()<OUT_FRACTION
    a,b=(OUT_DOMAIN[0],lo) if outside and rng.random()<.5 else (hi,OUT_DOMAIN[1]) if outside else (lo,hi)
    x[0]=np.exp(rng.uniform(np.log(a),np.log(b)))
    x[1]=np.exp(rng.uniform(np.log(.4),np.log(min(8,.55*x[0]))))
    return x,m,s,metal,E,ai,rate,rng

def target(x):
    'Unbounded transforms; out-of-range truth is never clipped.'
    x=np.asarray(x,float)
    if not np.isfinite(x[...,:4]).all() or np.any(x[...,:3]<=0) or np.any((x[...,3]<=.03)|(x[...,3]>=.45)):
        raise ValueError('Invalid structural primitive')
    s=(x[...,3]-.03)/.42
    return np.stack([np.log(x[...,0]/10),np.log(x[...,1]/2),np.log(x[...,2]),np.log(s/(1-s))],axis=-1)

def decode(z):
    from scipy.special import expit
    z=np.asarray(z,float)
    return np.stack([10*np.exp(z[...,0]),2*np.exp(z[...,1]),np.exp(z[...,2]),.03+.42*expit(z[...,3])],axis=-1)

def quantities(z,shape):
    x=decode(z);a=x[...,1];aspect=x[...,2];s=np.asarray(shape)
    diameter=np.where((s==0)&(aspect>1),a*(1+aspect**2)/aspect,2*a)
    return np.stack([x[...,0],diameter,a*aspect,x[...,3]],axis=-1)

def out_class(D,D_range=D_RANGE):
    D=np.asarray(D,float);lo,hi=D_range
    return np.where(D<lo,1,np.where(D>hi,2,0)).astype(np.int64)
