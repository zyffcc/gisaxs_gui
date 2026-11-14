'Count likelihood with declared model discrepancy; not pure counting GOF.'
from __future__ import annotations
import numpy as np
from scipy.special import xlogy,gammaln
from .features import dev_residual

FLOORS=(.05,.03,.10,0.)


def validate(y,mu,eps):
    y,mu=np.broadcast_arrays(np.asarray(y,float),np.asarray(mu,float))
    if not np.isfinite(y).all() or not np.isfinite(mu).all() or np.any(y<0) or np.any(mu<0) or not np.isfinite(eps) or eps<0:
        raise ValueError('Nonnegative finite counts, means, discrepancy required')
    return y,np.maximum(mu,1e-300)


def nll(y,mu,eps=.05):
    y,mu=validate(y,mu,eps)
    if eps==0:return mu-xlogy(y,mu)+gammaln(y+1)
    r=1./eps**2
    return gammaln(r)+gammaln(y+1)-gammaln(y+r)+r*np.log1p(mu/r)-xlogy(y,mu/(mu+r))


def residual(y,mu,eps=.05):
    y,mu=validate(y,mu,eps)
    if eps==0:return dev_residual(y,mu)
    r=1./eps**2
    d=2*(xlogy(y,y/mu)-(y+r)*np.log1p((y-mu)/(mu+r)))
    return np.sign(y-mu)*np.sqrt(np.maximum(d,0))


def compare_floors(y,mu,n_parameters):
    y,mu=validate(y,mu,.05);dof=max(1,y.size-int(n_parameters))
    return [dict(relative_model_error=eps,deviance=float(np.sum(residual(y,mu,eps)**2)),
        deviance_per_dof=float(np.sum(residual(y,mu,eps)**2)/dof),dof=dof,
        scope='original_Poisson_reference' if eps==0 else 'model_level_negative_binomial_reference_not_original_statistical_GOF') for eps in FLOORS]


def sample(mu,rng,eps=.05):
    _,mu=validate(np.zeros_like(mu),mu,eps)
    latent=mu if eps==0 else rng.gamma(1./eps**2,mu*eps**2)
    return rng.poisson(latent)


def native_discrepancy_factors(shape,rng,qstar,eps=.05):
    'One Gamma multiplier per disjoint observed-cut group.'
    from .neural_refine_input import stripe_indices,NATIVE_AF,Q
    from .features import AF_BANDS
    if tuple(shape)!=(2,len(NATIVE_AF),len(Q)) or eps<0:raise ValueError('Native shape and nonnegative discrepancy required')
    if eps==0:return np.ones(shape)
    r=1./eps**2
    out=rng.gamma(r,1./r,size=shape)
    ix,known=stripe_indices(qstar);keep=np.ones(len(Q),bool)
    if known:keep[ix]=False
    qs=np.flatnonzero(keep)
    for lo,hi in AF_BANDS:
        af=np.flatnonzero((NATIVE_AF>=lo)&(NATIVE_AF<hi))
        gain=rng.gamma(r,1./r,size=(2,len(qs)))
        for side in range(2):out[side][np.ix_(af,qs)]=gain[side][None,:]
    if known:
        gain=rng.gamma(r,1./r,size=(2,len(NATIVE_AF)))
        for side in range(2):out[side][:,ix]=gain[side][:,None]
    return out


def sample_native_partition(mu,rng,qstar,eps=.05):
    'qstar belongs to the noise-free observable mean, never hidden D.'
    _,mu=validate(np.zeros_like(mu),mu,eps)
    return rng.poisson(mu*native_discrepancy_factors(mu.shape,rng,qstar,eps))
