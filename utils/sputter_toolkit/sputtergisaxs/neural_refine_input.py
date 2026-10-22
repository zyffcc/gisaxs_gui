"""v3-NR inputs from one native photon histogram; frozen v2 stays unchanged.

The four existing AF bands are exact unions of native cells. The vertical
stripe is selected from an observed peak, never a hidden D label. NN inputs
may overlap; ``refinement_partition`` supplies a disjoint photon likelihood.
"""
from __future__ import annotations
import numpy as np
from .linecut_v3 import NATIVE_AF, NATIVE_EDGES, direct_features
from .features import AF_BANDS
from .tool_qinput import Q, Q_EDGES

SCHEMA='v3_neural_refine_input_1'
FEATURE_NAMES=('q_star_nm_inv','FWHM_nm_inv','Yoneda_deg','highq_log_slope')


def validate_native(Y,E,*,integer=True):
    y,e=np.asarray(Y,float),np.asarray(E,float)
    shape=(2,len(NATIVE_AF),len(Q))
    if y.shape!=shape or e.shape!=shape:
        raise ValueError('Expected original side/AF/q native histogram')
    if not np.isfinite(y).all() or not np.isfinite(e).all() or np.any(y<0) or np.any(e<0):
        raise ValueError('Finite nonnegative counts and exposure required')
    if integer and np.any(y!=np.rint(y)):
        raise ValueError('Native raw-count input must contain integer photons')
    if np.any((e==0)&(y>0)):
        raise ValueError('Positive counts outside supported exposure')
    return y,e


def stripe_indices(qstar):
    if qstar is None or not np.isfinite(qstar) or not Q_EDGES[0]<=qstar<=Q_EDGES[-1]:
        return np.flatnonzero((Q>=.15)&(Q<1.2)),False
    i=int(np.argmin(abs(Q-qstar)))
    return np.arange(max(0,i-1),min(len(Q),i+2)),True


def extract(Y,E,energy_ev,alpha_i_deg,*,metal=None,observations=None,integer=True):
    y,e=validate_native(Y,E,integer=integer)
    if not np.isfinite([energy_ev,alpha_i_deg]).all() or energy_ev<=0 or alpha_i_deg<=0:
        raise ValueError('Positive finite photon energy and incidence required')
    if metal not in (None,'Ag','Au'):raise ValueError('Declared Au/Ag required; Cu is not mapped')
    obs=direct_features(y,e) if observations is None else dict(observations)
    hy=[];he=[]
    for lo,hi in AF_BANDS:
        m=(NATIVE_AF>=lo)&(NATIVE_AF<hi)
        hy.append(y[:,m].sum(axis=1));he.append(e[:,m].sum(axis=1))
    ix,known=stripe_indices(obs.get('q_star_nm_inv'))
    vy=y[:,:,ix].sum(axis=2);ve=e[:,:,ix].sum(axis=2)
    qz=2*np.pi/(1239.841984/energy_ev)*(np.sin(np.radians(alpha_i_deg))+np.sin(np.radians(NATIVE_AF)))
    vals=np.zeros(len(FEATURE_NAMES));present=np.zeros(len(vals),bool)
    for j,name in enumerate(FEATURE_NAMES):
        value=obs.get(name)
        if value is not None and np.isfinite(value):vals[j]=value;present[j]=True
    return dict(schema=SCHEMA,H_Y=np.asarray(hy),H_E=np.asarray(he),V_Y=vy,V_E=ve,
        q_nm_inv=Q.copy(),qz_nm_inv=qz,vertical_q_indices=ix,vertical_peak_known=bool(known),
        vertical_source='observed_peak_three_native_columns' if known else 'no_peak_integrated_0.15_1.2_fallback',
        energy_ev=float(energy_ev),alpha_i_deg=float(alpha_i_deg),metal=metal,feature_values=vals,
        feature_present=present,observations=obs)


def rates_and_support(item):
    y=np.r_[item['H_Y'].reshape(-1),item['V_Y'].reshape(-1)]
    e=np.r_[item['H_E'].reshape(-1),item['V_E'].reshape(-1)]
    rate=np.divide(y,e,out=np.zeros_like(y),where=e>0)
    return rate,e>0


def self_normalize_cuts(rate,support):
    """Profile out one positive scale per side/band or vertical side.

    This deliberately removes between-cut optical amplitude information;
    shape/spacing may remain informative, absolute loading cannot follow.
    """
    rate=np.asarray(rate,float);support=np.asarray(support,bool)
    expected=4*2*len(Q)+2*len(NATIVE_AF)
    if rate.shape!=(expected,) or support.shape!=rate.shape:raise ValueError('NR projected cut vector required')
    result=np.zeros_like(rate)
    regions=[slice(j*len(Q),(j+1)*len(Q)) for j in range(8)]
    start=8*len(Q);regions += [slice(start+j*len(NATIVE_AF),start+(j+1)*len(NATIVE_AF)) for j in range(2)]
    for ix in regions:
        value=rate[ix];ok=support[ix]&(value>0)
        scale=max(float(np.median(value[ok])) if ok.any() else 1.,1e-12)
        result[ix]=np.arcsinh(value/scale)
    return result


def encode(item,*,cut_weights=None,normalization='global'):
    """Global scale preserves AF response; support and unknowns are explicit.

    No optional substrate/QCM truth is inferred here. Weights are frozen from
    TRAIN domain checks, not selected on a frame's fit or on TEST.
    """
    rate,support=rates_and_support(item)
    positive=rate[support&(rate>0)]
    scale=max(float(np.median(positive)) if len(positive) else 1.,1e-12)
    if normalization=='global':signal=np.arcsinh(rate/scale)
    elif normalization=='per_cut':signal=self_normalize_cuts(rate,support)
    else:raise ValueError('Declared global or per_cut normalization required')
    if cut_weights is not None:
        w=np.asarray(cut_weights,float)
        if w.shape!=(5,) or np.any(w<0) or not np.isfinite(w).all():raise ValueError('Five finite nonnegative TRAIN-fixed cut weights')
        signal[:4*2*len(Q)]*=np.repeat(w[:4],2*len(Q))
        signal[4*2*len(Q):]*=w[4]
        support[:4*2*len(Q)]&=np.repeat(w[:4]>0,2*len(Q))
        if w[4]==0:support[4*2*len(Q):]=False
    f=np.array(item['feature_values'],float).copy()
    f/=np.array([1.,1.,1.,5.])
    return np.r_[signal,support.astype(float),f,item['feature_present'].astype(float),
        float(item['vertical_peak_known']),np.log(item['energy_ev']/12000.),item['alpha_i_deg'],
        np.log1p(scale) if normalization=='global' else 0.,float(item.get('metal')=='Au'),float(item.get('metal') is not None)].astype(np.float32)


def refinement_partition(Y,E,qstar=None):
    """Disjoint 4AF horizontal cells plus native-AF vertical stripe.

    Each original photon is counted at most once. With no observed peak the
    vertical fallback is an NN feature only, not a second likelihood term.
    The return counts/support define a partition, not reconstructed pixels.
    """
    y,e=validate_native(Y,E)
    ix,known=stripe_indices(qstar)
    qkeep=np.ones(len(Q),bool)
    if known:qkeep[ix]=False
    ys=[];es=[];kinds=[]
    for b,(lo,hi) in enumerate(AF_BANDS):
        af=(NATIVE_AF>=lo)&(NATIVE_AF<hi)
        ys.append(y[:,af].sum(axis=1)[:,qkeep].ravel())
        es.append(e[:,af].sum(axis=1)[:,qkeep].ravel())
        kinds.extend([f'H{b}']*(2*int(qkeep.sum())))
    if known:
        ys.append(y[:,:,ix].sum(axis=2).ravel());es.append(e[:,:,ix].sum(axis=2).ravel())
        kinds.extend(['V']*(2*len(NATIVE_AF)))
    return np.concatenate(ys),np.concatenate(es),np.asarray(kinds)


def project_partition_rates(rate,E,qstar=None):
    """Expected photons in exactly the same disjoint partition as observed."""
    r,e=np.asarray(rate,float),np.asarray(E,float)
    if r.shape!=e.shape or np.any(r<0) or not np.isfinite(r).all():raise ValueError('Positive finite native rate required')
    # No integer validation for deterministic expectations.
    ix,known=stripe_indices(qstar);qkeep=np.ones(len(Q),bool)
    if known:qkeep[ix]=False
    values=[]
    for lo,hi in AF_BANDS:
        af=(NATIVE_AF>=lo)&(NATIVE_AF<hi)
        values.append((r[:,af]*e[:,af]).sum(axis=1)[:,qkeep].ravel())
    if known:values.append((r*e)[:,:,ix].sum(axis=2).ravel())
    return np.concatenate(values)
