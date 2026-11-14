'Shared positive-count measurement operator for NR production and refine.'
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .neural_refine_input import NATIVE_AF,Q,extract,encode
from .linecut_v3 import NATIVE_EDGES
from .neural_refine_likelihood import sample_native_partition

RECIPE='NR_mean2_PET_host_peak_preserved_bgbridge_psfq_AFblur_tilt_response_groupGamma5'


@lru_cache(maxsize=16)
def native(path,key,exposure_s):
    with np.load(Path(path),allow_pickle=False) as z:
        y=np.array(z[key+'__public_atomic_Y'],dtype=float,copy=True)
        e=np.array(z[key+'__public_atomic_P'],dtype=float,copy=True)*exposure_s
    y.setflags(write=False);e.setflags(write=False)
    return y,e


def empirical_background(y,e,qstar,width):
    v=np.divide(y,e,out=np.zeros_like(y,dtype=float),where=e>0)
    v=gaussian_filter1d(v,2.,axis=2,mode='nearest')
    if qstar is not None and width is not None:
        lo=max(0,min(int(np.searchsorted(Q,qstar-1.5*width)),len(Q)-2))
        hi=max(lo+1,min(int(np.searchsorted(Q,qstar+1.5*width)),len(Q)-1));t=np.linspace(0,1,hi-lo+1)
        v[:,:,lo:hi+1]=v[:,:,lo,None]*(1-t)+v[:,:,hi,None]*t
    for side in range(2):
        for j in range(len(NATIVE_AF)):
            good=e[side,j]>0
            if good.any():v[side,j,~good]=np.interp(Q[~good],Q[good],v[side,j,good])
    return np.maximum(v,0.)


def response(rate,psf_q,af_blur,*,tilt=0.,detector_response=None):
    'Linear positive mean operator; no noise or extra count normalization.'
    if psf_q<0 or af_blur<0:raise ValueError('Nonnegative resolution widths required')
    v=np.asarray(rate,float)
    if v.shape!=(2,len(NATIVE_AF),len(Q)) or not np.isfinite(v).all() or (v<0).any():raise ValueError('Positive native mean rate required')
    if psf_q:v=gaussian_filter1d(v,psf_q/.025,axis=2,mode='nearest')
    if af_blur:
        dist=NATIVE_AF[:,None]-NATIVE_AF[None,:]
        transfer=np.exp(-.5*(dist/af_blur)**2)*np.diff(NATIVE_EDGES)[None,:]
        transfer/=transfer.sum(axis=1,keepdims=True)
        v=np.einsum('ij,sjq->siq',transfer,v,optimize=True)
    v=v*np.exp(tilt*(NATIVE_AF-.4))[None,:,None]
    if detector_response is not None:
        d=np.asarray(detector_response,float)
        if d.shape!=(2,1,len(Q)) or not np.isfinite(d).all() or (d<=0).any():raise ValueError('Positive side/q detector response required')
        v=v*d
    return v


def simulate(d,meta,reference,counts,signal_rate,*,components=True):
    'A single original physical family and noisy acquisition.'
    cy,e=native(str(Path(counts)/meta['array_file']),meta['key'],meta['exposure_per_frame_s'])
    ry,re=native(str(Path(counts)/meta['array_file']),reference['key'],reference['exposure_per_frame_s'])
    rng=np.random.default_rng(d['seed']+100000000);good=e>0
    raw=np.asarray(signal_rate,float);q90=max(float(np.quantile(raw[good],.9)),1e-100)
    crate=np.divide(cy,e,out=np.zeros_like(e),where=good)
    scale=max(float(np.quantile(crate[good],.9)),1e-4)*np.exp(rng.uniform(-.35,.35))
    target=raw*(scale/q90)
    obs=reference['observations'];stationary_host='_pet_' in reference['stem'].lower()
    bg=empirical_background(ry,re,None if stationary_host else obs.get('q_star_nm_inv'),None if stationary_host else obs.get('FWHM_nm_inv'))
    bg90=max(float(np.quantile(bg[good],.9)),1e-20);gain=float(np.exp(rng.uniform(np.log(.02),np.log(1.2))))
    bg=bg*scale/bg90*gain
    psf=float(rng.uniform(.002,.035));af=float(rng.uniform(.002,.025));tilt=float(rng.normal(0,.06))
    detector=np.exp(rng.normal(0,.02,size=(2,1,len(Q))))
    mean=response(target+bg,psf,af,tilt=tilt,detector_response=detector)
    mitem=extract(mean*e,e,d['energy_ev'],d['alpha_i_deg'],metal=d['metal'],integer=False)
    qstar=mitem['observations'].get('q_star_nm_inv')
    y=sample_native_partition(mean*e,rng,qstar,.05)
    item=extract(y,e,d['energy_ev'],d['alpha_i_deg'],metal=d['metal'])
    arrays=dict(encoded=encode(item),H_Y=item['H_Y'],H_E=item['H_E'],V_Y=item['V_Y'],V_E=item['V_E'],
        feature_values=item['feature_values'],feature_present=item['feature_present'],qz_nm_inv=item['qz_nm_inv'])
    receipt=dict(recipe=RECIPE,PSF_q_nm_inv=psf,AF_blur_deg=af,AF_tilt=tilt,background_gain=gain,
        overall_rate_scale=float(scale),native_physics_q90=float(q90),noise_partition_qstar_nm_inv=qstar,
        observations=item['observations'],vertical_source=item['vertical_source'],
        stationary_host_peak_preserved=stationary_host,
        background_reference_key=reference['key'],background_source='earliest observed same TRAIN scan; empirical bridge; not certified zero-metal host')
    if components:
        signal=response(target,psf,af,tilt=tilt,detector_response=detector)
        signal_item=extract(signal*e,e,d['energy_ev'],d['alpha_i_deg'],metal=d['metal'],integer=False)
        bg_item=extract(np.maximum(mean-signal,0.)*e,e,d['energy_ev'],d['alpha_i_deg'],metal=d['metal'],integer=False)
        receipt.update(target_only_features=signal_item['observations'],background_only_features=bg_item['observations'],
            target_photon_fraction=float((signal*e).sum()/max((mean*e).sum(),1e-30)))
    return arrays,receipt
