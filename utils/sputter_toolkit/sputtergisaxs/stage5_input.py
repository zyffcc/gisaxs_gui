"""Real detector photons -> the stage5 atomic/feature/image input contract.

Does not infer noise metadata from a normalised tensor or resize old cuts.
Y is raw counts; N is sum(valid pixels * that frame's exposure), exactly once.
Unknown optional inputs have explicit false flags. No raw image is retained.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .prediction import FeatureMap
from .tool_qinput import AF_EDGES,Q_EDGES,Q
from .features import AF_BANDS,V_EDGES
# Identical native bins; CPU geometry must not import optional CUDA kernels.
NATIVE_EDGES=np.unique(np.round(np.r_[np.arange(.02,1.50001,.01),AF_EDGES,V_EDGES,np.array(AF_BANDS).ravel()],10))
NATIVE_AF=(NATIVE_EDGES[:-1]+NATIVE_EDGES[1:])/2

class AtomicMap:
    def __init__(self,geometry,shape,alpha_i_deg,*,sample_roll_deg=0.,crop=None,exclude_rings=()):
        # Shared validated geometry convention; no change to producer-pinned
        # FeatureMap. Rays below follow that same implementation explicitly.
        self.feature_map=FeatureMap(geometry,shape,alpha_i_deg,sample_roll_deg,crop,exclude_rings)
        self.shape=tuple(shape);self.meta=self.feature_map.geometry_meta
        r0,r1,c0,c1=self.feature_map.crop;rr,cc=np.mgrid[r0:r1,c0:c1]
        if isinstance(geometry,(str,Path)):
            import pyFAI
            g=pyFAI.load(str(geometry));p3,p1,p2=g.calc_pos_zyx(d1=rr.astype(float),d2=cc.astype(float));u=np.stack([p3,p2,-p1])
            wavelength_nm=float(g.wavelength)*1e9
        else:
            g=dict(geometry);L=float(g['sdd_mm']);pixel=g['pixel_mm'];pr,pc=(float(pixel),float(pixel)) if np.isscalar(pixel) else map(float,pixel)
            u=np.stack([np.full(rr.shape,L),(cc-float(g['db_col']))*pc,(float(g['db_row'])-rr)*pr]);wavelength_nm=1239.841984/float(g['energy_ev'])
        u/=np.linalg.norm(u,axis=0);ai=np.radians(alpha_i_deg);roll=np.radians(self.meta['sample_roll_deg'])
        ns=np.array([-np.sin(ai),np.cos(ai)*np.sin(roll),np.cos(ai)*np.cos(roll)]);xs=np.array([np.cos(ai),np.sin(ai)*np.sin(roll),np.sin(ai)*np.cos(roll)]);ys=np.cross(ns,xs)
        af=np.degrees(np.arcsin(np.clip(np.einsum('i,ijk->jk',ns,u),-1,1)));qvec=2*np.pi/wavelength_nm*(u-np.array([1.,0.,0.])[:,None,None]);qy=np.einsum('i,ijk->jk',ys,qvec)
        qa=np.searchsorted(Q_EDGES,abs(qy),side='right')-1;fa=np.searchsorted(NATIVE_EDGES,af,side='right')-1
        good=(qa>=0)&(qa<len(Q))&(fa>=0)&(fa<len(NATIVE_AF))
        for centre,half in exclude_rings:good&=abs(np.linalg.norm(qvec,axis=0)-centre)>half
        self.indices=np.ravel_multi_index((rr[good],cc[good]),shape)
        self.bins=(((qy[good]>0).astype(int)*len(NATIVE_AF)+fa[good])*len(Q)+qa[good]).astype(np.int32)
        self.atomic_shape=(2,len(NATIVE_AF),len(Q))

    def project(self,image,exposure_s,mask=None):
        im=np.asarray(image,float)
        if im.shape!=self.shape or not np.isfinite(exposure_s) or exposure_s<=0:raise ValueError('Wrong image geometry or invalid exposure')
        valid=np.isfinite(im)&(im>=0)
        if mask is not None:
            if np.shape(mask)!=self.shape:raise ValueError('Wrong detector mask shape')
            valid&=np.asarray(mask,bool)
        keep=valid.ravel()[self.indices];bins=self.bins[keep];size=int(np.prod(self.atomic_shape))
        Y=np.bincount(bins,weights=im.ravel()[self.indices[keep]],minlength=size).reshape(self.atomic_shape)
        N=np.bincount(bins,minlength=size).reshape(self.atomic_shape)*float(exposure_s)
        return Y,N

def model_input(Y,N,energy_ev,alpha_i_deg,metal,*,preY=None,preN=None,material=None,thickness_nm=None,qcm_rate=None):
    """Build5mandatory/40feature/32x70 image with no optional default truth.

    Current and reference rates use their own exposures/pixel supports. The
    relative-pre descriptor compares integrated RATES over common atom support,
    reproducing the synthetic law where supports happen to be identical.
    Missing reference/zero pre counts withholds that descriptor.
    """
    from .stage5_measurement import counts_profiles,describe
    Y,N=np.asarray(Y,float),np.asarray(N,float)
    if Y.shape!=(2,len(NATIVE_AF),len(Q)) or N.shape!=Y.shape or not np.isfinite(Y).all() or not np.isfinite(N).all() or np.any(Y<0) or np.any(N<0):
        raise ValueError('Invalid original photon/count support')
    if str(metal).lower() not in ('au','ag'):raise ValueError('Metal must be Au or Ag')
    hy,hn,vy,vn,image,scale=counts_profiles(Y,N);desc=describe(hy,hn,vy,vn)
    mandatory=np.array([(float(energy_ev)-14000)/6000,(float(alpha_i_deg)-.425)/.275,float(str(metal).lower()=='au'),np.log(max(scale,1e-6)),np.log(max(float(np.median(N[N>0])) if np.any(N>0) else 0.,1e-6))],np.float32)
    pre=np.zeros_like(image);known=np.zeros(4,bool);optional=np.zeros(3,np.float32)
    if (preY is None)!=(preN is None):raise ValueError('Pre counts and support must be supplied together')
    if preY is not None:
        pY,pN=np.asarray(preY,float),np.asarray(preN,float)
        if pY.shape!=Y.shape or pN.shape!=N.shape or not np.isfinite(pY).all() or not np.isfinite(pN).all() or np.any(pY<0) or np.any(pN<0):raise ValueError('Invalid provided reference')
        py,pn,pvy,pvn,pim,pscale=counts_profiles(pY,pN);pre[0]=np.log1p(np.expm1(pim[0])*pscale/scale)*pim[1];pre[1]=pim[1];known[2]=True
        # Synthetic post/pre shared N. For unequal real exposures use rates
        # on identical atom support, weighted by current effective exposure.
        common=(N>0)&(pN>0)
        current_rate=np.divide(Y,N,out=np.zeros_like(Y),where=N>0);reference_rate=np.divide(pY,pN,out=np.zeros_like(pY),where=pN>0)
        from .features import AF_BANDS
        for b,(lo,hi) in enumerate(AF_BANDS):
            good=common&((NATIVE_AF>=lo)&(NATIVE_AF<hi))[None,:,None]
            a=float((current_rate*N)[good].sum());z=float((reference_rate*N)[good].sum())
            if a>0 and z>0:desc['features'][13+b]=np.log(a/z);desc['features'][33+b]=1
    if material is not None:
        from .tool_prior import MATERIALS
        if material not in MATERIALS:raise ValueError('Unknown supplied layer class')
        known[0]=True;optional[0]=MATERIALS.index(material)
    for k,value,index in [(1,thickness_nm,1),(3,qcm_rate,2)]:
        if value is not None:
            if not np.isfinite(value) or value<=0:raise ValueError('Provided optional scale must be positive')
            known[k]=True;optional[index]=value
    return {'image':image,'pre':pre,'features':desc['features'],'mandatory':mandatory,'optional':optional,'known':known,
            'Y':hy,'N':hn,'VY':vy,'VN':vn,'scale':scale,'descriptor':desc,
            'atomic_exposure_sum':float(N.sum()),'mean_rate_not_second_normalized':True}
