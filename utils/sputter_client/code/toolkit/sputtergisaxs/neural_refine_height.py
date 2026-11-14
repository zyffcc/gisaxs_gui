'Independent surface height for NR cylinder / flat-base half ellipsoid.'
from __future__ import annotations
from functools import lru_cache
import numpy as np
from .linecut_v3 import Model,size_rule
from .polymer_linecut_v3 import depth_rule
from .formfactor import jinc


class HeightModel(Model):
    def __init__(self,cut,height_nm,shape='cylinder',approximation='DA',critical_angle_deg=.16,*,nsize=224,nz=64):
        if not np.isfinite(height_nm) or height_nm<=0:raise ValueError('Positive finite physical height required')
        super().__init__(cut,shape,approximation,critical_angle_deg,nsize=nsize,nz=nz,
            cylinder_tangent=False,hemisphere_tangent=False,side_response='independent')
        self._height_nm=float(height_nm)

    @property
    def height_nm(self):return self._height_nm

    @lru_cache(maxsize=4)
    def _moments(self,R,sigma_log):


        size_order=int(np.ceil(self.nsize*max(1.,self.height_nm/R))) if self.shape!='cylinder' else self.nsize
        z,w=size_rule(size_order);radii=R*np.exp(sigma_log*z-.5*sigma_log**2)
        P=np.zeros_like(self.qpar);M=np.zeros_like(self.qpar,dtype=complex)
        if self.shape=='cylinder':
            h=self.height_nm
            phase=np.exp(.5j*self.kz*h)*np.sinc(self.kz*h/(2*np.pi))
            vertical=np.sum(self.coeff*phase,axis=0)
            for radius,weight in zip(radii,w):
                F=(radius/R)**2*jinc(radius*self.qpar)*vertical[:,None,None]
                P+=weight*abs(F)**2;M+=weight*F
        else:
            aspect=self.height_nm/R
            phase_per_radius=float(self.qpar.max())+float(abs(self.kz).max())*aspect
            for radius,weight in zip(radii,w):
                order=max(self.nz,int(np.ceil(.75*phase_per_radius*radius))+12)
                order=16*int(np.ceil(order/16));t,v=depth_rule(order)
                rr=radius*np.sqrt(np.maximum(1-t*t,0));area=v*radius*np.pi*rr**2/R**3
                zp=aspect*radius*t
                field=np.sum(self.coeff[:,None,:]*np.exp(1j*self.kz[:,None,:]*zp[None,:,None]),axis=0)
                lat=jinc(rr[:,None,None,None]*self.qpar[None,:,:,:])
                F=np.einsum('z,zafq,za->afq',area,lat,field,optimize=True)
                P+=weight*abs(F)**2;M+=weight*F
        return P,M
