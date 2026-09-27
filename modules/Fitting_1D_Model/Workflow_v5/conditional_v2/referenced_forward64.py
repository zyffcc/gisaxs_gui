"""Evaluate a V5 candidate on new q while keeping its observation-grid normalizers.

For visualization/diagnosis only: preserve the median-background and first-five
resolution coefficients from reference_q. No fitting or model-weight changes.
"""
import numpy as np
from numpy_forward64 import component
from flow_codec import COMBOS

def terms(c,i,h,q):
    q=np.asarray(q,'float64');p=c['params'][i,h].astype('float64');g=c['globals'][i,h].astype('float64');types=COMBOS[int(c['combos'][i,h])];active=types>0
    wl=c['weights'][i,h,active].astype('float64');w=np.exp(wl-wl.max());w/=w.sum()
    r=np.exp(p[:,0]*np.log(100));sr=.02+.88*p[:,1];height=2*np.exp(p[:,2]*np.log(250));sh=.02+.88*p[:,3];distance=3*np.exp(p[:,4]*np.log(500/3));sd=.05+.85*p[:,5]
    particle=np.zeros_like(q)
    for slot,weight in enumerate(w):particle+=weight*component(q,int(types[slot]),r[slot],sr[slot],height[slot],sh[slot],distance[slot],sd[slot],c['d'][i,h,slot]>0)
    bg=np.exp(np.log(1e-6)+g[0]*np.log(1e4));rs=.007*np.exp(g[1]*np.log(.013/.007));nu=5+5*g[2];ra=10*np.exp(g[3]*np.log(100))
    shape=1/(1+(q/rs)**nu)
    return particle,shape,bg,ra

def reference_coefficients(c,i,h,reference_q):
    particle,shape,bg,ra=terms(c,i,h,reference_q)
    return dict(background=float(bg*np.median(particle)),resolution=float(ra*particle[:5].max()/max(shape[:5].max(),1e-30)) if c['res'][i,h]>0 else 0.)

def referenced_forward(c,i,h,q,reference_q):
    particle,shape,_,_=terms(c,i,h,q);co=reference_coefficients(c,i,h,reference_q)
    return np.maximum(particle+co['background']+co['resolution']*shape,1e-30)
