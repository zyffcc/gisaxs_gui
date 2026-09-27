"""Direct independent float64 forward from candidate fields, without encode clipping."""
import numpy as np
from numpy_forward64 import component
from flow_codec import COMBOS
def direct_forward(cand,i,j,q):
    c=int(cand['combos'][i,j]);typ=COMBOS[c];p=cand['params'][i,j].astype('float64');g=cand['globals'][i,j].astype('float64')
    active=typ>0;wl=cand['weights'][i,j,active].astype('float64');w=np.exp(wl-wl.max());w/=w.sum()
    r=np.exp(p[:,0]*np.log(100.));sr=.02+.88*p[:,1];h=np.exp(np.log(2.)+p[:,2]*np.log(250.));sh=.02+.88*p[:,3]
    dist=np.exp(np.log(3.)+p[:,4]*np.log(500./3.));sd=.05+.85*p[:,5]
    particle=np.zeros_like(q,dtype='float64')
    for slot,weight in enumerate(w):particle+=weight*component(q,int(typ[slot]),r[slot],sr[slot],h[slot],sh[slot],dist[slot],sd[slot],cand['d'][i,j,slot]>0)
    result=particle+np.exp(np.log(1e-6)+g[0]*np.log(1e4))*np.median(particle)
    if cand['res'][i,j]>0:
        rs=np.exp(np.log(.007)+g[1]*np.log(.013/.007));rn=5+5*g[2];ra=np.exp(np.log(10)+g[3]*np.log(100))
        shape=1/(1+(q/rs)**rn);result+=ra*particle[:5].max()/max(shape[:5].max(),1e-30)*shape
    return np.maximum(result,1e-30)
