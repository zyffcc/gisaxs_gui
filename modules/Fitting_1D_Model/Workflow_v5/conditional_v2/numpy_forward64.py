"""Independent float64 V5-qres3 forward on one native valid q array."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/'.scientific_deps'))
import numpy as np
from scipy.special import j1,expit
from flow_codec import COMBOS
from collections import OrderedDict

def nodes(mu,sigma,n,nsig):
    xx=np.linspace(max(mu-nsig*sigma,0.),mu+nsig*sigma,n);w=np.exp(-.5*((xx-mu)/max(sigma,1e-12))**2);return np.maximum(xx,1e-8),w/w.sum()
def sphere(x):
    a=np.abs(x)<.1;safe=np.where(a,1.,x);reg=3*(np.sin(safe)-safe*np.cos(safe))/safe**3;x2=x*x
    return np.where(a,1-x2/10+x2*x2/280-x2**3/15120,reg)
def radial(x):
    a=np.abs(x)<1e-4;safe=np.where(a,1.,x);return np.where(a,1-x*x/8+x**4/192,2*j1(safe)/safe)
def physical(theta,c):
    x=np.asarray(theta,dtype='float64');p=expit(x[:24]).reshape(4,6);r=np.exp(p[:,0]*np.log(100));sr=.02+.88*p[:,1];h=np.exp(np.log(2)+p[:,2]*np.log(250));sh=.02+.88*p[:,3]
    lower=np.maximum(np.log(3),np.log(2*r*1.001));dist=np.exp(lower+p[:,4]*(np.log(500)-lower));sd=.05+.85*p[:,5];k=np.count_nonzero(COMBOS[c]);wl=np.r_[x[24:24+k-1],0.];w=np.exp(wl-wl.max());w/=w.sum();gg=expit(x[27:31])
    glob=np.array([np.exp(np.log(1e-6)+gg[0]*np.log(1e4)),np.exp(np.log(.007)+gg[1]*np.log(.013/.007)),5+5*gg[2],np.exp(np.log(10)+gg[3]*np.log(100))])
    return r,sr,h,sh,dist,sd,w,glob
def component(q,typ,r,sr,h,sh,dist,sd,d_present):
    if typ==1:
        rr,w=nodes(r,r*sr,25,4);form=w@(sphere(rr[:,None]*q[None,:])**2)
    elif typ==3:
        rr,w=nodes(r,r*sr,26,3);form=w@(radial(rr[:,None]*q[None,:])**2)
    else:
        rr,wr=nodes(r,r*sr,13,4);hh,wh=nodes(h,h*sh,13,4);alpha=np.linspace(0,np.pi/2,24);wa=np.sin(alpha);wa/=wa.sum()
        fr=radial(rr[:,None,None]*np.sin(alpha)[None,:,None]*q[None,None,:]);fh=np.sinc(hh[:,None,None]*np.cos(alpha)[None,:,None]*q[None,None,:]/(2*np.pi))
        fm=np.sum(wr[:,None,None]*fr**2,0);hm=np.sum(wh[:,None,None]*fh**2,0);form=np.sum(wa[:,None]*fm*hm,0)
    if d_present:
        lp=-np.pi*q*q*(dist*sd)**2;phi=np.exp(lp);form*=(-np.expm1(2*lp))/np.maximum((-np.expm1(lp))**2+4*phi*np.sin(.5*q*dist)**2,1e-15)
    return form
def forward(theta,c,g,q):
    q=np.asarray(q,dtype='float64');r,sr,h,sh,dist,sd,w,glob=physical(theta,c);types=COMBOS[c];particle=np.zeros_like(q)
    for j,weight in enumerate(w):particle+=weight*component(q,types[j],r[j],sr[j],h[j],sh[j],dist[j],sd[j],bool(g&(1<<j)))
    bg,rs,rn,ra=glob;y=particle+bg*np.median(particle)
    if g&16:
        shape=1/(1+(np.maximum(q,0)/rs)**rn);y+=ra*particle[:5].max()/max(shape[:5].max(),1e-30)*shape
    return np.maximum(y,1e-30)

class CachedForward:
    """Reuse unchanged component forms during finite-difference Jacobian calls."""
    def __init__(self,c,g,q):self.c=int(c);self.g=int(g);self.q=np.asarray(q,dtype='float64');self.cache=OrderedDict()
    def __call__(self,theta):
        q=self.q;r,sr,h,sh,dist,sd,w,glob=physical(theta,self.c);types=COMBOS[self.c];particle=np.zeros_like(q)
        for j,weight in enumerate(w):
            typ=int(types[j]);present=bool(self.g&(1<<j));key=(typ,r[j],sr[j],h[j] if typ==2 else 0.,sh[j] if typ==2 else 0.,dist[j] if present else 0.,sd[j] if present else 0.,present)
            if key in self.cache:form=self.cache[key];self.cache.move_to_end(key)
            else:
                form=component(q,typ,r[j],sr[j],h[j],sh[j],dist[j],sd[j],present);self.cache[key]=form
                if len(self.cache)>256:self.cache.popitem(last=False)
            particle+=weight*form
        bg,rs,rn,ra=glob;y=particle+bg*np.median(particle)
        if self.g&16:
            shape=1/(1+(np.maximum(q,0)/rs)**rn);y+=ra*particle[:5].max()/max(shape[:5].max(),1e-30)*shape
        return np.maximum(y,1e-30)
