'Stage-five joint private physics posterior; separate from old artifacts.'
import math
import numpy as np
import torch
from torch import nn
from . import tool_prior,stage5_prior

DIM=25
META_DIM=16
INPUT_MODES=('features','qimage','hybrid')
FEATURE_SCALE=np.array([1.,1.]*4+[.4]+[2.]*4+[3.]*4+[1.]*3,np.float32)

def encode_primitives(x):
    z=tool_prior.encode(x);z[...,:4]=stage5_prior.target(x)
    return z.astype(np.float32)

def decode_primitives(z):
    x=tool_prior.decode(z);x[...,:4]=stage5_prior.decode(z[...,:4])
    return x

def encode_inverse_primitives(x,out):
    'Physical structural support is bounded; outside D is inactive.'
    x=np.asarray(x,float);out=np.broadcast_to(np.asarray(out,int),x.shape[:-1])
    if np.any((out<0)|(out>2)) or np.any(stage5_prior.out_class(x[...,0])!=out):
        raise ValueError('Range class disagrees with physical truth')
    z=encode_primitives(x);lo,hi=stage5_prior.D_RANGE
    u=(np.log(x[...,0])-np.log(lo))/np.log(hi/lo);inside=out==0
    if np.any(inside&((u<=0)|(u>=1))):raise ValueError('C0 truth must be strictly inside the open prior range')

    d=np.zeros_like(u);d[inside]=np.log(u[inside])-np.log1p(-u[inside]);z[...,0]=d
    upper=np.where(inside,np.minimum(8,.55*x[...,0]),np.where(out==1,.55*lo,8.))
    ur=(np.log(x[...,1])-np.log(.4))/np.log(upper/.4);ua=(np.log(x[...,2])-np.log(.1))/np.log(20.)
    if np.any((ur<=0)|(ur>=1)|(ua<=0)|(ua>=1)):raise ValueError('Radius/aspect truth outside open structural support')
    z[...,1]=np.log(ur)-np.log1p(-ur);z[...,2]=np.log(ua)-np.log1p(-ua)
    return z

def decode_inverse_primitives(z,out):
    'Outside-D draws keep other labels but have D=NaN, not an invented D.'
    from scipy.special import expit
    z=np.asarray(z,float);out=np.broadcast_to(np.asarray(out,int),z.shape[:-1])
    if np.any((out<0)|(out>2)):raise ValueError('Invalid posterior range class')
    safe=z.copy();safe[...,:3]=0.;x=decode_primitives(safe);lo,hi=stage5_prior.D_RANGE
    D=np.exp(np.log(lo)+expit(z[...,0])*np.log(hi/lo));upper=np.where(out==0,np.minimum(8,.55*D),np.where(out==1,.55*lo,8.))
    ur=expit(z[...,1]);ua=expit(z[...,2]);x[...,0]=np.where(out==0,D,np.nan)
    x[...,1]=np.exp((1-ur)*np.log(.4)+ur*np.log(upper));x[...,2]=np.exp((1-ua)*np.log(.1)+ua*np.log(2.))
    return x

def inverse_quantities(z,out,shape):
    'Derive public quantities per joint draw, after inverse decoding.'
    x=decode_inverse_primitives(z,out);a=x[...,1];asp=x[...,2];shape=np.asarray(shape)
    diameter=np.where((shape==0)&(asp>1),a*(1+asp**2)/asp,2*a)
    return np.stack([x[...,0],diameter,a*asp,x[...,3]],axis=-1)

def prior_public_log_prob(x,out):
    'Unconditional public prior density in the declared inverse coordinates.'
    x=np.asarray(x,float);out=np.broadcast_to(np.asarray(out,int),x.shape[:-1]);z=encode_inverse_primitives(x,out)
    lo,hi=stage5_prior.D_RANGE;ud=(np.log(x[...,0])-np.log(lo))/np.log(hi/lo);v=(x[...,3]-.03)/.42

    zz=z.astype(np.float64);upper=np.where(out==0,np.minimum(8,.55*x[...,0]),np.where(out==1,.55*lo,8.))
    def logistic_log_density(t):return -np.logaddexp(0.,-t)-np.logaddexp(0.,t)
    ans=np.log(np.where(out==0,.8,.1)/2)+logistic_log_density(zz[...,2])+logistic_log_density(zz[...,3])-2*np.log(1-2e-5)+logistic_log_density(zz[...,1])
    inside=out==0;dpart=np.zeros_like(ud)
    dpart[inside]=logistic_log_density(zz[...,0][inside])
    for c,(a,b) in enumerate(((3.,lo),(hi,100.)),1):
        sel=out==c
        if not np.any(sel):continue
        radius=x[...,1][sel];start=np.log(np.maximum(a,radius/.55));end=np.log(b);cut=np.log(8/.55)
        left=np.minimum(end,cut);right=np.maximum(start,cut);constant=np.log(.55/.4)
        integral=np.where(start<left,np.log((left+constant)/(start+constant)),0.)+np.maximum(end-right,0.)/np.log(20)
        density=integral/np.log(b/a)
        if np.any(density<=0):raise ValueError('Zero conditional-radius prior density')
        dpart[sel]=np.log(density)+np.log(np.log(upper[sel]/.4))
    return (ans+dpart).astype(np.float32)

def valid_physical_draws(x):
    'Do not clip or silently discard posterior mass outside training physics.'
    x=np.asarray(x,float)
    return (np.isfinite(x).all(-1)&(x[...,0]>=3)&(x[...,0]<=100)&
            (x[...,1]>=.4)&(x[...,1]<=np.minimum(8,.55*x[...,0]))&
            (x[...,2]>=.1)&(x[...,2]<=2)&(x[...,3]>=.03)&(x[...,3]<=.45))

def active_masks(material,primitives,out):
    'Inactive layer placeholders are not trained as observed labels.'
    material=np.asarray(material,int);x=np.asarray(primitives);out=np.asarray(out)
    keep=np.ones((len(material),DIM),bool)
    for i,mi in enumerate(material):
        keep[i]=tool_prior.active_dimensions(tool_prior.MATERIALS[mi],x[i,6],x[i,7])
    keep[:,0]&=out==0
    return keep

def optional_masks(indices,epoch=0,mode='random',device=None):
    'Identical deterministic availability for all architectures and CAL.'
    ids=torch.as_tensor(indices,device=device,dtype=torch.int64)
    if mode=='none':return torch.zeros(len(ids),4,device=ids.device,dtype=torch.bool)
    if mode=='all':return torch.ones(len(ids),4,device=ids.device,dtype=torch.bool)
    if mode=='layer':
        a=torch.zeros(len(ids),4,device=ids.device,dtype=torch.bool);a[:,:2]=True;return a
    if mode!='random':raise ValueError('Unknown optional mode')

    code=(ids*1103515245+(int(epoch)+1)*12345+951907)&2147483647
    u=(code%10000).float()/10000
    flags=torch.stack([((code*(j*2+3)+7411*(j+1))%10000)>=5000 for j in range(4)],1)
    flags[u<.5]=False;flags[u>=.75]=True
    return flags

def prepare(image,pre,features,mandatory,optional,known):
    'Return6channel q image,40features and16metadata, no host paths.'
    n=len(features);dev=features.device
    if features.shape!=(n,40) or mandatory.shape!=(n,5) or optional.shape!=(n,3) or known.shape!=(n,4):
        raise ValueError('Stage5 required/optional schema differs')
    if not torch.isfinite(mandatory).all():raise ValueError('Invalid measured geometry/noise metadata')
    known=known.bool().clone();mi=torch.where(known[:,0],optional[:,0],0)
    if torch.any(known[:,0]&((mi<0)|(mi>4)|(mi!=mi.long())|~torch.isfinite(mi))):raise ValueError('Invalid supplied material')

    known[:,1]&=~(known[:,0]&(mi==0))
    thick=torch.where(known[:,1],optional[:,1],torch.ones(n,device=dev))
    qcm=torch.where(known[:,3],optional[:,2],torch.ones(n,device=dev))
    if not torch.isfinite(thick).all() or torch.any(thick<=0) or not torch.isfinite(qcm).all() or torch.any(qcm<=0):
        raise ValueError('Invalid supplied thickness or QCM')
    mt=torch.zeros(n,META_DIM,device=dev,dtype=mandatory.dtype);mt[:,:5]=mandatory
    mt[:,5:10]=torch.nn.functional.one_hot(mi.long(),5)*known[:,0,None]
    mt[:,10]=known[:,0];mt[:,11]=torch.log(thick/10)*known[:,1];mt[:,12]=known[:,1]
    mt[:,13]=torch.log(qcm/.2)*known[:,3];mt[:,14]=known[:,3];mt[:,15]=known[:,2]
    vals=features[:,:20].clone();mask=features[:,20:].bool().clone()
    mask[:,13:17]&=known[:,2,None]
    vals=torch.where(mask,vals,torch.zeros_like(vals))
    scale=torch.as_tensor(FEATURE_SCALE,device=dev)
    feat=torch.cat([vals/scale,mask.to(vals.dtype)],1)
    im=None
    if image is not None:
        if image.shape!=(n,2,32,70) or pre.shape!=image.shape:raise ValueError('Invalid q image')

        p=torch.where(known[:,2,None,None,None],pre,torch.zeros_like(pre))
        from .tool_qinput import AF,Q
        coords=np.stack(np.meshgrid((AF-.76)/.74,(Q-.925)/.875,indexing='ij'))
        c=torch.as_tensor(coords,device=dev,dtype=image.dtype)[None].expand(n,-1,-1,-1)
        im=torch.cat([image,p,c],1)
    if not torch.isfinite(feat).all():raise ValueError('Nonfinite supplied observed feature')
    return im,feat,mt,known

def mixture_log_prob(logits,mu,L,z,categories,active,cpf):
    'Proper marginal Gaussian likelihood for each declared active subset.'
    n=len(z);pi=torch.log_softmax(logits,-1);b=torch.arange(n,device=z.device)[:,None]
    k=categories[:,None]*cpf+torch.arange(cpf,device=z.device)
    means=mu[b,k];factors=L[b,k];terms=torch.empty(n,device=z.device,dtype=z.dtype)
    patterns,inverse=torch.unique(active,dim=0,return_inverse=True)
    for j,pattern in enumerate(patterns):
        rows=inverse==j;ix=torch.nonzero(pattern).ravel();d=len(ix)
        if d==DIM:chol=factors[rows]
        else:
            fac=factors[rows][:,:,ix,:];cov=fac@fac.transpose(-1,-2)

            chol=torch.linalg.cholesky(cov)
        delta=z[rows][:,None,ix]-means[rows][:,:,ix]
        v=torch.linalg.solve_triangular(chol,delta[...,None],upper=False).squeeze(-1)
        logn=-.5*(v*v).sum(-1)-torch.log(torch.diagonal(chol,dim1=-2,dim2=-1)).sum(-1)-.5*d*math.log(2*math.pi)
        terms[rows]=torch.logsumexp(pi[b,k][rows]+logn,-1)
    return terms

def public_log_prob(logits,mu,L,z,shape,out,active,cpf):
    'Joint structure/shape/range CAL score; private quantities marginalised.'
    terms=[]
    public=active.clone();public[:,4:]=False
    for material in range(5):
        terms.append(mixture_log_prob(logits,mu,L,z,out*10+material*2+shape,public,cpf))
    return torch.logsumexp(torch.stack(terms,1),1)

class JointPosterior(nn.Module):
    def __init__(self,mode='features',components=2,hidden=192):
        super().__init__()
        if mode not in INPUT_MODES:raise ValueError('Unknown input branch')
        self.mode=mode;self.cpf=components;self.K=30*components;self.dim=DIM
        self.ntri=DIM*(DIM+1)//2
        if mode!='features':
            self.conv=nn.Sequential(nn.Conv2d(6,24,3,stride=(2,1),padding=1),nn.SiLU(),
                nn.Conv2d(24,32,3,stride=(2,1),padding=1),nn.SiLU(),
                nn.AdaptiveAvgPool2d((4,70)),nn.Flatten())
        size=META_DIM+(40 if mode!='qimage' else 0)+(32*4*70 if mode!='features' else 0)
        self.embed=nn.Sequential(nn.Linear(size,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.out=nn.Linear(hidden,self.K*(1+DIM+self.ntri))
        row,col=torch.tril_indices(DIM,DIM);self.register_buffer('tril_row',row);self.register_buffer('tril_col',col)
        nn.init.normal_(self.out.weight,std=.002);nn.init.zeros_(self.out.bias)
        with torch.no_grad():
            bias=self.out.bias.reshape(self.K,-1);diag=torch.nonzero(row==col).ravel()
            bias[:,1+DIM+diag]=math.log(math.expm1(1.8))
            bias[:,1:1+DIM]=torch.randn(self.K,DIM)*.1

    def forward(self,im,feat,meta):
        values=[meta]
        if self.mode!='qimage':values.append(feat)
        if self.mode!='features':values.append(self.conv(im))
        emb=self.embed(torch.cat(values,1));raw=self.out(emb).reshape(-1,self.K,1+DIM+self.ntri)
        tri=raw[:,:,1+DIM:];L=torch.zeros(*tri.shape[:2],DIM,DIM,device=tri.device,dtype=tri.dtype)
        L[:,:,self.tril_row,self.tril_col]=tri
        diag=self.tril_row==self.tril_col;ix=self.tril_row[diag]
        L[:,:,ix,ix]=torch.nn.functional.softplus(tri[:,:,diag])+.03
        logits=raw[:,:,0]

        oc=torch.logsumexp(logits.reshape(-1,3,10,self.cpf),dim=(2,3))
        return logits,raw[:,:,1:1+DIM],L,oc,emb

    def loss(self,im,feat,meta,z,cat,active,out,reduction='mean'):
        pi,mu,L,oc,emb=self(im,feat,meta)
        nll=-mixture_log_prob(pi,mu,L,z,out*10+cat,active,self.cpf)
        ce=torch.nn.functional.cross_entropy(oc,out,reduction='none')

        return nll.mean() if reduction=='mean' else (nll,nll-ce,ce)

    @torch.no_grad()
    def sample(self,im,feat,meta,n=256,generator=None,condition_out=None):
        pi,mu,L,oc,emb=self(im,feat,meta);pclass=torch.softmax(oc,-1)
        if condition_out is not None:
            c=torch.as_tensor(condition_out,device=pi.device,dtype=torch.int64)
            if c.ndim==0:c=c.expand(len(meta))
            if c.shape!=(len(meta),) or torch.any((c<0)|(c>2)):raise ValueError('Invalid joint range condition')
            ixclass=torch.arange(self.K,device=pi.device)//(10*self.cpf)
            pi=pi.masked_fill(ixclass[None]!=c[:,None],-torch.inf)
        probs=torch.softmax(pi,-1)
        ix=torch.multinomial(probs,n,replacement=True,generator=generator);b=torch.arange(len(meta),device=meta.device)[:,None]
        means=mu[b,ix];chol=L[b,ix];eps=torch.randn(*means.shape,device=meta.device,generator=generator)
        z=means+(chol@eps[...,None]).squeeze(-1)
        cat=ix//self.cpf
        return {'z':z.cpu().numpy(),'out_class':(cat//10).cpu().numpy(),
                'material':((cat%10)//2).cpu().numpy(),'shape':(cat%2).cpu().numpy(),
                'p_out_class':pclass.cpu().numpy(),'embedding':emb.cpu().numpy(),
                'condition_out':condition_out if isinstance(condition_out,(int,type(None))) else c.cpu().numpy()}

def ood_fit(emb):
    'TRAIN embedding location/covariance only; CAL sets the fixed quantile.'
    a=np.asarray(emb,float);mu=a.mean(0);cov=np.cov(a,rowvar=False)
    ridge=max(float(np.trace(cov)/len(mu)),1e-12)*.001
    precision=np.linalg.inv(cov+ridge*np.eye(len(mu)))
    return {'mean':mu,'precision':precision,'ridge':ridge}

def ood_score(emb,fit):
    delta=np.asarray(emb,float)-fit['mean']
    return np.einsum('bi,ij,bj->b',delta,fit['precision'],delta)
