'Independent bounded primitives for the v0.6 joint posterior candidate.'
import numpy as np
from .forward import SampleParams
from .forward_v06 import SSCAParams
from .tool_qinput import angles,AF,Q,AF_EDGES,Q_EDGES,GRID_SHAPE,tensor
from .forward_v06 import simulate_v06,complex_amplitudes,ssca_intensity
from .forward import simulate
MATERIALS=('Si','SiO2','TiO2','PS','PMMA')
NAMES=('D','a_fraction','aspect','sigma_log','omega_rel','kappa','layer_density','layer_thickness','sigma_top','xi_top','pore_gain','pore_xi','db_fraction','db_xi','flat_fraction','ring_fraction','psf_AF_bins','psf_q_bins','qscale','af_shift_deg','mean_counts','ring_q','ring_sigma','sigma_si','xi_si')
DOMAIN={'version':'v1.1-candidate','energy_ev':[8000,14000],'interlayer_nm':[.5,100],'metals':['Au','Ag'],'numeric_receipt_required':True,'unknown_layer_is_not_verified_in_domain':True}
BOUNDS=np.array([(3,100),(0,1),(.1,2),(.03,.45),(.06,.6),(0,1),(.08,1.8),(.5,100),(.02,2),(1,100),(.03,20),(.5,20),(.0001,3),(.5,40),(.0001,.3),(.0001,3),(.03,1),(.03,1),(.97,1.03),(-.015,.015),(.5,5000),(1.8,2.3),(.03,.15),(.02,.6),(3,100)],float)
LOG=np.array([i not in (1,3,5,18,19,21) for i in range(len(NAMES))]);TRANS=BOUNDS.copy();TRANS[LOG]=np.log(TRANS[LOG])

def decode(z):
    from scipy.special import expit
    x=TRANS[:,0]+expit(z)*(TRANS[:,1]-TRANS[:,0]);x[...,LOG]=np.exp(x[...,LOG]);x[...,1]=np.exp(np.log(.4)+x[...,1]*(np.log(np.minimum(8,.55*x[...,0]))-np.log(.4)))
    return x
def encode(x):
    x=np.asarray(x,float).copy();x[...,1]=(np.log(x[...,1])-np.log(.4))/(np.log(np.minimum(8,.55*x[...,0]))-np.log(.4));x[...,LOG]=np.log(x[...,LOG]);u=(x-TRANS[:,0])/(TRANS[:,1]-TRANS[:,0]);u=np.clip(u,1e-6,1-1e-6);return np.log(u/(1-u))
def active_dimensions(material,density=None,thickness=None):
    keep=np.ones(len(NAMES),bool)
    if material=='Si':keep[[6,7]]=False
    if material!='TiO2' or density is not None and density>=.98 or thickness is not None and thickness<=0:keep[[10,11]]=False
    return keep
def physical(x,material,shape,metal,E,ai):
    x=np.asarray(x);s=SampleParams(metal=metal,substrate=material,layer_thickness=0 if material=='Si' else x[7],layer_density=1 if material=='Si' else x[6],sigma_top=x[8],xi_top=x[9],pore_gain=x[10],pore_xi=x[11],sigma_si=x[23],xi_si=x[24],energy_ev=E,alpha_i_deg=ai)
    p=SSCAParams(D_nm=x[0],base_radius_nm=x[1],height_nm=x[1]*x[2],sigma_log=x[3],omega_rel=x[4],kappa=x[5],shape=shape)
    return s,p
def observables(x,shape):
    x=np.asarray(x);asp=x[...,2];a=x[...,1];diam=2*a if shape=='hemiellipsoid' else np.where(asp>1,a*(1+asp*asp)/asp,2*a)
    return np.stack([x[...,0],diam,a*asp],axis=-1)

def image_mean(x,material,shape,metal,E,ai,*,nbins=448,nz=192,subcells=2):
    s,p=physical(x,material,shape,metal,E,ai)
    if subcells==1:qc,ac=Q,AF
    else:
        u=(np.arange(subcells)+.5)/subcells;qc=(Q_EDGES[:-1,None]+np.diff(Q_EDGES)[:,None]*u).ravel();ac=(AF_EDGES[:-1,None]+np.diff(AF_EDGES)[:,None]*u).ravel()
    ang=angles(E,ai,x[18],x[19],q_centres=qc,af_centres=ac);out=simulate_v06(s,p,ang,nbins=nbins,nz=nz);reference=max(float(np.quantile(out['particles'],.8)),1e-30)
    bg={'db_amp':x[12]*reference,'db_xi_nm':x[13],'flat':x[14]*reference,'ring_amp':x[15]*reference,'ring_q_nm_inv':x[21],'ring_sigma':x[22]}
    q=ang['qpar'];qt=ang['q_total'];mean=out['total']+bg['flat']+bg['db_amp']/(1+(q*bg['db_xi_nm'])**2)**2+bg['ring_amp']*np.exp(-.5*((qt-bg['ring_q_nm_inv'])/bg['ring_sigma'])**2)
    pre=simulate_v06(s,None,ang)['total']+bg['flat']+bg['db_amp']/(1+(q*bg['db_xi_nm'])**2)**2+bg['ring_amp']*np.exp(-.5*((qt-bg['ring_q_nm_inv'])/bg['ring_sigma'])**2)
    from scipy.ndimage import gaussian_filter
    sig=(x[16]*subcells,x[17]*subcells);mean=gaussian_filter(mean,sig,mode='nearest');pre=gaussian_filter(pre,sig,mode='nearest')
    if subcells>1:
        n,m=GRID_SHAPE;mean=mean.reshape(n,subcells,m,subcells).mean(axis=(1,3));pre=pre.reshape(n,subcells,m,subcells).mean(axis=(1,3))
    flux=x[20]/max(float(np.quantile(mean,.8)),1e-30)
    return mean*flux,pre*flux,{'sample':s.to_dict(),'particle':vars(p),'background':bg,'resolution_final_grid_bins':[float(x[16]),float(x[17])],'flux':float(flux),'counts_rng_required':True,'size_nodes':nbins,'nz_floor':nz,'subcells':subcells,'topology_stage':'not_generated','QCM_rate_is_not_mass_constraint':True}

def draw(seed):
    rng=np.random.default_rng(seed);material=str(rng.choice(MATERIALS));shape=str(rng.choice(['cap','hemiellipsoid']));metal=str(rng.choice(['Au','Ag']));z=np.log(rng.uniform(.00001,.99999,len(NAMES)));u=np.exp(z);z=np.log(u/(1-u));x=decode(z);E=float(rng.uniform(*DOMAIN['energy_ev']));ai=float(rng.uniform(.15,.7));rate=float(np.exp(rng.uniform(np.log(.05),np.log(.5))))
    return x,material,shape,metal,E,ai,rate,rng

class PreparedOperator:
    'Exact reuse of fields/form factors when only correlation/noise changes.'
    FIXED=(1,2,3,6,7,8,9,10,11,18,19,23,24)
    def __init__(self,x,material,shape,metal,E,ai,nbins=448,nz=192,subcells=2,backend='cpu'):
        self.x=np.asarray(x).copy();self.material=material;self.shape=shape;self.metal=metal;self.E=E;self.ai=ai;self.nbins=nbins;self.nz=nz;self.subcells=subcells;self.backend=backend
        s,p=physical(x,material,shape,metal,E,ai);self.sample=s;self.particle=p;u=(np.arange(subcells)+.5)/subcells;qc=(Q_EDGES[:-1,None]+np.diff(Q_EDGES)[:,None]*u).ravel();ac=(AF_EDGES[:-1,None]+np.diff(AF_EDGES)[:,None]*u).ravel();self.ang=angles(E,ai,x[18],x[19],q_centres=qc,af_centres=ac)
        if backend=='cuda':
            from .forward_v06_gpu import complex_amplitudes_cuda
            self.F,self.R,self.weights,self.pref=complex_amplitudes_cuda(s,p.state(),self.ang,nbins,nz)
        elif backend=='cpu':self.F,self.R,self.weights,self.pref=complex_amplitudes(s,p.state(),self.ang,nbins,nz)
        else:raise ValueError('Unknown evaluator backend')
        self.base=simulate(s,None,self.ang)['total']
    def evaluate(self,x):
        x=np.asarray(x)
        if not np.allclose(x[list(self.FIXED)],self.x[list(self.FIXED)],rtol=0,atol=1e-12):raise ValueError('Cached physical field changed; cannot reuse operator')
        q=self.ang['qpar'];qt=self.ang['q_total']
        if self.backend=='cuda':
            from .forward_v06_gpu import ssca_cuda
            v=ssca_cuda(self.F,self.R,self.weights,q.ravel(),x[0],x[4]*x[0],x[5])
        else:v=ssca_intensity(self.F,self.R,self.weights,q.ravel(),x[0],x[4]*x[0],x[5])
        Ip=(self.pref*(self.x[0]/x[0])**2*v).reshape(q.shape)
        if not np.isfinite(Ip).all() or Ip.min()<-1e-7*max(float(abs(Ip).max()),1e-30):raise ValueError('Invalid cached SSCA positivity')
        Ip=np.maximum(Ip,0);reference=max(float(np.quantile(Ip,.8)),1e-30);bg={'db_amp':x[12]*reference,'db_xi_nm':x[13],'flat':x[14]*reference,'ring_amp':x[15]*reference,'ring_q_nm_inv':x[21],'ring_sigma':x[22]};background=bg['flat']+bg['db_amp']/(1+(q*bg['db_xi_nm'])**2)**2+bg['ring_amp']*np.exp(-.5*((qt-bg['ring_q_nm_inv'])/bg['ring_sigma'])**2)
        from scipy.ndimage import gaussian_filter
        sig=(x[16]*self.subcells,x[17]*self.subcells);mean=gaussian_filter(Ip+self.base+background,sig,mode='nearest');pre=gaussian_filter(self.base+background,sig,mode='nearest');n,m=GRID_SHAPE;mean=mean.reshape(n,self.subcells,m,self.subcells).mean(axis=(1,3));pre=pre.reshape(n,self.subcells,m,self.subcells).mean(axis=(1,3));flux=x[20]/max(float(np.quantile(mean,.8)),1e-30)
        s,p=physical(x,self.material,self.shape,self.metal,self.E,self.ai);meta={'sample':s.to_dict(),'particle':vars(p),'background':bg,'resolution_final_grid_bins':[float(x[16]),float(x[17])],'flux':float(flux),'size_nodes':self.nbins,'nz_floor':self.nz,'subcells':self.subcells,'backend':self.backend,'operator':'ExactFandfieldreusesamefixedprimitiveprefix;newSSCA D/omega/kappa andnoise','topology_stage':'not_generated','QCM_rate_is_not_mass_constraint':True}
        return mean*flux,pre*flux,meta

def conditional_D(rng,a):
    'D|a under original logD and conditional loga prior, rejection exact.'
    lo=max(3.,a/.55);Lmin=np.log(min(8,.55*lo)/.4)
    while True:
        D=float(np.exp(rng.uniform(np.log(lo),np.log(100))))
        if rng.random()<Lmin/np.log(min(8,.55*D)/.4):return D
