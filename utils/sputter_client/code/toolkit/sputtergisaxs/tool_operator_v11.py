'v1.1 exact field reuse with adaptive positive measurement integration.'
import numpy as np
from .tool_prior import physical
from .forward import _stack,simulate
from .forward_v06 import complex_amplitudes,ssca_intensity
from .tool_qinput import angles
from .cell_response import adaptive_nodes,integrate

class AdaptiveOperator:
    FIXED=(0,1,2,3,6,7,8,9,10,11,18,19,23,24)
    def __init__(self,x,material,shape,metal,E,ai,order=4,nbins=448,nz=192,backend='cuda'):
        self.x=np.asarray(x,float).copy();self.material=material;self.shape=shape;self.metal=metal;self.E=E;self.ai=ai;self.order=order;self.nbins=nbins;self.nz=nz;self.backend=backend;s,p=physical(x,material,shape,metal,E,ai);self.sample=s;self.particle=p;s._measurement_af_shift=x[19];s._measurement_qscale=x[18];nd,_,_,_,_=_stack(s,p.state(),nbins);self.ac,self.aw,self.qc,self.qw,self.rule_meta=adaptive_nodes(s,p,nd,order);self.ang=angles(E,ai,x[18],x[19],q_centres=self.qc,af_centres=self.ac)
        if backend=='cuda':
            from .forward_v06_gpu import complex_amplitudes_cuda
            self.F,self.R,self.weights,self.pref=complex_amplitudes_cuda(s,p.state(),self.ang,nbins,nz)
        elif backend=='cpu':self.F,self.R,self.weights,self.pref=complex_amplitudes(s,p.state(),self.ang,nbins,nz)
        else:raise ValueError('Unknown backend')
        self.base=simulate(s,None,self.ang)['total'];self.base=np.where(self.ang['alpha_f']>0,self.base,0.)
    def evaluate(self,x):
        x=np.asarray(x,float)
        if not np.allclose(x[list(self.FIXED)],self.x[list(self.FIXED)],rtol=0,atol=1e-12):raise ValueError('Adaptive field family changed D/shape/optics/geometry')
        q=self.ang['qpar'];qt=self.ang['q_total']
        if self.backend=='cuda':
            from .forward_v06_gpu import ssca_cuda
            v=ssca_cuda(self.F,self.R,self.weights,q.ravel(),x[0],x[4]*x[0],x[5])
        else:v=ssca_intensity(self.F,self.R,self.weights,q.ravel(),x[0],x[4]*x[0],x[5])
        Ip=(self.pref*v).reshape(q.shape)
        if not np.isfinite(Ip).all() or Ip.min()<-1e-7*max(float(abs(Ip).max()),1e-30):raise ValueError('Nonfinite/negative SSCA')
        Ip=np.maximum(Ip,0)
        def project(v):return integrate(v,self.ac,self.aw,self.qc,self.qw,x[16],x[17])[0]
        particle=project(Ip);base=project(self.base);reference=max(float(np.quantile(particle,.8)),1e-30);bg={'db_amp':x[12]*reference,'db_xi_nm':x[13],'flat':x[14]*reference,'ring_amp':x[15]*reference,'ring_q_nm_inv':x[21],'ring_sigma':x[22]};background=project(bg['flat']+bg['db_amp']/(1+(q*bg['db_xi_nm'])**2)**2+bg['ring_amp']*np.exp(-.5*((qt-bg['ring_q_nm_inv'])/bg['ring_sigma'])**2));mean=particle+base+background;pre=base+background;flux=x[20]/max(float(np.quantile(mean,.8)),1e-30);s,p=physical(x,self.material,self.shape,self.metal,self.E,self.ai)
        return mean*flux,pre*flux,{'sample':s.to_dict(),'particle':vars(p),'background':bg,'flux':float(flux),'operator':'PositiveGaussianbin-response adaptiveGL withcritical/phase/correlation cuts; finite4sigmahalo','resolution_final_grid_bins':[float(x[16]),float(x[17])],'rule':self.rule_meta,'backend':self.backend,'size_nodes':self.nbins,'nz_floor':self.nz,'topology_stage':'not_output','reference_amplitude':'quantileofprojectedparticleonfixedfinalbins,notlatentnode-countquantile'}
