'Positive adaptive pixel-bin/PSF quadrature for the v1.1 training operator.'
import numpy as np
from scipy.special import ndtr,roots_legendre
from .tool_qinput import AF_EDGES,Q_EDGES

def rule(edges,order,halo,extra=(),max_step=None):
    edges=np.asarray(edges,float);lo,hi=edges[0]-halo,edges[-1]+halo;cuts=np.unique(np.r_[lo,hi,edges,np.asarray(extra,float)]);cuts=cuts[(cuts>=lo)&(cuts<=hi)]
    if max_step is not None:
        cuts=np.unique(np.concatenate([np.linspace(a,b,max(2,int(np.ceil((b-a)/max_step))+1)) for a,b in zip(cuts[:-1],cuts[1:])]))
    t,w=roots_legendre(order);width=np.diff(cuts);nodes=((cuts[:-1,None]+cuts[1:,None])/2+width[:,None]*t/2).ravel();weights=(width[:,None]*w/2).ravel();return nodes,weights

def response(edges,nodes,weights,sigma):
    edges=np.asarray(edges,float);raw=ndtr((edges[1:,None]-nodes[None,:])/sigma)-ndtr((edges[:-1,None]-nodes[None,:])/sigma);W=raw*weights[None,:];s=W.sum(axis=1)
    if np.any(s<=0):raise ValueError('PSF response has no numerical support')

    return W/s[:,None],float(np.max(abs(s/np.diff(edges)-1)))

def adaptive_nodes(sample,particle,nd,order=4,psf_min=.03,psf_max=1.):
    da=float(AF_EDGES[1]-AF_EDGES[0]);dq=float(Q_EDGES[1]-Q_EDGES[0]);ai=np.radians(sample.alpha_i_deg);E=sample.energy_ev;wl=1239.841984/E;k=2*np.pi/wl;shift=getattr(sample,'_measurement_af_shift',0.);qscale=getattr(sample,'_measurement_qscale',1.)
    extra=[-shift];minimum=psf_min*da
    extra+=list((AF_EDGES[:,None]+np.array([-2,2])[None,:]*minimum).ravel())
    for n in nd:

        delta=max(float(np.real(n)),0.)
        if delta<=0:continue
        critical=np.sqrt(2*delta);width=max(abs(float(np.imag(n)))/max(critical,1e-12),1e-6);extra+=list(np.degrees(critical+np.array([-4,-1,0,1,4])*width)-shift)
    thickness=max(float(sample.layer_thickness),.5);max_step=max(.0001,min(da, np.degrees(wl/(8*thickness))))
    ac,aw=rule(AF_EDGES,order,4*psf_max*da,extra,max_step)


    D=particle.D_nm;amin=max(0.,float(ac.min())+shift);amax=float(ac.max())+shift;omega_min=.06;qextra=[]
    for m in range(1,int(np.ceil((Q_EDGES[-1]+4*psf_max*dq)*qscale*D/(2*np.pi)))+1):
        qp=2*np.pi*m/D;gamma=(2*np.pi*np.pi*m*m*omega_min**2/D+1e-4)/qscale
        if gamma>=dq/2:continue
        af=np.radians([amin,amax]);qx=k*(np.cos(af)-np.cos(ai))-qp*qp/(2*k*np.cos(af));qlo=float(np.sqrt(max(qp*qp-float(np.max(qx*qx)),0)))/qscale;qhi=qp/qscale
        a,b=qlo-6*gamma,qhi+6*gamma;qextra+=list(np.linspace(a,b,max(2,int(np.ceil((b-a)/gamma))+1)))
    qc,qw=rule(Q_EDGES,order,4*psf_max*dq,qextra)
    return ac,aw,qc,qw,{'AF_nodes':len(ac),'q_nodes':len(qc),'order':order,'critical_phase_and_correlation_cuts':True,'D_fixed_for_reuse':True,'min_sigma_final_bins':psf_min,'max_sigma_final_bins':psf_max,'AF_max_segment_deg':max_step}

def integrate(image,ac,aw,qc,qw,sigma_af_bins,sigma_q_bins):
    Wa,ea=response(AF_EDGES,ac,aw,sigma_af_bins*(AF_EDGES[1]-AF_EDGES[0]));Wq,eq=response(Q_EDGES,qc,qw,sigma_q_bins*(Q_EDGES[1]-Q_EDGES[0]));return Wa@np.asarray(image).reshape(len(ac),len(qc))@Wq.T,{'constant_response_raw_error_AF':ea,'constant_response_raw_error_q':eq}
