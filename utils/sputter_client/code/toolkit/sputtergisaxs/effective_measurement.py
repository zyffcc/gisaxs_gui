'Positive native-bin measurement for effective-host v2 research.'
import numpy as np
from .tool_qinput import AF_EDGES,Q_EDGES,angles
from .stage5_measurement import NATIVE_EDGES,NATIVE_AF,Q,counts_profiles,describe
from .cell_response import adaptive_nodes,rule,response
from .forward import SampleParams
from .effective_host import metal_intensity,host_transport,combine_host


class EffectiveOperator:
    'One immutable physical context; D changes require rebuilding context.'
    def __init__(self,context,*,af_order=4,q_order=4,qscale=1.,af_shift=0.,
                 psf_af_bins=.2,psf_q_bins=.2,fast_qy=False,F_order=8,backend='cpu',fast_critical=False):
        self.c=context;self.psf_a=psf_af_bins;self.psf_q=psf_q_bins
        self.qscale=qscale;self.shift=af_shift;self.fast=fast_qy;self.F_order=F_order
        if backend not in ('cpu','cuda'):raise ValueError('Expected cpu/cuda backend')
        self.backend=backend
        if not .03<=psf_af_bins<=1 or not .03<=psf_q_bins<=1:
            raise ValueError('PSF outside declared final-bin range')

        s=SampleParams(energy_ev=context['E'],alpha_i_deg=context['ai'],
                       layer_thickness=sum(context['thick']))
        s._measurement_af_shift=af_shift;s._measurement_qscale=qscale
        nds=context['post_nds']
        if fast_critical:


            vals=np.array(nds,complex);idx=np.argsort(vals.real)
            selected=idx[np.linspace(0,len(idx)-1,5).astype(int)]
            nds=list(vals[selected])+[context['host'].decrement,vals[-1],vals[-2]]
        _,_,self.qc,self.qw,meta=adaptive_nodes(s,context['particle'],nds,q_order)
        da=float(AF_EDGES[1]-AF_EDGES[0]);extra=[-af_shift]
        extra.extend((NATIVE_EDGES[:,None]+np.array([-2,2])[None,:]*(.03*da)).ravel())
        for decrement in nds:
            if decrement.real<=0:continue
            critical=np.sqrt(2*decrement.real);width=max(abs(decrement.imag)/critical,1e-6)
            extra.extend(np.degrees(critical+np.array([-4,-1,0,1,4])*width)-af_shift)
        self.ac,self.aw=rule(NATIVE_EDGES,af_order,4*da,extra,meta['AF_max_segment_deg'])
        self.Wa,ea=response(NATIVE_EDGES,self.ac,self.aw,psf_af_bins*da)
        self.Wq,eq=response(Q_EDGES,self.qc,self.qw,psf_q_bins*(Q_EDGES[1]-Q_EDGES[0]))
        self.meta={**meta,'AF_nodes':len(self.ac),'q_nodes':len(self.qc),
                   'AF_order':af_order,'q_order':q_order,'F_order':F_order,
                   'native_AF_edges':len(NATIVE_EDGES),'complete_stack_phase_nm':sum(context['thick']),
                   'raw_constant_AF':ea,'raw_constant_q':eq,'fast_qy_uncertified':fast_qy,
                   'fast_critical_uncertified':fast_critical}

    def evaluate(self,*,af_block=12,db_xi_nm=4.,power_exponent=2.,scatter_depth_nm=5.):
        c=self.c;projected=np.zeros((len(NATIVE_AF),len(Q)))
        db=np.zeros_like(projected);power=np.zeros_like(projected);ring=np.zeros_like(projected)
        transported=np.zeros(len(NATIVE_AF));positive_weights=np.zeros(len(NATIVE_AF))
        for start in range(0,len(self.ac),af_block):
            stop=min(start+af_block,len(self.ac));af=self.ac[start:stop]+self.shift
            good=af>0
            if not good.any():continue
            aa=af[good];ang=angles(c['E'],c['ai'],self.qscale,0.,q_centres=self.qc,af_centres=aa)
            q=abs(self.qc)*self.qscale if self.fast else ang['qpar']
            if self.backend=='cuda':
                from .effective_cuda import metal_intensity_cuda
                I=metal_intensity_cuda(c,aa,q,order=self.F_order)
            else:I=metal_intensity(c,aa,q,order=self.F_order)
            wa=self.Wa[:,start:stop][:,good]


            projected+=wa@(I@self.Wq.T)
            qt=ang['q_total'];qp=ang['qpar']
            T,valid=host_transport(c,aa,depth_nm=scatter_depth_nm)
            if not valid.all():raise ValueError('Host transport has unsupported denominator')
            transported+=wa@T
            positive_weights+=wa@np.ones(len(aa))
            db+=wa@((1/(1+(qp*db_xi_nm)**2)**2)@self.Wq.T)
            power+=wa@(((1+qp/.05)**(-power_exponent))@self.Wq.T)
            ring+=wa@(np.exp(-.5*((qt-2.1)/.1)**2)@self.Wq.T)
        if np.any(positive_weights<=0):raise ValueError('No positive-angle transport support')
        transported/=positive_weights
        return {'metal':projected,'transport':transported[:,None],
                'db':db,'power':power,'ring':ring,'flat':np.ones_like(projected),
                'meta':self.meta,'transport_definition':'positive-bin-averaged conditional field-power ratio; empirical response not inverted'}


def compose(parts,host_rate,*,metal_gain=1.,host_gain=1.,db_gain=0.,
            power_gain=0.,flat_gain=0.,ring_gain=0.,gamma=0.,transmitted_fraction=1.):
    'All gains convert components to common count-rate units first.'
    gains=[metal_gain,host_gain,db_gain,power_gain,flat_gain,ring_gain]
    if not np.isfinite(gains).all() or min(gains)<0:raise ValueError('Finite nonnegative gains required')
    h=np.asarray(host_rate,float)*host_gain+db_gain*parts['db']+power_gain*parts['power']
    total=combine_host(metal_gain*parts['metal'],h,gamma=gamma,
                       transmission=parts['transport'],transmitted_fraction=transmitted_fraction)
    return total+flat_gain*parts['flat']+ring_gain*parts['ring']


def observe_rates(rate,full_N,public_N,rng):
    'One photon draw on disjoint ring-kept/ring-rejected atoms, exact supports.'
    N=np.asarray(full_N,float);P=np.asarray(public_N,float)
    if N.shape!=(2,len(NATIVE_AF),len(Q)) or P.shape!=N.shape or not np.isfinite([N,P]).all() or np.any(P<0) or np.any(N<P):
        raise ValueError('Public support must be a subset of complete support')
    r=np.broadcast_to(np.asarray(rate,float),N.shape)
    if not np.isfinite(r).all() or np.any(r<0):raise ValueError('Finite positive rates required')
    py=rng.poisson(P*r);ry=rng.poisson((N-P)*r);y=py+ry
    hy,hn,vy,vn,image,scale=counts_profiles(y,N)
    phy,phn,pvy,pvn,_,_=counts_profiles(py,P);d=describe(phy,phn,pvy,pvn)
    return dict(image=image,features=d['features'],descriptor=d,scale=scale,
                Y=hy,N=hn,VY=vy,VN=vn,public_Y=phy,public_N=phn,public_VY=pvy,public_VN=pvn,
                atomic_Y=y,atomic_N=N,public_atomic_Y=py,public_atomic_N=P)
