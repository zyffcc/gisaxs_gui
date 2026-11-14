'Float64 CUDA adapter for the new effective-host amplitudes.'
import numpy as np
from .formfactor import cap_profile,hemiellipsoid_profile,_gauss_legendre
from .effective_host import field_arrays,_cell_field,metal_quadrature
from .optics import nd,wavelength_nm


def _cell_field_torch(kz,down,up,distance,*,all_shallow=False):
    'Same critical/deep branches as CPU; evaluate only selected exponentials.'
    import torch
    kz,down,up,distance=torch.broadcast_tensors(kz,down,up,distance)
    phase=kz*distance
    if all_shallow:


        return (down+up)*torch.cos(phase)+1j*kz*(down-up)*distance*torch.sinc(phase/np.pi)
    deep=abs(phase.imag)>4;shallow=~deep
    out=torch.empty_like(phase)
    out[shallow]=(down[shallow]+up[shallow])*torch.cos(phase[shallow])+1j*kz[shallow]*(down[shallow]-up[shallow])*distance[shallow]*torch.sinc(phase[shallow]/np.pi)
    out[deep]=down[deep]*torch.exp(1j*phase[deep])+up[deep]*torch.exp(-1j*phase[deep])
    return out


def _static(context,order,device,quadrature,inc):
    'One bounded device cache per context, with CPU-identical boundary gather.'
    import torch
    signature=(str(device),order,context['E'],context['ai'],tuple(context['post_nds']),
               tuple(context['thick']),tuple(context['sigma']),id(quadrature))
    old=context.get('_cuda_static')
    if old is not None and old[0]==signature:return old[1]
    e=context['edges'];cells=len(e)-1;rows=[]
    for z,rad,weight in quadrature:
        upward=np.clip(np.searchsorted(e,z,side='right')-1,0,cells-1)
        layer=cells-upward;distance=e[upward+1]-z
        Ei=_cell_field(inc['kz'][layer,0],inc['T'][layer,0],inc['R'][layer,0],distance)
        device_row=tuple(torch.as_tensor(value,device=device,dtype=dtype) for value,dtype in
            [(layer,torch.long),(distance,torch.float64),(Ei,torch.complex128),
             (rad,torch.float64),(weight,torch.complex128)])
        rows.append(device_row+(layer,distance))
    context['_cuda_static']=(signature,rows)
    return rows


def metal_intensity_cuda(context,af_deg,qpar_nm_inv,*,order=8,af_block=12):
    import torch
    from .bessel_cuda import j1
    from .forward_v06_gpu import ssca_cuda
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if torch.cuda.get_device_capability()[0]<7:raise RuntimeError('This runtime requires compute capability>=7.0; do not retry oldP100')
    af=np.atleast_1d(af_deg);q=np.asarray(qpar_nm_inv,float);separable=q.ndim==1
    if q.ndim==1:q=np.broadcast_to(q,(len(af),len(q)))
    if q.ndim!=2 or q.shape[0]!=len(af) or not np.isfinite(q).all() or np.any(q<0):raise ValueError('Expected finite AF/q grid')
    p=context['particle'];quadrature=metal_quadrature(context,order)
    out=np.zeros(q.shape);k=2*np.pi/wavelength_nm(context['E']);pref=(k*k/(4*np.pi))**2*context['n']
    device=torch.device('cuda',torch.cuda.current_device())
    free,_=torch.cuda.mem_get_info(device)
    largest=max(len(z) for z,_,_ in quadrature)


    one_ray_bytes=max(1,largest*q.shape[1]*8*12)
    static_bytes=sum(len(z) for z,_,_ in quadrature)*56
    available=max(0,int(free*.5)-static_bytes)
    if available<one_ray_bytes:raise MemoryError('One exact AF ray exceeds the safe device buffer; use a larger device, not fewer quadrature nodes')
    af_block=max(1,min(af_block,int(available//one_ray_bytes)))
    context['_last_cuda_AF_block']=af_block
    with torch.no_grad():
        for start in range(0,len(af),af_block):
            stop=min(start+af_block,len(af));qq=torch.as_tensor(np.array(q[start:stop]),device='cuda',dtype=torch.float64)
            inc,rec=field_arrays(context,af[start:stop])
            static=_static(context,order,device,quadrature,inc)


            kz,down,up=(torch.as_tensor(rec[key],device=device,dtype=torch.complex128) for key in ('kz','T','R'))
            F=torch.empty((len(context['R']),)+qq.shape,device='cuda',dtype=torch.complex128)
            for mark,(layer,distance,Ei,rr,ww,cpu_layer,cpu_distance) in enumerate(static):
                shallow=float(np.max(abs(rec['kz'][cpu_layer].imag*cpu_distance[:,None])))<=3.9
                field=_cell_field_torch(kz[layer].T,down[layer].T,up[layer].T,distance,all_shallow=shallow)*Ei[None,:]
                if separable:
                    arg=rr[:,None]*qq[0,None,:];safe=torch.where(abs(arg)<1e-7,torch.ones_like(arg),arg)
                    J=torch.where(abs(arg)<1e-7,1-arg*arg/8,2*j1(arg)/safe)
                    F[mark]=field@(ww[:,None]*J)
                else:
                    arg=rr[None,:,None]*qq[:,None,:];safe=torch.where(abs(arg)<1e-7,torch.ones_like(arg),arg)
                    J=torch.where(abs(arg)<1e-7,1-arg*arg/8,2*j1(arg)/safe)
                    F[mark]=torch.sum(field[:,:,None]*ww[None,:,None]*J,dim=1)
            power=ssca_cuda(F.reshape(len(context['R']),-1),context['R'],context['w'],qq.ravel(),p.D_nm,p.omega_rel*p.D_nm,p.kappa)

            if not np.isfinite(power).all() or float(power.min()) < -1e-7*max(float(abs(power).max()),1e-30):raise ValueError('CUDA SSCA positivity failure')
            out[start:stop]=pref*np.maximum(power,0).reshape(tuple(qq.shape))
    return out
