'Native v3 module adapter for NR pilot, independent of frozen tools.'
from __future__ import annotations
import numpy as np
from .linecut_v3 import Linecut,Model,NATIVE_AF
from .tool_qinput import Q,Q_EDGES
from .substrate_bimodal_v3 import BimodalModel
from .polymer_linecut_v3 import EmbeddedKernel
from .network_linecut_v3 import NetworkModel
from .scenario_ts import build_context,metal_intensity,TSParams
from .effective_host import HostOptics
from .optics import nd

MODULES=('island','Si_bimodal','highq_self','polymer_embed','TS_layer')


def cut_for(af,energy_ev,alpha_i_deg):
    e=np.ones((2,len(af),len(Q)))
    return Linecut(np.zeros_like(e),e,np.asarray(af),Q_EDGES,energy_ev,alpha_i_deg)


def rates(parameters,*,nsize=224,nz=64,af_block=8,af_values=None):
    'Return noise-free full native rates from one declared physical family.'
    d=parameters;module=d['module']
    if module not in MODULES or d['metal'] not in ('Ag','Au'):raise ValueError('Known NR module and element required')
    p=np.asarray(d['p'],float);shape=d['shape'];ap=d['approximation'];out=[]
    angles=NATIVE_AF if af_values is None else np.asarray(af_values,float)
    if module=='polymer_embed':
        R=np.exp(p[1]);b=p[11]*min(R,d['polymer_thickness_nm'])
        if not 0<=b<=d['polymer_thickness_nm']:raise ValueError('Embedded base crosses unimplemented polymer bottom interface')
        max_r=R*np.exp(p[2]*8-.5*p[2]**2)
        if max_r-b>=600:raise ValueError('Lognormal quadrature exceeds declared optical z-domain')
    for lo in range(0,len(angles),af_block):
        c=cut_for(angles[lo:lo+af_block],d['energy_ev'],d['alpha_i_deg'])
        if module=='polymer_embed':
            optical=nd('PS',energy_ev=12000.)
            kernel=EmbeddedKernel(c,d['polymer_thickness_nm'],d['metal'],optical.real,optical.imag,
                nsize=nsize,nz=nz,host_material='PS' if d['polymer_type']=='PS' else None)
            if d['polymer_type']=='PSBMA':kernel.delta12=2e-6;kernel.beta12=2e-8
            value=kernel.rates(p,ap)
        elif module=='TS_layer':
            physical=TSParams(**d['TS']);ctx=build_context(HostOptics(roughness_nm=d['interface_rms_nm']),physical,d['metal'],d['energy_ev'],d['alpha_i_deg'])
            grid=Model(c,'cylinder','DA',d['critical_angle_deg'])
            intensity=metal_intensity(ctx,c.af_deg,grid.qpar.reshape(len(c.af_deg),-1),order=8).reshape(grid.qpar.shape)
            signal=np.einsum('afq,q->af',intensity,grid.qw)
            A,B,C=np.exp(np.array([p[6:9],p[10:13]])).T
            tail=np.einsum('fq,q->f',(grid.qy/.5)**(-p[9]),grid.qw)
            value=A[:,None,None]*signal[None]+B[:,None,None]+C[:,None,None]*tail[None,None]
        else:
            if 'height_nm' in d:
                from .neural_refine_height import HeightModel
                model=HeightModel(c,d['height_nm'],shape,ap,d['critical_angle_deg'],nsize=nsize,nz=nz)
            else:
                model=Model(c,shape,ap,d['critical_angle_deg'],nsize=nsize,nz=nz,side_response='independent',
                    cylinder_tangent=False,hemisphere_tangent=False)
            if module=='island':value=model.atomic_rates(p)
            else:
                signal=model.signal(np.exp(p[0]),np.exp(p[1]),p[2],p[3],correlated=module=='Si_bimodal')
                if module=='Si_bimodal':
                    Rs,ratio=np.exp(p[11:13]);power=4 if shape=='cylinder' else 6
                    if 'height_nm' in d:
                        hs=1. if shape=='cylinder' else Rs
                        small=HeightModel(c,hs,shape,ap,d['critical_angle_deg'],nsize=nsize,nz=nz)
                        volume_ratio=(Rs**2*hs/(np.exp(p[1])**2*d['height_nm']))**2
                        signal=signal+ratio*volume_ratio*small.signal(1.,Rs,.15,1.,correlated=False)
                    else:signal=signal+ratio*(Rs/np.exp(p[1]))**power*model.signal(1.,Rs,.15,1.,correlated=False)
                A,B,C=np.exp(np.array([p[4:7],p[8:11]])).T
                tail=np.einsum('fq,q->f',(model.qy/.5)**(-p[7]),model.qw)
                value=A[:,None,None]*signal[None]+B[:,None,None]+C[:,None,None]*tail[None,None]
        if not np.isfinite(value).all() or np.any(value<0):raise ValueError('Nonpositive or nonfinite native forward')
        out.append(value)
    return np.concatenate(out,axis=1)


def draw(module,metal,rng,*,energy_ev=12000.,alpha_i_deg=.4,anchor_Dcorr=None,free_height=False):
    'DEV prior, not a frozen production training recipe or new TEST.'
    if module not in MODULES:raise ValueError(module)
    D=float(np.exp(rng.uniform(np.log(4.),np.log(90.))))
    if anchor_Dcorr is not None:D=float(np.clip(anchor_Dcorr*np.exp(rng.uniform(np.log(.65),np.log(1.6))),2.,120.))
    if module=='TS_layer':D=max(3.,D)
    R=float(np.clip(D*np.exp(rng.uniform(np.log(.05),np.log(.38))),.35,12.))
    shape=str(rng.choice(['cylinder','hemisphere']));ap=str(rng.choice(['DA','LMA_common_D']))
    sigma=float(rng.uniform(.03,.45));omega=float(rng.uniform(.04,.8))

    p=np.array([np.log(D),np.log(R),sigma,omega,0.,-12.,-12.,rng.uniform(.3,3.),rng.normal(0,.15),-12.,-12.])
    d=dict(module=module,metal=metal,shape=shape,approximation=ap,energy_ev=float(energy_ev),alpha_i_deg=float(alpha_i_deg),
        critical_angle_deg=float(rng.uniform(.10,.27)),p=p.tolist())
    truth=dict(R_number_mean_nm=R,radius_sigma_log=sigma,height_nm=1. if shape=='cylinder' else R,
        height_scope='fixed_1nm_cylinder' if shape=='cylinder' else 'hemisphere_h_equals_R',D_nm=D)
    if module=='Si_bimodal':
        small=float(rng.uniform(.5,1.5));ratio=float(np.exp(rng.uniform(np.log(.1),np.log(500))))
        p=np.r_[p,np.log(small),np.log(ratio)];truth.update(R_small_number_mean_nm=small,secondary_number_ratio=ratio)
    elif module=='highq_self':truth.pop('D_nm')
    elif module=='polymer_embed':
        H=float(rng.uniform(20.,80.));b=float(rng.uniform(0,min(2.5*R,H)))
        p=np.r_[p,b/min(R,H),rng.uniform(np.log(.5),np.log(1.8))]
        p[[4,8]]-=np.log(max(abs(2*nd(metal,energy_ev=energy_ev))**2,1e-18))
        d.update(shape='hemisphere',polymer_thickness_nm=H,polymer_type=str(rng.choice(['PS','PSBMA'])))
        truth.update(height_nm=R,height_scope='hemisphere_mark_radius',burial_base_depth_nm=b,polymer_thickness_nm=H,
            exposed_height_at_number_mean_nm=max(R-b,0.))
    elif module=='TS_layer':
        ts=dict(domain_nm=D,width_rel=float(rng.uniform(.08,1.3)),fraction=float(rng.uniform(.05,.95)),
            thickness_nm=float(np.exp(rng.uniform(np.log(.4),np.log(60.)))),depth_corr_nm=float(np.exp(rng.uniform(np.log(.3),np.log(50.)))))
        rms=float(rng.uniform(.05,3.));p=np.array([np.log(D),ts['width_rel'],0.,np.log(ts['thickness_nm']),np.log(ts['depth_corr_nm']),rms,0.,-30.,-30.,1.5,0.,-30.,-30.])
        d.update(TS=ts,interface_rms_nm=rms)
        truth=dict(spectral_domain_nm=D,film_thickness_nm=ts['thickness_nm'],buried_interface_rms_nm=rms,
            metal_fraction=ts['fraction'],depth_corr_nm=ts['depth_corr_nm'],scope='effective_TS_spectrum_not_proven_connectivity')
    if free_height and module in ('island','Si_bimodal','highq_self'):
        height=float(R*np.exp(rng.uniform(np.log(.25),np.log(2.))))
        d['height_nm']=height;truth['height_nm']=height
        truth['height_scope']='independent cylinder common height' if shape=='cylinder' else 'independent mean half-ellipsoid height, common h/R across marks'
        if module=='Si_bimodal':
            truth['small_height_scope']='fixed cylinder1nm or small hemisphere height=Rs; not shared with main-population free height'
    d['p']=p.tolist();d['truth']=truth
    return d
