"""Named joint coordinates and physical decoding for the new NR modules.

Independent of frozen v2 weights/schema. Inactive coordinates never become
fake training labels. Invalid posterior draws retain their mass and reason;
they are not silently clipped or redrawn. An optimizer may use its own
declared feasible bounds after these proposals have been recorded.
"""
from __future__ import annotations
import copy
import numpy as np

NAMES=('D_nm','R_number_mean_nm','radius_sigma_log','spacing_disorder',
       'height_nm','R_small_number_mean_nm','secondary_number_ratio',
       'burial_base_depth_nm','polymer_thickness_nm','polymer_density_factor',
       'spectral_domain_nm','width_rel','metal_fraction','film_thickness_nm',
       'depth_corr_nm','buried_interface_rms_nm','critical_angle_deg')
INDEX={k:i for i,k in enumerate(NAMES)}
LOG1P={'burial_base_depth_nm'}
LOGIT={'metal_fraction'}


def encode_truth(d):
    t=d['truth'];p=np.asarray(d['p']);m=d['module'];values={}
    if m!='TS_layer':
        values.update(R_number_mean_nm=t['R_number_mean_nm'],radius_sigma_log=t['radius_sigma_log'])
        if m!='highq_self':values.update(D_nm=t['D_nm'],spacing_disorder=p[3])
        if 'height_nm' in d:values['height_nm']=d['height_nm']
        if m=='Si_bimodal':values.update(R_small_number_mean_nm=t['R_small_number_mean_nm'],secondary_number_ratio=t['secondary_number_ratio'])
        if m=='polymer_embed':values.update(burial_base_depth_nm=t['burial_base_depth_nm'],polymer_thickness_nm=d['polymer_thickness_nm'],polymer_density_factor=np.exp(p[12]))
        else:values['critical_angle_deg']=d['critical_angle_deg']
    else:
        values.update(spectral_domain_nm=d['TS']['domain_nm'],width_rel=d['TS']['width_rel'],metal_fraction=d['TS']['fraction'],
            film_thickness_nm=d['TS']['thickness_nm'],depth_corr_nm=d['TS']['depth_corr_nm'],buried_interface_rms_nm=d['interface_rms_nm'])
    z=np.zeros(len(NAMES));active=np.zeros(len(NAMES),bool)
    for name,value in values.items():
        i=INDEX[name];active[i]=True
        if name in LOG1P:z[i]=np.log1p(value)
        elif name in LOGIT:z[i]=np.log(value/(1-value))
        else:z[i]=np.log(value)
    if not np.isfinite(z).all():raise ValueError('Nonfinite original physical truth coordinate')
    return z,active


def decode(prototype,z,active):
    """Update BOTH p and module physics dictionaries, preserving the branch.

    Prototype supplies declared nuisance initial values, not posterior truth.
    Missing shape-conditioned height is derived again after changing R.
    """
    z=np.asarray(z,float);active=np.asarray(active,bool)
    if z.shape!=(len(NAMES),) or active.shape!=z.shape:raise ValueError('NR named-coordinate vector required')
    d=copy.deepcopy(prototype);p=np.asarray(d['p'],float).copy();v={};issues=[]
    if not np.isfinite(z[active]).all():return dict(valid=False,reasons=['nonfinite_active_chart'],parameters=d)
    with np.errstate(over='ignore',invalid='ignore'):
        for name in np.array(NAMES)[active]:
            a=z[INDEX[name]]
            v[name]=float(np.expm1(a) if name in LOG1P else 1/(1+np.exp(-a)) if name in LOGIT else np.exp(a))
    if any(not np.isfinite(x) or x<0 or x==0 and k not in LOG1P or k in LOGIT and x>=1 for k,x in v.items()):
        return dict(valid=False,reasons=['out_of_positive_physical_chart'],parameters=d)
    m=d['module']
    old_burial=p[11]*min(np.exp(p[1]),d['polymer_thickness_nm']) if m=='polymer_embed' else None
    if m=='TS_layer':
        mapping={'spectral_domain_nm':'domain_nm','width_rel':'width_rel','metal_fraction':'fraction','film_thickness_nm':'thickness_nm','depth_corr_nm':'depth_corr_nm'}
        for name,key in mapping.items():
            if name in v:d['TS'][key]=v[name]
        if 'buried_interface_rms_nm' in v:d['interface_rms_nm']=v['buried_interface_rms_nm']
        ts=d['TS'];p[:6]=[np.log(ts['domain_nm']),ts['width_rel'],0.,np.log(ts['thickness_nm']),np.log(ts['depth_corr_nm']),d['interface_rms_nm']]
        from .scenario_ts import TSParams
        try:TSParams(**ts).validate()
        except ValueError as exc:issues.append(str(exc))
    else:
        for name,i,logged in [('D_nm',0,True),('R_number_mean_nm',1,True),('radius_sigma_log',2,False),('spacing_disorder',3,False)]:
            if name in v:p[i]=np.log(v[name]) if logged else v[name]
        R=np.exp(p[1])
        if 'height_nm' in v:d['height_nm']=v['height_nm']
        if 'critical_angle_deg' in v:d['critical_angle_deg']=v['critical_angle_deg']
        if m=='Si_bimodal':
            for name,i in [('R_small_number_mean_nm',11),('secondary_number_ratio',12)]:
                if name in v:p[i]=np.log(v[name])
        if m=='polymer_embed':
            if 'polymer_thickness_nm' in v:d['polymer_thickness_nm']=v['polymer_thickness_nm']
            H=d['polymer_thickness_nm'];b=v.get('burial_base_depth_nm',old_burial)
            p[11]=b/min(R,H)
            if 'polymer_density_factor' in v:p[12]=np.log(v['polymer_density_factor'])
            if b>H:issues.append('burial_crosses_unimplemented_bottom_interface')
            max_r=R*np.exp(8*p[2]-.5*p[2]**2)
            if max_r-b>=600.:issues.append('outside_implemented_optical_z_domain')
    d['p']=p.tolist()
    if not np.isfinite(p).all():issues.append('nonfinite_decoded_physical_parameters')
    # This is a proposal, not new ground truth or calibrated posterior label.
    d.pop('truth',None)
    return dict(valid=not issues,reasons=issues,parameters=d,chart_values=v)


def branch(d):
    # TS ignores the metadata shape/ap; uncorrelated self scattering is P in
    # both DA and LMA. Do not teach a classifier arbitrary invisible labels.
    if d['module']=='TS_layer':return ('TS_layer','network','none','none','film_thickness')
    ap='self' if d['module']=='highq_self' else d['approximation']
    return (d['module'],d['shape'],ap,d.get('polymer_type','none'),'independent_height' if 'height_nm' in d else 'conditional_height')
