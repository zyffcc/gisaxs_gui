'Candidate joint coordinate chart, not a frozen prior or public model.'
import math
import numpy as np
from scipy.special import expit
from .forward_v06 import SSCAParams
from .effective_host import HostOptics

SCHEMA = 'effective_v2_candidate_chart_1'
SCENES = tuple(dict(shape=shape, layer=layer, buried=buried, graded=graded)
               for shape in ('cap', 'hemiellipsoid') for layer in (False, True)
               for buried in ((False, True) if layer else (False,))
               for graded in (False, True))
CORE = ('D_nm', 'radius_fraction', 'aspect', 'sigma_log', 'omega_rel', 'kappa')


SPEC = (
    ('delta12', 1e-8, 1e-5, True), ('beta12', 1e-10, 8e-7, True),
    ('thickness', .5, 300., True), ('transition_nm', .03, 30., True),
    ('burial_fraction', 0., 1., False), ('host_fraction', .001, 10., True),
    ('db_xi', .5, 40., True), ('db_fraction', .001, 3., True),
    ('power_exponent', 1., 4., False), ('power_fraction', .001, 3., True),
    ('flat_fraction', .0001, .3, True), ('ring_fraction', .00001, .5, True),
    ('gamma', -.8, .8, False), ('transmitted_fraction', 0., 1., False),
    ('scatter_depth', .5, 30., True), ('psf_af', .03, 1., True),
    ('psf_q', .03, 1., True), ('qscale', .97, 1.03, False),
    ('af_shift', -.015, .015, False))
COORDINATES = CORE + tuple(s[0] for s in SPEC) + ('log_target_scale',)
DIM = len(COORDINATES)
SHARED = ((3.6, 80., True), None, (.1, 2., False), (.03, .45, False),
          (.06, .6, False), (0., 1., False))


def _index(value, size, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 0<=value<size:
        raise ValueError('Invalid '+name)
    return int(value)


def _keys(keys):
    keys=tuple(keys)
    if any(not isinstance(k,str) or not k for k in keys) or len(set(keys))!=len(keys):
        raise ValueError('Unique nonempty TRAIN template keys required')
    return keys


def _encode(value, bounds):
    lo,hi,log=bounds;value=float(value)
    if not math.isfinite(value) or not lo<value<hi:
        raise ValueError('Label at endpoint/outside candidate open chart')
    x=math.log(value) if log else value
    a,b=(math.log(lo),math.log(hi)) if log else (lo,hi)
    u=(x-a)/(b-a)
    if not 0<u<1:
        raise ValueError('Label not representable in open chart at floating precision')
    return math.log(u)-math.log1p(-u)


def _decode(value,bounds):
    lo,hi,log=bounds;a,b=(math.log(lo),math.log(hi)) if log else (lo,hi)
    u=float(expit(value));x=a+(b-a)*u
    result=math.exp(x) if log else x
    if not 0<u<1 or not lo<result<hi:
        raise ValueError('Original joint coordinate saturated at floating precision; no substitute draw')
    return result


def template_compatibility(energy_ev,alpha_i_deg,templates,*,host_category=None):
    'Pre-sampling mask; template0 always available, unknown never implies Si.'
    _keys(t['key'] for t in templates)
    if host_category not in (None,'unknown','Si','oxide_particles','polymer'):
        raise ValueError('Unknown optional host category')
    if not np.isfinite([energy_ev,alpha_i_deg]).all() or min(energy_ev,alpha_i_deg)<=0:
        raise ValueError('Positive mandatory geometry required')
    legal=[True]
    for t in templates:
        legal.append(t.get('split')=='TRAIN' and energy_ev==12000. and
                     abs(float(t['alpha_i_deg'])-alpha_i_deg)<1e-12 and
                     (host_category in (None,'unknown') or t['host_category']==host_category))
    return np.broadcast_to(np.array(legal,bool),(len(SCENES),len(legal))).copy()


def area_factor(shape,aspect):
    return ((1+aspect*aspect)/(2*aspect))**2 if shape=='cap' and aspect>1 else 1.


def radius_upper(D,sigma,aspect,shape):
    'Sufficient analytic bound on *average* occupation at every depth.'
    g=area_factor(shape,aspect)
    return min(8.,D*math.sqrt(.98*math.sqrt(3)/(2*math.pi*math.exp(2*sigma*sigma)*g)))


def active_coordinates(coarse,template):
    c=SCENES[_index(coarse,len(SCENES),'scene')]
    if isinstance(template,(bool,np.bool_)) or not isinstance(template,(int,np.integer)) or template<0:
        raise ValueError('Nonnegative template class required')
    active=np.ones(DIM,bool)
    for name in ('delta12','beta12','thickness'):
        active[COORDINATES.index(name)]=c['layer']
    active[COORDINATES.index('transition_nm')]=c['graded']
    active[COORDINATES.index('burial_fraction')]=c['buried']
    active[COORDINATES.index('host_fraction')]=template>0
    return active


def encode(row,template_keys=()):
    'Encode an existing candidate label only when exactly representable.'
    keys=_keys(template_keys);p=SSCAParams(**row['particle']);p.state()
    if p.film_thickness_nm!=0 or p.average_material:
        raise ValueError('Effective v2 chart is islands without averaged-film mode')
    HostOptics(complex(row['delta12'],row['beta12']),row['thickness'],
               transition_nm=row.get('transition_nm',0.)).validate()
    burial=float(row.get('burial',0.))
    if not math.isfinite(burial) or not 0<=burial<=row['thickness']:
        raise ValueError('Burial must remain in the first finite host')
    c=dict(shape=p.shape,layer=row['thickness']>0,buried=row.get('burial',0.)>0,
           graded=row.get('transition_nm',0.)>0)
    if c not in SCENES:
        raise ValueError('Burial requires a finite first host layer')
    key=row.get('host_template_key')
    if key is not None and key not in keys:
        raise ValueError('Unregistered TRAIN template label')
    template=keys.index(key)+1 if key is not None else 0
    coarse=SCENES.index(c);active=active_coordinates(coarse,template)
    aspect=p.height_nm/p.base_radius_nm
    core=(p.D_nm,p.base_radius_nm,aspect,p.sigma_log,p.omega_rel,p.kappa)
    upper=radius_upper(p.D_nm,p.sigma_log,aspect,p.shape)
    z=np.zeros(DIM)
    for i,value in enumerate(core):
        z[i]=_encode(value,(.4,upper,True) if i==1 else SHARED[i])
    for i,(name,lo,hi,log) in enumerate(SPEC,start=len(CORE)):
        if not active[i]:
            continue
        value=row.get(name)
        if name=='burial_fraction':
            limit=min(row['thickness'],.9*p.height_nm)
            value=row['burial']/limit
        if value is None:
            raise ValueError('Missing active forward nuisance '+name)
        z[i]=_encode(value,(lo,hi,log))
    brightness=float(row['target_scale'])
    if not math.isfinite(brightness) or brightness<=0:
        raise ValueError('Positive original brightness required')
    z[-1]=math.log(brightness)
    return dict(z=z,active=active,coarse=coarse,template=template,schema=SCHEMA,
                labels_changed=False,structural_release=False)


def decode(z,coarse,template,*,metal,energy_ev,alpha_i_deg,template_keys=(),template_metadata=None):
    'Decode the complete original joint draw; no clipping/drop/redraw.'
    keys=_keys(template_keys);coarse=_index(coarse,len(SCENES),'scene')
    template=_index(template,len(keys)+1,'template');c=SCENES[coarse]
    if template:
        if template_metadata is None or tuple(t['key'] for t in template_metadata)!=keys:
            raise ValueError('Actual bound TRAIN template metadata required')
        if not template_compatibility(energy_ev,alpha_i_deg,template_metadata)[coarse,template]:
            raise ValueError('Original sampled template is incompatible; no substitute draw')
    active=active_coordinates(coarse,template);z=np.asarray(z,float)
    if z.shape!=(DIM,) or not np.isfinite(z[active]).all():
        raise ValueError('Complete finite active joint coordinates required')
    if (metal not in ('Au','Ag') or not np.isfinite([energy_ev,alpha_i_deg]).all() or
            energy_ev<=0 or alpha_i_deg<=0):
        raise ValueError('Mandatory Au/Ag and positive geometry required')
    D=_decode(z[0],SHARED[0]);aspect=_decode(z[2],SHARED[2]);sigma=_decode(z[3],SHARED[3])
    radius=_decode(z[1],(.4,radius_upper(D,sigma,aspect,c['shape']),True))
    p=SSCAParams(D_nm=D,base_radius_nm=radius,height_nm=radius*aspect,sigma_log=sigma,
                 omega_rel=_decode(z[4],SHARED[4]),kappa=_decode(z[5],SHARED[5]),shape=c['shape'])
    p.state();row=dict(particle=vars(p),metal=metal,E=float(energy_ev),ai=float(alpha_i_deg),
                      host_template_key=keys[template-1] if template else None)
    for i,(name,lo,hi,log) in enumerate(SPEC,start=len(CORE)):
        row[name]=_decode(z[i],(lo,hi,log)) if active[i] else 0.
    row['burial']=row.pop('burial_fraction')*min(row['thickness'],.9*p.height_nm)
    try:row['target_scale']=math.exp(z[-1])
    except OverflowError:raise ValueError('Original joint brightness overflow, no substitute draw')
    if row['target_scale']<=0:
        raise ValueError('Original joint brightness underflow, no substitute draw')
    row.update(schema=SCHEMA,coarse=coarse,template=template,structural_release=False,
               conditioned_outside_D=False,nuisance_prior_resampled=False,
               diameter_nm=p.diameter(),basal_diameter_nm=2*radius,
               mean_occupation_sufficient_bound=2*math.pi*radius*radius/(math.sqrt(3)*D*D)*math.exp(2*sigma*sigma)*area_factor(p.shape,aspect),
               numerical_certificate=False,prior_frozen=False)
    return row
