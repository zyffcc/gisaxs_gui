'Explicit51-coordinate exploratory mainline chart; old44D stays unchanged.'
from dataclasses import asdict
import itertools
import math

import numpy as np

from . import scenario_chart as old
from .effective_host import HostOptics
from .scenario_coverage_variants import _WideSpreadParams
from .scenario_steps import StepProfile
from .scenario_template import TemplateGrowth
from .scenario_ts import TSParams
from .v2_scene import _encode, _decode, _keys, radius_upper

SCHEMA = 'exploratory_mainline_four_scenario_joint_chart_1'
SCENARIO_NAMES = old.SCENARIO_NAMES
MODIFIERS = ('a', 'b', 'c', 'd')
COMPOSITION_ORDER = ('d', 'c', 'a', 'b')
EXTRA_SPEC = (
    ('small_base_radius_nm', .5, 1.5, True),
    ('cluster_sigma_log', .03, .20, False),
    ('cluster_aspect', .3, 2., False),
    ('remaining_occupation_fraction', .01, .9, False),
    ('lowq_corner_nm_inv', .015, .5, True),
    ('lowq_gain', .001, 10., True),
    ('lowq_exponent', .5, 6., False))
SPEC = old.SPEC + EXTRA_SPEC
CORE = old.CORE
COORDINATES = CORE + tuple(s[0] for s in SPEC) + ('log_target_scale',)
DIM = len(COORDINATES)
INDEX = {k:i for i,k in enumerate(COORDINATES)}
_catalog = []
for _base in old.CATALOG:
    _legal = ['b']
    if _base['scenario'] in ('S1', 'S2'):
        _legal += ['a', 'd']
    if _base['layer']:
        _legal += ['c']
    for _n in range(len(_legal)+1):
        for _selected in itertools.combinations(sorted(_legal), _n):
            _backgrounds = itertools.product(('lorentzian','soft_power'),('host','instrument')) if 'b' in _selected else [(None,None)]
            for _kind,_origin in _backgrounds:
                _catalog.append(dict(_base,id=len(_catalog),base_coarse=_base['id'],
                    modifiers=list(_selected),combination='+'.join(_selected) or 'baseline',
                    lowq_kind=_kind,lowq_origin=_origin))
CATALOG = tuple(_catalog)
del _catalog, _base, _legal, _n, _selected, _backgrounds, _kind, _origin


def catalog_entry(coarse):
    if (isinstance(coarse,(bool,np.bool_)) or not isinstance(coarse,(int,np.integer))
            or not 0 <= coarse < len(CATALOG)):
        raise ValueError('Invalid explicit mainline coarse class')
    return dict(CATALOG[int(coarse)])


def scenario_classes(scenario):
    if isinstance(scenario,(int,np.integer)) and not isinstance(scenario,(bool,np.bool_)):
        if not 0<=scenario<len(SCENARIO_NAMES):raise ValueError('Invalid mainline scenario index')
        scenario=SCENARIO_NAMES[int(scenario)]
    if scenario not in SCENARIO_NAMES:raise ValueError('Explicit S1--S4 required')
    return tuple(c['id'] for c in CATALOG if c['scenario']==scenario)


def active_coordinates(coarse,template):
    c=catalog_entry(coarse);base=old.active_coordinates(c['base_coarse'],template)
    active=np.r_[base[:-1],np.zeros(len(EXTRA_SPEC),bool),base[-1]]
    for name,*_ in EXTRA_SPEC[:4]:active[INDEX[name]]='a' in c['modifiers']
    for name,*_ in EXTRA_SPEC[4:]:active[INDEX[name]]='b' in c['modifiers']
    if c['lowq_kind']=='lorentzian':active[INDEX['lowq_exponent']]=False
    return active


def template_compatibility(E,ai,templates,*,host_category=None):
    base=old.template_compatibility(E,ai,templates,host_category=host_category)
    return np.asarray([base[c['base_coarse']] for c in CATALOG],bool)


def _core_bounds(index,c):
    if index==3 and 'd' in c['modifiers']:return (.03,.65,False)
    if index==4 and 'd' in c['modifiers']:return (.03,1.,False)
    return old._CORE_BOUNDS[index]


def _bounds(name,c,values):
    bounds={k:(lo,hi,log) for k,lo,hi,log in SPEC}[name]
    if 'c' in c['modifiers']:
        if name=='delta12':return (1e-9,3e-5,True)
        if name=='beta12':return (1e-11,3e-6,True)
        if name=='thickness':return (.2,500.,True)
        if name in ('roughness_nm','buried_roughness_nm'):
            return (.01,min(10.,.25*values['thickness']),True)
    if 'b' in c['modifiers']:
        if name=='psf_af':return (.03,3.,True)
        if name=='psf_q':return (.03,4.,True)
    return bounds


def _entry(row):
    if 'mainline_coarse' not in row:raise ValueError('Explicit mainline coarse label required')
    c=catalog_entry(row['mainline_coarse'])
    if (row['scenario']!=c['scenario'] or (row['thickness']>0)!=c['layer']
            or list(row.get('modifiers',[]))!=c['modifiers']):
        raise ValueError('Physical mainline row differs from its discrete declaration')
    if row['scenario']!='S4':
        p=row['particle']
        if (p['shape']!=c['shape'] or row['structure_factor']!=c['structure_factor']
                or row['decoupling']!=c['decoupling']):
            raise ValueError('Mainline particle law differs from its coarse label')
        if c['decoupling']=='DA' and p['kappa']!=0:raise ValueError('DA kappa must remain exactly zero')
    if 'b' in c['modifiers'] and (row['lowq']['lowq_kind']!=c['lowq_kind'] or row['lowq']['lowq_origin']!=c['lowq_origin']):
        raise ValueError('Low-q law differs from its actual discrete label')
    return c


def encode(row,template_keys=()):
    keys=_keys(template_keys);c=_entry(row);key=row.get('host_template_key')
    if row.get('transition_nm',0.)!=0 or row.get('burial',0.)!=0:
        raise ValueError('This mainline chart has no unencoded gradient/burial coordinate')
    if not c['layer'] and any(row[k]!=0 for k in ('delta12','beta12','thickness')):
        raise ValueError('Absent finite layer must have zero optical constants/thickness')
    if key is not None and key not in keys:raise ValueError('Unregistered TRAIN host template')
    template=keys.index(key)+1 if key is not None else 0
    active=active_coordinates(c['id'],template);z=np.zeros(DIM);values=dict(row)
    HostOptics(complex(row['delta12'],row['beta12']),row['thickness'],
        roughness_nm=row['roughness_nm'],buried_roughness_nm=row['buried_roughness_nm']).validate()
    if row['scenario']!='S4':
        p=_WideSpreadParams(**row['particle']);p.state();aspect=p.height_nm/p.base_radius_nm
        if p.film_thickness_nm!=0 or p.average_material:
            raise ValueError('Mainline mean profile replaces additive primitive-film modes')
        upper=8. if c['scenario']=='S3' else radius_upper(p.D_nm,p.sigma_log,aspect,p.shape)
        core=(p.D_nm,p.base_radius_nm,aspect,p.sigma_log,p.omega_rel,p.kappa)
        for i,value in enumerate(core):
            if active[i]:z[i]=_encode(value,(.4,upper,True) if i==1 else _core_bounds(i,c))
    if c['scenario']=='S2':
        growth=TemplateGrowth(**row['growth']);growth.gain(float(row['loading']))
        values.update(growth_base_gain=growth.base_gain,growth_gain=growth.growth_gain,
            growth_loading_scale=growth.loading_scale,growth_power=growth.power)
    if c['scenario']=='S3':
        profile=StepProfile(**row['steps']);edges,fractions=profile.arrays(row['thickness'])
        if len(fractions)!=c['step_count'] or np.any(fractions>profile.coverage):
            raise ValueError('Mainline step profile differs from the original occupation law')
        values.update(step_burial_fraction=-edges[0]/row['thickness'],step_exposed_nm=edges[-1],
            step_coverage=profile.coverage,step_split_fraction=edges[1]/edges[0] if len(fractions)==3 else 0.)
        values.update({'step_fill_'+str(i):v/profile.coverage for i,v in enumerate(fractions)})
    if c['scenario']=='S4':
        ts=TSParams(**row['ts']);ts.validate();values.update({'ts_'+k:v for k,v in asdict(ts).items()})
    values.update(row.get('small_clusters',{}));values.update(row.get('lowq',{}))
    for i,(name,*_) in enumerate(SPEC,start=len(CORE)):
        if active[i]:z[i]=_encode(values[name],_bounds(name,c,values))
    if not math.isfinite(row['target_scale']) or row['target_scale']<=0:raise ValueError('Positive brightness required')
    z[-1]=math.log(row['target_scale'])
    return dict(z=z,active=active,coarse=c['id'],template=template,scenario=c['scenario_index'],
        scenario_id=c['scenario'],schema=SCHEMA,labels_changed=True,structural_release=False)


def decode(z,coarse,template,*,metal,E,ai,template_keys=(),template_metadata=None):
    c=catalog_entry(coarse);keys=_keys(template_keys)
    if (isinstance(template,(bool,np.bool_)) or not isinstance(template,(int,np.integer))
            or not 0<=template<=len(keys)):
        raise ValueError('Invalid mainline template class')
    if metal not in ('Au','Ag') or not np.isfinite([E,ai]).all() or min(E,ai)<=0:
        raise ValueError('Positive Au/Ag geometry required')
    if template and template_metadata is not None:
        if tuple(t['key'] for t in template_metadata)!=keys or not template_compatibility(E,ai,template_metadata)[coarse,template]:
            raise ValueError('Original TRAIN template geometry/identity differs')
    active=active_coordinates(coarse,template);z=np.asarray(z,float)
    if z.shape!=(DIM,) or not np.isfinite(z[active]).all():raise ValueError('Finite complete51D joint required')
    values={}
    for i,(name,*_) in enumerate(SPEC,start=len(CORE)):
        values[name]=_decode(z[i],_bounds(name,c,values)) if active[i] else 0.
    try:brightness=math.exp(float(z[-1]))
    except OverflowError:raise ValueError('Brightness overflow; no redraw')
    if not math.isfinite(brightness) or brightness<=0:raise ValueError('Positive finite brightness required')
    row={name:values[name] for name,*_ in old.COMMON_SPEC}
    row.update(scenario=c['scenario'],scenario_index=c['scenario_index'],mainline_coarse=int(coarse),
        coarse=int(coarse),base_coarse=c['base_coarse'],template=int(template),modifiers=c['modifiers'],
        combination=c['combination'],E=float(E),ai=float(ai),metal=metal,
        host_template_key=keys[int(template)-1] if template else None,
        transition_nm=0.,burial=0.,target_scale=brightness,structure_factor=c['structure_factor'],
        decoupling=c['decoupling'],eta=values['eta'],particle=None,schema=SCHEMA,
        conditioned_outside_D=False,nuisance_prior_resampled=False,structural_release=False)
    if c['scenario']!='S4':
        D=_decode(z[0],_core_bounds(0,c));aspect=_decode(z[2],_core_bounds(2,c));sigma=_decode(z[3],_core_bounds(3,c))
        upper=8. if c['scenario']=='S3' else radius_upper(D,sigma,aspect,c['shape'])
        radius=_decode(z[1],(.4,upper,True))
        p=_WideSpreadParams(D_nm=D,base_radius_nm=radius,height_nm=radius*aspect,sigma_log=sigma,
            omega_rel=_decode(z[4],_core_bounds(4,c)) if active[4] else .2,
            kappa=_decode(z[5],_core_bounds(5,c)) if active[5] else 0.,shape=c['shape'])
        p.state();row.update(particle=asdict(p),D_nm=D,diameter_nm=p.diameter(),
            height_nm=p.height_nm if c['scenario']!='S3' else None,gap_nm=None,
            D_definition='conditional centre-density length in S1/S2; correlation step in S3; not SEM nearest neighbour',
            diameter_definition='main population median maximum footprint diameter')
    if c['scenario']=='S2':
        row.update(growth=dict(base_gain=values['growth_base_gain'],growth_gain=values['growth_gain'],
            loading_scale=values['growth_loading_scale'],power=values['growth_power']),loading=values['loading'],
            host_peak_not_metal_D=True,loading_definition='conditional decoration proxy, not QCM')
    if c['scenario']=='S3':
        b=values['step_burial_fraction']*row['thickness'];h=values['step_exposed_nm'];coverage=values['step_coverage']
        edges=(-b,0.,h) if c['step_count']==2 else (-b,-b*values['step_split_fraction'],0.,h)
        profile=StepProfile(edges,tuple(values['step_fill_'+str(i)]*coverage for i in range(c['step_count'])),coverage)
        profile.arrays(row['thickness']);row.update(steps=asdict(profile),depth_interval_nm=[-b,h],depth_span_nm=b+h,
            exposed_span_nm=h,burial_span_nm=b,coverage_fraction=coverage,
            diameter_definition='median effective column maximum footprint diameter',
            height_missing_reason='step depth support is not island height',gap_missing_reason='no validated centre configuration')
    if c['scenario']=='S4':
        ts=TSParams(**{k:values['ts_'+k] for k in ('domain_nm','width_rel','fraction','thickness_nm','depth_corr_nm')})
        ts.validate();row.update(ts=asdict(ts),D_nm=None,diameter_nm=None,height_nm=None,gap_nm=None,
            domain_nm=ts.domain_nm,width_rel_intrinsic=ts.width_rel,coverage_fraction=ts.fraction,
            island_quantity_missing_reason='not applicable to effective two-phase S4',
            merged_phase_assumed=True,connected_topology_verified=False)
    if 'a' in c['modifiers']:row['small_clusters']={name:values[name] for name,*_ in EXTRA_SPEC[:4]}
    if 'b' in c['modifiers']:
        row['lowq']={name:values[name] for name,*_ in EXTRA_SPEC[4:]}
        row['lowq'].update(lowq_kind=c['lowq_kind'],lowq_origin=c['lowq_origin'])
        if c['lowq_kind']=='lorentzian':row['lowq']['lowq_exponent']=2.
    HostOptics(complex(row['delta12'],row['beta12']),row['thickness'],
        roughness_nm=row['roughness_nm'],buried_roughness_nm=row['buried_roughness_nm']).validate()
    return row
