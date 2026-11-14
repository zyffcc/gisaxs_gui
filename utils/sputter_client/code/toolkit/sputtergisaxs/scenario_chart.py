'Joint coordinate chart for the simplified physical S1--S4 scenarios.'
from dataclasses import asdict
import math

import numpy as np

from .effective_host import HostOptics
from .forward_v06 import SSCAParams
from .scenario_steps import StepProfile
from .scenario_template import TemplateGrowth
from .scenario_ts import TSParams
from .v2_scene import _encode, _decode, _keys, radius_upper


SCHEMA = 'simplified_four_scenario_joint_chart_1'
SCENARIO_NAMES = ('S1', 'S2', 'S3', 'S4')
_VARIANTS = (('paracrystal', 'DA'), ('paracrystal', 'SSCA'),
             ('harddisk', 'DA'))
_catalog = []
for _s in ('S1', 'S2', 'S3'):
    for _shape in ('cap', 'hemiellipsoid'):
        for _sf, _da in (_VARIANTS[:2] if _s == 'S3' else _VARIANTS):
            for _layout in ((2, 3) if _s == 'S3' else (False, True)):
                _catalog.append(dict(id=len(_catalog), scenario=_s,
                    scenario_index=SCENARIO_NAMES.index(_s), shape=_shape,
                    structure_factor=_sf, decoupling=_da,
                    layer=True if _s == 'S3' else bool(_layout),
                    step_count=int(_layout) if _s == 'S3' else 0,
                    step_layout=('two_buried_one_surface' if _layout == 3
                                 else 'one_buried_one_surface') if _s == 'S3' else None))
for _layer in (False, True):
    _catalog.append(dict(id=len(_catalog), scenario='S4', scenario_index=3,
        shape=None, structure_factor='teubner_strey', decoupling=None,
        layer=_layer, step_count=0, step_layout=None))
CATALOG = tuple(_catalog)
SCENES = CATALOG
del _catalog, _s, _shape, _sf, _da, _layout, _layer

CORE = ('D_nm', 'radius_coordinate', 'aspect', 'sigma_log', 'omega_rel', 'kappa')
COMMON_SPEC = (
    ('delta12', 1e-8, 1e-5, True), ('beta12', 1e-10, 8e-7, True),
    ('thickness', .5, 300., True),
    ('roughness_nm', .001, 5., True), ('buried_roughness_nm', .001, 5., True),
    ('host_fraction', .001, 10., True),
    ('db_xi', .5, 40., True), ('db_fraction', .001, 3., True),
    ('power_exponent', 1., 4., False), ('power_fraction', .001, 3., True),
    ('flat_fraction', .0001, .3, True), ('ring_fraction', .00001, .5, True),
    ('gamma', -1., 1., False), ('transmitted_fraction', 0., 1., False),
    ('scatter_depth', .5, 30., True), ('psf_af', .03, 1., True),
    ('psf_q', .03, 1., True), ('qscale', .97, 1.03, False),
    ('af_shift', -.015, .015, False))
EXTRA_SPEC = (
    ('eta', .03, .65, False),
    ('growth_base_gain', .001, 10., True),
    ('growth_gain', .001, 30., True),
    ('growth_loading_scale', .001, 100., True),
    ('growth_power', .25, 3., False), ('loading', .001, 100., True),
    ('step_burial_fraction', 0., 1., False),
    ('step_exposed_nm', .1, 60., True),
    ('step_split_fraction', 0., 1., False),
    ('step_coverage', .01, .99, False),
    ('step_fill_0', .001, .999, False),
    ('step_fill_1', .001, .999, False),
    ('step_fill_2', .001, .999, False),
    ('ts_domain_nm', 3., 200., True),
    ('ts_width_rel', .03, 1.35, True),
    ('ts_fraction', .01, .99, False),
    ('ts_thickness_nm', .2, 60., True),
    ('ts_depth_corr_nm', .2, 120., True))
SPEC = COMMON_SPEC + EXTRA_SPEC
COORDINATES = CORE + tuple(s[0] for s in SPEC) + ('log_target_scale',)
DIM = len(COORDINATES)
INDEX = {name: i for i, name in enumerate(COORDINATES)}
_CORE_BOUNDS = ((3.6, 80., True), None, (.1, 2., False),
                (.03, .45, False), (.06, .6, False), (0., 1., False))


def catalog_entry(coarse):
    if (isinstance(coarse, (bool, np.bool_)) or not isinstance(coarse, (int, np.integer))
            or not 0 <= coarse < len(CATALOG)):
        raise ValueError('Invalid physical-scenario coarse class')
    return dict(CATALOG[int(coarse)])


def scenario_classes(scenario):
    'Coarse IDs for directly conditioned scenario sampling; no rejection draw.'
    if isinstance(scenario, (int, np.integer)) and not isinstance(scenario, (bool, np.bool_)):
        if not 0 <= scenario < len(SCENARIO_NAMES):
            raise ValueError('Invalid scenario index')
        scenario = SCENARIO_NAMES[int(scenario)]
    if scenario not in SCENARIO_NAMES:
        raise ValueError('Scenario must be S1--S4')
    return tuple(c['id'] for c in CATALOG if c['scenario'] == scenario)


def active_coordinates(coarse, template):
    c = catalog_entry(coarse)
    if (isinstance(template, (bool, np.bool_)) or not isinstance(template, (int, np.integer))
            or template < 0):
        raise ValueError('Nonnegative template class required')
    active = np.zeros(DIM, bool)
    for name, *_ in COMMON_SPEC:
        active[INDEX[name]] = True
    for name in ('delta12', 'beta12', 'thickness'):
        active[INDEX[name]] = c['layer']
    active[INDEX['host_fraction']] = template > 0
    active[-1] = True
    if c['scenario'] != 'S4':
        active[:4] = True
        active[INDEX['omega_rel']] = c['structure_factor'] == 'paracrystal'
        active[INDEX['kappa']] = c['decoupling'] == 'SSCA'
        active[INDEX['eta']] = c['structure_factor'] == 'harddisk'
    if c['scenario'] == 'S2':
        for name in ('growth_base_gain', 'growth_gain', 'growth_loading_scale',
                     'growth_power', 'loading'):
            active[INDEX[name]] = True
    if c['scenario'] == 'S3':
        for name in ('step_burial_fraction', 'step_exposed_nm', 'step_coverage',
                     'step_fill_0', 'step_fill_1'):
            active[INDEX[name]] = True
        active[INDEX['step_split_fraction']] = c['step_count'] == 3
        active[INDEX['step_fill_2']] = c['step_count'] == 3
    if c['scenario'] == 'S4':
        for name in ('ts_domain_nm', 'ts_width_rel', 'ts_fraction',
                     'ts_thickness_nm', 'ts_depth_corr_nm'):
            active[INDEX[name]] = True
    return active


def template_compatibility(E, ai, templates, *, host_category=None):
    'TRAIN-only same-geometry/category templates and physical applicability.'
    _keys(t['key'] for t in templates)
    if host_category not in (None, 'unknown', 'Si', 'oxide_particles', 'polymer'):
        raise ValueError('Unknown optional host category')
    if not np.isfinite([E, ai]).all() or min(E, ai) <= 0:
        raise ValueError('Positive energy and incidence angle required')
    legal = np.zeros((len(CATALOG), len(templates)+1), bool)
    compatible = np.array([True] + [
        t.get('split') == 'TRAIN' and E == 12000.
        and abs(float(t['alpha_i_deg'])-ai) < 1e-12
        and (host_category in (None, 'unknown') or t['host_category'] == host_category)
        for t in templates], bool)
    for c in CATALOG:
        if host_category == 'Si' and (c['scenario'] in ('S2', 'S3') or c['layer']):
            continue
        legal[c['id']] = compatible
    return legal


def _variant(row):
    scenario = row.get('scenario')
    if scenario not in SCENARIO_NAMES:
        raise ValueError('Explicit S1--S4 physical scenario required')
    layer = float(row['thickness']) > 0
    if scenario == 'S4':
        query = dict(scenario=scenario, layer=layer)
    else:
        p = row['particle'] if isinstance(row['particle'], SSCAParams) else SSCAParams(**row['particle'])
        p.state()
        sf = row.get('structure_factor', 'paracrystal')
        da = row.get('decoupling', 'SSCA')
        if (sf, da) not in _VARIANTS:
            raise ValueError('Harddisk currently implements DA only; no duplicate HD-SSCA class')
        if da == 'DA' and p.kappa != 0:
            raise ValueError('DA has exactly kappa=0; do not relabel a coupled draw')
        query = dict(scenario=scenario, layer=layer, shape=p.shape,
                     structure_factor=sf, decoupling=da)
        if scenario == 'S3':
            steps = row['steps']
            steps = steps if isinstance(steps, StepProfile) else StepProfile(**steps)
            edges, fractions = steps.arrays(row['thickness'])
            if len(fractions) == 3 and (edges[2] != 0 or edges[1] >= 0):
                raise ValueError('Three-step chart requires two buried and one surface segment')
            query['step_count'] = len(fractions)
    found = [c for c in CATALOG if all(c[k] == v for k, v in query.items())]
    if len(found) != 1:
        raise ValueError('Forward row has no unique declared coarse variant')
    return found[0]


def encode(row, template_keys=()):
    'Encode the original complete joint row without changing its active truth.'
    keys = _keys(template_keys); c = _variant(row)
    if row.get('transition_nm', 0.) != 0:
        raise ValueError('Simplified chart uses a sharp host; no gradient relabeling')
    HostOptics(complex(row['delta12'], row['beta12']), row['thickness'],
               roughness_nm=row.get('roughness_nm', .01),
               buried_roughness_nm=row.get('buried_roughness_nm', .01)).validate()
    if not c['layer'] and any(row[k] != 0 for k in ('delta12', 'beta12', 'thickness')):
        raise ValueError('Absent finite layer must have zero host constants/thickness')
    key = row.get('host_template_key')
    if key is not None and key not in keys:
        raise ValueError('Unregistered TRAIN template')
    template = keys.index(key)+1 if key is not None else 0
    active = active_coordinates(c['id'], template); z = np.zeros(DIM)
    values = dict(row)
    values.setdefault('roughness_nm', .01); values.setdefault('buried_roughness_nm', .01)
    if c['scenario'] != 'S4':
        p = row['particle'] if isinstance(row['particle'], SSCAParams) else SSCAParams(**row['particle'])
        if p.film_thickness_nm != 0 or p.average_material:
            raise ValueError('Use scenario profile, not additive island-film modes')
        aspect = p.height_nm/p.base_radius_nm
        upper = 8. if c['scenario'] == 'S3' else radius_upper(p.D_nm, p.sigma_log, aspect, p.shape)
        core = (p.D_nm, p.base_radius_nm, aspect, p.sigma_log, p.omega_rel, p.kappa)
        for i, value in enumerate(core):
            if active[i]:
                z[i] = _encode(value, (.4, upper, True) if i == 1 else _CORE_BOUNDS[i])
    if c['scenario'] == 'S2':
        growth = row['growth']; growth = growth if isinstance(growth, TemplateGrowth) else TemplateGrowth(**growth)
        growth.gain(float(row['loading']))
        values.update(growth_base_gain=growth.base_gain, growth_gain=growth.growth_gain,
                      growth_loading_scale=growth.loading_scale, growth_power=growth.power)
    if c['scenario'] == 'S3':
        steps = row['steps']; steps = steps if isinstance(steps, StepProfile) else StepProfile(**steps)
        edges, fractions = steps.arrays(row['thickness'])
        if np.any(fractions > steps.coverage):
            raise ValueError('Metal step fraction cannot exceed column footprint occupation')
        values.update(step_burial_fraction=-edges[0]/row['thickness'],
                      step_exposed_nm=edges[-1], step_coverage=steps.coverage,
                      step_split_fraction=edges[1]/edges[0] if len(fractions) == 3 else 0.)
        for i, value in enumerate(fractions):
            values['step_fill_'+str(i)] = value/steps.coverage
    if c['scenario'] == 'S4':
        ts = row['ts']; ts = ts if isinstance(ts, TSParams) else TSParams(**ts)
        ts.validate()
        values.update({'ts_'+k: v for k, v in asdict(ts).items()})
    for i, (name, lo, hi, log) in enumerate(SPEC, start=len(CORE)):
        if active[i]:
            if name not in values:
                raise ValueError('Missing active joint nuisance '+name)
            z[i] = _encode(values[name], (lo, hi, log))
    brightness = float(row['target_scale'])
    if not math.isfinite(brightness) or brightness <= 0:
        raise ValueError('Positive original brightness required')
    z[-1] = math.log(brightness)
    return dict(z=z, active=active, coarse=c['id'], template=template,
                scenario=c['scenario_index'], scenario_id=c['scenario'], schema=SCHEMA,
                labels_changed=False, structural_release=False)


def decode(z, coarse, template, *, metal, E, ai, template_keys=(), template_metadata=None):
    'Replay one complete original joint draw; failures are never substituted.'
    keys = _keys(template_keys); c = catalog_entry(coarse)
    if (isinstance(template, (bool, np.bool_)) or not isinstance(template, (int, np.integer))
            or not 0 <= template <= len(keys)):
        raise ValueError('Invalid template class')
    if (metal not in ('Au', 'Ag') or not np.isfinite([E, ai]).all() or min(E, ai) <= 0):
        raise ValueError('Positive mandatory Au/Ag geometry required')
    if template and template_metadata is not None:
        if tuple(t['key'] for t in template_metadata) != keys or not template_compatibility(E, ai, template_metadata)[coarse, template]:
            raise ValueError('Original template geometry/TRAIN identity differs')
    active = active_coordinates(coarse, template); z = np.asarray(z, float)
    if z.shape != (DIM,) or not np.isfinite(z[active]).all():
        raise ValueError('Finite complete active joint coordinates required')
    values = {name: _decode(z[i], (lo, hi, log)) if active[i] else 0.
              for i, (name, lo, hi, log) in enumerate(SPEC, start=len(CORE))}
    try:
        brightness = math.exp(z[-1])
    except OverflowError:
        raise ValueError('Original brightness overflow; no redraw')
    if not math.isfinite(brightness) or brightness <= 0:
        raise ValueError('Original brightness under/overflow; no redraw')
    row = {name: values[name] for name, *_ in COMMON_SPEC}
    row.update(scenario=c['scenario'], scenario_index=c['scenario_index'],
        coarse=int(coarse), template=int(template), E=float(E), ai=float(ai), metal=metal,
        host_template_key=keys[int(template)-1] if template else None,
        transition_nm=0., burial=0., target_scale=brightness,
        structure_factor=c['structure_factor'], decoupling=c['decoupling'],
        eta=values['eta'], particle=None, schema=SCHEMA, structural_release=False,
        conditioned_outside_D=False, nuisance_prior_resampled=False)
    if c['scenario'] != 'S4':
        D = _decode(z[0], _CORE_BOUNDS[0]); aspect = _decode(z[2], _CORE_BOUNDS[2])
        sigma = _decode(z[3], _CORE_BOUNDS[3])
        upper = 8. if c['scenario'] == 'S3' else radius_upper(D, sigma, aspect, c['shape'])
        radius = _decode(z[1], (.4, upper, True))
        p = SSCAParams(D_nm=D, base_radius_nm=radius, height_nm=aspect*radius,
                      sigma_log=sigma, omega_rel=_decode(z[4], _CORE_BOUNDS[4]) if active[4] else .2,
                      kappa=_decode(z[5], _CORE_BOUNDS[5]) if active[5] else 0., shape=c['shape'])
        p.state(); row['particle'] = asdict(p)
        row['D_nm'] = D; row['diameter_nm'] = p.diameter()
        row['height_nm'] = p.height_nm if c['scenario'] != 'S3' else None
        row['gap_nm'] = None
        row['D_definition'] = 'conditional lateral correlation step, not certified 2D nearest neighbour'
        row['diameter_definition'] = 'median maximum lateral primitive diameter'
    if c['scenario'] == 'S2':
        row['growth'] = dict(base_gain=values['growth_base_gain'], growth_gain=values['growth_gain'],
                             loading_scale=values['growth_loading_scale'], power=values['growth_power'])
        row['loading'] = values['loading']; row['host_peak_not_metal_D'] = True
        row['loading_definition'] = 'free conditional decoration proxy; not QCM or total metal amount'
    if c['scenario'] == 'S3':
        b = values['step_burial_fraction']*row['thickness']; h = values['step_exposed_nm']
        edges = (-b, 0., h) if c['step_count'] == 2 else (-b, -b*values['step_split_fraction'], 0., h)
        coverage = values['step_coverage']
        fractions = tuple(values['step_fill_'+str(i)]*coverage for i in range(c['step_count']))
        profile = StepProfile(tuple(edges), fractions, coverage); profile.arrays(row['thickness'])
        row['steps'] = asdict(profile); row['depth_interval_nm'] = [-b, h]
        row['depth_span_nm'] = b+h; row['exposed_span_nm'] = h; row['burial_span_nm'] = b
        row['diameter_definition'] = 'median effective column maximum footprint diameter'
        row['height_missing_reason'] = 'step support is a depth interval, not an island height'
        row['coverage_fraction'] = coverage
        row['coverage_definition'] = 'effective mean footprint occupation; union coverage unvalidated'
        row['gap_missing_reason'] = 'no validated centre configuration'
    if c['scenario'] == 'S4':
        ts = TSParams(**{k: values['ts_'+k] for k in ('domain_nm', 'width_rel', 'fraction', 'thickness_nm', 'depth_corr_nm')})
        ts.validate(); row['ts'] = asdict(ts)
        row.update(D_nm=None, diameter_nm=None, height_nm=None, gap_nm=None,
                   island_quantity_missing_reason='not applicable to effective two-phase S4',
                   merged_phase_assumed=True, connected_topology_verified=False,
                   domain_nm=ts.domain_nm, width_rel_intrinsic=ts.width_rel,
                   coverage_fraction=ts.fraction,
                   coverage_definition='conditional planar phase-cut fraction, not measured top-surface coverage')
    HostOptics(complex(row['delta12'], row['beta12']), row['thickness'],
               roughness_nm=row['roughness_nm'], buried_roughness_nm=row['buried_roughness_nm']).validate()
    return row
