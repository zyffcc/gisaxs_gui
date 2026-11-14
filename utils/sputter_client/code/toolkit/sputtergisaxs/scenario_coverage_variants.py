'Paired coverage experiments around an original S1--S4 physical row.'
from dataclasses import replace

import numpy as np

from .effective_host import (HostOptics, build_context, metal_amplitudes,
                             metal_intensity)
from .effective_measurement import compose
from .forward_v06 import IntegratedState, SSCAParams
from .formfactor import _gauss_legendre
from .optics import wavelength_nm
from .scenario_forward import (QUICK, context as original_context,
    expected_scale, harddisk_intensity, mean_forward as baseline_forward)
from .scenario_measurement import NativeScenarioOperator, _peak_qy
from .scenario_template import TemplateGrowth, compose_template
from .scenario_ts import labels as ts_labels
from .stage5_measurement import NATIVE_AF, NATIVE_EDGES, angles
from .tool_qinput import AF_EDGES, Q_EDGES
from .v2_observation import CountFrame
from .v2_scene import radius_upper
from .cell_response import rule, response


SCHEMA = 'paired_scenario_coverage_variants_1'
VARIANTS = ('baseline', 'a', 'b', 'c', 'd')
VARIANT_NAMESPACE = 782420
RANGES = dict(
    a=dict(small_base_radius_nm=[.5, 1.5],
        cluster_sigma_log=[.03, .20], cluster_aspect=[.3, 2.],
        remaining_occupation_fraction=[.01, .9]),
    b=dict(lowq_corner_nm_inv=[.015, .5], lowq_relative_gain=[.001, 10.],
        lowq_exponent=[.5, 6.], psf_af_bins=[.03, 3.], psf_q_bins=[.03, 4.]),
    c=dict(delta12=[1e-9, 3e-5], beta12=[1e-11, 3e-6], thickness_nm=[.2, 500.],
        roughness_nm=[.01, 10.], buried_roughness_nm=[.01, 10.],
        roughness_upper_thickness_fraction=.25),
    d=dict(sigma_log=[.03, .65], paracrystal_omega_rel=[.03, 1.]))


def _applicability(row, variant):
    scene = row.get('scenario')
    if scene not in ('S1', 'S2', 'S3', 'S4'):
        raise ValueError('Original physical S1--S4 row required')
    if variant in ('a', 'd') and scene not in ('S1', 'S2'):
        return False, 'This surface-island coverage variant applies only to S1/S2'
    if variant == 'c' and float(row['thickness']) <= 0:
        return False, 'No optional finite layer in this original row; known bare Si is not relabelled'
    return True, None


def draw_variant(row, variant, *, seed_root=None, family_id=None,
                 namespace=VARIANT_NAMESPACE):
    'One deterministic extension draw; no forward/feature rejection or redraw.'
    if variant not in VARIANTS:
        raise ValueError('Coverage variant must be baseline/a/b/c/d')
    fid = row.get('family_id') if family_id is None else family_id
    if fid is None or isinstance(fid, (bool, np.bool_)) or int(fid) != fid or int(fid) < 0:
        raise ValueError('Original nonnegative family ID required')
    if seed_root is None:
        seed = row.get('noise_seed')
        if not seed:
            raise ValueError('Original root seed must be provided or recorded in noise_seed')
        seed_root = seed[0]
    applicable, reason = _applicability(row, variant)
    stream = [int(seed_root), int(namespace), int(fid), VARIANTS.index(variant)]
    rng = np.random.default_rng(np.random.SeedSequence(stream))
    log = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    uniform = lambda lo, hi: float(rng.uniform(lo, hi))
    params = {}
    if applicable and variant == 'a':
        params = dict(small_base_radius_nm=log(.5, 1.5),
            cluster_sigma_log=uniform(.03, .20), cluster_aspect=uniform(.3, 2.),
            cluster_shape=row['particle']['shape'],
            remaining_occupation_fraction=uniform(.01, .9))
    elif applicable and variant == 'b':
        params = dict(lowq_kind=str(rng.choice(['lorentzian', 'soft_power'])),
            lowq_origin=str(rng.choice(['host', 'instrument'])),
            lowq_corner_nm_inv=log(.015, .5), lowq_gain=log(.001, 10.),
            lowq_exponent=uniform(.5, 6.), psf_af=log(.03, 3.), psf_q=log(.03, 4.))
    elif applicable and variant == 'c':
        params = dict(delta12=log(1e-9, 3e-5), beta12=log(1e-11, 3e-6),
                      thickness=log(.2, 500.))
        rough_upper = min(10., .25*params['thickness'])
        params.update(roughness_nm=log(.01, rough_upper),
                      buried_roughness_nm=log(.01, rough_upper))
    elif applicable and variant == 'd':
        params = dict(sigma_log=uniform(.03, .65))
        if row['structure_factor'] == 'paracrystal':
            params['omega_rel'] = uniform(.03, 1.)
    return dict(schema=SCHEMA, variant=variant, family_id=int(fid),
        scenario=row['scenario'], extension_seed=stream,
        original_photon_seed=row.get('noise_seed'), applicable=applicable,
        not_applicable_reason=reason, parameters=params,
        old_chart_label_unchanged=True, extension_in_old_chart=False,
        ranges_status='preregisterable research candidates, not validated facts',
        structural_release=False)


class _WideSpreadParams(SSCAParams):
    'The original marked law, explicitly extended to sigma<=.65 here only.'
    def state(self):
        values = [self.D_nm, self.base_radius_nm, self.height_nm,
                  self.sigma_log, self.omega_rel, self.kappa]
        if not np.isfinite(values).all() or min(values[:3]+[self.omega_rel]) <= 0:
            raise ValueError('Finite positive wide-spread particle scales required')
        if (self.shape not in ('cap', 'hemiellipsoid') or not 0 <= self.kappa <= 1
                or not .03 <= self.sigma_log <= .65):
            raise ValueError('Outside the independent wide-spread coverage domain')
        aspect = self.height_nm/self.base_radius_nm
        if not .1 <= aspect <= 2:
            raise ValueError('Outside the unchanged aspect domain')
        return IntegratedState(n_islands=2/(np.sqrt(3)*self.D_nm**2),
            a_med=self.base_radius_nm, aspect=aspect, s_log=self.sigma_log,
            omega_rel=self.omega_rel, shape=self.shape,
            film_thickness=self.film_thickness_nm, use_avg_material=self.average_material)


def _host(row):
    scale = 12000./row['E']
    return HostOptics(complex(row['delta12']*scale**2, row['beta12']*scale**3),
        row['thickness'], roughness_nm=row.get('roughness_nm', .01),
        buried_roughness_nm=row.get('buried_roughness_nm', .01), transition_nm=0.)


def _cell_means(c, edges):
    'Reintegrate the actual marks on common cells; no averaged-cell remap.'
    t, wt = _gauss_legendre(2)
    result = np.zeros(len(edges)-1)
    widths = np.diff(edges)
    for a, h, w in zip(c['R'], c['H'], c['w']):
        lo = np.clip(edges[:-1], 0., h)
        hi = np.clip(edges[1:], 0., h)
        dz = hi-lo
        z = lo[:, None]+dz[:, None]*t
        tip = (h-hi[:, None])+dz[:, None]*(1-t)
        area = (tip*(a*a/h+z) if c['shape'] == 'cap'
                else a*a*tip*(h+z)/(h*h))
        result += w*np.pi*dz*(area@wt)*c['n']/widths
    return result


def _small_cluster_context(row, parameters, recipe):
    main = original_context(row, recipe)
    p = SSCAParams(**row['particle'])
    a = parameters['cluster_aspect']
    radius = parameters['small_base_radius_nm']


    small_p = SSCAParams(D_nm=1000., base_radius_nm=radius,
        height_nm=radius*a, sigma_log=parameters['cluster_sigma_log'],
        omega_rel=.2, kappa=0., shape=parameters['cluster_shape'])
    initial = build_context(main['host'], small_p, row['metal'], row['E'], row['ai'],
        nbins=recipe['marks'], ndepth=recipe['depth'], reference_mesh='compact')


    maximum_area = initial['max_phi']/initial['n']
    remaining = .999-main['max_phi']
    if not np.isfinite(maximum_area) or maximum_area <= 0 or remaining <= 0:
        raise ValueError('Original main occupation leaves no declared small-cluster budget')
    density = parameters['remaining_occupation_fraction']*remaining/maximum_area
    small_p = replace(small_p, D_nm=float(np.sqrt(2/(np.sqrt(3)*density))))
    small = build_context(main['host'], small_p, row['metal'], row['E'], row['ai'],
        nbins=recipe['marks'], ndepth=recipe['depth'], reference_mesh='compact')
    edges = np.unique(np.r_[main['edges'], small['edges']])
    widths = np.diff(edges); middle = (edges[:-1]+edges[1:])/2
    phi = _cell_means(main, edges)+_cell_means(small, edges)
    if not np.isfinite(phi).all() or np.any(phi < 0) or np.any(phi > 1):
        raise ValueError('Combined physical metal fraction is not in [0,1]')
    from .optics import nd
    base = np.zeros(len(widths), complex)
    base[(middle < 0) & (middle > -main['host'].thickness_nm)] = main['host'].decrement
    base[middle < -main['host'].thickness_nm] = nd('SiO2', energy_ev=row['E'])
    post = base+(nd(row['metal'], energy_ev=row['E'])-base)*phi
    sigma = np.zeros(len(widths)+2)
    for z, rough in ((0., main['host'].roughness_nm),
                     (-main['host'].thickness_nm, main['host'].buried_roughness_nm)):
        where = np.flatnonzero(edges[::-1] == z)
        if len(where): sigma[int(where[0])+1] = max(sigma[int(where[0])+1], rough)
    common = dict(edges=edges, phi=phi, base=base,
        pre_nds=[0j]+base[::-1].tolist()+[nd('Si', energy_ev=row['E'])],
        post_nds=[0j]+post[::-1].tolist()+[nd('Si', energy_ev=row['E'])],
        thick=widths[::-1].tolist(), sigma=sigma.tolist(), mass_nm=float(phi@widths))
    main.update(common); small.update(common)
    shared_field_cache = {}; main['_field_cache'] = small['_field_cache'] = shared_field_cache
    main['_small_cluster_context'] = small
    metadata = dict(main_number_density_nm2=float(main['n']),
        small_number_density_nm2=float(small['n']),
        small_median_base_radius_nm=float(small_p.base_radius_nm),
        small_median_maximum_footprint_diameter_nm=float(small_p.diameter()),
        main_mass_nm=float(main['primitive_mass_nm']),
        small_mass_nm=float(small['primitive_mass_nm']),
        total_mass_nm=float(phi@widths), maximum_cell_phi=float(phi.max()),
        sufficient_point_phi_bound=float(main['max_phi']+small['max_phi']),
        common_mean_field=True, small_structure_factor='independent self term',
        population_fluctuation_cross='assumed zero',
        main_D_and_density_unchanged=True, total_metal_mass_held_fixed=False,
        physical_overlap_or_connectivity_verified=False)
    return main, metadata


def _two_population_intensity(c, af, q, *, order=4, reference=False):
    main = (harddisk_intensity(c, af, q, order=order, reference=reference)
            if '_harddisk_table' in c else metal_intensity(c, af, q, order=order))
    small = c['_small_cluster_context']
    amplitudes = metal_amplitudes(small, af, q, order=order)
    self_term = np.einsum('m,maq->aq', small['w'], abs(amplitudes)**2)
    k = 2*np.pi/wavelength_nm(c['E'])
    return main+self_term*(k*k/(4*np.pi))**2*small['n']


def _cuts(nodes, weights, order):
    grouped = np.asarray(nodes).reshape(-1, order)
    width = np.asarray(weights).reshape(-1, order).sum(axis=1)
    middle = (grouped[:, 0]+grouped[:, -1])/2
    return np.unique(np.r_[middle-width/2, middle+width/2])


class _CoverageOperator(NativeScenarioOperator):
    'Rebuild one native positive-bin PSF, rather than blur a projected image.'
    def __init__(self, context, *, psf_af_bins, psf_q_bins, **kwargs):
        if (not np.isfinite([psf_af_bins, psf_q_bins]).all()
                or not .03 <= psf_af_bins <= 3. or not .03 <= psf_q_bins <= 4.):
            raise ValueError('Outside declared coverage PSF range')
        super().__init__(context, psf_af_bins=min(psf_af_bins, 1.),
                         psf_q_bins=min(psf_q_bins, 1.), **kwargs)
        da = float(AF_EDGES[1]-AF_EDGES[0]); dq = float(Q_EDGES[1]-Q_EDGES[0])
        if psf_af_bins > 1:
            self.ac, self.aw = rule(NATIVE_EDGES, self.order, 4*psf_af_bins*da,
                _cuts(self.ac, self.aw, self.order), self.meta['AF_max_segment_deg'])
        if psf_q_bins > 1:
            extra = list(_cuts(self.qc, self.qw, self.order))
            p = context.get('particle')
            if p is not None:
                aa = np.unique([max(self.ac.min()+self.shift, 0.),
                    max(self.ac.max()+self.shift, 0.), context['ai']])
                maximum = Q_EDGES[-1]+4*psf_q_bins*dq
                for m in range(1, max(1, int(np.ceil(maximum*self.qscale*p.D_nm/(2*np.pi))))+1):
                    peak = 2*np.pi*m/p.D_nm
                    width = (2*np.pi*np.pi*m*m*p.omega_rel**2/p.D_nm+1e-4)/self.qscale
                    if width >= dq/2: continue
                    peaks = _peak_qy(peak, aa, context['E'], context['ai'], self.qscale)
                    if len(peaks):
                        lo, hi = float(peaks.min()-6*width), float(peaks.max()+6*width)
                        extra.extend(np.linspace(lo, hi, max(2, int(np.ceil((hi-lo)/width))+1)))
            self.qc, self.qw = rule(Q_EDGES, self.order, 4*psf_q_bins*dq, extra, dq)
        self.Wa, ea = response(NATIVE_EDGES, self.ac, self.aw, psf_af_bins*da)
        self.Wq, eq = response(Q_EDGES, self.qc, self.qw, psf_q_bins*dq)
        self.psf_a = float(psf_af_bins); self.psf_q = float(psf_q_bins)
        self.meta.update(AF_nodes=len(self.ac), q_nodes=len(self.qc),
            raw_constant_AF=ea, raw_constant_q=eq, psf_af_bins=self.psf_a,
            psf_q_bins=self.psf_q, psf_applied_once=True,
            empirical_template_reblurred=False, coverage_PSF_range=True)

    def lowq_component(self, parameters):
        result = np.zeros((len(NATIVE_AF), len(Q_EDGES)-1))
        for start in range(0, len(self.ac), 64):
            stop = min(start+64, len(self.ac))
            af = self.ac[start:stop]+self.shift; good = af > 0
            if not good.any(): continue
            qp = angles(self.c['E'], self.c['ai'], self.qscale, 0.,
                q_centres=self.qc, af_centres=af[good])['qpar']
            x = qp/parameters['lowq_corner_nm_inv']
            latent = (1/(1+x*x) if parameters['lowq_kind'] == 'lorentzian'
                      else (1+x)**(-parameters['lowq_exponent']))
            result += self.Wa[:, start:stop][:, good]@(latent@self.Wq.T)
        return result


def forward(physical, P, hostarr, *, variant=None, sidecar=None,
            recipe=QUICK, backend='cpu', disk=None):
    'Original-style (rate, pre, receipt); unsupported variants return None rates.'
    if sidecar is None:
        sidecar = draw_variant(physical, 'baseline' if variant is None else variant)
    elif variant is not None and variant != sidecar.get('variant'):
        raise ValueError('Variant and original recorded sidecar differ')
    variant = sidecar.get('variant')
    if sidecar.get('schema') != SCHEMA or variant not in VARIANTS or sidecar.get('scenario') != physical.get('scenario'):
        raise ValueError('Original physical row and coverage sidecar differ')
    applicable, reason = _applicability(physical, variant)
    if applicable != sidecar.get('applicable'):
        raise ValueError('Recorded variant applicability differs')
    if not applicable:
        return None, None, dict(coverage_variant=sidecar, status='not_applicable',
            missing_reason=reason, structural_release=False)
    if variant == 'baseline':
        rate, pre, metadata = baseline_forward(physical, P, hostarr,
                                               recipe=recipe, backend=backend, disk=disk)
        return rate, pre, dict(metadata, coverage_variant=sidecar,
                              baseline_calculation_unchanged=True)
    row = dict(physical); params = sidecar['parameters']; physical_changes = {}
    if variant == 'c':
        old_thickness = float(row['thickness'])
        row.update(params)
        if row['scenario'] == 'S3':
            steps = dict(row['steps']); edges = np.asarray(steps['edges_nm'], float)
            edges = np.where(edges < 0, edges*row['thickness']/old_thickness, edges)
            steps['edges_nm'] = edges.tolist(); row['steps'] = steps
        c = original_context(row, recipe)
        physical_changes = dict(layer_parameters=params, original_layer_thickness=old_thickness,
            S3_buried_depth_fraction_held_fixed=row['scenario'] == 'S3')
    elif variant == 'd':
        old = dict(row['particle']); p = dict(old); p.update(params)
        aspect = old['height_nm']/old['base_radius_nm']
        old_upper = radius_upper(old['D_nm'], old['sigma_log'], aspect, old['shape'])*.999999
        new_upper = radius_upper(p['D_nm'], p['sigma_log'], aspect, p['shape'])*.999999
        lo = .400001
        if old_upper <= lo or new_upper <= lo or not lo <= old['base_radius_nm'] <= old_upper:
            raise ValueError('Original or widened declared radius interval is empty/incompatible')
        quantile = np.log(old['base_radius_nm']/lo)/np.log(old_upper/lo)
        p['base_radius_nm'] = float(lo*np.exp(quantile*np.log(new_upper/lo)))
        p['height_nm'] = p['base_radius_nm']*aspect
        row['particle'] = p
        wide = _WideSpreadParams(**p)
        if row.get('decoupling') == 'DA': wide = replace(wide, kappa=0.)
        c = build_context(_host(row), wide, row['metal'], row['E'], row['ai'],
            nbins=recipe['marks'], ndepth=recipe['depth'], reference_mesh='compact')
        physical_changes = dict(particle=p, radius_prior_quantile=float(quantile),
            radius_rebounded_for_actual_wide_occupation=True,
            total_metal_mass_held_fixed=False,
            mark_law='same fixed-normal [-8,8] positive quadrature, sigma<=.65; tails and speed are candidate limitations')
    elif variant == 'a':
        c, physical_changes = _small_cluster_context(row, params, recipe)
    else:
        c = original_context(row, recipe); row['psf_af'] = params['psf_af']; row['psf_q'] = params['psf_q']
        physical_changes = dict(background_is_metal=False, lowq_origin=params['lowq_origin'])
    fn = None
    if row.get('structure_factor') == 'harddisk' and row['scenario'] != 'S4':
        if disk is None: raise ValueError('Original bound hard-disk table required; no extrapolation')
        c['_harddisk_table'] = disk; c['_harddisk_eta'] = row['eta']; fn = harddisk_intensity
    if variant == 'a': fn = _two_population_intensity
    op = _CoverageOperator(c, scenario='S1' if row['scenario'] == 'S2' else row['scenario'],
        reference=recipe['reference'], qscale=row['qscale'], af_shift=row['af_shift'],
        psf_af_bins=row['psf_af'], psf_q_bins=row['psf_q'], backend=backend, intensity_fn=fn)
    parts = op.evaluate(db_xi_nm=row['db_xi'], power_exponent=row['power_exponent'],
                        scatter_depth_nm=row['scatter_depth'])
    ref = max(float(np.quantile(parts['metal'], .8)), 1e-30)
    empirical = np.zeros_like(P, dtype=float); pre = None; missing = 1.
    key = row.get('host_template_key')
    if key:
        shape = hostarr[key+'__shape']; valid = hostarr[key+'__valid']
        empirical = np.where(valid, shape, parts['db'])*ref*row['host_fraction']
        pre = CountFrame(shape*row['target_scale']*P*valid, P*valid, False, None)
        missing = float((~valid).mean())
    lowq = op.lowq_component(params)*ref*params['lowq_gain'] if variant == 'b' else 0.
    low_host = lowq if variant == 'b' and params['lowq_origin'] == 'host' else 0.
    low_instrument = lowq if variant == 'b' and params['lowq_origin'] == 'instrument' else 0.
    background = ref*(row['flat_fraction']*parts['flat']+row['ring_fraction']*parts['ring'])+low_instrument
    if row['scenario'] == 'S2':
        host = empirical+low_host+ref*(row['db_fraction']*parts['db']+row['power_fraction']*parts['power'])
        mixed = compose_template(parts['metal'], host, loading=row['loading'],
            growth=TemplateGrowth(**row['growth']), cross_coefficient=row['gamma'],
            transmission=parts['transport'], transmitted_fraction=row['transmitted_fraction'],
            instrumental_rate=background, host_source=key or 'parametric_host')
        rate = mixed['rate']; composition = mixed['metadata']
    else:
        rate = compose(parts, empirical+low_host, db_gain=ref*row['db_fraction'],
            power_gain=ref*row['power_fraction'], flat_gain=ref*row['flat_fraction'],
            ring_gain=ref*row['ring_fraction'], gamma=row['gamma'],
            transmitted_fraction=row['transmitted_fraction'])+low_instrument
        composition = dict(host_source=key, cross='effective intensity correlation')
    scale = expected_scale(rate, P)
    if not np.isfinite(scale) or scale <= 0: raise ValueError('No positive candidate input scale')
    rate = np.broadcast_to(rate*row['target_scale']/scale, P.shape)
    if row['scenario'] == 'S4': labels = ts_labels(c)
    elif row['scenario'] == 'S3': labels = c['labels']
    else:
        labels = dict(D_nm=float(c['particle'].D_nm),
            D_definition='conditional primitive centre density length; not SEM nearest neighbour',
            lateral_diameter_nm=float(c['particle'].diameter()),
            diameter_definition='main population median maximum footprint diameter',
            island_height_nm=float(c['particle'].height_nm), gap_nm=None,
            coverage=None, conditional_geometry=True, structural_release=False)
    return rate, pre, dict(coverage_variant=sidecar, status='computed_candidate',
        physical_changes=physical_changes, operator=op.meta, recipe=recipe, labels=labels,
        expected_scale=scale, metal_reference=ref, host_missing_fraction=missing,
        composition=composition, mean_rate_support_applied=False,
        extension_in_old_chart=False, conditional_parameters_calibrated=False,
        numerical_certificate=False, structural_release=False)
