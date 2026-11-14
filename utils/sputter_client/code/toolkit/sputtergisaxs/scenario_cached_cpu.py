'Per-draw CPU caching of the existing four-scenario mean forward.'
import hashlib
from pathlib import Path

import numpy as np

from .effective_host import _cell_field, field_arrays, metal_quadrature
from .effective_measurement import compose
from .formfactor import jinc
from .forward_v06 import ssca_intensity
from .optics import wavelength_nm
from .scenario_forward import (
    QUICK, context, expected_scale, harddisk_intensity,
)
from .scenario_measurement import NativeScenarioOperator
from .scenario_steps import step_vertical_amplitude
from .scenario_template import TemplateGrowth, compose_template
from .scenario_ts import labels as ts_labels
from .v2_observation import CountFrame


SCHEMA = 'four_scenario_per_draw_cached_CPU_1'
SOURCE_FILES = (
    'scenario_cached_cpu.py', 'scenario_forward.py', 'scenario_measurement.py',
    'effective_host.py', 'effective_measurement.py', 'formfactor.py',
    'forward_v06.py', 'scenario_steps.py', 'scenario_template.py',
    'scenario_ts.py', 'multilayer.py', 'optics.py',
)


def source_receipt():
    'Executing source identity, not an inherited numerical certificate.'
    root = Path(__file__).resolve().parent
    pins = {name: hashlib.sha256((root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in SOURCE_FILES}
    return dict(schema=SCHEMA, backend='cpu', source_pins=pins,
        cache_scope='one original context/operator; never across posterior draws',
        arithmetic='original float64/complex128', original_nodes_and_AF_blocks=True,
        original_photon_observation_not_changed=True, surrogate_model=False,
        original_numerical_certificate_reused=False,
        numerical_status='requires actual original-TRAIN fixed-draw comparison')


def _power_checked(power, tolerance, reason):
    if (not np.isfinite(power).all()
            or power.min() < -tolerance*max(float(abs(power).max()), 1e-30)):
        raise ValueError(reason)
    return np.maximum(power, 0.)


class CachedCPUScenarioOperator(NativeScenarioOperator):
    'Original measurement operator with a single bounded surface-q cache.'
    def __init__(self, context, scenario='S1', reference=False, qscale=1.,
                 af_shift=0., psf_af_bins=.2, psf_q_bins=.2, backend='cpu',
                 intensity_fn=None):
        if backend != 'cpu':
            raise ValueError('Per-draw cached operator is a CPU implementation')
        super().__init__(context, scenario=scenario, reference=reference,
            qscale=qscale, af_shift=af_shift, psf_af_bins=psf_af_bins,
            psf_q_bins=psf_q_bins, backend=backend, intensity_fn=intensity_fn)
        self._surface_q_cache = None
        self.meta.update(cached_CPU_schema=SCHEMA,
            surface_q_cache_scope='single original operator/context',
            separable_B_cached=False, SSCA_q_terms_cached=False,
            harddisk_q_table_cached=False,
            incident_field_and_cell_gather_cached=False,
            S3_common_vertical_factored=False, cache_crosses_draws=False,
            precision_or_nodes_changed=False,
            cached_CPU_original_numerical_certificate_reused=False)

    def _surface_cache(self, q, incident):
        c = self.c; p = c['particle']


        quadrature = metal_quadrature(c, self.F_order)
        radii = np.asarray(c['R'], float); weights = np.asarray(c['w'], float)
        harddisk = self.intensity_fn is harddisk_intensity
        table = c.get('_harddisk_table') if harddisk else None
        eta = c.get('_harddisk_eta') if harddisk else None
        signature = (id(quadrature), self.F_order, q.shape, q.tobytes(),
            radii.tobytes(), weights.tobytes(), float(p.D_nm),
            float(p.omega_rel), float(p.kappa), harddisk, id(table), eta,
            c['E'], c['ai'], tuple(c['post_nds']), tuple(c['thick']), tuple(c['sigma']))
        previous = self._surface_q_cache
        if previous is not None and previous['signature'] == signature:
            return previous
        marks = []
        edges = c['edges']; cells = len(edges)-1
        for z, rad, weight in quadrature:
            if np.any(z < edges[0]-1e-12) or np.any(z > edges[-1]+1e-12):
                raise ValueError('Field query outside represented effective reference')
            upward = np.clip(np.searchsorted(edges, z, side='right')-1, 0, cells-1)
            layer = cells-upward; distance = edges[upward+1]-z
            Ei = _cell_field(incident['kz'][layer, 0], incident['T'][layer, 0],
                             incident['R'][layer, 0], distance)
            B = weight[:, None]*jinc(rad[:, None]*q[None, :])
            marks.append((layer, distance, Ei, B))
        cache = dict(signature=signature, marks=tuple(marks))
        self.meta['separable_B_cached'] = True
        self.meta['incident_field_and_cell_gather_cached'] = True
        if harddisk:
            n = 2/(np.sqrt(3)*p.D_nm**2)
            exclusion = np.sqrt(eta/(np.pi*n))
            cache['S'] = table(np.column_stack((np.full(q.size, eta), q*exclusion)))
            self.meta['harddisk_q_table_cached'] = True
        else:
            dr = radii-float(weights@radii)
            phase = np.exp(1j*p.kappa*dr[:, None]*q[None, :])
            cache.update(phase=phase,
                p0=np.exp(1j*q*p.D_nm-.5*(q*p.omega_rel*p.D_nm)**2-p.D_nm/1e4),
                p2=weights@(phase**2))
            self.meta['SSCA_q_terms_cached'] = True
        self._surface_q_cache = cache
        return cache

    def _surface_intensity(self, af, q):
        c = self.c; weights = c['w']
        incident, reciprocal = field_arrays(c, af)
        cache = self._surface_cache(q, incident)
        F = np.empty((len(c['R']), len(af), len(q)), complex)
        for mark, (layer, distance, Ei, B) in enumerate(cache['marks']):
            Ef = _cell_field(reciprocal['kz'][layer].T,
                             reciprocal['T'][layer].T,
                             reciprocal['R'][layer].T, distance)
            F[mark] = (Ef*Ei[None, :])@B
        if not np.isfinite(F).all():
            raise ValueError('Nonfinite contrast amplitudes')
        self_term = np.einsum('j,jk->k', weights,
                             abs(F.reshape(len(c['R']), -1))**2).reshape(F.shape[1:])
        if self.intensity_fn is harddisk_intensity:
            mean = np.einsum('m,maq->aq', weights, F)
            power = self_term+abs(mean)**2*(cache['S'][None, :]-1)
            power = _power_checked(power, 1e-10, 'Nonpositive marked hard-disk DA')
        else:
            phase = cache['phase'][:, None, :]
            fp = np.einsum('j,jk->k', weights,
                          (F*phase).reshape(len(c['R']), -1)).reshape(F.shape[1:])
            fm = np.einsum('j,jk->k', weights,
                          (F/phase).reshape(len(c['R']), -1)).reshape(F.shape[1:])
            p0 = cache['p0'][None, :]; p2 = cache['p2'][None, :]
            power = self_term+2*np.real(p0*fp*np.conj(fm)/(1-p0*p2))
            power = _power_checked(power, 1e-7, 'SSCA positivity failure')
        k = 2*np.pi/wavelength_nm(c['E'])
        return power*(k*k/(4*np.pi))**2*c['n']

    def _step_intensity(self, af, q):
        c = self.c; p = c['particle']; R = c['R']; weights = c['w']
        qq = np.asarray(q, float)
        if qq.ndim == 1:
            qq = np.broadcast_to(qq, (len(af), len(qq)))
        if (qq.ndim != 2 or qq.shape[0] != len(af)
                or not np.isfinite(qq).all() or np.any(qq < 0)):
            raise ValueError('Finite nonnegative exact ray q grid required')


        vertical = step_vertical_amplitude(c, af, order=self.F_order,
                                           method='analytic')
        lateral = np.pi*R[:, None, None]**2*jinc(R[:, None, None]*qq[None, :, :])
        raw = ssca_intensity(lateral.reshape(len(R), -1), R, weights,
            qq.ravel(), p.D_nm, p.omega_rel*p.D_nm, p.kappa).reshape(qq.shape)
        power = raw*abs(vertical)[:, None]**2
        power = _power_checked(power, 1e-7, 'Effective-step SSCA positivity failure')
        self.meta['S3_common_vertical_factored'] = True


        pref = (2*np.pi/wavelength_nm(c['E']))**4/(16*np.pi*np.pi)*c['n']
        return pref*power

    def _intensity(self, af, q):
        aa = np.atleast_1d(np.asarray(af, float)); qq = np.asarray(q, float)
        if (self.scenario == 'S1' and not self.reference and qq.ndim == 1
                and (self.intensity_fn is None or self.intensity_fn is harddisk_intensity)):
            if not np.isfinite(qq).all() or np.any(qq < 0):
                raise ValueError('Invalid separable q grid')
            return self._surface_intensity(aa, qq)
        if self.scenario == 'S3' and self.intensity_fn is None:
            return self._step_intensity(aa, qq)
        return super()._intensity(aa, qq)


def mean_forward_cached(row, P, hostarr, *, recipe=QUICK, backend='cpu', disk=None):
    'Original per-draw composition using the independent cached CPU operator.'
    if backend != 'cpu':
        raise ValueError('mean_forward_cached requires backend=cpu')
    c = context(row, recipe); fn = None
    if row.get('structure_factor') == 'harddisk' and row['scenario'] != 'S4':
        if disk is None:
            raise ValueError('Bound hard-disk table required')
        c['_harddisk_table'] = disk; c['_harddisk_eta'] = row['eta']
        fn = harddisk_intensity
    op = CachedCPUScenarioOperator(c, scenario='S1' if row['scenario']=='S2' else row['scenario'],
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
    background = ref*(row['flat_fraction']*parts['flat']+row['ring_fraction']*parts['ring'])
    if row['scenario'] == 'S2':
        host = empirical+ref*(row['db_fraction']*parts['db']+row['power_fraction']*parts['power'])
        mixed = compose_template(parts['metal'], host, loading=row['loading'],
            growth=TemplateGrowth(**row['growth']), cross_coefficient=row['gamma'],
            transmission=parts['transport'], transmitted_fraction=row['transmitted_fraction'],
            instrumental_rate=background, host_source=key or 'parametric_host')
        rate = mixed['rate']; composition = mixed['metadata']
    else:
        rate = compose(parts, empirical, db_gain=ref*row['db_fraction'],
            power_gain=ref*row['power_fraction'], flat_gain=ref*row['flat_fraction'],
            ring_gain=ref*row['ring_fraction'], gamma=row['gamma'],
            transmitted_fraction=row['transmitted_fraction'])
        composition = {'host_source':key, 'cross':'effective intensity correlation'}
    scale = expected_scale(rate, P)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('No positive expected input scale')
    rate = np.broadcast_to(rate*row['target_scale']/scale, P.shape)
    if row['scenario'] == 'S4':
        labels = ts_labels(c)
    elif row['scenario'] == 'S3':
        labels = c['labels']
    else:
        labels = dict(D_nm=float(c['particle'].D_nm),
            D_definition='conditional radial correlation step' if row['structure_factor']=='paracrystal'
                else 'conditional hex-equivalent centre density length',
            lateral_diameter_nm=float(c['particle'].diameter()),
            diameter_definition='median maximum footprint diameter',
            island_height_nm=float(c['particle'].height_nm),
            coverage=None, gap_nm=None, conditional_geometry=True, structural_release=False)
    return rate, pre, dict(operator=op.meta, recipe=recipe, labels=labels,
        expected_scale=scale, metal_reference=ref, host_missing_fraction=missing,
        composition=composition, mean_rate_support_applied=False,
        approximations='S2 effective intensity cross; S3 marked columns; S4 two-phase correlation, topology unverified',
        cached_CPU_receipt=source_receipt())
