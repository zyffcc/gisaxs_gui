'Executed four-scenario mean forward, before a single photon observation.'
from dataclasses import replace
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .effective_host import HostOptics, build_context, metal_amplitudes
from .effective_measurement import compose
from .forward_v06 import SSCAParams
from .optics import wavelength_nm
from .scenario_template import TemplateGrowth, compose_template
from .scenario_steps import StepProfile, build_step_context, step_vertical_amplitude
from .scenario_ts import TSParams, build_context as ts_context, labels as ts_labels
from .v2_observation import CountFrame
from .formfactor import jinc


QUICK = dict(marks=32, depth=16, reference=False)
REFERENCE = dict(marks=64, depth=32, reference=True)


def context(row, recipe=QUICK):
    "Exactly the row's optical/metal state, including loading in the field."
    E = row['E']; scale = 12000./E
    host = HostOptics(complex(row['delta12']*scale**2, row['beta12']*scale**3),
        row['thickness'], roughness_nm=row.get('roughness_nm', .01),
        buried_roughness_nm=row.get('buried_roughness_nm', .01), transition_nm=0.)
    scene = row['scenario']
    if scene == 'S4':
        return ts_context(host, TSParams(**row['ts']), metal=row['metal'],
                          energy_ev=E, alpha_i_deg=row['ai'])
    particle = SSCAParams(**row['particle'])
    if row.get('decoupling') == 'DA':
        particle = replace(particle, kappa=0.)
    if scene == 'S3':
        return build_step_context(host, particle, StepProfile(**row['steps']),
            row['metal'], E, row['ai'], nbins=recipe['marks'])
    return build_context(host, particle, row['metal'], E, row['ai'],
        burial_nm=0., nbins=recipe['marks'], ndepth=recipe['depth'],reference_mesh='compact')


def harddisk_intensity(c, af, q, *, order=4, reference=False):
    "BornAgain's stored analytic hard-disk S(q), marked DA, not SSCA."
    if c.get('scenario') == 'S3':
        raise ValueError('Independent S3 occupation is incompatible with density-defined harddisk; use paracrystal')
    F = metal_amplitudes(c, af, q, order=order)
    qq = np.asarray(q)
    if qq.ndim == 1: qq = np.broadcast_to(qq, F.shape[1:])
    table = c['_harddisk_table']; eta = c['_harddisk_eta']; p = c['particle']
    n = 2/(np.sqrt(3)*p.D_nm**2)
    exclusion = np.sqrt(eta/(np.pi*n))
    x = np.column_stack((np.full(qq.size, eta), qq.ravel()*exclusion))
    S = table(x).reshape(qq.shape)
    mean = np.einsum('m,maq->aq', c['w'], F)
    self_term = np.einsum('m,maq->aq', c['w'], abs(F)**2)
    power = self_term + abs(mean)**2*(S-1)
    if power.min() < -1e-10*max(float(abs(power).max()), 1e-30):
        raise ValueError('Nonpositive marked hard-disk DA')
    k = 2*np.pi/wavelength_nm(c['E'])
    return np.maximum(power, 0.)*(k*k/(4*np.pi))**2*c['n']


def harddisk_table(arrays):
    return RegularGridInterpolator((arrays['eta'], arrays['qr']), arrays['S'],
                                   bounds_error=True)


def expected_scale(rate, P):
    from .tool_qinput import AF_EDGES, GRID_SHAPE
    from .stage5_measurement import NATIVE_AF
    ci = np.searchsorted(AF_EDGES, NATIVE_AF, side='right')-1
    y = np.zeros(GRID_SHAPE); n = np.zeros(GRID_SHAPE)
    np.add.at(y, ci, (np.broadcast_to(rate, P.shape)*P).sum(axis=0))
    np.add.at(n, ci, P.sum(axis=0))
    r = np.divide(y, n, out=np.zeros_like(y), where=n>0)
    return float(np.quantile(r[n>0], .8)) if np.any(n>0) else 0.


def mean_forward(row, P, hostarr, *, recipe=QUICK, backend='cpu', disk=None):
    'Return atomic mean rate, optional candidate pre-shape, honest labels.'
    from .scenario_measurement import NativeScenarioOperator
    c = context(row, recipe)
    fn = None
    if row.get('structure_factor') == 'harddisk' and row['scenario'] != 'S4':
        if disk is None: raise ValueError('Bound hard-disk table required')
        c['_harddisk_table'] = disk; c['_harddisk_eta'] = row['eta']
        fn = harddisk_intensity
    op = NativeScenarioOperator(c, scenario='S1' if row['scenario']=='S2' else row['scenario'],
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
    background = ref*(row['flat_fraction']*parts['flat'] + row['ring_fraction']*parts['ring'])
    if row['scenario'] == 'S2':
        host = empirical + ref*(row['db_fraction']*parts['db']+row['power_fraction']*parts['power'])
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
    if not np.isfinite(scale) or scale<=0: raise ValueError('No positive expected input scale')
    rate = np.broadcast_to(rate*row['target_scale']/scale, P.shape)
    if row['scenario']=='S4':
        labels = ts_labels(c)
    elif row['scenario']=='S3':
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
        approximations='S2 effective intensity cross; S3 marked columns; S4 two-phase correlation, topology unverified')
