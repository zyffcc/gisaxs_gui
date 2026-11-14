'Bound new51D clean-mean previews of original posterior medoids.'
from contextlib import ExitStack
from numbers import Integral
from pathlib import Path

import numpy as np

from .scenario_mainline_chart import SCHEMA as CHART_SCHEMA, DIM, decode
from .scenario_mainline_forward import mean_forward
from .scenario_mainline_posterior import MainlineRuntime
from .scenario_mainline_training import BANK_SCHEMAS, read, bound, generating_source_sha
from .scenario_forward import QUICK, harddisk_table
from .scenario_fixed_af_observation import observe_candidate_frame_fixed_af
from .scenario_runtime import _array_sha, _frame_identity, _inside, _sha, _source_sha
from .v2_observation import CountFrame
from .v2_ppc import GATES

SCHEMA = 'exploratory_mainline51_original_medoid_preview_1'
ERRORS = (ValueError, TypeError, KeyError, IndexError, OverflowError,
          FloatingPointError, RuntimeError, MemoryError, OSError)


def _execution_pins(runtime):
    'Recheck sources actually executed by the already strict runtime.'
    current = Path(__file__).resolve().parents[1]
    pins = runtime.declaration['source_pins']
    if runtime.pins != pins or runtime.receipt.get('source_pins') != pins:
        raise ValueError('Original51D runtime source receipt differs')
    for relative, digest in pins.items():
        hasher=generating_source_sha if relative in runtime.declaration['bank']['source_pins'] else _source_sha
        if hasher(_inside(current, relative)) != digest:
            raise ValueError('Executing original51D source changed: '+relative)
    return dict(pins)


def _asset(plan_path, plan, name, supplied):
    path = bound(plan_path.parent, plan['bindings'][name])
    if supplied is not None:
        given = Path(supplied).resolve()
        if _sha(given) != plan['bindings'][name]['sha256']:
            raise ValueError('Original51D preview asset differs: '+name)
        path = given
    return path


def _descriptor(mu, public, band):
    frame = CountFrame(mu, public.P, False)
    if band is not None:
        return observe_candidate_frame_fixed_af(frame, AF_band=band, estimate_stat_se=False)


    out = observe_candidate_frame_fixed_af(frame, AF_band=0, estimate_stat_se=False)
    out.update(AF_band=None, support='unresolved', comparison_band_source='no_observed_comparison_band',
               peak_measurability_known=False, no_peak_with_sufficient_support=False,
               bands_support=[None]*4)
    for key in ('q_star', 'FWHM', 'highq_ratio', 'valley_ratio'):
        out['values'][key] = None
        out['feature_missing_reasons'][key] = 'no_observed_comparison_band'
    return out


def _failure(index, band, error):
    return dict(index=index, forward_failed=True, AF_band=band,
        values={key: None for key in GATES},
        feature_missing_reasons={key: '尚待模拟核对：'+str(error) for key in GATES},
        failure_reason=str(error), representative_clean_mean=True,
        posterior_predictive_replica=False, original_joint_modified=False,
        actual_joint_dim=DIM, old44D_forward_used=False,
        numerical_certificate=False, calibrated=False, structural_release=False)


def preview_medoid_features(sampled, candidates, full, public, *, runtime=None,
        bank_plan=None, feature_certificate=None, host_shapes=None,
        host_metadata=None, harddisk=None, backend='cpu', observed=None):
    'Compatible preview signature; old certificates are explicitly rejected.'
    from .scenario_candidates import observe_candidate_frame
    identity = _frame_identity(full, public)
    observed = (observe_candidate_frame(public,
        estimate_stat_se=identity['photon_statistics_known']) if observed is None else observed)
    band = observed['AF_band']
    source = dict(kind='actual_new51D_original_joint_representatives', certified=False,
        numerical_certificate=False, actual_joint_dim=DIM,
        preview_source_sha256=_source_sha(Path(__file__)),
        expanded_numerical_equivalence='not yet certified',
        old44D_certificate_borrowed=False)
    error = None
    paths = None
    with ExitStack() as stack:
        hosts = disk = None
        try:
            if not isinstance(runtime, MainlineRuntime):
                raise TypeError('Actual source-bound MainlineRuntime required')
            if backend not in ('cpu', 'cuda'):
                raise ValueError('Explicit CPU/CUDA preview backend required')
            if feature_certificate is not None:
                raise ValueError('No old44D feature certificate applies to new51D physics')
            if sampled.get('receipt') != runtime.receipt or sampled.get('frame_identity') != identity:
                raise ValueError('Original51D sampled runtime/full-public observation differs')
            if bank_plan is None:
                raise ValueError('Original51D generating bank plan required')
            pins = _execution_pins(runtime)
            plan_path = Path(bank_plan).resolve(); plan = read(plan_path)
            bank = runtime.declaration['bank']
            if (_sha(plan_path) != bank['plan_sha256'] or plan.get('schema') not in BANK_SCHEMAS or plan.get('schema')!=bank['schema']
                    or plan.get('physics_schema') != CHART_SCHEMA
                    or plan.get('status') != 'frozen' or plan.get('TEST_rows') != 0
                    or plan.get('real_TEST_decoded') is not False
                    or plan['catalog'] != runtime.declaration['catalog']
                    or plan['coordinates'] != runtime.declaration['coordinates']
                    or plan['source_pins'] != bank['source_pins']
                    or plan['quick_recipe'] != QUICK):
                raise ValueError('Original actual51D plan/chart/source/recipe differs')
            hp = _asset(plan_path, plan, 'host_shapes.npz', host_shapes)
            hm = _asset(plan_path, plan, 'host_shapes.json', host_metadata)
            dp = _asset(plan_path, plan, 'harddisk.npz', harddisk)
            if tuple(read(hm)['rows']) != runtime.templates:
                raise ValueError('Bound original TRAIN host-template metadata differs')
            hosts = stack.enter_context(np.load(hp, allow_pickle=False))
            disk = harddisk_table(stack.enter_context(np.load(dp, allow_pickle=False)))
            paths = dict(plan=plan_path, hosts=hp, metadata=hm, disk=dp)
            source.update(plan_sha256=_sha(plan_path), host_shapes_sha256=_sha(hp),
                host_metadata_sha256=_sha(hm), harddisk_sha256=_sha(dp),
                runtime_receipt=dict(runtime.receipt), execution_source_pins=pins)
        except ERRORS as exception:
            error = type(exception).__name__+': '+str(exception)
        output = {}
        for scene, group in candidates['scenarios'].items():
            part = sampled['scenarios'][scene]
            records = {int(draw['index']): draw for draw in part['draws']}
            original = part.get('original_joint')
            group_error = error
            try:
                count = part['original_draws']
                if (len(records) != count or [draw['index'] for draw in part['draws']] != list(range(count))
                        or original is None or np.shape(original['z']) != (count, DIM)
                        or any(np.shape(original[key]) != (count,) for key in ('coarse', 'template'))):
                    raise ValueError('Original51D joint draw denominator/order/dimension changed')
            except ERRORS as exception:
                group_error = type(exception).__name__+': '+str(exception)
            descriptors = []
            for candidate in group['candidates']:
                index = candidate['original_draw_index']
                try:
                    if group_error is not None:
                        raise ValueError(group_error)
                    if (isinstance(index, bool) or not isinstance(index, Integral)
                            or index not in records or candidate.get('scenario') != scene
                            or candidate.get('representative_is_original_medoid') is not True):
                        raise ValueError('Original same-scenario medoid index required')
                    record = records[index]
                    if record.get('decode_failed') or record.get('physical') is None:
                        raise ValueError(record.get('reason', 'Original joint draw did not decode'))
                    geometry = sampled['geometry']
                    row = decode(original['z'][index], int(original['coarse'][index]),
                        int(original['template'][index]), metal=geometry['metal'],
                        E=geometry['energy_ev'], ai=geometry['alpha_i_deg'],
                        template_keys=runtime.keys, template_metadata=runtime.templates)
                    if row != record['physical'] or row['scenario'] != scene:
                        raise ValueError('Original51D complete physical medoid changed')
                    rate, _, receipt = mean_forward(row, full.P, hosts,
                        recipe=QUICK, backend=backend, disk=disk)
                    raw_mu = np.asarray(rate, float)*public.P
                    if (raw_mu.shape != np.shape(public.P) or not np.isfinite(raw_mu).all()
                            or np.any(raw_mu < 0) or np.any(raw_mu[public.P == 0] != 0)
                            or float(raw_mu.sum()) <= 0 or float(public.Y.sum()) <= 0):
                        raise ValueError('Actual supported representative mean/observation is not positive finite')
                    gain = float(public.Y.sum())/float(raw_mu.sum()); mu = raw_mu*gain
                    descriptor = _descriptor(mu, public, band)
                    descriptor.update(index=int(index), fitted_scalar_gain=gain,
                        gain_kind='Poisson_global_scalar_MLE' if identity['photon_statistics_known'] else
                            'shape_moment_scalar; not PoissonMLE',
                        representative_clean_mean=True, original_joint_modified=False,
                        count_support_applications=1, posterior_predictive_replica=False,
                        raw_expected_photons_sha256=_array_sha(raw_mu),
                        profiled_mean_photons_sha256=_array_sha(mu), forward_receipt=receipt,
                        actual_joint_dim=DIM, modifiers=list(row['modifiers']),
                        old44D_forward_used=False, numerical_certificate=False,
                        calibrated=False, structural_release=False)
                except ERRORS as exception:
                    descriptor = _failure(index, band, type(exception).__name__+': '+str(exception))
                descriptors.append(descriptor)
            output[scene] = dict(draws=descriptors,
                original_joint_sha256={key: _array_sha(value) for key, value in original.items()}
                    if original is not None else None,
                original_medoid_indices=[candidate['original_draw_index'] for candidate in group['candidates']])
        if _frame_identity(full, public) != identity:
            raise ValueError('Original full/public observation changed during preview')
        if paths is not None:
            _execution_pins(runtime)
            for name, digest in (('plan', source['plan_sha256']), ('hosts', source['host_shapes_sha256']),
                                 ('metadata', source['host_metadata_sha256']), ('disk', source['harddisk_sha256'])):
                if _sha(paths[name]) != digest:
                    raise ValueError('Original51D preview asset changed during execution: '+name)
    return dict(schema=SCHEMA, scenarios=output,
        match_kind='profiled_clean_mean_original51D_medoids', source=source,
        frame_identity=identity, mean_image_is_not_posterior_average=True,
        full_posterior_predictive_check=False, new_photon_replica_drawn=False,
        original_physical_records_changed=False, actual_joint_dim=DIM,
        numerical_certificate=False, calibrated=False, structural_release=False)
