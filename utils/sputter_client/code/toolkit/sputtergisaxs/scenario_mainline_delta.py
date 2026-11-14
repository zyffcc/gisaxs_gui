'Optional signed deposition-change inputs, separate from shape-only ``pre``.'
import hashlib
import json
from pathlib import Path

import numpy as np

from .scenario_reference_change import reference_change, _frame_identity
from .stage5_measurement import Q, counts_profiles


SCHEMA = 'mainline_signed_reference_delta_1'
PROFILE_SHAPE = (8, len(Q))
VECTOR_DIM = 2 * int(np.prod(PROFILE_SHAPE)) + 4
ARRAY_FIELDS = ('delta', 'delta_known', 'platform', 'platform_known',
                'raw_difference', 'raw_known', 'conditional_difference',
                'conditional_known', 'global_ratio', 'global_ratio_known')
SOURCE_FILES = ('sputtergisaxs/scenario_mainline_delta.py',
    'sputtergisaxs/scenario_reference_change.py', 'sputtergisaxs/reference_changes.py',
    'sputtergisaxs/v2_observation.py', 'sputtergisaxs/stage5_measurement.py',
    'sputtergisaxs/features.py', 'sputtergisaxs/tool_qinput.py')
INPUT_SPEC = dict(schema=SCHEMA, profiles='eight fixed AF-band/side signed profiles',
    transform='asinh(conditional signed difference/current observed rate q80)',
    shape=list(PROFILE_SHAPE), vector_dim=VECTOR_DIM,
    platform=['relative_slope', 'comparable_highq_bins/16'],
    input_exclusions=['reference scalar c', 'DB change length', 'absolute metal amount',
                      'host/template class truth', 'latent size or metal spacing'],
    unprovided_or_failed_platform='all delta values unknown; main input continues',
    signed_difference_is_CountFrame=False, existing_pre_is_delta_input=False)


def source_pins():
    root = Path(__file__).resolve().parents[1]
    return {name: hashlib.sha256((root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for name in SOURCE_FILES}


def empty_arrays():
    result = {name: np.zeros(PROFILE_SHAPE, bool if name.endswith('known') else np.float32)
              for name in ('delta', 'delta_known', 'raw_difference', 'raw_known',
                           'conditional_difference', 'conditional_known')}
    result.update(platform=np.zeros(2, np.float32), platform_known=np.zeros(2, bool),
                  global_ratio=np.zeros(len(Q), np.float32), global_ratio_known=np.zeros(len(Q), bool))
    return result


def _stored(values, valid):
    values = np.asarray(values, float); known = np.asarray(valid, bool) & np.isfinite(values)
    return np.where(known, values, 0.).astype(np.float32), known


def paired_delta(current, reference, **provenance):
    'Return arrays and compact provenance using the original scalar rule.'
    change = reference_change(current, reference, **provenance)
    result = empty_arrays(); raw = change.get('raw_change')
    conditional = change.get('conditional_subtraction'); scale = None
    if raw is not None:
        result['raw_difference'], result['raw_known'] = _stored(raw['difference'], raw['valid'])
        n = current.P * (current.exposure_s if change['actual_exposures_normalized'] else 1.)
        _, _, _, _, _, scale = counts_profiles(current.Y, n)
    if conditional is not None and conditional.get('difference') is not None:
        difference = np.asarray(conditional['difference'], float)
        result['conditional_difference'], result['conditional_known'] = _stored(difference, conditional['valid'])
        if conditional['shape_condition_met'] and scale is not None and np.isfinite(scale) and scale > 0:
            result['delta'], result['delta_known'] = _stored(np.arcsinh(difference/scale), conditional['valid'])
    global_scale = change.get('global_scale')
    if global_scale is not None:
        ratio = np.asarray(global_scale['current_to_reference_expected_ratio'], float)
        result['global_ratio'], result['global_ratio_known'] = _stored(ratio, np.isfinite(ratio))
        if bool(global_scale['shape_condition_met']):
            result['platform'][:] = (global_scale['relative_slope'],
                                     global_scale['highq_comparable_ratio_q_bins']/16.)
            result['platform_known'][:] = True
    summary = dict(schema=SCHEMA, status=change['status'], source_pins=source_pins(),
        current_source=change.get('current_source'), reference_source=change.get('reference_source'),
        geometry_identity=change.get('geometry_identity'),
        current_frame_identity=change.get('current_frame_identity') or _frame_identity(current),
        reference_frame_identity=change.get('reference_frame_identity'),
        common_native_support_sha256=change.get('common_native_support_sha256'),
        common_pixel_support_verified=change['common_pixel_support_verified'],
        common_support_scope=change.get('common_support_scope'),
        current_exposure_s=current.exposure_s,
        reference_exposure_s=reference.exposure_s if reference is not None else None,
        actual_exposures_normalized=change.get('actual_exposures_normalized', False),
        relative_exposure_or_dose_unknown=change.get('relative_exposure_or_dose_unknown', True),
        current_photon_counts_known=bool(current.photon_counts_known),
        reference_photon_counts_known=bool(reference.photon_counts_known) if reference is not None else None,
        highq_interval_nm_inv=change['highq_interval_nm_inv'],
        highq_interval_adapted=False, platform_relative_slope_limit=change['platform_relative_slope_limit'],
        c=global_scale['c'] if global_scale is not None else None,
        c_numeric_available=bool(global_scale is not None and global_scale['c'] is not None),
        c_known=bool(global_scale is not None and global_scale['shape_condition_met']),
        platform_condition_met=bool(global_scale is not None and global_scale['shape_condition_met']),
        relative_slope=global_scale['relative_slope'] if global_scale is not None else None,
        comparable_highq_bins=global_scale['highq_comparable_ratio_q_bins'] if global_scale is not None else 0,
        c_conditional_se=global_scale.get('c_conditional_se') if global_scale is not None else None,
        c_formula=global_scale.get('formula') if global_scale is not None else None,
        observed_current_rate_q80=scale, delta_known_points=int(result['delta_known'].sum()),
        raw_known_points=int(result['raw_known'].sum()),
        signed_negative_points=int((result['raw_difference'][result['raw_known']] < 0).sum()),
        conditional_negative_points=int((result['conditional_difference'][result['conditional_known']] < 0).sum()),
        conditional_DB_views=conditional.get('summaries', []) if conditional is not None else [],
        shared_reference_windows_not_independent=True,
        covariance_given_fixed_scales='c_t*c_u*Var(reference rate); c estimation contributes shared covariance',
        DB_change_length_is_particle_size=False, metal_only_scattering_extracted=False,
        reference_known_zero_metal=False, conditional_parameters_calibrated=False,
        reference_noise_added=False, TEST_read=False)
    return dict(result, summary=summary)


def prepare(batch, *, enabled=None):
    'Separate NumPy encoder branch; disabled references cannot carry signal.'
    values = np.asarray(batch['delta'], float); known = np.asarray(batch['delta_known'])
    platform = np.asarray(batch['platform'], float); pknown = np.asarray(batch['platform_known'])
    if (values.ndim != 3 or values.shape[1:] != PROFILE_SHAPE or known.shape != values.shape
            or platform.shape != (len(values), 2) or pknown.shape != platform.shape
            or not np.isin(known, (0, 1)).all() or not np.isin(pknown, (0, 1)).all()):
        raise ValueError('Original signed-delta values and explicit presence masks required')
    known = known.astype(bool); pknown = pknown.astype(bool)
    if enabled is not None:
        enable = np.asarray(enabled)
        if enable.shape != (len(values),) or not np.isin(enable, (0, 1)).all():
            raise ValueError('One explicit reference-presence flag per row required')
        known &= enable[:, None, None].astype(bool); pknown &= enable[:, None].astype(bool)
    values = np.where(known, values, 0.); platform = np.where(pknown, platform, 0.)
    if not np.isfinite(values).all() or not np.isfinite(platform).all():
        raise ValueError('Known signed-delta inputs must be finite')
    vector = np.concatenate((values.reshape(len(values), -1), known.reshape(len(values), -1),
                             platform, pknown), axis=1).astype(np.float32)
    return dict(vector=vector, signed_profiles=values.astype(np.float32), known=known,
                schema=SCHEMA, input_spec=INPUT_SPEC)


class DeltaBank:
    'Small completed optional bank joined strictly by original family ID.'
    def __init__(self, path, *, expected_bank_plan_sha256):
        from .stage5_partial import sha
        self.root = Path(path).resolve()
        self.receipt = json.loads((self.root/'run.json').read_text(encoding='utf8'))
        side = (self.root/'run.json.sha256').read_text(encoding='ascii').split()[0]
        if (sha(self.root/'run.json') != side or self.receipt.get('schema') != SCHEMA
                or self.receipt.get('status') != 'completed'
                or self.receipt['identity']['bank_plan_sha256'] != expected_bank_plan_sha256
                or self.receipt.get('TEST_read') is not False):
            raise ValueError('Completed original-plan-bound delta bank required')
        if any(self.receipt['identity']['source_pins'].get(k)!=v for k,v in source_pins().items()):
            raise ValueError('Executing signed-delta definitions differ from the completed optional bank')
        payload = {k: [] for k in ('family_id',) + ARRAY_FIELDS}; seen = []
        for shard in self.receipt['shards']:
            file = (self.root/shard['path']).resolve()
            if self.root not in file.parents or sha(file) != shard['sha256']:
                raise ValueError('Completed delta shard changed')
            with np.load(file, allow_pickle=False) as arrays:
                ids = arrays['family_id']
                if ids.tolist() != shard['family_ids'] or np.any(ids % 10 == 9):
                    raise ValueError('Original TRAIN/CAL delta identities changed')
                seen.extend(map(int, ids))
                for key in payload: payload[key].append(np.array(arrays[key], copy=True))
        if seen != self.receipt['family_ids'] or len(set(seen)) != len(seen):
            raise ValueError('All original delta family slots must remain')
        self.arrays = {k: np.concatenate(v) for k, v in payload.items()}
        self.lookup = {fid: i for i, fid in enumerate(seen)}

    def for_families(self, family_ids, *, enabled=None):
        ids = [int(fid) for fid in family_ids]
        if any(fid % 10 == 9 or fid not in self.lookup for fid in ids):
            raise ValueError('Only recorded original TRAIN/CAL families may use delta inputs')
        indices = np.asarray([self.lookup[fid] for fid in ids], int)
        batch = {k: self.arrays[k][indices] for k in ('delta', 'delta_known', 'platform', 'platform_known')}
        return prepare(batch, enabled=enabled)
