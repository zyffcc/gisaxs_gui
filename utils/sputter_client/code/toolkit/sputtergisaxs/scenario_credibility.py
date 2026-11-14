'Actual four-scenario OOD, geometric-range and variant diagnostics.'
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from .resolvable import coverage, in_range
from .scenario_chart import CATALOG, SCENARIO_NAMES, decode
from .scenario_posterior import legal_from_metadata
from .scenario_runtime import (
    ScenarioRuntime, _array_sha, _frame_identity, _inside, _labels, _sha,
    _source_sha, _state_digest,
)
from .stage5_measurement import counts_profiles
from .stage5_posterior import ood_score
from .v2_input import HOST_CLASSES, INPUT_KEYS, prepare
from .v2_observation import frame_input, ppc_features


SCHEMA = 'four_scenario_credibility_diagnostic_1'
REGIMES = ('mandatory_only', 'recorded_host')
GROUPS = HOST_CLASSES + ('unknown',)
VARIANT_LIMIT = .15
MINIMUM_GROUP_DRAWS = 2
D_SEMANTIC_VERSION = 'conditional_primitive_density_or_correlation_1'
D_DENSITY_DEFINITION = ('conditional hex-equivalent metal-primitive centre density length '
                        'sqrt(2/(sqrt(3)*n)); not SEM nearest neighbour')
D_REPEAT_DEFINITION = ('conditional effective-column lateral correlation repeat; '
                       'independent of column centre density, not SEM nearest neighbour')
POLICY = dict(
    OOD='existing TRAIN fit and complete original CAL99; not p(x)',
    OOD_primary_threshold='overall CAL99 for the exact input-presence regime',
    OOD_auxiliary='original host CAL99, descriptive and never selected to obtain an inlier',
    optional='legacy none/host OOD; pre/thickness/QCM require the exact-presence supplemental artifact; OOD alone does not validate conditional parameter intervals',
    range='original D draws versus resolvable.coverage(public pixel-frame support)',
    range_domain='existing rule: outside probability >.5 OR median outside observed AF-band intervals',
    variant='posterior coarse/axis groups, each original member retained; at least two draws per group',
    spread='(max median - min median) / ((max median + min median)/2)',
    variant_limit=VARIANT_LIMIT,
    variant_qualification='same-definition original-draw groups at the user 15% gate; S-level calibration and S-level PPC are bound separately by release',
    no_original_draws_removed=True, no_prior_nuisance_redraw=True,
    TEST_read=False, structural_release=False)


def D_semantics(scene, records=()):
    'Clarify actual forward meanings without editing original decoder labels.'
    raw = sorted({r.get('conditional_labels', {}).get('D_definition')
                  for r in records if r.get('conditional_labels', {}).get('D_definition') is not None})
    return dict(version=D_SEMANTIC_VERSION, raw_D_definitions=raw,
        D_definition=D_DENSITY_DEFINITION if scene in ('S1','S2') else
                     D_REPEAT_DEFINITION if scene == 'S3' else 'not applicable to two-phase S4',
        centre_density_length_available=scene in ('S1','S2'),
        density_relation='n=2/(sqrt(3)*D^2)' if scene in ('S1','S2') else
                         'n=coverage/E(pi*Rfoot^2), independent of D' if scene == 'S3' else None,
        paracrystal_repeat_equals_density_length_assumption=scene in ('S1','S2'),
        scope='upstream conditional metal-primitive population, not all decorating metal centres'
              if scene == 'S2' else 'conditional metal-primitive population' if scene == 'S1' else
              'effective-column correlation repeat; not a metal-centre spacing' if scene == 'S3' else
              'two-phase morphology without island centres',
        SEM_nearest_neighbor_comparable=False, original_D_values_unchanged=True)


def _finite(value):
    return (not isinstance(value, (bool, np.bool_)) and np.isscalar(value)
            and isinstance(value, (int, float, np.number)) and np.isfinite(value))


def _same_array(left, right):
    a, b = np.asarray(left), np.asarray(right)
    return a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b, equal_nan=True)


def _spread(medians):
    a = np.asarray(medians, float)
    if len(a) < 2 or not np.isfinite(a).all() or np.any(a <= 0):
        return None
    return float((a.max()-a.min())/((a.max()+a.min())/2))


def _median_group(values):
    good = [float(v) for v in values if _finite(v) and v > 0]
    complete = len(good) == len(values)
    enough = len(values) >= MINIMUM_GROUP_DRAWS
    return dict(original_draws=len(values), resolved_original_draws=len(good),
                median_D_nm=float(np.median(good)) if complete and enough else None,
                minimum_original_draws=MINIMUM_GROUP_DRAWS,
                reason=None if complete and enough else
                ('insufficient_original_draws' if not enough else 'unresolved_original_D_draws'))


class ScenarioCredibility:
    'Bind one actual runtime to an optional completed OOD receipt.'
    def __init__(self, runtime, ood_run=None, *, optional_ood_run=None):
        if not isinstance(runtime, ScenarioRuntime):
            raise TypeError('Actual source-bound ScenarioRuntime required')
        self.runtime = runtime
        self.ood_path = None
        self.ood = None
        self.fits = {}
        self.ood_binding = None
        self.optional_ood = None
        if ood_run is not None:
            path = Path(ood_run).resolve()
            if path.is_dir():
                path /= 'run.json'
            self._load_ood(path)
        if optional_ood_run is not None:
            from .scenario_optional_credibility import load_optional_ood
            self._check_runtime()
            self.optional_ood=load_optional_ood(runtime,optional_ood_run)

    def _check_runtime(self):
        r = self.runtime
        if _sha(r.path) != r.receipt['weights_sha256']:
            raise ValueError('Actual selected weights changed')
        if _state_digest(r.net.state_dict()) != r.receipt['state_sha256']:
            raise ValueError('Actual runtime tensor state changed')
        for name, digest in dict(r.pins, **r.receipt['execution_source_pins']).items():
            relative = name if '/' in name else 'sputtergisaxs/'+name
            if _source_sha(_inside(r.package, relative)) != digest:
                raise ValueError('Actual runtime source changed: '+relative)

    def _load_ood(self, path):
        self._check_runtime()
        digest = _sha(path)
        value = json.loads(path.read_text(encoding='utf8'))
        identity = value.get('identity', {})
        r = self.runtime
        if (value.get('status') != 'completed_OOD_diagnostic'
                or identity.get('schema') != 'four_scenario_embedding_OOD_diagnostic_1'
                or identity.get('runtime') != r.receipt
                or identity.get('bank') != r.declaration['bank']
                or identity.get('regimes') != list(REGIMES)
                or identity.get('groups') != list(GROUPS)
                or identity.get('training_optional_policy') != r.declaration['optional_policy']
                or identity.get('TEST_scientific_rows_decoded') is not False
                or value.get('TEST_scientific_rows_decoded') is not False
                or identity.get('p_x_estimated') is not False
                or value.get('p_x_estimated') is not False
                or identity.get('independent_OOD_validation_performed') is not False
                or value.get('independent_OOD_validation_performed') is not False):
            raise ValueError('Actual complete non-TEST OOD/runtime/bank regime binding required')
        pins = identity.get('source_pins', {})
        if any(pins.get(k) != v for k, v in r.pins.items()):
            raise ValueError('Original OOD generating/training source differs')
        for name in ('scripts/v2_scenario_ood.py', 'sputtergisaxs/stage5_posterior.py',
                     'sputtergisaxs/scenario_runtime.py', 'sputtergisaxs/scenario_posterior.py'):
            if name not in pins:
                raise ValueError('Original OOD fit/scoring source not bound: '+name)
        for name, source_digest in pins.items():
            if _source_sha(_inside(r.package, name)) != source_digest:
                raise ValueError('Actual OOD source changed: '+name)
        if identity.get('source_sha256') != pins['scripts/v2_scenario_ood.py']:
            raise ValueError('Actual OOD producer source identity differs')
        ids = identity.get('original_family_ids', {})
        records = value.get('records', [])
        if set(ids) != {'TRAIN', 'CAL'} or not records:
            raise ValueError('Complete original TRAIN/CAL OOD IDs required')
        for split in ('TRAIN', 'CAL'):
            if any(isinstance(fid, bool) or not isinstance(fid, int) or fid < 0
                   or not (fid % 10 < 8 if split == 'TRAIN' else fid % 10 == 8)
                   for fid in ids[split]):
                raise ValueError('OOD metadata contains a wrong split or TEST ID')
        expected = ids['TRAIN']+ids['CAL']
        if (len(set(expected)) != len(expected) or len(records) != len(expected)
                or any(sorted(rec.get('family_id') for rec in records if rec.get('split') == split)
                       != sorted(ids[split]) for split in ('TRAIN', 'CAL'))
                or any(rec.get('index') != rec['family_id']
                       or rec.get('split') != ('TRAIN' if i < len(ids['TRAIN']) else 'CAL')
                       or rec.get('host_category') not in GROUPS
                       or rec.get('scenario') not in range(4)
                       for i, rec in enumerate(records))):
            raise ValueError('Completed OOD original IDs/order/denominators differ')


        record_ids = [rec['family_id'] for rec in records]
        artifact = _inside(path.parent, value['diagnostic_file'])
        if _sha(artifact) != value['diagnostic_sha256']:
            raise ValueError('Actual completed OOD diagnostic SHA differs')

        with np.load(artifact, allow_pickle=False) as archive:
            fam = archive['family_id']; splits = archive['split_index']; hosts = archive['host_index']
            if (fam.dtype.kind not in 'iu' or splits.dtype.kind not in 'iu' or hosts.dtype.kind not in 'iu'
                    or fam.tolist() != record_ids
                    or splits.tolist() != [0 if rec['split'] == 'TRAIN' else 1 for rec in records]
                    or hosts.tolist() != [GROUPS.index(rec['host_category']) for rec in records]):
                raise ValueError('Actual OOD array row identity differs')
            scores = archive['score']; valid = archive['valid_score']
            if (scores.shape != (len(records), len(REGIMES)) or valid.shape != scores.shape
                    or not np.array_equal(valid, np.isfinite(scores))):
                raise ValueError('Actual OOD score completeness differs')
            train = splits == 0; cal = splits == 1
            for i, regime in enumerate(REGIMES):
                report = value['reports'][regime]
                if (report.get('original_TRAIN_rows') != int(train.sum())
                        or report.get('original_CAL_rows') != int(cal.sum())):
                    raise ValueError('Actual OOD report original denominator differs')
                scopes = [('overall', cal, report['overall'])]
                scopes += [(g, cal & (hosts == GROUPS.index(g)), report['host_groups'][g])
                           for g in GROUPS]
                for name, sel, scope in scopes:
                    cut = scope['own_CAL_threshold']; count = int(sel.sum())
                    complete = count >= 30 and np.isfinite(scores[sel, i]).all()
                    expected_cut = float(np.quantile(scores[sel, i], .99)) if complete else None
                    if (cut.get('original_CAL_rows') != count
                            or cut.get('valid_CAL_scores') != int(np.isfinite(scores[sel, i]).sum())
                            or cut.get('minimum_CAL') != 30 or cut.get('value') != expected_cut
                            or scope.get('grouping_qualified_for_threshold') != (expected_cut is not None)):
                        raise ValueError('Original '+regime+'/'+name+' CAL99 threshold differs')
                if report['TRAIN_fit_available']:
                    if not np.isfinite(scores[train, i]).all() or train.sum() < 2:
                        raise ValueError('OOD TRAIN fit uses an incomplete original denominator')
                    mean = archive[regime+'__mean'].copy()
                    precision = archive[regime+'__precision'].copy()
                    ridge = float(archive[regime+'__ridge'])
                    dim = r.declaration.get('embedding_dim', r.declaration['hidden'])
                    if (mean.shape != (dim,) or precision.shape != (dim, dim)
                            or not np.isfinite(mean).all() or not np.isfinite(precision).all()
                            or not np.isfinite(ridge) or ridge <= 0):
                        raise ValueError('Actual OOD TRAIN fit dimension/finite parameters differ')
                    self.fits[regime] = dict(mean=mean, precision=precision, ridge=ridge)
        if _sha(path) != digest or _sha(artifact) != value['diagnostic_sha256']:
            raise ValueError('OOD receipt/artifact changed during binding')
        self.ood_path = path; self.ood = value
        self.ood_binding = dict(run_sha256=digest, diagnostic_sha256=value['diagnostic_sha256'],
            source_pins=pins, original_TRAIN_rows=len(ids['TRAIN']), original_CAL_rows=len(ids['CAL']),
            independent_OOD_validation_performed=False, TEST_scientific_rows_decoded=False)

    def _input_and_verify(self, full, public, sampled, optional):
        import torch
        self._check_runtime()
        r = self.runtime
        if (sampled.get('receipt') != r.receipt
                or sampled.get('frame_identity') != _frame_identity(full, public)
                or sampled.get('conditioned_outside_D') is not False
                or sampled.get('nuisance_prior_resampled') is not False
                or sampled.get('scenarios_pruned') is not False):
            raise ValueError('Original runtime/frame/unconditioned joint receipt differs')
        geometry = sampled['geometry']; n = sampled['original_draws_per_applicable_scenario']
        seed = sampled['original_seed']
        if (not isinstance(n, Integral) or isinstance(n, (bool, np.bool_)) or n < 2
                or not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0
                or set(sampled['scenarios']) != set(SCENARIO_NAMES)):
            raise ValueError('Original scenario/seed/draw count differs')
        frame = frame_input(full, public, geometry['energy_ev'], geometry['alpha_i_deg'],
                            geometry['metal'], **optional)
        if frame['geometry'] != geometry:
            raise ValueError('Original mandatory geometry differs')
        observed = ppc_features(public, estimate_stat_se=sampled['frame_identity']['photon_statistics_known'])
        if observed != sampled.get('observed'):
            raise ValueError('Original observed features or selected AF band changed')
        arrays = {k:np.stack([frame[k]]) for k in INPUT_KEYS}
        prepared = prepare(arrays, r.mode)
        encoded = {k:torch.as_tensor(prepared[k], device=r.device)
                   for k in ('vector', 'image') if prepared[k] is not None}
        legal = legal_from_metadata(geometry['energy_ev'], geometry['alpha_i_deg'], r.templates,
            host_categories=[optional['host_category']], device=r.device)
        generator = torch.Generator(device=r.device).manual_seed(int(seed))
        replay = r.net.sample_all_applicable(encoded, int(n), legal_templates=legal,
            generator=generator, host_categories=[optional['host_category']])
        hashes = {}
        for scene in SCENARIO_NAMES:
            old = sampled['scenarios'][scene]; new = replay['samples'][scene]
            if (old['applicable'] != bool(new['applicable'][0])
                    or old['not_applicable_reason'] != new['not_applicable_reason'][0]):
                raise ValueError('Original optional context or scenario applicability changed')
            joint = old['original_joint']
            if set(joint) != {'z', 'coarse', 'template', 'component'}:
                raise ValueError('All original joint coordinates/classes/components required')
            for key in joint:
                if not _same_array(joint[key], new[key][0].detach().cpu().numpy()):
                    raise ValueError('Original optional context/joint draw differs: '+scene+'/'+key)
            count = int(n) if old['applicable'] else 0
            if (old['original_draws'] != count or len(old['draws']) != count
                    or [rec.get('index') for rec in old['draws']] != list(range(count))):
                raise ValueError('Original draw indices/denominator changed')
            for i, rec in enumerate(old['draws']):
                try:
                    row = decode(joint['z'][i], int(joint['coarse'][i]), int(joint['template'][i]),
                        metal=geometry['metal'], E=geometry['energy_ev'], ai=geometry['alpha_i_deg'],
                        template_keys=r.keys, template_metadata=r.templates)
                except (ValueError, OverflowError, FloatingPointError):
                    if rec.get('decode_failed') is not True or rec.get('physical') is not None:
                        raise ValueError('Original decoder failure status changed')
                else:
                    if (rec.get('decode_failed') is not False or rec.get('physical') != row
                            or rec.get('conditional_labels') != _labels(row) or row['scenario'] != scene):
                        raise ValueError('Original decoded physical labels changed')
            hashes[scene] = {key:_array_sha(val) for key, val in joint.items()}
        if _frame_identity(full, public) != sampled['frame_identity']:
            raise ValueError('Original frame changed during verification')
        expected_legal = legal[0].detach().cpu().numpy().any(axis=1)
        return frame, encoded, expected_legal, hashes

    def _ood(self, frame, encoded):
        import torch
        flags = np.asarray(frame['known'], bool)
        regime = 'recorded_host' if flags[0] else 'mandatory_only'
        host = HOST_CLASSES[int(frame['optional'][0])] if flags[0] else 'unknown'
        result = dict(score=None, OOD=None, threshold=None, input_regime=regime,
            host_category=host, host_threshold=None, host_OOD=None,
            threshold_status='missing', threshold_available=False,
            input_regime_matched=False, score_bound=False, OOD_qualified=False,
            independent_OOD_validation_performed=False, p_x_estimated=False,
            structural_release=False, missing_reason=None)
        if flags[1:].any() or (self.ood is None and self.optional_ood is not None):
            from .scenario_optional_credibility import score_optional_ood
            return score_optional_ood(self.runtime,frame,encoded,self.optional_ood,result)
        if self.ood is None:
            result['missing_reason'] = 'Completed matched TRAIN/CAL OOD artifact not supplied'
            return result
        artifact = _inside(self.ood_path.parent, self.ood['diagnostic_file'])
        if (_sha(self.ood_path) != self.ood_binding['run_sha256']
                or _sha(artifact) != self.ood_binding['diagnostic_sha256']):
            raise ValueError('Bound OOD artifact changed before scoring')
        report = self.ood['reports'][regime]
        cut = report['overall']['own_CAL_threshold']
        local = report['host_groups'][host]['own_CAL_threshold']
        result.update(threshold=cut['value'], threshold_status=cut['status'],
                      threshold_available=cut['value'] is not None, input_regime_matched=True,
                      host_threshold=local['value'], host_threshold_status=local['status'])
        if regime not in self.fits:
            result['missing_reason'] = report['fit_missing_reason']
            return result
        with torch.no_grad():
            embedding = self.runtime.net.embedding(encoded).detach().cpu().numpy()
        if not np.isfinite(embedding).all():
            result['missing_reason'] = 'Actual inference embedding is nonfinite'
            return result
        score = float(ood_score(embedding.astype(float), self.fits[regime])[0])
        if not np.isfinite(score):
            result['missing_reason'] = 'Actual inference OOD score is nonfinite'
            return result
        result.update(score=score, OOD=bool(score > cut['value']) if cut['value'] is not None else None,
            host_OOD=bool(score > local['value']) if local['value'] is not None else None,
            embedding_sha256=_array_sha(embedding), score_bound=True,
            missing_reason=None if cut['value'] is not None else 'Complete original CAL99 threshold unavailable')
        return result

    def _range(self, scene, part, observed, cov):
        if not part['applicable'] or scene == 'S4':
            return dict(status='not_applicable', original_draws=part['original_draws'],
                outside_range=[None]*part['original_draws'], outside_probability=None, outside_probability_bounds=None,
                domain_out_of_range=None, structural_range_verified=False,
                reason=part['not_applicable_reason'] if not part['applicable'] else 'S4 has no island D')
        intervals = cov['D_corr_intervals_nm']
        values = [rec.get('conditional_labels', {}).get('D_nm') for rec in part['draws']]
        flags = [not bool(in_range(v, intervals)) if _finite(v) and v > 0 else None for v in values]
        n = len(flags); known = sum(v is not None for v in flags); outside = sum(v is True for v in flags)
        complete = known == n
        p = outside/n if complete else None
        median = float(np.median(values)) if complete else None
        band = observed.get('AF_band')
        band_intervals = cov['bands'][int(band)]['D_corr_intervals_nm'] if band in range(4) else []
        band_flag = not bool(in_range(median, band_intervals)) if median is not None else None
        domain = bool(p > .5 or band_flag) if p is not None else None
        return dict(status='geometric_search_domain_diagnostic', original_draws=n,
            resolved_original_draws=known, unresolved_original_draws=n-known,
            outside_range=flags, outside_probability=p,
            outside_probability_bounds=[outside/n, (outside+n-known)/n],
            median_D_nm=median, observed_AF_band=band,
            observed_band_intervals_nm=band_intervals, median_outside_observed_band=band_flag,
            domain_out_of_range=domain, threshold_outside_probability=.5,
            coverage_definition=cov['definition'], structural_range_verified=False,
            host_peak_is_not_metal_D=scene == 'S2',
            reason=('Geometric search convention only; requires independent recovery and PPC'
                    if complete else 'Original decoder/quantity failures retained; no subset probability'))

    def _variants(self, scene, part, expected_legal):
        if not part['applicable'] or scene == 'S4':
            return dict(status='not_applicable', original_draws=part['original_draws'],
                relative_spread=None, threshold=VARIANT_LIMIT, diagnostic_pass=None, qualified=None,
                reason=part['not_applicable_reason'] if not part['applicable'] else 'S4 has no island D')
        expected = [dict(c) for c in CATALOG if c['scenario'] == scene and expected_legal[c['id']]]
        joint = part['original_joint']; coarse = np.asarray(joint['coarse'])
        classes = []
        definitions = {}
        for c in expected:
            records = [rec for i, rec in enumerate(part['draws']) if coarse[i] == c['id']]
            labels = [rec.get('conditional_labels', {}) for rec in records]
            defs = sorted(set(x.get('D_definition') for x in labels if x.get('D_definition') is not None))
            declared = ('conditional hex-equivalent centre density length; not SEM nearest neighbour'
                        if c['structure_factor'] == 'harddisk' else
                        'conditional lateral correlation step, not certified 2D nearest neighbour')
            if any(d != declared for d in defs):
                raise ValueError('Original variant D definition differs from its physical catalog')
            canonical = D_semantics(scene)['D_definition']
            item = dict(c, raw_D_definition=declared, D_definition=canonical,
                        **_median_group([x.get('D_nm') for x in labels]))
            classes.append(item)
            definitions.setdefault(canonical, []).append(item)
        panels = []
        for definition, items in definitions.items():
            medians = [item['median_D_nm'] for item in items if item['median_D_nm'] is not None]
            spread = _spread(medians)
            axes = {}
            for axis in ('shape', 'structure_factor', 'decoupling', 'layer', 'step_count'):
                groups = []
                axis_values = list(dict.fromkeys(item[axis] for item in items))
                for value in axis_values:
                    ids = [item['id'] for item in items if item[axis] == value]
                    original = [rec.get('conditional_labels', {}).get('D_nm')
                                for i, rec in enumerate(part['draws']) if coarse[i] in ids]
                    groups.append(dict(value=value, coarse_ids=ids, **_median_group(original)))
                grouped_spread = _spread([g['median_D_nm'] for g in groups if g['median_D_nm'] is not None])
                axes[axis] = dict(groups=groups, relative_spread=grouped_spread,
                    all_expected_groups_resolved=all(g['median_D_nm'] is not None for g in groups),
                    threshold_pass_for_resolved_groups=(grouped_spread <= VARIANT_LIMIT
                                                       if grouped_spread is not None else None))
            panels.append(dict(D_definition=definition, coarse_ids=[v['id'] for v in items],
                relative_spread=spread, resolved_classes=len(medians), expected_classes=len(items),
                all_expected_classes_resolved=len(medians) == len(items),
                threshold_pass_for_resolved_classes=spread <= VARIANT_LIMIT if spread is not None else None,
                axes=axes))
        same_definition = len(panels) == 1
        whole_spread = panels[0]['relative_spread'] if same_definition else None


        any_failed = any(p['relative_spread'] is not None and p['relative_spread'] > VARIANT_LIMIT
                         for p in panels)
        complete = bool(panels) and all(p['all_expected_classes_resolved']
                                       and p['relative_spread'] is not None for p in panels)
        diagnostic_pass = False if any_failed else (True if complete else None)
        missing = []
        if not complete:
            missing.append('Some legal variant groups have missing or fewer than two original draws, or no comparable variant pair')
        return dict(status='posterior_variant_group_diagnostic', original_draws=part['original_draws'],
            classes=classes, definition_panels=panels, threshold=VARIANT_LIMIT,
            relative_spread=whole_spread, cross_definition_comparison_performed=False,
            all_expected_classes_resolved=bool(classes) and all(c['median_D_nm'] is not None for c in classes),
            threshold_pass_for_resolved_classes=(whole_spread <= VARIANT_LIMIT
                                                if whole_spread is not None else None),
            diagnostic_pass=diagnostic_pass,
            diagnostic_status='failed' if diagnostic_pass is False else 'passed' if diagnostic_pass is True else 'unknown',
            grouped_from_original_joint=True, independent_optimizer_refit_required=False,
            per_coarse_calibration_required=False, per_coarse_PPC_required=False, qualified=None,
            qualification_owner='release binds original S-level independent calibration and S-level PPC',
            missing_reasons=missing,
            D_semantics=D_semantics(scene, part['draws']),
            definition_notes=(['S1/S2 SF variants share the actual primitive centre-density length; original raw labels retained; paracrystal also assumes equal radial repeat']
                if scene in ('S1','S2') else ['S3 correlation repeat does not determine column centre density']),
            reason='Registered original-draw variant diagnostic; public release additionally binds S-level calibration and PPC')

    def assess(self, full, public, sampled, *, pre=None, host_category=None,
               thickness_nm=None, qcm_rate=None):
        'Diagnose exactly the supplied original joint pool, never replace it.'
        optional = dict(pre=pre, host_category=host_category,
                        thickness_nm=thickness_nm, qcm_rate=qcm_rate)
        frame, encoded, legal, hashes = self._input_and_verify(full, public, sampled, optional)
        _, hn, _, _, _, _ = counts_profiles(public.Y, public.P)
        cov = coverage(hn)


        observed = sampled['observed']
        scenes = {scene:dict(applicable=sampled['scenarios'][scene]['applicable'],
            joint_sha256=hashes[scene],
            range=self._range(scene, sampled['scenarios'][scene], observed, cov),
            variants=self._variants(scene, sampled['scenarios'][scene], legal))
            for scene in SCENARIO_NAMES}
        inputs = {k:_array_sha(frame[k]) for k in INPUT_KEYS}
        return dict(schema=SCHEMA, policy=dict(POLICY), runtime_receipt=dict(self.runtime.receipt),
            frame_identity=dict(sampled['frame_identity']), geometry=dict(sampled['geometry']),
            optional_presence=np.asarray(frame['known'], bool).tolist(), input_sha256=inputs,
            original_seed=sampled['original_seed'],
            original_draws_per_applicable_scenario=sampled['original_draws_per_applicable_scenario'],
            joint_sha256=hashes, deterministic_context_verification=True,
            context_verification_scope='same-seed original-pool compatibility; original sampler input SHA was not recorded',
            original_optional_input_identity_independently_bound=False,
            original_scientific_pool_replaced=False, OOD_binding=self.ood_binding,
            optional_OOD_binding=self.optional_ood['binding'] if self.optional_ood is not None else None,
            OOD=self._ood(frame, encoded), coverage=cov, scenarios=scenes,
            helper_source_sha256=_source_sha(__file__),
            structural_release=False,
            missing_release_evidence=['Quantity-specific simulation calibration',
                                      'Original-draw PPC', 'Qualified model-variant stability'])
