'Source-bound four-scenario research runtime; no automatic structure release.'
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from .scenario_chart import CATALOG, COORDINATES, SCHEMA, SCENARIO_NAMES, decode
from .scenario_posterior import ScenarioCandidate, legal_from_metadata
from .v2_input import INPUT_KEYS, MODES, prepare
from .v2_observation import CountFrame, frame_input, ppc_features
from .v2_ppc import GATES, evaluate


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sha(path):
    path = Path(path); raw = path.read_bytes()
    return hashlib.sha256(raw.replace(b'\r\n', b'\n') if path.suffix == '.py' else raw).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _inside(root, relative):
    root = Path(root).resolve(); path = (root/relative).resolve()
    if root not in path.parents:
        raise ValueError('Bound source/asset path escapes its declared root')
    return path


def _state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _array_sha(value):
    a = np.ascontiguousarray(value)
    digest = hashlib.sha256(str((a.dtype, tuple(a.shape))).encode())
    digest.update(a.tobytes())
    return digest.hexdigest()


def _frame_identity(full, public):
    if not isinstance(full, CountFrame) or not isinstance(public, CountFrame):
        raise TypeError('Original CountFrame full/public observations required')
    return dict(full_Y=_array_sha(full.Y), full_P=_array_sha(full.P),
                public_Y=_array_sha(public.Y), public_P=_array_sha(public.P),
                photon_statistics_known=bool(full.photon_counts_known and public.photon_counts_known))


def _public_features(observed):
    values = observed['values']; q = values['q_star']; w = values['FWHM']
    return dict(D_corr_nm=2*np.pi/q if q is not None and q > 0 else None,
        width_rel=w/q if q is not None and q > 0 and w is not None else None,
        Yoneda_alpha_f_deg=values['Yoneda'], q_star_nm_inv=q, FWHM_nm_inv=w,
        missing_reasons=dict(D_corr=observed['feature_missing_reasons']['q_star'],
            width_rel=observed['feature_missing_reasons']['FWHM'],
            Yoneda=observed['feature_missing_reasons']['Yoneda']),
        empirical_Yoneda_not_unique_critical_angle=True,
        no_structure_model_used=True)


def _labels(physical):
    scene = physical['scenario']
    result = dict(scenario=scene, D_nm=physical.get('D_nm'),
        diameter_nm=physical.get('diameter_nm'), height_nm=physical.get('height_nm'),
        gap_nm=None, coverage_fraction=physical.get('coverage_fraction'),
        D_definition=physical.get('D_definition'),
        diameter_definition=physical.get('diameter_definition'),
        coverage_definition=physical.get('coverage_definition'),
        gap_missing_reason=physical.get('gap_missing_reason', 'no validated centre configuration'),
        validated=False, structural_release=False)
    if physical['structure_factor'] == 'harddisk':
        result['D_definition'] = 'conditional hex-equivalent centre density length; not SEM nearest neighbour'
    if scene == 'S3':
        result.update(height_nm=None, depth_interval_nm=physical['depth_interval_nm'],
            depth_span_nm=physical['depth_span_nm'], exposed_span_nm=physical['exposed_span_nm'],
            burial_span_nm=physical['burial_span_nm'],
            height_missing_reason='step-column depth support is not an island height')
    elif scene == 'S4':
        result.update(D_nm=None, diameter_nm=None, height_nm=None,
            merged_phase_assumed=True, connected_topology_verified=False,
            island_quantity_missing_reason='not applicable to effective two-phase S4')
    elif scene == 'S2':
        result['host_peak_not_metal_D'] = True
    return result


def _conditional_summaries(scene, records):
    names = ('D_nm', 'diameter_nm', 'height_nm', 'gap_nm', 'coverage_fraction',
             'depth_span_nm', 'exposed_span_nm', 'burial_span_nm')
    output = {}
    for name in names:
        values = [r.get('conditional_labels', {}).get(name) for r in records]
        good = [float(v) for v in values if v is not None and np.isscalar(v) and np.isfinite(v)]
        item = dict(median=None, interval68=None, interval95=None,
                    original_draws=len(records), resolved_original_draws=len(good),
                    calibrated=False, structural_release=False)
        if good and len(good) == len(records):
            bounds = np.quantile(good, [.025, .16, .5, .84, .975])
            item.update(median=float(bounds[2]), interval68=bounds[[1, 3]].tolist(),
                        interval95=bounds[[0, 4]].tolist(), reason='uncalibrated conditional research interval')
        else:
            item['reason'] = ('not applicable to this conditional model' if not good else
                              'original draws missing; no interval from a retained subset')
        output[name] = item
    return output


class ScenarioRuntime:
    'Load actual best.pt, retain every applicable conditional hypothesis.'
    def __init__(self, checkpoint, *, device='cpu', package_root=None):
        import torch
        self.path = Path(checkpoint).resolve()
        self.package = (Path(__file__).resolve().parent.parent if package_root is None
                        else Path(package_root).resolve())
        side = self.path.with_name(self.path.name+'.sha256')
        digest = side.read_text(encoding='ascii').split()[0]
        if _sha(self.path) != digest:
            raise ValueError('Selected checkpoint SHA differs from its sidecar')
        try:
            saved = torch.load(self.path, map_location='cpu', weights_only=True)
        except TypeError as error:
            raise RuntimeError('This runtime requires torch safe weights_only loading; no unrestricted pickle fallback') from error
        if _sha(self.path) != digest:
            raise ValueError('Checkpoint changed during load; retry the stable selected checkpoint')
        d = saved['declaration']; bank = d['bank']
        if (d.get('schema') != 'four_scenario_training_1' or d.get('fixture') is not False
                or d.get('TEST_inspected') is not False or bank.get('fixture') is not False
                or bank.get('TEST_scientific_rows_decoded') is not False
                or d['physics_schema'] != SCHEMA or d['mode'] not in MODES):
            raise ValueError('Actual source-bound non-TEST four-scenario training declaration required')
        if (d['dim'] != len(d['coordinates']) or d['dim'] != d['continuous_dim']
                or d['coordinates'] != list(COORDINATES)
                or d['catalog'] != list(CATALOG) or d['coarse_classes'] != len(d['catalog'])):
            raise ValueError('Bound chart/catalog/continuous coordinate declarations differ')
        self.templates = tuple(d['template_metadata']); self.keys = tuple(d['template_keys'])
        if (self.keys != tuple(t['key'] for t in self.templates) or len(set(self.keys)) != len(self.keys)
                or d['template_classes'] != len(self.keys)+1
                or any(t.get('split') != 'TRAIN' for t in self.templates)):
            raise ValueError('Unique source-bound TRAIN template catalog required')
        self.pins = dict(bank['source_pins'])
        for relative, value in d['source_pins'].items():
            if relative in self.pins and self.pins[relative] != value:
                raise ValueError('Training/generating source identity conflict')
            self.pins[relative] = value
        for relative, value in self.pins.items():
            if _source_sha(_inside(self.package, relative)) != value:
                raise ValueError('Actual checkpoint source differs: '+relative)
        if _state_digest(saved['model']) != saved['state_sha256']:
            raise ValueError('Selected checkpoint tensor state differs')
        if (isinstance(saved['epoch'], bool) or not isinstance(saved['epoch'], int)
                or saved['epoch'] < 0 or not np.isfinite(saved['CAL_NLL'])):
            raise ValueError('Selected complete CAL epoch required')
        self.declaration = d; self.device = torch.device(device); self.mode = d['mode']
        self.net = ScenarioCandidate(self.mode, d['dim'], d['coarse_classes'], d['template_classes'],
            hidden=d['hidden'], components=d['components'], catalog=d['catalog'],
            bank_scenario_prior=d['bank_scenario_prior']).to(self.device)
        self.net.load_state_dict(saved['model'], strict=True); self.net.eval()
        self.receipt = dict(schema='four_scenario_research_runtime_1', weights_sha256=digest,
            state_sha256=saved['state_sha256'], selected_epoch=saved['epoch'], CAL_NLL=saved['CAL_NLL'],
            plan_sha256=bank['plan_sha256'], input_mode=self.mode, training_phase=d['phase'],
            final_family_target_met=d['final_family_target_met'],
            execution_source_pins={name:_source_sha(Path(__file__).parent/name)
                                   for name in ('scenario_runtime.py', 'scenario_posterior.py')},
            calibrated=False, OOD_qualified=False, structural_release=False)

    def sample(self, full, public, energy_ev, alpha_i_deg, metal, *, pre=None,
               host_category=None, thickness_nm=None, qcm_rate=None, n=256, seed=672720261018):
        'Features and unvalidated fixed-S draws; optional inputs may be absent.'
        import torch
        if (not isinstance(n, Integral) or isinstance(n, (bool, np.bool_)) or n < 2
                or not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0):
            raise ValueError('At least two original draws and a nonnegative integer seed required')
        frame = frame_input(full, public, energy_ev, alpha_i_deg, metal, pre=pre,
            host_category=host_category, thickness_nm=thickness_nm, qcm_rate=qcm_rate)
        identity = _frame_identity(full, public)
        observed = ppc_features(public, estimate_stat_se=identity['photon_statistics_known'])
        batch = {k:np.stack([frame[k]]) for k in INPUT_KEYS}
        x = prepare(batch, self.mode)
        encoded = {k:torch.as_tensor(x[k], device=self.device) for k in ('vector', 'image') if x[k] is not None}
        legal = legal_from_metadata(energy_ev, alpha_i_deg, self.templates,
                                   host_categories=[host_category], device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        sampled = self.net.sample_all_applicable(encoded, int(n), legal_templates=legal,
            generator=generator, host_categories=[host_category])
        results = {}
        for scene in SCENARIO_NAMES:
            part = sampled['samples'][scene]; applicable = bool(part['applicable'][0])
            original = {k:part[k][0].detach().cpu().numpy() for k in ('z', 'coarse', 'template', 'component')}
            records = []
            if applicable:
                for i in range(int(n)):
                    try:
                        row = decode(original['z'][i], int(original['coarse'][i]), int(original['template'][i]),
                            metal=metal, E=energy_ev, ai=alpha_i_deg, template_keys=self.keys,
                            template_metadata=self.templates)
                        if row['scenario'] != scene:
                            raise ValueError('Original fixed-S class changed its physical scenario')
                        records.append(dict(index=i, decode_failed=False, physical=row,
                                            conditional_labels=_labels(row)))
                    except (ValueError, OverflowError, FloatingPointError) as error:
                        records.append(dict(index=i, decode_failed=True, reason=str(error),
                                            physical=None, conditional_labels={}))
            results[scene] = dict(applicable=applicable,
                not_applicable_reason=part['not_applicable_reason'][0], original_draws=len(records),
                original_joint=original, draws=records,
                conditional_quantities=_conditional_summaries(scene, records),
                public_quantities=dict(D_nm=None, diameter_nm=None, height_nm=None, gap_nm=None,
                                       coverage_fraction=None),
                missing_reason='No source-bound simulation calibration/OOD/per-frame release receipt',
                calibrated=False, structural_release=False)
        probabilities = sampled['scenario_comparison']['probabilities'][0].detach().cpu().numpy()
        return dict(receipt=dict(self.receipt), geometry=dict(frame['geometry']), frame_identity=identity,
            observed=observed, features=_public_features(observed), scenarios=results,
            reference_scenario_probabilities=dict(zip(SCENARIO_NAMES, probabilities.tolist())),
            scenario_probabilities_are_reference_only=True, scenarios_pruned=False,
            original_seed=int(seed), original_draws_per_applicable_scenario=int(n),
            conditioned_outside_D=False, nuisance_prior_resampled=False,
            calibrated=False, structural_release=False)

    def posterior_predictive(self, sampled, full, public, *, bank_plan, feature_certificate,
                             host_shapes=None, host_metadata=None, harddisk=None,
                             backend='cpu', seed=672820261018):
        'One forward/count replica per original draw, with original public P.'
        from .scenario_forward import mean_forward, harddisk_table, QUICK
        if sampled['receipt'] != self.receipt or _frame_identity(full, public) != sampled['frame_identity']:
            raise ValueError('Original posterior weights or full/public frame changed')
        if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0:
            raise ValueError('Nonnegative original replica seed required')
        if backend not in ('cpu', 'cuda'):
            raise ValueError('Explicit CPU/CUDA predictive backend required')
        if not sampled['frame_identity']['photon_statistics_known']:
            return dict(status='not_ready_unknown_photon_statistics', features=sampled['features'],
                scenarios={s:dict(status='not_verified_unknown_photon_statistics', structural_release=False)
                           for s in SCENARIO_NAMES},
                missing_reason='Detector units are not established photon counts; no Poisson replicas/statistical gate',
                calibrated=False, structural_release=False)
        plan_path = Path(bank_plan).resolve(); plan = _json(plan_path)
        bank = self.declaration['bank']; generation_pins = bank['source_pins']
        for relative, value in generation_pins.items():
            if _source_sha(_inside(self.package, relative)) != value:
                raise ValueError('Original predictive generating source changed: '+relative)
        if (_sha(plan_path) != bank['plan_sha256'] or plan['source_pins'] != generation_pins
                or plan['catalog'] != self.declaration['catalog']
                or plan['coordinates'] != self.declaration['coordinates']):
            raise ValueError('Original generating plan/source/chart binding differs')
        certificate_path = Path(feature_certificate).resolve(); cert = _json(certificate_path)
        if (_sha(certificate_path) != bank['certificate_sha256']
                or cert.get('schema') != 'simple_scenario_feature_certificate_1'
                or cert.get('passed') is not True or cert.get('TEST_decoded') is not False
                or cert.get('quick') != QUICK or cert['quick'] != plan['quick_recipe']
                or any(cert.get('source_pins', {}).get(k) != v for k, v in generation_pins.items())):
            raise ValueError('Actual source-bound schedule feature certificate required')
        def bound(name, supplied):
            item = plan['bindings'][name]
            path = _inside(plan_path.parent, item['path']) if supplied is None else Path(supplied).resolve()
            if _sha(path) != item['sha256']:
                raise ValueError('Actual TRAIN asset changed: '+name)
            return path
        hp = bound('host_shapes.npz', host_shapes); hm = bound('host_shapes.json', host_metadata)
        dp = bound('harddisk.npz', harddisk)
        if tuple(_json(hm)['rows']) != self.templates:
            raise ValueError('Original normalized TRAIN template metadata differs')
        observed = sampled['observed']; band = observed['AF_band']; output = {}
        with np.load(hp, allow_pickle=False) as hostarr, np.load(dp, allow_pickle=False) as diskarr:
            disk = harddisk_table(diskarr)
            for scene_index, scene in enumerate(SCENARIO_NAMES):
                part = sampled['scenarios'][scene]
                if not part['applicable']:
                    output[scene] = dict(status='not_applicable', reason=part['not_applicable_reason'],
                                         structural_release=False)
                    continue
                replicas = []; original = part['original_joint']
                for i, record in enumerate(part['draws']):
                    try:
                        if record['decode_failed']:
                            raise ValueError('Original joint decoder failure: '+record['reason'])
                        physical = decode(original['z'][i], int(original['coarse'][i]), int(original['template'][i]),
                            metal=sampled['geometry']['metal'], E=sampled['geometry']['energy_ev'],
                            ai=sampled['geometry']['alpha_i_deg'], template_keys=self.keys,
                            template_metadata=self.templates)
                        if physical != record['physical'] or physical['scenario'] != scene:
                            raise ValueError('Original joint physical draw changed')
                        rate, _, receipt = mean_forward(physical, full.P, hostarr, recipe=QUICK,
                                                        backend=backend, disk=disk)
                        mu = np.asarray(rate, float)*public.P
                        if not np.isfinite(mu).all() or np.any(mu < 0) or np.any(mu[public.P == 0] != 0):
                            raise ValueError('Unsupported original expected photon mean')
                        rng = np.random.default_rng(np.random.SeedSequence([int(seed), scene_index, i]))
                        photons = rng.poisson(mu)
                        extracted = ppc_features(CountFrame(photons, public.P, True),
                                                 AF_band=band, estimate_stat_se=False)
                        if band is None:
                            extracted['AF_band'] = None; extracted['support'] = 'unresolved'
                            for key in ('q_star', 'FWHM', 'highq_ratio', 'valley_ratio'):
                                extracted['values'][key] = None
                                extracted['feature_missing_reasons'][key] = 'no_observed_peak_band'
                        extracted.update(index=i, outside_range=None,
                            expected_photons_sha256=_array_sha(mu), replica_photons_sha256=_array_sha(photons),
                            forward_receipt=receipt)
                    except (ValueError, OverflowError, FloatingPointError, RuntimeError, MemoryError, OSError) as error:
                        extracted = dict(index=i, AF_band=band, forward_failed=True, outside_range=None,
                            values={k:None for k in GATES},
                            feature_missing_reasons={k:'original_draw_failure' for k in GATES},
                            failure_reason=type(error).__name__+': '+str(error), structural_release=False)
                    replicas.append(extracted)
                if len(replicas) != part['original_draws']:
                    raise ValueError('Original conditional draw count changed')
                output[scene] = dict(status='research_individual_draw_PPC_complete', draws=replicas,
                    PPC=evaluate(observed, replicas, expected_draws=part['original_draws']),
                    calibrated=False, OOD_qualified=False, structural_release=False)
        if _frame_identity(full, public) != sampled['frame_identity']:
            raise ValueError('Original observation changed during predictive calculation')
        return dict(status='four_scenario_research_PPC_complete', scenarios=output,
            feature_certificate_sha256=_sha(certificate_path), host_shapes_sha256=_sha(hp),
            harddisk_sha256=_sha(dp), backend=backend, original_replica_seed=int(seed),
            posterior_images_averaged=False, count_support_applications=1,
            calibrated=False, OOD_qualified=False, structural_release=False)
