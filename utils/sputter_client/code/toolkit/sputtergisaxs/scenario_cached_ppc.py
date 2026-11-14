'Optional certified CPU caching for original four-scenario posterior draws.'
from numbers import Integral
from pathlib import Path

import numpy as np

from .scenario_cached_cpu import mean_forward_cached
from .scenario_chart import SCENARIO_NAMES, decode
from .scenario_cpu_certificate import cpu_certificate
from .scenario_forward import QUICK, harddisk_table, mean_forward
from .scenario_runtime import _array_sha, _frame_identity, _inside, _json, _sha, _source_sha
from .v2_observation import CountFrame, ppc_features
from .v2_ppc import GATES, evaluate


S4_FEATURES = ('q_star', 'FWHM', 'Yoneda')


def _clean_features(mu, public, band, index):
    'Retain the existing mean response, separately from its photon replica.'
    try:
        if mu is None:
            raise ValueError('Original forward mean unavailable')
        clean = ppc_features(CountFrame(mu, public.P, False),
                             AF_band=band, estimate_stat_se=False)
        if band is None:
            clean['AF_band'] = None; clean['support'] = 'unresolved'
            for key in ('q_star', 'FWHM', 'highq_ratio', 'valley_ratio'):
                clean['values'][key] = None
                clean['feature_missing_reasons'][key] = 'no_observed_peak_band'
        clean.update(index=index, expected_photons_sha256=_array_sha(mu),
                     photon_count_statistics_known=False, Poisson_draw=False)
        return clean
    except (ValueError, TypeError, KeyError, IndexError, OverflowError,
            FloatingPointError, RuntimeError, MemoryError, OSError) as error:
        return dict(index=index, AF_band=band, forward_failed=True,
            values={k:None for k in GATES},
            feature_missing_reasons={k:'original_mean_feature_failure' for k in GATES},
            failure_reason=type(error).__name__+': '+str(error),
            expected_photons_sha256=_array_sha(mu) if mu is not None else None,
            photon_count_statistics_known=False, Poisson_draw=False,
            structural_release=False)


def clean_feature_summary(draws, expected_draws):
    'Same complete-pool quantiles as independent S4 feature calibration.'
    output = {}
    for key in S4_FEATURES:
        values = np.asarray([d.get('values', {}).get(key)
                             if not d.get('forward_failed') and d.get('values', {}).get(key) is not None
                             else np.nan for d in draws], float)
        valid = bool(len(values) == expected_draws and expected_draws >= 2
                     and np.isfinite(values).all() and np.all(values > 0))
        bounds = np.quantile(values, [.025, .16, .5, .84, .975]) if valid else None
        output[key] = dict(median=float(bounds[2]) if valid else None,
            interval68=bounds[[1,3]].tolist() if valid else None,
            interval95=bounds[[0,4]].tolist() if valid else None,
            original_draws=int(expected_draws),
            missing_original_draws=int(np.sum(~np.isfinite(values) | (values <= 0))),
            valid_complete_original_joint=valid, calibrated=False,
            target='latent_mean_response_features', Poisson_draw=False,
            units='degree' if key == 'Yoneda' else 'nm^-1')
    return output


def posterior_predictive_cached(runtime, sampled, full, public, *, bank_plan,
                                feature_certificate, cpu_cache_certificate,
                                host_shapes=None, host_metadata=None, harddisk=None,
                                backend='cpu', seed=672820261018):
    'One original forward/count replica per draw, with optional CPU caching.'
    helper_sha = _source_sha(Path(__file__))
    if sampled['receipt'] != runtime.receipt or _frame_identity(full, public) != sampled['frame_identity']:
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
            CPU_acceleration=dict(status='not_executed_unknown_photon_statistics',
                                  structural_release=False),
            cached_PPC_source_sha256=helper_sha, calibrated=False, structural_release=False)

    plan_path = Path(bank_plan).resolve(); plan = _json(plan_path)
    bank = runtime.declaration['bank']; generation_pins = bank['source_pins']
    for relative, value in generation_pins.items():
        if _source_sha(_inside(runtime.package, relative)) != value:
            raise ValueError('Original predictive generating source changed: '+relative)
    if (_sha(plan_path) != bank['plan_sha256'] or plan['source_pins'] != generation_pins
            or plan['catalog'] != runtime.declaration['catalog']
            or plan['coordinates'] != runtime.declaration['coordinates']):
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
    if tuple(_json(hm)['rows']) != runtime.templates:
        raise ValueError('Original normalized TRAIN template metadata differs')

    if cpu_cache_certificate is None:
        acceleration = dict(status='not_supplied', enabled_scenarios=[],
            applicable_in_current_environment=False, reason='original_forward_no_CPU_cache_certificate',
            structural_release=False)
    else:
        acceleration = cpu_certificate(cpu_cache_certificate)
    enabled = (set(acceleration['enabled_scenarios'])
               if backend == 'cpu' and acceleration['applicable_in_current_environment'] else set())
    acceleration = dict(acceleration, requested_backend=backend,
        used_scenarios=[], original_runtime_receipt_changed=False,
        original_feature_certificate_reused_for_caching=False,
        numerical_certificate_is_structure_evidence=False, structural_release=False)
    if backend != 'cpu':
        acceleration['dispatch_reason'] = 'original_forward_for_non_CPU_backend'

    observed = sampled['observed']; band = observed['AF_band']; output = {}; used = set()
    with np.load(hp, allow_pickle=False) as hostarr, np.load(dp, allow_pickle=False) as diskarr:
        disk = harddisk_table(diskarr)
        for scene_index, scene in enumerate(SCENARIO_NAMES):
            part = sampled['scenarios'][scene]
            if not part['applicable']:
                output[scene] = dict(status='not_applicable', reason=part['not_applicable_reason'],
                                     structural_release=False)
                continue
            replicas = []; latent = []; original = part['original_joint']
            forward = mean_forward_cached if scene in enabled else mean_forward
            for i, record in enumerate(part['draws']):
                mu = None
                try:
                    if record['decode_failed']:
                        raise ValueError('Original joint decoder failure: '+record['reason'])
                    physical = decode(original['z'][i], int(original['coarse'][i]), int(original['template'][i]),
                        metal=sampled['geometry']['metal'], E=sampled['geometry']['energy_ev'],
                        ai=sampled['geometry']['alpha_i_deg'], template_keys=runtime.keys,
                        template_metadata=runtime.templates)
                    if physical != record['physical'] or physical['scenario'] != scene:
                        raise ValueError('Original joint physical draw changed')
                    rate, _, receipt = forward(physical, full.P, hostarr, recipe=QUICK,
                                                backend=backend, disk=disk)
                    if scene in enabled:
                        used.add(scene)
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
                if scene == 'S4':
                    latent.append(_clean_features(mu, public, band, i))
            if len(replicas) != part['original_draws']:
                raise ValueError('Original conditional draw count changed')
            output[scene] = dict(status='research_individual_draw_PPC_complete', draws=replicas,
                PPC=evaluate(observed, replicas, expected_draws=part['original_draws']),
                calibrated=False, OOD_qualified=False, structural_release=False)
            if scene == 'S4':
                output[scene].update(latent_feature_draws=latent,
                    conditional_features=clean_feature_summary(latent, part['original_draws']),
                    latent_response_binding=dict(runtime_receipt=runtime.receipt,
                        frame_identity=sampled['frame_identity'],
                        joint_sha256={k:_array_sha(v) for k,v in original.items()},
                        original_seed=sampled['original_seed'],
                        original_draws=part['original_draws'], AF_band=band,
                        extraction_source_sha256=helper_sha),
                    latent_response_extra_forward_calls=0,
                    latent_response_count_support_applications=1)
    if _frame_identity(full, public) != sampled['frame_identity']:
        raise ValueError('Original observation changed during predictive calculation')
    acceleration['used_scenarios'] = [s for s in SCENARIO_NAMES if s in used]
    return dict(status='four_scenario_research_PPC_complete', scenarios=output,
        feature_certificate_sha256=_sha(certificate_path), host_shapes_sha256=_sha(hp),
        harddisk_sha256=_sha(dp), backend=backend, original_replica_seed=int(seed),
        posterior_images_averaged=False, count_support_applications=1,
        CPU_acceleration=acceleration, cached_PPC_source_sha256=helper_sha,
        calibrated=False, OOD_qualified=False, structural_release=False)
