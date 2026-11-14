'Observed AF bounds as geometry for a complete conditional research posterior.'
from numbers import Integral
from pathlib import Path
import re

import numpy as np

from .scenario_chart import CATALOG, COORDINATES, DIM, SCENARIO_NAMES, decode
from .scenario_peak_geometry_posterior import PeakGeometryScenarioCandidate, input_spec, encode_pair
from .scenario_posterior import legal_from_metadata
from .scenario_peak_highq_runtime import PeakHighqScenarioRuntime, REQUIRED_TRAINING_SOURCES as HIGHQ_SOURCES
from .scenario_minimal_runtime import MinimalScenarioRuntime, _domain_summary
from .scenario_runtime import (_sha, _source_sha, _json, _inside, _state_digest,
    _array_sha, _frame_identity, _labels, _conditional_summaries, _public_features)
from .features import AF_BANDS
from .v2_observation import ppc_features

TRAINING_SCHEMA = 'four_scenario_peak_geometry_training_1'
ARCHITECTURE = 'peak_geometry_joint'
SAMPLE_SEED = 681720261013
REQUIRED_TRAINING_SOURCES = tuple(dict.fromkeys(HIGHQ_SOURCES+(
    'scripts/v2_peak_geometry_train.py', 'sputtergisaxs/scenario_peak_geometry_posterior.py',
    'sputtergisaxs/features.py')))


class PeakGeometryScenarioRuntime(PeakHighqScenarioRuntime):
    'Source-bound AF-geometry model; existing original joint/PPC operations.'
    def __init__(self, checkpoint, *, training_package, producer_package, device='cpu'):
        import torch
        self.path = Path(checkpoint).resolve(); self.package = Path(__file__).resolve().parent.parent
        self.training_package = Path(training_package).resolve()
        self.producer_package = Path(producer_package).resolve()
        digest = self.path.with_name(self.path.name+'.sha256').read_text(encoding='ascii').split()[0]
        if _sha(self.path) != digest:
            raise ValueError('Actual AF-geometry selected checkpoint SHA differs')
        try:
            saved = torch.load(self.path, map_location='cpu', weights_only=True)
        except TypeError as error:
            raise RuntimeError('Safe weights_only checkpoint loading required') from error
        if _sha(self.path) != digest:
            raise ValueError('Selected AF-geometry checkpoint changed during load')
        d = saved['declaration']; bank = d['bank']; mode = d.get('feature_mode')
        if mode not in ('qwidth', 'qwidth_highq'):
            raise ValueError('Explicit supported geometry-aware feature mode required')
        spec = input_spec(mode)
        if (d.get('schema') != TRAINING_SCHEMA or d.get('architecture') != ARCHITECTURE
                or d.get('input_spec') != spec or d.get('fixture') is not False
                or d.get('TEST_inspected') is not False or bank.get('fixture') is not False
                or bank.get('TEST_scientific_rows_decoded') is not False
                or bank.get('original_families') != 30000 or bank['split_rows']['TEST'] != 0
                or d['dim'] != DIM or d['continuous_dim'] != DIM
                or d['catalog'] != list(CATALOG) or d['coordinates'] != list(COORDINATES)
                or d['coarse_classes'] != len(CATALOG) or d['compact_data']['bankreceipt'] != bank):
            raise ValueError('Actual original44D AF-geometry training/chart/compact declaration required')
        self.templates = tuple(d['template_metadata']); self.keys = tuple(d['template_keys'])
        if (self.keys != tuple(t['key'] for t in self.templates) or len(set(self.keys)) != len(self.keys)
                or d['template_classes'] != len(self.keys)+1 or any(t.get('split') != 'TRAIN' for t in self.templates)):
            raise ValueError('Unique source-bound TRAIN template catalog required')
        build_path = self.training_package/'BUILD_INFO.json'; build = _json(build_path)
        commit = d.get('training_commit'); build_sha = _sha(build_path)
        if (not isinstance(commit, str) or re.fullmatch('[0-9a-f]{40}', commit) is None
                or build.get('git_commit') != commit or build.get('git_dirty') is not False
                or d.get('training_BUILD_INFO_sha256') != build_sha):
            raise ValueError('Checkpoint actual clean training commit/BUILD_INFO binding required')
        if not set(REQUIRED_TRAINING_SOURCES) <= set(d['source_pins']):
            raise ValueError('Original and AF-geometry training source pins incomplete')
        for relative, value in d['source_pins'].items():
            if (_source_sha(_inside(self.training_package, relative)) != value
                    or _source_sha(_inside(self.package, relative)) != value):
                raise ValueError('Actual executing/original AF-geometry training source differs: '+relative)
        for relative, value in bank['source_pins'].items():
            if (_source_sha(_inside(self.producer_package, relative)) != value
                    or _source_sha(_inside(self.package, relative)) != value):
                raise ValueError('Actual executing/original generating source differs: '+relative)
        if (_state_digest(saved['model']) != saved['state_sha256']
                or not isinstance(saved['epoch'], int) or isinstance(saved['epoch'], bool)
                or saved['epoch'] < 0 or not np.isfinite(saved['CAL_NLL'])
                or not np.isfinite(saved['CAL_objective_NLL'])):
            raise ValueError('Actual selected complete-epoch tensor state required')
        self.declaration = d; self.feature_mode = mode; self.device = torch.device(device)
        if self.device.type not in ('cpu', 'cuda'):
            raise ValueError('CPU or CUDA device required')
        self.net = PeakGeometryScenarioCandidate(DIM, len(CATALOG), len(self.keys)+1,
            feature_mode=mode, components=d['components'], hidden=d['hidden'],
            catalog=d['catalog'], bank_scenario_prior=d['bank_scenario_prior']).to(self.device)
        self.net.load_state_dict(saved['model'], strict=True); self.net.eval()
        self.receipt = dict(schema='observed_AF_geometry_research_runtime_1', weights_sha256=digest,
            state_sha256=saved['state_sha256'], selected_epoch=saved['epoch'], CAL_NLL=saved['CAL_NLL'],
            CAL_objective_NLL=saved['CAL_objective_NLL'], training_commit=commit,
            training_BUILD_INFO_sha256=build_sha, plan_sha256=bank['plan_sha256'],
            architecture=ARCHITECTURE, feature_mode=mode, input_mode=mode+'_observed_AF_geometry',
            input_dim=spec['input_dim'], input_spec=spec, training_source_pins=d['source_pins'],
            execution_source_pins={name:_source_sha(Path(__file__).parent/name) for name in (
                'scenario_peak_geometry_runtime.py', 'scenario_peak_geometry_posterior.py',
                'scenario_peak_highq_runtime.py', 'scenario_peak_highq_posterior.py',
                'scenario_minimal_runtime.py', 'scenario_posterior.py', 'scenario_runtime.py',
                'features.py', 'v2_ppc.py', 'v2_observation.py')},
            predictive_implementation='Original minimal gain-fit per-joint-draw PPC; highq mode adds original same-AF gate',
            calibrated=False, OOD_qualified=False, structural_release=False)

    def sample(self, full, public, energy_ev, alpha_i_deg, metal, *, host_category=None, n=256, seed=SAMPLE_SEED):
        'Observed band bounds, complete fixed-S original joint samples.'
        import torch
        if (not isinstance(n, Integral) or isinstance(n, (bool, np.bool_)) or n < 2
                or not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or seed < 0):
            raise ValueError('At least two original draws and nonnegative integer seed required')
        identity = _frame_identity(full, public)
        if (np.any(public.P > full.P) or np.any(public.Y > full.Y)
                or np.any((public.P == full.P) & (public.Y != full.Y))):
            raise ValueError('Public observations must remain a subset of original full counts')
        observed = ppc_features(public, estimate_stat_se=identity['photon_statistics_known'])
        band = observed['AF_band']; bounds = None if band is None else AF_BANDS[int(band)]
        x = encode_pair(observed['values']['q_star'], observed['values']['FWHM'], energy_ev, alpha_i_deg, metal,
            AF_band=band, alpha_f_bounds_deg=bounds, highq_ratio=observed['values']['highq_ratio'],
            feature_mode=self.feature_mode)
        encoded = dict(vector=torch.as_tensor(x['vector'], device=self.device), image=None)
        legal = legal_from_metadata(energy_ev, alpha_i_deg, self.templates,
            host_categories=[host_category], device=self.device)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        sampled = self.net.sample_all_applicable(encoded, int(n), legal_templates=legal,
            generator=generator, host_categories=[host_category])
        scenes = {}
        for scene in SCENARIO_NAMES:
            part = sampled['samples'][scene]; applicable = bool(part['applicable'][0])
            joint = {k:part[k][0].detach().cpu().numpy() for k in ('z', 'coarse', 'template', 'component')}
            records = []
            if applicable:
                for i in range(int(n)):
                    try:
                        physical = decode(joint['z'][i], int(joint['coarse'][i]), int(joint['template'][i]),
                            metal=metal, E=float(energy_ev), ai=float(alpha_i_deg),
                            template_keys=self.keys, template_metadata=self.templates)
                        if physical['scenario'] != scene:
                            raise ValueError('Original fixed-S joint draw changed scenario')
                        labels = _labels(physical)
                        if scene == 'S4':
                            labels['domain_nm'] = float(physical['domain_nm'])
                        records.append(dict(index=i, decode_failed=False, physical=physical, conditional_labels=labels))
                    except (ValueError, OverflowError, FloatingPointError) as error:
                        records.append(dict(index=i, decode_failed=True, reason=str(error), physical=None, conditional_labels={}))
            summaries = _conditional_summaries(scene, records)
            if scene == 'S4':
                summaries['domain_nm'] = _domain_summary(records)
            scenes[scene] = dict(applicable=applicable, not_applicable_reason=part['not_applicable_reason'][0],
                original_draws=len(records), original_joint=joint, draws=records, conditional_quantities=summaries,
                public_quantities=dict(D_nm=None, diameter_nm=None, height_nm=None, gap_nm=None, coverage_fraction=None),
                missing_reason='No independent calibration/TEST and frame release receipt',
                calibrated=False, structural_release=False)
        probabilities = sampled['scenario_comparison']['probabilities'][0].detach().cpu().numpy()
        pair = bool(x['complete_feature_pair'][0])
        highq = self.feature_mode == 'qwidth_highq' and bool(x['highq_known'][0])
        return dict(receipt=dict(self.receipt), frame_identity=identity,
            geometry=dict(energy_ev=float(energy_ev), alpha_i_deg=float(alpha_i_deg), metal=metal),
            host_category_input=host_category, input_vector_sha256=_array_sha(x['vector']),
            input_AF_band=band, input_AF_bounds_deg=list(bounds) if bounds is not None else None,
            observed_AF_geometry_known=bounds is not None, input_pair_available=pair,
            input_highq_available=bool(highq), input_triplet_available=bool(pair and highq),
            observed=observed, features=_public_features(observed), scenarios=scenes,
            reference_scenario_probabilities=dict(zip(SCENARIO_NAMES, probabilities.tolist())),
            scenario_probabilities_are_reference_only=True, scenarios_pruned=False,
            original_seed=int(seed), original_draws_per_applicable_scenario=int(n),
            conditioned_outside_D=False, nuisance_prior_resampled=False, calibrated=False, structural_release=False)

    def posterior_predictive(self, sampled, full, public, **kwargs):
        'Same original pools/gain; highq gate exists only for highq-input mode.'
        if self.feature_mode == 'qwidth':
            result = MinimalScenarioRuntime.posterior_predictive(self, sampled, full, public, **kwargs)
        else:
            result = super().posterior_predictive(sampled, full, public, **kwargs)
        result.update(feature_mode=self.feature_mode, observed_AF_bounds_deg=sampled['input_AF_bounds_deg'],
            runtime_source_sha256=_source_sha(__file__), calibrated=False, structural_release=False)
        return result
