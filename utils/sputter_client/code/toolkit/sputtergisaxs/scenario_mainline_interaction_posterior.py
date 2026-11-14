'Independent data/coarse interaction head for the original51D joint model.'
import json
from numbers import Integral
from pathlib import Path

import torch
from torch import nn

from .scenario_mainline_chart import SCHEMA, DIM, COORDINATES, CATALOG
from .scenario_mainline_posterior import (
    MainlineCandidate as _BaseMainlineCandidate,
    MainlineRuntime as _BaseMainlineRuntime,
    architecture_spec as _base_architecture_spec,
    ARCHITECTURE as BASE_ARCHITECTURE)
from .scenario_mainline_training import generating_source_sha, BANK_SCHEMAS
from .scenario_runtime import _sha, _source_sha, _state_digest

ARCHITECTURE = 'mainline_template_interaction_gain_anchor_optional_signed_delta_v1'
TRAINING_SCHEMA = 'exploratory_mainline_interaction_joint_training_1'
DEFAULT_HIDDEN = 256
DEFAULT_COMPONENTS = 8
MODULE_SOURCE = 'sputtergisaxs/scenario_mainline_interaction_posterior.py'

__all__ = ('MainlineInteractionCandidate', 'InteractionRuntime',
    'architecture_spec', 'ARCHITECTURE', 'TRAINING_SCHEMA',
    'DEFAULT_HIDDEN', 'DEFAULT_COMPONENTS')


def architecture_spec():
    'Declare topology; actual hidden/K stay in the bound training declaration.'
    spec = _base_architecture_spec()
    spec.update(name=ARCHITECTURE, base_architecture=BASE_ARCHITECTURE,
        template_categorical_head=dict(
            input='same data embedding concatenated with the same12D coarse embedding',
            network=['Linear(hidden+12,hidden)', 'SiLU', 'Linear(hidden,T)'],
            output='conditional template logits for the existing physical legal mask',
            data_coarse_interaction=True),
        configurable_dimensions=dict(hidden='positive integer in declaration.hidden',
            components='positive integer in declaration.components'),
        defaults=dict(hidden=DEFAULT_HIDDEN, components=DEFAULT_COMPONENTS),
        unchanged=['continuous chart and decoder', 'coarse categorical head',
            'coarse/template embeddings', 'full joint conditional mixture',
            'gain anchor and covariance floors', 'optional signed delta branch',
            'legal masks and fixed-S sampler'],
        old_weights_compatible=False, production_selected=False,
        independent_TRAIN_CAL_experiment=True)
    return spec


def _dimension(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError('Positive integer interaction ' + name + ' required')
    return int(value)


class MainlineInteractionCandidate(_BaseMainlineCandidate):
    'Same complete joint model, with a nonlinear template categorical head.'
    def __init__(self, mode, continuous_dim=DIM, coarse_classes=len(CATALOG),
            template_classes=1, *, components=DEFAULT_COMPONENTS,
            hidden=DEFAULT_HIDDEN, catalog=None, bank_scenario_prior=None):
        hidden = _dimension(hidden, 'hidden')
        components = _dimension(components, 'components')
        super().__init__(mode, continuous_dim, coarse_classes, template_classes,
            components=components, hidden=hidden, catalog=catalog,
            bank_scenario_prior=bank_scenario_prior)
        self.template_logits = nn.Sequential(nn.Linear(hidden + 12, hidden),
            nn.SiLU(), nn.Linear(hidden, self.T))
        self.architecture = ARCHITECTURE
        self.validated = False
        self.structural_release = False


class InteractionRuntime(_BaseMainlineRuntime):
    'Strict independent weights; inherit the unchanged complete joint sampler.'
    def __init__(self, checkpoint, *, device='cpu', package_root=None):
        self.path = Path(checkpoint).resolve()
        package = Path(package_root).resolve() if package_root is not None else Path(__file__).resolve().parents[1]
        side = self.path.with_name(self.path.name + '.sha256')
        digest = side.read_text(encoding='ascii').split()[0]
        if _sha(self.path) != digest:
            raise ValueError('Selected interaction checkpoint SHA differs')
        saved = torch.load(self.path, map_location='cpu', weights_only=True)
        if _sha(self.path) != digest:
            raise ValueError('Selected interaction checkpoint changed during read')
        declaration = saved['declaration']; self.declaration = declaration
        if (declaration.get('schema') != TRAINING_SCHEMA
                or declaration.get('architecture') != ARCHITECTURE
                or declaration.get('architecture_spec') != architecture_spec()
                or declaration.get('TEST_inspected') is not False
                or declaration.get('physics_schema') != SCHEMA
                or declaration['dim'] != DIM or declaration['continuous_dim'] != DIM
                or declaration['coordinates'] != list(COORDINATES)
                or declaration['catalog'] != list(CATALOG)
                or declaration['bank'].get('schema') not in BANK_SCHEMAS
                or declaration['bank'].get('TEST_scientific_rows_decoded') is not False):
            raise ValueError('Explicit new interaction architecture/chart declaration required')
        hidden = _dimension(declaration['hidden'], 'hidden')
        components = _dimension(declaration['components'], 'components')
        self.templates = tuple(declaration['template_metadata'])
        self.keys = tuple(declaration['template_keys'])
        if (self.keys != tuple(t['key'] for t in self.templates)
                or len(set(self.keys)) != len(self.keys)
                or declaration['coarse_classes'] != len(CATALOG)
                or declaration['template_classes'] != len(self.keys) + 1
                or any(t.get('split') != 'TRAIN' for t in self.templates)):
            raise ValueError('Unique original TRAIN templates/catalog dimensions differ')
        self.pins = dict(declaration['source_pins'])
        current = Path(__file__).resolve().parents[1]
        if (MODULE_SOURCE not in self.pins
                or not set(declaration['bank']['source_pins']).issubset(self.pins)
                or any(self.pins[k] != value
                    for k, value in declaration['bank']['source_pins'].items())):
            raise ValueError('Interaction module and original generating sources must remain bound')
        for relative, value in self.pins.items():
            digest_fn = generating_source_sha if relative in declaration['bank']['source_pins'] else _source_sha
            if digest_fn(package / relative) != value or digest_fn(current / relative) != value:
                raise ValueError('Actual trained/executing interaction source differs: ' + relative)
        if _state_digest(saved['model']) != saved['state_sha256']:
            raise ValueError('Complete interaction state digest differs')
        run_path = self.path.parent / 'run.json'
        if _sha(run_path) != run_path.with_name(run_path.name + '.sha256').read_text(encoding='ascii').split()[0]:
            raise ValueError('Original interaction training run SHA differs')
        run = json.loads(run_path.read_text(encoding='utf8'))
        if (run.get('declaration') != declaration or run.get('best') is None
                or any(saved[k] != run['best'][k] for k in ('epoch', 'state_sha256', 'CAL_NLL'))):
            raise ValueError('Selected complete interaction epoch differs from its original run')
        self.device = torch.device(device); self.mode = declaration['mode']; self.package = package
        self.net = MainlineInteractionCandidate(self.mode, DIM, len(CATALOG), len(self.keys) + 1,
            components=components, hidden=hidden, catalog=CATALOG,
            bank_scenario_prior=declaration['bank_scenario_prior']).to(self.device)
        self.net.load_state_dict(saved['model'], strict=True); self.net.eval()
        self.receipt = dict(schema='actual_exploratory_mainline_interaction_runtime_1',
            weights_sha256=digest, state_sha256=saved['state_sha256'],
            selected_epoch=saved['epoch'], CAL_NLL=saved['CAL_NLL'],
            architecture=ARCHITECTURE, bank=declaration['bank'], mode=self.mode,
            source_pins=self.pins, training_terminal_status=run.get('status'),
            trained=True, calibrated=False, exploratory=True,
            production_selected=False, structural_release=False)
