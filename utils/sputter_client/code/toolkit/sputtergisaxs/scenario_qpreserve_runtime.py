'Separate strict loader for the q-preserving four-scenario candidate.'
from pathlib import Path

import numpy as np

from .scenario_chart import CATALOG, COORDINATES, SCHEMA
from .scenario_qpreserve import (BASELINE_ARCHITECTURE, Q_PRESERVE_ARCHITECTURE, GAIN_ANCHOR_ARCHITECTURE,
                                architecture_spec, make_candidate)
from .scenario_runtime import (ScenarioRuntime, _sha, _source_sha, _inside,
                               _state_digest)
from .v2_input import MODES


def safe_load_checkpoint(checkpoint):
    'Read a stable selected checkpoint with its original SHA sidecar.'
    import torch
    path = Path(checkpoint).resolve()
    digest = path.with_name(path.name+'.sha256').read_text(encoding='ascii').split()[0]
    if _sha(path) != digest:
        raise ValueError('Selected checkpoint SHA differs from its sidecar')
    try:
        saved = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError as error:
        raise RuntimeError('Safe weights_only checkpoint loading required; no unrestricted fallback') from error
    if _sha(path) != digest:
        raise ValueError('Checkpoint changed during safe load')
    return saved, digest


def _declared_architecture(declaration):
    architecture = declaration.get('architecture', BASELINE_ARCHITECTURE)
    expected = architecture_spec(architecture)
    if ('architecture' in declaration or 'architecture_spec' in declaration):
        if declaration.get('architecture_spec') != expected:
            raise ValueError('Selected checkpoint encoder architecture specification differs')
    return architecture, expected


class ScenarioQRuntime(ScenarioRuntime):
    'Explicit nonbaseline construction; inherited research methods stay unchanged.'
    def __init__(self, checkpoint, *, device='cpu', package_root=None):
        import torch
        self.path = Path(checkpoint).resolve()
        self.package = (Path(__file__).resolve().parent.parent if package_root is None
                        else Path(package_root).resolve())
        saved, digest = safe_load_checkpoint(self.path)
        d = saved['declaration']; bank = d['bank']
        architecture, specification = _declared_architecture(d)
        if architecture not in (Q_PRESERVE_ARCHITECTURE, GAIN_ANCHOR_ARCHITECTURE):
            raise ValueError('ScenarioQRuntime requires an explicitly declared supported architecture')
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
        encoder_path = 'sputtergisaxs/scenario_qpreserve.py'
        if (encoder_path not in d['source_pins']
                or d['source_pins'][encoder_path] != _source_sha(Path(__file__).parent/'scenario_qpreserve.py')):
            raise ValueError('Actual executing q-preserving encoder is not bound by the training declaration')
        if architecture == GAIN_ANCHOR_ARCHITECTURE:
            relative = 'sputtergisaxs/scenario_gain_anchor.py'
            if (relative not in d['source_pins'] or
                    d['source_pins'][relative] != _source_sha(Path(__file__).parent/'scenario_gain_anchor.py')):
                raise ValueError('Actual gain anchor is not bound by the training declaration')
        if _state_digest(saved['model']) != saved['state_sha256']:
            raise ValueError('Selected checkpoint tensor state differs')
        if (isinstance(saved['epoch'], bool) or not isinstance(saved['epoch'], int)
                or saved['epoch'] < 0 or not np.isfinite(saved['CAL_NLL'])):
            raise ValueError('Selected complete CAL epoch required')
        self.declaration = d; self.device = torch.device(device); self.mode = d['mode']
        self.net = make_candidate(self.mode, d['dim'], d['coarse_classes'], d['template_classes'],
            architecture=architecture, hidden=d['hidden'], components=d['components'],
            catalog=d['catalog'], bank_scenario_prior=d['bank_scenario_prior']).to(self.device)
        self.net.load_state_dict(saved['model'], strict=True); self.net.eval()
        self.receipt = dict(schema='four_scenario_research_runtime_1', weights_sha256=digest,
            state_sha256=saved['state_sha256'], selected_epoch=saved['epoch'], CAL_NLL=saved['CAL_NLL'],
            plan_sha256=bank['plan_sha256'], input_mode=self.mode, training_phase=d['phase'],
            final_family_target_met=d['final_family_target_met'], architecture=architecture,
            architecture_spec=specification,
            execution_source_pins={name:_source_sha(Path(__file__).parent/name) for name in
                ('scenario_runtime.py', 'scenario_posterior.py', 'scenario_qpreserve.py',
                 'scenario_qpreserve_runtime.py')},
            calibrated=False, OOD_qualified=False, structural_release=False)
        if architecture == GAIN_ANCHOR_ARCHITECTURE:
            self.receipt['execution_source_pins']['scenario_gain_anchor.py'] = _source_sha(Path(__file__).parent/'scenario_gain_anchor.py')


def load_runtime(checkpoint, *, device='cpu', package_root=None):
    'Select a strict actual loader; a missing legacy architecture is baseline.'
    saved, digest = safe_load_checkpoint(checkpoint)
    architecture, _ = _declared_architecture(saved['declaration'])
    del saved
    cls = ScenarioRuntime if architecture == BASELINE_ARCHITECTURE else ScenarioQRuntime
    runtime = cls(checkpoint, device=device, package_root=package_root)
    if runtime.receipt['weights_sha256'] != digest:
        raise ValueError('Selected checkpoint changed between architecture inspection and construction')
    return runtime
