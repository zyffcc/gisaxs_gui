'TRAIN/CAL reader for the source-bound simplified four-scenario bank.'
import hashlib
import itertools
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from .stage5_partial import sha, selected_array
from .v2_dataset import array_header, read_split
from .v2_input import INPUT_KEYS, HOST_CLASSES, prepare
from .v2_training import optional_missing
from .scenario_chart import (SCHEMA, COORDINATES, CATALOG, active_coordinates,
                             template_compatibility, decode)
from .stage5_measurement import NATIVE_AF, Q
from .tool_qinput import GRID_SHAPE


BANK_SCHEMA = 'four_scenario_bank_1'


def source_sha(path):
    "Match the generator's LF-normalized Windows/Linux source definition."
    path=Path(path); data=path.read_bytes()
    return hashlib.sha256(data.replace(b'\r\n',b'\n') if path.suffix=='.py' else data).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def _inside(root, relative):
    path = (root/relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError('Bank/source path escapes its declared root: '+str(relative))
    return path


def _bound(root, item):
    path = _inside(root, item['path'])
    if sha(path) != item['sha256']:
        raise ValueError('Bound bank input changed: '+item['path'])
    return path


def _shard_header(path):
    "Read the producer's top-level prefix without interpreting receipts."
    prefix = []
    with Path(path).open('r', encoding='utf8') as stream:
        for line in stream:
            if re.match(r'^  "receipts"\s*:', line):
                return json.loads(''.join(prefix).rstrip().rstrip(',')+'}')
            prefix.append(line)
    raise ValueError('Completed original shard has no receipt-prefix boundary')


def shapes(dim):
    'Executed existing q/input sizes plus plan-defined joint dimensions.'
    return dict(image=(2,)+tuple(GRID_SHAPE), pre=(2,)+tuple(GRID_SHAPE),
                features=(40,), mandatory=(5,), optional=(3,), known=(4,),
                public_N=(8, len(Q)), mean_rate=(2, len(NATIVE_AF), len(Q)),
                z=(dim,), active=(dim,), coarse=(), template=(), scenario=(),
                geometry=(2,), public_visible_peak=())


class ScenarioBank:
    'Complete production/pilot reader; no synthetic-fixture bypass.'
    def __init__(self, bank, package=None, *, fixture=False, allow_fixture=False):
        if fixture is not False or allow_fixture is not False:
            raise ValueError('ScenarioBank has no interface-fixture bypass')
        path = Path(bank).resolve(); self.base = path if path.is_dir() else path.parent
        self.package = Path(package).resolve() if package is not None else Path(__file__).resolve().parent.parent
        self.plan = _json(self.base/'plan.json'); plan = self.plan
        run = _json(self.base/'run.json')
        if (plan.get('schema') != BANK_SCHEMA or plan.get('status') != 'frozen'
                or run.get('schema') != BANK_SCHEMA or run.get('status') != 'completed'
                or plan.get('fixture') is not False or run.get('fixture') is not False
                or run.get('plan_sha256') != sha(self.base/'plan.json')):
            raise ValueError('Complete source-bound four-scenario generated bank required')
        if plan.get('real_TEST_decoded') is not False:
            raise ValueError('Production generation must not use real TEST scans')
        if plan.get('physics_schema') != SCHEMA:
            raise ValueError('Original scenario chart schema differs')
        self.coordinates = tuple(plan['coordinates']); self.dim = len(self.coordinates)
        self.catalog = tuple(plan['catalog']); self.shapes = shapes(self.dim)
        if (self.coordinates != tuple(COORDINATES) or self.catalog != tuple(CATALOG)
                or self.dim < 1 or not self.catalog):
            raise ValueError('Source-bound joint coordinate/catalog declaration differs')
        pins = plan['source_pins']
        required = ('sputtergisaxs/scenario_chart.py', 'sputtergisaxs/scenario_forward.py',
                    'sputtergisaxs/scenario_measurement.py', 'scripts/v2_scenario_bank.py',
                    'sputtergisaxs/v2_input.py', 'sputtergisaxs/v2_observation.py')
        if not set(required).issubset(pins):
            raise ValueError('Generator/decoder/input source pins are incomplete')
        for relative, digest in pins.items():
            if source_sha(_inside(self.package, relative)) != digest:
                raise ValueError('Original generating source changed: '+relative)
        self.bindings = {name:_bound(self.base, item) for name, item in plan['bindings'].items()}
        split = _json(self.bindings['split.json']); policy = split['synthetic_split']
        if (policy != plan['synthetic_split'] or policy['TRAIN_family_remainders'] != list(range(8))
                or policy['CAL_family_remainders'] != [8] or policy['TEST_family_remainders'] != [9]):
            raise ValueError('Original synthetic family split/seed policy differs')
        support = _json(self.bindings['support.json'])
        if (support.get('TEST_counts_decoded') is not False
                or support['split_sha256'] != sha(self.bindings['split.json'])
                or support['support_sha256'] != sha(self.bindings['supports.npz'])):
            raise ValueError('Original support is not bound TRAIN-only data')
        original = _json(self.bindings['original_host.json'])
        normalized = _json(self.bindings['host_shapes.json'])
        train = {s['scan'] for s in split['scans'] if s['split'] == 'TRAIN'}
        if (original['array_sha256'] != sha(self.bindings['original_host.npz'])
                or original['split_sha256'] != sha(self.bindings['split.json'])
                or original.get('no_CAL_TEST_input') is not True):
            raise ValueError('Original empirical host library is not bound TRAIN-only data')
        self.templates = normalized['rows']; self.keys = tuple(t['key'] for t in self.templates)
        old_templates = {t['key']:t for t in original['rows']}
        if (len(set(self.keys)) != len(self.keys) or set(self.keys) != set(old_templates)
                or any(t.get('split') != 'TRAIN' or t.get('scan') not in train for t in self.templates)):
            raise ValueError('Empirical template catalog is not unique TRAIN-only data')
        for t in self.templates:
            if (not np.isfinite(t.get('normalization_scale', np.nan)) or t['normalization_scale'] <= 0
                    or any(t.get(k) != v for k, v in old_templates[t['key']].items())):
                raise ValueError('Template normalization changed its source metadata')

        with zipfile.ZipFile(self.bindings['assignments.npz']) as archive:
            header, dtype = array_header(archive, 'family_id')
            if len(header) != 1 or dtype.kind not in 'iu':
                raise ValueError('Integer original family identities required')
            self.families = selected_array(archive, 'family_id')
            scene_shape, scene_dtype = array_header(archive, 'scenario')
            if scene_shape != header or scene_dtype.kind not in 'iu':
                raise ValueError('Original scenario-assignment shape differs')
            selected = np.flatnonzero(self.families % 10 != 9)

            assigned_scenes = selected_array(archive, 'scenario', selected)
            if np.any(assigned_scenes < 0) or np.any(assigned_scenes >= 4):
                raise ValueError('Selected physical scenario assignment is invalid')
            self.scenario_by_family = dict(zip(map(int, self.families[selected]), map(int, assigned_scenes)))
        if (len(self.families) != plan['families'] or run['families'] != plan['families']
                or np.any(self.families < 0) or len(np.unique(self.families)) != len(self.families)):
            raise ValueError('Original complete family count/identities differ')
        self.split_rows = {name:int(np.sum(self.families % 10 < 8 if name == 'TRAIN'
                              else self.families % 10 == (8 if name == 'CAL' else 9)))
                           for name in ('TRAIN', 'CAL', 'TEST')}
        self.scenario_split_rows = {}
        for name in ('TRAIN', 'CAL'):
            keep = self.families[selected] % 10 < 8 if name == 'TRAIN' else self.families[selected] % 10 == 8
            self.scenario_split_rows[name] = {'S'+str(s+1):int(np.sum(assigned_scenes[keep] == s)) for s in range(4)}
        if plan.get('TEST_rows') != self.split_rows['TEST'] or run.get('TEST_rows') != self.split_rows['TEST']:
            raise ValueError('Declared original TEST count differs')
        if self.split_rows['TEST'] == 0 and run.get('TEST_decoded') is not False:
            raise ValueError('Pilot claims unexpected TEST decoding')
        if self.split_rows['TEST'] == 0 and run.get('scenario_counts') != {
                name:self.scenario_split_rows['TRAIN'][name]+self.scenario_split_rows['CAL'][name]
                for name in self.scenario_split_rows['TRAIN']}:
            raise ValueError('Original pilot scenario counts differ from selected identities')
        seen = set(); paths = set(); files = []; common = None
        for item in run['shards']:
            file = _bound(self.base, item)
            if file in paths:
                raise ValueError('Repeated generated shard')
            paths.add(file)
            receipt = _shard_header(file.parent/'run.json')
            if (receipt.get('status') != 'completed' or receipt.get('plan_sha256') != run['plan_sha256']
                    or receipt.get('source_pins') != pins or receipt.get('checkpoint_file') != file.name
                    or receipt.get('checkpoint_sha256') != item['sha256']
                    or receipt.get('rows') != item['rows']):
                raise ValueError('Original completed shard/source/checksum differs')
            identity = (receipt.get('certificate_sha256'), receipt.get('backend'))
            if not all(isinstance(v, str) and v for v in identity):
                raise ValueError('Original numerical gate/backend identity is missing')
            if common is None:
                common = identity
            elif identity != common:
                raise ValueError('Mixed original numerical gate or backend')
            with zipfile.ZipFile(file) as archive:
                for key in ('family_id', 'index'):
                    header, dtype = array_header(archive, key)
                    if len(header) != 1 or dtype.kind not in 'iu':
                        raise ValueError('Integer original row/family identity required')
                families = selected_array(archive, 'family_id'); index = selected_array(archive, 'index')
                n = len(families)
                if (n != item['rows'] or len(index) != n or not np.array_equal(index, families)
                        or families.tolist() != receipt['family_ids'] or len(np.unique(index)) != n
                        or seen.intersection(map(int, index))):
                    raise ValueError('Original shard identities/count/duplicates differ')
                seen.update(map(int, index))
                for key, expected in self.shapes.items():
                    header, dtype = array_header(archive, key)
                    if header != (n,)+expected:
                        raise ValueError('Original scientific array shape differs: '+key)
                    if (key == 'z' and dtype != np.dtype('float64')
                            or key in ('known', 'active', 'public_visible_peak') and dtype.kind != 'b'
                            or key in ('coarse', 'template', 'scenario') and dtype.kind not in 'iu'):
                        raise ValueError('Original joint class/coordinate/flag dtype differs: '+key)
            files.append((file, item['sha256']))
        if seen != set(map(int, self.families)):
            raise ValueError('Original generated bank is incomplete; no dropped families')
        self.files = tuple(files); self.shards = self.files
        if not self.files:
            raise ValueError('No complete generated shards')
        self.receipt = dict(schema=BANK_SCHEMA, bank_run_sha256=sha(self.base/'run.json'),
            plan_sha256=run['plan_sha256'], original_families=len(self.families),
            original_rows=len(seen), split_rows=dict(self.split_rows),
            scenario_split_rows=self.scenario_split_rows,
            source_pins=pins, fixture=False, validation_mode='source_bound_four_scenario_generated_bank',
            physics_schema=plan['physics_schema'], continuous_dim=self.dim,
            coarse_classes=len(self.catalog), template_classes=len(self.templates)+1,
            certificate_sha256=common[0], backend=common[1],
            TEST_scientific_rows_decoded=False, structural_release=False)

    def indices(self, split):
        if split not in ('TRAIN', 'CAL'):
            raise ValueError('Scenario training reader only selects TRAIN or CAL')
        return self.families[self.families % 10 < 8 if split == 'TRAIN' else self.families % 10 == 8].copy()

    def iterate(self, split, *, epoch=0):
        'Yield only requested rows, shuffled for TRAIN and fixed for CAL.'
        if split not in ('TRAIN', 'CAL'):
            raise ValueError('Scenario training reader never decodes TEST science')
        rng = np.random.default_rng(np.random.SeedSequence([672520, int(epoch)]))
        order = rng.permutation(len(self.files)) if split == 'TRAIN' else np.arange(len(self.files))
        for j in order:
            path, digest = self.files[int(j)]
            arrays, _ = read_split(path, digest, split, self.shapes)
            row_order = rng.permutation(len(arrays['index'])) if split == 'TRAIN' else np.arange(len(arrays['index']))
            for i in row_order:
                row = {k:v[i] for k, v in arrays.items()}
                c, t, s = int(row['coarse']), int(row['template']), int(row['scenario'])
                if not 0 <= c < len(self.catalog) or not 0 <= t <= len(self.templates) or not 0 <= s < 4:
                    raise ValueError('Original scenario/coarse/template class outside plan')
                if self.catalog[c]['scenario_index'] != s:
                    raise ValueError('Physical scenario is not its coarse catalog class')
                if self.scenario_by_family[int(row['family_id'])] != s:
                    raise ValueError('Original physical scenario assignment changed')
                active = active_coordinates(c, t)
                if not np.array_equal(active, row['active']) or not np.isfinite(row['z'][active]).all():
                    raise ValueError('Original active joint coordinates differ')
                if (not np.isfinite(row['public_N']).all() or np.any(row['public_N'] < 0)
                        or np.any(row['public_N'] != np.rint(row['public_N']))
                        or np.any(row['features'][33:37] != 0)):
                    raise ValueError('Original pixel-frame support/shape-only reference differs')
                if not np.isfinite(row['mean_rate']).all() or np.any(row['mean_rate'] < 0):
                    raise ValueError('Stored actual mean rate is nonfinite or negative')
                E, ai = row['geometry']; mandatory = row['mandatory']
                if (mandatory[2] not in (0, 1) or not np.allclose(mandatory[:2],
                        [(E-14000)/6000, (ai-.425)/.275], rtol=0, atol=2e-6)):
                    raise ValueError('Mandatory geometry differs from encoder metadata')
                physical = decode(row['z'], c, t, metal='Au' if mandatory[2] == 1 else 'Ag',
                    E=E, ai=ai, template_keys=self.keys, template_metadata=self.templates)
                row['physical'] = physical
                row['diameter_nm'] = physical['diameter_nm']
                yield row

    def iter_batches(self, split, *, batch_size=256, mode='features', epoch=0, device='cpu'):
        'Yield (encoded, original joint targets, legal coarse/template mask).'
        if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
            raise ValueError('Positive batch size required')
        rows = self.iterate(split, epoch=epoch if split == 'TRAIN' else 0)
        while True:
            block = list(itertools.islice(rows, int(batch_size)))
            if not block:
                return
            yield batch_inputs(block, mode, epoch if split == 'TRAIN' else 0, device,
                               templates=self.templates)


def batch_inputs(rows, mode, epoch, device, *, templates):
    'Existing four-mode encoding and missingness with new exact joint labels.'
    import torch
    if not rows:
        raise ValueError('Nonempty original row batch required')
    inputs = [optional_missing(r, epoch) for r in rows]
    batch = {key:np.stack([r[key] for r in inputs]) for key in INPUT_KEYS}
    encoded = prepare(batch, mode)
    tensors = {key:torch.as_tensor(encoded[key], device=device)
               for key in ('vector', 'image') if encoded[key] is not None}
    legal = []
    for original, visible in zip(rows, inputs):
        cat = HOST_CLASSES[int(visible['optional'][0])] if visible['known'][0] else None
        legal.append(template_compatibility(*original['geometry'], templates, host_category=cat))
    target = {key:torch.as_tensor(np.stack([r[key] for r in rows]), device=device)
              for key in ('z', 'active', 'coarse', 'template', 'scenario')}
    return tensors, target, torch.as_tensor(np.stack(legal), device=device)
