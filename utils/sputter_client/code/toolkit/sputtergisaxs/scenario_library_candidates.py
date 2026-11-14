'Finite TRAIN-library conditional candidates, not a calibrated posterior.'
from collections import OrderedDict
import hashlib
import json
from numbers import Integral
from pathlib import Path
import re
import time
import zipfile

import numpy as np

from .scenario_candidates import (_finite, _matching, _peak_kind,
    build_candidates, observe_candidate_frame)
from .scenario_fixed_af_observation import observe_candidate_frame_fixed_af
from .scenario_chart import (CATALOG, COORDINATES, SCENARIO_NAMES,
    active_coordinates, decode, template_compatibility)
from .scenario_runtime import _array_sha, _frame_identity, _labels, _sha, _source_sha
from .scenario_training import ScenarioBank
from .stage5_partial import selected_array
from .v2_dataset import read_split
from .v2_observation import CountFrame, SHAPE
from .v2_ppc import GATES, STAT_SOURCES

SCHEMA = 'finite_TRAIN_library_conditional_candidates_1'
INDEX_SCHEMA = 'compact_original_TRAIN_joint_library_index_1'
CACHE_SCHEMA = 'bounded_original_TRAIN_mean_cache_1'
POLICY = dict(shortlist_per_scenario=32, tie_break='original_family_id',
    preselection='stored_noisy_q_FWHM_highq_Yoneda_minimax; missing_flags_penalized',
    acceptance='all_measured_features_comparable_and_max_noise_aware_error_le_1',
    weights='equal_over_accepted_particles',
    fallback='all_decodable_shortlisted_particles_if_acceptance_empty',
    geometry='same_metal_energy_within1pct_incidence_within.01degree',
    template='original_TRAIN_template_geometry_and_optional_host_legality',
    support='actual_public_pixel_frames_once',
    fluence='single_global_sum_ratio; Poisson_MLE_if_counts_known_else_shape_moment_only',
    intervals='empirical_conditional_clusters_not_calibrated',
    shortlist_truncation=True, TEST_read=False, structural_release=False)
FIELDS = ('features', 'mandatory', 'geometry', 'z', 'coarse', 'template',
          'scenario', 'active')
SOURCE_FILES = ('sputtergisaxs/scenario_library_candidates.py',
    'sputtergisaxs/scenario_fixed_af_observation.py',
    'sputtergisaxs/scenario_candidates.py', 'sputtergisaxs/scenario_chart.py',
    'sputtergisaxs/scenario_runtime.py', 'sputtergisaxs/v2_observation.py',
    'sputtergisaxs/v2_ppc.py', 'sputtergisaxs/v2_dataset.py',
    'sputtergisaxs/stage5_partial.py', 'sputtergisaxs/stage5_measurement.py',
    'sputtergisaxs/prediction.py', 'sputtergisaxs/features.py',
    'sputtergisaxs/effective_host.py', 'sputtergisaxs/forward_v06.py',
    'sputtergisaxs/scenario_steps.py', 'sputtergisaxs/scenario_template.py',
    'sputtergisaxs/scenario_ts.py', 'sputtergisaxs/v2_scene.py')
DECODER_SOURCES = ('scenario_chart.py','effective_host.py','forward_v06.py',
                   'scenario_steps.py','scenario_template.py','scenario_ts.py','v2_scene.py')


def _write_json(path, value):
    temporary = path.with_name(path.name+'.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
        allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def _save_npz(path, arrays):
    temporary = path.with_name(path.name+'.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _group(scan):
    return '_'.join(str(scan).split('_')[:2])


def _receipts(path, allowed):
    'Decode only original TRAIN receipt dictionaries, never all science.'
    text = Path(path).read_text(encoding='utf8')
    marker = re.search(r'"receipts"\s*:\s*\[', text)
    if marker is None:
        raise ValueError('Original generating receipts are missing')
    depth = 0; quoted = False; escape = False; begin = None; output = {}
    for pos in range(marker.end(), len(text)):
        ch = text[pos]
        if quoted:
            if escape: escape = False
            elif ch == '\\': escape = True
            elif ch == '"': quoted = False
            continue
        if ch == '"': quoted = True
        elif ch == '{':
            if depth == 0: begin = pos
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                raw = text[begin:pos+1]
                match = re.search(r'"id"\s*:\s*(\d+)', raw)
                if match is None: raise ValueError('Original family identity is missing')
                fid = int(match.group(1))
                if fid in allowed:
                    if fid % 10 >= 8 or fid in output:
                        raise ValueError('Non-TRAIN or repeated original receipt')
                    output[fid] = json.loads(raw)
        elif ch == ']' and depth == 0:
            break
    return output


def _gate(observed, name):
    actual = observed.get('values', {}).get(name); value = GATES[name]
    se = observed.get('stat_se', {}).get(name)
    if se is not None:
        if not _finite(se, nonnegative=True) or observed.get('stat_se_source', {}).get(name) not in STAT_SOURCES:
            raise ValueError('Only declared count-statistical SE may widen a feature gate')
        if _finite(actual): value = max(value, 2*float(se)/(1. if name == 'Yoneda' else float(actual)))
    return float(value)


def _match_score(match):
    'Minimax acceptance, with missing measured features kept as failures.'
    ratios = []
    for feature in match['features'].values():
        if feature['comparable']:
            ratios.append(float(feature['error'])/float(feature['effective_gate']))
        elif feature['status'] == '代表图案缺测':
            ratios.append(4.)
    peak = match['peak_measurability']
    if peak['comparable'] and not peak['matching']:
        ratios.append(4.)
    return float(max(ratios)) if ratios else None


class TrainLibraryCandidates:
    'Small original-joint index plus bounded, lazy source-bound means.'
    def __init__(self, bank, *, producer_package=None, shortlist_per_scenario=32):
        source = bank if isinstance(bank, ScenarioBank) else ScenarioBank(bank, package=producer_package)
        self.bank = source; self.base = source.base
        self._budget(shortlist_per_scenario)
        arrays = {k: [] for k in ('family_id',)+FIELDS+('location_file', 'location_row')}
        metadata = {}; shards = []
        windows = {r['key']:r for r in json.loads(source.bindings['support.json'].read_text(encoding='utf8'))['rows']}
        template_by_key = {t['key']:t for t in source.templates}
        for file_id, (path, digest) in enumerate(source.files):
            rows, header = read_split(path, digest, 'TRAIN', {k:source.shapes[k] for k in FIELDS})
            ids = rows['family_id']; receipts = _receipts(path.parent/'run.json', set(map(int, ids)))
            if set(receipts) != set(map(int, ids)):
                raise ValueError('Original TRAIN receipts and scientific row identities differ')
            shards.append(dict(path=path.relative_to(source.base).as_posix(), sha256=digest,
                receipt_path=(path.parent/'run.json').relative_to(source.base).as_posix(),
                receipt_sha256=_sha(path.parent/'run.json')))
            for key in ('family_id',)+FIELDS: arrays[key].append(rows[key])
            arrays['location_file'].append(np.full(len(ids), file_id, dtype=np.int32))
            arrays['location_row'].append(np.array(header['selected_row_indices'], dtype=np.int32))
            for j, fid0 in enumerate(ids):
                fid = int(fid0); rec = receipts[fid]; original = rec['physical']


                E, ai = float(original['E']), float(original['ai'])
                if not np.allclose(rows['geometry'][j], [E,ai], rtol=0, atol=2e-6):
                    raise ValueError('Original receipt and stored geometry differ')
                groups = set(rec['donor_groups']) | set(original.get('donor_groups', []))
                groups.add(windows[rec['support_window']]['group'])
                key = original.get('host_template_key')
                if key:
                    raw = template_by_key[key]['raw_sha256']
                    groups.update(_group(t['scan']) for t in source.templates if t['raw_sha256'] == raw)
                metadata[str(fid)] = dict(E=E, ai=ai, category=original['category'],
                    donor_groups=sorted(groups), support_window=rec['support_window'],
                    noise_seed=original.get('noise_seed'),
                    count_support_contract=original.get('count_support_contract'),
                    metadata_fields_are_not_physical_label_substitutes=True)
        arrays = {k:np.concatenate(v,axis=0) for k,v in arrays.items()}
        order = np.argsort(arrays['family_id'], kind='stable')
        self.arrays = {k:v[order] for k,v in arrays.items()}
        self.metadata = metadata; self.templates = tuple(source.templates); self.keys = tuple(source.keys)
        self.catalog = tuple(source.catalog); self.coordinates = tuple(source.coordinates)
        self.original = dict(bank_receipt=source.receipt, shards=shards,
            producer_package_hint=str(source.package), bank_root_hint=str(source.base),
            TRAIN_ids_sha256=_array_sha(self.arrays['family_id']),
            original_TRAIN_families=len(source.indices('TRAIN')),
            source_scope='original_complete_bank; only_TRAIN_science_decoded',
            CAL_scientific_rows_decoded=False, TEST_scientific_rows_decoded=False)
        self.index_binding = None; self._loader = None; self._cache_binding = None
        self._mean_cache = OrderedDict(); self._verified_files = set()
        self._validate()

    def _budget(self, value):
        if isinstance(value, bool) or not isinstance(value, Integral) or not 1 <= value <= 32:
            raise ValueError('Fixed shortlist budget must be an integer from1 to32 per scenario')
        self.shortlist = int(value)

    def _validate(self):
        a = self.arrays; ids = np.asarray(a['family_id'])
        if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(ids) != self.original['original_TRAIN_families']
                or np.any(ids < 0) or np.any(ids % 10 >= 8) or np.any(np.diff(ids) <= 0)
                or set(self.metadata) != set(map(str,map(int,ids)))
                or _array_sha(ids) != self.original['TRAIN_ids_sha256']):
            raise ValueError('Compact index must preserve the full unique original TRAIN population')
        if self.catalog != tuple(CATALOG) or self.coordinates != tuple(COORDINATES):
            raise ValueError('Original catalog/joint coordinates differ from the executing decoder')
        for name in DECODER_SOURCES:
            decoder_pin = self.original['bank_receipt']['source_pins']['sputtergisaxs/'+name]
            if _source_sha(Path(__file__).with_name(name)) != decoder_pin:
                raise ValueError('Executing physical decoder differs from original generating source: '+name)
        dim = len(self.coordinates); n = len(ids)
        expected = dict(features=(40,), mandatory=(5,), geometry=(2,), z=(dim,),
            active=(dim,), coarse=(), template=(), scenario=(), location_file=(), location_row=())
        for key, shape in expected.items():
            if a[key].shape != (n,)+shape:
                raise ValueError('Compact original TRAIN column shape differs: '+key)
        if (not np.isfinite(a['features']).all() or np.any(a['features'][:,33:37]!=0)
                or not np.isfinite(a['mandatory']).all()
                or a['z'].dtype != np.dtype('float64')
                or a['active'].dtype.kind != 'b'
                or any(a[k].dtype.kind not in 'iu' for k in
                       ('coarse','template','scenario','location_file','location_row'))
                or np.any(a['location_file']<0) or np.any(a['location_file']>=len(self.original['shards']))
                or np.any(a['location_row']<0)):
            raise ValueError('Original finite descriptors/joint dtype or shard locations differ')
        for j, fid in enumerate(ids):
            c,t,s = int(a['coarse'][j]),int(a['template'][j]),int(a['scenario'][j])
            if not 0 <= c < len(self.catalog) or not 0 <= t <= len(self.templates) or not 0 <= s < 4:
                raise ValueError('Original class outside the source-bound catalog')
            if self.catalog[c]['scenario_index'] != s:
                raise ValueError('Original scenario and coarse class differ')
            active = active_coordinates(c,t)
            if not np.array_equal(active,a['active'][j]) or not np.isfinite(a['z'][j,active]).all():
                raise ValueError('Original active joint vector differs')
            info = self.metadata[str(int(fid))]; E,ai = info['E'],info['ai']
            if (a['mandatory'][j,2] not in (0,1) or not np.allclose(a['geometry'][j],[E,ai],rtol=0,atol=2e-6)
                    or not np.allclose(a['mandatory'][j,:2],[(E-14000)/6000,(ai-.425)/.275],rtol=0,atol=2e-6)):
                raise ValueError('Original mandatory input and acquisition geometry differ')
        self._row_by_id = {int(fid):i for i,fid in enumerate(ids)}

    @classmethod
    def from_index(cls, directory, *, mean_loader=None, mean_cache=None, bank_root=None,
                   shortlist_per_scenario=32):
        'Load small TRAIN-only NPZ/JSON; original bank is optional and lazy.'
        directory = Path(directory).resolve(); manifest = directory/'index.json'
        side = manifest.with_name(manifest.name+'.sha256')
        if not side.is_file() or side.read_text(encoding='ascii').strip().split()[0] != _sha(manifest):
            raise ValueError('Compact index manifest SHA sidecar differs')
        info = json.loads(manifest.read_text(encoding='utf8'))
        array_path = directory/info['arrays']['path']
        if (array_path.parent != directory or info.get('schema') != INDEX_SCHEMA
                or info.get('TRAIN_only') is not True or info.get('TEST_scientific_rows_decoded') is not False
                or _sha(array_path) != info['arrays']['sha256']):
            raise ValueError('Bound original TRAIN compact index required')
        self = cls.__new__(cls); self.bank = None
        self.base = Path(bank_root).resolve() if bank_root is not None else None
        self._budget(shortlist_per_scenario)
        with np.load(array_path,allow_pickle=False) as data:
            self.arrays = {k:np.array(data[k],copy=True) for k in ('family_id',)+FIELDS+('location_file','location_row')}
        self.original = info['original']; self.metadata = info['metadata']
        self.catalog = tuple(info['catalog']); self.coordinates = tuple(info['coordinates'])
        self.templates = tuple(info['templates']); self.keys = tuple(t['key'] for t in self.templates)
        self.index_binding = dict(index_sha256=_sha(manifest),arrays_sha256=info['arrays']['sha256'])
        self._loader = mean_loader; self._cache_binding = None
        self._mean_cache = OrderedDict(); self._verified_files = set()
        self._validate()
        if mean_cache is not None: self._load_mean_cache(mean_cache)
        return self

    def write_index(self, directory, *, overwrite=False):
        'Write only small original TRAIN columns and provenance, no means.'
        directory = Path(directory).resolve(); directory.mkdir(parents=True,exist_ok=True)
        path = directory/'index.npz'; manifest = directory/'index.json'
        if not overwrite and (path.exists() or manifest.exists()):
            raise FileExistsError('Compact index already exists; explicit overwrite required')
        _save_npz(path,self.arrays)
        info = dict(schema=INDEX_SCHEMA, arrays=dict(path=path.name,sha256=_sha(path)),
            original=self.original, metadata=self.metadata, catalog=self.catalog,
            coordinates=self.coordinates, templates=self.templates, TRAIN_only=True,
            original_joint_preserved=True, mean_images_copied=False,
            TEST_scientific_rows_decoded=False, structural_release=False)
        _write_json(manifest,info)
        manifest.with_name(manifest.name+'.sha256').write_text(_sha(manifest)+'\n',encoding='ascii')
        return dict(index_json=str(manifest),index_npz=str(path),index_sha256=_sha(manifest),
                    arrays_sha256=_sha(path),bytes=path.stat().st_size+manifest.stat().st_size)

    def _get_mean(self, fid):
        fid = int(fid)
        if fid not in self._row_by_id or fid % 10 >= 8:
            raise ValueError('Only indexed original TRAIN means may be read')
        if fid in self._mean_cache:
            self._mean_cache.move_to_end(fid); return self._mean_cache[fid]
        i = self._row_by_id[fid]; a = self.arrays
        item = self.original['shards'][int(a['location_file'][i])]
        source = dict(kind='sealed_original_TRAIN_mean', family_id=fid,
            original_shard_sha256=item['sha256'], original_row=int(a['location_row'][i]),
            stored_mean_precision='original_float32', original_joint_preserved=True)
        if self.base is not None:
            path = (self.base/item['path']).resolve()
            if self.base not in path.parents:
                raise ValueError('Original mean path escapes its explicit bank root')
            if path not in self._verified_files:
                if _sha(path) != item['sha256']: raise ValueError('Original sealed mean shard changed')
                self._verified_files.add(path)
            with zipfile.ZipFile(path) as archive:
                raw = selected_array(archive,'mean_rate',[int(a['location_row'][i])])[0]
                identity = selected_array(archive,'family_id',[int(a['location_row'][i])])[0]
            if int(identity) != fid: raise ValueError('Lazy original mean family differs')
            source['verification'] = 'original_shard_SHA_and_row_family'
        elif self._loader is not None:
            loaded = self._loader(fid)
            if isinstance(loaded,tuple) and len(loaded)==2:
                raw,receipt = loaded; source['supplied_loader_receipt'] = receipt
            else: raw = loaded
            source['verification'] = 'caller_supplied_original_mean; caller_responsible_for_source_binding'
        else:
            raise ValueError('Original mean unavailable; provide bounded mean cache, bank_root or mean_loader')
        raw = np.asarray(raw)
        if raw.shape != SHAPE or not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError('Original mean must be a finite nonnegative two-sided native AF/q rate')
        source['mean_array_sha256'] = _array_sha(raw)
        value = (raw.astype(float),source); self._mean_cache[fid] = value
        if len(self._mean_cache)>128: self._mean_cache.popitem(last=False)
        return value

    def write_mean_cache(self, family_ids, directory, *, overwrite=False):
        'Explicit bounded cache only (max128 rows), not full-bank copying.'
        ids = sorted(set(map(int,family_ids)))
        if not ids or len(ids)>128 or any(fid not in self._row_by_id for fid in ids):
            raise ValueError('Choose1 to128 unique indexed TRAIN family IDs')
        directory = Path(directory).resolve(); directory.mkdir(parents=True,exist_ok=True)
        path = directory/'means.npz'; manifest = directory/'means.json'
        if not overwrite and (path.exists() or manifest.exists()):
            raise FileExistsError('Mean cache already exists; explicit overwrite required')
        means = []; sources = {}
        for fid in ids:
            mean,source = self._get_mean(fid); means.append(mean); sources[str(fid)] = source
        _save_npz(path,dict(family_id=np.asarray(ids,np.int64),mean_rate=np.stack(means)))
        _write_json(manifest,dict(schema=CACHE_SCHEMA,bank_receipt=self.original['bank_receipt'],
            original_TRAIN_ids_sha256=self.original['TRAIN_ids_sha256'],
            arrays=dict(path=path.name,sha256=_sha(path)),family_ids=ids,sources=sources,
            TEST_scientific_rows_decoded=False, structural_release=False))
        manifest.with_name(manifest.name+'.sha256').write_text(_sha(manifest)+'\n',encoding='ascii')
        return dict(manifest=str(manifest),arrays=str(path),bytes=path.stat().st_size+manifest.stat().st_size)

    def _load_mean_cache(self, directory):
        directory = Path(directory).resolve(); path = directory/'means.json'
        side = path.with_name(path.name+'.sha256')
        if not side.is_file() or side.read_text(encoding='ascii').strip().split()[0] != _sha(path):
            raise ValueError('Bounded mean-cache manifest SHA sidecar differs')
        info = json.loads(path.read_text(encoding='utf8')); arrays = directory/info['arrays']['path']
        if (arrays.parent != directory or info.get('schema') != CACHE_SCHEMA
                or info.get('bank_receipt') != self.original['bank_receipt']
                or info.get('original_TRAIN_ids_sha256') != self.original['TRAIN_ids_sha256']
                or info.get('TEST_scientific_rows_decoded') is not False
                or _sha(arrays) != info['arrays']['sha256']):
            raise ValueError('Bound compact mean cache differs from original TRAIN index')
        with np.load(arrays,allow_pickle=False) as data:
            ids = data['family_id']; means = data['mean_rate']
            if ids.tolist()!=info['family_ids'] or len(ids)>128 or means.shape!=(len(ids),)+SHAPE:
                raise ValueError('Bounded original mean-cache identities/shape differ')
            for fid0,mean in zip(ids,means):
                fid = int(fid0)
                if fid not in self._row_by_id or fid%10>=8 or not np.isfinite(mean).all() or np.any(mean<0):
                    raise ValueError('Mean cache contains invalid/non-TRAIN science')
                source = dict(info['sources'][str(fid)],cached_mean_array_sha256=_array_sha(mean),
                    verification='original_mean_cache_SHA_and_bank_lineage')
                self._mean_cache[fid] = (mean.astype(float),source)
        self._cache_binding = dict(manifest_sha256=_sha(path),arrays_sha256=info['arrays']['sha256'])

    def _rank(self, indices, observed):
        'Stored descriptors shortlist only; actual support is always rechecked.'
        features = self.arrays['features'][indices].astype(float); b = observed.get('AF_band')
        values = dict(Yoneda=features[:,8]); flags = dict(Yoneda=features[:,28]==1)
        if b is not None:
            if isinstance(b,bool) or not isinstance(b,Integral) or int(b) not in range(4):
                raise ValueError('Original observed AF band must be0..3 or missing')
            b = int(b)
            with np.errstate(over='ignore',invalid='ignore'):
                q = np.exp(features[:,2*b]); width = q*np.exp(features[:,2*b+1])
                values.update(q_star=q,FWHM=width,highq_ratio=np.exp(features[:,9+b]))
            flags.update(q_star=features[:,20+2*b]==1,
                FWHM=(features[:,20+2*b]==1)&(features[:,21+2*b]==1),
                highq_ratio=features[:,29+b]==1)
        errors = []; names = []; missing = np.zeros(len(indices),int)
        for name in ('q_star','FWHM','highq_ratio','Yoneda'):
            actual = observed.get('values',{}).get(name)
            if not _finite(actual) or name not in values: continue
            usable = flags[name]&np.isfinite(values[name])&(values[name]>0)
            err = np.abs(values[name]-actual)/(1. if name=='Yoneda' else actual)/_gate(observed,name)
            errors.append(np.where(usable,err,4.)); missing += ~usable; names.append(name)
        peak = _peak_kind(observed)
        if peak=='no_resolved_peak' and b is not None:


            errors.append(np.where(features[:,20+2*b]==1,4.,0.)); names.append('known_no_peak_preference')
        maximum = np.max(np.stack(errors,axis=1),axis=1) if errors else np.zeros(len(indices))
        squared = np.sum(np.stack(errors,axis=1)**2,axis=1) if errors else np.zeros(len(indices))
        order = np.lexsort((self.arrays['family_id'][indices],squared,missing,maximum))[:self.shortlist]
        return indices[order], dict(features=names, original_legal_families=len(indices),
            selected_families=len(order), insufficient_measured_ranking_information=not bool(errors),
            stored_features_are_not_actual_support_acceptance=True), maximum[order]

    def query(self, full, public, E, ai, metal, *, host_category=None,
              exclude_groups=(), observed=None, max_per_scenario=3):
        'Return candidates, original pools, per-row matches and ABC diagnostics.'
        begin = time.perf_counter(); identity = _frame_identity(full,public)
        if metal not in ('Au','Ag') or not np.isfinite([E,ai]).all() or min(E,ai)<=0:
            raise ValueError('Positive required energy/incidence and Au/Ag metal')
        observed = (observe_candidate_frame(public,estimate_stat_se=identity['photon_statistics_known'])
                    if observed is None else observed)
        if not isinstance(observed,dict) or not isinstance(observed.get('values'),dict):
            raise ValueError('Original observed feature descriptor required')
        for name in GATES: _gate(observed,name)
        a = self.arrays; ids = a['family_id']; exclude = set(map(str,exclude_groups))
        legality = template_compatibility(float(E),float(ai),self.templates,host_category=host_category)
        geom = a['geometry'].astype(float)
        allowed = ((a['mandatory'][:,2]==(1 if metal=='Au' else 0))
            &(np.abs(geom[:,0]/float(E)-1)<=.01)&(np.abs(geom[:,1]-float(ai))<=.01))
        allowed &= legality[a['coarse'].astype(int),a['template'].astype(int)]
        allowed &= np.asarray([not exclude.intersection(self.metadata[str(int(fid))]['donor_groups']) for fid in ids])
        sampled = dict(candidate_source='原TRAIN模拟库有限特征条件集合；未校准ABC式近似',
            receipt=dict(kind=SCHEMA,input_mode='finite_TRAIN_feature_library'), frame_identity=identity,
            input_sources=dict(posterior_input_mode='原TRAIN库：实测可用特征；完整联合物理行',
                reference_frame='本入口未以未核实的前态推断零金属',
                temporal_continuation='本入口未自动利用相邻帧；需另外报告峰移动证据'),
            scenarios={}, structural_release=False)
        predictive = dict(scenarios={},match_kind='finite_library_profiled_clean_means_not_full_PPC',
            frame_identity=identity,structural_release=False)
        reports = {}; all_records = {}
        for s,scene in enumerate(SCENARIO_NAMES):
            applicable = not (host_category=='Si' and scene in ('S2','S3'))
            if not applicable:
                sampled['scenarios'][scene] = dict(applicable=False,
                    not_applicable_reason='明确裸Si不采用宿主模板或渗入情景',original_draws=0,draws=[])
                predictive['scenarios'][scene] = dict(draws=[])
                reports[scene] = dict(applicable=False,reason='known_Si_excludes_S2_S3')
                continue
            indices = np.flatnonzero(allowed&(a['scenario']==s))
            selected,ranking,stored_scores = self._rank(indices,observed)
            draws = []; descriptions = []; checks = []
            for ordinal,(i,stored_score) in enumerate(zip(selected,stored_scores)):
                fid = int(ids[i]); info = self.metadata[str(fid)]; c,t = int(a['coarse'][i]),int(a['template'][i])
                draw = dict(index=ordinal,family_id=fid,source_family_id=fid,
                    candidate_source='原TRAIN族'+str(fid)+'的完整联合条件样本',decode_failed=False)
                desc = None; match = None; score = None
                try:
                    physical = decode(a['z'][i],c,t,metal=metal,E=info['E'],ai=info['ai'],
                        template_keys=self.keys,template_metadata=self.templates)
                    if physical['scenario']!=scene: raise ValueError('Original scenario changed during joint decode')
                    physical.update(category=info['category'],source_family_id=fid,
                        donor_groups=info['donor_groups'],noise_seed=info['noise_seed'],
                        count_support_contract=info['count_support_contract'])
                    draw.update(physical=physical,conditional_labels=_labels(physical))
                except (ValueError,TypeError,KeyError,OverflowError,FloatingPointError) as error:
                    draw.update(decode_failed=True,physical=None,reason=str(error))
                try:
                    if draw['decode_failed']: raise ValueError('Original joint decode failed: '+draw['reason'])
                    rate,mean_source = self._get_mean(fid)
                    raw_mu = rate*public.P
                    total = float(raw_mu.sum()); photons = float(public.Y.sum())
                    if not np.isfinite(raw_mu).all() or total<=0 or photons<=0:
                        raise ValueError('原mean在实测有效支持上无正强度或实测无正计数')
                    gain = photons/total; mu = gain*raw_mu
                    desc = observe_candidate_frame_fixed_af(CountFrame(mu,public.P,False),
                        AF_band=observed.get('AF_band'),estimate_stat_se=False)
                    desc.update(index=ordinal,family_id=fid,mean_source=mean_source,
                        fitted_scalar_gain=float(gain),count_support_applications=1,
                        gain_kind='Poisson_global_scalar_MLE' if identity['photon_statistics_known'] else
                            'shape_moment_scalar; not PoissonMLE',
                        intensity_alignment_only=not identity['photon_statistics_known'],
                        photon_statistics_inferred_from_gain=False,
                        original_joint_modified=False,posterior_predictive_replica=False,
                        representative_clean_mean=True,raw_expected_photons_sha256=_array_sha(raw_mu),
                        profiled_mean_photons_sha256=_array_sha(mu),
                        original_geometry=[info['E'],info['ai']],query_geometry=[float(E),float(ai)],
                        geometry_is_tolerance_matched_not_recomputed=bool(info['E']!=E or info['ai']!=ai))
                    match = _matching(observed,desc); score = _match_score(match)
                except (ValueError,TypeError,KeyError,OverflowError,FloatingPointError,RuntimeError,OSError,MemoryError) as error:
                    desc = dict(index=ordinal,family_id=fid,forward_failed=True,
                        AF_band=observed.get('AF_band'),values={k:None for k in GATES},
                        feature_missing_reasons={k:'尚待模拟核对：'+str(error) for k in GATES},
                        failure_reason=str(error),posterior_predictive_replica=False)
                    match = _matching(observed,desc)
                accepted = bool(not draw['decode_failed'] and match['matches_measured_features']
                    and score is not None and score<=1+1e-12)
                checks.append(dict(index=ordinal,family_id=fid,decode_failed=draw['decode_failed'],
                    actual_support_checked=not desc['forward_failed'],accepted=accepted,
                    maximum_normalized_error=score,stored_shortlist_error=float(stored_score),matching=match,
                    failure_reason=desc.get('failure_reason',draw.get('reason'))))
                draws.append(draw); descriptions.append(desc)
            accepted_indices = [r['index'] for r in checks if r['accepted']]
            fallback = not accepted_indices
            chosen = list(range(len(draws))) if fallback else accepted_indices
            joint = {k:np.array(a[k][selected[chosen]],copy=True) for k in ('z','coarse','template')}


            selected_draws = [dict(draws[j],index=k,original_shortlist_index=j) for k,j in enumerate(chosen)]
            selected_desc = [dict(descriptions[j],index=k,original_shortlist_index=j) for k,j in enumerate(chosen)]
            source_text = ('原TRAIN库最近完整集合；尚无接受粒子，非数据验证结构' if fallback else
                '原TRAIN库接受集合的等权ABC式条件近似；非校准联合后验')
            for record in selected_draws:
                record['candidate_source'] = source_text+'（原族'+str(record['family_id'])+'）'
            sampled['scenarios'][scene] = dict(applicable=True,original_draws=len(selected_draws),
                draws=selected_draws,original_joint=joint,candidate_source=source_text,
                library_all_original_shortlist_draws=len(draws),library_fallback=fallback)
            predictive['scenarios'][scene] = dict(draws=selected_desc,
                original_joint_sha256={k:_array_sha(v) for k,v in joint.items()})
            reports[scene] = dict(applicable=True,ranking=ranking,original_shortlist=len(draws),
                accepted_particles=len(accepted_indices),ESS=len(accepted_indices),
                displayed_ensemble_particles=len(selected_draws),fallback=fallback,
                approximate_ABC_only=not fallback,calibrated_posterior=False,
                empirical_interval_degenerate_possible=len(selected_draws)<3,
                shortlist_failures=sum(r['decode_failed'] or not r['actual_support_checked'] for r in checks),
                accepted_family_ids=[r['family_id'] for r in checks if r['accepted']],
                selected_ensemble_family_ids=[d['family_id'] for d in selected_draws],
                accepted_equal_weights=[1/len(accepted_indices)]*len(accepted_indices) if accepted_indices else [],
                fallback_equal_weights=[1/len(selected_draws)]*len(selected_draws) if fallback and selected_draws else [],
                checks=checks, original_joint_sha256={k:_array_sha(v) for k,v in joint.items()},
                no_capacity_claim_from_empty_bounded_shortlist=True,structural_release=False)
            all_records[scene] = dict(draws=draws,predictive=descriptions)
        candidates = build_candidates(sampled,observed,predictive,max_per_scenario=max_per_scenario)
        for scene,part in candidates['scenarios'].items():
            original_by_index = {d['index']:d for d in sampled['scenarios'][scene]['draws']}
            for candidate in part['candidates']:
                candidate['representative_family_id'] = original_by_index[candidate['representative_index']]['family_id']
                candidate['library_ensemble'] = {k:reports[scene][k] for k in
                    ('accepted_particles','ESS','fallback','displayed_ensemble_particles','calibrated_posterior')}
                candidate['interval_calibration_status'] = '未校准；有限条件集合内区间'
        execution = {relative:_source_sha(Path(__file__).resolve().parent.parent/relative) for relative in SOURCE_FILES}
        binding = dict(schema=SCHEMA,policy=dict(POLICY,shortlist_per_scenario=self.shortlist),
            original=self.original,index_binding=self.index_binding,mean_cache_binding=self._cache_binding,
            execution_source_pins=execution,frame_identity=identity,
            query_geometry=dict(E=float(E),ai=float(ai),metal=metal,host_category=host_category or 'unknown'),
            excluded_donor_groups=sorted(exclude),observed_AF_band=observed.get('AF_band'),
            observed_peak_kind=_peak_kind(observed),
            missing_mode='all_known_measured_features_only; unknown_absence_never_a_match',
            gain_kind='Poisson_global_scalar_MLE' if identity['photon_statistics_known'] else
                'shape_moment_scalar; not PoissonMLE',
            original_TRAIN_families=len(ids),CAL_science_read=False,TEST_read=False,
            global_intensity_profiled=any(r['actual_support_checked'] for p in reports.values()
                for r in p.get('checks',[])),posterior_predictive_check=False,structural_release=False)
        return dict(schema=SCHEMA,candidates=candidates,sampled=sampled,predictive=predictive,
            observed=observed,acceptance=reports,developer_original_shortlist=all_records,
            binding=binding,elapsed_s=time.perf_counter()-begin,
            status='needs_prior_candidates' if candidates['status']=='needs_prior_candidates'
                else 'conditional_library_candidates',structural_release=False)
