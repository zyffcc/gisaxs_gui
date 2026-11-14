'Read committed TRAIN/CAL rows without interpreting TEST payloads.'
import hashlib
import json
import re
import zipfile
from pathlib import Path
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def selected_array(archive, key, selected=None):
    with archive.open(key+'.npy') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError('Unsupported original NPY version: '+str(version))
        if dtype.hasobject or fortran or not shape:
            raise ValueError('Numeric C-order row array required')
        row_bytes = int(np.prod(shape[1:], dtype=int))*dtype.itemsize
        indices = np.arange(shape[0]) if selected is None else np.asarray(selected, int)
        if np.any(indices < 0) or np.any(indices >= shape[0]) or np.any(np.diff(indices) <= 0):
            raise ValueError('Selected rows must be sorted unique identities')
        base = f.tell()
        out = np.empty((len(indices),)+shape[1:], dtype=dtype)
        for j, i in enumerate(indices):
            f.seek(base+int(i)*row_bytes)
            raw = f.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError('Truncated committed NPY row')
            out[j] = np.frombuffer(raw, dtype=dtype).reshape(shape[1:])
        return out


def metadata(path, allowed):
    'Parse header identities and ONLY allowed record dictionaries.'
    text = Path(path).read_text(encoding='utf-8')
    marker = re.search(r'"records"\s*:\s*\[', text)
    if marker is None:
        raise ValueError('Original shard metadata missing')
    header = json.loads(text[:marker.start()].rstrip().rstrip(',')+'}')
    records = {}; depth = 0; quote = False; escape = False; begin = None
    for pos in range(marker.end(), len(text)):
        ch = text[pos]
        if quote:
            if escape: escape = False
            elif ch == '\\': escape = True
            elif ch == '"': quote = False
            continue
        if ch == '"': quote = True
        elif ch == '{':
            if depth == 0: begin = pos
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                raw = text[begin:pos+1]
                identity = re.search(r'"global_family_id"\s*:\s*(\d+)', raw)
                if identity is None: raise ValueError('Missing original family identity')
                fid = int(identity.group(1))
                if fid in allowed:
                    if fid % 10 == 9: raise ValueError('Forbidden TEST record')
                    row = json.loads(raw)
                    ix = fid*4+int(row['sibling'])
                    if ix in records: raise ValueError('Duplicate original metadata row')
                    records[ix] = row
        elif ch == ']' and depth == 0: break
    return header, records


KEYS = ('index', 'family_id', 'features', 'mandatory', 'optional_truth',
        'primitives', 'shape', 'out_class', 'visible_peak', 'N')


def snapshot(data, out):
    data, out = Path(data), Path(out)
    if out.exists(): raise ValueError('Snapshot output must be fresh')
    arrays = {k: [] for k in KEYS}; qs = []; widths = []; states = []
    sources = []; common = None; original_seen = set(); failed = []
    for task in sorted(data.glob('task[0-9][0-9][0-9]')):
        if int(task.name[4:]) >= 50: continue
        dp = task/'declaration.json'
        d = json.loads(dp.read_text())
        ref = {k:v for k,v in d.items() if k not in ('task_id','assigned_family_range')}
        if common is None: common = ref
        if ref != common or d['siblings'] != 4 or d['seed_base'] != 1500000:
            raise ValueError('Mixed original recipe')
        for jp in sorted(task.glob('shard*.json')):
            if jp.with_suffix('.npz').exists(): continue
            header, _ = metadata(jp, set())
            if header['successes']:


                raise ValueError('Committed successful metadata has no NPZ')
            if header['failures']:
                failed.append({'path':str(jp.relative_to(data)),
                               'failures':header['failures'],
                               'original_indices':header['indices'],
                               'metadata_sha256':sha(jp)})
        for path in sorted(task.glob('shard*.npz')):
            side = path.with_name(path.name+'.sha256'); jp = path.with_suffix('.json')
            if not side.exists() or not jp.exists(): continue
            digest = side.read_text().split()[0]
            if sha(path) != digest: raise ValueError('Committed shard changed')
            with zipfile.ZipFile(path) as z:
                ids = selected_array(z, 'index'); fam = selected_array(z, 'family_id')
                if not np.array_equal(ids//4, fam) or len(np.unique(ids)) != len(ids):
                    raise ValueError('Original identities differ')
                keep = np.flatnonzero(fam % 10 != 9)
                header, records = metadata(jp, set(fam[keep].tolist()))
                if ids.tolist() != header['successful_indices'] or original_seen.intersection(header['indices']):
                    raise ValueError('Successful or duplicate original identities differ')
                original_seen.update(header['indices'])
                if header['failures']: failed.append({'path':str(jp.relative_to(data)), 'failures':header['failures']})
                for k in KEYS:
                    arrays[k].append(selected_array(z, k, keep))
                for ix in ids[keep]:
                    m = records[int(ix)]
                    qs.append(m.get('q_star')); widths.append(m.get('width')); states.append(m.get('support', 'unresolved'))
            sources.append({'path':str(path.relative_to(data)), 'sha256':digest,
                            'metadata_sha256':sha(jp), 'sidecar_sha256':sha(side),
                            'declaration_sha256':sha(dp), 'original_rows':len(header['indices']),
                            'selected_TRAIN_CAL_rows':len(keep)})
    if not sources: raise ValueError('No complete committed shards')
    if failed: raise ValueError('Original failed rows retained; repair diagnosis required: '+repr(failed))
    a = {k:np.concatenate(v) for k,v in arrays.items()}
    a.update(q_star=np.array(qs, float), width=np.array(widths, float), support=np.array(states))
    order = np.argsort(a['index']); a = {k:v[order] for k,v in a.items()}
    if np.any(a['family_id'] % 10 == 9): raise ValueError('TEST decoded into snapshot')
    fid, counts = np.unique(a['family_id'], return_counts=True)
    if np.any(counts != 4): raise ValueError('Incomplete original family')
    out.mkdir(parents=True)
    np.savez_compressed(out/'rows.npz', **a)
    digest = sha(out/'rows.npz'); (out/'rows.npz.sha256').write_text(digest+'  rows.npz\n', encoding='utf-8')
    record = {'schema':'stage5-exploratory-TRAIN-CAL-snapshot-1',
              'status':'completed_snapshot_not_final_bank', 'not_final_candidate':True,
              'declaration':common, 'sources':sources, 'rows_sha256':digest,
              'TRAIN_families':fid[fid%10<8].tolist(), 'CAL_families':fid[fid%10==8].tolist(),
              'TEST_values_decoded':False, 'limitations':'Available completed shard timing can favour faster families; not final random population recovery.'}
    (out/'snapshot.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    return record
