'Read specified v2 family splits before decoding scientific row values.'
from pathlib import Path
import zipfile
import numpy as np
from .stage5_partial import selected_array, sha


def array_header(archive, key):
    'Read an NPY header only, without materialising its scientific column.'
    with archive.open(key + '.npy') as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError('Unsupported shard NPY version')
    if not shape or fortran or dtype.hasobject:
        raise ValueError('Numeric C-order row arrays required')
    return shape, dtype


def read_split(path, digest, split, expected_shapes, *, allow_test=False):
    'Decode only requested rows, preserving original family and row identity.'
    path = Path(path)
    if split not in ('TRAIN', 'CAL', 'TEST'):
        raise ValueError('Explicit single family split required')
    if split == 'TEST' and not allow_test:
        raise ValueError('TEST decoding requires the independent evaluation caller')
    if not expected_shapes or set(expected_shapes) & {'family_id', 'index'}:
        raise ValueError('Explicit scientific row shapes required')
    if sha(path) != digest:
        raise ValueError('Completed shard SHA differs')
    with zipfile.ZipFile(path) as archive:
        identity_headers = {key: array_header(archive, key) for key in ('family_id', 'index')}
        for shape, dtype in identity_headers.values():
            if len(shape) != 1 or dtype.kind not in 'iu':
                raise ValueError('Integer row identities required')
        family = selected_array(archive, 'family_id')
        index = selected_array(archive, 'index')
        n = len(family)
        if len(index) != n or np.any(family < 0) or np.any(index < 0) or len(np.unique(index)) != n:
            raise ValueError('Invalid/repeated row identity')
        keep = np.flatnonzero(family % 10 < 8 if split == 'TRAIN' else family % 10 == (8 if split == 'CAL' else 9))
        headers = {}


        for key, expected in expected_shapes.items():
            shape, dtype = array_header(archive, key)
            if shape != (n,) + tuple(expected):
                raise ValueError('Scientific row shape differs: ' + key)
            headers[key] = dict(shape=list(shape), dtype=dtype.str)
        arrays = {key: selected_array(archive, key, keep) for key in expected_shapes}
    arrays.update(family_id=family[keep], index=index[keep])
    return arrays, dict(split=split, original_rows=n, selected_rows=len(keep), selected_row_indices=keep.tolist(),
                        scientific_headers=headers, shard_sha256=digest, TEST_scientific_rows_decoded=split == 'TEST')
