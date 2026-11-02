"""Lossless compact storage of the float32 form-factor tables for shipping (Windows package).

The tables (~/.cache/gisaxs1d_tables/ff_{1,2,3}_number_v2.npy, 24.6 MB each) are smooth log F(x, s)
values; shuffling the four bytes of every float32 into planes before zlib shrinks them ~3.6x
(24.6 -> 6.8 MB) and restores them bit for bit.

    python3 tools/windows/tablepack.py pack OUT_DIR        # .npy in the cache -> OUT_DIR/*.fft
    python3 tools/windows/tablepack.py unpack IN_DIR [CACHE]   # *.fft -> cache (default ~/.cache/gisaxs1d_tables)
"""
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np

CACHE = Path.home()/'.cache'/'gisaxs1d_tables'
NAMES = [f'ff_{t}_number_v2.npy' for t in (1, 2, 3)]
MAGIC = b'GISAXS1D-FFT1\n'


def pack_array(a):
    a = np.ascontiguousarray(a, np.float32)
    header = json.dumps(dict(shape=list(a.shape), dtype='float32', sha256=hashlib.sha256(a.tobytes()).hexdigest())).encode()
    planes = a.view(np.uint8).reshape(-1, 4).T.copy()
    return MAGIC+len(header).to_bytes(4, 'little')+header+zlib.compress(planes.tobytes(), 9)


def unpack_array(blob):
    if not blob.startswith(MAGIC):
        raise ValueError('not a packed table')
    i = len(MAGIC)
    n = int.from_bytes(blob[i:i+4], 'little')
    header = json.loads(blob[i+4:i+4+n])
    planes = np.frombuffer(zlib.decompress(blob[i+4+n:]), np.uint8).reshape(4, -1)
    a = planes.T.copy().view(np.float32).reshape(header['shape'])
    if hashlib.sha256(a.tobytes()).hexdigest() != header['sha256']:
        raise ValueError('table checksum mismatch')
    return a


def pack(out_dir, cache=CACHE):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        blob = pack_array(np.load(Path(cache)/name))
        (out_dir/(name[:-4]+'.fft')).write_bytes(blob)
        print(name, f'{len(blob)/1e6:.1f} MB')


def unpack(in_dir, cache=CACHE):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        target = cache/name
        a = unpack_array((Path(in_dir)/(name[:-4]+'.fft')).read_bytes())
        if target.exists() and np.array_equal(np.load(target), a):
            continue
        np.save(target, a)
        print('wrote', target)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'pack':
        pack(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == 'unpack':
        unpack(sys.argv[2], *(sys.argv[3:4] or [CACHE]))
    else:
        print(__doc__)
