"""Check every toolkit file against MANIFEST.json (sha256), or write the manifest.

    python verify.py            -> VERIFY_OK or the files that differ
    python verify.py --write    -> (re)write MANIFEST.json for the current files (packaging only)
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP = {'MANIFEST.json', 'verify.py'}


def files():
    for p in sorted(ROOT.rglob('*')):
        rel = p.relative_to(ROOT).as_posix()
        if p.is_file() and rel not in SKIP and '__pycache__' not in rel and not rel.startswith('runs/'):
            yield rel, p


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


if __name__ == '__main__':
    if '--write' in sys.argv:
        manifest = {rel: sha(p) for rel, p in files()}
        text = json.dumps(dict(files=manifest), indent=1) + '\n'
        (ROOT / 'MANIFEST.json').write_bytes(text.encode('utf-8'))  # LF on every platform, like the other files
        print('MANIFEST_WRITTEN', len(manifest))
    else:
        want = json.loads((ROOT / 'MANIFEST.json').read_text())['files']
        bad = [rel for rel, digest in want.items() if not (ROOT / rel).is_file() or sha(ROOT / rel) != digest]
        print('VERIFY_OK' if not bad else 'VERIFY_FAILED ' + ' '.join(bad[:20]), len(want))
        sys.exit(1 if bad else 0)
