"""Identify the selected research bundle without importing its optional numerical runtime."""
from hashlib import sha256
from pathlib import Path

from ...application.workflow_v5 import bundled_workflow, fit1d_kit_home


def model_provenance(method: str, model_path=None) -> dict:
    if method == 'experimental':
        return {'kind': 'numerical', 'manifest_sha256': None}
    if method == 'fit1d':
        root, name = fit1d_kit_home(), 'FILES_SHA256.json'
    elif method == 'stable':
        root, name = bundled_workflow().parent / 'stable_blue_rc_v1', 'MANIFEST.json'
    else:
        root, name = Path(model_path or bundled_workflow()), 'conditional_fast_manifest_v2.json'
    manifest = root / name
    return {'kind': method, 'folder': str(root), 'manifest': name,
            'manifest_sha256': sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
            'identity_scope': 'manifest fingerprint; not a per-run verification of every model file'}
