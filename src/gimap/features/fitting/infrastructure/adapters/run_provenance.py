"""Identify the network or kit a 1D Predict run used without importing its numerical runtime."""
from hashlib import sha256

from ...application.workflow_v5 import fit1d_kit_home


def model_provenance(method: str) -> dict:
    if method == "experimental":
        return {"kind": "numerical", "manifest_sha256": None}
    if method == "posterior":
        from ...domain.posterior import network
        record = network.path().with_suffix(".json")
        return {"kind": "posterior", "folder": str(record.parent), "manifest": record.name,
                "manifest_sha256": sha256(record.read_bytes()).hexdigest() if record.is_file() else None,
                "identity_scope": "the network's record (it names the SHA-256 of its weights, checked when loaded)"}
    root, name = fit1d_kit_home(), "FILES_SHA256.json"
    manifest = root / name
    return {"kind": method, "folder": str(root), "manifest": name,
            "manifest_sha256": sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
            "identity_scope": "manifest fingerprint; not a per-run verification of every model file"}
