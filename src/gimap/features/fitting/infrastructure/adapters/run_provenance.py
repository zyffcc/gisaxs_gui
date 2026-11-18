"""Identify the network a 1D Predict run used without importing its numerical runtime."""
from hashlib import sha256


def model_provenance(method: str = "posterior") -> dict:
    """The record of Predict parameters' network (it names the SHA-256 of the weights, checked when they are
    loaded) and its fingerprint."""
    from ...domain.posterior import network
    record = network.path().with_suffix(".json")
    return {"kind": "posterior", "folder": str(record.parent), "manifest": record.name,
            "manifest_sha256": sha256(record.read_bytes()).hexdigest() if record.is_file() else None,
            "identity_scope": "the network's record (it names the SHA-256 of its weights, checked when loaded)"}
