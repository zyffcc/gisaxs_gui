"""Process-live single-use authority for one Phase-A v7 worker stage."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import hmac
from pathlib import Path
import secrets
from typing import Mapping, Sequence
from weakref import WeakKeyDictionary

from .k1_phase_a_contract_v7 import (
    canonical_json,
    portable_identity,
    validate_launch_binding_payload,
)
from .k1_staging_files_v5 import read_only_identity


CAPABILITY_SCHEMA = "gisaxs.posterior_v8.k1_phase_a_job_local_capability/v1"


class V7PhaseAInputCapability:
    __slots__ = ("_nonce", "__weakref__")

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        raise TypeError("Phase-A capability is opaque and worker-minted")

    def __reduce__(self):
        raise TypeError("Phase-A capability cannot be serialized")

    def audit_payload(self) -> dict[str, object]:
        return deepcopy(_record(self)["audit"])


_KEY = secrets.token_bytes(32)
_LIVE: WeakKeyDictionary[V7PhaseAInputCapability, dict[str, object]] = WeakKeyDictionary()


def _sealed(record: Mapping[str, object]) -> bytes:
    return canonical_json(
        {
            "audit": record["audit"],
            "state": record["state"],
            "pre_sha256": record["pre_sha256"],
            "post_sha256": record["post_sha256"],
        }
    ).encode("utf-8")


def _reseal(capability: V7PhaseAInputCapability, record: dict[str, object]) -> None:
    record["seal"] = hmac.digest(_KEY, capability._nonce + _sealed(record), "sha256")


def _record(capability: V7PhaseAInputCapability) -> dict[str, object]:
    if type(capability) is not V7PhaseAInputCapability:
        raise TypeError("unsupported Phase-A capability type")
    record = _LIVE.get(capability)
    if record is None:
        raise RuntimeError("Phase-A capability was not minted by this live worker process")
    expected = hmac.digest(_KEY, capability._nonce + _sealed(record), "sha256")
    if not hmac.compare_digest(record["seal"], expected):
        raise RuntimeError("Phase-A capability seal is invalid")
    return record


def _rehash(rows: Sequence[Mapping[str, object]]) -> str:
    replay = []
    for row in rows:
        identity = portable_identity(
            read_only_identity(Path(str(row["path"])), str(row["role"]))
        )
        if identity != row["identity"]:
            raise RuntimeError(f"Phase-A staged {row['role']} changed")
        replay.append({"role": row["role"], "path": row["path"], "identity": identity})
    return sha256(canonical_json(replay).encode("utf-8")).hexdigest()


def _mint_phase_a_capability(
    launch_binding: Mapping[str, object], staged_rows: Sequence[Mapping[str, object]]
) -> V7PhaseAInputCapability:
    launch_binding = validate_launch_binding_payload(launch_binding)
    rows = deepcopy(list(staged_rows))
    pre = _rehash(rows)
    core = {
        "schema": CAPABILITY_SCHEMA,
        "stage": launch_binding["stage"],
        "slurm_job_id": launch_binding["slurm_job_id"],
        "launch_binding_sha256": launch_binding["binding_sha256"],
        "staged_inputs": rows,
        "pre_mint_rehash_sha256": pre,
        "authorization": {
            "live_registry_required": True,
            "single_use": True,
            "serialized_payload_authorizes_use": False,
            "job_local_read_only_single_link_inputs": True,
        },
    }
    audit = {**core, "capability_sha256": sha256(canonical_json(core).encode()).hexdigest()}
    capability = object.__new__(V7PhaseAInputCapability)
    capability._nonce = secrets.token_bytes(32)
    record = {
        "audit": audit,
        "rows": rows,
        "state": "minted",
        "pre_sha256": None,
        "post_sha256": None,
    }
    _reseal(capability, record)
    _LIVE[capability] = record
    return capability


def _validate_phase_a_capability(
    capability: V7PhaseAInputCapability, *, stage: str, phase: str
) -> str:
    if phase not in {"pre_use", "post_use"}:
        raise ValueError("unsupported Phase-A capability phase")
    record = _record(capability)
    if record["audit"]["stage"] != stage:
        raise RuntimeError("Phase-A capability belongs to another stage")
    expected = "minted" if phase == "pre_use" else "pre_validated"
    if record["state"] != expected:
        raise RuntimeError("Phase-A capability is stale, reused, or out of order")
    observed = _rehash(record["rows"])
    if phase == "pre_use":
        if observed != record["audit"]["pre_mint_rehash_sha256"]:
            raise RuntimeError("Phase-A inputs changed after capability mint")
        record["pre_sha256"] = observed
        record["state"] = "pre_validated"
    else:
        if observed != record["pre_sha256"]:
            raise RuntimeError("Phase-A inputs changed while the stage executed")
        record["post_sha256"] = observed
        record["state"] = "consumed"
    _reseal(capability, record)
    return observed


def _consumed_capability_payload(
    capability: V7PhaseAInputCapability,
) -> dict[str, object]:
    record = _record(capability)
    if record["state"] != "consumed":
        raise RuntimeError("Phase-A capability has not completed post-use validation")
    return {
        "capability": deepcopy(record["audit"]),
        "pre_use_rehash_sha256": record["pre_sha256"],
        "post_use_rehash_sha256": record["post_sha256"],
        "pre_post_equal": record["pre_sha256"] == record["post_sha256"],
    }


def _claim_completion_capability(
    capability: V7PhaseAInputCapability,
) -> dict[str, object]:
    """Spend the consumed capability exactly once on its stage completion."""

    payload = _consumed_capability_payload(capability)
    record = _record(capability)
    record["state"] = "completion_claimed"
    _reseal(capability, record)
    return payload


__all__ = ["CAPABILITY_SCHEMA", "V7PhaseAInputCapability"]
