"""Process-live, single-use authority for one audited Phase-B worker."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import hmac
import secrets
from typing import Callable, Mapping
from weakref import WeakKeyDictionary

from .grouped_artifact_v5 import canonical_json


V5_K1_PHASE_B_CAPABILITY_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_b_job_local_capability/v1"
)


class V5K1PhaseBInputCapability:
    """Opaque live object; serialized audit bytes never authorize publication."""

    __slots__ = ("_nonce", "__weakref__")

    def __new__(cls, *args, **kwargs):
        del args, kwargs
        raise TypeError("Phase-B capability is opaque and worker-minted")

    def __reduce__(self):
        raise TypeError("Phase-B capability cannot be serialized")

    def audit_payload(self) -> dict[str, object]:
        return deepcopy(_record(self)["audit"])


_KEY = secrets.token_bytes(32)
_LIVE: WeakKeyDictionary[V5K1PhaseBInputCapability, dict[str, object]] = (
    WeakKeyDictionary()
)


def _sealed(record: Mapping[str, object]) -> bytes:
    return canonical_json(
        {
            "audit": record["audit"],
            "state": record["state"],
            "pre_recheck_sha256": record["pre_recheck_sha256"],
            "post_recheck_sha256": record["post_recheck_sha256"],
        }
    ).encode("utf-8")


def _record(capability: V5K1PhaseBInputCapability) -> dict[str, object]:
    if type(capability) is not V5K1PhaseBInputCapability:
        raise TypeError("unsupported Phase-B capability type")
    record = _LIVE.get(capability)
    if record is None:
        raise RuntimeError("Phase-B capability was not minted by this live worker")
    expected = hmac.digest(_KEY, capability._nonce + _sealed(record), "sha256")
    if not hmac.compare_digest(record["seal"], expected):
        raise RuntimeError("Phase-B capability seal is invalid")
    return record


def _reseal(
    capability: V5K1PhaseBInputCapability, record: dict[str, object]
) -> None:
    record["seal"] = hmac.digest(
        _KEY, capability._nonce + _sealed(record), "sha256"
    )


def _launch_core(launch_chain: Mapping[str, object]) -> dict[str, str]:
    if not isinstance(launch_chain, Mapping):
        raise TypeError("Phase-B launch chain must be a mapping")
    stage = launch_chain.get("launch_stage")
    job_id = launch_chain.get("slurm_job_id")
    plan = launch_chain.get("launch_plan")
    if stage not in {"engineering_smoke", "formal_gate"}:
        raise ValueError("Phase-B launch stage is unsupported")
    if not isinstance(job_id, str) or not job_id.isdigit() or job_id.startswith("0"):
        raise ValueError("Phase-B launch job id is malformed")
    if not isinstance(plan, Mapping):
        raise ValueError("Phase-B launch plan evidence is missing")
    plan_sha = plan.get("plan_sha256")
    if not isinstance(plan_sha, str) or len(plan_sha) != 64:
        raise ValueError("Phase-B launch plan SHA-256 is malformed")
    return {
        "launch_stage": stage,
        "slurm_job_id": job_id,
        "launch_plan_sha256": plan_sha,
    }


def _mint_phase_b_capability(
    launch_chain: Mapping[str, object],
    recheck: Callable[[], None],
) -> V5K1PhaseBInputCapability:
    if not callable(recheck):
        raise TypeError("Phase-B capability recheck must be callable")
    launch = _launch_core(launch_chain)
    recheck()
    core = {
        "schema": V5_K1_PHASE_B_CAPABILITY_SCHEMA,
        **launch,
        "launch_chain_sha256": sha256(
            canonical_json(dict(launch_chain)).encode("utf-8")
        ).hexdigest(),
        "authorization": {
            "live_registry_required": True,
            "single_stage_completion": True,
            "serialized_payload_authorizes_publication": False,
            "rechecks_launch_and_all_original_job_local_inputs": True,
        },
    }
    audit = {
        **core,
        "capability_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }
    capability = object.__new__(V5K1PhaseBInputCapability)
    capability._nonce = secrets.token_bytes(32)
    record = {
        "audit": audit,
        "launch_chain": deepcopy(dict(launch_chain)),
        "recheck": recheck,
        "state": "minted",
        "pre_recheck_sha256": None,
        "post_recheck_sha256": None,
    }
    _reseal(capability, record)
    _LIVE[capability] = record
    return capability


def _recheck_digest(record: Mapping[str, object]) -> str:
    record["recheck"]()
    return sha256(canonical_json(record["launch_chain"]).encode("utf-8")).hexdigest()


def _validate_phase_b_capability(
    capability: V5K1PhaseBInputCapability, *, phase: str
) -> str:
    if phase not in {"pre_execution", "post_execution"}:
        raise ValueError("unsupported Phase-B capability phase")
    record = _record(capability)
    expected = "minted" if phase == "pre_execution" else "pre_validated"
    if record["state"] != expected:
        raise RuntimeError("Phase-B capability is stale, reused, or out of order")
    observed = _recheck_digest(record)
    if observed != record["audit"]["launch_chain_sha256"]:
        raise RuntimeError("Phase-B launch chain changed during capability use")
    if phase == "pre_execution":
        record["pre_recheck_sha256"] = observed
        record["state"] = "pre_validated"
    else:
        if observed != record["pre_recheck_sha256"]:
            raise RuntimeError("Phase-B evidence changed across execution")
        record["post_recheck_sha256"] = observed
        record["state"] = "consumed"
    _reseal(capability, record)
    return observed


def _consumed_phase_b_capability_payload(
    capability: V5K1PhaseBInputCapability,
) -> dict[str, object]:
    record = _record(capability)
    if record["state"] != "consumed":
        raise RuntimeError("Phase-B capability has not completed post-execution validation")
    return {
        "capability": deepcopy(record["audit"]),
        "pre_execution_recheck_sha256": record["pre_recheck_sha256"],
        "post_execution_recheck_sha256": record["post_recheck_sha256"],
        "pre_post_equal": (
            record["pre_recheck_sha256"] == record["post_recheck_sha256"]
        ),
    }


def _recheck_consumed_phase_b_capability(
    capability: V5K1PhaseBInputCapability,
) -> None:
    record = _record(capability)
    if record["state"] != "consumed":
        raise RuntimeError("Phase-B capability is not ready for publication")
    if _recheck_digest(record) != record["post_recheck_sha256"]:
        raise RuntimeError("Phase-B evidence changed before publication")


def _claim_phase_b_completion(
    capability: V5K1PhaseBInputCapability,
) -> dict[str, object]:
    _recheck_consumed_phase_b_capability(capability)
    payload = _consumed_phase_b_capability_payload(capability)
    record = _record(capability)
    record["state"] = "completion_claimed"
    _reseal(capability, record)
    return payload


__all__ = [
    "V5K1PhaseBInputCapability",
    "V5_K1_PHASE_B_CAPABILITY_SCHEMA",
]
