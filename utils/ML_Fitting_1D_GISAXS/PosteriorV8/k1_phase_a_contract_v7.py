"""Strict launch and completion contracts for the K1 Phase-A v7 DAG."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from .k1_staging_files_v5 import read_only_identity, read_only_json


PHASE_A_PLAN_SCHEMA = "gisaxs.posterior_v8.maxwell_k1_phase_a_dag_launch/v7"
PHASE_A_PLAN_VERSION = "posterior_v8_transactional_evidence_bound_phase_a_dag_v7"
PHASE_A_RECEIPT_SCHEMA = "gisaxs.posterior_v8.k1_phase_a_submission_receipt/v1"
PHASE_A_RELEASE_SCHEMA = "gisaxs.posterior_v8.k1_phase_a_release_completion/v1"
PHASE_A_FAILURE_SCHEMA = "gisaxs.posterior_v8.k1_phase_a_launch_failure/v1"
PHASE_A_COMPLETION_SCHEMA = "gisaxs.posterior_v8.k1_phase_a_stage_completion/v1"
PHASE_A_LAUNCH_BINDING_SCHEMA = "gisaxs.posterior_v8.k1_phase_a_launch_binding/v1"

STAGES = (
    "regression",
    "cross_platform_gate",
    "smoke_dataset",
    "smoke_gate",
    "full_dataset",
    "full_gate",
)
DEPENDENCY = {
    "regression": None,
    "cross_platform_gate": "regression",
    "smoke_dataset": "cross_platform_gate",
    "smoke_gate": "smoke_dataset",
    "full_dataset": "smoke_gate",
    "full_gate": "full_dataset",
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_JOB_ID = re.compile(r"[1-9][0-9]*\Z")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def job_id(value: object, name: str) -> str:
    if not isinstance(value, str) or _JOB_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a positive Slurm job id")
    return value


def self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    payload = deepcopy(dict(core))
    payload[field] = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return payload


def _validate_self_hash(
    value: Mapping[str, object], field: str, name: str
) -> dict[str, object]:
    payload = deepcopy(dict(value))
    supplied = digest(payload.pop(field, None), f"{name} {field}")
    if supplied != sha256(canonical_json(payload).encode("utf-8")).hexdigest():
        raise ValueError(f"{name} self hash does not reproduce")
    return {**payload, field: supplied}


def validate_plan(value: Mapping[str, object]) -> dict[str, object]:
    payload = _validate_self_hash(value, "plan_sha256", "Phase-A plan")
    if payload.get("schema_version") != PHASE_A_PLAN_SCHEMA:
        raise ValueError("Phase-A v7 rejects legacy or unsupported launch plans")
    if payload.get("version") != PHASE_A_PLAN_VERSION:
        raise ValueError("unsupported Phase-A plan version")
    if payload.get("stage_order") != list(STAGES):
        raise ValueError("Phase-A stage order drifted")
    jobs = payload.get("jobs")
    layout = payload.get("layout")
    if not isinstance(jobs, Mapping) or set(jobs) != set(STAGES):
        raise ValueError("Phase-A plan job inventory drifted")
    if not isinstance(layout, Mapping):
        raise ValueError("Phase-A plan layout is missing")
    for stage in STAGES:
        job = jobs[stage]
        if not isinstance(job, Mapping) or job.get("depends_on") != DEPENDENCY[stage]:
            raise ValueError(f"Phase-A {stage} dependency drifted")
        if layout.get(f"{stage}_completion") is None:
            raise ValueError(f"Phase-A {stage} completion path is missing")
    for name in ("plan", "receipt", "release_completion", "failure_audit"):
        if not isinstance(layout.get(name), str):
            raise ValueError(f"Phase-A layout {name} is missing")
    return payload


def validate_receipt(value: Mapping[str, object], plan: Mapping[str, object]) -> dict[str, object]:
    expected_plan = validate_plan(plan)
    payload = _validate_self_hash(value, "receipt_sha256", "Phase-A submission receipt")
    if (
        payload.get("schema") != PHASE_A_RECEIPT_SCHEMA
        or payload.get("status") != "ALL_JOBS_HELD"
        or payload.get("plan_sha256") != expected_plan["plan_sha256"]
    ):
        raise ValueError("Phase-A submission receipt identity drifted")
    jobs = payload.get("job_ids")
    if not isinstance(jobs, Mapping) or set(jobs) != set(STAGES):
        raise ValueError("Phase-A submission receipt job inventory drifted")
    normalized = {stage: job_id(jobs[stage], f"{stage} job id") for stage in STAGES}
    if len(set(normalized.values())) != len(STAGES):
        raise ValueError("Phase-A submission receipt reuses a Slurm job id")
    expected_edges = [
        [DEPENDENCY[stage], stage, normalized[DEPENDENCY[stage]], normalized[stage]]
        for stage in STAGES
        if DEPENDENCY[stage] is not None
    ]
    if payload.get("dependency_edges") != expected_edges:
        raise ValueError("Phase-A submission receipt dependency edges drifted")
    snapshots = payload.get("held_scheduler_snapshots")
    if not isinstance(snapshots, Mapping) or set(snapshots) != set(STAGES):
        raise ValueError("Phase-A submission receipt lacks scheduler hold evidence")
    for stage in STAGES:
        snapshot = snapshots[stage]
        dependency = DEPENDENCY[stage]
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("job_id") != normalized[stage]
            or snapshot.get("job_state") != "PENDING"
            or snapshot.get("reason") != "JobHeldUser"
            or (
                dependency is None
                and snapshot.get("dependency") is not None
            )
            or (
                dependency is not None
                and not str(snapshot.get("dependency", "")).startswith(
                    f"afterok:{normalized[dependency]}"
                )
            )
        ):
            raise ValueError(f"Phase-A {stage} scheduler hold evidence drifted")
    return payload


def validate_release(
    value: Mapping[str, object],
    plan: Mapping[str, object],
    receipt: Mapping[str, object],
) -> dict[str, object]:
    expected_plan = validate_plan(plan)
    expected_receipt = validate_receipt(receipt, expected_plan)
    payload = _validate_self_hash(value, "release_sha256", "Phase-A release completion")
    if (
        payload.get("schema") != PHASE_A_RELEASE_SCHEMA
        or payload.get("status") != "ALL_JOBS_RELEASED"
        or payload.get("plan_sha256") != expected_plan["plan_sha256"]
        or payload.get("receipt_sha256") != expected_receipt["receipt_sha256"]
        or payload.get("release_order") != list(reversed(STAGES))
    ):
        raise ValueError("Phase-A release completion identity drifted")
    released = payload.get("released_job_ids")
    expected_ids = [expected_receipt["job_ids"][stage] for stage in reversed(STAGES)]
    if released != expected_ids:
        raise ValueError("Phase-A released job inventory drifted")
    attempts = payload.get("release_attempts")
    if not isinstance(attempts, list) or len(attempts) != len(STAGES):
        raise ValueError("Phase-A release attempts are incomplete")
    for stage, attempt in zip(reversed(STAGES), attempts, strict=True):
        if not isinstance(attempt, Mapping):
            raise ValueError("Phase-A release attempt is invalid")
        snapshot = attempt.get("scheduler_snapshot")
        if (
            attempt.get("stage") != stage
            or attempt.get("job_id") != expected_receipt["job_ids"][stage]
            or attempt.get("returncode") != 0
            or not isinstance(snapshot, Mapping)
            or snapshot.get("job_id") != expected_receipt["job_ids"][stage]
            or snapshot.get("reason") == "JobHeldUser"
        ):
            raise ValueError(f"Phase-A {stage} release evidence drifted")
    return payload


def portable_identity(identity: Mapping[str, object]) -> dict[str, object]:
    return {
        name: identity[name]
        for name in (
            "sha256",
            "byte_count",
            "mode_octal",
            "uid",
            "gid",
            "link_count",
        )
    }


def validate_stage_completion(
    value: Mapping[str, object],
    *,
    expected_stage: str,
    plan: Mapping[str, object],
    receipt: Mapping[str, object],
    release: Mapping[str, object],
    verify_artifacts: bool = True,
) -> dict[str, object]:
    if expected_stage not in STAGES:
        raise ValueError("unsupported Phase-A completion stage")
    expected_plan = validate_plan(plan)
    expected_receipt = validate_receipt(receipt, expected_plan)
    expected_release = validate_release(release, expected_plan, expected_receipt)
    payload = _validate_self_hash(value, "completion_sha256", "Phase-A stage completion")
    if (
        payload.get("schema") != PHASE_A_COMPLETION_SCHEMA
        or payload.get("status") != "COMPLETE"
        or payload.get("stage") != expected_stage
        or payload.get("plan_sha256") != expected_plan["plan_sha256"]
        or payload.get("receipt_sha256") != expected_receipt["receipt_sha256"]
        or payload.get("release_sha256") != expected_release["release_sha256"]
        or payload.get("slurm_job_id") != expected_receipt["job_ids"][expected_stage]
    ):
        raise ValueError("Phase-A stage completion launch binding drifted")
    launch = payload.get("launch_binding")
    if not isinstance(launch, Mapping):
        raise ValueError("Phase-A stage completion has no launch binding")
    launch = validate_launch_binding_payload(launch, expected_stage=expected_stage)
    if (
        launch.get("schema") != PHASE_A_LAUNCH_BINDING_SCHEMA
        or launch.get("status") != "VALIDATED"
        or launch.get("stage") != expected_stage
        or launch.get("slurm_job_id") != payload["slurm_job_id"]
        or launch.get("plan_sha256") != payload["plan_sha256"]
        or launch.get("receipt_sha256") != payload["receipt_sha256"]
        or launch.get("release_sha256") != payload["release_sha256"]
        or launch.get("binding_sha256") != payload.get("launch_binding_sha256")
    ):
        raise ValueError("Phase-A completion embedded launch binding drifted")
    capability = payload.get("job_local_capability")
    if (
        not isinstance(capability, Mapping)
        or capability.get("pre_post_equal") is not True
        or not isinstance(capability.get("capability"), Mapping)
        or capability["capability"].get("launch_binding_sha256")
        != launch["binding_sha256"]
        or capability["capability"].get("stage") != expected_stage
        or capability["capability"].get("slurm_job_id") != payload["slurm_job_id"]
    ):
        raise ValueError("Phase-A completion capability binding drifted")
    upstream = DEPENDENCY[expected_stage]
    expected_upstream_id = None if upstream is None else expected_receipt["job_ids"][upstream]
    if payload.get("upstream_stage") != upstream or payload.get("upstream_job_id") != expected_upstream_id:
        raise ValueError("Phase-A stage completion upstream binding drifted")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Phase-A stage completion artifact inventory is invalid")
    if verify_artifacts:
        for item in artifacts:
            if not isinstance(item, Mapping) or set(item) != {"role", "path", "identity"}:
                raise ValueError("Phase-A completion artifact row is invalid")
            observed = portable_identity(
                read_only_identity(Path(str(item["path"])), f"Phase-A {item['role']}")
            )
            if observed != item["identity"]:
                raise ValueError(f"Phase-A {item['role']} identity changed")
    return payload


def load_launch_transaction(
    plan_path: Path,
    receipt_path: Path,
    release_path: Path,
    *,
    expected_plan_sha256: str,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    plan, plan_file = read_only_json(plan_path, "Phase-A launch plan")
    plan = validate_plan(plan)
    if plan["plan_sha256"] != digest(expected_plan_sha256, "expected plan SHA-256"):
        raise ValueError("Phase-A plan differs from the Slurm export binding")
    receipt, receipt_file = read_only_json(receipt_path, "Phase-A submission receipt")
    receipt = validate_receipt(receipt, plan)
    release, release_file = read_only_json(release_path, "Phase-A release completion")
    release = validate_release(release, plan, receipt)
    identities = {
        "plan": portable_identity(plan_file),
        "receipt": portable_identity(receipt_file),
        "release_completion": portable_identity(release_file),
    }
    return plan, receipt, release, identities


def launch_binding(
    *,
    stage: str,
    slurm_job_id: str,
    plan: Mapping[str, object],
    receipt: Mapping[str, object],
    release: Mapping[str, object],
    transaction_files: Mapping[str, object],
    upstream_completion: Mapping[str, object] | None,
    upstream_completion_file: Mapping[str, object] | None,
) -> dict[str, object]:
    if stage not in STAGES:
        raise ValueError("unsupported Phase-A stage")
    current = job_id(slurm_job_id, "current Slurm job id")
    if receipt["job_ids"][stage] != current:
        raise ValueError("current Slurm job id differs from the submission receipt")
    upstream = DEPENDENCY[stage]
    if upstream is None:
        if upstream_completion is not None or upstream_completion_file is not None:
            raise ValueError("regression must not have an upstream completion")
        upstream_payload = None
    else:
        if upstream_completion is None or upstream_completion_file is None:
            raise ValueError(f"Phase-A {stage} requires its upstream completion")
        validated = validate_stage_completion(
            upstream_completion,
            expected_stage=upstream,
            plan=plan,
            receipt=receipt,
            release=release,
        )
        upstream_payload = {
            "stage": upstream,
            "job_id": receipt["job_ids"][upstream],
            "completion_sha256": validated["completion_sha256"],
            "file": portable_identity(upstream_completion_file),
        }
    core = {
        "schema": PHASE_A_LAUNCH_BINDING_SCHEMA,
        "status": "VALIDATED",
        "stage": stage,
        "slurm_job_id": current,
        "plan_sha256": plan["plan_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
        "release_sha256": release["release_sha256"],
        "transaction_files": deepcopy(dict(transaction_files)),
        "upstream": upstream_payload,
    }
    return self_hashed(core, "binding_sha256")


def validate_launch_binding_payload(
    value: Mapping[str, object], *, expected_stage: str | None = None
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Phase-A launch binding must be an object")
    payload = _validate_self_hash(value, "binding_sha256", "Phase-A launch binding")
    stage = payload.get("stage")
    if (
        payload.get("schema") != PHASE_A_LAUNCH_BINDING_SCHEMA
        or payload.get("status") != "VALIDATED"
        or stage not in STAGES
        or (expected_stage is not None and stage != expected_stage)
    ):
        raise ValueError("Phase-A launch binding identity drifted")
    job_id(payload.get("slurm_job_id"), "Phase-A launch binding job id")
    for name in ("plan_sha256", "receipt_sha256", "release_sha256"):
        digest(payload.get(name), f"Phase-A launch binding {name}")
    return payload


def completion_payload(
    launch: Mapping[str, object],
    artifacts: Sequence[Mapping[str, object]],
    job_local_capability: Mapping[str, object],
) -> dict[str, object]:
    launch = validate_launch_binding_payload(launch)
    if (
        not isinstance(job_local_capability, Mapping)
        or job_local_capability.get("pre_post_equal") is not True
        or not isinstance(job_local_capability.get("capability"), Mapping)
        or job_local_capability["capability"].get("launch_binding_sha256")
        != launch["binding_sha256"]
        or job_local_capability["capability"].get("stage") != launch["stage"]
        or job_local_capability["capability"].get("slurm_job_id")
        != launch["slurm_job_id"]
    ):
        raise ValueError("Phase-A completion requires a consumed matching capability")
    core = {
        "schema": PHASE_A_COMPLETION_SCHEMA,
        "status": "COMPLETE",
        "stage": launch["stage"],
        "slurm_job_id": launch["slurm_job_id"],
        "upstream_stage": DEPENDENCY[launch["stage"]],
        "upstream_job_id": None if launch["upstream"] is None else launch["upstream"]["job_id"],
        "plan_sha256": launch["plan_sha256"],
        "receipt_sha256": launch["receipt_sha256"],
        "release_sha256": launch["release_sha256"],
        "launch_binding_sha256": launch["binding_sha256"],
        "launch_binding": deepcopy(dict(launch)),
        "job_local_capability": deepcopy(dict(job_local_capability)),
        "artifacts": deepcopy(list(artifacts)),
    }
    return self_hashed(core, "completion_sha256")


__all__ = [
    "DEPENDENCY",
    "PHASE_A_COMPLETION_SCHEMA",
    "PHASE_A_FAILURE_SCHEMA",
    "PHASE_A_LAUNCH_BINDING_SCHEMA",
    "PHASE_A_PLAN_SCHEMA",
    "PHASE_A_PLAN_VERSION",
    "PHASE_A_RECEIPT_SCHEMA",
    "PHASE_A_RELEASE_SCHEMA",
    "STAGES",
    "canonical_json",
    "completion_payload",
    "digest",
    "job_id",
    "launch_binding",
    "load_launch_transaction",
    "portable_identity",
    "self_hashed",
    "validate_plan",
    "validate_launch_binding_payload",
    "validate_receipt",
    "validate_release",
    "validate_stage_completion",
]
