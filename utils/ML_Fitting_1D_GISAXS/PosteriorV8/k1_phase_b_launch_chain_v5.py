"""Audited launch-chain and pinned-wrapper contract for K1 Phase-B."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Mapping

from .grouped_artifact_v5 import canonical_json
from .immutable_submission_file_v5 import copy_immutable_submission_file
from .k1_phase_a_cross_platform_v5 import file_identity as strict_file_identity
from .k1_phase_b_contract_v5 import (
    CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS,
    FULL_FILE_IDENTITY_FIELDS,
    V5_K1_PHASE_B_RESULT_FILENAME,
    V5_K1_PHASE_B_SCHEMA,
    V5_K1_PHASE_B_VERSION,
    cross_node_stable_file_identity,
    digest,
    strict_json_object,
    validate_cross_node_stable_file_identity,
)
from .k1_phase_b_publication_v5 import (
    V5_K1_PHASE_B_COMPLETION_FILENAME,
    V5_K1_PHASE_B_COMPLETION_SCHEMA,
    V5_K1_PHASE_B_COMPLETION_VERSION,
)


V5_K1_PHASE_B_LAUNCH_SCHEMA = "gisaxs.posterior_v8.maxwell_k1_phase_b_dag_launch/v14"
V5_K1_PHASE_B_LAUNCH_VERSION = (
    "posterior_v8_closed_interval_tolerance_phase_b_dag_v14"
)
V5_K1_PHASE_B_LAUNCH_COMPLETION_SCHEMA = (
    "gisaxs.posterior_v8.maxwell_k1_phase_b_launch_completion/v13"
)
V5_K1_PHASE_B_LAUNCH_COMPLETION_VERSION = (
    "posterior_v8_closed_interval_tolerance_release_completion_v13"
)
LAUNCH_STAGES = ("engineering_smoke", "formal_gate")
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_CROSS_NODE_STABLE_IDENTITY_FIELDS = CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS
_FULL_FILE_IDENTITY_FIELDS = FULL_FILE_IDENTITY_FIELDS


def _stat_tuple(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
        status.st_nlink,
    )


def _cross_node_stable_file_identity(value: object) -> dict[str, object] | None:
    """Canonicalize an audited file identity without mount-local ``st_dev``."""

    return cross_node_stable_file_identity(value)


def _cross_node_file_identity_matches(observed: object, expected: object) -> bool:
    """Compare every cross-node-stable field of two complete file identities."""

    observed_stable = _cross_node_stable_file_identity(observed)
    return observed_stable is not None and observed_stable == (
        _cross_node_stable_file_identity(expected)
    )


def copy_v5_k1_phase_b_submission_wrapper(
    source: Path,
    target: Path,
    *,
    expected_source_identity: Mapping[str, object],
) -> dict[str, object]:
    """Copy the verified wrapper through O_NOFOLLOW fds into an exclusive target."""
    return copy_immutable_submission_file(
        source,
        target,
        expected_source_identity=expected_source_identity,
        name="Phase-B wrapper",
    )


def _read_self_hashed_json(
    path: Path, *, name: str, self_field: str
) -> tuple[dict[str, object], dict[str, object]]:
    identity = strict_file_identity(path, name=name, require_read_only=True)
    descriptor = os.open(
        Path(str(identity["path"])),
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_dev != identity["device"]
            or before.st_ino != identity["inode"]
            or stat.S_IMODE(before.st_mode) != identity["mode"]
            or before.st_size != identity["byte_count"]
            or before.st_mtime_ns != identity["mtime_ns"]
            or before.st_ctime_ns != identity["ctime_ns"]
            or before.st_nlink != identity["nlink"]
        ):
            raise RuntimeError(f"{name} changed before its bound read")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after_fd = os.fstat(descriptor)
        after_path = os.stat(Path(str(identity["path"])), follow_symlinks=False)
        if _stat_tuple(before) != _stat_tuple(after_fd) or _stat_tuple(
            after_fd
        ) != _stat_tuple(after_path):
            raise RuntimeError(f"{name} changed during its bound read")
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)
    if len(encoded) != identity["byte_count"] or sha256(encoded).hexdigest() != identity[
        "sha256"
    ]:
        raise RuntimeError(f"{name} bytes do not match its bound identity")
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not UTF-8 JSON") from exc
    payload = strict_json_object(text, name)
    after = strict_file_identity(path, name=name, require_read_only=True)
    if after != identity:
        raise RuntimeError(f"{name} changed while it was parsed")
    supplied = digest(payload.get(self_field), f"{name} self SHA-256")
    core = dict(payload)
    core.pop(self_field, None)
    if sha256(canonical_json(core).encode("utf-8")).hexdigest() != supplied:
        raise ValueError(f"{name} self SHA-256 does not reproduce")
    return payload, identity


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseBLaunchRuntime:
    plan_path: Path
    expected_plan_sha256: str
    expected_plan_file_identity: Mapping[str, object]
    receipt_path: Path
    launch_completion_path: Path
    launch_stage: str


def _inspect_plan(
    runtime: V5K1PhaseBLaunchRuntime, *, output_dir: Path
) -> tuple[dict[str, object], dict[str, object]]:
    if runtime.launch_stage not in LAUNCH_STAGES:
        raise ValueError("Phase-B launch stage is unsupported")
    plan, plan_identity = _read_self_hashed_json(
        runtime.plan_path, name="Phase-B launch plan", self_field="plan_sha256"
    )
    if plan.get("plan_sha256") != digest(
        runtime.expected_plan_sha256, "expected launch-plan SHA-256"
    ):
        raise ValueError("Phase-B launch plan self identity drift detected")
    expected_plan_identity = dict(runtime.expected_plan_file_identity)
    # st_dev identifies the client mount namespace, not the Lustre object. Maxwell
    # login and worker nodes can therefore report different values for the same
    # inode. Bind every cross-node-stable field and retain the launcher device in
    # the immutable receipt instead of comparing the mount-local value.
    if not _cross_node_file_identity_matches(plan_identity, expected_plan_identity):
        raise ValueError("Phase-B launch plan file identity drift detected")
    if (plan.get("schema_version"), plan.get("version")) != (
        V5_K1_PHASE_B_LAUNCH_SCHEMA,
        V5_K1_PHASE_B_LAUNCH_VERSION,
    ):
        raise ValueError("Phase-B launch plan schema/version is incompatible")
    layout = plan.get("layout")
    jobs = plan.get("jobs")
    if not isinstance(layout, Mapping) or not isinstance(jobs, Mapping):
        raise ValueError("Phase-B launch plan layout/jobs are missing")
    if Path(str(layout.get("plan"))).resolve() != runtime.plan_path.resolve():
        raise ValueError("Phase-B launch plan path does not bind itself")
    if Path(str(layout.get("receipt"))).resolve() != runtime.receipt_path.resolve():
        raise ValueError("Phase-B launch receipt path drift detected")
    if Path(str(layout.get("launch_completion"))).resolve() != (
        runtime.launch_completion_path.resolve()
    ):
        raise ValueError("Phase-B launch completion path drift detected")
    job = jobs.get(runtime.launch_stage)
    if not isinstance(job, Mapping):
        raise ValueError("Phase-B stage is missing from the launch plan")
    expected_output_name = (
        "smoke_output" if runtime.launch_stage == "engineering_smoke" else "formal_output"
    )
    if Path(str(job.get("environment", {}).get("POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT"))).resolve() != (
        output_dir.resolve()
    ) or Path(str(layout.get(expected_output_name))).resolve() != output_dir.resolve():
        raise ValueError("Phase-B stage/output binding drift detected")
    if job.get("environment", {}).get("POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_STAGE") != (
        runtime.launch_stage
    ):
        raise ValueError("Phase-B launch stage environment binding is missing")
    if (
        job.get("wrapper") != layout.get("submission_wrapper")
        or job.get("environment", {}).get("POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN")
        != str(runtime.plan_path)
        or job.get("environment", {}).get("POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT")
        != str(runtime.receipt_path)
        or job.get("environment", {}).get(
            "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION"
        )
        != str(runtime.launch_completion_path)
    ):
        raise ValueError("Phase-B stage launch-path binding is incomplete")
    expected_dependency = (
        None if runtime.launch_stage == "engineering_smoke" else "engineering_smoke"
    )
    if (
        job.get("depends_on") != expected_dependency
        or job.get("submit_held") is not True
    ):
        raise ValueError("Phase-B stage hold/afterok plan binding is incompatible")
    wrapper_binding = plan.get("submission_wrapper")
    if not isinstance(wrapper_binding, Mapping) or set(wrapper_binding) != {
        "path",
        "sha256",
        "byte_count",
        "mode",
        "nlink",
    }:
        raise ValueError("Phase-B pinned wrapper binding is incomplete")
    wrapper_identity = strict_file_identity(
        Path(str(wrapper_binding["path"])),
        name="pinned Phase-B submission wrapper",
        require_read_only=True,
    )
    if any(wrapper_identity[name] != wrapper_binding[name] for name in ("sha256", "byte_count", "mode", "nlink")):
        raise ValueError("Phase-B pinned wrapper content identity drift detected")
    stable_plan_identity = _cross_node_stable_file_identity(expected_plan_identity)
    stable_wrapper_identity = _cross_node_stable_file_identity(wrapper_identity)
    if stable_plan_identity is None or stable_wrapper_identity is None:
        raise ValueError("Phase-B cross-node audit identity fields are incomplete")
    evidence = {
        "path": str(runtime.plan_path.resolve()),
        "plan_sha256": plan["plan_sha256"],
        "file_identity": stable_plan_identity,
        "cross_node_identity_policy": {
            "bound_fields": list(_CROSS_NODE_STABLE_IDENTITY_FIELDS),
            "device_field": "mount_namespace_local_not_cross_node_bound",
            "canonical_evidence_excludes_device": True,
        },
        "submission_wrapper_identity": stable_wrapper_identity,
    }
    return plan, evidence


def _validate_smoke_completion(
    output_dir: Path,
    *,
    expected_job_id: str,
    plan_evidence: Mapping[str, object],
) -> dict[str, object]:
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("upstream smoke output directory is missing")
    if stat.S_IMODE(output_dir.stat().st_mode) & _WRITE_BITS:
        raise ValueError("upstream smoke output directory remains writable")
    completion, completion_identity = _read_self_hashed_json(
        output_dir / V5_K1_PHASE_B_COMPLETION_FILENAME,
        name="upstream smoke completion",
        self_field="completion_payload_sha256",
    )
    if (completion.get("schema_version"), completion.get("version"), completion.get("status")) != (
        V5_K1_PHASE_B_COMPLETION_SCHEMA,
        V5_K1_PHASE_B_COMPLETION_VERSION,
        "COMPLETE",
    ):
        raise ValueError("upstream smoke completion schema/status is incompatible")
    if (
        completion.get("launch_stage") != "engineering_smoke"
        or completion.get("slurm_job_id") != expected_job_id
        or Path(str(completion.get("output_dir"))).resolve() != output_dir.resolve()
        or completion.get("launch_plan") != dict(plan_evidence)
    ):
        raise ValueError("upstream smoke completion launch binding drift detected")
    result, result_identity = _read_self_hashed_json(
        output_dir / V5_K1_PHASE_B_RESULT_FILENAME,
        name="upstream smoke result",
        self_field="result_payload_sha256",
    )
    expected_result = completion.get("result")
    expected_result_fields = {
        *CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS,
        "result_payload_sha256",
    }
    if not isinstance(expected_result, Mapping) or set(expected_result) != (
        expected_result_fields
    ):
        raise ValueError("upstream smoke completion result identity is missing")
    expected_result_identity = validate_cross_node_stable_file_identity(
        {
            name: expected_result[name]
            for name in CROSS_NODE_STABLE_FILE_IDENTITY_FIELDS
        }
    )
    observed_result_identity = _cross_node_stable_file_identity(result_identity)
    capability = completion.get("job_local_capability")
    if (
        observed_result_identity is None
        or expected_result_identity is None
        or observed_result_identity != expected_result_identity
        or result["result_payload_sha256"]
        != expected_result["result_payload_sha256"]
        or not isinstance(capability, Mapping)
        or capability.get("pre_post_equal") is not True
        or result.get("job_local_capability") != capability
    ):
        raise ValueError("upstream smoke result does not match its completion marker")
    launch_chain = result.get("launch_chain")
    if (
        result.get("schema_version") != V5_K1_PHASE_B_SCHEMA
        or result.get("version") != V5_K1_PHASE_B_VERSION
        or result.get("status") != "engineering_throughput_smoke_completed_fail_closed"
        or not isinstance(launch_chain, Mapping)
        or launch_chain.get("launch_stage") != "engineering_smoke"
        or launch_chain.get("slurm_job_id") != expected_job_id
        or launch_chain.get("launch_plan") != dict(plan_evidence)
    ):
        raise ValueError("upstream smoke result launch/stage/job binding is invalid")
    stable_completion_identity = _cross_node_stable_file_identity(completion_identity)
    if stable_completion_identity is None:
        raise ValueError("upstream smoke completion file identity is incomplete")
    return {
        "completion_identity": stable_completion_identity,
        "completion_payload_sha256": completion["completion_payload_sha256"],
        "result_identity": observed_result_identity,
        "result_payload_sha256": result["result_payload_sha256"],
        "launch_stage": "engineering_smoke",
        "slurm_job_id": expected_job_id,
    }


def inspect_v5_k1_phase_b_launch_chain(
    runtime: V5K1PhaseBLaunchRuntime,
    *,
    output_dir: Path,
    slurm_job_id: str,
) -> dict[str, object]:
    """Validate scheduler-read-back launch evidence and any upstream smoke."""

    plan, plan_evidence = _inspect_plan(runtime, output_dir=output_dir)
    receipt, receipt_identity = _read_self_hashed_json(
        runtime.receipt_path,
        name="Phase-B held submission receipt",
        self_field="receipt_sha256",
    )
    if (
        receipt.get("schema_version") != V5_K1_PHASE_B_LAUNCH_SCHEMA
        or receipt.get("version") != V5_K1_PHASE_B_LAUNCH_VERSION
        or receipt.get("status") != "all_jobs_held"
        or receipt.get("plan_sha256") != plan["plan_sha256"]
        or _cross_node_stable_file_identity(receipt.get("plan_file_identity"))
        != plan_evidence["file_identity"]
        or _cross_node_stable_file_identity(
            receipt.get("submission_wrapper_identity")
        )
        != plan_evidence["submission_wrapper_identity"]
        or Path(str(receipt.get("plan_path"))).resolve() != runtime.plan_path.resolve()
        or Path(str(receipt.get("receipt_path"))).resolve()
        != runtime.receipt_path.resolve()
        or Path(str(receipt.get("launch_completion_path"))).resolve()
        != runtime.launch_completion_path.resolve()
        or receipt.get("formal_submitted_held") is not True
        or receipt.get("all_jobs_submitted_held") is not True
        or receipt.get("formal_release_completed") is not False
        or receipt.get("cancellation_attempted") is not False
        or receipt.get("failure") is not None
    ):
        raise ValueError("Phase-B submission receipt binding is incompatible")
    job_ids = receipt.get("job_ids")
    if not isinstance(job_ids, Mapping) or set(job_ids) != set(LAUNCH_STAGES):
        raise ValueError("Phase-B submission receipt job IDs are incomplete")
    smoke_job_id = str(job_ids["engineering_smoke"])
    formal_job_id = str(job_ids["formal_gate"])
    if (
        not smoke_job_id.isdigit()
        or smoke_job_id.startswith("0")
        or not formal_job_id.isdigit()
        or formal_job_id.startswith("0")
        or smoke_job_id == formal_job_id
    ):
        raise ValueError("Phase-B submission receipt job IDs are malformed")
    expected_current_job = str(job_ids[runtime.launch_stage])
    if expected_current_job != slurm_job_id:
        raise ValueError("current Slurm job is not the receipt-bound stage job")
    if receipt.get("dependency_edges") != [["engineering_smoke", "formal_gate"]]:
        raise ValueError("Phase-B receipt afterok edge is incompatible")
    attempts = receipt.get("submission_attempts")
    if not isinstance(attempts, list) or len(attempts) != 2:
        raise ValueError("Phase-B receipt submission attempts are incomplete")
    attempts_by_stage = {
        value.get("stage"): value for value in attempts if isinstance(value, Mapping)
    }
    if set(attempts_by_stage) != set(LAUNCH_STAGES):
        raise ValueError("Phase-B receipt stage attempts are incompatible")
    smoke_argv = attempts_by_stage["engineering_smoke"].get("argv")
    formal_argv = attempts_by_stage["formal_gate"].get("argv")
    if (
        not isinstance(smoke_argv, list)
        or not isinstance(formal_argv, list)
        or "--hold" not in smoke_argv
        or "--hold" not in formal_argv
        or "--kill-on-invalid-dep=yes" not in smoke_argv
        or "--kill-on-invalid-dep=yes" not in formal_argv
        or f"--dependency=afterok:{smoke_job_id}" not in formal_argv
        or plan["layout"]["submission_wrapper"] in smoke_argv
        or plan["layout"]["submission_wrapper"] in formal_argv
        or attempts_by_stage["engineering_smoke"].get("script_bytes_recorded") is not False
        or attempts_by_stage["formal_gate"].get("script_bytes_recorded") is not False
        or attempts_by_stage["engineering_smoke"].get("script_identity", {}).get("sha256")
        != plan["submission_wrapper"]["sha256"]
        or attempts_by_stage["formal_gate"].get("script_identity", {}).get("sha256")
        != plan["submission_wrapper"]["sha256"]
    ):
        raise ValueError("Phase-B receipt does not prove the held afterok submission")
    held = receipt.get("held_scheduler_snapshots")
    if not isinstance(held, Mapping) or set(held) != set(LAUNCH_STAGES):
        raise ValueError("Phase-B receipt lacks scheduler hold evidence")
    for stage, expected_job in job_ids.items():
        dependency = None if stage == "engineering_smoke" else smoke_job_id
        snapshot = held[stage]
        if (
            not isinstance(snapshot, Mapping)
            or snapshot.get("job_id") != expected_job
            or snapshot.get("job_state") != "PENDING"
            or snapshot.get("reason") != "JobHeldUser"
            or (
                dependency is None and snapshot.get("dependency") is not None
            )
            or (
                dependency is not None
                and not str(snapshot.get("dependency", "")).startswith(
                    f"afterok:{dependency}"
                )
            )
        ):
            raise ValueError(f"Phase-B {stage} scheduler hold evidence drifted")

    stable_receipt_identity = _cross_node_stable_file_identity(receipt_identity)
    if stable_receipt_identity is None:
        raise ValueError("Phase-B submission receipt file identity is incomplete")
    launch_completion, launch_completion_identity = _read_self_hashed_json(
        runtime.launch_completion_path,
        name="Phase-B launch completion",
        self_field="launch_completion_sha256",
    )
    if (
        launch_completion.get("schema_version")
        != V5_K1_PHASE_B_LAUNCH_COMPLETION_SCHEMA
        or launch_completion.get("version") != V5_K1_PHASE_B_LAUNCH_COMPLETION_VERSION
        or launch_completion.get("status") != "ALL_JOBS_RELEASED"
        or _cross_node_stable_file_identity(
            launch_completion.get("plan_file_identity")
        )
        != plan_evidence["file_identity"]
        or _cross_node_stable_file_identity(
            launch_completion.get("submission_receipt_identity")
        )
        != stable_receipt_identity
        or launch_completion.get("job_ids") != dict(job_ids)
        or launch_completion.get("release_order") != list(reversed(LAUNCH_STAGES))
        or launch_completion.get("released_job_ids")
        != [job_ids[stage] for stage in reversed(LAUNCH_STAGES)]
        or launch_completion.get("plan_sha256") != plan["plan_sha256"]
        or launch_completion.get("official_launch_chain_complete") is not True
    ):
        raise ValueError("Phase-B launch completion/release binding is incompatible")
    releases = launch_completion.get("release_attempts")
    if not isinstance(releases, list) or len(releases) != len(LAUNCH_STAGES):
        raise ValueError("Phase-B release audit is incomplete")
    for stage, attempt in zip(reversed(LAUNCH_STAGES), releases, strict=True):
        snapshot = attempt.get("scheduler_snapshot") if isinstance(attempt, Mapping) else None
        if (
            not isinstance(attempt, Mapping)
            or attempt.get("stage") != stage
            or attempt.get("job_id") != job_ids[stage]
            or attempt.get("argv") != ["scontrol", "release", job_ids[stage]]
            or attempt.get("returncode") != 0
            or not isinstance(snapshot, Mapping)
            or snapshot.get("job_id") != job_ids[stage]
            or snapshot.get("reason") == "JobHeldUser"
        ):
            raise ValueError(f"Phase-B {stage} release audit is incompatible")
    stable_launch_completion_identity = _cross_node_stable_file_identity(
        launch_completion_identity
    )
    if stable_launch_completion_identity is None:
        raise ValueError("Phase-B launch completion file identity is incomplete")
    transaction_evidence = {
        "submission_receipt_identity": stable_receipt_identity,
        "submission_receipt_sha256": receipt["receipt_sha256"],
        "launch_completion_identity": stable_launch_completion_identity,
        "launch_completion_sha256": launch_completion["launch_completion_sha256"],
        "job_ids": dict(job_ids),
        "held_scheduler_snapshots": dict(held),
        "release_attempts": releases,
    }
    result: dict[str, object] = {
        "launch_stage": runtime.launch_stage,
        "slurm_job_id": slurm_job_id,
        "launch_plan": plan_evidence,
        "launch_transaction_evidence": transaction_evidence,
        "launch_transaction_evidence_sha256": sha256(
            canonical_json(transaction_evidence).encode("utf-8")
        ).hexdigest(),
        "formal_prerequisite_evidence": None,
        "formal_prerequisite_evidence_sha256": None,
    }
    if runtime.launch_stage == "engineering_smoke":
        return result

    smoke_output = Path(str(plan["layout"]["smoke_output"]))
    smoke = _validate_smoke_completion(
        smoke_output,
        expected_job_id=smoke_job_id,
        plan_evidence=plan_evidence,
    )
    formal_evidence = {
        "launch_transaction_evidence_sha256": result[
            "launch_transaction_evidence_sha256"
        ],
        "formal_job_id": formal_job_id,
        "upstream_smoke_job_id": smoke_job_id,
        "afterok_edge": ["engineering_smoke", "formal_gate"],
        "upstream_smoke": smoke,
    }
    result["formal_prerequisite_evidence"] = formal_evidence
    result["formal_prerequisite_evidence_sha256"] = sha256(
        canonical_json(formal_evidence).encode("utf-8")
    ).hexdigest()
    return result


def assert_v5_k1_phase_b_launch_chain_unchanged(
    before: Mapping[str, object],
    runtime: V5K1PhaseBLaunchRuntime,
    *,
    output_dir: Path,
    slurm_job_id: str,
) -> None:
    if inspect_v5_k1_phase_b_launch_chain(
        runtime, output_dir=output_dir, slurm_job_id=slurm_job_id
    ) != dict(before):
        raise RuntimeError("Phase-B launch/afterok evidence changed during execution")


def add_v5_k1_phase_b_launch_runtime_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--launch-stage", required=True, choices=LAUNCH_STAGES)
    parser.add_argument("--launch-plan", required=True, type=Path)
    parser.add_argument("--launch-plan-sha256", required=True)
    parser.add_argument("--launch-plan-file-sha256", required=True)
    parser.add_argument("--launch-plan-byte-count", required=True, type=int)
    parser.add_argument("--launch-plan-mode", required=True, type=int)
    parser.add_argument("--launch-plan-device", required=True, type=int)
    parser.add_argument("--launch-plan-inode", required=True, type=int)
    parser.add_argument("--launch-plan-mtime-ns", required=True, type=int)
    parser.add_argument("--launch-plan-ctime-ns", required=True, type=int)
    parser.add_argument("--launch-plan-nlink", required=True, type=int)
    parser.add_argument("--launch-receipt", required=True, type=Path)
    parser.add_argument("--launch-completion", required=True, type=Path)


def v5_k1_phase_b_launch_runtime_from_args(
    args: argparse.Namespace,
) -> V5K1PhaseBLaunchRuntime:
    return V5K1PhaseBLaunchRuntime(
        plan_path=args.launch_plan,
        expected_plan_sha256=args.launch_plan_sha256,
        expected_plan_file_identity={
            "path": str(args.launch_plan.resolve()),
            "sha256": args.launch_plan_file_sha256,
            "byte_count": args.launch_plan_byte_count,
            "mode": args.launch_plan_mode,
            "device": args.launch_plan_device,
            "inode": args.launch_plan_inode,
            "mtime_ns": args.launch_plan_mtime_ns,
            "ctime_ns": args.launch_plan_ctime_ns,
            "nlink": args.launch_plan_nlink,
        },
        receipt_path=args.launch_receipt,
        launch_completion_path=args.launch_completion,
        launch_stage=args.launch_stage,
    )


__all__ = [
    "LAUNCH_STAGES",
    "V5K1PhaseBLaunchRuntime",
    "V5_K1_PHASE_B_LAUNCH_COMPLETION_SCHEMA",
    "V5_K1_PHASE_B_LAUNCH_COMPLETION_VERSION",
    "V5_K1_PHASE_B_LAUNCH_SCHEMA",
    "V5_K1_PHASE_B_LAUNCH_VERSION",
    "add_v5_k1_phase_b_launch_runtime_arguments",
    "assert_v5_k1_phase_b_launch_chain_unchanged",
    "copy_v5_k1_phase_b_submission_wrapper",
    "inspect_v5_k1_phase_b_launch_chain",
    "v5_k1_phase_b_launch_runtime_from_args",
]
