"""Plan or submit the audited Maxwell V5.2 K1 Phase-B DAG.

Dry-run is the default and performs no writes.  Submit mode is restricted to
``max-wgs`` and publishes an immutable plan before submitting an engineering
throughput smoke followed by the formal single-branch gate with ``afterok``.
Every submission replays the source archive/tree and all Phase-A evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import shlex
import socket
import subprocess
import tempfile
from typing import Callable, Mapping, Sequence

from .k1_phase_a_cross_platform_v5 import file_identity as strict_file_identity
from .k1_staging_files_v5 import read_only_bytes_identity
from .k1_phase_b_launch_chain_v5 import (
    V5_K1_PHASE_B_LAUNCH_COMPLETION_SCHEMA,
    V5_K1_PHASE_B_LAUNCH_COMPLETION_VERSION,
    V5_K1_PHASE_B_LAUNCH_SCHEMA,
    V5_K1_PHASE_B_LAUNCH_VERSION,
    copy_v5_k1_phase_b_submission_wrapper,
)
from .k1_phase_b_launch_inputs_v5 import (
    CURRENT_RUN_ROOT_NAME,
    MAXWELL_DUST_ROOT,
    PHASE_B_WRAPPER,
    V5K1PhaseBLaunchConfig,
    inspect_v5_k1_phase_a_inputs,
    inspect_v5_k1_phase_b_source,
    resolve_v5_k1_phase_b_run_root,
)


PHASE_ROOT_NAME = "k1_phase_b_v5_2_dag_v15"
PLAN_FILENAME = "launch-plan-v14.json"
RECEIPT_FILENAME = "submission-receipt-v14.json"
LAUNCH_COMPLETION_FILENAME = "launch-completion-v13.json"
RELEASE_FAILURE_FILENAME = "launch-release-failure-v2.json"
_WRAPPER = PHASE_B_WRAPPER
_PARSABLE_JOB_ID = re.compile(r"([1-9][0-9]*)(?:;[A-Za-z0-9._-]+)?\Z")
_SAFE_INHERITED_ENVIRONMENT = ("HOME", "LANG", "PATH", "SHELL", "USER")
_STAGES = ("engineering_smoke", "formal_gate")


@dataclass(frozen=True)
class V5CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class V5CommandRequest:
    argv: tuple[str, ...]
    script_bytes: bytes | None = None
    script_sha256: str | None = None

    def __iter__(self):
        return iter(self.argv)

    def __len__(self) -> int:
        return len(self.argv)

    def __getitem__(self, index):
        return self.argv[index]


CommandRunner = Callable[[V5CommandRequest], V5CommandResult]


class V5K1PhaseBLaunchError(RuntimeError):
    """Submission failed after its exclusive audit root was reserved."""

    def __init__(self, message: str, *, receipt_path: Path) -> None:
        super().__init__(message)
        self.receipt_path = receipt_path


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _safe_export_value(value: object, name: str) -> str:
    result = str(value)
    if any(token in result for token in (",", "\n", "\r", "\0")):
        raise ValueError(f"{name} cannot be represented in the Slurm export contract")
    return result


def _export_argument(environment: Mapping[str, object]) -> str:
    assignments = [
        f"{name}={_safe_export_value(value, name)}" for name, value in environment.items()
    ]
    return "--export=" + ",".join((*_SAFE_INHERITED_ENVIRONMENT, *assignments))


def _job_command(
    job: Mapping[str, object],
    dependency_job_id: str | None,
    *,
    runtime_environment: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    command = ["sbatch", "--parsable", "--hold", "--kill-on-invalid-dep=yes"]
    if dependency_job_id is not None:
        command.append(f"--dependency=afterok:{dependency_job_id}")
    command.extend(
        (
            f"--job-name={job['job_name']}",
            f"--output={job['stdout']}",
            f"--error={job['stderr']}",
            _export_argument(
                {
                    **job["environment"],
                    **({} if runtime_environment is None else runtime_environment),
                }
            ),
        )
    )
    return tuple(command)


def build_v5_k1_phase_b_launch_plan(
    config: V5K1PhaseBLaunchConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Validate every immutable input and return a deterministic write-free DAG."""

    if not isinstance(config, V5K1PhaseBLaunchConfig):
        raise TypeError("config must be V5K1PhaseBLaunchConfig")
    source = inspect_v5_k1_phase_b_source(config, allowed_root)
    run_root = resolve_v5_k1_phase_b_run_root(config.run_root, allowed_root)
    logs = run_root / "logs"
    if not logs.is_dir() or logs.is_symlink():
        raise FileNotFoundError("the versioned run root must contain a real logs directory")
    source_root = Path(str(source["source_root"]))
    source_archive = Path(str(source["archive_path"]))
    if (
        source_root == run_root
        or source_root.is_relative_to(run_root)
        or run_root.is_relative_to(source_root)
        or source_archive == run_root
        or source_archive.is_relative_to(run_root)
        or source_archive == source_root
        or source_archive.is_relative_to(source_root)
    ):
        raise ValueError("immutable source/archive and writable run outputs must be separate")
    phase_a = inspect_v5_k1_phase_a_inputs(
        config, source=source, run_root=run_root, allowed_root=allowed_root
    )

    phase_root = run_root / PHASE_ROOT_NAME
    results_root = phase_root / "results"
    audit_root = phase_root / "audit"
    layout = {
        "phase_root": str(phase_root),
        "results_root": str(results_root),
        "audit_root": str(audit_root),
        "smoke_output": str(results_root / "engineering-smoke-parents2"),
        "formal_output": str(results_root / "formal-all512"),
        "plan": str(audit_root / PLAN_FILENAME),
        "receipt": str(audit_root / RECEIPT_FILENAME),
        "launch_completion": str(audit_root / LAUNCH_COMPLETION_FILENAME),
        "release_failure": str(audit_root / RELEASE_FAILURE_FILENAME),
        "submission_root": str(audit_root / "submission-v7"),
        "submission_wrapper": str(audit_root / "submission-v7/phase-b-gate-v6.sbatch"),
        "logs": str(logs),
    }
    if phase_root.exists() or phase_root.is_symlink():
        raise FileExistsError("refusing to reuse the versioned K1 Phase-B root")
    if any(logs.glob("k1-phase-b-v5-2-v15-*.out")) or any(
        logs.glob("k1-phase-b-v5-2-v15-*.err")
    ):
        raise FileExistsError("refusing to reuse existing K1 Phase-B Slurm logs")

    source_environment = {
        "POSTERIOR_V8_SOURCE_ROOT": str(source_root),
        "POSTERIOR_V8_V5_SOURCE_ARCHIVE": str(source_archive),
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_SHA256": source["archive_sha256"],
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_ARCHIVE_BYTE_COUNT": source["archive_byte_count"],
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_MANIFEST_SHA256": source["manifest_sha256"],
        "POSTERIOR_V8_V5_EXPECTED_SOURCE_TREE_SHA256": source["source_tree_sha256"],
    }
    artifact_environment_names = {
        "dataset": "POSTERIOR_V8_V5_K1_PHASE_B_DATASET",
        "dataset_binding": "POSTERIOR_V8_V5_K1_PHASE_B_DATASET_BINDING",
        "cross_platform_pass_marker": ("POSTERIOR_V8_V5_K1_PHASE_B_CROSS_PLATFORM_PASS_MARKER"),
        "result": "POSTERIOR_V8_V5_K1_PHASE_B_RESULT",
        "model": "POSTERIOR_V8_V5_K1_PHASE_B_MODEL",
        "model_provenance": "POSTERIOR_V8_V5_K1_PHASE_B_MODEL_PROVENANCE",
        "phase_a_completion": "POSTERIOR_V8_V5_K1_PHASE_B_PHASE_A_COMPLETION",
    }
    phase_a_environment: dict[str, object] = {}
    for name, environment_name in artifact_environment_names.items():
        identity = phase_a["artifacts"][name]
        phase_a_environment[environment_name] = identity["path"]
        phase_a_environment[f"{environment_name}_SHA256"] = identity["sha256"]
        phase_a_environment[f"{environment_name}_BYTE_COUNT"] = identity["byte_count"]
    gate = phase_a["cross_platform_gate"]
    for field, environment_name in (
        ("gate_claim_sha256", "EXPECTED_GATE_CLAIM_SHA256"),
        ("reference_file_sha256", "EXPECTED_REFERENCE_SHA256"),
        ("reference_file_byte_count", "EXPECTED_REFERENCE_BYTE_COUNT"),
        ("reference_manifest_sha256", "EXPECTED_REFERENCE_MANIFEST_SHA256"),
        ("scientific_content_sha256", "EXPECTED_SCIENTIFIC_SHA256"),
        ("comparison_result_sha256", "EXPECTED_COMPARISON_SHA256"),
    ):
        phase_a_environment[f"POSTERIOR_V8_V5_K1_PHASE_B_{environment_name}"] = gate[field]
    common_environment = {**source_environment, **phase_a_environment}
    wrapper = Path(layout["submission_wrapper"])
    jobs = {
        "engineering_smoke": {
            "job_name": "gisaxs-v5-2-k1-phase-b-smoke",
            "wrapper": str(wrapper),
            "stdout": str(logs / "k1-phase-b-v5-2-v15-smoke-%j.out"),
            "stderr": str(logs / "k1-phase-b-v5-2-v15-smoke-%j.err"),
            "environment": {
                **common_environment,
                "POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT": layout["smoke_output"],
                "POSTERIOR_V8_V5_K1_PHASE_B_MODE": "smoke",
                "POSTERIOR_V8_V5_K1_PHASE_B_SMOKE_PARENTS": 2,
                "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_STAGE": "engineering_smoke",
                "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN": layout["plan"],
                "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT": layout["receipt"],
                "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION": layout[
                    "launch_completion"
                ],
            },
            "depends_on": None,
            "submit_held": True,
        },
        "formal_gate": {
            "job_name": "gisaxs-v5-2-k1-phase-b-formal",
            "wrapper": str(wrapper),
            "stdout": str(logs / "k1-phase-b-v5-2-v15-formal-%j.out"),
            "stderr": str(logs / "k1-phase-b-v5-2-v15-formal-%j.err"),
            "environment": {
                **common_environment,
                "POSTERIOR_V8_V5_K1_PHASE_B_OUTPUT": layout["formal_output"],
                "POSTERIOR_V8_V5_K1_PHASE_B_MODE": "formal",
                "POSTERIOR_V8_V5_K1_PHASE_B_FORMAL_ACK": "YES",
                "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_STAGE": "formal_gate",
                "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN": layout["plan"],
                "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_RECEIPT": layout["receipt"],
                "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_COMPLETION": layout[
                    "launch_completion"
                ],
            },
            "depends_on": "engineering_smoke",
            "submit_held": True,
        },
    }
    preview_ids = {stage: f"{stage.upper()}_JOB_ID" for stage in _STAGES}
    preview = {
        stage: list(
            _job_command(
                jobs[stage],
                None
                if jobs[stage]["depends_on"] is None
                else preview_ids[str(jobs[stage]["depends_on"])],
            )
        )
        for stage in _STAGES
    }
    core = {
        "schema_version": V5_K1_PHASE_B_LAUNCH_SCHEMA,
        "version": V5_K1_PHASE_B_LAUNCH_VERSION,
        "scientific_role": (
            "single_branch_phase_b_capacity_and_throughput_diagnostic_not_full_k1_"
            "or_model_acceptance"
        ),
        "run_root": str(run_root),
        "source": source,
        "source_archive": {
            "path": source["archive_path"],
            "sha256": source["archive_sha256"],
            "byte_count": source["archive_byte_count"],
            "manifest_sha256": source["manifest_sha256"],
            "source_tree_sha256": source["source_tree_sha256"],
            "verification": "archive_manifest_and_exact_read_only_tree_recomputed",
        },
        "phase_a_inputs": phase_a,
        "phase_b_scope": {
            "topology": ["sphere"],
            "branch_pattern_id": 0,
            "formal_clean_parent_count": 512,
            "proposal_count_per_parent": 32,
            "full_k1_all_legal_branches": False,
            "model_acceptance_evidence": False,
        },
        "full_k1_all_legal_branches_pending_fail_closed": True,
        "layout": layout,
        "wrapper_sha256": source["required_file_identity"][_WRAPPER.as_posix()]["sha256"],
        "submission_wrapper": {
            "path": layout["submission_wrapper"],
            "sha256": source["required_file_identity"][_WRAPPER.as_posix()]["sha256"],
            "byte_count": source["required_file_identity"][_WRAPPER.as_posix()][
                "byte_count"
            ],
            "mode": 0o400,
            "nlink": 1,
        },
        "stage_order": list(_STAGES),
        "jobs": jobs,
        "submission_preview": preview,
        "dependency_edges": [["engineering_smoke", "formal_gate"]],
        "slurm_output_paths_overridden_and_audited": True,
        "login_node_heavy_compute_allowed": False,
    }
    return {**core, "plan_sha256": sha256(_canonical_json(core).encode()).hexdigest()}


def _write_json_exclusive(
    path: Path, payload: Mapping[str, object], *, name: str
) -> dict[str, object]:
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fchmod(stream.fileno(), 0o400)
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite launch audit: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    identity = strict_file_identity(path, name=name, require_read_only=True)
    if identity["mode"] != 0o400 or identity["nlink"] != 1:
        raise RuntimeError(f"{name} final permissions/link count are invalid")
    return identity


def _default_runner(request: V5CommandRequest) -> V5CommandResult:
    completed = subprocess.run(  # noqa: S603
        request.argv,
        input=request.script_bytes,
        check=False,
        capture_output=True,
    )
    return V5CommandResult(
        completed.returncode,
        completed.stdout.decode("utf-8", errors="replace"),
        completed.stderr.decode("utf-8", errors="replace"),
    )


def _job_id(result: V5CommandResult, stage: str) -> str:
    if not isinstance(result, V5CommandResult):
        raise TypeError("command runner must return V5CommandResult")
    if result.returncode != 0:
        raise RuntimeError(
            f"{stage} submission failed with return code {result.returncode}; "
            "scheduler output is retained only by digest"
        )
    match = _PARSABLE_JOB_ID.fullmatch(result.stdout.strip())
    if match is None:
        raise RuntimeError(f"{stage} returned an invalid parsable Slurm job id")
    return match.group(1)


def _text_identity(value: str) -> dict[str, object]:
    encoded = value.encode("utf-8")
    return {"sha256": sha256(encoded).hexdigest(), "byte_count": len(encoded)}


def _scheduler_snapshot(
    result: V5CommandResult,
    *,
    stage: str,
    expected_job_id: str,
    expected_dependency_job_id: str | None,
    expected_held: bool,
) -> dict[str, object]:
    if not isinstance(result, V5CommandResult) or result.returncode != 0:
        raise RuntimeError(f"could not read back Slurm state for {stage}")
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Slurm returned an ambiguous state for {stage}")
    fields: dict[str, str] = {}
    for token in shlex.split(lines[0]):
        if "=" in token:
            name, value = token.split("=", 1)
            fields[name] = value
    if fields.get("JobId") != expected_job_id:
        raise RuntimeError(f"Slurm state job id drifted for {stage}")
    state = fields.get("JobState")
    reason = fields.get("Reason")
    dependency = fields.get("Dependency", "")
    normalized_dependency = None if dependency in {"", "(null)", "None"} else dependency
    if expected_held:
        if state != "PENDING" or reason != "JobHeldUser":
            raise RuntimeError(f"Slurm did not retain {stage} in a user hold")
    elif state not in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "COMPLETED"}:
        raise RuntimeError(f"Slurm returned an unsupported released state for {stage}")
    elif reason == "JobHeldUser":
        raise RuntimeError(f"Slurm still reports {stage} as user-held after release")
    if expected_dependency_job_id is None:
        if normalized_dependency is not None:
            raise RuntimeError(f"Slurm attached an unexpected dependency to {stage}")
    elif not str(normalized_dependency).startswith(
        f"afterok:{expected_dependency_job_id}"
    ):
        raise RuntimeError(f"Slurm dependency drifted for {stage}")
    return {
        "job_id": expected_job_id,
        "job_state": state,
        "reason": reason,
        "dependency": normalized_dependency,
        "response": _text_identity(result.stdout),
    }


def _plan_runtime_environment(
    plan: Mapping[str, object], identity: Mapping[str, object]
) -> dict[str, object]:
    prefix = "POSTERIOR_V8_V5_K1_PHASE_B_LAUNCH_PLAN"
    return {
        f"{prefix}_SHA256": plan["plan_sha256"],
        f"{prefix}_FILE_SHA256": identity["sha256"],
        f"{prefix}_BYTE_COUNT": identity["byte_count"],
        f"{prefix}_MODE": identity["mode"],
        f"{prefix}_DEVICE": identity["device"],
        f"{prefix}_INODE": identity["inode"],
        f"{prefix}_MTIME_NS": identity["mtime_ns"],
        f"{prefix}_CTIME_NS": identity["ctime_ns"],
        f"{prefix}_NLINK": identity["nlink"],
    }


def _self_hashed(core: Mapping[str, object], field: str) -> dict[str, object]:
    return {**core, field: sha256(_canonical_json(core).encode()).hexdigest()}


def launch_v5_k1_phase_b_dag(
    config: V5K1PhaseBLaunchConfig,
    *,
    submit: bool = False,
    runner: CommandRunner | None = None,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Dry-run by default; submit smoke and formal jobs only with opt-in."""

    if type(submit) is not bool:
        raise TypeError("submit must be a bool")
    if allowed_root == MAXWELL_DUST_ROOT and any(
        value is not None for value in (runner, hostname, environment)
    ):
        raise RuntimeError("production Maxwell submission forbids injected test seams")
    plan = build_v5_k1_phase_b_launch_plan(config, allowed_root=allowed_root)
    if not submit:
        return {
            "status": "dry_run",
            "writes_performed": False,
            "submissions_performed": False,
            "plan": plan,
        }
    host = socket.gethostname() if hostname is None else hostname
    selected_environment = os.environ if environment is None else environment
    if not host.split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("K1 Phase-B DAG submission is restricted to max-wgs")
    if selected_environment.get("SLURM_JOB_ID"):
        raise RuntimeError("K1 Phase-B launcher must not run inside a Slurm allocation")

    layout = plan["layout"]
    Path(layout["phase_root"]).mkdir(mode=0o700, exist_ok=False)
    for name in ("results_root", "audit_root", "submission_root"):
        Path(layout[name]).mkdir(mode=0o700, exist_ok=False)
    plan_path = Path(layout["plan"])
    receipt_path = Path(layout["receipt"])
    source_wrapper_identity = plan["source"]["required_file_identity"][
        _WRAPPER.as_posix()
    ]
    pinned_wrapper_identity = copy_v5_k1_phase_b_submission_wrapper(
        Path(plan["source"]["source_root"]) / _WRAPPER,
        Path(layout["submission_wrapper"]),
        expected_source_identity=source_wrapper_identity,
    )
    Path(layout["submission_root"]).chmod(0o500)
    plan_identity = _write_json_exclusive(
        plan_path, plan, name="K1 Phase-B launch plan"
    )
    runtime_environment = _plan_runtime_environment(plan, plan_identity)

    command_runner = _default_runner if runner is None else runner
    attempts: list[dict[str, object]] = []
    held_scheduler_snapshots: dict[str, dict[str, object]] = {}
    release_attempts: list[dict[str, object]] = []
    cancellation_attempts: list[dict[str, object]] = []
    job_ids: dict[str, str] = {}
    receipt: dict[str, object] | None = None
    receipt_identity: dict[str, object] | None = None
    stage = _STAGES[0]

    def build_receipt(
        *, status: str, failure: Mapping[str, object] | None
    ) -> dict[str, object]:
        core = {
            "schema_version": V5_K1_PHASE_B_LAUNCH_SCHEMA,
            "version": V5_K1_PHASE_B_LAUNCH_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "plan_sha256": plan["plan_sha256"],
            "plan_file_identity": plan_identity,
            "submission_wrapper_identity": pinned_wrapper_identity,
            "source_archive_sha256": plan["source"]["archive_sha256"],
            "source_manifest_sha256": plan["source"]["manifest_sha256"],
            "source_tree_sha256": plan["source"]["source_tree_sha256"],
            "phase_a_result_sha256": plan["phase_a_inputs"]["artifacts"]["result"]["sha256"],
            "phase_a_model_sha256": plan["phase_a_inputs"]["artifacts"]["model"]["sha256"],
            "phase_a_dataset_sha256": plan["phase_a_inputs"]["artifacts"]["dataset"]["sha256"],
            "phase_a_dataset_binding_sha256": plan["phase_a_inputs"]["dataset_binding_sha256"],
            "plan_path": str(plan_path),
            "receipt_path": str(receipt_path),
            "launch_completion_path": str(layout["launch_completion"]),
            "job_ids": dict(job_ids),
            "dependency_edges": plan["dependency_edges"],
            "submission_attempts": attempts,
            "held_scheduler_snapshots": held_scheduler_snapshots,
            "cancellation_attempts": cancellation_attempts,
            "failure": None if failure is None else dict(failure),
            "execution_environment": {
                "hostname": host,
                "python_version": platform.python_version(),
            },
            "heavy_compute_performed_on_login_node": False,
            "cancellation_attempted": bool(cancellation_attempts),
            "secret_environment_captured": False,
            "model_acceptance_evidence": False,
            "full_k1_all_legal_branches_gate_passed": False,
            "all_jobs_submitted_held": status == "all_jobs_held",
            "formal_submitted_held": status == "all_jobs_held",
            "formal_release_completed": False,
        }
        return _self_hashed(core, "receipt_sha256")

    try:
        for stage in _STAGES:
            replayed_source = inspect_v5_k1_phase_b_source(config, allowed_root)
            if replayed_source != plan["source"]:
                raise RuntimeError("immutable source snapshot changed during DAG submission")
            replayed_phase_a = inspect_v5_k1_phase_a_inputs(
                config,
                source=replayed_source,
                run_root=Path(plan["run_root"]),
                allowed_root=allowed_root,
            )
            if replayed_phase_a != plan["phase_a_inputs"]:
                raise RuntimeError("immutable Phase-A evidence changed during DAG submission")
            if strict_file_identity(
                Path(layout["submission_wrapper"]),
                name="pinned Phase-B submission wrapper",
                require_read_only=True,
            ) != pinned_wrapper_identity:
                raise RuntimeError("pinned Phase-B submission wrapper changed")
            if strict_file_identity(
                plan_path, name="K1 Phase-B launch plan", require_read_only=True
            ) != plan_identity:
                raise RuntimeError("K1 Phase-B launch plan changed during submission")
            script_bytes, script_file_identity = read_only_bytes_identity(
                Path(layout["submission_wrapper"]),
                f"pinned Phase-B {stage} submission wrapper",
            )
            if (
                script_file_identity["sha256"] != pinned_wrapper_identity["sha256"]
                or script_file_identity["byte_count"]
                != pinned_wrapper_identity["byte_count"]
                or script_file_identity["link_count"] != 1
            ):
                raise RuntimeError("pinned Phase-B wrapper changed before stdin submission")
            dependency = plan["jobs"][stage]["depends_on"]
            command = _job_command(
                plan["jobs"][stage],
                None if dependency is None else job_ids[str(dependency)],
                runtime_environment=runtime_environment,
            )
            request = V5CommandRequest(
                argv=command,
                script_bytes=script_bytes,
                script_sha256=str(script_file_identity["sha256"]),
            )
            attempt = {
                "stage": stage,
                "argv": list(command),
                "script_identity": script_file_identity,
                "script_bytes_recorded": False,
            }
            attempts.append(attempt)
            result = command_runner(request)
            if isinstance(result, V5CommandResult):
                attempt.update(
                    {
                        "returncode": result.returncode,
                        "stdout": _text_identity(result.stdout),
                        "stderr": _text_identity(result.stderr),
                    }
                )
            job_ids[stage] = _job_id(result, stage)
            if list(job_ids.values()).count(job_ids[stage]) != 1:
                raise RuntimeError("Phase-B scheduler reused a Slurm job id")
            dependency_job_id = None if dependency is None else job_ids[str(dependency)]
            scheduler_result = command_runner(
                V5CommandRequest(
                    argv=("scontrol", "show", "job", "--oneliner", job_ids[stage])
                )
            )
            held_scheduler_snapshots[stage] = _scheduler_snapshot(
                scheduler_result,
                stage=stage,
                expected_job_id=job_ids[stage],
                expected_dependency_job_id=dependency_job_id,
                expected_held=True,
            )

        receipt = build_receipt(status="all_jobs_held", failure=None)
        receipt_identity = _write_json_exclusive(
            receipt_path, receipt, name="K1 Phase-B held submission receipt"
        )

        for released_stage in reversed(_STAGES):
            release_request = V5CommandRequest(
                argv=("scontrol", "release", job_ids[released_stage])
            )
            release_result = command_runner(release_request)
            attempt = {
                "stage": released_stage,
                "job_id": job_ids[released_stage],
                "argv": list(release_request.argv),
                "returncode": release_result.returncode,
                "stdout": _text_identity(release_result.stdout),
                "stderr": _text_identity(release_result.stderr),
            }
            release_attempts.append(attempt)
            if release_result.returncode != 0:
                raise RuntimeError(f"release failed for {released_stage}")
            dependency = plan["jobs"][released_stage]["depends_on"]
            scheduler_result = command_runner(
                V5CommandRequest(
                    argv=(
                        "scontrol",
                        "show",
                        "job",
                        "--oneliner",
                        job_ids[released_stage],
                    )
                )
            )
            attempt["scheduler_snapshot"] = _scheduler_snapshot(
                scheduler_result,
                stage=released_stage,
                expected_job_id=job_ids[released_stage],
                expected_dependency_job_id=(
                    None if dependency is None else job_ids[str(dependency)]
                ),
                expected_held=False,
            )

        completion_core = {
            "schema_version": V5_K1_PHASE_B_LAUNCH_COMPLETION_SCHEMA,
            "version": V5_K1_PHASE_B_LAUNCH_COMPLETION_VERSION,
            "status": "ALL_JOBS_RELEASED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": plan["plan_sha256"],
            "plan_file_identity": plan_identity,
            "submission_receipt_identity": receipt_identity,
            "job_ids": dict(job_ids),
            "release_order": list(reversed(_STAGES)),
            "released_job_ids": [job_ids[item] for item in reversed(_STAGES)],
            "release_attempts": release_attempts,
            "official_launch_chain_complete": True,
        }
        launch_completion = _self_hashed(
            completion_core, "launch_completion_sha256"
        )
        launch_completion_identity = _write_json_exclusive(
            Path(layout["launch_completion"]),
            launch_completion,
            name="K1 Phase-B launch completion",
        )
        return {
            "status": "all_jobs_released",
            "submission_receipt": receipt,
            "submission_receipt_identity": receipt_identity,
            "launch_completion": launch_completion,
            "launch_completion_identity": launch_completion_identity,
        }
    except (Exception, KeyboardInterrupt) as exc:
        for cancelled_stage in reversed(tuple(job_ids)):
            request = V5CommandRequest(argv=("scancel", job_ids[cancelled_stage]))
            try:
                result = command_runner(request)
                cancellation_attempts.append(
                    {
                        "stage": cancelled_stage,
                        "job_id": job_ids[cancelled_stage],
                        "argv": list(request.argv),
                        "returncode": result.returncode,
                        "stdout": _text_identity(result.stdout),
                        "stderr": _text_identity(result.stderr),
                    }
                )
            except BaseException as cancel_exc:
                cancellation_attempts.append(
                    {
                        "stage": cancelled_stage,
                        "job_id": job_ids[cancelled_stage],
                        "argv": list(request.argv),
                        "exception_type": type(cancel_exc).__name__,
                        "exception_message": str(cancel_exc)[:2000],
                    }
                )
        failure = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        if receipt is None:
            failed_receipt = build_receipt(status="submission_failed", failure=failure)
            _write_json_exclusive(
                receipt_path,
                failed_receipt,
                name="K1 Phase-B failed submission receipt",
            )
            failure_path = receipt_path
        else:
            failure_core = {
                "schema_version": V5_K1_PHASE_B_LAUNCH_COMPLETION_SCHEMA,
                "version": V5_K1_PHASE_B_LAUNCH_COMPLETION_VERSION,
                "status": "RELEASE_FAILED_NO_LAUNCH_COMPLETION",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": plan["plan_sha256"],
                "plan_file_identity": plan_identity,
                "submission_receipt_identity": receipt_identity,
                "job_ids": dict(job_ids),
                "release_attempts": release_attempts,
                "cancellation_attempts": cancellation_attempts,
                "failure": failure,
                "official_launch_chain_complete": False,
            }
            failure_path = Path(layout["release_failure"])
            _write_json_exclusive(
                failure_path,
                _self_hashed(failure_core, "release_failure_sha256"),
                name="K1 Phase-B release failure audit",
            )
        raise V5K1PhaseBLaunchError(
            f"K1 Phase-B DAG launch failed during {stage}; all known jobs were cancelled",
            receipt_path=failure_path,
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--run-root", required=True, type=Path)
    for option in (
        "dataset",
        "dataset-binding",
        "cross-platform-pass-marker",
        "result",
        "model",
        "model-provenance",
    ):
        parser.add_argument(f"--phase-a-{option}", required=True, type=Path)
        parser.add_argument(f"--phase-a-{option}-sha256", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = launch_v5_k1_phase_b_dag(
        V5K1PhaseBLaunchConfig(
            source_root=args.source_root,
            source_archive_path=args.source_archive,
            source_archive_sha256=args.source_archive_sha256,
            run_root=args.run_root,
            phase_a_dataset_path=args.phase_a_dataset,
            phase_a_dataset_sha256=args.phase_a_dataset_sha256,
            phase_a_dataset_binding_path=args.phase_a_dataset_binding,
            phase_a_dataset_binding_sha256=args.phase_a_dataset_binding_sha256,
            phase_a_cross_platform_pass_marker_path=(args.phase_a_cross_platform_pass_marker),
            phase_a_cross_platform_pass_marker_sha256=(
                args.phase_a_cross_platform_pass_marker_sha256
            ),
            phase_a_result_path=args.phase_a_result,
            phase_a_result_sha256=args.phase_a_result_sha256,
            phase_a_model_path=args.phase_a_model,
            phase_a_model_sha256=args.phase_a_model_sha256,
            phase_a_model_provenance_path=args.phase_a_model_provenance,
            phase_a_model_provenance_sha256=args.phase_a_model_provenance_sha256,
        ),
        submit=args.submit and not args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CURRENT_RUN_ROOT_NAME",
    "MAXWELL_DUST_ROOT",
    "PHASE_ROOT_NAME",
    "PLAN_FILENAME",
    "RECEIPT_FILENAME",
    "V5CommandRequest",
    "V5CommandResult",
    "V5K1PhaseBLaunchConfig",
    "V5K1PhaseBLaunchError",
    "V5_K1_PHASE_B_LAUNCH_SCHEMA",
    "V5_K1_PHASE_B_LAUNCH_VERSION",
    "build_v5_k1_phase_b_launch_plan",
    "launch_v5_k1_phase_b_dag",
    "main",
]
