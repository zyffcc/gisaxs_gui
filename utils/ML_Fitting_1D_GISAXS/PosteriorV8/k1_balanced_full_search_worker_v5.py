"""Slurm worker for one immutable balanced all-K1 full-search shard."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import stat
from typing import Mapping, Sequence

from .calibrated_search_threshold_v5 import (
    read_v5_checked_compatibility_calibration,
)
from .formal_production_search_worker_contract_v5 import (
    replay_v5_formal_production_stage_from_artifacts,
)
from .frozen_search_pipeline_v5 import (
    V5FrozenSearchExecution,
    execute_v5_frozen_search_shard,
)
from .grouped_artifact_v5 import canonical_json
from .grouped_dataset_v5 import read_v5_grouped_dataset
from .k1_balanced_full_search_authorization_v5 import (
    authorize_v5_k1_balanced_full_search_task,
    build_v5_k1_balanced_full_search_task_membership,
)
from .k1_balanced_full_search_parent_v5 import (
    project_v5_k1_balanced_full_search_parent,
)
from .k1_balanced_full_search_plan_v5 import (
    validate_v5_k1_balanced_full_search_plan,
)
from .k1_balanced_full_search_runtime_v5 import (
    V5K1BalancedFullSearchExecutableTask,
)
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks, read_regular_bytes
from .k1_training_chain_contract_v5 import digest
from .k1_training_chain_plan_v5 import fingerprint_v5_k1_training_source


V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_task_completion/v1"
)
V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_VERSION = (
    "posterior_v8_v5_2_input_pre_post_pipeline_receipt_completion_last_v1"
)
V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_FILENAME = (
    "task-authorization-v1.json"
)
V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_FILENAME = (
    "balanced-task-completion-v1.json"
)
V5_K1_BALANCED_FULL_SEARCH_TASK_FAILURE_FILENAME = "balanced-task-failure-v1.json"
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")


def _load_json(path: Path, name: str, maximum_bytes: int) -> dict[str, object]:
    encoded = read_regular_bytes(path, name, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def read_v5_k1_balanced_full_search_sealed_file(
    path: Path, name: str, expected_sha256: str | None = None
) -> str:
    selected = lexical_no_symlinks(path, name).resolve(strict=True)
    metadata = selected.stat()
    if (
        not selected.is_file()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"{name} must be a 0400/nlink1 regular file")
    observed = file_sha256(selected, name)
    if expected_sha256 is not None and observed != digest(
        expected_sha256, f"expected {name} SHA-256"
    ):
        raise ValueError(f"{name} file SHA-256 drifted")
    return observed


def _completion_self_sha256(path: Path, expected: str) -> str:
    payload = _load_json(path, "balanced dataset task completion", 8 * 1024 * 1024)
    supplied = digest(payload.pop("completion_sha256", None), "completion_sha256")
    if supplied != sha256(canonical_json(payload).encode()).hexdigest():
        raise ValueError("balanced dataset task completion SHA-256 does not reproduce")
    if supplied != digest(expected, "expected task completion SHA-256"):
        raise ValueError("balanced dataset task completion identity drifted")
    return supplied


def read_v5_k1_balanced_full_search_plan_file(
    path: Path,
    *,
    expected_plan_sha256: str,
    expected_plan_file_sha256: str,
) -> dict[str, object]:
    read_v5_k1_balanced_full_search_sealed_file(
        path,
        "balanced full-search launch plan",
        expected_plan_file_sha256,
    )
    plan = validate_v5_k1_balanced_full_search_plan(
        _load_json(path, "balanced full-search launch plan", 64 * 1024 * 1024)
    )
    if plan["plan_sha256"] != digest(expected_plan_sha256, "expected_plan_sha256"):
        raise ValueError("balanced full-search plan SHA-256 drifted")
    if str(path.resolve(strict=True)) != plan["layout"]["plan"]:
        raise ValueError("balanced full-search plan path escaped its frozen layout")
    return plan


def _task(plan: Mapping[str, object], array_task_id: int) -> Mapping[str, object]:
    if isinstance(array_task_id, bool) or not isinstance(array_task_id, int):
        raise TypeError("array_task_id must be an integer")
    rows = plan["parents"]
    if not 0 <= array_task_id < len(rows):
        raise ValueError("array_task_id is outside the frozen parent inventory")
    selected = rows[array_task_id]
    if selected["array_task_id"] != array_task_id:
        raise RuntimeError("array task mapping does not reproduce")
    return selected


def _worker_guard(
    *, dry_run: bool, hostname: str, environment: Mapping[str, str]
) -> None:
    if dry_run:
        return
    if hostname.split(".", 1)[0].startswith(("max-wgs", "max-fs-display")):
        raise RuntimeError("balanced full search is forbidden on max-wgs or max-fs-display")
    if not environment.get("SLURM_JOB_ID", "").isdigit() or not environment.get(
        "SLURM_ARRAY_TASK_ID", ""
    ).isdigit():
        raise RuntimeError("balanced full search requires a Slurm array worker")


def _source_bundle(source_root: Path, expected: str) -> dict[str, object]:
    observed = fingerprint_v5_k1_training_source(source_root)
    if observed["bundle_sha256"] != digest(expected, "source bundle SHA-256"):
        raise RuntimeError("immutable source snapshot bundle changed")
    if observed["source_snapshot_write_bits_set"] or observed[
        "source_files_with_write_bits"
    ]:
        raise ValueError("immutable source snapshot or required files are writable")
    return observed


def _read_parent(selected: Mapping[str, object]):
    binding = selected["parent"]
    path = Path(binding["path"])
    observed_file = read_v5_k1_balanced_full_search_sealed_file(
        path,
        "balanced full-search source parent",
        binding["artifact_sha256"],
    )
    parent, receipt = read_v5_grouped_dataset(path)
    if (
        observed_file != receipt.artifact_sha256
        or receipt.manifest_sha256 != binding["manifest_sha256"]
        or receipt.byte_count != binding["byte_count"]
    ):
        raise ValueError("balanced source parent artifact identity drifted")
    return parent, receipt


def read_v5_k1_balanced_full_search_task_input_identity(
    *,
    plan_path: Path,
    source_root: Path,
    source_archive: Path,
    schedule_path: Path,
    calibration_path: Path,
    selected: Mapping[str, object],
    plan: Mapping[str, object],
) -> dict[str, object]:
    return _task_input_identity_after_source_replay(
        plan_path=plan_path, source_archive=source_archive,
        schedule_path=schedule_path, calibration_path=calibration_path,
        selected=selected, plan=plan,
        source_bundle_sha256=_source_bundle(source_root, plan["source"]["bundle_sha256"])["bundle_sha256"],
    )


def _task_input_identity_after_source_replay(
    *, plan_path, source_archive, schedule_path, calibration_path,
    selected, plan, source_bundle_sha256,
) -> dict[str, object]:
    """Shared file checks after the caller has replayed the actual source tree."""
    if source_bundle_sha256 != plan["source"]["bundle_sha256"]:
        raise ValueError("replayed source bundle differs from the frozen search plan")
    parent_binding = selected["parent"]
    completion_binding = selected["task_completion"]
    completion_path = Path(completion_binding["path"])
    completion_file = read_v5_k1_balanced_full_search_sealed_file(
        completion_path,
        "balanced dataset task completion",
        completion_binding["file_sha256"],
    )
    completion_self = _completion_self_sha256(
        completion_path,
        completion_binding["completion_sha256"],
    )
    return {
        "launch_plan_file_sha256": read_v5_k1_balanced_full_search_sealed_file(
            plan_path, "balanced full-search launch plan"
        ),
        "source_archive_sha256": read_v5_k1_balanced_full_search_sealed_file(
            source_archive,
            "source archive",
            plan["source"]["archive_sha256"],
        ),
        "source_bundle_sha256": source_bundle_sha256,
        "local_sobol_schedule_file_sha256": read_v5_k1_balanced_full_search_sealed_file(
            schedule_path, "local Sobol schedule"
        ),
        "calibration_file_sha256": read_v5_k1_balanced_full_search_sealed_file(
            calibration_path,
            "compatibility calibration",
            plan["k1_stage"]["payload"]["protocol"]["calibration_identity"][
                "file_sha256"
            ],
        ),
        "source_parent_file_sha256": read_v5_k1_balanced_full_search_sealed_file(
            Path(parent_binding["path"]),
            "balanced full-search source parent",
            parent_binding["artifact_sha256"],
        ),
        "source_parent_manifest_sha256": parent_binding["manifest_sha256"],
        "source_parent_byte_count": parent_binding["byte_count"],
        "source_task_completion_file_sha256": completion_file,
        "source_task_completion_sha256": completion_self,
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, object]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return file_sha256(path, path.name)


def _seal_pipeline_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("balanced search output contains a symlink")
        if path.is_file():
            if path.stat().st_nlink != 1:
                raise ValueError("balanced search output contains a hard-linked file")
            path.chmod(0o400)
    directories = sorted(
        (value for value in root.rglob("*") if value.is_dir()),
        key=lambda value: len(value.parts),
        reverse=True,
    )
    for path in directories:
        path.chmod(0o500)


def run_v5_k1_balanced_full_search_task(
    plan_path: str | os.PathLike[str],
    array_task_id: int,
    *,
    expected_plan_sha256: str,
    expected_plan_file_sha256: str,
    source_root: str | os.PathLike[str],
    source_archive: str | os.PathLike[str],
    local_sobol_schedule_path: str | os.PathLike[str],
    calibration_path: str | os.PathLike[str],
    dry_run: bool = False,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Execute one plan-selected task and publish a second completion last."""

    if type(dry_run) is not bool:
        raise TypeError("dry_run must be a bool")
    host = socket.gethostname() if hostname is None else hostname
    env = os.environ if environment is None else environment
    _worker_guard(dry_run=dry_run, hostname=host, environment=env)
    paths = {
        "plan": Path(plan_path),
        "source_root": Path(source_root),
        "source_archive": Path(source_archive),
        "schedule": Path(local_sobol_schedule_path),
        "calibration": Path(calibration_path),
    }
    for name, path in paths.items():
        paths[name] = lexical_no_symlinks(path, name).resolve(strict=True)
        if not paths[name].is_relative_to(MAXWELL_DUST_ROOT):
            raise ValueError(f"{name} must remain under {MAXWELL_DUST_ROOT}")
    plan = read_v5_k1_balanced_full_search_plan_file(
        paths["plan"],
        expected_plan_sha256=expected_plan_sha256,
        expected_plan_file_sha256=expected_plan_file_sha256,
    )
    selected = _task(plan, array_task_id)
    output = Path(selected["search_output_root"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to reuse a balanced full-search output root")
    inputs_before = read_v5_k1_balanced_full_search_task_input_identity(
        plan_path=paths["plan"],
        source_root=paths["source_root"],
        source_archive=paths["source_archive"],
        schedule_path=paths["schedule"],
        calibration_path=paths["calibration"],
        selected=selected,
        plan=plan,
    )
    if dry_run:
        return {
            "status": "checked_dry_run",
            "writes_performed": False,
            "plan_sha256": plan["plan_sha256"],
            "array_task_id": array_task_id,
            "source_parent_artifact_sha256": selected["parent"]["artifact_sha256"],
            "input_identity": inputs_before,
            "search_evidence_created": False,
            "gradient_training_authorized": False,
        }
    if str(array_task_id) != env["SLURM_ARRAY_TASK_ID"]:
        raise RuntimeError("array_task_id differs from SLURM_ARRAY_TASK_ID")
    parent, _ = _read_parent(selected)
    projection = project_v5_k1_balanced_full_search_parent(
        parent,
        source_parent_artifact_sha256=selected["parent"]["artifact_sha256"],
        expected_balanced_dataset_plan_sha256=plan["identity_authorization"][
            "populations"
        ][selected["role"]]["plan_sha256"],
        expected_balanced_sobol_block_sha256=selected[
            "balanced_sobol_block_sha256"
        ],
        expected_role=selected["role"],
        expected_split_id=selected["split_id"],
        candidate_view_indices=tuple(plan["configuration"]["candidate_view_indices"]),
    )
    membership = build_v5_k1_balanced_full_search_task_membership(
        projection,
        array_task_id=array_task_id,
        shard_index=selected["shard_index"],
        split_offset=selected["split_offset"],
        source_selection_sha256=selected["selection_sha256"],
    )
    authorization = authorize_v5_k1_balanced_full_search_task(plan, membership)
    stage_binding = plan["k1_stage"]
    calibration_identity = stage_binding["payload"]["protocol"][
        "calibration_identity"
    ]
    calibration = read_v5_checked_compatibility_calibration(
        paths["calibration"],
        expected_artifact_sha256=calibration_identity["artifact_sha256"],
        expected_file_sha256=calibration_identity["file_sha256"],
    )
    stage = replay_v5_formal_production_stage_from_artifacts(
        stage_payload={
            **stage_binding["payload"],
            "stage_sha256": stage_binding["stage_sha256"],
        },
        schedule_path=paths["schedule"],
        calibration=calibration,
    )
    runtime = V5K1BalancedFullSearchExecutableTask(
        projection=projection,
        task_authorization=authorization,
        topology_schedule=stage.topology_schedule,
    )
    execution = V5FrozenSearchExecution(
        seed_schedule=stage.seed_schedule,
        optimizer_schedule=stage.optimizer_schedule,
        protocol=stage.protocol,
        calibration=calibration,
        launch_source_bundle_sha256=plan["source"]["bundle_sha256"],
        launch_plan_sha256=plan["plan_sha256"],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    authorization_path = output / V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_FILENAME
    _write_json_exclusive(authorization_path, authorization.to_payload())
    try:
        pipeline_completion = execute_v5_frozen_search_shard(
            runtime,
            execution,
            output,
            source_bundle_fingerprint=lambda: _source_bundle(
                paths["source_root"], plan["source"]["bundle_sha256"]
            )["bundle_sha256"],
            formal_production_authorization=authorization.formal_authorization,
        )
        inputs_after = read_v5_k1_balanced_full_search_task_input_identity(
            plan_path=paths["plan"],
            source_root=paths["source_root"],
            source_archive=paths["source_archive"],
            schedule_path=paths["schedule"],
            calibration_path=paths["calibration"],
            selected=selected,
            plan=plan,
        )
        if inputs_after != inputs_before:
            raise RuntimeError("immutable balanced search inputs changed during execution")
        pipeline_completion_path = output / "completion.json"
        pipeline_completion_file = file_sha256(
            pipeline_completion_path, "frozen pipeline completion"
        )
        _seal_pipeline_tree(output)
        core = {
            "schema": V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_SCHEMA,
            "version": V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_VERSION,
            "status": "PASS",
            "scientific_acceptance_evidence": False,
            "full_search_supervision_complete": False,
            "gradient_training_authorized": False,
            "plan_sha256": plan["plan_sha256"],
            "array_task_id": array_task_id,
            "slurm_job_id": env["SLURM_JOB_ID"],
            "hostname": host,
            "task_authorization": {
                "path": str(authorization_path),
                "file_sha256": file_sha256(
                    authorization_path, "task authorization"
                ),
                "task_authorization_sha256": authorization.sha256,
                "task_plan_sha256": authorization.task_plan_sha256,
            },
            "pipeline_completion": {
                "path": str(pipeline_completion_path),
                "file_sha256": pipeline_completion_file,
                "completion_sha256": pipeline_completion["completion_sha256"],
                "parent_artifact_sha256": pipeline_completion["parent"][
                    "artifact_sha256"
                ],
                "sidecar_artifact_sha256": pipeline_completion["sidecar"][
                    "artifact_sha256"
                ],
                "evidence_receipt_file_sha256": pipeline_completion[
                    "task_bound_evidence_receipt"
                ]["file_sha256"],
                "evidence_receipt_sha256": pipeline_completion[
                    "task_bound_evidence_receipt"
                ]["receipt_sha256"],
            },
            "counts": {
                "queries": pipeline_completion["sidecar"]["queries"],
                "branches": pipeline_completion["sidecar"]["branches"],
                "expected_exact_forward_calls": (
                    authorization.formal_authorization.expected_exact_forward_calls
                ),
            },
            "immutable_input_identity_pre": inputs_before,
            "immutable_input_identity_post": inputs_after,
            "pipeline_tree_files_sealed_before_task_completion": True,
            "task_completion_written_last": True,
        }
        completion = {
            **core,
            "completion_sha256": sha256(
                canonical_json(core).encode("utf-8")
            ).hexdigest(),
        }
        completion_path = output / V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_FILENAME
        _write_json_exclusive(completion_path, completion)
        output.chmod(0o500)
        return completion
    except (Exception, KeyboardInterrupt) as exc:
        failure_path = output / V5_K1_BALANCED_FULL_SEARCH_TASK_FAILURE_FILENAME
        if not failure_path.exists():
            failure_core = {
                "schema": V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_SCHEMA,
                "version": V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_VERSION,
                "status": "FAIL",
                "plan_sha256": plan["plan_sha256"],
                "array_task_id": array_task_id,
                "slurm_job_id": env.get("SLURM_JOB_ID"),
                "hostname": host,
                "exception_type": type(exc).__name__,
                "message": str(exc)[:2000],
                "partial_outputs_retained": True,
                "outputs_are_never_overwritten": True,
            }
            failure = {
                **failure_core,
                "failure_sha256": sha256(
                    canonical_json(failure_core).encode("utf-8")
                ).hexdigest(),
            }
            _write_json_exclusive(failure_path, failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--array-task-id", required=True, type=int)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-plan-file-sha256", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--local-sobol-schedule", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_v5_k1_balanced_full_search_task(
        args.plan,
        args.array_task_id,
        expected_plan_sha256=args.expected_plan_sha256,
        expected_plan_file_sha256=args.expected_plan_file_sha256,
        source_root=args.source_root,
        source_archive=args.source_archive,
        local_sobol_schedule_path=args.local_sobol_schedule,
        calibration_path=args.calibration,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_FILENAME",
    "V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_FILENAME",
    "V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_SCHEMA",
    "V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_VERSION",
    "V5_K1_BALANCED_FULL_SEARCH_TASK_FAILURE_FILENAME",
    "main",
    "read_v5_k1_balanced_full_search_plan_file",
    "read_v5_k1_balanced_full_search_sealed_file",
    "read_v5_k1_balanced_full_search_task_input_identity",
    "run_v5_k1_balanced_full_search_task",
]
