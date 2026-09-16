"""Completion-last collector for the 60 balanced all-K1 search tasks."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import stat
from typing import Mapping, Sequence

from .formal_production_search_plan_v5 import (
    V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE,
    V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE,
)
from .frozen_search_pipeline_contract_v5 import (
    V5_FROZEN_SEARCH_PIPELINE_SCHEMA,
    V5_SEARCH_PIPELINE_TRAINING_SCOPE,
)
from .grouped_artifact_v5 import canonical_json
from .k1_balanced_full_search_authorization_v5 import (
    V5K1BalancedFullSearchTaskAuthorization,
    V5K1BalancedFullSearchTaskMembership,
    authorize_v5_k1_balanced_full_search_task,
)
from .k1_balanced_full_search_worker_v5 import (
    MAXWELL_DUST_ROOT,
    V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_FILENAME,
    V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_FILENAME,
    V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_SCHEMA,
    V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_VERSION,
    read_v5_k1_balanced_full_search_plan_file,
    read_v5_k1_balanced_full_search_sealed_file,
    read_v5_k1_balanced_full_search_task_input_identity,
)
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks, read_regular_bytes
from .search_evidence_receipt_v5 import (
    V5_SEARCH_LABEL_PURPOSE_TRAINING,
    read_v5_search_evidence_receipt,
)


V5_K1_BALANCED_FULL_SEARCH_INVENTORY_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_inventory/v1"
)
V5_K1_BALANCED_FULL_SEARCH_INVENTORY_VERSION = (
    "posterior_v8_v5_2_all60_task_bound_receipts_replayed_v1"
)
V5_K1_BALANCED_FULL_SEARCH_COLLECTION_COMPLETION_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_collection_completion/v1"
)
V5_K1_BALANCED_FULL_SEARCH_COLLECTION_COMPLETION_VERSION = (
    "posterior_v8_v5_2_inventory_sealed_then_completion_last_v1"
)
V5_K1_BALANCED_FULL_SEARCH_INVENTORY_FILENAME = "full-search-inventory-v1.json"
_TASK_COMPLETION_FIELDS = {
    "schema",
    "version",
    "status",
    "scientific_acceptance_evidence",
    "full_search_supervision_complete",
    "gradient_training_authorized",
    "plan_sha256",
    "array_task_id",
    "slurm_job_id",
    "hostname",
    "task_authorization",
    "pipeline_completion",
    "counts",
    "immutable_input_identity_pre",
    "immutable_input_identity_post",
    "pipeline_tree_files_sealed_before_task_completion",
    "task_completion_written_last",
    "completion_sha256",
}


def _read_json(path: Path, name: str) -> dict[str, object]:
    raw = read_regular_bytes(path, name, maximum_bytes=64 * 1024 * 1024)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _self_hashed(
    payload: Mapping[str, object], *, field: str, name: str
) -> dict[str, object]:
    values = dict(payload)
    if set(values) != _TASK_COMPLETION_FIELDS and name == "task completion":
        raise ValueError("balanced full-search task completion fields drifted")
    supplied = values.pop(field, None)
    if not isinstance(supplied, str) or supplied != sha256(
        canonical_json(values).encode("utf-8")
    ).hexdigest():
        raise ValueError(f"{name} self SHA-256 does not reproduce")
    return dict(payload)


def _write_json_exclusive(
    path: Path, payload: Mapping[str, object], *, hash_field: str
) -> tuple[dict[str, object], str]:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path.name}")
    core = dict(payload)
    value = {
        **core,
        hash_field: sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    return value, file_sha256(path, path.name)


def _worker_guard(
    *, dry_run: bool, hostname: str, environment: Mapping[str, str]
) -> None:
    if dry_run:
        return
    if hostname.split(".", 1)[0].startswith(("max-wgs", "max-fs-display")):
        raise RuntimeError("balanced full-search collection is forbidden on max-wgs or max-fs-display")
    if not environment.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("balanced full-search collection requires a Slurm worker")
    if environment.get("SLURM_ARRAY_TASK_ID"):
        raise RuntimeError("balanced full-search collector cannot be an array task")


def _output_tree_is_sealed(root: Path) -> None:
    selected = lexical_no_symlinks(root, "balanced search output").resolve(strict=True)
    if not selected.is_dir() or stat.S_IMODE(selected.stat().st_mode) != 0o500:
        raise ValueError("balanced search output root must be a 0500 directory")
    for path in selected.rglob("*"):
        if path.is_symlink():
            raise ValueError("balanced search output contains a symlink")
        metadata = path.stat()
        if path.is_dir() and stat.S_IMODE(metadata.st_mode) != 0o500:
            raise ValueError("balanced search output directory is not 0500")
        if path.is_file() and (
            stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1
        ):
            raise ValueError("balanced search output file is not 0400/nlink1")


def _consumer_role(role: str) -> str:
    if role == "train":
        return V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE
    if role == "tuning_validation":
        return V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE
    raise ValueError("balanced full-search task role is unsupported")


def _task_completion(
    plan: Mapping[str, object], selected: Mapping[str, object]
) -> tuple[dict[str, object], Path, str]:
    output = Path(selected["search_output_root"])
    _output_tree_is_sealed(output)
    path = output / V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_FILENAME
    file_sha = read_v5_k1_balanced_full_search_sealed_file(
        path, "balanced full-search task completion"
    )
    completion = _self_hashed(
        _read_json(path, "balanced full-search task completion"),
        field="completion_sha256",
        name="task completion",
    )
    fixed = {
        "schema": V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_SCHEMA,
        "version": V5_K1_BALANCED_FULL_SEARCH_TASK_COMPLETION_VERSION,
        "status": "PASS",
        "scientific_acceptance_evidence": False,
        "full_search_supervision_complete": False,
        "gradient_training_authorized": False,
        "plan_sha256": plan["plan_sha256"],
        "array_task_id": selected["array_task_id"],
        "pipeline_tree_files_sealed_before_task_completion": True,
        "task_completion_written_last": True,
    }
    if any(completion.get(name) != value for name, value in fixed.items()):
        raise ValueError("balanced full-search task completion claims drifted")
    if completion["immutable_input_identity_pre"] != completion[
        "immutable_input_identity_post"
    ]:
        raise ValueError("balanced full-search immutable inputs changed during search")
    return completion, path, file_sha


def _checked_task(
    plan: Mapping[str, object], selected: Mapping[str, object]
) -> dict[str, object]:
    completion, completion_path, completion_file_sha = _task_completion(plan, selected)
    output = Path(selected["search_output_root"])
    authorization_path = output / V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_FILENAME
    authorization_file_sha = read_v5_k1_balanced_full_search_sealed_file(
        authorization_path, "balanced full-search task authorization"
    )
    authorization = V5K1BalancedFullSearchTaskAuthorization.from_payload(
        _read_json(authorization_path, "balanced full-search task authorization")
    )
    membership = V5K1BalancedFullSearchTaskMembership.from_payload(
        authorization.task_plan["membership"]
    )
    expected = authorize_v5_k1_balanced_full_search_task(plan, membership)
    if authorization.to_payload() != expected.to_payload():
        raise ValueError("balanced full-search task authorization escaped its plan")
    if completion["task_authorization"] != {
        "path": str(authorization_path),
        "file_sha256": authorization_file_sha,
        "task_authorization_sha256": authorization.sha256,
        "task_plan_sha256": authorization.task_plan_sha256,
    }:
        raise ValueError("task completion lost its task authorization binding")
    pipeline_path = output / "completion.json"
    pipeline_file_sha = read_v5_k1_balanced_full_search_sealed_file(
        pipeline_path, "frozen search completion"
    )
    pipeline = _self_hashed(
        _read_json(pipeline_path, "frozen search completion"),
        field="completion_sha256",
        name="pipeline completion",
    )
    formal = authorization.formal_authorization
    fixed_pipeline = {
        "schema": V5_FROZEN_SEARCH_PIPELINE_SCHEMA,
        "status": "complete",
        "pipeline_plan_sha256": authorization.task_plan_sha256,
        "launch_plan_sha256": plan["plan_sha256"],
        "scientific_scope": V5_SEARCH_PIPELINE_TRAINING_SCOPE,
        "label_purpose": V5_SEARCH_LABEL_PURPOSE_TRAINING,
        "full_training_label_claimed": True,
        "authorized_consumer_role": _consumer_role(selected["role"]),
    }
    if any(pipeline.get(name) != value for name, value in fixed_pipeline.items()):
        raise ValueError("frozen search completion claims drifted")
    if completion["pipeline_completion"] != {
        "path": str(pipeline_path),
        "file_sha256": pipeline_file_sha,
        "completion_sha256": pipeline["completion_sha256"],
        "parent_artifact_sha256": pipeline["parent"]["artifact_sha256"],
        "sidecar_artifact_sha256": pipeline["sidecar"]["artifact_sha256"],
        "evidence_receipt_file_sha256": pipeline[
            "task_bound_evidence_receipt"
        ]["file_sha256"],
        "evidence_receipt_sha256": pipeline["task_bound_evidence_receipt"][
            "receipt_sha256"
        ],
    }:
        raise ValueError("task completion lost its frozen pipeline binding")
    parent_path = Path(pipeline["parent"]["path"])
    sidecar_path = Path(pipeline["sidecar"]["path"])
    receipt_path = Path(pipeline["task_bound_evidence_receipt"]["path"])
    for path, name in (
        (parent_path, "projected search parent"),
        (sidecar_path, "search supervision sidecar"),
        (receipt_path, "task-bound evidence receipt"),
    ):
        read_v5_k1_balanced_full_search_sealed_file(path, name)
        if output not in path.resolve(strict=True).parents:
            raise ValueError(f"{name} escaped its task output root")
    receipt = read_v5_search_evidence_receipt(
        receipt_path,
        parent_dataset_path=parent_path,
        sidecar_path=sidecar_path,
        require_training_eligible=True,
        expected_consumer_role=formal.consumer_role,
    )
    counts = receipt.manifest["counts"]
    expected_counts = {
        "queries": formal.expected_query_count,
        "branches": formal.expected_branch_count,
        "expected_exact_forward_calls": formal.expected_exact_forward_calls,
    }
    if completion["counts"] != expected_counts or counts != {
        "queries": formal.expected_query_count,
        "branches": formal.expected_branch_count,
        "exact_forward_calls_used": formal.expected_exact_forward_calls,
    }:
        raise ValueError("balanced full-search task budget or counts drifted")
    return {
        "array_task_id": selected["array_task_id"],
        "role": selected["role"],
        "branch_id": selected["branch_id"],
        "shard_index": selected["shard_index"],
        "recipe_count": selected["recipe_count"],
        "source_parent_artifact_sha256": selected["parent"]["artifact_sha256"],
        "projected_parent_path": str(parent_path),
        "projected_parent_artifact_sha256": pipeline["parent"]["artifact_sha256"],
        "sidecar_path": str(sidecar_path),
        "sidecar_artifact_sha256": receipt.sidecar_artifact_sha256,
        "evidence_receipt_path": str(receipt_path),
        "evidence_receipt_file_sha256": receipt.file_sha256,
        "evidence_receipt_sha256": receipt.manifest["receipt_sha256"],
        "task_authorization_sha256": authorization.sha256,
        "task_plan_sha256": authorization.task_plan_sha256,
        "task_completion_path": str(completion_path),
        "task_completion_file_sha256": completion_file_sha,
        "task_completion_sha256": completion["completion_sha256"],
        "query_count": formal.expected_query_count,
        "branch_count": formal.expected_branch_count,
        "exact_forward_calls_used": formal.expected_exact_forward_calls,
    }


def _source_input_identity(
    *,
    plan: Mapping[str, object],
    plan_path: Path,
    source_root: Path,
    source_archive: Path,
    schedule_path: Path,
    calibration_path: Path,
) -> tuple[dict[str, object], ...]:
    return tuple(
        read_v5_k1_balanced_full_search_task_input_identity(
            plan_path=plan_path,
            source_root=source_root,
            source_archive=source_archive,
            schedule_path=schedule_path,
            calibration_path=calibration_path,
            selected=selected,
            plan=plan,
        )
        for selected in plan["parents"]
    )


def _aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_role = {}
    for role in ("train", "tuning_validation"):
        selected = tuple(value for value in rows if value["role"] == role)
        by_role[role] = {
            "task_count": len(selected),
            "clean_parent_count": sum(value["recipe_count"] for value in selected),
            "query_count": sum(value["query_count"] for value in selected),
            "branch_count": sum(value["branch_count"] for value in selected),
            "exact_forward_calls_used": sum(
                value["exact_forward_calls_used"] for value in selected
            ),
        }
    if by_role["train"]["task_count"] != 48 or by_role[
        "tuning_validation"
    ]["task_count"] != 12:
        raise ValueError("balanced full-search task totals drifted")
    if by_role["train"]["clean_parent_count"] != 13824 or by_role[
        "tuning_validation"
    ]["clean_parent_count"] != 3456:
        raise ValueError("balanced full-search parent totals drifted")
    return by_role


def _collect_actual(
    *,
    plan: Mapping[str, object],
    paths: Mapping[str, Path],
    inventory_path: Path,
    completion_path: Path,
    host: str,
    environment: Mapping[str, str],
) -> dict[str, object]:
    inputs_before = _source_input_identity(
        plan=plan,
        plan_path=paths["plan"],
        source_root=paths["source_root"],
        source_archive=paths["source_archive"],
        schedule_path=paths["schedule"],
        calibration_path=paths["calibration"],
    )
    rows = tuple(_checked_task(plan, selected) for selected in plan["parents"])
    if len(rows) != 60 or tuple(value["array_task_id"] for value in rows) != tuple(
        range(60)
    ):
        raise ValueError("balanced full-search task inventory is incomplete or unordered")
    totals = _aggregate(rows)
    inputs_after = _source_input_identity(
        plan=plan,
        plan_path=paths["plan"],
        source_root=paths["source_root"],
        source_archive=paths["source_archive"],
        schedule_path=paths["schedule"],
        calibration_path=paths["calibration"],
    )
    if inputs_after != inputs_before:
        raise RuntimeError("balanced full-search source inputs changed during collection")
    replay_rows = tuple(_checked_task(plan, selected) for selected in plan["parents"])
    if replay_rows != rows:
        raise RuntimeError("balanced full-search task evidence changed during collection")
    inventory_core = {
        "schema": V5_K1_BALANCED_FULL_SEARCH_INVENTORY_SCHEMA,
        "version": V5_K1_BALANCED_FULL_SEARCH_INVENTORY_VERSION,
        "status": "PASS",
        "plan_sha256": plan["plan_sha256"],
        "source_bundle_sha256": plan["source"]["bundle_sha256"],
        "identity_authorization_sha256": plan["identity_authorization_sha256"],
        "task_count": len(rows),
        "totals": totals,
        "tasks": list(rows),
        "immutable_input_identity_pre": list(inputs_before),
        "immutable_input_identity_post": list(inputs_after),
        "all_task_bound_search_evidence_replayed_twice": True,
        "full_search_supervision_complete": True,
        "tuning_exact_budget_summary_complete": False,
        "gradient_training_authorized": False,
        "scientific_acceptance_evidence": False,
    }
    inventory, inventory_file_sha = _write_json_exclusive(
        inventory_path, inventory_core, hash_field="inventory_sha256"
    )
    completion_core = {
        "schema": V5_K1_BALANCED_FULL_SEARCH_COLLECTION_COMPLETION_SCHEMA,
        "version": V5_K1_BALANCED_FULL_SEARCH_COLLECTION_COMPLETION_VERSION,
        "status": "PASS",
        "plan_sha256": plan["plan_sha256"],
        "slurm_job_id": environment["SLURM_JOB_ID"],
        "hostname": host,
        "inventory": {
            "path": str(inventory_path),
            "file_sha256": inventory_file_sha,
            "inventory_sha256": inventory["inventory_sha256"],
        },
        "task_count": len(rows),
        "totals": totals,
        "all_task_bound_search_evidence_replayed_twice": True,
        "full_search_supervision_complete": True,
        "tuning_exact_budget_summary_complete": False,
        "gradient_training_authorized": False,
        "phase_c_model_acceptance": False,
        "paper_model_acceptance": False,
        "inventory_sealed_before_completion": True,
        "completion_written_last": True,
    }
    completion, _ = _write_json_exclusive(
        completion_path, completion_core, hash_field="completion_sha256"
    )
    return completion


def collect_v5_k1_balanced_full_search(
    plan_path: str | os.PathLike[str],
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
    """Replay all task evidence and publish an inventory before completion."""

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
    completion_path = Path(plan["layout"]["completion"])
    inventory_path = completion_path.parent / V5_K1_BALANCED_FULL_SEARCH_INVENTORY_FILENAME
    for path in (inventory_path, completion_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError("refusing to reuse balanced full-search collection output")
    if dry_run:
        return {
            "status": "checked_dry_run",
            "writes_performed": False,
            "plan_sha256": plan["plan_sha256"],
            "expected_task_count": len(plan["parents"]),
            "full_search_supervision_complete": False,
            "gradient_training_authorized": False,
        }
    failure_path = Path(plan["layout"]["failure"])
    try:
        return _collect_actual(
            plan=plan,
            paths=paths,
            inventory_path=inventory_path,
            completion_path=completion_path,
            host=host,
            environment=env,
        )
    except (Exception, KeyboardInterrupt) as exc:
        if not failure_path.exists() and not failure_path.is_symlink():
            failure_core = {
                "schema": V5_K1_BALANCED_FULL_SEARCH_COLLECTION_COMPLETION_SCHEMA,
                "version": V5_K1_BALANCED_FULL_SEARCH_COLLECTION_COMPLETION_VERSION,
                "status": "FAIL",
                "plan_sha256": plan["plan_sha256"],
                "slurm_job_id": env.get("SLURM_JOB_ID"),
                "hostname": host,
                "exception_type": type(exc).__name__,
                "message": str(exc)[:2000],
                "partial_outputs_retained": True,
                "outputs_are_never_overwritten": True,
            }
            _write_json_exclusive(
                failure_path, failure_core, hash_field="failure_sha256"
            )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
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
    result = collect_v5_k1_balanced_full_search(
        args.plan,
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
    "V5_K1_BALANCED_FULL_SEARCH_COLLECTION_COMPLETION_SCHEMA",
    "V5_K1_BALANCED_FULL_SEARCH_COLLECTION_COMPLETION_VERSION",
    "V5_K1_BALANCED_FULL_SEARCH_INVENTORY_FILENAME",
    "V5_K1_BALANCED_FULL_SEARCH_INVENTORY_SCHEMA",
    "V5_K1_BALANCED_FULL_SEARCH_INVENTORY_VERSION",
    "collect_v5_k1_balanced_full_search",
    "main",
]
