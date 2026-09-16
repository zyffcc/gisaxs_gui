"""Collect immutable balanced K1 shards and prove actual train/tune disjointness."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import stat
from typing import Mapping, Sequence

from .grouped_artifact_v5 import canonical_json
from .k1_balanced_dataset_launch_plan_v5 import (
    MAXWELL_DUST_ROOT,
    replay_v5_k1_balanced_dataset_launch_inputs,
    validate_v5_k1_balanced_dataset_launch_plan,
)
from .k1_balanced_dataset_worker_v5 import (
    V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA,
    V5_K1_BALANCED_DATASET_COMPLETION_VERSION,
)
from .k1_dataset_disjointness_v5 import (
    V5K1GroupedArtifactBinding,
    build_v5_k1_train_tuning_disjointness_receipt,
    population_from_v5_k1_grouped_artifacts,
    write_v5_k1_train_tuning_disjointness_receipt,
)
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks, read_regular_bytes
from .k1_training_chain_contract_v5 import digest


V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_dataset_collection/v2"
)
V5_K1_BALANCED_DATASET_COLLECTION_VERSION = (
    "posterior_v8_v5_2_all12_actual_identity_collection_completion_last_v2"
)


def _load_json(path: Path, name: str, maximum_bytes: int) -> dict[str, object]:
    encoded = read_regular_bytes(path, name, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return value


def _immutable_file(path: Path, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a real regular file")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError(f"{name} must be 0400/nlink1")


def _load_plan(path: Path, expected_sha256: str) -> dict[str, object]:
    _immutable_file(path, "balanced dataset launch plan")
    plan = validate_v5_k1_balanced_dataset_launch_plan(
        _load_json(path, "balanced dataset launch plan", 16 * 1024 * 1024)
    )
    if plan["plan_sha256"] != digest(expected_sha256, "expected_plan_sha256"):
        raise ValueError("launch plan SHA-256 differs from the Slurm export binding")
    if str(path.resolve(strict=True)) != plan["layout"]["plan"]:
        raise ValueError("launch plan path is not its frozen publication path")
    return plan


def _worker_guard(hostname: str, environment: Mapping[str, str]) -> None:
    if hostname.split(".", 1)[0].startswith(("max-wgs", "max-fs-display")):
        raise RuntimeError("balanced K1 dataset collection is forbidden on Maxwell login nodes")
    job_id = environment.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise RuntimeError("balanced K1 dataset collection requires a Slurm worker")


def _completion(
    task: Mapping[str, object],
    *,
    plan_sha256: str,
    input_identity: Mapping[str, object],
) -> tuple[dict[str, object], V5K1GroupedArtifactBinding]:
    path = Path(task["completion"])
    _immutable_file(path, "balanced dataset task completion")
    value = _load_json(path, "balanced dataset task completion", 4 * 1024 * 1024)
    core = dict(value)
    supplied = digest(core.pop("completion_sha256", None), "completion_sha256")
    if supplied != sha256(canonical_json(core).encode()).hexdigest():
        raise ValueError("balanced dataset task completion SHA-256 does not reproduce")
    if (
        value.get("schema") != V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA
        or value.get("version") != V5_K1_BALANCED_DATASET_COMPLETION_VERSION
        or value.get("status") != "PASS"
        or value.get("plan_sha256") != plan_sha256
        or value.get("array_task_id") != task["array_task_id"]
        or value.get("task") != dict(task)
        or value.get("immutable_input_identity_pre") != input_identity
        or value.get("immutable_input_identity_post") != input_identity
        or value.get("completion_written_after_artifact_seal") is not True
        or value.get("scientific_acceptance_evidence") is not False
        or value.get("training_authorization_granted") is not False
    ):
        raise ValueError("balanced dataset task completion contract drifted")
    artifact = value.get("artifact")
    expected_fields = {
        "path",
        "artifact_sha256",
        "manifest_sha256",
        "byte_count",
        "mode_octal",
        "nlink",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != expected_fields:
        raise ValueError("balanced dataset task artifact identity is incomplete")
    if artifact["path"] != task["output"]:
        raise ValueError("balanced dataset task artifact path drifted")
    artifact_path = Path(artifact["path"])
    _immutable_file(artifact_path, "balanced grouped artifact")
    if (
        file_sha256(artifact_path, "balanced grouped artifact")
        != digest(artifact["artifact_sha256"], "artifact_sha256")
        or artifact_path.stat().st_size != artifact["byte_count"]
        or artifact["mode_octal"] != "0400"
        or artifact["nlink"] != 1
    ):
        raise ValueError("balanced grouped artifact identity changed after completion")
    return value, V5K1GroupedArtifactBinding(
        path=artifact_path,
        artifact_sha256=artifact["artifact_sha256"],
        manifest_sha256=artifact["manifest_sha256"],
    )


def _write_completion(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite balanced dataset collection completion")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    _immutable_file(path, "balanced dataset collection completion")


def collect_v5_k1_balanced_dataset(
    plan_path: str | os.PathLike[str],
    *,
    expected_plan_sha256: str,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Verify all tasks, derive actual populations, and publish completion last."""

    host = socket.gethostname() if hostname is None else hostname
    env = os.environ if environment is None else environment
    _worker_guard(host, env)
    path = lexical_no_symlinks(
        Path(plan_path), "K1 balanced dataset launch plan"
    ).resolve(strict=True)
    plan = _load_plan(path, expected_plan_sha256)
    input_before = replay_v5_k1_balanced_dataset_launch_inputs(
        plan, allowed_root=allowed_root
    )
    receipt_path = Path(plan["layout"]["train_tuning_disjointness_receipt"])
    completion_path = Path(plan["layout"]["dataset_completion"])
    if (
        receipt_path.exists()
        or receipt_path.is_symlink()
        or completion_path.exists()
        or completion_path.is_symlink()
    ):
        raise FileExistsError("refusing to reuse balanced dataset collection outputs")
    if not receipt_path.parent.is_dir() or completion_path.parent != receipt_path.parent:
        raise FileNotFoundError("balanced dataset audit directory must already exist")
    by_role = {"train": [], "tuning_validation": []}
    completion_shas = []
    for task in plan["tasks"]:
        completion, binding = _completion(
            task,
            plan_sha256=plan["plan_sha256"],
            input_identity=input_before,
        )
        by_role[task["role"]].append(binding)
        completion_shas.append(completion["completion_sha256"])
    balanced_sha = plan["balanced_dataset_plan"]["plan_sha256"]
    populations = tuple(
        population_from_v5_k1_grouped_artifacts(
            role=role,
            split_id=role,
            plan_sha256=balanced_sha,
            artifacts=tuple(by_role[role]),
            allowed_root=allowed_root,
        )
        for role in ("train", "tuning_validation")
    )
    expected_counts = plan["array"]["expected_clean_parent_counts"]
    observed_counts = {value.role: value.clean_parent_count for value in populations}
    if observed_counts != expected_counts:
        raise RuntimeError("actual train/tuning parent counts disagree with the launch plan")
    receipt = build_v5_k1_train_tuning_disjointness_receipt(populations)
    write_v5_k1_train_tuning_disjointness_receipt(receipt_path, receipt)
    receipt_file_sha = file_sha256(receipt_path, "train/tuning disjointness receipt")
    input_after = replay_v5_k1_balanced_dataset_launch_inputs(
        plan, allowed_root=allowed_root
    )
    if input_after != input_before:
        raise RuntimeError("balanced K1 immutable inputs changed during collection")
    core = {
        "schema": V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA,
        "version": V5_K1_BALANCED_DATASET_COLLECTION_VERSION,
        "status": "PASS",
        "scientific_acceptance_evidence": False,
        "training_authorization_granted": False,
        "phase_c_exclusion_proven": False,
        "plan_sha256": plan["plan_sha256"],
        "slurm_job_id": env["SLURM_JOB_ID"],
        "hostname": host,
        "task_count": len(plan["tasks"]),
        "task_completion_sha256s": completion_shas,
        "observed_clean_parent_counts": observed_counts,
        "populations": {
            value.role: {
                "recipe_set_sha256": value.recipe_set_sha256,
                "clean_group_set_sha256": value.clean_group_set_sha256,
                "clean_parent_count": value.clean_parent_count,
            }
            for value in populations
        },
        "train_tuning_disjointness_receipt": {
            "path": str(receipt_path),
            "file_sha256": receipt_file_sha,
            "receipt_sha256": receipt["receipt_sha256"],
            "train_tuning_claim_sha256": receipt["train_tuning_claim_sha256"],
            "mode_octal": "0400",
            "nlink": 1,
        },
        "immutable_input_identity_pre": input_before,
        "immutable_input_identity_post": input_after,
        "completion_written_after_receipt_seal": True,
    }
    result = {
        **core,
        "completion_sha256": sha256(canonical_json(core).encode()).hexdigest(),
    }
    _write_completion(completion_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = collect_v5_k1_balanced_dataset(
        args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA",
    "V5_K1_BALANCED_DATASET_COLLECTION_VERSION",
    "collect_v5_k1_balanced_dataset",
    "main",
]
