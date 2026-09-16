"""Collect Phase-C holdout shards and prove train/tune/holdout disjointness."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import socket
import stat
from typing import Mapping, Sequence

from .grouped_artifact_v5 import canonical_json
from .k1_dataset_disjointness_v5 import (
    V5K1RecipePopulation,
    build_v5_k1_dataset_disjointness_receipt,
    validate_v5_k1_train_tuning_disjointness_receipt,
    write_v5_k1_dataset_disjointness_receipt,
)
from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCHES,
    K1_PHASE_C_OBSERVATION_STRESS_STRATA,
    K1_PHASE_C_RANGE_STRESS_STRATA,
    K1_PHASE_C_SPLIT_ID,
)
from .k1_phase_c_holdout_launch_plan_v5 import (
    MAXWELL_DUST_ROOT,
    replay_v5_k1_phase_c_holdout_launch_inputs,
    validate_v5_k1_phase_c_holdout_launch_plan,
)
from .k1_phase_c_holdout_shard_v5 import (
    validate_v5_k1_phase_c_holdout_shard_payload,
)
from .k1_phase_c_holdout_worker_v5 import (
    V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_SCHEMA,
    V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_VERSION,
)
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks, read_regular_bytes
from .k1_training_chain_contract_v5 import digest


V5_K1_PHASE_C_HOLDOUT_COLLECTION_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_c_holdout_collection/v1"
)
V5_K1_PHASE_C_HOLDOUT_COLLECTION_VERSION = (
    "posterior_v8_v5_2_actual_all12_identity_three_way_disjoint_completion_last_v1"
)


def _load_json(path: Path, name: str, maximum_bytes: int) -> dict[str, object]:
    encoded = read_regular_bytes(path, name, maximum_bytes=maximum_bytes)
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one object")
    return value


def _immutable_file(path: Path, name: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a real regular file")
    metadata = path.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError(f"{name} must be 0400/nlink1")


def _completion(
    task: Mapping[str, object],
    *,
    plan_sha256: str,
    input_identity: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    completion_path = Path(task["completion"])
    _immutable_file(completion_path, "Phase-C task completion")
    completion = _load_json(
        completion_path, "Phase-C task completion", 4 * 1024 * 1024
    )
    core = dict(completion)
    supplied = digest(core.pop("completion_sha256", None), "completion_sha256")
    if supplied != sha256(canonical_json(core).encode()).hexdigest():
        raise ValueError("Phase-C task completion self-hash does not reproduce")
    if (
        completion.get("schema") != V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_SCHEMA
        or completion.get("version") != V5_K1_PHASE_C_HOLDOUT_TASK_COMPLETION_VERSION
        or completion.get("status") != "PASS"
        or completion.get("plan_sha256") != plan_sha256
        or completion.get("array_task_id") != task["array_task_id"]
        or completion.get("task") != dict(task)
        or completion.get("immutable_input_identity_pre") != input_identity
        or completion.get("immutable_input_identity_post") != input_identity
        or completion.get("completion_written_after_artifact_seal") is not True
        or completion.get("scientific_acceptance_evidence") is not False
        or completion.get("training_authorization_granted") is not False
    ):
        raise ValueError("Phase-C task completion contract drifted")
    artifact_identity = completion.get("artifact")
    artifact_path = Path(task["output"])
    if not isinstance(artifact_identity, Mapping) or artifact_identity.get(
        "path"
    ) != str(artifact_path):
        raise ValueError("Phase-C task artifact identity is incomplete")
    _immutable_file(artifact_path, "Phase-C holdout shard")
    metadata = artifact_path.stat()
    if (
        file_sha256(artifact_path, "Phase-C holdout shard")
        != digest(artifact_identity.get("file_sha256"), "artifact file SHA-256")
        or artifact_identity.get("byte_count") != metadata.st_size
        or artifact_identity.get("mode_octal") != "0400"
        or artifact_identity.get("nlink") != 1
    ):
        raise ValueError("Phase-C holdout shard changed after completion")
    artifact = validate_v5_k1_phase_c_holdout_shard_payload(
        _load_json(artifact_path, "Phase-C holdout shard", 256 * 1024 * 1024)
    )
    manifest = artifact["manifest"]
    if (
        artifact["artifact_self_sha256"]
        != artifact_identity.get("artifact_self_sha256")
        or manifest["manifest_sha256"] != artifact_identity.get("manifest_sha256")
        or manifest["phase_c_sobol_block_sha256"]
        != task["phase_c_sobol_block_sha256"]
        or manifest["branch_id"] != task["branch_id"]
        or manifest["branch_ordinal"] != task["branch_ordinal"]
        or manifest["selection_sha256"] != completion["selection_sha256"]
        or manifest["recipe_count"] != task["recipe_count"]
        or manifest["selected_sobol_indices"]
        != list(
            range(
                task["split_offset"],
                task["split_offset"] + task["recipe_count"],
            )
        )
    ):
        raise ValueError("Phase-C task artifact identity/window drifted")
    if completion_path.stat().st_mtime_ns < artifact_path.stat().st_mtime_ns:
        raise ValueError("Phase-C task completion was not written last")
    return completion, artifact


def _write_completion(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError("refusing to overwrite Phase-C collection completion")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o400)
    _immutable_file(path, "Phase-C collection completion")


def collect_v5_k1_phase_c_holdout(
    plan_path: str | os.PathLike[str],
    *,
    expected_plan_sha256: str,
    allowed_root: Path = MAXWELL_DUST_ROOT,
    hostname: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    host = socket.gethostname() if hostname is None else hostname
    env = os.environ if environment is None else environment
    if host.split(".", 1)[0].startswith(("max-wgs", "max-fs-display")):
        raise RuntimeError("Phase-C holdout collection is forbidden on Maxwell login nodes")
    if not env.get("SLURM_JOB_ID", "").isdigit():
        raise RuntimeError("Phase-C holdout collection requires a Slurm worker")
    path = lexical_no_symlinks(Path(plan_path), "Phase-C holdout launch plan").resolve(
        strict=True
    )
    _immutable_file(path, "Phase-C holdout launch plan")
    plan = validate_v5_k1_phase_c_holdout_launch_plan(
        _load_json(path, "Phase-C holdout launch plan", 16 * 1024 * 1024)
    )
    if plan["plan_sha256"] != digest(expected_plan_sha256, "expected_plan_sha256"):
        raise ValueError("Phase-C launch plan differs from the Slurm export")
    input_before = replay_v5_k1_phase_c_holdout_launch_inputs(
        plan, allowed_root=allowed_root
    )
    receipt_path = Path(plan["layout"]["three_way_disjointness_receipt"])
    completion_path = Path(plan["layout"]["holdout_completion"])
    if any(value.exists() or value.is_symlink() for value in (receipt_path, completion_path)):
        raise FileExistsError("refusing to reuse Phase-C collection outputs")
    if not receipt_path.parent.is_dir() or completion_path.parent != receipt_path.parent:
        raise FileNotFoundError("Phase-C audit directory must exist")
    recipe_sha256s = []
    clean_group_ids = []
    artifact_sha256s = []
    manifest_sha256s = []
    completion_sha256s = []
    branch_counts: Counter[str] = Counter()
    stress_cells: dict[str, Counter[tuple[str, str]]] = {
        branch.branch_id: Counter() for branch in K1_PHASE_C_BRANCHES
    }
    for task in plan["tasks"]:
        completion, artifact = _completion(
            task,
            plan_sha256=plan["plan_sha256"],
            input_identity=input_before,
        )
        manifest = artifact["manifest"]
        recipe_sha256s.extend(manifest["recipe_sha256s"])
        clean_group_ids.extend(manifest["clean_group_ids"])
        artifact_sha256s.append(completion["artifact"]["file_sha256"])
        manifest_sha256s.append(manifest["manifest_sha256"])
        completion_sha256s.append(completion["completion_sha256"])
        branch_counts[manifest["branch_id"]] += manifest["recipe_count"]
        for range_name, observation_name, count in manifest["stress_cell_counts"]:
            stress_cells[manifest["branch_id"]][(range_name, observation_name)] += count
    expected_per_branch = plan["configuration"]["parents_per_branch"]
    expected_branches = {value.branch_id: expected_per_branch for value in K1_PHASE_C_BRANCHES}
    if dict(branch_counts) != expected_branches:
        raise RuntimeError("Phase-C holdout branch balance failed")
    expected_cells = {
        (range_name, observation_name)
        for range_name in K1_PHASE_C_RANGE_STRESS_STRATA
        for observation_name in K1_PHASE_C_OBSERVATION_STRESS_STRATA
    }
    for branch_id, counts in stress_cells.items():
        if set(counts) != expected_cells or max(counts.values()) - min(counts.values()) > 1:
            raise RuntimeError(f"Phase-C stress cells are not balanced for {branch_id}")
    holdout = V5K1RecipePopulation(
        role="phase_c_holdout",
        split_id=K1_PHASE_C_SPLIT_ID,
        plan_sha256=plan["phase_c_plan"]["plan_sha256"],
        artifact_sha256s=tuple(artifact_sha256s),
        manifest_sha256s=tuple(manifest_sha256s),
        recipe_sha256s=tuple(recipe_sha256s),
        clean_group_ids=tuple(clean_group_ids),
    )
    if holdout.clean_parent_count != plan["array"]["expected_clean_parent_count"]:
        raise RuntimeError("Phase-C holdout parent count drifted")
    train_tuning = validate_v5_k1_train_tuning_disjointness_receipt(
        _load_json(
            Path(plan["train_tuning_receipt"]["path"]),
            "train/tuning disjointness receipt",
            16 * 1024 * 1024,
        )
    )
    if (
        train_tuning["receipt_sha256"]
        != plan["train_tuning_receipt"]["receipt_sha256"]
        or train_tuning["train_tuning_claim_sha256"]
        != plan["train_tuning_receipt"]["train_tuning_claim_sha256"]
    ):
        raise RuntimeError("train/tuning receipt identity drifted during collection")
    populations = tuple(
        V5K1RecipePopulation.from_payload(train_tuning["populations"][role])
        for role in ("train", "tuning_validation")
    ) + (holdout,)
    receipt = build_v5_k1_dataset_disjointness_receipt(populations)
    write_v5_k1_dataset_disjointness_receipt(receipt_path, receipt)
    input_after = replay_v5_k1_phase_c_holdout_launch_inputs(
        plan, allowed_root=allowed_root
    )
    if input_after != input_before:
        raise RuntimeError("Phase-C immutable inputs changed during collection")
    core = {
        "schema": V5_K1_PHASE_C_HOLDOUT_COLLECTION_SCHEMA,
        "version": V5_K1_PHASE_C_HOLDOUT_COLLECTION_VERSION,
        "status": "PASS",
        "scientific_acceptance_evidence": False,
        "training_authorization_granted": False,
        "phase_c_exclusion_proven": True,
        "plan_sha256": plan["plan_sha256"],
        "slurm_job_id": env["SLURM_JOB_ID"],
        "hostname": host,
        "task_count": len(plan["tasks"]),
        "task_completion_sha256s": completion_sha256s,
        "holdout_population": {
            "clean_parent_count": holdout.clean_parent_count,
            "recipe_set_sha256": holdout.recipe_set_sha256,
            "clean_group_set_sha256": holdout.clean_group_set_sha256,
            "branch_counts": dict(branch_counts),
            "all_25_stress_cells_per_branch": True,
            "stress_cell_max_minus_min_lte": 1,
        },
        "three_way_disjointness_receipt": {
            "path": str(receipt_path),
            "file_sha256": file_sha256(receipt_path, "three-way disjointness receipt"),
            "receipt_sha256": receipt["receipt_sha256"],
            "phase_c_exclusion_claim_sha256": receipt[
                "phase_c_exclusion_claim_sha256"
            ],
            "mode_octal": "0400",
            "nlink": 1,
        },
        "immutable_input_identity_pre": input_before,
        "immutable_input_identity_post": input_after,
        "completion_written_after_receipt_seal": True,
    }
    result = {**core, "completion_sha256": sha256(canonical_json(core).encode()).hexdigest()}
    _write_completion(completion_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = collect_v5_k1_phase_c_holdout(
        args.plan, expected_plan_sha256=args.expected_plan_sha256
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "V5_K1_PHASE_C_HOLDOUT_COLLECTION_SCHEMA",
    "V5_K1_PHASE_C_HOLDOUT_COLLECTION_VERSION",
    "collect_v5_k1_phase_c_holdout",
    "main",
]
