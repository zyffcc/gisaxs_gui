"""Build the all-K1 search plan from immutable dataset publications."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import stat
from typing import Mapping

from .compatibility_calibration import acquisition_policy_payload
from .formal_production_search_contract_v5 import V5FormalProductionSearchStage
from .grouped_artifact_v5 import canonical_json
from .grouped_dataset_v5 import observation_array, read_v5_grouped_dataset
from .k1_balanced_dataset_collector_v5 import (
    V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA,
    V5_K1_BALANCED_DATASET_COLLECTION_VERSION,
)
from .k1_balanced_dataset_launch_plan_v5 import (
    MAXWELL_DUST_ROOT,
    replay_v5_k1_balanced_dataset_launch_inputs,
    validate_v5_k1_balanced_dataset_launch_plan,
)
from .k1_balanced_dataset_worker_v5 import (
    V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA,
    V5_K1_BALANCED_DATASET_COMPLETION_VERSION,
)
from .k1_balanced_full_search_plan_v5 import (
    V5K1BalancedFullSearchParentBinding,
    build_v5_k1_balanced_full_search_plan,
)
from .k1_staging_files_v5 import file_sha256, lexical_no_symlinks, read_regular_bytes
from .k1_training_identity_runtime_v5 import (
    issue_v5_k1_training_identity_authorization_from_files,
)
from .simulation import NOISE_APPLICATION_VERSION


@dataclass(frozen=True, kw_only=True)
class V5K1BalancedFullSearchPlanConfig:
    """Explicit publication expectations for one new full-search run root."""

    source_archive_sha256: str
    source_bundle_sha256: str
    balanced_dataset_launch_plan_path: Path
    balanced_dataset_launch_plan_file_sha256: str
    balanced_dataset_launch_plan_sha256: str
    balanced_dataset_completion_path: Path
    balanced_dataset_completion_file_sha256: str
    balanced_dataset_completion_sha256: str
    train_tuning_receipt_path: Path
    train_tuning_receipt_file_sha256: str
    train_tuning_receipt_sha256: str
    train_tuning_claim_sha256: str
    phase_c_completion_path: Path
    phase_c_completion_file_sha256: str
    phase_c_completion_sha256: str
    three_way_receipt_path: Path
    three_way_receipt_file_sha256: str
    three_way_receipt_sha256: str
    phase_c_exclusion_claim_sha256: str
    k1_stage: V5FormalProductionSearchStage
    search_run_root: str


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _under_root(path: Path, allowed_root: Path, name: str) -> Path:
    selected = lexical_no_symlinks(path, name).resolve(strict=True)
    root = Path(allowed_root).resolve(strict=True)
    if selected == root or root not in selected.parents:
        raise ValueError(f"{name} must remain below {root}")
    return selected


def _sealed_regular(path: Path, name: str) -> tuple[Path, str, int]:
    selected = path.resolve(strict=True)
    metadata = selected.stat()
    if (
        not selected.is_file()
        or selected.is_symlink()
        or stat.S_IMODE(metadata.st_mode) != 0o400
        or metadata.st_nlink != 1
    ):
        raise ValueError(f"{name} must be a 0400/nlink1 regular file")
    return selected, file_sha256(selected, name), metadata.st_size


def _sealed_json(
    path: Path,
    name: str,
    *,
    allowed_root: Path,
    maximum_bytes: int,
) -> tuple[dict[str, object], Path, str]:
    selected = _under_root(path, allowed_root, name)
    _, file_sha, byte_count = _sealed_regular(selected, name)
    if byte_count > maximum_bytes:
        raise ValueError(f"{name} exceeds its maximum encoded size")
    encoded = read_regular_bytes(selected, name, maximum_bytes=maximum_bytes)
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return payload, selected, file_sha


def _self_hash(
    payload: Mapping[str, object], *, field: str, name: str
) -> str:
    core = dict(payload)
    supplied = _digest(core.pop(field, None), f"{name}.{field}")
    if supplied != sha256(canonical_json(core).encode("utf-8")).hexdigest():
        raise ValueError(f"{name} self SHA-256 does not reproduce")
    return supplied


def _expected_identity(config: V5K1BalancedFullSearchPlanConfig) -> dict[str, str]:
    names = (
        "source_archive_sha256",
        "source_bundle_sha256",
        "balanced_dataset_launch_plan_file_sha256",
        "balanced_dataset_launch_plan_sha256",
        "balanced_dataset_completion_file_sha256",
        "balanced_dataset_completion_sha256",
        "train_tuning_receipt_file_sha256",
        "train_tuning_receipt_sha256",
        "train_tuning_claim_sha256",
        "phase_c_completion_file_sha256",
        "phase_c_completion_sha256",
        "three_way_receipt_file_sha256",
        "three_way_receipt_sha256",
        "phase_c_exclusion_claim_sha256",
    )
    return {name: _digest(getattr(config, name), name) for name in names}


def _verify_parent_noise_policy(path: Path, artifact: Mapping[str, object]) -> None:
    """Reject incompatible reconstruction before authoring the batch plan.

    This worker-side check does not replace the projection's full policy and
    array equality checks, nor imply compatibility with a calibration artifact.
    """
    parent, receipt = read_v5_grouped_dataset(path)
    if (
        receipt.artifact_sha256 != artifact["artifact_sha256"]
        or receipt.manifest_sha256 != artifact["manifest_sha256"]
        or receipt.byte_count != artifact["byte_count"]
    ):
        raise ValueError("balanced grouped parent semantic receipt drifted")
    policies = parent.arrays[observation_array("acquisition_policy_id")]
    if not len(policies):
        raise ValueError("balanced grouped parent has no observation policies")
    for policy_id in set(map(str, policies)):
        policy = acquisition_policy_payload(policy_id)
        noise_version = policy["sigma"]["sigma_log_source"].split("|", 1)[0]
        if noise_version != NOISE_APPLICATION_VERSION:
            raise ValueError(
                "balanced grouped parent noise policy is incompatible with current "
                "reconstruction; preserve the historical artifact and use a "
                "version-matched dataset before authoring full search"
            )


def _task_binding(
    task: Mapping[str, object],
    *,
    plan_sha256: str,
    launch_input_identity: Mapping[str, object],
    allowed_root: Path,
) -> tuple[V5K1BalancedFullSearchParentBinding, str]:
    completion, completion_path, completion_file_sha = _sealed_json(
        Path(task["completion"]),
        "balanced dataset task completion",
        allowed_root=allowed_root,
        maximum_bytes=4 * 1024 * 1024,
    )
    completion_sha = _self_hash(
        completion,
        field="completion_sha256",
        name="balanced dataset task completion",
    )
    if (
        completion.get("schema") != V5_K1_BALANCED_DATASET_COMPLETION_SCHEMA
        or completion.get("version") != V5_K1_BALANCED_DATASET_COMPLETION_VERSION
        or completion.get("status") != "PASS"
        or completion.get("scientific_acceptance_evidence") is not False
        or completion.get("training_authorization_granted") is not False
        or completion.get("plan_sha256") != plan_sha256
        or completion.get("array_task_id") != task["array_task_id"]
        or completion.get("task") != dict(task)
        or completion.get("immutable_input_identity_pre") != launch_input_identity
        or completion.get("immutable_input_identity_post") != launch_input_identity
        or completion.get("completion_written_after_artifact_seal") is not True
    ):
        raise ValueError("balanced dataset task completion contract drifted")
    artifact = completion.get("artifact")
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "path",
        "artifact_sha256",
        "manifest_sha256",
        "byte_count",
        "mode_octal",
        "nlink",
    }:
        raise ValueError("balanced dataset task artifact identity is incomplete")
    artifact_path = _under_root(
        Path(artifact["path"]), allowed_root, "balanced grouped parent"
    )
    if str(artifact_path) != task["output"]:
        raise ValueError("balanced grouped parent path escaped its task")
    _, artifact_file_sha, artifact_byte_count = _sealed_regular(
        artifact_path, "balanced grouped parent"
    )
    if (
        artifact_file_sha != _digest(artifact["artifact_sha256"], "artifact_sha256")
        or artifact_byte_count != artifact["byte_count"]
        or artifact["mode_octal"] != "0400"
        or artifact["nlink"] != 1
    ):
        raise ValueError("balanced grouped parent changed after completion")
    _verify_parent_noise_policy(artifact_path, artifact)
    selection_sha = _digest(completion.get("selection_sha256"), "selection_sha256")
    return (
        V5K1BalancedFullSearchParentBinding(
            array_task_id=task["array_task_id"],
            role=task["role"],
            split_id=task["split_id"],
            branch_id=task["branch_id"],
            branch_ordinal=task["branch_ordinal"],
            balanced_sobol_block_sha256=task["balanced_sobol_block_sha256"],
            shard_index=task["shard_index"],
            split_offset=task["split_offset"],
            recipe_count=task["recipe_count"],
            selection_sha256=selection_sha,
            parent_path=str(artifact_path),
            artifact_sha256=artifact["artifact_sha256"],
            manifest_sha256=artifact["manifest_sha256"],
            byte_count=artifact_byte_count,
            task_completion_path=str(completion_path),
            task_completion_file_sha256=completion_file_sha,
            task_completion_sha256=completion_sha,
        ),
        completion_sha,
    )


def build_v5_k1_balanced_full_search_plan_from_publications(
    config: V5K1BalancedFullSearchPlanConfig,
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    """Replay every dataset publication and derive the 60-task search plan."""

    if not isinstance(config, V5K1BalancedFullSearchPlanConfig):
        raise TypeError("config must be V5K1BalancedFullSearchPlanConfig")
    allowed_root = Path(allowed_root)
    expected = _expected_identity(config)
    launch_payload, launch_path, launch_file_sha = _sealed_json(
        config.balanced_dataset_launch_plan_path,
        "balanced dataset launch plan",
        allowed_root=allowed_root,
        maximum_bytes=16 * 1024 * 1024,
    )
    launch = validate_v5_k1_balanced_dataset_launch_plan(launch_payload)
    if (
        str(launch_path) != launch["layout"]["plan"]
        or launch_file_sha
        != expected["balanced_dataset_launch_plan_file_sha256"]
        or launch["plan_sha256"]
        != expected["balanced_dataset_launch_plan_sha256"]
    ):
        raise ValueError("balanced dataset launch-plan identity drifted")
    launch_inputs = replay_v5_k1_balanced_dataset_launch_inputs(
        launch, allowed_root=allowed_root
    )
    balanced, balanced_path, balanced_file_sha = _sealed_json(
        config.balanced_dataset_completion_path,
        "balanced dataset completion",
        allowed_root=allowed_root,
        maximum_bytes=16 * 1024 * 1024,
    )
    balanced_sha = _self_hash(
        balanced,
        field="completion_sha256",
        name="balanced dataset completion",
    )
    if (
        str(balanced_path) != launch["layout"]["dataset_completion"]
        or balanced_file_sha
        != expected["balanced_dataset_completion_file_sha256"]
        or balanced_sha != expected["balanced_dataset_completion_sha256"]
        or balanced.get("schema") != V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA
        or balanced.get("version") != V5_K1_BALANCED_DATASET_COLLECTION_VERSION
        or balanced.get("status") != "PASS"
        or balanced.get("plan_sha256") != launch["plan_sha256"]
        or balanced.get("task_count") != len(launch["tasks"])
        or balanced.get("immutable_input_identity_pre") != launch_inputs
        or balanced.get("immutable_input_identity_post") != launch_inputs
        or balanced.get("completion_written_after_receipt_seal") is not True
    ):
        raise ValueError("balanced dataset collection completion drifted")
    authorization = issue_v5_k1_training_identity_authorization_from_files(
        source_archive_sha256=expected["source_archive_sha256"],
        source_bundle_sha256=expected["source_bundle_sha256"],
        balanced_dataset_completion_path=balanced_path,
        train_tuning_receipt_path=config.train_tuning_receipt_path,
        phase_c_completion_path=config.phase_c_completion_path,
        three_way_receipt_path=config.three_way_receipt_path,
    )
    expected_authorization = {
        name: expected[name]
        for name in (
            "source_archive_sha256",
            "source_bundle_sha256",
            "balanced_dataset_completion_file_sha256",
            "balanced_dataset_completion_sha256",
            "train_tuning_receipt_file_sha256",
            "train_tuning_receipt_sha256",
            "train_tuning_claim_sha256",
            "phase_c_completion_file_sha256",
            "phase_c_completion_sha256",
            "three_way_receipt_file_sha256",
            "three_way_receipt_sha256",
            "phase_c_exclusion_claim_sha256",
        )
    }
    if any(
        getattr(authorization, name) != value
        for name, value in expected_authorization.items()
    ):
        raise ValueError("K1 training population authorization differs from expectations")
    rows = tuple(
        _task_binding(
            task,
            plan_sha256=launch["plan_sha256"],
            launch_input_identity=launch_inputs,
            allowed_root=allowed_root,
        )
        for task in launch["tasks"]
    )
    parents = tuple(value[0] for value in rows)
    completion_shas = [value[1] for value in rows]
    if (
        balanced.get("task_completion_sha256s") != completion_shas
        or balanced.get("observed_clean_parent_counts")
        != launch["array"]["expected_clean_parent_counts"]
    ):
        raise ValueError("balanced dataset completion lost its exact task inventory")
    return build_v5_k1_balanced_full_search_plan(
        source_archive_sha256=expected["source_archive_sha256"],
        source_bundle_sha256=expected["source_bundle_sha256"],
        balanced_dataset_launch_plan_file_sha256=launch_file_sha,
        balanced_dataset_launch_plan_sha256=launch["plan_sha256"],
        balanced_dataset_completion_file_sha256=balanced_file_sha,
        balanced_dataset_completion_sha256=balanced_sha,
        identity_authorization=authorization,
        k1_stage=config.k1_stage,
        search_run_root=config.search_run_root,
        parents=parents,
    )


__all__ = [
    "V5K1BalancedFullSearchPlanConfig",
    "build_v5_k1_balanced_full_search_plan_from_publications",
]
