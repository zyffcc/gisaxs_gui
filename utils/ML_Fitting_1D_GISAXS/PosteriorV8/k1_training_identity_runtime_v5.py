"""Replay published K1 dataset receipts into an identity-only training gate."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Mapping

from .grouped_artifact_v5 import canonical_json
from .k1_balanced_dataset_collector_v5 import (
    V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA,
    V5_K1_BALANCED_DATASET_COLLECTION_VERSION,
)
from .k1_dataset_disjointness_v5 import (
    V5_K1_DATASET_POPULATION_ROLES,
    V5K1RecipePopulation,
    validate_v5_k1_dataset_disjointness_receipt,
    validate_v5_k1_train_tuning_disjointness_receipt,
)
from .k1_phase_c_holdout_collector_v5 import (
    V5_K1_PHASE_C_HOLDOUT_COLLECTION_SCHEMA,
    V5_K1_PHASE_C_HOLDOUT_COLLECTION_VERSION,
)
from .k1_staging_files_v5 import lexical_no_symlinks, read_regular_bytes
from .k1_training_identity_contract_v5 import (
    V5K1TrainingIdentityAuthorization,
    V5K1TrainingPopulationIdentity,
)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def _completion(
    payload: Mapping[str, object],
    *,
    schema: str,
    version: str,
    name: str,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be an object")
    value = dict(payload)
    supplied = _digest(value.pop("completion_sha256", None), f"{name}.completion_sha256")
    if supplied != sha256(canonical_json(value).encode("utf-8")).hexdigest():
        raise ValueError(f"{name} self hash does not reproduce")
    if (
        value.get("schema") != schema
        or value.get("version") != version
        or value.get("status") != "PASS"
        or value.get("scientific_acceptance_evidence") is not False
        or value.get("training_authorization_granted") is not False
        or value.get("completion_written_after_receipt_seal") is not True
        or value.get("immutable_input_identity_pre")
        != value.get("immutable_input_identity_post")
    ):
        raise ValueError(f"{name} completion contract is incomplete")
    return dict(payload)


def _population(value: V5K1RecipePopulation) -> V5K1TrainingPopulationIdentity:
    return V5K1TrainingPopulationIdentity(
        role=value.role,
        split_id=value.split_id,
        plan_sha256=value.plan_sha256,
        artifact_sha256s=value.artifact_sha256s,
        manifest_sha256s=value.manifest_sha256s,
        clean_parent_count=value.clean_parent_count,
        recipe_set_sha256=value.recipe_set_sha256,
        clean_group_set_sha256=value.clean_group_set_sha256,
    )


def issue_v5_k1_training_identity_authorization(
    *,
    source_archive_sha256: str,
    source_bundle_sha256: str,
    balanced_dataset_completion: Mapping[str, object],
    balanced_dataset_completion_file_sha256: str,
    train_tuning_receipt: Mapping[str, object],
    train_tuning_receipt_file_sha256: str,
    phase_c_completion: Mapping[str, object],
    phase_c_completion_file_sha256: str,
    three_way_receipt: Mapping[str, object],
    three_way_receipt_file_sha256: str,
) -> V5K1TrainingIdentityAuthorization:
    """Issue a non-gradient gate after all four published artifacts replay."""

    balanced = _completion(
        balanced_dataset_completion,
        schema=V5_K1_BALANCED_DATASET_COLLECTION_SCHEMA,
        version=V5_K1_BALANCED_DATASET_COLLECTION_VERSION,
        name="balanced_dataset_completion",
    )
    phase_c = _completion(
        phase_c_completion,
        schema=V5_K1_PHASE_C_HOLDOUT_COLLECTION_SCHEMA,
        version=V5_K1_PHASE_C_HOLDOUT_COLLECTION_VERSION,
        name="phase_c_completion",
    )
    train_tuning = validate_v5_k1_train_tuning_disjointness_receipt(
        train_tuning_receipt
    )
    three_way = validate_v5_k1_dataset_disjointness_receipt(three_way_receipt)
    balanced_file_sha = _digest(
        balanced_dataset_completion_file_sha256,
        "balanced_dataset_completion_file_sha256",
    )
    train_tuning_file_sha = _digest(
        train_tuning_receipt_file_sha256, "train_tuning_receipt_file_sha256"
    )
    phase_c_file_sha = _digest(
        phase_c_completion_file_sha256, "phase_c_completion_file_sha256"
    )
    three_way_file_sha = _digest(
        three_way_receipt_file_sha256, "three_way_receipt_file_sha256"
    )

    balanced_receipt = balanced.get("train_tuning_disjointness_receipt")
    phase_c_receipt = phase_c.get("three_way_disjointness_receipt")
    if not isinstance(balanced_receipt, Mapping) or not isinstance(
        phase_c_receipt, Mapping
    ):
        raise ValueError("collector completions do not bind their receipts")
    if (
        balanced_receipt.get("file_sha256") != train_tuning_file_sha
        or balanced_receipt.get("receipt_sha256") != train_tuning["receipt_sha256"]
        or balanced_receipt.get("train_tuning_claim_sha256")
        != train_tuning["train_tuning_claim_sha256"]
        or balanced_receipt.get("mode_octal") != "0400"
        or balanced_receipt.get("nlink") != 1
    ):
        raise ValueError("balanced completion receipt binding drifted")
    if (
        phase_c_receipt.get("file_sha256") != three_way_file_sha
        or phase_c_receipt.get("receipt_sha256") != three_way["receipt_sha256"]
        or phase_c_receipt.get("phase_c_exclusion_claim_sha256")
        != three_way["phase_c_exclusion_claim_sha256"]
        or phase_c_receipt.get("mode_octal") != "0400"
        or phase_c_receipt.get("nlink") != 1
    ):
        raise ValueError("Phase-C completion receipt binding drifted")
    if (
        phase_c.get("phase_c_exclusion_proven") is not True
        or balanced.get("phase_c_exclusion_proven") is not False
        or train_tuning["train_tuning_claim_sha256"]
        != three_way["train_tuning_claim_sha256"]
    ):
        raise ValueError("train/tune/Phase-C exclusion claims do not form one chain")

    for role in ("train", "tuning_validation"):
        if train_tuning["populations"][role] != three_way["populations"][role]:
            raise ValueError("three-way receipt changed the train/tuning population")
        completion_population = balanced["populations"].get(role)
        receipt_population = three_way["populations"][role]
        if completion_population != {
            "recipe_set_sha256": receipt_population["recipe_set_sha256"],
            "clean_group_set_sha256": receipt_population["clean_group_set_sha256"],
            "clean_parent_count": receipt_population["clean_parent_count"],
        }:
            raise ValueError("balanced completion population summary drifted")
    phase_c_population = three_way["populations"]["phase_c_holdout"]
    holdout_summary = phase_c.get("holdout_population")
    if not isinstance(holdout_summary, Mapping) or any(
        holdout_summary.get(name) != phase_c_population[name]
        for name in (
            "recipe_set_sha256",
            "clean_group_set_sha256",
            "clean_parent_count",
        )
    ):
        raise ValueError("Phase-C completion population summary drifted")
    if (
        three_way.get("all_recipe_sets_pairwise_disjoint") is not True
        or three_way.get("all_clean_group_sets_pairwise_disjoint") is not True
        or any(
            count != 0
            for pair in three_way["pairwise_intersection_counts"].values()
            for count in pair.values()
        )
    ):
        raise ValueError("three-way identity receipt is not disjoint")

    populations = tuple(
        _population(V5K1RecipePopulation.from_payload(three_way["populations"][role]))
        for role in V5_K1_DATASET_POPULATION_ROLES
    )
    return V5K1TrainingIdentityAuthorization(
        source_archive_sha256=_digest(source_archive_sha256, "source_archive_sha256"),
        source_bundle_sha256=_digest(source_bundle_sha256, "source_bundle_sha256"),
        balanced_dataset_completion_file_sha256=balanced_file_sha,
        balanced_dataset_completion_sha256=balanced["completion_sha256"],
        train_tuning_receipt_file_sha256=train_tuning_file_sha,
        train_tuning_receipt_sha256=train_tuning["receipt_sha256"],
        train_tuning_claim_sha256=train_tuning["train_tuning_claim_sha256"],
        phase_c_completion_file_sha256=phase_c_file_sha,
        phase_c_completion_sha256=phase_c["completion_sha256"],
        three_way_receipt_file_sha256=three_way_file_sha,
        three_way_receipt_sha256=three_way["receipt_sha256"],
        phase_c_exclusion_claim_sha256=three_way["phase_c_exclusion_claim_sha256"],
        populations=populations,
    )


def _read_immutable_json(path: Path, name: str, maximum_bytes: int) -> tuple[dict[str, object], str]:
    selected = lexical_no_symlinks(path, name).resolve(strict=True)
    metadata = selected.stat()
    if not selected.is_file() or stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise ValueError(f"{name} must be 0400/nlink1")
    encoded = read_regular_bytes(selected, name, maximum_bytes=maximum_bytes)
    try:
        payload = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one object")
    return payload, sha256(encoded).hexdigest()


def issue_v5_k1_training_identity_authorization_from_files(
    *,
    source_archive_sha256: str,
    source_bundle_sha256: str,
    balanced_dataset_completion_path: str | os.PathLike[str],
    train_tuning_receipt_path: str | os.PathLike[str],
    phase_c_completion_path: str | os.PathLike[str],
    three_way_receipt_path: str | os.PathLike[str],
) -> V5K1TrainingIdentityAuthorization:
    """Replay the four immutable publications and issue the identity gate."""

    balanced_path = lexical_no_symlinks(
        Path(balanced_dataset_completion_path), "balanced dataset completion"
    ).resolve(strict=True)
    train_tuning_path = lexical_no_symlinks(
        Path(train_tuning_receipt_path), "train/tuning receipt"
    ).resolve(strict=True)
    phase_c_path = lexical_no_symlinks(
        Path(phase_c_completion_path), "Phase-C completion"
    ).resolve(strict=True)
    three_way_path = lexical_no_symlinks(
        Path(three_way_receipt_path), "three-way receipt"
    ).resolve(strict=True)
    balanced, balanced_file_sha = _read_immutable_json(
        balanced_path,
        "balanced dataset completion",
        16 * 1024 * 1024,
    )
    train_tuning, train_tuning_file_sha = _read_immutable_json(
        train_tuning_path,
        "train/tuning receipt",
        32 * 1024 * 1024,
    )
    phase_c, phase_c_file_sha = _read_immutable_json(
        phase_c_path,
        "Phase-C completion",
        16 * 1024 * 1024,
    )
    three_way, three_way_file_sha = _read_immutable_json(
        three_way_path,
        "three-way receipt",
        64 * 1024 * 1024,
    )
    balanced_receipt = balanced.get("train_tuning_disjointness_receipt")
    phase_c_receipt = phase_c.get("three_way_disjointness_receipt")
    if not isinstance(balanced_receipt, Mapping) or not isinstance(
        phase_c_receipt, Mapping
    ):
        raise ValueError("collector completions do not bind their receipt paths")
    if Path(balanced_receipt.get("path", "")).resolve(strict=True) != train_tuning_path:
        raise ValueError("balanced completion points at another train/tuning receipt")
    if Path(phase_c_receipt.get("path", "")).resolve(strict=True) != three_way_path:
        raise ValueError("Phase-C completion points at another three-way receipt")
    return issue_v5_k1_training_identity_authorization(
        source_archive_sha256=source_archive_sha256,
        source_bundle_sha256=source_bundle_sha256,
        balanced_dataset_completion=balanced,
        balanced_dataset_completion_file_sha256=balanced_file_sha,
        train_tuning_receipt=train_tuning,
        train_tuning_receipt_file_sha256=train_tuning_file_sha,
        phase_c_completion=phase_c,
        phase_c_completion_file_sha256=phase_c_file_sha,
        three_way_receipt=three_way,
        three_way_receipt_file_sha256=three_way_file_sha,
    )


__all__ = [
    "issue_v5_k1_training_identity_authorization",
    "issue_v5_k1_training_identity_authorization_from_files",
]
