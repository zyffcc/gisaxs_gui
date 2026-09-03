"""Fail-closed input contract for balanced twelve-branch K=1 training.

This contract sits between checked grouped artifacts and the existing V5.2
trainer.  It does not promote pilot search labels, select a paper checkpoint,
or reuse the Phase-A single-branch weights as evidence of K1 coverage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
    V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
)
from .formal_production_search_plan_v5 import (
    V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED,
)
from .grouped_dataset_v5 import V5_GROUPED_DATASET_SCHEMA, V5_GROUPED_DATASET_VERSION
from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCHES,
    K1_PHASE_C_SPLIT_ID,
    v5_k1_phase_c_contract_payload,
)
from .k1_training_chain_dataset_audit_v5 import V5_K1_PARENT_SET_HASH_SEMANTICS
from .model_v5_contract import model_v5_contract_payload
from .search_evidence_receipt_v5 import (
    V5_SEARCH_TRAINING_RECEIPT_PROMOTION_ENABLED,
)
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_COORDINATE_VERSION,
    V5_SOBOL_RECIPE_DIM,
)


V5_K1_TRAINING_INVENTORY_SCHEMA = "gisaxs.posterior_v8.k1_training_input_inventory/v1"
V5_K1_TRAINING_INVENTORY_VERSION = (
    "posterior_v8_v5_2_balanced_all12_train_tuning_input_inventory_v1"
)
V5_K1_TRAINING_CHAIN_SCHEMA = "gisaxs.posterior_v8.k1_training_tuning_chain/v1"
V5_K1_TRAINING_CHAIN_VERSION = "posterior_v8_v5_2_all12_seed_array_training_tuning_handoff_v1"
V5_K1_TRAINING_MODES = ("engineering_e1", "formal_multiseed")
V5_K1_FORMAL_MINIMUM_MODEL_SEEDS = 5
V5_K1_TUNING_SUMMARY_RUNTIME_AVAILABLE = False
V5_K1_FORMAL_TRAINING_AUTHORIZATION_ADAPTER_AVAILABLE = False
V5_K1_TRAIN_SPLIT_ID = "train"
V5_K1_TUNING_SPLIT_ID = "tuning_validation"

_SHA256_LENGTH = 64
_ARTIFACT_FIELDS = frozenset(
    {
        "path",
        "role",
        "split_id",
        "artifact_sha256",
        "manifest_sha256",
        "clean_parent_count",
        "branch_counts",
        "sidecar_path",
        "sidecar_artifact_sha256",
        "sidecar_manifest_sha256",
        "evidence_receipt_path",
        "evidence_receipt_sha256",
        "full_training_eligible",
    }
)


def canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty stripped string")
    return value


def _optional_digest(value: object, name: str) -> str | None:
    return None if value is None else digest(value, name)


def _model_contract_sha256() -> str:
    return sha256(canonical_json(model_v5_contract_payload()).encode("utf-8")).hexdigest()


def v5_k1_training_runtime_capabilities() -> dict[str, object]:
    search_receipts = V5_SEARCH_TRAINING_RECEIPT_PROMOTION_ENABLED is True
    formal_membership = V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED is True
    full = search_receipts and formal_membership
    authorization_adapter = V5_K1_FORMAL_TRAINING_AUTHORIZATION_ADAPTER_AVAILABLE
    return {
        "search_training_receipt_promotion_enabled": search_receipts,
        "formal_production_training_promotion_enabled": formal_membership,
        "full_search_supervision_consumable_by_grouped_trainer": full,
        "formal_training_authorization_adapter_available": authorization_adapter,
        "tuning_exact_budget_summary_runtime_available": (V5_K1_TUNING_SUMMARY_RUNTIME_AVAILABLE),
        "formal_chain_submission_ready": bool(
            full and authorization_adapter and V5_K1_TUNING_SUMMARY_RUNTIME_AVAILABLE
        ),
        "blocked_interfaces": [
            *(
                []
                if search_receipts
                else ["search_evidence_receipt_v5 rejects model_development_full_training"]
            ),
            *(
                []
                if formal_membership
                else ["formal_production_search_plan_v5 disables training promotion"]
            ),
            *(
                []
                if V5_K1_TUNING_SUMMARY_RUNTIME_AVAILABLE
                else ["no runtime produces V5TuningCheckpointEvaluation exact-budget summaries"]
            ),
            *(
                []
                if authorization_adapter
                else [
                    "K1 chain has not bound the formal training authorization identity"
                ]
            ),
        ],
    }


@dataclass(frozen=True, kw_only=True)
class V5K1TrainingArtifact:
    """One checked grouped shard and its optional completed-search overlay."""

    path: str
    role: str
    split_id: str
    artifact_sha256: str
    manifest_sha256: str
    clean_parent_count: int
    branch_counts: tuple[tuple[str, int], ...]
    sidecar_path: str | None = None
    sidecar_artifact_sha256: str | None = None
    sidecar_manifest_sha256: str | None = None
    evidence_receipt_path: str | None = None
    evidence_receipt_sha256: str | None = None
    full_training_eligible: bool = False

    def __post_init__(self) -> None:
        path = _text(self.path, "path")
        role = _text(self.role, "role")
        expected_split = {
            "train": V5_K1_TRAIN_SPLIT_ID,
            "tuning_validation": V5_K1_TUNING_SPLIT_ID,
        }.get(role)
        if expected_split is None or self.split_id != expected_split:
            raise ValueError("artifact role and split_id must be train or tuning_validation")
        digest(self.artifact_sha256, "artifact_sha256")
        digest(self.manifest_sha256, "manifest_sha256")
        parent_count = positive_integer(self.clean_parent_count, "clean_parent_count")
        counts = tuple(self.branch_counts)
        branch_ids = tuple(value.branch_id for value in K1_PHASE_C_BRANCHES)
        if tuple(name for name, _ in counts) != branch_ids:
            raise ValueError("branch_counts must contain the canonical twelve K1 branches")
        checked_counts = []
        for name, count in counts:
            if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral):
                raise TypeError(f"branch_counts[{name}] must be an integer")
            if int(count) < 0:
                raise ValueError(f"branch_counts[{name}] must be non-negative")
            checked_counts.append((name, int(count)))
        if sum(count for _, count in checked_counts) != parent_count:
            raise ValueError("branch_counts do not sum to clean_parent_count")
        overlay_values = (
            self.sidecar_path,
            self.sidecar_artifact_sha256,
            self.sidecar_manifest_sha256,
            self.evidence_receipt_path,
            self.evidence_receipt_sha256,
        )
        if any(value is None for value in overlay_values) and any(
            value is not None for value in overlay_values
        ):
            raise ValueError("sidecar and evidence identities must be all supplied or all absent")
        if overlay_values[0] is not None:
            _text(self.sidecar_path, "sidecar_path")
            digest(self.sidecar_artifact_sha256, "sidecar_artifact_sha256")
            digest(self.sidecar_manifest_sha256, "sidecar_manifest_sha256")
            _text(self.evidence_receipt_path, "evidence_receipt_path")
            digest(self.evidence_receipt_sha256, "evidence_receipt_sha256")
        if type(self.full_training_eligible) is not bool:
            raise TypeError("full_training_eligible must be a bool")
        if self.full_training_eligible and overlay_values[0] is None:
            raise ValueError("full-training eligibility requires sidecar and evidence identities")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "clean_parent_count", parent_count)
        object.__setattr__(self, "branch_counts", tuple(checked_counts))

    def audit_payload(self) -> dict[str, object]:
        value = asdict(self)
        value["branch_counts"] = dict(self.branch_counts)
        return value

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "V5K1TrainingArtifact":
        if not isinstance(payload, Mapping) or frozenset(payload) != _ARTIFACT_FIELDS:
            raise ValueError("training artifact fields are incomplete or unsupported")
        values = dict(payload)
        raw_counts = values.pop("branch_counts")
        if not isinstance(raw_counts, Mapping):
            raise ValueError("branch_counts must be an object")
        branch_ids = tuple(value.branch_id for value in K1_PHASE_C_BRANCHES)
        if set(raw_counts) != set(branch_ids):
            raise ValueError("branch_counts must contain exactly twelve K1 branches")
        values["branch_counts"] = tuple((name, raw_counts[name]) for name in branch_ids)
        return cls(**values)


def _aggregate_branch_counts(
    artifacts: Sequence[V5K1TrainingArtifact],
) -> dict[str, int]:
    return {
        branch.branch_id: sum(dict(value.branch_counts)[branch.branch_id] for value in artifacts)
        for branch in K1_PHASE_C_BRANCHES
    }


def _validate_balanced_split(
    artifacts: tuple[V5K1TrainingArtifact, ...], role: str
) -> dict[str, object]:
    if not artifacts or any(value.role != role for value in artifacts):
        raise ValueError(f"{role} artifacts must be non-empty and role-pure")
    counts = _aggregate_branch_counts(artifacts)
    if any(value < 1 for value in counts.values()):
        raise ValueError(f"{role} does not cover every legal K1 branch")
    if max(counts.values()) - min(counts.values()) > 1:
        raise ValueError(f"{role} K1 generating branches are not balanced")
    return {
        "split_id": artifacts[0].split_id,
        "clean_parent_count": sum(value.clean_parent_count for value in artifacts),
        "branch_counts": counts,
        "all_twelve_generating_branches_present": True,
        "branch_count_max_minus_min_lte": 1,
    }


def _artifact_bundle_sha256(artifacts: Sequence[V5K1TrainingArtifact]) -> str:
    values = [
        {
            "role": value.role,
            "path": value.path,
            "artifact_sha256": value.artifact_sha256,
            "manifest_sha256": value.manifest_sha256,
            "sidecar_artifact_sha256": value.sidecar_artifact_sha256,
            "sidecar_manifest_sha256": value.sidecar_manifest_sha256,
            "evidence_receipt_sha256": value.evidence_receipt_sha256,
        }
        for value in artifacts
    ]
    return sha256(canonical_json(values).encode("utf-8")).hexdigest()


def build_v5_k1_training_inventory(
    *,
    source_archive_sha256: str,
    source_bundle_sha256: str,
    train_artifacts: Sequence[V5K1TrainingArtifact],
    tuning_artifacts: Sequence[V5K1TrainingArtifact],
    train_parent_set_sha256: str,
    tuning_parent_set_sha256: str,
    train_tuning_disjointness_receipt_sha256: str,
    k1_phase_c_disjointness_receipt_sha256: str,
    phase_a_result_payload_sha256: str | None = None,
) -> dict[str, object]:
    """Build a self-verifying inventory; callers must derive counts from artifacts."""

    train = tuple(train_artifacts)
    tuning = tuple(tuning_artifacts)
    if not all(isinstance(value, V5K1TrainingArtifact) for value in (*train, *tuning)):
        raise TypeError("artifact sequences contain invalid values")
    paths = tuple(value.path for value in (*train, *tuning))
    overlay_paths = tuple(
        path
        for value in (*train, *tuning)
        for path in (value.sidecar_path, value.evidence_receipt_path)
        if path is not None
    )
    if len(set((*paths, *overlay_paths))) != len(paths) + len(overlay_paths):
        raise ValueError("dataset, sidecar, and evidence paths must be globally unique")
    train_summary = _validate_balanced_split(train, "train")
    tuning_summary = _validate_balanced_split(tuning, "tuning_validation")
    train_parent_sha = digest(train_parent_set_sha256, "train_parent_set_sha256")
    tuning_parent_sha = digest(tuning_parent_set_sha256, "tuning_parent_set_sha256")
    if train_parent_sha == tuning_parent_sha:
        raise ValueError("train and tuning parent-set identities must differ")
    phase_c = v5_k1_phase_c_contract_payload()
    artifacts = (*train, *tuning)
    full_eligible = all(value.full_training_eligible for value in artifacts)
    core = {
        "schema": V5_K1_TRAINING_INVENTORY_SCHEMA,
        "version": V5_K1_TRAINING_INVENTORY_VERSION,
        "scientific_role": "balanced_k1_train_and_tuning_inputs_not_model_acceptance",
        "source_archive_sha256": digest(source_archive_sha256, "source_archive_sha256"),
        "source_bundle_sha256": digest(source_bundle_sha256, "source_bundle_sha256"),
        "dataset_contract": {
            "schema": V5_GROUPED_DATASET_SCHEMA,
            "version": V5_GROUPED_DATASET_VERSION,
            "dataset_manifest_bundle_sha256": _artifact_bundle_sha256(artifacts),
        },
        "model_contract_sha256": _model_contract_sha256(),
        "coordinate_contract": {
            "schema": V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
            "version": V5_SOBOL_RECIPE_COORDINATE_VERSION,
            "sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
            "dimension": V5_SOBOL_RECIPE_DIM,
        },
        "amplitude_range_assignment_contract": {
            "schema": V5_AMPLITUDE_RANGE_ASSIGNMENT_SCHEMA,
            "version": V5_AMPLITUDE_RANGE_ASSIGNMENT_VERSION,
        },
        "splits": {
            "parent_set_hash_semantics": V5_K1_PARENT_SET_HASH_SEMANTICS,
            "train": train_summary,
            "tuning_validation": tuning_summary,
            "train_parent_set_sha256": train_parent_sha,
            "tuning_parent_set_sha256": tuning_parent_sha,
            "train_tuning_disjoint": True,
            "train_tuning_disjointness_receipt_sha256": digest(
                train_tuning_disjointness_receipt_sha256,
                "train_tuning_disjointness_receipt_sha256",
            ),
            "k1_phase_c_split_id": K1_PHASE_C_SPLIT_ID,
            "k1_phase_c_contract_sha256": phase_c["contract_sha256"],
            "train_and_tuning_are_disjoint_from_k1_phase_c": True,
            "k1_phase_c_disjointness_receipt_sha256": digest(
                k1_phase_c_disjointness_receipt_sha256,
                "k1_phase_c_disjointness_receipt_sha256",
            ),
        },
        "phase_a": {
            "result_payload_sha256": _optional_digest(
                phase_a_result_payload_sha256, "phase_a_result_payload_sha256"
            ),
            "role": "single_branch_capacity_diagnostic_only",
            "counts_as_full_k1_coverage": False,
            "weights_accepted_as_k1_training_initialization_by_this_contract": False,
        },
        "artifacts": {
            "train": [value.audit_payload() for value in train],
            "tuning_validation": [value.audit_payload() for value in tuning],
        },
        "full_search_supervision": {
            "all_artifacts_declare_training_eligible": full_eligible,
            "current_runtime_capabilities": v5_k1_training_runtime_capabilities(),
        },
        "claim_limits": {
            "inventory_is_training_completion": False,
            "inventory_is_k1_phase_c_pass": False,
            "inventory_is_paper_model_acceptance": False,
        },
    }
    return {
        **core,
        "inventory_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def validate_v5_k1_training_inventory(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("K1 training inventory must be a mapping")
    value = dict(payload)
    supplied_sha = digest(value.pop("inventory_sha256", None), "inventory_sha256")
    if supplied_sha != sha256(canonical_json(value).encode("utf-8")).hexdigest():
        raise ValueError("K1 training inventory SHA-256 does not reproduce")
    expected_fields = {
        "schema",
        "version",
        "scientific_role",
        "source_archive_sha256",
        "source_bundle_sha256",
        "dataset_contract",
        "model_contract_sha256",
        "coordinate_contract",
        "amplitude_range_assignment_contract",
        "splits",
        "phase_a",
        "artifacts",
        "full_search_supervision",
        "claim_limits",
    }
    if set(value) != expected_fields:
        raise ValueError("K1 training inventory fields are incomplete or unsupported")
    if (value["schema"], value["version"]) != (
        V5_K1_TRAINING_INVENTORY_SCHEMA,
        V5_K1_TRAINING_INVENTORY_VERSION,
    ):
        raise ValueError("K1 training inventory schema/version drifted")
    artifacts = value["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "train",
        "tuning_validation",
    }:
        raise ValueError("inventory artifacts must contain train and tuning_validation")
    train = tuple(V5K1TrainingArtifact.from_payload(item) for item in artifacts["train"])
    tuning = tuple(
        V5K1TrainingArtifact.from_payload(item) for item in artifacts["tuning_validation"]
    )
    replay = build_v5_k1_training_inventory(
        source_archive_sha256=value["source_archive_sha256"],
        source_bundle_sha256=value["source_bundle_sha256"],
        train_artifacts=train,
        tuning_artifacts=tuning,
        train_parent_set_sha256=value["splits"]["train_parent_set_sha256"],
        tuning_parent_set_sha256=value["splits"]["tuning_parent_set_sha256"],
        train_tuning_disjointness_receipt_sha256=value["splits"][
            "train_tuning_disjointness_receipt_sha256"
        ],
        k1_phase_c_disjointness_receipt_sha256=value["splits"][
            "k1_phase_c_disjointness_receipt_sha256"
        ],
        phase_a_result_payload_sha256=value["phase_a"]["result_payload_sha256"],
    )
    if dict(payload) != replay:
        raise ValueError("K1 training inventory drifted from live contracts or derived totals")
    return replay


def write_v5_k1_training_inventory(path: str | Path, payload: Mapping[str, object]) -> Path:
    validated = validate_v5_k1_training_inventory(payload)
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError("inventory parent directory does not exist")
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(validated, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return target


__all__ = [
    "V5K1TrainingArtifact",
    "V5_K1_FORMAL_TRAINING_AUTHORIZATION_ADAPTER_AVAILABLE",
    "V5_K1_FORMAL_MINIMUM_MODEL_SEEDS",
    "V5_K1_TRAINING_CHAIN_SCHEMA",
    "V5_K1_TRAINING_CHAIN_VERSION",
    "V5_K1_TRAINING_INVENTORY_SCHEMA",
    "V5_K1_TRAINING_INVENTORY_VERSION",
    "V5_K1_TRAINING_MODES",
    "build_v5_k1_training_inventory",
    "canonical_json",
    "digest",
    "positive_integer",
    "v5_k1_training_runtime_capabilities",
    "validate_v5_k1_training_inventory",
    "write_v5_k1_training_inventory",
]
