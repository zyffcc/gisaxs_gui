"""Immutable identity gate between balanced K1 data and formal training.

The gate proves only that the published train and tuning populations are the
ones excluded from the published Phase-C holdout.  It deliberately cannot
authorize gradients: full-search supervision and exact-budget tuning remain
separate, later gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from .k1_phase_c_contract_v5 import K1_PHASE_C_SPLIT_ID


V5_K1_TRAINING_IDENTITY_AUTHORIZATION_SCHEMA = (
    "gisaxs.posterior_v8.k1_training_identity_authorization/v1"
)
V5_K1_TRAINING_IDENTITY_AUTHORIZATION_VERSION = (
    "posterior_v8_v5_2_actual_train_tune_phase_c_exclusion_gate_v1"
)
V5_K1_TRAINING_IDENTITY_ROLES = (
    "train",
    "tuning_validation",
    "phase_c_holdout",
)
_SPLIT_BY_ROLE = {
    "train": "train",
    "tuning_validation": "tuning_validation",
    "phase_c_holdout": K1_PHASE_C_SPLIT_ID,
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _digests(values: Sequence[str], name: str) -> tuple[str, ...]:
    checked = tuple(_digest(value, f"{name}[{index}]") for index, value in enumerate(values))
    if not checked or len(checked) != len(set(checked)):
        raise ValueError(f"{name} must be non-empty and unique")
    return tuple(sorted(checked))


@dataclass(frozen=True, eq=False, kw_only=True)
class V5K1TrainingPopulationIdentity:
    """Small immutable summary of one actual recipe/clean-group population."""

    role: str
    split_id: str
    plan_sha256: str
    artifact_sha256s: tuple[str, ...]
    manifest_sha256s: tuple[str, ...]
    clean_parent_count: int
    recipe_set_sha256: str
    clean_group_set_sha256: str

    def __post_init__(self) -> None:
        if self.role not in V5_K1_TRAINING_IDENTITY_ROLES:
            raise ValueError("population role is unsupported")
        if self.split_id != _SPLIT_BY_ROLE[self.role]:
            raise ValueError("population role and split_id disagree")
        object.__setattr__(self, "plan_sha256", _digest(self.plan_sha256, "plan_sha256"))
        artifacts = _digests(self.artifact_sha256s, "artifact_sha256s")
        manifests = _digests(self.manifest_sha256s, "manifest_sha256s")
        if len(artifacts) != len(manifests):
            raise ValueError("population artifact and manifest inventories disagree")
        object.__setattr__(self, "artifact_sha256s", artifacts)
        object.__setattr__(self, "manifest_sha256s", manifests)
        object.__setattr__(
            self,
            "clean_parent_count",
            _positive_integer(self.clean_parent_count, "clean_parent_count"),
        )
        object.__setattr__(
            self, "recipe_set_sha256", _digest(self.recipe_set_sha256, "recipe_set_sha256")
        )
        object.__setattr__(
            self,
            "clean_group_set_sha256",
            _digest(self.clean_group_set_sha256, "clean_group_set_sha256"),
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "role": self.role,
            "split_id": self.split_id,
            "plan_sha256": self.plan_sha256,
            "artifact_sha256s": list(self.artifact_sha256s),
            "manifest_sha256s": list(self.manifest_sha256s),
            "clean_parent_count": self.clean_parent_count,
            "recipe_set_sha256": self.recipe_set_sha256,
            "clean_group_set_sha256": self.clean_group_set_sha256,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V5K1TrainingPopulationIdentity":
        if not isinstance(payload, Mapping) or set(payload) != {
            "role",
            "split_id",
            "plan_sha256",
            "artifact_sha256s",
            "manifest_sha256s",
            "clean_parent_count",
            "recipe_set_sha256",
            "clean_group_set_sha256",
        }:
            raise ValueError("population identity fields are incomplete or unsupported")
        return cls(
            role=payload["role"],
            split_id=payload["split_id"],
            plan_sha256=payload["plan_sha256"],
            artifact_sha256s=tuple(payload["artifact_sha256s"]),
            manifest_sha256s=tuple(payload["manifest_sha256s"]),
            clean_parent_count=payload["clean_parent_count"],
            recipe_set_sha256=payload["recipe_set_sha256"],
            clean_group_set_sha256=payload["clean_group_set_sha256"],
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class V5K1TrainingIdentityAuthorization:
    """Role-separated proof that train/tune exclude the formal holdout."""

    source_archive_sha256: str
    source_bundle_sha256: str
    balanced_dataset_completion_file_sha256: str
    balanced_dataset_completion_sha256: str
    train_tuning_receipt_file_sha256: str
    train_tuning_receipt_sha256: str
    train_tuning_claim_sha256: str
    phase_c_completion_file_sha256: str
    phase_c_completion_sha256: str
    three_way_receipt_file_sha256: str
    three_way_receipt_sha256: str
    phase_c_exclusion_claim_sha256: str
    populations: tuple[V5K1TrainingPopulationIdentity, ...]
    schema: str = V5_K1_TRAINING_IDENTITY_AUTHORIZATION_SCHEMA
    version: str = V5_K1_TRAINING_IDENTITY_AUTHORIZATION_VERSION

    def __post_init__(self) -> None:
        if (self.schema, self.version) != (
            V5_K1_TRAINING_IDENTITY_AUTHORIZATION_SCHEMA,
            V5_K1_TRAINING_IDENTITY_AUTHORIZATION_VERSION,
        ):
            raise ValueError("unsupported K1 training identity authorization")
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
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        populations = tuple(self.populations)
        if not all(isinstance(value, V5K1TrainingPopulationIdentity) for value in populations):
            raise TypeError("populations contain an invalid value")
        if tuple(value.role for value in populations) != V5_K1_TRAINING_IDENTITY_ROLES:
            raise ValueError("populations must use canonical train/tune/Phase-C order")
        if populations[0].plan_sha256 != populations[1].plan_sha256:
            raise ValueError("train and tuning populations must share one plan")
        if len({value.recipe_set_sha256 for value in populations}) != len(populations):
            raise ValueError("population recipe-set identities must be distinct")
        if len({value.clean_group_set_sha256 for value in populations}) != len(populations):
            raise ValueError("population clean-group identities must be distinct")
        object.__setattr__(self, "populations", populations)

    def population(self, role: str) -> V5K1TrainingPopulationIdentity:
        try:
            return next(value for value in self.populations if value.role == role)
        except StopIteration as exc:  # pragma: no cover - guarded by __post_init__
            raise ValueError("population role is unsupported") from exc

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "scientific_role": "actual_identity_exclusion_gate_not_gradient_authorization",
            "source_archive_sha256": self.source_archive_sha256,
            "source_bundle_sha256": self.source_bundle_sha256,
            "balanced_dataset_completion_file_sha256": (
                self.balanced_dataset_completion_file_sha256
            ),
            "balanced_dataset_completion_sha256": self.balanced_dataset_completion_sha256,
            "train_tuning_receipt_file_sha256": self.train_tuning_receipt_file_sha256,
            "train_tuning_receipt_sha256": self.train_tuning_receipt_sha256,
            "train_tuning_claim_sha256": self.train_tuning_claim_sha256,
            "phase_c_completion_file_sha256": self.phase_c_completion_file_sha256,
            "phase_c_completion_sha256": self.phase_c_completion_sha256,
            "three_way_receipt_file_sha256": self.three_way_receipt_file_sha256,
            "three_way_receipt_sha256": self.three_way_receipt_sha256,
            "phase_c_exclusion_claim_sha256": self.phase_c_exclusion_claim_sha256,
            "populations": {
                value.role: value.audit_payload() for value in self.populations
            },
            "phase_c_exclusion_proven": True,
            "full_search_supervision_complete": False,
            "tuning_exact_budget_summary_complete": False,
            "training_authorization_granted": False,
            "scientific_acceptance_evidence": False,
        }

    @property
    def sha256(self) -> str:
        return sha256(_canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {**self.audit_payload(), "authorization_sha256": self.sha256}

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V5K1TrainingIdentityAuthorization":
        if not isinstance(payload, Mapping):
            raise TypeError("K1 training identity authorization must be an object")
        values = dict(payload)
        supplied = _digest(values.pop("authorization_sha256", None), "authorization_sha256")
        populations = values.pop("populations", None)
        if not isinstance(populations, Mapping) or set(populations) != set(
            V5_K1_TRAINING_IDENTITY_ROLES
        ):
            raise ValueError("authorization population inventory is incomplete")
        fixed_claims = {
            "scientific_role": "actual_identity_exclusion_gate_not_gradient_authorization",
            "phase_c_exclusion_proven": True,
            "full_search_supervision_complete": False,
            "tuning_exact_budget_summary_complete": False,
            "training_authorization_granted": False,
            "scientific_acceptance_evidence": False,
        }
        for name, expected in fixed_claims.items():
            if values.pop(name, None) != expected:
                raise ValueError(f"authorization fixed claim drifted: {name}")
        authorization = cls(
            **values,
            populations=tuple(
                V5K1TrainingPopulationIdentity.from_payload(populations[role])
                for role in V5_K1_TRAINING_IDENTITY_ROLES
            ),
        )
        if supplied != authorization.sha256:
            raise ValueError("K1 training identity authorization SHA-256 does not reproduce")
        return authorization


__all__ = [
    "V5K1TrainingIdentityAuthorization",
    "V5K1TrainingPopulationIdentity",
    "V5_K1_TRAINING_IDENTITY_AUTHORIZATION_SCHEMA",
    "V5_K1_TRAINING_IDENTITY_AUTHORIZATION_VERSION",
    "V5_K1_TRAINING_IDENTITY_ROLES",
]
