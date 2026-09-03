"""Canonical global plan for paper-scale V5 exact-search production.

The plan is a write-free preregistration artifact, not an executor.  It binds
checked scientific inputs to explicit non-overlapping train and tuning-
validation shards.  Promotion is expressed by a per-shard authorization whose
consumer role is derived from the frozen split; a plan-level Boolean is never
enough to make an artifact training eligible.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from .calibrated_search_threshold_v5 import V5CheckedCompatibilityCalibration
from .formal_label_observation_policy_v5 import (
    V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
    formal_label_observation_policy_payload,
)
from .formal_production_search_contract_v5 import (
    V5FormalProductionSearchShard,
    V5FormalProductionSearchStage,
    V5FormalProductionSourceIdentity,
    V5_FORMAL_PRODUCTION_ALLOWED_SPLITS,
    V5_FORMAL_PRODUCTION_QUERY_CONTRACT_SHA256,
    V5_FORMAL_PRODUCTION_STAGE_IDS,
    formal_production_query_contract_payload,
    plan_v5_formal_production_search_shard,
)
from .grouped_artifact_v5 import canonical_json
from .sobol_design_v5 import V5SobolDesign, v5_clean_group_id
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
)
from .split_design_v5 import V5SplitPlan


V5_FORMAL_PRODUCTION_SEARCH_PLAN_SCHEMA = (
    "gisaxs.posterior_v8.formal_production_search_launch_plan/v2"
)
V5_FORMAL_PRODUCTION_SEARCH_PLAN_VERSION = (
    "posterior_v8_role_isolated_per_shard_training_authorization_v2"
)
V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED = True
V5_FORMAL_PRODUCTION_SCIENTIFIC_SCOPE = (
    "paper_full_calibrated_model_free_exact_search_labels_for_model_development"
)
V5_FORMAL_PRODUCTION_AUTHORIZATION_SCHEMA = (
    "gisaxs.posterior_v8.formal_production_search_authorization/v1"
)
V5_FORMAL_PRODUCTION_AUTHORIZATION_VERSION = (
    "posterior_v8_plan_source_shard_protocol_membership_role_binding_v1"
)
V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE = "gradient_training"
V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE = "tuning_validation_only"
V5_FORMAL_PRODUCTION_CONSUMER_ROLES = (
    V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE,
    V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE,
)
V5_FORMAL_PRODUCTION_FORBIDDEN_TRAINING_SPLITS = (
    "calibration",
    "test",
    "reference",
    "ood",
    "ood_topology",
    "ood_range_width",
    "ood_weak_component",
    "ood_acquisition_policy",
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _member_identity_sha256(shard: V5FormalProductionSearchShard) -> str:
    return sha256(
        canonical_json(
            [
                {
                    "sobol_index": value.sobol_index,
                    "clean_group_id": value.clean_group_id,
                    "design_point_sha256": value.design_point_sha256,
                    "query_design_sha256": value.query_design_sha256,
                    "observation_selection_sha256": (
                        value.observation_selection.audit_sha256
                    ),
                    "selected_view_index": (
                        value.observation_selection.selected_view_index
                    ),
                }
                for value in shard.recipes
            ]
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, eq=False, kw_only=True)
class V5FormalProductionSearchAuthorization:
    """One plan-derived authorization for exactly one formal shard and role."""

    study_id: str
    consumer_role: str
    launch_plan_sha256: str
    launch_source_bundle_sha256: str
    source_identity_sha256: str
    shard_plan_sha256: str
    stage_id: str
    stage_sha256: str
    target_split: str
    output_relative_path: str
    split_plan_sha256: str
    sobol_design_sha256: str
    protocol_sha256: str
    seed_schedule_sha256: str
    optimizer_schedule_sha256: str
    calibration_artifact_sha256: str
    recipe_sobol_indices: tuple[int, ...]
    clean_group_ids: tuple[str, ...]
    recipe_membership_sha256: str
    expected_query_count: int
    expected_branch_count: int
    expected_exact_forward_calls: int
    schema: str = V5_FORMAL_PRODUCTION_AUTHORIZATION_SCHEMA
    version: str = V5_FORMAL_PRODUCTION_AUTHORIZATION_VERSION

    def __post_init__(self) -> None:
        for name in ("study_id", "stage_id", "output_relative_path"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if self.target_split not in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS:
            raise ValueError("formal authorization split is not train or tuning_validation")
        expected_role = (
            V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE
            if self.target_split == "train"
            else V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE
        )
        if self.consumer_role != expected_role:
            raise ValueError("formal authorization consumer role disagrees with its split")
        for name in (
            "launch_plan_sha256",
            "launch_source_bundle_sha256",
            "source_identity_sha256",
            "shard_plan_sha256",
            "stage_sha256",
            "split_plan_sha256",
            "sobol_design_sha256",
            "protocol_sha256",
            "seed_schedule_sha256",
            "optimizer_schedule_sha256",
            "calibration_artifact_sha256",
            "recipe_membership_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        indices = tuple(self.recipe_sobol_indices)
        if (
            not indices
            or any(
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
                or int(value) < 0
                for value in indices
            )
            or tuple(sorted(set(int(value) for value in indices)))
            != tuple(int(value) for value in indices)
        ):
            raise ValueError("authorization Sobol indices must be non-empty and ordered")
        groups = tuple(_digest(value, "clean_group_id") for value in self.clean_group_ids)
        if len(groups) != len(indices) or len(set(groups)) != len(groups):
            raise ValueError("authorization clean-group membership is incomplete or duplicated")
        object.__setattr__(self, "recipe_sobol_indices", tuple(int(value) for value in indices))
        object.__setattr__(self, "clean_group_ids", groups)
        for name in (
            "expected_query_count",
            "expected_branch_count",
            "expected_exact_forward_calls",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, int(value))
        if self.schema != V5_FORMAL_PRODUCTION_AUTHORIZATION_SCHEMA or self.version != (
            V5_FORMAL_PRODUCTION_AUTHORIZATION_VERSION
        ):
            raise ValueError("unsupported formal-production authorization contract")
        if self.expected_query_count != len(indices):
            raise ValueError("authorization query count disagrees with recipe membership")

    @property
    def gradient_training_eligible(self) -> bool:
        return self.consumer_role == V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "study_id": self.study_id,
            "consumer_role": self.consumer_role,
            "gradient_training_eligible": self.gradient_training_eligible,
            "launch_plan_sha256": self.launch_plan_sha256,
            "launch_source_bundle_sha256": self.launch_source_bundle_sha256,
            "source_identity_sha256": self.source_identity_sha256,
            "shard_plan_sha256": self.shard_plan_sha256,
            "stage_id": self.stage_id,
            "stage_sha256": self.stage_sha256,
            "target_split": self.target_split,
            "output_relative_path": self.output_relative_path,
            "split_plan_sha256": self.split_plan_sha256,
            "sobol_design_sha256": self.sobol_design_sha256,
            "protocol_sha256": self.protocol_sha256,
            "seed_schedule_sha256": self.seed_schedule_sha256,
            "optimizer_schedule_sha256": self.optimizer_schedule_sha256,
            "calibration_artifact_sha256": self.calibration_artifact_sha256,
            "recipe_sobol_indices": list(self.recipe_sobol_indices),
            "clean_group_ids": list(self.clean_group_ids),
            "recipe_membership_sha256": self.recipe_membership_sha256,
            "expected_query_count": self.expected_query_count,
            "expected_branch_count": self.expected_branch_count,
            "expected_exact_forward_calls": self.expected_exact_forward_calls,
            "forbidden_training_splits": list(
                V5_FORMAL_PRODUCTION_FORBIDDEN_TRAINING_SPLITS
            ),
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, object]:
        return {**self.audit_payload(), "authorization_sha256": self.sha256}

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V5FormalProductionSearchAuthorization":
        if not isinstance(payload, Mapping):
            raise TypeError("formal authorization payload must be an object")
        values = dict(payload)
        supplied = _digest(
            values.pop("authorization_sha256", None), "authorization_sha256"
        )
        forbidden = values.pop("forbidden_training_splits", None)
        gradient = values.pop("gradient_training_eligible", None)
        values["recipe_sobol_indices"] = tuple(values.get("recipe_sobol_indices", ()))
        values["clean_group_ids"] = tuple(values.get("clean_group_ids", ()))
        authorization = cls(**values)
        if forbidden != list(V5_FORMAL_PRODUCTION_FORBIDDEN_TRAINING_SPLITS):
            raise ValueError("formal authorization forbidden-split policy drifted")
        if gradient is not authorization.gradient_training_eligible:
            raise ValueError("formal authorization gradient role claim drifted")
        if supplied != authorization.sha256:
            raise ValueError("formal authorization SHA-256 does not reproduce")
        return authorization


@dataclass(frozen=True, eq=False)
class V5FormalProductionSearchPlan:
    """Canonical global identity for every formal train/validation search shard."""

    study_id: str
    source: V5FormalProductionSourceIdentity
    split_plan: V5SplitPlan
    sobol_design: V5SobolDesign
    calibration: V5CheckedCompatibilityCalibration
    candidate_view_indices: tuple[int, ...]
    stages: tuple[V5FormalProductionSearchStage, ...]
    shards: tuple[V5FormalProductionSearchShard, ...]
    schema: str = V5_FORMAL_PRODUCTION_SEARCH_PLAN_SCHEMA
    version: str = V5_FORMAL_PRODUCTION_SEARCH_PLAN_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _text(self.study_id, "study_id"))
        if not isinstance(self.source, V5FormalProductionSourceIdentity):
            raise TypeError("source has an invalid type")
        if not isinstance(self.split_plan, V5SplitPlan):
            raise TypeError("split_plan has an invalid type")
        if not isinstance(self.sobol_design, V5SobolDesign):
            raise TypeError("sobol_design has an invalid type")
        if (
            self.sobol_design.coordinate_names != V5_SOBOL_RECIPE_COORDINATE_NAMES
            or self.sobol_design.coordinate_contract_sha256
            != V5_SOBOL_RECIPE_COORDINATE_SHA256
        ):
            raise ValueError("global Sobol design escaped the direct V5 contract")
        if not isinstance(self.calibration, V5CheckedCompatibilityCalibration):
            raise TypeError("calibration must be a checked compatibility calibration")
        views = tuple(self.candidate_view_indices)
        if not views:
            raise ValueError("candidate_view_indices cannot be empty")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or int(value) < 0
            for value in views
        ) or len(set(views)) != len(views):
            raise ValueError("candidate_view_indices must be unique non-negative integers")
        views = tuple(int(value) for value in views)

        stages = tuple(self.stages)
        if not all(isinstance(value, V5FormalProductionSearchStage) for value in stages):
            raise TypeError("stages contain an invalid value")
        if tuple(value.stage_id for value in stages) != V5_FORMAL_PRODUCTION_STAGE_IDS:
            raise ValueError("formal production requires ordered K1, K2, ALL34 stages")
        if len({value.sha256 for value in stages}) != len(stages) or len(
            {value.protocol.protocol_id for value in stages}
        ) != len(stages):
            raise ValueError("formal production stages require unique identities")
        for stage in stages:
            if stage.protocol.calibration_identity != self.calibration.identity:
                raise ValueError("a stage escaped the checked global calibration identity")

        shards = tuple(self.shards)
        if not shards or not all(
            isinstance(value, V5FormalProductionSearchShard) for value in shards
        ):
            raise ValueError("formal production requires explicit shard membership")
        stage_order = {
            value: index for index, value in enumerate(V5_FORMAL_PRODUCTION_STAGE_IDS)
        }
        split_order = {
            value: index
            for index, value in enumerate(V5_FORMAL_PRODUCTION_ALLOWED_SPLITS)
        }
        canonical_shards = tuple(
            sorted(
                shards,
                key=lambda value: (
                    stage_order[value.stage.stage_id],
                    split_order[value.target_split],
                    value.recipes[0].sobol_index,
                    value.output_relative_path,
                ),
            )
        )
        if shards != canonical_shards:
            raise ValueError("formal-production shards must use canonical order")
        by_stage = {value.stage_id: value for value in stages}
        seen_indices: set[int] = set()
        seen_groups: set[str] = set()
        seen_outputs: set[str] = set()
        seen_shas: set[str] = set()
        coverage = {
            (stage_id, split): 0
            for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS
            for split in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS
        }
        for shard in shards:
            expected_stage = by_stage[shard.stage.stage_id]
            if shard.stage.sha256 != expected_stage.sha256:
                raise ValueError("shard stage is not a member of the global stage plan")
            if (
                shard.split_plan_sha256 != self.split_plan.sha256
                or shard.sobol_design_sha256 != self.sobol_design.sha256
                or shard.candidate_view_indices != views
            ):
                raise ValueError("shard escaped a frozen global design identity")
            replay = plan_v5_formal_production_search_shard(
                split_plan=self.split_plan,
                sobol_design=self.sobol_design,
                stage=expected_stage,
                target_split=shard.target_split,
                sobol_indices=tuple(value.sobol_index for value in shard.recipes),
                output_relative_path=shard.output_relative_path,
                candidate_view_indices=views,
            )
            if replay.audit_payload() != shard.audit_payload() or replay.sha256 != shard.sha256:
                raise ValueError("shard does not replay from the frozen global contracts")
            for recipe in shard.recipes:
                if self.split_plan.split_for_index(recipe.sobol_index) != shard.target_split:
                    raise ValueError("recipe is not a member of its declared split")
                if v5_clean_group_id(
                    self.split_plan, self.sobol_design, recipe.sobol_index
                ) != recipe.clean_group_id:
                    raise ValueError("recipe clean-group identity does not reproduce")
                if recipe.sobol_index in seen_indices or recipe.clean_group_id in seen_groups:
                    raise ValueError("formal-production shards overlap clean parents")
                seen_indices.add(recipe.sobol_index)
                seen_groups.add(recipe.clean_group_id)
            if shard.output_relative_path in seen_outputs or shard.sha256 in seen_shas:
                raise ValueError("formal-production shard output or plan identity is duplicated")
            seen_outputs.add(shard.output_relative_path)
            seen_shas.add(shard.sha256)
            coverage[(shard.stage.stage_id, shard.target_split)] += 1
        if any(value == 0 for value in coverage.values()):
            raise ValueError("every stage requires explicit train and tuning-validation shards")
        train = {
            recipe.sobol_index
            for shard in shards
            if shard.target_split == "train"
            for recipe in shard.recipes
        }
        validation = {
            recipe.sobol_index
            for shard in shards
            if shard.target_split == "tuning_validation"
            for recipe in shard.recipes
        }
        if train & validation:
            raise ValueError("train and tuning-validation clean parents overlap")
        if self.schema != V5_FORMAL_PRODUCTION_SEARCH_PLAN_SCHEMA or self.version != (
            V5_FORMAL_PRODUCTION_SEARCH_PLAN_VERSION
        ):
            raise ValueError("unsupported formal-production global plan contract")
        object.__setattr__(self, "candidate_view_indices", views)
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "shards", shards)

    def _membership_summary(self) -> list[dict[str, object]]:
        rows = []
        for stage_id in V5_FORMAL_PRODUCTION_STAGE_IDS:
            for split in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS:
                selected = tuple(
                    recipe.sobol_index
                    for shard in self.shards
                    if shard.stage.stage_id == stage_id and shard.target_split == split
                    for recipe in shard.recipes
                )
                rows.append(
                    {
                        "stage_id": stage_id,
                        "split": split,
                        "shard_count": sum(
                            shard.stage.stage_id == stage_id
                            and shard.target_split == split
                            for shard in self.shards
                        ),
                        "recipe_count": len(selected),
                        "sobol_indices_sha256": sha256(
                            canonical_json(list(selected)).encode("utf-8")
                        ).hexdigest(),
                    }
                )
        return rows

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "study_id": self.study_id,
            "scientific_scope": V5_FORMAL_PRODUCTION_SCIENTIFIC_SCOPE,
            "source": self.source.audit_payload(),
            "source_identity_sha256": self.source.sha256,
            "split_plan": {
                "payload": json.loads(self.split_plan.canonical_json),
                "sha256": self.split_plan.sha256,
            },
            "sobol_design": {
                "payload": self.sobol_design.payload(),
                "sha256": self.sobol_design.sha256,
            },
            "query_contract": formal_production_query_contract_payload(),
            "query_contract_sha256": V5_FORMAL_PRODUCTION_QUERY_CONTRACT_SHA256,
            "observation_selection_contract": {
                "payload": formal_label_observation_policy_payload(),
                "policy_id": V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
                "policy_sha256": V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
                "candidate_view_indices": list(self.candidate_view_indices),
                "selected_views_per_clean_parent": 1,
                "measurement_sigma_required": True,
            },
            "calibration_identity": self.calibration.identity.audit_payload(),
            "calibration_identity_sha256": self.calibration.identity.sha256,
            "stage_order": list(V5_FORMAL_PRODUCTION_STAGE_IDS),
            "stages": [
                {**value.audit_payload(), "stage_sha256": value.sha256}
                for value in self.stages
            ],
            "shards": [
                {**value.audit_payload(), "shard_plan_sha256": value.sha256}
                for value in self.shards
            ],
            "membership_summary": self._membership_summary(),
            "non_overlap": {
                "statistical_unit": "clean_physical_recipe",
                "all_shard_sobol_indices_unique": True,
                "all_clean_group_ids_unique": True,
                "train_tuning_validation_disjoint": True,
                "all_output_relative_paths_unique": True,
            },
            "promotion_boundary": {
                "formal_membership_verifier_available": True,
                "training_promotion_enabled_by_this_module": (
                    V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED
                ),
                "authorization_schema": V5_FORMAL_PRODUCTION_AUTHORIZATION_SCHEMA,
                "authorization_version": V5_FORMAL_PRODUCTION_AUTHORIZATION_VERSION,
                "train_consumer_role": V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE,
                "tuning_consumer_role": V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE,
                "forbidden_training_splits": list(
                    V5_FORMAL_PRODUCTION_FORBIDDEN_TRAINING_SPLITS
                ),
                "consumer_role_is_derived_from_frozen_split": True,
                "tuning_validation_can_supply_gradients": False,
                "evidence_reader_must_pass_before_membership_replay": True,
                "membership_proof_alone_authorizes_training": False,
            },
        }

    @property
    def canonical_json(self) -> str:
        return canonical_json(self.audit_payload())

    @property
    def sha256(self) -> str:
        return sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def launch_plan_sha256(self) -> str:
        return self.sha256

    def to_payload(self) -> dict[str, object]:
        return {**self.audit_payload(), "plan_sha256": self.sha256}

    def to_json(self) -> str:
        return json.dumps(self.to_payload(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def build_v5_formal_production_search_plan(
    *,
    study_id: str,
    source: V5FormalProductionSourceIdentity,
    split_plan: V5SplitPlan,
    sobol_design: V5SobolDesign,
    calibration: V5CheckedCompatibilityCalibration,
    candidate_view_indices: Sequence[int],
    stages: Sequence[V5FormalProductionSearchStage],
    shards: Sequence[V5FormalProductionSearchShard],
) -> V5FormalProductionSearchPlan:
    """Canonicalize an already explicit, write-free formal launch plan."""

    stage_order = {
        value: index for index, value in enumerate(V5_FORMAL_PRODUCTION_STAGE_IDS)
    }
    split_order = {
        value: index
        for index, value in enumerate(V5_FORMAL_PRODUCTION_ALLOWED_SPLITS)
    }
    ordered_shards = tuple(
        sorted(
            tuple(shards),
            key=lambda value: (
                stage_order[value.stage.stage_id],
                split_order[value.target_split],
                value.recipes[0].sobol_index,
                value.output_relative_path,
            ),
        )
    )
    return V5FormalProductionSearchPlan(
        study_id=study_id,
        source=source,
        split_plan=split_plan,
        sobol_design=sobol_design,
        calibration=calibration,
        candidate_view_indices=tuple(candidate_view_indices),
        stages=tuple(stages),
        shards=ordered_shards,
    )


def authorize_v5_formal_production_search_shard(
    plan: V5FormalProductionSearchPlan,
    *,
    shard_plan_sha256: str,
) -> V5FormalProductionSearchAuthorization:
    """Derive one immutable role authorization from exact global-plan membership."""

    if not V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED:  # pragma: no cover
        raise RuntimeError("formal-production promotion is disabled")
    if not isinstance(plan, V5FormalProductionSearchPlan):
        raise TypeError("plan must be a V5FormalProductionSearchPlan")
    shard_sha = _digest(shard_plan_sha256, "shard_plan_sha256")
    matching = tuple(value for value in plan.shards if value.sha256 == shard_sha)
    if len(matching) != 1:
        raise ValueError("authorization requires one unique shard in the global plan")
    shard = matching[0]
    stage = shard.stage
    consumer_role = (
        V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE
        if shard.target_split == "train"
        else V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE
    )
    return V5FormalProductionSearchAuthorization(
        study_id=plan.study_id,
        consumer_role=consumer_role,
        launch_plan_sha256=plan.sha256,
        launch_source_bundle_sha256=plan.source.bundle_sha256,
        source_identity_sha256=plan.source.sha256,
        shard_plan_sha256=shard.sha256,
        stage_id=stage.stage_id,
        stage_sha256=stage.sha256,
        target_split=shard.target_split,
        output_relative_path=shard.output_relative_path,
        split_plan_sha256=plan.split_plan.sha256,
        sobol_design_sha256=plan.sobol_design.sha256,
        protocol_sha256=stage.protocol.sha256,
        seed_schedule_sha256=stage.seed_schedule.sha256,
        optimizer_schedule_sha256=stage.optimizer_schedule.sha256,
        calibration_artifact_sha256=plan.calibration.identity.artifact_sha256,
        recipe_sobol_indices=tuple(value.sobol_index for value in shard.recipes),
        clean_group_ids=tuple(value.clean_group_id for value in shard.recipes),
        recipe_membership_sha256=_member_identity_sha256(shard),
        expected_query_count=shard.expected_query_count,
        expected_branch_count=shard.expected_branch_count,
        expected_exact_forward_calls=shard.expected_exact_forward_calls,
    )


__all__ = [
    "V5FormalProductionSearchAuthorization",
    "V5FormalProductionSearchPlan",
    "V5_FORMAL_PRODUCTION_AUTHORIZATION_SCHEMA",
    "V5_FORMAL_PRODUCTION_AUTHORIZATION_VERSION",
    "V5_FORMAL_PRODUCTION_CONSUMER_ROLES",
    "V5_FORMAL_PRODUCTION_FORBIDDEN_TRAINING_SPLITS",
    "V5_FORMAL_PRODUCTION_SCIENTIFIC_SCOPE",
    "V5_FORMAL_PRODUCTION_SEARCH_PLAN_SCHEMA",
    "V5_FORMAL_PRODUCTION_SEARCH_PLAN_VERSION",
    "V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE",
    "V5_FORMAL_PRODUCTION_TRAINING_PROMOTION_ENABLED",
    "V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE",
    "authorize_v5_formal_production_search_shard",
    "build_v5_formal_production_search_plan",
]
