"""Plan-bound task authorization for balanced all-K1 full search.

The balanced dataset artifact is the population identity, while exact search
uses a lossless one-observation projection of that artifact.  This module is
the explicit adapter between those two identities and the mature formal search
evidence contract.  It authorizes execution of one search task only; no object
here claims that search evidence exists or that gradient training may start.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import PurePosixPath
from typing import Mapping, Sequence

import numpy as np

from .contract import topology_from_id
from .formal_label_observation_policy_v5 import (
    V5FormalLabelObservationSelection,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
)
from .formal_production_search_plan_v5 import (
    V5FormalProductionSearchAuthorization,
    V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE,
    V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE,
)
from .grouped_artifact_v5 import canonical_json
from .grouped_dataset_v5 import clean_array, observation_array
from .k1_balanced_dataset_plan_v5 import (
    K1_BALANCED_FORMAL_VIEW_INDICES,
)
from .k1_balanced_full_search_parent_v5 import (
    V5K1BalancedFullSearchParentProjection,
)
from .k1_balanced_full_search_plan_v5 import (
    V5_K1_BALANCED_FULL_SEARCH_CANDIDATE_VIEW_INDICES,
    validate_v5_k1_balanced_full_search_plan,
)
from .k1_forced_sobol_recipe_v5 import V5K1PersistedForcedCleanRecipe
from .k1_forced_universal_query_v5 import V5K1ForcedUniversalQuerySet


V5_K1_BALANCED_FULL_SEARCH_TASK_MEMBERSHIP_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_task_membership/v1"
)
V5_K1_BALANCED_FULL_SEARCH_TASK_MEMBERSHIP_VERSION = (
    "posterior_v8_v5_2_original_artifact_projected_parent_member_binding_v1"
)
V5_K1_BALANCED_FULL_SEARCH_TASK_PLAN_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_task_plan/v1"
)
V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_task_authorization/v1"
)
V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_VERSION = (
    "posterior_v8_v5_2_plan_identity_projection_formal_search_adapter_v1"
)


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < int(positive):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _payload_sha256(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5K1BalancedFullSearchRecipeMember:
    """One artifact-authoritative projected parent member."""

    sobol_index: int
    clean_group_id: str
    recipe_sha256: str
    sobol_design_sha256: str
    query_set_sha256: str
    observation_selection_sha256: str
    selected_view_index: int
    generating_topology_id: int
    branch_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sobol_index", _integer(self.sobol_index, "sobol_index")
        )
        object.__setattr__(
            self,
            "selected_view_index",
            _integer(self.selected_view_index, "selected_view_index"),
        )
        object.__setattr__(
            self,
            "generating_topology_id",
            _integer(self.generating_topology_id, "generating_topology_id"),
        )
        topology_from_id(self.generating_topology_id)
        object.__setattr__(
            self, "branch_count", _integer(self.branch_count, "branch_count", positive=True)
        )
        for name in (
            "clean_group_id",
            "recipe_sha256",
            "sobol_design_sha256",
            "query_set_sha256",
            "observation_selection_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))

    def audit_payload(self) -> dict[str, object]:
        return {
            "sobol_index": self.sobol_index,
            "clean_group_id": self.clean_group_id,
            "recipe_sha256": self.recipe_sha256,
            "sobol_design_sha256": self.sobol_design_sha256,
            "query_set_sha256": self.query_set_sha256,
            "observation_selection_sha256": self.observation_selection_sha256,
            "selected_view_index": self.selected_view_index,
            "generating_topology_id": self.generating_topology_id,
            "branch_count": self.branch_count,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V5K1BalancedFullSearchRecipeMember":
        if not isinstance(payload, Mapping) or set(payload) != {
            "sobol_index",
            "clean_group_id",
            "recipe_sha256",
            "sobol_design_sha256",
            "query_set_sha256",
            "observation_selection_sha256",
            "selected_view_index",
            "generating_topology_id",
            "branch_count",
        }:
            raise ValueError("balanced full-search recipe member fields are unsupported")
        return cls(**payload)


@dataclass(frozen=True, eq=False, kw_only=True)
class V5K1BalancedFullSearchTaskMembership:
    """Non-promoting membership proof for one projected search parent."""

    array_task_id: int
    source_parent_artifact_sha256: str
    balanced_dataset_plan_sha256: str
    balanced_sobol_block_sha256: str
    role: str
    split_id: str
    shard_index: int
    split_offset: int
    source_selection_sha256: str
    projected_parent_sha256: str
    candidate_view_indices: tuple[int, ...]
    members: tuple[V5K1BalancedFullSearchRecipeMember, ...]
    schema: str = V5_K1_BALANCED_FULL_SEARCH_TASK_MEMBERSHIP_SCHEMA
    version: str = V5_K1_BALANCED_FULL_SEARCH_TASK_MEMBERSHIP_VERSION

    def __post_init__(self) -> None:
        if (self.schema, self.version) != (
            V5_K1_BALANCED_FULL_SEARCH_TASK_MEMBERSHIP_SCHEMA,
            V5_K1_BALANCED_FULL_SEARCH_TASK_MEMBERSHIP_VERSION,
        ):
            raise ValueError("unsupported balanced full-search task membership")
        object.__setattr__(
            self, "array_task_id", _integer(self.array_task_id, "array_task_id")
        )
        object.__setattr__(self, "shard_index", _integer(self.shard_index, "shard_index"))
        object.__setattr__(self, "split_offset", _integer(self.split_offset, "split_offset"))
        for name in (
            "source_parent_artifact_sha256",
            "balanced_dataset_plan_sha256",
            "balanced_sobol_block_sha256",
            "source_selection_sha256",
            "projected_parent_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        if self.role not in ("train", "tuning_validation") or self.split_id != self.role:
            raise ValueError("balanced full-search membership role/split is invalid")
        views = tuple(
            _integer(value, "candidate_view_index")
            for value in self.candidate_view_indices
        )
        if (
            views != V5_K1_BALANCED_FULL_SEARCH_CANDIDATE_VIEW_INDICES
            or len(views) != len(set(views))
        ):
            raise ValueError("balanced full-search candidate views drifted")
        rows = tuple(self.members)
        if not rows or not all(
            isinstance(value, V5K1BalancedFullSearchRecipeMember) for value in rows
        ):
            raise ValueError("balanced full-search task membership has no recipe rows")
        indices = tuple(value.sobol_index for value in rows)
        if indices != tuple(range(indices[0], indices[0] + len(indices))):
            raise ValueError("balanced full-search recipe indices must be contiguous and ordered")
        if len({value.clean_group_id for value in rows}) != len(rows) or len(
            {value.recipe_sha256 for value in rows}
        ) != len(rows):
            raise ValueError("balanced full-search recipe/group membership is duplicated")
        if any(value.selected_view_index not in views for value in rows):
            raise ValueError("balanced full-search selected view escaped its candidate pool")
        designs = {value.sobol_design_sha256 for value in rows}
        if len(designs) != 1:
            raise ValueError("balanced full-search shard mixed Sobol designs")
        selection_core = {
            "balanced_dataset_plan_sha256": self.balanced_dataset_plan_sha256,
            "balanced_sobol_block_sha256": self.balanced_sobol_block_sha256,
            "split_offset": self.split_offset,
            "requested_count": len(rows),
            "actual_count": len(rows),
            "selected_sobol_indices": list(indices),
            "view_indices": list(K1_BALANCED_FORMAL_VIEW_INDICES),
            "selection_mode": "shard_index",
            "shard_index": self.shard_index,
            "generating_candidate_only": True,
        }
        if self.source_selection_sha256 != _payload_sha256(selection_core):
            raise ValueError("balanced full-search source selection does not reproduce")
        object.__setattr__(self, "candidate_view_indices", views)
        object.__setattr__(self, "members", rows)

    @property
    def recipe_membership_sha256(self) -> str:
        return _payload_sha256([value.audit_payload() for value in self.members])

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "scientific_role": "artifact_to_projected_search_parent_membership_not_training_evidence",
            "array_task_id": self.array_task_id,
            "source_parent_artifact_sha256": self.source_parent_artifact_sha256,
            "balanced_dataset_plan_sha256": self.balanced_dataset_plan_sha256,
            "balanced_sobol_block_sha256": self.balanced_sobol_block_sha256,
            "role": self.role,
            "split_id": self.split_id,
            "shard_index": self.shard_index,
            "split_offset": self.split_offset,
            "source_view_indices": list(K1_BALANCED_FORMAL_VIEW_INDICES),
            "source_selection_sha256": self.source_selection_sha256,
            "projected_parent_sha256": self.projected_parent_sha256,
            "candidate_view_indices": list(self.candidate_view_indices),
            "observation_policy_id": V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
            "observation_policy_sha256": V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
            "members": [value.audit_payload() for value in self.members],
            "recipe_membership_sha256": self.recipe_membership_sha256,
            "one_curve_blind_sigma_present_view_per_clean_parent": True,
            "search_evidence_created": False,
            "gradient_training_authorized": False,
        }

    @property
    def sha256(self) -> str:
        return _payload_sha256(self.audit_payload())

    def to_payload(self) -> dict[str, object]:
        return {**self.audit_payload(), "membership_sha256": self.sha256}

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V5K1BalancedFullSearchTaskMembership":
        if not isinstance(payload, Mapping):
            raise TypeError("balanced full-search task membership must be an object")
        values = dict(payload)
        supplied = _digest(values.pop("membership_sha256", None), "membership_sha256")
        fixed = {
            "scientific_role": "artifact_to_projected_search_parent_membership_not_training_evidence",
            "source_view_indices": list(K1_BALANCED_FORMAL_VIEW_INDICES),
            "observation_policy_id": V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
            "observation_policy_sha256": V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
            "one_curve_blind_sigma_present_view_per_clean_parent": True,
            "search_evidence_created": False,
            "gradient_training_authorized": False,
        }
        for name, expected in fixed.items():
            if values.pop(name, None) != expected:
                raise ValueError(f"balanced full-search membership {name} drifted")
        recipe_membership = _digest(
            values.pop("recipe_membership_sha256", None),
            "recipe_membership_sha256",
        )
        values["candidate_view_indices"] = tuple(values["candidate_view_indices"])
        values["members"] = tuple(
            V5K1BalancedFullSearchRecipeMember.from_payload(value)
            for value in values["members"]
        )
        result = cls(**values)
        if recipe_membership != result.recipe_membership_sha256:
            raise ValueError("balanced full-search recipe membership SHA-256 drifted")
        if supplied != result.sha256:
            raise ValueError("balanced full-search task membership SHA-256 drifted")
        return result


def build_v5_k1_balanced_full_search_task_membership(
    projection: V5K1BalancedFullSearchParentProjection,
    *,
    array_task_id: int,
    shard_index: int,
    split_offset: int,
    source_selection_sha256: str,
) -> V5K1BalancedFullSearchTaskMembership:
    """Bind a strictly replayed projection to its original balanced shard."""

    if not isinstance(projection, V5K1BalancedFullSearchParentProjection):
        raise TypeError("projection has an invalid type")
    rows = []
    parent = projection.parent
    for index, (recipe, query_set, selection) in enumerate(
        zip(
            projection.recipes,
            projection.query_sets,
            projection.observation_selections,
            strict=True,
        )
    ):
        if not isinstance(recipe, V5K1PersistedForcedCleanRecipe) or not isinstance(
            query_set, V5K1ForcedUniversalQuerySet
        ) or not isinstance(selection, V5FormalLabelObservationSelection):
            raise TypeError("projection recipe, query-set, or selection type is invalid")
        identity = recipe.identity
        if (
            query_set.identity != identity
            or selection.recipe_seed != recipe.recipe_seed
            or tuple(selection.candidate_view_indices)
            != V5_K1_BALANCED_FULL_SEARCH_CANDIDATE_VIEW_INDICES
            or identity.sobol_index
            != int(parent.arrays[clean_array("sobol_index")][index])
            or identity.clean_group_id
            != str(parent.arrays[clean_array("clean_group_id")][index])
            or identity.recipe_sha256
            != str(parent.arrays[clean_array("recipe_sha256")][index])
            or identity.sobol_design_sha256
            != str(parent.arrays[clean_array("sobol_design_sha256")][index])
            or index
            != int(parent.arrays[observation_array("recipe_index")][index])
            or selection.selected_view_index
            != int(parent.arrays[observation_array("view_index")][index])
        ):
            raise ValueError("projected recipe/query/observation membership drifted")
        rows.append(
            V5K1BalancedFullSearchRecipeMember(
                sobol_index=identity.sobol_index,
                clean_group_id=identity.clean_group_id,
                recipe_sha256=identity.recipe_sha256,
                sobol_design_sha256=identity.sobol_design_sha256,
                query_set_sha256=query_set.sha256,
                observation_selection_sha256=selection.audit_sha256,
                selected_view_index=selection.selected_view_index,
                generating_topology_id=query_set.generating_topology_id,
                branch_count=sum(
                    len(value.feasible_wire_pattern_ids)
                    for value in query_set.topology_queries
                ),
            )
        )
    audit = projection.audit
    return V5K1BalancedFullSearchTaskMembership(
        array_task_id=array_task_id,
        source_parent_artifact_sha256=audit["source_parent_artifact_sha256"],
        balanced_dataset_plan_sha256=audit["balanced_dataset_plan_sha256"],
        balanced_sobol_block_sha256=audit["balanced_sobol_block_sha256"],
        role=audit["role"],
        split_id=audit["split_id"],
        shard_index=shard_index,
        split_offset=split_offset,
        source_selection_sha256=source_selection_sha256,
        projected_parent_sha256=projection.sha256,
        candidate_view_indices=tuple(audit["candidate_view_indices"]),
        members=tuple(rows),
    )


@dataclass(frozen=True, eq=False)
class V5K1BalancedFullSearchTaskAuthorization:
    """Execution-only task plan plus its formal evidence authorization."""

    task_plan: Mapping[str, object]
    formal_authorization: V5FormalProductionSearchAuthorization
    schema: str = V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_SCHEMA
    version: str = V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_VERSION

    def __post_init__(self) -> None:
        if (self.schema, self.version) != (
            V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_SCHEMA,
            V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_VERSION,
        ):
            raise ValueError("unsupported balanced full-search task authorization")
        if not isinstance(self.task_plan, Mapping) or set(self.task_plan) != {
            "schema",
            "global_launch_plan_sha256",
            "source",
            "identity_authorization_sha256",
            "array_task_id",
            "stage_sha256",
            "parent_binding",
            "membership",
            "membership_sha256",
            "output_relative_path",
        }:
            raise ValueError("balanced full-search task plan fields are unsupported")
        if self.task_plan["schema"] != V5_K1_BALANCED_FULL_SEARCH_TASK_PLAN_SCHEMA:
            raise ValueError("balanced full-search task plan schema drifted")
        if not isinstance(
            self.formal_authorization, V5FormalProductionSearchAuthorization
        ):
            raise TypeError("formal_authorization has an invalid type")
        membership = V5K1BalancedFullSearchTaskMembership.from_payload(
            self.task_plan["membership"]
        )
        task_sha = self.task_plan_sha256
        authorization = self.formal_authorization
        if (
            self.task_plan["membership_sha256"] != membership.sha256
            or self.task_plan["array_task_id"] != membership.array_task_id
            or authorization.launch_plan_sha256
            != self.task_plan["global_launch_plan_sha256"]
            or authorization.launch_source_bundle_sha256
            != self.task_plan["source"]["bundle_sha256"]
            or authorization.source_identity_sha256
            != self.task_plan["identity_authorization_sha256"]
            or authorization.shard_plan_sha256 != task_sha
            or authorization.stage_sha256 != self.task_plan["stage_sha256"]
            or authorization.output_relative_path
            != self.task_plan["output_relative_path"]
            or authorization.recipe_membership_sha256
            != membership.recipe_membership_sha256
        ):
            raise ValueError("balanced full-search formal authorization binding drifted")

    @property
    def task_plan_sha256(self) -> str:
        return _payload_sha256(dict(self.task_plan))

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "scientific_role": "one_balanced_all_k1_full_search_task_execution_authorization",
            "task_plan": dict(self.task_plan),
            "task_plan_sha256": self.task_plan_sha256,
            "formal_production_authorization": self.formal_authorization.to_payload(),
            "formal_production_authorization_sha256": self.formal_authorization.sha256,
            "search_execution_authorized": True,
            "search_evidence_created": False,
            "full_search_supervision_complete": False,
            "gradient_training_authorized": False,
        }

    @property
    def sha256(self) -> str:
        return _payload_sha256(self.audit_payload())

    def to_payload(self) -> dict[str, object]:
        return {**self.audit_payload(), "task_authorization_sha256": self.sha256}

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V5K1BalancedFullSearchTaskAuthorization":
        if not isinstance(payload, Mapping):
            raise TypeError("balanced full-search task authorization must be an object")
        values = dict(payload)
        supplied = _digest(
            values.pop("task_authorization_sha256", None),
            "task_authorization_sha256",
        )
        task_plan_sha = _digest(
            values.pop("task_plan_sha256", None), "task_plan_sha256"
        )
        formal_sha = _digest(
            values.pop("formal_production_authorization_sha256", None),
            "formal_production_authorization_sha256",
        )
        fixed = {
            "scientific_role": "one_balanced_all_k1_full_search_task_execution_authorization",
            "search_execution_authorized": True,
            "search_evidence_created": False,
            "full_search_supervision_complete": False,
            "gradient_training_authorized": False,
        }
        for name, expected in fixed.items():
            if values.pop(name, None) != expected:
                raise ValueError(f"balanced full-search authorization {name} drifted")
        formal = V5FormalProductionSearchAuthorization.from_payload(
            values.pop("formal_production_authorization")
        )
        result = cls(
            task_plan=values.pop("task_plan"),
            formal_authorization=formal,
            schema=values.pop("schema"),
            version=values.pop("version"),
        )
        if values:
            raise ValueError("balanced full-search authorization has extra fields")
        if task_plan_sha != result.task_plan_sha256 or formal_sha != formal.sha256:
            raise ValueError("balanced full-search nested authorization SHA-256 drifted")
        if supplied != result.sha256:
            raise ValueError("balanced full-search task authorization SHA-256 drifted")
        return result


def authorize_v5_k1_balanced_full_search_task(
    plan_payload: Mapping[str, object],
    membership: V5K1BalancedFullSearchTaskMembership,
) -> V5K1BalancedFullSearchTaskAuthorization:
    """Derive one formal execution authorization from unique global membership."""

    plan = validate_v5_k1_balanced_full_search_plan(plan_payload)
    if not isinstance(membership, V5K1BalancedFullSearchTaskMembership):
        raise TypeError("membership has an invalid type")
    matches = tuple(
        value
        for value in plan["parents"]
        if value["array_task_id"] == membership.array_task_id
    )
    if len(matches) != 1:
        raise ValueError("task membership is absent or duplicated in the global plan")
    parent = matches[0]
    if (
        parent["role"] != membership.role
        or parent["split_id"] != membership.split_id
        or parent["balanced_sobol_block_sha256"]
        != membership.balanced_sobol_block_sha256
        or parent["shard_index"] != membership.shard_index
        or parent["split_offset"] != membership.split_offset
        or parent["recipe_count"] != len(membership.members)
        or parent["selection_sha256"] != membership.source_selection_sha256
        or parent["parent"]["artifact_sha256"]
        != membership.source_parent_artifact_sha256
    ):
        raise ValueError("task membership escaped its global parent binding")
    identity = plan["identity_authorization"]
    populations = identity["populations"]
    population = populations[membership.role]
    if population["plan_sha256"] != membership.balanced_dataset_plan_sha256:
        raise ValueError("task membership escaped the authorized dataset plan")
    stage_binding = plan["k1_stage"]
    stage = stage_binding["payload"]
    allowed_topologies = set(stage["selected_topology_ids"])
    if any(
        value.generating_topology_id not in allowed_topologies
        for value in membership.members
    ):
        raise ValueError("task membership escaped the all-K1 stage")
    run_root = PurePosixPath(plan["layout"]["run_root"])
    output_root = PurePosixPath(parent["search_output_root"])
    try:
        output_relative_path = output_root.relative_to(run_root).as_posix()
    except ValueError as exc:  # pragma: no cover - global plan guards this
        raise ValueError("task output escaped the global run root") from exc
    if not output_relative_path or ".." in PurePosixPath(output_relative_path).parts:
        raise ValueError("task output relative path is unsafe")
    task_plan = {
        "schema": V5_K1_BALANCED_FULL_SEARCH_TASK_PLAN_SCHEMA,
        "global_launch_plan_sha256": plan["plan_sha256"],
        "source": dict(plan["source"]),
        "identity_authorization_sha256": plan["identity_authorization_sha256"],
        "array_task_id": membership.array_task_id,
        "stage_sha256": stage_binding["stage_sha256"],
        "parent_binding": dict(parent),
        "membership": membership.to_payload(),
        "membership_sha256": membership.sha256,
        "output_relative_path": output_relative_path,
    }
    task_plan_sha = _payload_sha256(task_plan)
    protocol = stage["protocol"]
    calibration = protocol["calibration_identity"]
    consumer_role = (
        V5_FORMAL_PRODUCTION_TRAIN_CONSUMER_ROLE
        if membership.role == "train"
        else V5_FORMAL_PRODUCTION_TUNING_CONSUMER_ROLE
    )
    members = membership.members
    formal = V5FormalProductionSearchAuthorization(
        study_id=f"k1-balanced-all12-full-search-{plan['plan_sha256'][:16]}",
        consumer_role=consumer_role,
        launch_plan_sha256=plan["plan_sha256"],
        launch_source_bundle_sha256=plan["source"]["bundle_sha256"],
        source_identity_sha256=plan["identity_authorization_sha256"],
        shard_plan_sha256=task_plan_sha,
        stage_id="K1",
        stage_sha256=stage_binding["stage_sha256"],
        target_split=membership.split_id,
        output_relative_path=output_relative_path,
        split_plan_sha256=membership.balanced_dataset_plan_sha256,
        sobol_design_sha256=members[0].sobol_design_sha256,
        protocol_sha256=stage["protocol_sha256"],
        seed_schedule_sha256=stage["local_sobol_schedule_sha256"],
        optimizer_schedule_sha256=stage["optimizer_schedule_sha256"],
        calibration_artifact_sha256=calibration["artifact_sha256"],
        recipe_sobol_indices=tuple(value.sobol_index for value in members),
        clean_group_ids=tuple(value.clean_group_id for value in members),
        recipe_membership_sha256=membership.recipe_membership_sha256,
        expected_query_count=len(members),
        expected_branch_count=sum(value.branch_count for value in members),
        expected_exact_forward_calls=(
            sum(value.branch_count for value in members)
            * stage["exact_forward_call_budget_per_branch"]
        ),
    )
    return V5K1BalancedFullSearchTaskAuthorization(
        task_plan=task_plan,
        formal_authorization=formal,
    )


__all__ = [
    "V5K1BalancedFullSearchRecipeMember",
    "V5K1BalancedFullSearchTaskAuthorization",
    "V5K1BalancedFullSearchTaskMembership",
    "V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_SCHEMA",
    "V5_K1_BALANCED_FULL_SEARCH_TASK_AUTHORIZATION_VERSION",
    "V5_K1_BALANCED_FULL_SEARCH_TASK_MEMBERSHIP_SCHEMA",
    "V5_K1_BALANCED_FULL_SEARCH_TASK_MEMBERSHIP_VERSION",
    "V5_K1_BALANCED_FULL_SEARCH_TASK_PLAN_SCHEMA",
    "authorize_v5_k1_balanced_full_search_task",
    "build_v5_k1_balanced_full_search_task_membership",
]
