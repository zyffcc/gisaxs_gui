"""Immutable building blocks for paper-scale V5 exact-search launches.

This module is deliberately write-free.  It freezes the scientific identity
of the three staged searches and records every clean parent, direct-Sobol
cross-topology query, and curve-blind sigma-present observation selection in a
per-shard plan.  Execution and training promotion remain separate concerns.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from pathlib import PurePosixPath
import re
from typing import Mapping, Sequence

import numpy as np

from .calibrated_search_threshold_v5 import V5CalibrationArtifactIdentity
from .contract import NUM_TOPOLOGIES, topology_from_id
from .exact_search_schedule_v5 import (
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
)
from .formal_label_observation_policy_v5 import (
    V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
    V5FormalLabelObservationSelection,
    select_v5_formal_label_observation,
)
from .frozen_search_pipeline_contract_v5 import V5SelectedTopologySearchSchedule
from .grouped_artifact_v5 import canonical_json
from .search_supervision_contract_v5 import (
    V5FrozenExactSearchProtocol,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)
from .sobol_design_v5 import (
    V5SobolDesign,
    materialize_v5_design_points_for_indices,
)
from .sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_NAMES,
    V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
    V5_SOBOL_RECIPE_COORDINATE_SHA256,
    V5_SOBOL_RECIPE_COORDINATE_VERSION,
)
from .sobol_universal_query_design_v5 import (
    V5_SOBOL_UNIVERSAL_POINT_HASH_VERSION,
    V5_SOBOL_UNIVERSAL_QUERY_DESIGN_SCHEMA,
    V5_SOBOL_UNIVERSAL_QUERY_DESIGN_VERSION,
    materialize_v5_sobol_universal_topology_query_design,
)
from .split_design_v5 import V5SplitPlan
from .universal_query_contract_v5 import (
    V5_CONTEXT_BRANCH_KEY_VERSION,
    V5_GLOBAL_BRANCH_KEY_VERSION,
    V5_TOPOLOGY_QUERY_VERSION,
)


V5_FORMAL_PRODUCTION_SHARD_SCHEMA = (
    "gisaxs.posterior_v8.formal_production_search_shard/v1"
)
V5_FORMAL_PRODUCTION_SHARD_VERSION = (
    "posterior_v8_explicit_parent_complete_query_branch_membership_v2"
)
V5_FORMAL_PRODUCTION_SOURCE_SCHEMA = (
    "gisaxs.posterior_v8.formal_production_source_identity/v1"
)
V5_FORMAL_PRODUCTION_SOURCE_VERSION = (
    "posterior_v8_portable_complete_bundle_and_required_file_digests_v1"
)
V5_FORMAL_PRODUCTION_STAGE_SCHEMA = (
    "gisaxs.posterior_v8.formal_production_search_stage/v1"
)
V5_FORMAL_PRODUCTION_STAGE_VERSION = (
    "posterior_v8_k1_then_k1_through_k2_then_all34_explicit_budget_v1"
)
V5_FORMAL_PRODUCTION_STAGE_IDS = ("K1", "K2", "ALL34")
V5_FORMAL_PRODUCTION_ALLOWED_SPLITS = ("train", "tuning_validation")

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_STAGE_TOPOLOGY_IDS = {
    "K1": tuple(
        value for value in range(NUM_TOPOLOGIES) if len(topology_from_id(value)) == 1
    ),
    "K2": tuple(
        value for value in range(NUM_TOPOLOGIES) if len(topology_from_id(value)) <= 2
    ),
    "ALL34": tuple(range(NUM_TOPOLOGIES)),
}
_STAGE_SCOPE = {
    "K1": "all_and_only_K1_topologies",
    "K2": "all_K1_through_K2_topologies",
    "ALL34": "all_34_K1_through_K4_topologies",
}


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _ordered_indices(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("sobol_indices must be a sequence of integers")
    result = tuple(_nonnegative_integer(value, "sobol_index") for value in values)
    if not result:
        raise ValueError("sobol_indices cannot be empty")
    if tuple(sorted(set(result))) != result:
        raise ValueError("sobol_indices must be unique and strictly increasing")
    return result


def _candidate_views(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise TypeError("candidate_view_indices must be a sequence of integers")
    result = tuple(
        _nonnegative_integer(value, "candidate view index") for value in values
    )
    if not result or len(result) != len(set(result)):
        raise ValueError("candidate_view_indices must be non-empty and unique")
    return result


def _safe_output_root(value: object, *, stage_id: str, split: str) -> str:
    text = _text(value, "output_relative_path")
    if "\\" in text or text.endswith("/"):
        raise ValueError("output_relative_path must be one canonical POSIX path")
    path = PurePosixPath(text)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError("output_relative_path must be safe and relative")
    expected = ("labels", stage_id.lower(), split)
    if path.parts[:3] != expected or len(path.parts) < 4:
        raise ValueError(
            "output_relative_path must be below " + "/".join(expected)
        )
    return path.as_posix()


def formal_production_query_contract_payload() -> dict[str, object]:
    """Return the complete version identity used to replay direct query catalogs."""

    return {
        "direct_coordinate_schema": V5_SOBOL_RECIPE_COORDINATE_SCHEMA,
        "direct_coordinate_version": V5_SOBOL_RECIPE_COORDINATE_VERSION,
        "direct_coordinate_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
        "coordinate_names": list(V5_SOBOL_RECIPE_COORDINATE_NAMES),
        "universal_query_design_schema": V5_SOBOL_UNIVERSAL_QUERY_DESIGN_SCHEMA,
        "universal_query_design_version": V5_SOBOL_UNIVERSAL_QUERY_DESIGN_VERSION,
        "universal_point_hash_version": V5_SOBOL_UNIVERSAL_POINT_HASH_VERSION,
        "topology_query_version": V5_TOPOLOGY_QUERY_VERSION,
        "global_branch_key_version": V5_GLOBAL_BRANCH_KEY_VERSION,
        "context_branch_key_version": V5_CONTEXT_BRANCH_KEY_VERSION,
        "per_parent_query_design_sha256_is_frozen_in_shard": True,
        "curve_or_search_result_conditions_query": False,
    }


V5_FORMAL_PRODUCTION_QUERY_CONTRACT_SHA256 = sha256(
    canonical_json(formal_production_query_contract_payload()).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5FormalProductionSourceIdentity:
    """Portable identity from ``fingerprint_v5_frozen_search_source``."""

    bundle_sha256: str
    bundle_file_count: int
    required_file_sha256: tuple[tuple[str, str], ...]
    schema: str = V5_FORMAL_PRODUCTION_SOURCE_SCHEMA
    version: str = V5_FORMAL_PRODUCTION_SOURCE_VERSION

    def __post_init__(self) -> None:
        bundle = _digest(self.bundle_sha256, "source bundle SHA-256")
        count = _positive_integer(self.bundle_file_count, "bundle_file_count")
        rows = tuple(self.required_file_sha256)
        if not rows:
            raise ValueError("required source-file hashes cannot be empty")
        canonical_rows = []
        for raw_path, raw_digest in rows:
            path = PurePosixPath(_text(raw_path, "required source path"))
            if path.is_absolute() or "." in path.parts or ".." in path.parts:
                raise ValueError("required source paths must be safe and relative")
            canonical_rows.append(
                (path.as_posix(), _digest(raw_digest, f"source SHA-256[{path}]") )
            )
        canonical_tuple = tuple(sorted(canonical_rows))
        if len({value[0] for value in canonical_tuple}) != len(canonical_tuple):
            raise ValueError("required source paths must be unique")
        if count < len(canonical_tuple):
            raise ValueError("bundle_file_count cannot be smaller than its required subset")
        if self.schema != V5_FORMAL_PRODUCTION_SOURCE_SCHEMA or self.version != (
            V5_FORMAL_PRODUCTION_SOURCE_VERSION
        ):
            raise ValueError("unsupported formal-production source schema")
        object.__setattr__(self, "bundle_sha256", bundle)
        object.__setattr__(self, "bundle_file_count", count)
        object.__setattr__(self, "required_file_sha256", canonical_tuple)

    @classmethod
    def from_fingerprint(
        cls, fingerprint: Mapping[str, object]
    ) -> "V5FormalProductionSourceIdentity":
        if not isinstance(fingerprint, Mapping):
            raise TypeError("source fingerprint must be a mapping")
        hashes = fingerprint.get("required_file_sha256")
        if not isinstance(hashes, Mapping):
            raise ValueError("source fingerprint has no required-file hashes")
        return cls(
            bundle_sha256=fingerprint.get("bundle_sha256"),
            bundle_file_count=fingerprint.get("bundle_file_count"),
            required_file_sha256=tuple((str(key), value) for key, value in hashes.items()),
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "bundle_sha256": self.bundle_sha256,
            "bundle_file_count": self.bundle_file_count,
            "required_file_sha256": dict(self.required_file_sha256),
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, eq=False)
class V5FormalProductionSearchStage:
    """One exact staged topology scope with its concrete search schedules."""

    stage_id: str
    topology_schedule: V5SelectedTopologySearchSchedule
    seed_schedule: V5FrozenLocalSobolSchedule
    optimizer_schedule: V5FrozenExactOptimizerSchedule
    protocol: V5FrozenExactSearchProtocol
    schema: str = V5_FORMAL_PRODUCTION_STAGE_SCHEMA
    version: str = V5_FORMAL_PRODUCTION_STAGE_VERSION

    def __post_init__(self) -> None:
        stage_id = _text(self.stage_id, "stage_id")
        if stage_id not in V5_FORMAL_PRODUCTION_STAGE_IDS:
            raise ValueError(
                f"stage_id must be one of {V5_FORMAL_PRODUCTION_STAGE_IDS}"
            )
        if not isinstance(self.topology_schedule, V5SelectedTopologySearchSchedule):
            raise TypeError("topology_schedule has an invalid type")
        if self.topology_schedule.selected_topology_ids != _STAGE_TOPOLOGY_IDS[stage_id]:
            raise ValueError(f"{stage_id} topology scope is incomplete or over-broad")
        if not isinstance(self.seed_schedule, V5FrozenLocalSobolSchedule):
            raise TypeError("seed_schedule has an invalid type")
        if not isinstance(self.optimizer_schedule, V5FrozenExactOptimizerSchedule):
            raise TypeError("optimizer_schedule has an invalid type")
        if not isinstance(self.protocol, V5FrozenExactSearchProtocol):
            raise TypeError("protocol has an invalid type")
        if self.protocol.protocol_tier != V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED:
            raise ValueError("formal production requires paper_full_calibrated protocol")
        if not isinstance(self.protocol.calibration_identity, V5CalibrationArtifactIdentity):
            raise ValueError("formal production protocol has no checked calibration identity")
        bindings = (
            (
                self.protocol.exact_forward_call_budget,
                self.seed_schedule.point_count,
                "exact-forward budget",
            ),
            (self.protocol.seed_schedule_id, self.seed_schedule.schedule_id, "seed schedule ID"),
            (self.protocol.seed_schedule_sha256, self.seed_schedule.sha256, "seed schedule SHA-256"),
            (
                self.protocol.optimizer_schedule_id,
                self.optimizer_schedule.schedule_id,
                "optimizer schedule ID",
            ),
            (
                self.protocol.optimizer_schedule_sha256,
                self.optimizer_schedule.sha256,
                "optimizer schedule SHA-256",
            ),
            (
                self.protocol.termination_policy_id,
                self.optimizer_schedule.termination_policy_id,
                "termination policy",
            ),
        )
        for actual, expected, name in bindings:
            if actual != expected:
                raise ValueError(f"formal-production {name} does not reproduce")
        if self.optimizer_schedule.direct_scout_seed_count > self.seed_schedule.point_count:
            raise ValueError("direct scout prefix exceeds the exact-forward budget")
        if self.schema != V5_FORMAL_PRODUCTION_STAGE_SCHEMA or self.version != (
            V5_FORMAL_PRODUCTION_STAGE_VERSION
        ):
            raise ValueError("unsupported formal-production stage contract")
        object.__setattr__(self, "stage_id", stage_id)

    @property
    def exact_forward_call_budget(self) -> int:
        return self.protocol.exact_forward_call_budget

    @property
    def selected_topology_ids(self) -> tuple[int, ...]:
        return self.topology_schedule.selected_topology_ids

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "stage_id": self.stage_id,
            "stage_order": V5_FORMAL_PRODUCTION_STAGE_IDS.index(self.stage_id),
            "topology_scope": _STAGE_SCOPE[self.stage_id],
            "selected_topology_ids": list(self.selected_topology_ids),
            "selected_topologies": [
                list(topology_from_id(value)) for value in self.selected_topology_ids
            ],
            "exact_forward_call_budget_per_branch": self.exact_forward_call_budget,
            "topology_schedule": self.topology_schedule.audit_payload(),
            "topology_schedule_sha256": self.topology_schedule.sha256,
            "local_sobol_schedule": self.seed_schedule.audit_payload(),
            "local_sobol_schedule_sha256": self.seed_schedule.sha256,
            "optimizer_schedule": self.optimizer_schedule.audit_payload(),
            "optimizer_schedule_sha256": self.optimizer_schedule.sha256,
            "protocol": self.protocol.audit_payload(),
            "protocol_sha256": self.protocol.sha256,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class V5FormalProductionRecipeMember:
    sobol_index: int
    clean_group_id: str
    design_point_sha256: str
    query_design_sha256: str
    generating_topology_id: int
    branch_count: int
    observation_selection: V5FormalLabelObservationSelection

    def __post_init__(self) -> None:
        index = _nonnegative_integer(self.sobol_index, "sobol_index")
        topology_from_id(self.generating_topology_id)
        _positive_integer(self.branch_count, "branch_count")
        for value, name in (
            (self.clean_group_id, "clean_group_id"),
            (self.design_point_sha256, "design_point_sha256"),
            (self.query_design_sha256, "query_design_sha256"),
        ):
            _digest(value, name)
        if not isinstance(
            self.observation_selection, V5FormalLabelObservationSelection
        ):
            raise TypeError("observation_selection has an invalid type")
        if self.observation_selection.recipe_seed != index:
            raise ValueError("observation selection escaped its Sobol parent seed")

    def audit_payload(self) -> dict[str, object]:
        return {
            "sobol_index": self.sobol_index,
            "clean_group_id": self.clean_group_id,
            "design_point_sha256": self.design_point_sha256,
            "query_design_sha256": self.query_design_sha256,
            "generating_topology_id": self.generating_topology_id,
            "branch_count": self.branch_count,
            "selected_view_index": self.observation_selection.selected_view_index,
            "observation_selection_audit": self.observation_selection.audit_payload(),
            "observation_selection_sha256": self.observation_selection.audit_sha256,
        }


@dataclass(frozen=True, eq=False)
class V5FormalProductionSearchShard:
    """Exact formal shard member; its SHA is the execution shard-plan SHA."""

    stage: V5FormalProductionSearchStage
    target_split: str
    output_relative_path: str
    split_plan_sha256: str
    sobol_design_sha256: str
    candidate_view_indices: tuple[int, ...]
    recipes: tuple[V5FormalProductionRecipeMember, ...]
    schema: str = V5_FORMAL_PRODUCTION_SHARD_SCHEMA
    version: str = V5_FORMAL_PRODUCTION_SHARD_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.stage, V5FormalProductionSearchStage):
            raise TypeError("stage has an invalid type")
        if self.target_split not in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS:
            raise ValueError(
                f"target_split must be one of {V5_FORMAL_PRODUCTION_ALLOWED_SPLITS}"
            )
        path = _safe_output_root(
            self.output_relative_path,
            stage_id=self.stage.stage_id,
            split=self.target_split,
        )
        _digest(self.split_plan_sha256, "split_plan_sha256")
        _digest(self.sobol_design_sha256, "sobol_design_sha256")
        views = _candidate_views(self.candidate_view_indices)
        recipes = tuple(self.recipes)
        if not recipes or not all(
            isinstance(value, V5FormalProductionRecipeMember) for value in recipes
        ):
            raise ValueError("formal-production shard requires recipe members")
        indices = tuple(value.sobol_index for value in recipes)
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("shard recipe indices must be unique and strictly increasing")
        for recipe in recipes:
            if recipe.generating_topology_id not in self.stage.selected_topology_ids:
                raise ValueError("generating topology escaped the staged topology scope")
            if recipe.observation_selection.candidate_view_indices != views:
                raise ValueError("recipe observation selection escaped the frozen pool")
            if recipe.observation_selection.policy_id != (
                V5_FORMAL_LABEL_OBSERVATION_POLICY_ID
            ) or recipe.observation_selection.policy_sha256 != (
                V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256
            ):
                raise ValueError("recipe observation selection uses a stale policy")
        if self.schema != V5_FORMAL_PRODUCTION_SHARD_SCHEMA or self.version != (
            V5_FORMAL_PRODUCTION_SHARD_VERSION
        ):
            raise ValueError("unsupported formal-production shard contract")
        object.__setattr__(self, "output_relative_path", path)
        object.__setattr__(self, "candidate_view_indices", views)
        object.__setattr__(self, "recipes", recipes)

    @property
    def expected_query_count(self) -> int:
        return len(self.recipes)

    @property
    def expected_branch_count(self) -> int:
        return sum(value.branch_count for value in self.recipes)

    @property
    def expected_exact_forward_calls(self) -> int:
        return self.expected_branch_count * self.stage.exact_forward_call_budget

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "stage_id": self.stage.stage_id,
            "stage_sha256": self.stage.sha256,
            "target_split": self.target_split,
            "output_relative_path": self.output_relative_path,
            "split_plan_sha256": self.split_plan_sha256,
            "sobol_design_sha256": self.sobol_design_sha256,
            "query_contract_sha256": V5_FORMAL_PRODUCTION_QUERY_CONTRACT_SHA256,
            "observation_policy_id": V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
            "observation_policy_sha256": V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
            "candidate_view_indices": list(self.candidate_view_indices),
            "selected_topology_ids": list(self.stage.selected_topology_ids),
            "exact_forward_call_budget_per_branch": (
                self.stage.exact_forward_call_budget
            ),
            "recipes": [value.audit_payload() for value in self.recipes],
            "expected_query_count": self.expected_query_count,
            "expected_branch_count": self.expected_branch_count,
            "expected_exact_forward_calls": self.expected_exact_forward_calls,
            "one_sigma_present_observation_per_clean_parent": True,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


def plan_v5_formal_production_search_shard(
    *,
    split_plan: V5SplitPlan,
    sobol_design: V5SobolDesign,
    stage: V5FormalProductionSearchStage,
    target_split: str,
    sobol_indices: Sequence[int],
    output_relative_path: str,
    candidate_view_indices: Sequence[int] = (0, 1),
) -> V5FormalProductionSearchShard:
    """Replay a write-free shard from explicit clean-parent memberships."""

    if not isinstance(split_plan, V5SplitPlan):
        raise TypeError("split_plan has an invalid type")
    if not isinstance(sobol_design, V5SobolDesign):
        raise TypeError("sobol_design has an invalid type")
    if not isinstance(stage, V5FormalProductionSearchStage):
        raise TypeError("stage has an invalid type")
    if (
        sobol_design.coordinate_names != V5_SOBOL_RECIPE_COORDINATE_NAMES
        or sobol_design.coordinate_contract_sha256
        != V5_SOBOL_RECIPE_COORDINATE_SHA256
    ):
        raise ValueError("Sobol design is not bound to direct V5 recipe coordinates")
    if target_split not in V5_FORMAL_PRODUCTION_ALLOWED_SPLITS:
        raise ValueError(
            f"target_split must be one of {V5_FORMAL_PRODUCTION_ALLOWED_SPLITS}"
        )
    indices = _ordered_indices(sobol_indices)
    for index in indices:
        if split_plan.split_for_index(index) != target_split:
            raise ValueError("Sobol parent escaped its declared train/validation split")
    views = _candidate_views(candidate_view_indices)
    points = materialize_v5_design_points_for_indices(
        split_plan, sobol_design, indices
    )
    members = []
    for point in points:
        query = materialize_v5_sobol_universal_topology_query_design(
            point,
            sobol_design,
            selected_topology_ids=stage.selected_topology_ids,
        )
        branch_count = sum(
            len(value.feasible_wire_pattern_ids)
            for value in query.topology_queries
        )
        members.append(
            V5FormalProductionRecipeMember(
                sobol_index=point.sobol_index,
                clean_group_id=point.clean_group_id,
                design_point_sha256=query.design_point_sha256,
                query_design_sha256=query.sha256,
                generating_topology_id=query.generating_topology_id,
                branch_count=branch_count,
                observation_selection=select_v5_formal_label_observation(
                    point.sobol_index, views
                ),
            )
        )
    return V5FormalProductionSearchShard(
        stage=stage,
        target_split=target_split,
        output_relative_path=output_relative_path,
        split_plan_sha256=split_plan.sha256,
        sobol_design_sha256=sobol_design.sha256,
        candidate_view_indices=views,
        recipes=tuple(members),
    )


def topology_ids_for_v5_formal_production_stage(stage_id: str) -> tuple[int, ...]:
    """Expose an immutable copy of the preregistered staged topology scope."""

    selected = _text(stage_id, "stage_id")
    try:
        return _STAGE_TOPOLOGY_IDS[selected]
    except KeyError as exc:
        raise ValueError(
            f"stage_id must be one of {V5_FORMAL_PRODUCTION_STAGE_IDS}"
        ) from exc


__all__ = [
    "V5FormalProductionRecipeMember",
    "V5FormalProductionSearchShard",
    "V5FormalProductionSearchStage",
    "V5FormalProductionSourceIdentity",
    "V5_FORMAL_PRODUCTION_ALLOWED_SPLITS",
    "V5_FORMAL_PRODUCTION_QUERY_CONTRACT_SHA256",
    "V5_FORMAL_PRODUCTION_SHARD_SCHEMA",
    "V5_FORMAL_PRODUCTION_SHARD_VERSION",
    "V5_FORMAL_PRODUCTION_SOURCE_SCHEMA",
    "V5_FORMAL_PRODUCTION_SOURCE_VERSION",
    "V5_FORMAL_PRODUCTION_STAGE_IDS",
    "V5_FORMAL_PRODUCTION_STAGE_SCHEMA",
    "V5_FORMAL_PRODUCTION_STAGE_VERSION",
    "formal_production_query_contract_payload",
    "plan_v5_formal_production_search_shard",
    "topology_ids_for_v5_formal_production_stage",
]
