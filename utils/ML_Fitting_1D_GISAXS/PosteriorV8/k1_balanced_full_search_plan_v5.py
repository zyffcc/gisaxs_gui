"""Write-free global plan for balanced all-K1 full-search supervision."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Mapping, Sequence

import numpy as np

from .contract import topology_from_id
from .formal_label_observation_policy_v5 import (
    V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
)
from .formal_production_search_contract_v5 import (
    V5_FORMAL_PRODUCTION_STAGE_SCHEMA,
    V5_FORMAL_PRODUCTION_STAGE_VERSION,
    V5FormalProductionSearchStage,
    topology_ids_for_v5_formal_production_stage,
)
from .grouped_artifact_v5 import canonical_json
from .k1_balanced_dataset_plan_v5 import (
    K1_BALANCED_FORMAL_RECIPES_PER_SHARD,
    K1_BALANCED_FORMAL_TRAIN_PARENTS_PER_BRANCH,
    K1_BALANCED_FORMAL_TUNING_PARENTS_PER_BRANCH,
)
from .k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES
from .k1_training_identity_contract_v5 import V5K1TrainingIdentityAuthorization
from .search_supervision_contract_v5 import (
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)


V5_K1_BALANCED_FULL_SEARCH_PLAN_SCHEMA = (
    "gisaxs.posterior_v8.k1_balanced_full_search_launch_plan/v1"
)
V5_K1_BALANCED_FULL_SEARCH_PLAN_VERSION = (
    "posterior_v8_v5_2_actual_identity_bound_all12_full_search_array_v1"
)
V5_K1_BALANCED_FULL_SEARCH_CANDIDATE_VIEW_INDICES = (0, 1)
V5_K1_BALANCED_FULL_SEARCH_TASK_COUNT = 60
MAXWELL_DUST_ROOT = PurePosixPath("/data/dust/user/zhaiyufe")


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


def _absolute_dust_path(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty absolute path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or path == MAXWELL_DUST_ROOT:
        raise ValueError(f"{name} must be a child of {MAXWELL_DUST_ROOT}")
    try:
        path.relative_to(MAXWELL_DUST_ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} must be under {MAXWELL_DUST_ROOT}") from exc
    return str(path)


@dataclass(frozen=True, kw_only=True)
class V5K1BalancedFullSearchParentBinding:
    """One immutable balanced parent shard and its completion-last proof."""

    array_task_id: int
    role: str
    split_id: str
    branch_id: str
    branch_ordinal: int
    balanced_sobol_block_sha256: str
    shard_index: int
    split_offset: int
    recipe_count: int
    selection_sha256: str
    parent_path: str
    artifact_sha256: str
    manifest_sha256: str
    byte_count: int
    task_completion_path: str
    task_completion_file_sha256: str
    task_completion_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "array_task_id", _integer(self.array_task_id, "array_task_id")
        )
        if self.role not in ("train", "tuning_validation") or self.split_id != self.role:
            raise ValueError("parent role/split binding is invalid")
        branches = tuple(value.branch_id for value in K1_PHASE_C_BRANCHES)
        ordinal = _integer(self.branch_ordinal, "branch_ordinal")
        if ordinal >= len(branches) or branches[ordinal] != self.branch_id:
            raise ValueError("parent branch ID/ordinal binding is invalid")
        object.__setattr__(self, "branch_ordinal", ordinal)
        shard_index = _integer(self.shard_index, "shard_index")
        object.__setattr__(self, "shard_index", shard_index)
        split_offset = _integer(self.split_offset, "split_offset")
        if split_offset != shard_index * K1_BALANCED_FORMAL_RECIPES_PER_SHARD:
            raise ValueError("parent split offset disagrees with its shard index")
        object.__setattr__(self, "split_offset", split_offset)
        count = _integer(self.recipe_count, "recipe_count", positive=True)
        if count != K1_BALANCED_FORMAL_RECIPES_PER_SHARD:
            raise ValueError("full-search parent must contain one complete formal shard")
        object.__setattr__(self, "recipe_count", count)
        object.__setattr__(
            self, "byte_count", _integer(self.byte_count, "byte_count", positive=True)
        )
        for name in (
            "balanced_sobol_block_sha256",
            "selection_sha256",
            "artifact_sha256",
            "manifest_sha256",
            "task_completion_file_sha256",
            "task_completion_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        for name in ("parent_path", "task_completion_path"):
            object.__setattr__(
                self, name, _absolute_dust_path(getattr(self, name), name)
            )

    def audit_payload(self, *, search_run_root: str) -> dict[str, object]:
        branch_stem = f"branch-{self.branch_ordinal:02d}-shard-{self.shard_index:06d}"
        output_root = str(
            PurePosixPath(search_run_root) / "shards" / self.role / branch_stem
        )
        return {
            "array_task_id": self.array_task_id,
            "role": self.role,
            "split_id": self.split_id,
            "branch_id": self.branch_id,
            "branch_ordinal": self.branch_ordinal,
            "balanced_sobol_block_sha256": self.balanced_sobol_block_sha256,
            "shard_index": self.shard_index,
            "split_offset": self.split_offset,
            "recipe_count": self.recipe_count,
            "selection_sha256": self.selection_sha256,
            "parent": {
                "path": self.parent_path,
                "artifact_sha256": self.artifact_sha256,
                "manifest_sha256": self.manifest_sha256,
                "byte_count": self.byte_count,
                "mode_octal": "0400",
                "nlink": 1,
            },
            "task_completion": {
                "path": self.task_completion_path,
                "file_sha256": self.task_completion_file_sha256,
                "completion_sha256": self.task_completion_sha256,
                "mode_octal": "0400",
                "nlink": 1,
            },
            "search_output_root": output_root,
        }

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, object]
    ) -> "V5K1BalancedFullSearchParentBinding":
        if not isinstance(payload, Mapping):
            raise TypeError("full-search parent binding must be an object")
        value = dict(payload)
        parent = value.pop("parent", None)
        completion = value.pop("task_completion", None)
        value.pop("search_output_root", None)
        if not isinstance(parent, Mapping) or set(parent) != {
            "path",
            "artifact_sha256",
            "manifest_sha256",
            "byte_count",
            "mode_octal",
            "nlink",
        }:
            raise ValueError("parent artifact binding is incomplete")
        if not isinstance(completion, Mapping) or set(completion) != {
            "path",
            "file_sha256",
            "completion_sha256",
            "mode_octal",
            "nlink",
        }:
            raise ValueError("parent completion binding is incomplete")
        if (
            parent["mode_octal"] != "0400"
            or parent["nlink"] != 1
            or completion["mode_octal"] != "0400"
            or completion["nlink"] != 1
        ):
            raise ValueError("parent and completion must be 0400/nlink1")
        return cls(
            **value,
            parent_path=parent["path"],
            artifact_sha256=parent["artifact_sha256"],
            manifest_sha256=parent["manifest_sha256"],
            byte_count=parent["byte_count"],
            task_completion_path=completion["path"],
            task_completion_file_sha256=completion["file_sha256"],
            task_completion_sha256=completion["completion_sha256"],
        )


def _validate_stage_payload(payload: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("K1 stage payload must be an object")
    stage = dict(payload)
    if set(stage) != {
        "schema",
        "version",
        "stage_id",
        "stage_order",
        "topology_scope",
        "selected_topology_ids",
        "selected_topologies",
        "exact_forward_call_budget_per_branch",
        "topology_schedule",
        "topology_schedule_sha256",
        "local_sobol_schedule",
        "local_sobol_schedule_sha256",
        "optimizer_schedule",
        "optimizer_schedule_sha256",
        "protocol",
        "protocol_sha256",
    }:
        raise ValueError("K1 stage fields are incomplete or unsupported")
    selected = topology_ids_for_v5_formal_production_stage("K1")
    if (
        stage["schema"] != V5_FORMAL_PRODUCTION_STAGE_SCHEMA
        or stage["version"] != V5_FORMAL_PRODUCTION_STAGE_VERSION
        or stage["stage_id"] != "K1"
        or stage["stage_order"] != 0
        or tuple(stage["selected_topology_ids"]) != selected
        or stage["selected_topologies"]
        != [list(topology_from_id(value)) for value in selected]
        or stage["topology_scope"] != "all_and_only_K1_topologies"
    ):
        raise ValueError("full-search stage is not all and only K1")
    for payload_name, hash_name in (
        ("topology_schedule", "topology_schedule_sha256"),
        ("local_sobol_schedule", "local_sobol_schedule_sha256"),
        ("optimizer_schedule", "optimizer_schedule_sha256"),
        ("protocol", "protocol_sha256"),
    ):
        nested = stage[payload_name]
        if not isinstance(nested, Mapping) or stage[hash_name] != sha256(
            canonical_json(nested).encode("utf-8")
        ).hexdigest():
            raise ValueError(f"K1 stage {payload_name} binding drifted")
    protocol = stage["protocol"]
    if (
        protocol.get("protocol_tier")
        != V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED
        or protocol.get("same_budget_for_every_branch") is not True
        or protocol.get("uses_neural_proposal_scores") is not False
        or not isinstance(protocol.get("calibration_identity"), Mapping)
    ):
        raise ValueError("K1 stage lacks a self-consistent full calibrated protocol")
    if stage["exact_forward_call_budget_per_branch"] != protocol.get(
        "exact_forward_call_budget"
    ):
        raise ValueError("K1 stage exact-forward budget binding drifted")
    return stage


def _core(
    *,
    source_archive_sha256: str,
    source_bundle_sha256: str,
    balanced_dataset_launch_plan_file_sha256: str,
    balanced_dataset_launch_plan_sha256: str,
    balanced_dataset_completion_file_sha256: str,
    balanced_dataset_completion_sha256: str,
    identity_authorization: V5K1TrainingIdentityAuthorization,
    stage_payload: Mapping[str, object],
    search_run_root: str,
    parents: Sequence[V5K1BalancedFullSearchParentBinding],
) -> dict[str, object]:
    source_archive = _digest(source_archive_sha256, "source_archive_sha256")
    source_bundle = _digest(source_bundle_sha256, "source_bundle_sha256")
    balanced_completion_file = _digest(
        balanced_dataset_completion_file_sha256,
        "balanced_dataset_completion_file_sha256",
    )
    balanced_completion = _digest(
        balanced_dataset_completion_sha256,
        "balanced_dataset_completion_sha256",
    )
    if not isinstance(identity_authorization, V5K1TrainingIdentityAuthorization):
        raise TypeError("identity_authorization has an invalid type")
    if (
        identity_authorization.source_archive_sha256 != source_archive
        or identity_authorization.source_bundle_sha256 != source_bundle
        or identity_authorization.balanced_dataset_completion_file_sha256
        != balanced_completion_file
        or identity_authorization.balanced_dataset_completion_sha256
        != balanced_completion
    ):
        raise ValueError("identity authorization escaped the full-search inputs")
    root = _absolute_dust_path(search_run_root, "search_run_root")
    stage = _validate_stage_payload(stage_payload)
    rows = tuple(parents)
    if len(rows) != V5_K1_BALANCED_FULL_SEARCH_TASK_COUNT or not all(
        isinstance(value, V5K1BalancedFullSearchParentBinding) for value in rows
    ):
        raise ValueError("full-search plan requires exactly 60 parent shard bindings")
    if tuple(value.array_task_id for value in rows) != tuple(range(len(rows))):
        raise ValueError("full-search parent task IDs must be canonical and contiguous")
    expected_shards = {"train": 4, "tuning_validation": 1}
    for role, shards_per_branch in expected_shards.items():
        role_rows = tuple(value for value in rows if value.role == role)
        for branch in K1_PHASE_C_BRANCHES:
            selected = tuple(value for value in role_rows if value.branch_id == branch.branch_id)
            if (
                len(selected) != shards_per_branch
                or tuple(value.shard_index for value in selected)
                != tuple(range(shards_per_branch))
            ):
                raise ValueError("full-search parents are not balanced across all K1 branches")
    expected_counts = {
        "train": K1_BALANCED_FORMAL_TRAIN_PARENTS_PER_BRANCH * 12,
        "tuning_validation": K1_BALANCED_FORMAL_TUNING_PARENTS_PER_BRANCH * 12,
    }
    observed_counts = {
        role: sum(value.recipe_count for value in rows if value.role == role)
        for role in expected_counts
    }
    if observed_counts != expected_counts:
        raise ValueError("full-search parent counts disagree with the frozen E1 populations")
    paths = tuple(
        path
        for value in rows
        for path in (value.parent_path, value.task_completion_path)
    )
    hashes = tuple(value.artifact_sha256 for value in rows)
    if len(paths) != len(set(paths)) or len(hashes) != len(set(hashes)):
        raise ValueError("full-search parent paths or artifact identities are duplicated")
    for role in expected_counts:
        population = identity_authorization.population(role)
        selected = tuple(value for value in rows if value.role == role)
        if (
            population.artifact_sha256s
            != tuple(sorted(value.artifact_sha256 for value in selected))
            or population.manifest_sha256s
            != tuple(sorted(value.manifest_sha256 for value in selected))
            or population.clean_parent_count != observed_counts[role]
        ):
            raise ValueError("full-search parents escaped the authorized population")
    parent_payloads = [value.audit_payload(search_run_root=root) for value in rows]
    outputs = tuple(value["search_output_root"] for value in parent_payloads)
    if len(outputs) != len(set(outputs)):
        raise ValueError("full-search output roots are not unique")
    return {
        "schema": V5_K1_BALANCED_FULL_SEARCH_PLAN_SCHEMA,
        "version": V5_K1_BALANCED_FULL_SEARCH_PLAN_VERSION,
        "scientific_role": "all_k1_model_free_full_search_supervision_not_training_completion",
        "source": {
            "archive_sha256": source_archive,
            "bundle_sha256": source_bundle,
        },
        "identity_authorization": identity_authorization.to_payload(),
        "identity_authorization_sha256": identity_authorization.sha256,
        "balanced_dataset": {
            "launch_plan_file_sha256": _digest(
                balanced_dataset_launch_plan_file_sha256,
                "balanced_dataset_launch_plan_file_sha256",
            ),
            "launch_plan_sha256": _digest(
                balanced_dataset_launch_plan_sha256,
                "balanced_dataset_launch_plan_sha256",
            ),
            "completion_file_sha256": _digest(
                balanced_completion_file,
                "balanced_dataset_completion_file_sha256",
            ),
            "completion_sha256": _digest(
                balanced_completion,
                "balanced_dataset_completion_sha256",
            ),
        },
        "k1_stage": {
            "payload": stage,
            "stage_sha256": sha256(
                canonical_json(stage).encode("utf-8")
            ).hexdigest(),
        },
        "configuration": {
            "candidate_view_indices": list(
                V5_K1_BALANCED_FULL_SEARCH_CANDIDATE_VIEW_INDICES
            ),
            "observation_policy_id": V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
            "observation_policy_sha256": V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
            "one_curve_blind_sigma_present_view_per_clean_parent": True,
            "all_and_only_k1_topologies_per_parent": True,
            "every_feasible_wire_branch_per_topology": True,
        },
        "layout": {
            "run_root": root,
            "plan": str(PurePosixPath(root) / "k1-balanced-full-search-plan-v1.json"),
            "logs": str(PurePosixPath(root) / "logs"),
            "shards": str(PurePosixPath(root) / "shards"),
            "completion": str(PurePosixPath(root) / "audit" / "completion-v1.json"),
            "failure": str(PurePosixPath(root) / "audit" / "failure-v1.json"),
        },
        "array": {
            "task_count": len(rows),
            "array_spec": f"0-{len(rows) - 1}",
            "clean_parent_counts": observed_counts,
        },
        "parents": parent_payloads,
        "claim_limits": {
            "search_execution_authorized": True,
            "full_search_supervision_complete": False,
            "gradient_training_authorized": False,
            "tuning_exact_budget_summary_complete": False,
            "phase_c_model_acceptance": False,
            "paper_model_acceptance": False,
        },
    }


def build_v5_k1_balanced_full_search_plan(
    *,
    source_archive_sha256: str,
    source_bundle_sha256: str,
    balanced_dataset_launch_plan_file_sha256: str,
    balanced_dataset_launch_plan_sha256: str,
    balanced_dataset_completion_file_sha256: str,
    balanced_dataset_completion_sha256: str,
    identity_authorization: V5K1TrainingIdentityAuthorization,
    k1_stage: V5FormalProductionSearchStage,
    search_run_root: str,
    parents: Sequence[V5K1BalancedFullSearchParentBinding],
) -> dict[str, object]:
    """Build a full-search execution plan without reading curves or writing files."""

    if not isinstance(k1_stage, V5FormalProductionSearchStage):
        raise TypeError("k1_stage has an invalid type")
    core = _core(
        source_archive_sha256=source_archive_sha256,
        source_bundle_sha256=source_bundle_sha256,
        balanced_dataset_launch_plan_file_sha256=(
            balanced_dataset_launch_plan_file_sha256
        ),
        balanced_dataset_launch_plan_sha256=balanced_dataset_launch_plan_sha256,
        balanced_dataset_completion_file_sha256=(
            balanced_dataset_completion_file_sha256
        ),
        balanced_dataset_completion_sha256=balanced_dataset_completion_sha256,
        identity_authorization=identity_authorization,
        stage_payload=k1_stage.audit_payload(),
        search_run_root=search_run_root,
        parents=parents,
    )
    return {**core, "plan_sha256": sha256(canonical_json(core).encode()).hexdigest()}


def validate_v5_k1_balanced_full_search_plan(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Replay plan derivations and reject any relaxed completion claim."""

    if not isinstance(payload, Mapping):
        raise TypeError("balanced full-search plan must be an object")
    value = dict(payload)
    supplied = _digest(value.pop("plan_sha256", None), "plan_sha256")
    if supplied != sha256(canonical_json(value).encode()).hexdigest():
        raise ValueError("balanced full-search plan SHA-256 does not reproduce")
    if set(value) != {
        "schema",
        "version",
        "scientific_role",
        "source",
        "identity_authorization",
        "identity_authorization_sha256",
        "balanced_dataset",
        "k1_stage",
        "configuration",
        "layout",
        "array",
        "parents",
        "claim_limits",
    }:
        raise ValueError("balanced full-search plan fields are incomplete or unsupported")
    if (
        value["schema"] != V5_K1_BALANCED_FULL_SEARCH_PLAN_SCHEMA
        or value["version"] != V5_K1_BALANCED_FULL_SEARCH_PLAN_VERSION
        or value["claim_limits"]
        != {
            "search_execution_authorized": True,
            "full_search_supervision_complete": False,
            "gradient_training_authorized": False,
            "tuning_exact_budget_summary_complete": False,
            "phase_c_model_acceptance": False,
            "paper_model_acceptance": False,
        }
    ):
        raise ValueError("balanced full-search schema/version or claim limits drifted")
    authorization = V5K1TrainingIdentityAuthorization.from_payload(
        value["identity_authorization"]
    )
    if value["identity_authorization_sha256"] != authorization.sha256:
        raise ValueError("balanced full-search identity authorization hash drifted")
    stage = value["k1_stage"]
    if not isinstance(stage, Mapping) or set(stage) != {"payload", "stage_sha256"}:
        raise ValueError("balanced full-search K1 stage binding is incomplete")
    stage_payload = _validate_stage_payload(stage["payload"])
    if stage["stage_sha256"] != sha256(
        canonical_json(stage_payload).encode()
    ).hexdigest():
        raise ValueError("balanced full-search K1 stage hash drifted")
    layout = value["layout"]
    dataset = value["balanced_dataset"]
    source = value["source"]
    parents = tuple(
        V5K1BalancedFullSearchParentBinding.from_payload(row)
        for row in value["parents"]
    )
    replay = _core(
        source_archive_sha256=source["archive_sha256"],
        source_bundle_sha256=source["bundle_sha256"],
        balanced_dataset_launch_plan_file_sha256=dataset[
            "launch_plan_file_sha256"
        ],
        balanced_dataset_launch_plan_sha256=dataset["launch_plan_sha256"],
        balanced_dataset_completion_file_sha256=dataset["completion_file_sha256"],
        balanced_dataset_completion_sha256=dataset["completion_sha256"],
        identity_authorization=authorization,
        stage_payload=stage_payload,
        search_run_root=layout["run_root"],
        parents=parents,
    )
    if value != replay:
        raise ValueError("balanced full-search plan does not replay from its inputs")
    return {**value, "plan_sha256": supplied}


def write_v5_k1_balanced_full_search_plan(
    path: str | os.PathLike[str], payload: Mapping[str, object]
) -> Path:
    """Exclusively publish the validated scientific plan as 0400/nlink1."""

    plan = validate_v5_k1_balanced_full_search_plan(payload)
    target = Path(path)
    if not target.is_absolute() or str(target) != plan["layout"]["plan"]:
        raise ValueError("full-search plan path must equal its frozen layout path")
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite a balanced full-search plan")
    if not target.parent.is_dir():
        raise FileNotFoundError("balanced full-search plan parent does not exist")
    encoded = json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o400)
    metadata = target.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise RuntimeError("balanced full-search plan is not 0400/nlink1")
    return target


__all__ = [
    "MAXWELL_DUST_ROOT",
    "V5K1BalancedFullSearchParentBinding",
    "V5_K1_BALANCED_FULL_SEARCH_CANDIDATE_VIEW_INDICES",
    "V5_K1_BALANCED_FULL_SEARCH_PLAN_SCHEMA",
    "V5_K1_BALANCED_FULL_SEARCH_PLAN_VERSION",
    "V5_K1_BALANCED_FULL_SEARCH_TASK_COUNT",
    "build_v5_k1_balanced_full_search_plan",
    "validate_v5_k1_balanced_full_search_plan",
    "write_v5_k1_balanced_full_search_plan",
]
