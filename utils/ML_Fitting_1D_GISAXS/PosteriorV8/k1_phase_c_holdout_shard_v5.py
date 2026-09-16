"""Immutable recipe-only shards for the formal K1 Phase-C holdout."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path
import socket
import stat
from typing import Mapping, Sequence

import numpy as np

from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_OBSERVATION_STRESS_STRATA,
    K1_PHASE_C_RANGE_STRESS_STRATA,
)
from .k1_phase_c_holdout_recipe_v5 import (
    V5K1PhaseCHoldoutCleanRecipe,
    decode_v5_k1_phase_c_holdout_recipe_identity,
    materialize_v5_k1_phase_c_holdout_recipe,
)
from .k1_phase_c_plan_v5 import (
    V5K1PhaseCPlan,
    V5K1PhaseCSobolBlock,
    validate_v5_k1_phase_c_plan,
)
from .sobol_design_v5 import materialize_v5_unit_coordinates_for_indices
from .sobol_recipe_coordinates_v5 import v5_sobol_recipe_design


V5_K1_PHASE_C_HOLDOUT_SHARD_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_c_holdout_recipe_shard/v1"
)
V5_K1_PHASE_C_HOLDOUT_SHARD_VERSION = (
    "posterior_v8_v5_2_branch_pure_stress_bound_recipe_only_shard_v1"
)
MAXWELL_DUST_ROOT = Path("/data/dust/user/zhaiyufe")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _positive_integer(value: object, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _under_root(path: Path, allowed_root: Path) -> Path:
    if not path.is_absolute():
        raise ValueError("Phase-C shard output must be absolute")
    root = allowed_root.resolve(strict=True)
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Phase-C shard output must be under {root}") from exc
    if not relative.parts:
        raise ValueError("Phase-C shard output must not be the allowed root")
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("Phase-C shard output must not traverse a symlink")
        if not current.exists():
            break
    if not lexical.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("Phase-C shard parent resolves outside the allowed root")
    return lexical


@dataclass(frozen=True)
class V5K1PhaseCHoldoutShardPlan:
    phase_c_plan: V5K1PhaseCPlan
    block: V5K1PhaseCSobolBlock
    split_offset: int
    requested_count: int
    selected_indices: tuple[int, ...]
    selection_sha256: str

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": V5_K1_PHASE_C_HOLDOUT_SHARD_SCHEMA,
            "version": V5_K1_PHASE_C_HOLDOUT_SHARD_VERSION,
            "phase_c_plan_sha256": self.phase_c_plan.sha256,
            "phase_c_sobol_block": self.block.audit_payload(),
            "split_offset": self.split_offset,
            "requested_count": self.requested_count,
            "selected_sobol_indices": list(self.selected_indices),
            "selection_sha256": self.selection_sha256,
        }


def plan_v5_k1_phase_c_holdout_shard(
    *,
    phase_c_plan: V5K1PhaseCPlan,
    block_sha256: str,
    start: int,
    count: int,
) -> V5K1PhaseCHoldoutShardPlan:
    plan = validate_v5_k1_phase_c_plan(phase_c_plan)
    if not plan.formal:
        raise ValueError("production holdout shards require the formal Phase-C plan")
    matches = tuple(value for value in plan.sobol_blocks if value.block_sha256 == block_sha256)
    if len(matches) != 1:
        raise ValueError("block_sha256 must select one Phase-C block")
    offset = _nonnegative_integer(start, "start")
    selected_count = _positive_integer(count, "count")
    block = matches[0]
    if offset >= block.parent_count:
        raise ValueError("Phase-C shard starts beyond its branch block")
    stop = min(offset + selected_count, block.parent_count)
    indices = tuple(range(block.sobol_index_start + offset, block.sobol_index_start + stop))
    core = {
        "phase_c_plan_sha256": plan.sha256,
        "phase_c_sobol_block_sha256": block.block_sha256,
        "split_offset": offset,
        "requested_count": selected_count,
        "selected_sobol_indices": list(indices),
    }
    return V5K1PhaseCHoldoutShardPlan(
        phase_c_plan=plan,
        block=block,
        split_offset=offset,
        requested_count=selected_count,
        selected_indices=indices,
        selection_sha256=sha256(_canonical_json(core).encode()).hexdigest(),
    )


def materialize_v5_k1_phase_c_holdout_shard(
    shard_plan: V5K1PhaseCHoldoutShardPlan,
) -> tuple[V5K1PhaseCHoldoutCleanRecipe, ...]:
    if not isinstance(shard_plan, V5K1PhaseCHoldoutShardPlan):
        raise TypeError("shard_plan must be V5K1PhaseCHoldoutShardPlan")
    replay = plan_v5_k1_phase_c_holdout_shard(
        phase_c_plan=shard_plan.phase_c_plan,
        block_sha256=shard_plan.block.block_sha256,
        start=shard_plan.split_offset,
        count=shard_plan.requested_count,
    )
    if replay != shard_plan:
        raise ValueError("Phase-C holdout shard plan does not reproduce")
    design = v5_sobol_recipe_design(scramble_seed=shard_plan.block.scramble_seed)
    points = materialize_v5_unit_coordinates_for_indices(design, shard_plan.selected_indices)
    return tuple(
        materialize_v5_k1_phase_c_holdout_recipe(
            plan=shard_plan.phase_c_plan,
            block=shard_plan.block,
            sobol_index=index,
            original_unit_coordinates=point,
        )
        for index, point in zip(shard_plan.selected_indices, points, strict=True)
    )


def build_v5_k1_phase_c_holdout_shard_payload(
    shard_plan: V5K1PhaseCHoldoutShardPlan,
) -> dict[str, object]:
    recipes = materialize_v5_k1_phase_c_holdout_shard(shard_plan)
    identities = tuple(
        decode_v5_k1_phase_c_holdout_recipe_identity(
            value.canonical_json, expected_sha256=value.sha256
        )
        for value in recipes
    )
    range_counts = {
        name: sum(value.range_stress_stratum == name for value in identities)
        for name in K1_PHASE_C_RANGE_STRESS_STRATA
    }
    observation_counts = {
        name: sum(value.observation_stress_stratum == name for value in identities)
        for name in K1_PHASE_C_OBSERVATION_STRESS_STRATA
    }
    stress_cell_counts = [
        [
            range_name,
            observation_name,
            sum(
                value.range_stress_stratum == range_name
                and value.observation_stress_stratum == observation_name
                for value in identities
            ),
        ]
        for observation_name in K1_PHASE_C_OBSERVATION_STRESS_STRATA
        for range_name in K1_PHASE_C_RANGE_STRESS_STRATA
    ]
    manifest_core = {
        "schema": V5_K1_PHASE_C_HOLDOUT_SHARD_SCHEMA,
        "version": V5_K1_PHASE_C_HOLDOUT_SHARD_VERSION,
        "phase_c_plan_sha256": shard_plan.phase_c_plan.sha256,
        "phase_c_sobol_block_sha256": shard_plan.block.block_sha256,
        "branch_id": shard_plan.block.branch_id,
        "branch_ordinal": shard_plan.block.branch_ordinal,
        "selection_sha256": shard_plan.selection_sha256,
        "selected_sobol_indices": list(shard_plan.selected_indices),
        "recipe_sha256s": [value.recipe_sha256 for value in identities],
        "clean_group_ids": [value.clean_group_id for value in identities],
        "range_stress_counts": range_counts,
        "observation_stress_counts": observation_counts,
        "stress_cell_counts": stress_cell_counts,
        "recipe_count": len(identities),
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": sha256(_canonical_json(manifest_core).encode()).hexdigest(),
    }
    core = {
        "manifest": manifest,
        "canonical_recipes": [value.canonical_json for value in recipes],
        "scientific_acceptance_evidence": False,
        "training_authorization_granted": False,
    }
    return {**core, "artifact_self_sha256": sha256(_canonical_json(core).encode()).hexdigest()}


def validate_v5_k1_phase_c_holdout_shard_payload(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("Phase-C holdout shard must be an object")
    value = dict(payload)
    supplied = value.pop("artifact_self_sha256", None)
    if supplied != sha256(_canonical_json(value).encode()).hexdigest():
        raise ValueError("Phase-C holdout artifact self-hash does not reproduce")
    if set(value) != {
        "manifest",
        "canonical_recipes",
        "scientific_acceptance_evidence",
        "training_authorization_granted",
    }:
        raise ValueError("Phase-C holdout shard fields are unsupported")
    if value["scientific_acceptance_evidence"] is not False or value[
        "training_authorization_granted"
    ] is not False:
        raise ValueError("Phase-C holdout data cannot claim model authorization")
    manifest = value["manifest"]
    if not isinstance(manifest, Mapping):
        raise ValueError("Phase-C holdout manifest is missing")
    manifest_core = dict(manifest)
    manifest_sha = manifest_core.pop("manifest_sha256", None)
    if manifest_sha != sha256(_canonical_json(manifest_core).encode()).hexdigest():
        raise ValueError("Phase-C holdout manifest SHA-256 does not reproduce")
    recipes = value["canonical_recipes"]
    if not isinstance(recipes, list) or len(recipes) != manifest["recipe_count"]:
        raise ValueError("Phase-C holdout recipe count drifted")
    identities = tuple(
        decode_v5_k1_phase_c_holdout_recipe_identity(
            encoded, expected_sha256=expected_sha
        )
        for encoded, expected_sha in zip(
            recipes, manifest["recipe_sha256s"], strict=True
        )
    )
    if [value.clean_group_id for value in identities] != manifest["clean_group_ids"]:
        raise ValueError("Phase-C holdout clean-group inventory drifted")
    if [value.sobol_index for value in identities] != manifest["selected_sobol_indices"]:
        raise ValueError("Phase-C holdout Sobol index inventory drifted")
    if {value.phase_c_plan_sha256 for value in identities} != {
        manifest["phase_c_plan_sha256"]
    } or {value.phase_c_sobol_block_sha256 for value in identities} != {
        manifest["phase_c_sobol_block_sha256"]
    }:
        raise ValueError("Phase-C holdout plan/block binding drifted")
    if {value.branch_id for value in identities} != {manifest["branch_id"]}:
        raise ValueError("Phase-C holdout shard is not branch-pure")
    observed_range = {
        name: sum(value.range_stress_stratum == name for value in identities)
        for name in K1_PHASE_C_RANGE_STRESS_STRATA
    }
    observed_observation = {
        name: sum(value.observation_stress_stratum == name for value in identities)
        for name in K1_PHASE_C_OBSERVATION_STRESS_STRATA
    }
    observed_cells = [
        [
            range_name,
            observation_name,
            sum(
                value.range_stress_stratum == range_name
                and value.observation_stress_stratum == observation_name
                for value in identities
            ),
        ]
        for observation_name in K1_PHASE_C_OBSERVATION_STRESS_STRATA
        for range_name in K1_PHASE_C_RANGE_STRESS_STRATA
    ]
    if observed_range != manifest["range_stress_counts"] or observed_observation != manifest[
        "observation_stress_counts"
    ] or observed_cells != manifest["stress_cell_counts"]:
        raise ValueError("Phase-C holdout stress counts drifted")
    return {**value, "artifact_self_sha256": supplied}


def write_v5_k1_phase_c_holdout_shard(
    shard_plan: V5K1PhaseCHoldoutShardPlan,
    path: str | os.PathLike[str],
    *,
    allowed_root: Path = MAXWELL_DUST_ROOT,
) -> dict[str, object]:
    if socket.gethostname().split(".", 1)[0].startswith("max-wgs"):
        raise RuntimeError("Phase-C holdout materialization is forbidden on max-wgs")
    target = _under_root(Path(path), allowed_root)
    if target.exists() or target.is_symlink():
        raise FileExistsError("refusing to overwrite a Phase-C holdout shard")
    payload = validate_v5_k1_phase_c_holdout_shard_payload(
        build_v5_k1_phase_c_holdout_shard_payload(shard_plan)
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    target.chmod(0o400)
    metadata = target.stat()
    if stat.S_IMODE(metadata.st_mode) != 0o400 or metadata.st_nlink != 1:
        raise RuntimeError("Phase-C holdout shard is not 0400/nlink1")
    return payload


__all__ = [
    "MAXWELL_DUST_ROOT",
    "V5_K1_PHASE_C_HOLDOUT_SHARD_SCHEMA",
    "V5_K1_PHASE_C_HOLDOUT_SHARD_VERSION",
    "V5K1PhaseCHoldoutShardPlan",
    "build_v5_k1_phase_c_holdout_shard_payload",
    "materialize_v5_k1_phase_c_holdout_shard",
    "plan_v5_k1_phase_c_holdout_shard",
    "validate_v5_k1_phase_c_holdout_shard_payload",
    "write_v5_k1_phase_c_holdout_shard",
]
