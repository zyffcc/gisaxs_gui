"""Replayable branch-balanced Sobol plan for the K1 Phase-C gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping

import numpy as np

from .k1_phase_c_contract_v5 import (
    K1_PHASE_C_BRANCHES,
    K1_PHASE_C_BRANCH_BY_ID,
    K1_PHASE_C_MASTER_SCRAMBLE_SEED,
    K1_PHASE_C_OBSERVATION_STRESS_STRATA,
    K1_PHASE_C_RANGE_STRESS_STRATA,
    K1_PHASE_C_SPLIT_ID,
    V5_K1_PHASE_C_SOBOL_BLOCK_VERSION,
    V5_K1_PHASE_C_STRESS_ASSIGNMENT_VERSION,
    V5K1PhaseCBranch,
    canonical_json,
    positive_integer,
    v5_k1_phase_c_contract_payload,
    validate_v5_k1_phase_c_contract,
)
from .sobol_design_v5 import V5_SOBOL_DESIGN_SCHEMA, V5_SOBOL_DESIGN_VERSION
from .sobol_recipe_coordinates_v5 import V5_SOBOL_RECIPE_COORDINATE_SHA256


V5_K1_PHASE_C_PLAN_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_sobol_plan/v1"
V5_K1_PHASE_C_PLAN_VERSION = "posterior_v8_k1_branch_specific_scramble_balanced_stress_cells_v1"
K1_PHASE_C_MIN_FIXTURE_PARENTS_PER_BRANCH = 25
_UINT32_MAX = (1 << 32) - 1


def _uint32(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= _UINT32_MAX:
        raise ValueError(f"{name} must fit in uint32")
    return result


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCSobolBlock:
    branch_id: str
    branch_ordinal: int
    split_id: str
    scramble_seed: int
    sobol_index_start: int
    parent_count: int
    sobol_design_sha256: str
    block_sha256: str

    def audit_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCPlan:
    master_scramble_seed: int
    formal: bool
    parents_per_branch: int
    total_parent_count: int
    contract_sha256: str
    sobol_blocks: tuple[V5K1PhaseCSobolBlock, ...]
    canonical_json: str
    sha256: str
    schema_version: str = V5_K1_PHASE_C_PLAN_SCHEMA
    version: str = V5_K1_PHASE_C_PLAN_VERSION

    def audit_payload(self) -> dict[str, object]:
        return json.loads(self.canonical_json)


def _branch_scramble_seed(master_seed: int, branch: V5K1PhaseCBranch) -> int:
    raw = sha256(
        f"{V5_K1_PHASE_C_SOBOL_BLOCK_VERSION}\0{master_seed}\0{branch.branch_id}".encode("ascii")
    ).digest()
    return int.from_bytes(raw[:4], "big")


def _derive_block(
    branch: V5K1PhaseCBranch,
    *,
    branch_ordinal: int,
    master_seed: int,
    parent_count: int,
) -> V5K1PhaseCSobolBlock:
    scramble_seed = _branch_scramble_seed(master_seed, branch)
    design_payload = {
        "sobol_schema": V5_SOBOL_DESIGN_SCHEMA,
        "sobol_version": V5_SOBOL_DESIGN_VERSION,
        "coordinate_contract_sha256": V5_SOBOL_RECIPE_COORDINATE_SHA256,
        "scramble_seed": scramble_seed,
        "bits": 52,
    }
    design_sha = sha256(canonical_json(design_payload).encode("utf-8")).hexdigest()
    block_core = {
        "version": V5_K1_PHASE_C_SOBOL_BLOCK_VERSION,
        "branch": branch.audit_payload(),
        "branch_ordinal": branch_ordinal,
        "split_id": K1_PHASE_C_SPLIT_ID,
        "master_scramble_seed": master_seed,
        "scramble_seed": scramble_seed,
        "sobol_index_start": 0,
        "parent_count": parent_count,
        "sobol_design_sha256": design_sha,
        "cross_branch_coordinate_reuse": False,
    }
    return V5K1PhaseCSobolBlock(
        branch_id=branch.branch_id,
        branch_ordinal=branch_ordinal,
        split_id=K1_PHASE_C_SPLIT_ID,
        scramble_seed=scramble_seed,
        sobol_index_start=0,
        parent_count=parent_count,
        sobol_design_sha256=design_sha,
        block_sha256=sha256(canonical_json(block_core).encode("utf-8")).hexdigest(),
    )


def build_v5_k1_phase_c_plan(
    *,
    formal: bool = True,
    parents_per_branch: int | None = None,
    master_scramble_seed: int = K1_PHASE_C_MASTER_SCRAMBLE_SEED,
) -> V5K1PhaseCPlan:
    """Build the formal plan or a non-claiming >=25-parent-per-branch fixture."""

    if type(formal) is not bool:
        raise TypeError("formal must be a bool")
    seed = _uint32(master_scramble_seed, "master_scramble_seed")
    contract = v5_k1_phase_c_contract_payload()
    formal_count = int(contract["population"]["parents_per_generating_branch"])
    count = (
        formal_count
        if parents_per_branch is None
        else positive_integer(parents_per_branch, "parents_per_branch")
    )
    if formal and count != formal_count:
        raise ValueError("formal K1 Phase-C plan must use the frozen parent count")
    if not formal and count < K1_PHASE_C_MIN_FIXTURE_PARENTS_PER_BRANCH:
        raise ValueError("fixture plan needs at least 25 parents per branch for all stress cells")
    blocks = tuple(
        _derive_block(
            branch,
            branch_ordinal=index,
            master_seed=seed,
            parent_count=count,
        )
        for index, branch in enumerate(K1_PHASE_C_BRANCHES)
    )
    if len({value.scramble_seed for value in blocks}) != len(blocks):
        raise RuntimeError("branch-specific Sobol scramble seeds collided")
    if len({value.sobol_design_sha256 for value in blocks}) != len(blocks):
        raise RuntimeError("branch-specific Sobol designs are not independent")
    core = {
        "schema": V5_K1_PHASE_C_PLAN_SCHEMA,
        "version": V5_K1_PHASE_C_PLAN_VERSION,
        "contract_sha256": contract["contract_sha256"],
        "master_scramble_seed": seed,
        "formal": formal,
        "parents_per_branch": count,
        "total_parent_count": count * len(blocks),
        "stress_assignment_version": V5_K1_PHASE_C_STRESS_ASSIGNMENT_VERSION,
        "sobol_blocks": [value.audit_payload() for value in blocks],
    }
    encoded = canonical_json(core)
    result = V5K1PhaseCPlan(
        master_scramble_seed=seed,
        formal=formal,
        parents_per_branch=count,
        total_parent_count=count * len(blocks),
        contract_sha256=str(contract["contract_sha256"]),
        sobol_blocks=blocks,
        canonical_json=encoded,
        sha256=sha256(encoded.encode("utf-8")).hexdigest(),
    )
    return validate_v5_k1_phase_c_plan(result, contract=contract)


def validate_v5_k1_phase_c_plan(
    plan: V5K1PhaseCPlan,
    *,
    contract: Mapping[str, object] | None = None,
) -> V5K1PhaseCPlan:
    if not isinstance(plan, V5K1PhaseCPlan):
        raise TypeError("plan must be a V5K1PhaseCPlan")
    live = validate_v5_k1_phase_c_contract(
        v5_k1_phase_c_contract_payload() if contract is None else contract
    )
    if (
        plan.schema_version != V5_K1_PHASE_C_PLAN_SCHEMA
        or plan.version != V5_K1_PHASE_C_PLAN_VERSION
    ):
        raise ValueError("unsupported K1 Phase-C plan schema/version")
    if type(plan.formal) is not bool:
        raise TypeError("plan.formal must be a bool")
    seed = _uint32(plan.master_scramble_seed, "master_scramble_seed")
    count = positive_integer(plan.parents_per_branch, "parents_per_branch")
    formal_count = int(live["population"]["parents_per_generating_branch"])
    if plan.formal and count != formal_count:
        raise ValueError("formal K1 Phase-C plan has the wrong parent count")
    if not plan.formal and count < K1_PHASE_C_MIN_FIXTURE_PARENTS_PER_BRANCH:
        raise ValueError("fixture plan cannot cover all 25 stress cells")
    expected_blocks = tuple(
        _derive_block(
            branch,
            branch_ordinal=index,
            master_seed=seed,
            parent_count=count,
        )
        for index, branch in enumerate(K1_PHASE_C_BRANCHES)
    )
    if plan.sobol_blocks != expected_blocks:
        raise ValueError("K1 Phase-C Sobol blocks do not replay")
    core = {
        "schema": V5_K1_PHASE_C_PLAN_SCHEMA,
        "version": V5_K1_PHASE_C_PLAN_VERSION,
        "contract_sha256": live["contract_sha256"],
        "master_scramble_seed": seed,
        "formal": plan.formal,
        "parents_per_branch": count,
        "total_parent_count": count * len(expected_blocks),
        "stress_assignment_version": V5_K1_PHASE_C_STRESS_ASSIGNMENT_VERSION,
        "sobol_blocks": [value.audit_payload() for value in expected_blocks],
    }
    encoded = canonical_json(core)
    expected = (
        count * len(expected_blocks),
        live["contract_sha256"],
        encoded,
        sha256(encoded.encode("utf-8")).hexdigest(),
    )
    actual = (plan.total_parent_count, plan.contract_sha256, plan.canonical_json, plan.sha256)
    if actual != expected:
        raise ValueError("K1 Phase-C plan identity does not reproduce")
    return plan


def planned_v5_k1_phase_c_stress_cell(
    plan: V5K1PhaseCPlan,
    *,
    branch_id: str,
    sobol_index: int,
) -> tuple[str, str]:
    """Return the balanced 5x5 stress cell for one branch-local Sobol index."""

    validate_v5_k1_phase_c_plan(plan)
    if branch_id not in K1_PHASE_C_BRANCH_BY_ID:
        raise ValueError("branch_id is not in the frozen K1 Phase-C catalog")
    block = next(value for value in plan.sobol_blocks if value.branch_id == branch_id)
    if isinstance(sobol_index, (bool, np.bool_)) or not isinstance(sobol_index, Integral):
        raise TypeError("sobol_index must be an integer")
    index = int(sobol_index)
    if index < 0:
        raise ValueError("sobol_index must be non-negative")
    stop = block.sobol_index_start + block.parent_count
    if not block.sobol_index_start <= index < stop:
        raise ValueError("sobol_index is outside its K1 Phase-C block")
    offset = index - block.sobol_index_start
    return (
        K1_PHASE_C_RANGE_STRESS_STRATA[offset % len(K1_PHASE_C_RANGE_STRESS_STRATA)],
        K1_PHASE_C_OBSERVATION_STRESS_STRATA[
            (offset // len(K1_PHASE_C_RANGE_STRESS_STRATA))
            % len(K1_PHASE_C_OBSERVATION_STRESS_STRATA)
        ],
    )


__all__ = [
    "K1_PHASE_C_MIN_FIXTURE_PARENTS_PER_BRANCH",
    "V5_K1_PHASE_C_PLAN_SCHEMA",
    "V5_K1_PHASE_C_PLAN_VERSION",
    "V5K1PhaseCPlan",
    "V5K1PhaseCSobolBlock",
    "build_v5_k1_phase_c_plan",
    "planned_v5_k1_phase_c_stress_cell",
    "validate_v5_k1_phase_c_plan",
]
