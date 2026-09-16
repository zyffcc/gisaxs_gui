from __future__ import annotations

import copy

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_holdout_shard_v5 import (
    build_v5_k1_phase_c_holdout_shard_payload,
    plan_v5_k1_phase_c_holdout_shard,
    validate_v5_k1_phase_c_holdout_shard_payload,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_phase_c_plan_v5 import (
    build_v5_k1_phase_c_plan,
)


@pytest.fixture(scope="module")
def plan():
    return build_v5_k1_phase_c_plan(formal=True)


@pytest.mark.parametrize("branch_ordinal", range(12))
def test_every_k1_branch_materializes_as_branch_pure(plan, branch_ordinal):
    block = plan.sobol_blocks[branch_ordinal]
    shard = plan_v5_k1_phase_c_holdout_shard(
        phase_c_plan=plan,
        block_sha256=block.block_sha256,
        start=0,
        count=1,
    )
    payload = validate_v5_k1_phase_c_holdout_shard_payload(
        build_v5_k1_phase_c_holdout_shard_payload(shard)
    )
    assert payload["manifest"]["branch_id"] == block.branch_id
    assert payload["manifest"]["recipe_count"] == 1


def test_first_complete_stress_cycle_is_exactly_balanced(plan):
    block = plan.sobol_blocks[-1]
    shard = plan_v5_k1_phase_c_holdout_shard(
        phase_c_plan=plan,
        block_sha256=block.block_sha256,
        start=0,
        count=25,
    )
    payload = build_v5_k1_phase_c_holdout_shard_payload(shard)
    manifest = payload["manifest"]
    assert set(manifest["range_stress_counts"].values()) == {5}
    assert set(manifest["observation_stress_counts"].values()) == {5}
    assert {row[2] for row in manifest["stress_cell_counts"]} == {1}


def test_shard_validation_rejects_recipe_inventory_drift(plan):
    block = plan.sobol_blocks[0]
    shard = plan_v5_k1_phase_c_holdout_shard(
        phase_c_plan=plan,
        block_sha256=block.block_sha256,
        start=0,
        count=2,
    )
    payload = build_v5_k1_phase_c_holdout_shard_payload(shard)
    damaged = copy.deepcopy(payload)
    damaged["canonical_recipes"].reverse()
    core = dict(damaged)
    core.pop("artifact_self_sha256")
    from hashlib import sha256
    import json

    damaged["artifact_self_sha256"] = sha256(
        json.dumps(core, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(ValueError):
        validate_v5_k1_phase_c_holdout_shard_payload(damaged)


def test_shard_plan_rejects_nonformal_fixture():
    fixture = build_v5_k1_phase_c_plan(formal=False, parents_per_branch=25)
    with pytest.raises(ValueError, match="formal"):
        plan_v5_k1_phase_c_holdout_shard(
            phase_c_plan=fixture,
            block_sha256=fixture.sobol_blocks[0].block_sha256,
            start=0,
            count=1,
        )
