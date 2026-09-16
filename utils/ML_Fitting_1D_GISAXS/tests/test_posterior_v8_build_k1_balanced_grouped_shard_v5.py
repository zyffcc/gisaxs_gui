from __future__ import annotations

import pytest

from PosteriorV8.build_k1_balanced_grouped_shard_v5 import (
    materialize_v5_k1_grouped_recipe_specs,
    plan_v5_k1_balanced_grouped_shard,
)
from PosteriorV8.k1_balanced_dataset_plan_v5 import build_v5_k1_balanced_dataset_plan
from PosteriorV8.k1_forced_sobol_recipe_v5 import decode_v5_k1_forced_recipe_identity


def _dataset_plan():
    return build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=101,
        tuning_master_scramble_seed=303,
        train_parents_per_branch=5,
        tuning_parents_per_branch=3,
    )


def test_branch_local_shard_plan_and_specs_replay_exactly():
    dataset_plan = _dataset_plan()
    block = dataset_plan.blocks[7]
    shard = plan_v5_k1_balanced_grouped_shard(
        dataset_plan=dataset_plan,
        block_sha256=block.block_sha256,
        start=1,
        count=3,
        view_indices=(0, 2),
    )

    specs = materialize_v5_k1_grouped_recipe_specs(shard)

    assert shard.selected_indices == (1, 2, 3)
    assert len(specs) == 3
    assert {value.split_id for value in specs} == {block.split_id}
    assert {value.sobol_design_sha256 for value in specs} == {
        block.sobol_design_sha256
    }
    for spec in specs:
        identity = decode_v5_k1_forced_recipe_identity(
            spec.recipe.canonical_json,
            expected_sha256=spec.recipe.sha256,
        )
        assert identity.branch_id == block.branch_id
        assert identity.balanced_sobol_block_sha256 == block.block_sha256
        assert identity.clean_group_id == spec.clean_group_id


def test_shard_index_truncates_only_at_branch_boundary():
    dataset_plan = _dataset_plan()
    block = dataset_plan.blocks[0]

    shard = plan_v5_k1_balanced_grouped_shard(
        dataset_plan=dataset_plan,
        block_sha256=block.block_sha256,
        shard_index=1,
        count=3,
    )

    assert shard.split_offset == 3
    assert shard.selected_indices == (3, 4)


def test_shard_plan_rejects_ambiguous_or_out_of_range_selection():
    dataset_plan = _dataset_plan()
    block = dataset_plan.blocks[0]
    with pytest.raises(ValueError, match="exactly one"):
        plan_v5_k1_balanced_grouped_shard(
            dataset_plan=dataset_plan,
            block_sha256="0" * 64,
            start=0,
            count=1,
        )
    with pytest.raises(ValueError, match="exactly one of"):
        plan_v5_k1_balanced_grouped_shard(
            dataset_plan=dataset_plan,
            block_sha256=block.block_sha256,
            start=0,
            shard_index=0,
            count=1,
        )
    with pytest.raises(ValueError, match="beyond"):
        plan_v5_k1_balanced_grouped_shard(
            dataset_plan=dataset_plan,
            block_sha256=block.block_sha256,
            start=block.parent_count,
            count=1,
        )
