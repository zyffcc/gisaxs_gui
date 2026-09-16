from dataclasses import replace
import json

import pytest

from PosteriorV8.clean_recipe_forward_v5 import validate_v5_clean_recipe_like
from PosteriorV8.k1_balanced_dataset_plan_v5 import (
    build_frozen_v5_k1_balanced_dataset_plan,
    build_v5_k1_balanced_dataset_plan,
)
from PosteriorV8.k1_forced_sobol_recipe_v5 import (
    V5K1ForcedSobolCleanRecipe,
    decode_v5_k1_forced_recipe_identity,
    persisted_v5_k1_forced_clean_recipe_from_json,
)
from PosteriorV8.k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES
from PosteriorV8.sobol_design_v5 import materialize_v5_unit_coordinates_for_indices
from PosteriorV8.sobol_recipe_coordinates_v5 import v5_sobol_recipe_design


def _plan():
    return build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=11,
        tuning_master_scramble_seed=29,
        train_parents_per_branch=2,
        tuning_parents_per_branch=1,
    )


def _recipe(block_index: int = 0, sobol_index: int = 0):
    plan = _plan()
    block = plan.blocks[block_index]
    design = v5_sobol_recipe_design(scramble_seed=block.scramble_seed)
    coordinates = materialize_v5_unit_coordinates_for_indices(design, (sobol_index,))[0]
    return V5K1ForcedSobolCleanRecipe.create(
        plan=plan,
        block=block,
        sobol_index=sobol_index,
        original_unit_coordinates=coordinates,
    )


@pytest.mark.parametrize("block_index", range(24))
def test_forced_recipe_carries_truthful_raw_and_transformed_provenance(block_index):
    recipe = validate_v5_clean_recipe_like(_recipe(block_index))
    payload = json.loads(recipe.canonical_json)
    branch = K1_PHASE_C_BRANCHES[block_index % 12]

    assert payload["source"]["generating_branch_id"] == branch.branch_id
    assert payload["source"]["original_unit_coordinates"] == list(
        recipe.forcing.original_coordinates
    )
    assert payload["source"]["forced_unit_coordinates"] == list(
        recipe.forcing.forced_coordinates
    )
    assert recipe.query.topology == (branch.shape,)
    assert recipe.target.pattern_id == branch.pattern_id
    assert recipe.query.feasible_wire_pattern_ids == (branch.pattern_id,)


def test_recipe_observation_seed_and_group_id_are_split_and_branch_specific():
    plan = _plan()
    recipes = []
    for block in plan.blocks:
        design = v5_sobol_recipe_design(scramble_seed=block.scramble_seed)
        coordinates = materialize_v5_unit_coordinates_for_indices(design, (0,))[0]
        recipes.append(
            V5K1ForcedSobolCleanRecipe.create(
                plan=plan,
                block=block,
                sobol_index=0,
                original_unit_coordinates=coordinates,
            )
        )

    assert len({value.clean_group_id for value in recipes}) == 24
    assert len({value.recipe_seed for value in recipes}) == 24


def test_recipe_rejects_wrong_block_index_and_identity_tampering():
    recipe = _recipe()
    with pytest.raises(ValueError, match="outside"):
        V5K1ForcedSobolCleanRecipe.create(
            plan=recipe.plan,
            block=recipe.block,
            sobol_index=recipe.block.parent_count,
            original_unit_coordinates=recipe.forcing.original_coordinates,
        )
    with pytest.raises(ValueError, match="identity"):
        replace(recipe, sha256="0" * 64)
    wrong_point = list(recipe.forcing.original_coordinates)
    wrong_point[5] = 0.25 if wrong_point[5] != 0.25 else 0.75
    with pytest.raises(ValueError, match="design/index"):
        V5K1ForcedSobolCleanRecipe.create(
            plan=recipe.plan,
            block=recipe.block,
            sobol_index=recipe.sobol_index,
            original_unit_coordinates=wrong_point,
        )


def test_persisted_identity_decoder_replays_nested_recipe_contracts():
    recipe = _recipe(block_index=17, sobol_index=0)

    identity = decode_v5_k1_forced_recipe_identity(
        recipe.canonical_json,
        expected_sha256=recipe.sha256,
    )

    assert identity.branch_id == recipe.block.branch_id
    assert identity.clean_group_id == recipe.clean_group_id
    assert identity.balanced_dataset_plan_sha256 == recipe.plan.sha256
    assert identity.balanced_sobol_block_sha256 == recipe.block.block_sha256


def test_persisted_full_decoder_recovers_artifact_authoritative_clean_recipe():
    recipe = _recipe(block_index=17, sobol_index=0)

    replay = persisted_v5_k1_forced_clean_recipe_from_json(
        recipe.canonical_json,
        recipe.sha256,
    )

    assert validate_v5_clean_recipe_like(replay) is replay
    assert replay.identity.branch_id == recipe.block.branch_id
    assert replay.clean_group_id == recipe.clean_group_id
    assert replay.assigned_split == recipe.assigned_split
    assert replay.recipe_seed == recipe.recipe_seed
    assert replay.physics == recipe.physics
    assert replay.grid == recipe.grid


def test_persisted_identity_decoder_rejects_nested_forcing_drift():
    recipe = _recipe()
    payload = json.loads(recipe.canonical_json)
    payload["source"]["forced_unit_coordinates"][0] = 0.125
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

    with pytest.raises(ValueError, match="branch transform"):
        decode_v5_k1_forced_recipe_identity(encoded)


@pytest.mark.parametrize(
    ("branch_ordinal", "sobol_index"),
    ((6, 42), (7, 2)),
)
def test_forced_recipe_coupled_boundary_has_a_replayable_varying_mask(
    branch_ordinal,
    sobol_index,
):
    plan = build_frozen_v5_k1_balanced_dataset_plan()
    block = next(
        value
        for value in plan.blocks
        if value.role == "train" and value.branch_ordinal == branch_ordinal
    )
    design = v5_sobol_recipe_design(scramble_seed=block.scramble_seed)
    coordinates = materialize_v5_unit_coordinates_for_indices(
        design,
        (sobol_index,),
    )[0]
    recipe = V5K1ForcedSobolCleanRecipe.create(
        plan=plan,
        block=block,
        sobol_index=sobol_index,
        original_unit_coordinates=coordinates,
    )

    codec = recipe.query.codec_for(recipe.target.pattern_id)

    assert any(codec.varying_mask)
    assert codec.encode(*codec.decode(recipe.target.local_target_unit)).unit_cube == (
        recipe.target.local_target_unit
    )
