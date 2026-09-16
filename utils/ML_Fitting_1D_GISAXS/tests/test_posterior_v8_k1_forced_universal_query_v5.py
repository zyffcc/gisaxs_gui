from dataclasses import replace
import json

import pytest

from PosteriorV8.formal_production_search_contract_v5 import (
    topology_ids_for_v5_formal_production_stage,
)
from PosteriorV8.k1_balanced_dataset_plan_v5 import (
    build_v5_k1_balanced_dataset_plan,
)
from PosteriorV8.k1_forced_sobol_recipe_v5 import V5K1ForcedSobolCleanRecipe
from PosteriorV8.k1_forced_universal_query_v5 import (
    materialize_v5_k1_forced_universal_query_set,
)
from PosteriorV8.k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES
from PosteriorV8.sobol_design_v5 import materialize_v5_unit_coordinates_for_indices
from PosteriorV8.sobol_recipe_coordinates_v5 import v5_sobol_recipe_design


def _recipe(branch_ordinal: int) -> V5K1ForcedSobolCleanRecipe:
    plan = build_v5_k1_balanced_dataset_plan(
        train_master_scramble_seed=11,
        tuning_master_scramble_seed=29,
        train_parents_per_branch=2,
        tuning_parents_per_branch=1,
    )
    block = next(
        value
        for value in plan.blocks
        if value.role == "train" and value.branch_ordinal == branch_ordinal
    )
    design = v5_sobol_recipe_design(scramble_seed=block.scramble_seed)
    coordinates = materialize_v5_unit_coordinates_for_indices(design, (0,))[0]
    return V5K1ForcedSobolCleanRecipe.create(
        plan=plan,
        block=block,
        sobol_index=0,
        original_unit_coordinates=coordinates,
    )


@pytest.mark.parametrize("branch_ordinal", range(12))
def test_forced_query_set_replays_every_k1_topology_and_the_generating_query(
    branch_ordinal,
):
    recipe = _recipe(branch_ordinal)

    query_set = materialize_v5_k1_forced_universal_query_set(
        recipe.canonical_json,
        expected_recipe_sha256=recipe.sha256,
    )

    assert query_set.selected_topology_ids == (
        topology_ids_for_v5_formal_production_stage("K1")
    )
    generating = next(
        value
        for value in query_set.topology_queries
        if value.topology_id == query_set.generating_topology_id
    )
    branch = K1_PHASE_C_BRANCHES[branch_ordinal]
    assert generating.geometry == recipe.query
    assert generating.amplitude == recipe.amplitude_query
    assert generating.feasible_wire_pattern_ids == (branch.pattern_id,)
    assert query_set.audit_payload()["training_authorization_granted"] is False


def test_forced_query_set_rejects_a_recipe_payload_drift():
    recipe = _recipe(0)
    payload = json.loads(recipe.canonical_json)
    payload["source"]["forced_unit_coordinates"][0] = 0.125
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))

    with pytest.raises(ValueError, match="branch transform"):
        materialize_v5_k1_forced_universal_query_set(encoded)


def test_forced_query_set_is_immutable_and_self_replaying():
    recipe = _recipe(3)
    query_set = materialize_v5_k1_forced_universal_query_set(recipe.canonical_json)

    with pytest.raises(ValueError, match="does not reproduce"):
        replace(query_set, sha256="0" * 64)
