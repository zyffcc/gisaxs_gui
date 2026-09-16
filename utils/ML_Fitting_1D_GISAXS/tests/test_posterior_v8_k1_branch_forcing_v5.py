from dataclasses import replace

import numpy as np
import pytest

from PosteriorV8.contextual_branch_catalog import PRESENCE_POLICIES
from PosteriorV8.contract import NUM_TOPOLOGIES
from PosteriorV8.k1_branch_forcing_v5 import (
    V5_K1_BRANCH_FORCED_COORDINATE_NAMES,
    V5K1ForcedSobolCoordinates,
    force_v5_k1_branch_coordinates,
    v5_k1_branch_forcing_contract,
)
from PosteriorV8.k1_phase_c_contract_v5 import K1_PHASE_C_BRANCHES
from PosteriorV8.sobol_recipe_coordinates_v5 import (
    V5_SOBOL_RECIPE_COORDINATE_INDEX,
    V5_SOBOL_RECIPE_DIM,
)
from PosteriorV8.sobol_recipe_physics_v5 import direct_v5_physics_from_sobol


def _base_coordinates() -> tuple[float, ...]:
    return tuple(float(value) for value in np.linspace(0.01, 0.99, V5_SOBOL_RECIPE_DIM))


def test_forcing_contract_is_self_bound_and_names_only_discrete_selectors():
    contract = v5_k1_branch_forcing_contract()

    assert len(contract["contract_sha256"]) == 64
    assert tuple(contract["forced_coordinate_names"]) == (
        V5_K1_BRANCH_FORCED_COORDINATE_NAMES
    )
    assert contract["continuous_coordinate_policy"] == (
        "preserve_every_non_forced_coordinate_bit_exactly"
    )


@pytest.mark.parametrize("branch", K1_PHASE_C_BRANCHES, ids=lambda value: value.branch_id)
def test_all_twelve_forced_blocks_materialize_the_exact_requested_branch(branch):
    original = _base_coordinates()
    result = force_v5_k1_branch_coordinates(original, branch_id=branch.branch_id)
    physics = direct_v5_physics_from_sobol(result.forced_coordinates, sobol_index=17)

    assert physics.query.topology == (branch.shape,)
    assert physics.target.pattern_id == branch.pattern_id
    assert physics.query.feasible_wire_pattern_ids == (branch.pattern_id,)
    assert (physics.query.component_bounds[0].D is not None) is branch.d_present
    assert (physics.query.resolution_bounds is not None) is branch.resolution_present
    assert result.audit_payload()["branch"] == branch.audit_payload()

    forced_indices = {
        V5_SOBOL_RECIPE_COORDINATE_INDEX[name]
        for name in V5_K1_BRANCH_FORCED_COORDINATE_NAMES
    }
    assert all(
        result.forced_coordinates[index] == value
        for index, value in enumerate(original)
        if index not in forced_indices
    )
    assert result.forced_coordinates[
        V5_SOBOL_RECIPE_COORDINATE_INDEX["discrete.topology"]
    ] == (branch.topology_id + 0.5) / NUM_TOPOLOGIES
    expected_policy = "required" if branch.d_present else "absent"
    assert result.forced_coordinates[
        V5_SOBOL_RECIPE_COORDINATE_INDEX["geometry.slot_1.D_policy"]
    ] == (PRESENCE_POLICIES.index(expected_policy) + 0.5) / len(PRESENCE_POLICIES)


def test_forcing_replays_and_tampering_fails_closed():
    branch = K1_PHASE_C_BRANCHES[0]
    result = force_v5_k1_branch_coordinates(_base_coordinates(), branch_id=branch.branch_id)

    assert V5K1ForcedSobolCoordinates(**result.__dict__) == result
    with pytest.raises(ValueError, match="do not replay"):
        replace(
            result,
            forced_coordinates=(0.25, *result.forced_coordinates[1:]),
        )
    with pytest.raises(ValueError, match="audit identity"):
        replace(result, sha256="0" * 64)
    with pytest.raises(ValueError, match="canonical K1 catalog"):
        force_v5_k1_branch_coordinates(_base_coordinates(), branch_id="not-a-branch")


def test_each_branch_transform_has_a_unique_audit_identity():
    values = {
        force_v5_k1_branch_coordinates(_base_coordinates(), branch_id=branch.branch_id).sha256
        for branch in K1_PHASE_C_BRANCHES
    }

    assert len(values) == 12
