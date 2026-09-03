from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_v5 import (
    V5_BOUNDS_QUERY_SCHEMA,
    branch_condition,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_sampling_v5 import (
    full_range_v5_query,
    sample_v5_bounds_query,
    sample_v5_solution_target,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import (
    branch_pattern_id,
    decode_branch_pattern,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import BRANCH_CODEC_VERSION
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import TOPOLOGIES
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_numeric_canonicalization_v5 import (
    v5_numeric_policy_sha256,
)


def test_query_is_reproducible_truth_independent_and_per_axis_heterogeneous():
    first = sample_v5_bounds_query(
        TOPOLOGIES[8],
        query_seed=20260903,
        d_policies=("optional", "optional"),
        resolution_presence_policy="optional",
    )
    replay = sample_v5_bounds_query(
        TOPOLOGIES[8],
        query_seed=20260903,
        d_policies=("optional", "optional"),
        resolution_presence_policy="optional",
    )
    target_a = sample_v5_solution_target(first, target_seed=11)
    target_b = sample_v5_solution_target(first, target_seed=12)

    assert first == replay
    assert first.sha256 == replay.sha256
    assert V5_BOUNDS_QUERY_SCHEMA in first.canonical_json
    payload = json.loads(first.canonical_json)
    assert payload["branch_codec_version"] == BRANCH_CODEC_VERSION
    assert payload["numeric_policy_sha256"] == v5_numeric_policy_sha256(
        first.numeric_policy_version
    )
    assert len(first.bounds_embedding) == 78
    assert len(first.available_dimension_mask) == 26
    assert target_a.query is first and target_b.query is first
    assert (
        target_a.local_target_unit != target_b.local_target_unit
        or target_a.pattern_id != target_b.pattern_id
    )
    assert len({value.regime for value in first.axis_designs}) > 1
    assert len({value.placement for value in first.axis_designs}) > 1
    assert first.component_bounds[0] != first.component_bounds[1]


def test_optional_d_and_resolution_create_multiple_feasible_hard_branches():
    query = full_range_v5_query(("sphere",), query_seed=7)

    assert query.policy_wire_pattern_ids == (0, 1, 16, 17)
    assert query.feasible_wire_pattern_ids == (0, 1, 16, 17)
    conditions = {
        pattern: branch_condition(query, pattern) for pattern in query.feasible_wire_pattern_ids
    }
    absent = conditions[0]
    present = conditions[17]
    assert absent.available_dimension_mask[4:6] == (True, True)
    assert absent.active_dimension_mask[4:6] == (False, False)
    assert present.active_dimension_mask[4:6] == (True, True)
    assert absent.available_dimension_mask[-2:] == (True, True)
    assert absent.active_dimension_mask[-2:] == (False, False)
    assert present.active_dimension_mask[-2:] == (True, True)
    assert all(
        not varying or active
        for varying, active in zip(present.varying_dimension_mask, present.active_dimension_mask)
    )


def test_heterogeneous_same_shape_single_d_assignments_remain_distinct():
    query = sample_v5_bounds_query(
        ("sphere", "sphere"),
        query_seed=991,
        d_policies=("optional", "optional"),
        resolution_presence_policy="absent",
    )
    first_d = branch_pattern_id((True, False, False, False), False)
    second_d = branch_pattern_id((False, True, False, False), False)

    assert query.component_bounds[0] != query.component_bounds[1]
    assert first_d in query.policy_wire_pattern_ids
    assert second_d in query.policy_wire_pattern_ids
    # Continuous hard-core feasibility may remove a branch, but it cannot
    # identify two heterogeneous assignments as the same wire representative.
    assert decode_branch_pattern(first_d) != decode_branch_pattern(second_d)


@pytest.mark.parametrize("component_count", range(1, 5))
def test_solution_targets_roundtrip_for_k1_through_k4(component_count):
    topology = next(value for value in TOPOLOGIES if len(value) == component_count)
    query = full_range_v5_query(topology, query_seed=component_count)
    target = sample_v5_solution_target(query, target_seed=100 + component_count)
    codec = query.codec_for(target.pattern_id)
    latent, resolution = codec.decode(target.local_target_unit)
    replay = codec.encode(latent, resolution)

    assert np.allclose(replay.unit_cube, target.local_target_unit, rtol=0.0, atol=5.0e-12)
    active = np.asarray(codec.active_mask, dtype=bool)
    coordinates = np.asarray(target.local_target_unit)
    assert np.all(coordinates[~active] == 0.5)
    assert np.all(coordinates[active] >= 0.0)
    assert np.all(coordinates[active] <= 1.0)


def test_solution_target_rejects_stale_scientific_contract_version():
    query = full_range_v5_query(("sphere",), query_seed=12)
    target = sample_v5_solution_target(query, target_seed=13)

    with pytest.raises(ValueError, match="unsupported V5 local solution-target version"):
        replace(target, version="stale_slot_contract")


def test_query_and_branch_inputs_fail_closed():
    query = full_range_v5_query(("sphere",), query_seed=3)
    with pytest.raises(ValueError, match="not feasible"):
        query.codec_for(2)
    with pytest.raises(ValueError, match="must match topology"):
        sample_v5_bounds_query(("sphere",), query_seed=1, d_policies=("optional", "absent"))
    with pytest.raises(ValueError, match="does not reproduce"):
        replace(query, sha256="0" * 64)
    with pytest.raises(ValueError, match="not feasible"):
        sample_v5_solution_target(query, target_seed=4, pattern_id=2)
