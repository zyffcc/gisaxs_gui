from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import V5AmplitudeQuery
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_sampling_v5 import (
    full_range_v5_query,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    CYLINDER,
    SPHERE,
    TOPOLOGIES,
    ClosedInterval,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5_contract import MODEL_V5_INPUT_KEYS
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.preprocessing import preprocess_curve
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.uncertainty_provenance_v5 import (
    V5UncertaintyProvenance,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_v5 import (
    V5TopologyQuery,
    build_v5_universal_candidate_context,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_training_contract_v5 import (
    V5_CROSS_TOPOLOGY_VERIFIED_SEARCH_STAGE,
    V5_GENERATING_TOPOLOGY_WARMUP_STAGE,
    universal_training_contract_v5_payload,
    validate_universal_training_contract_v5,
)


def _curve():
    q = np.geomspace(1.0e-3, 1.0, 64)
    intensity = 100.0 + 5.0e4 / (1.0 + (q / 0.03) ** 4)
    return preprocess_curve(q, intensity, 0.02 * intensity)


def _amplitude_query(query, intervals=None):
    component_intensities = (
        tuple(ClosedInterval(0.0, 1.0) for _ in query.topology)
        if intervals is None
        else tuple(intervals)
    )
    return V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 1.0e8),
        k=ClosedInterval(1.0e-2, 1.0e8),
        component_intensities=component_intensities,
        resolution_presence_policy=query.resolution_presence_policy,
        int_res=ClosedInterval(0.0, 1.0e8),
    )


def _entry(topology, seed):
    query = full_range_v5_query(topology, query_seed=seed)
    return V5TopologyQuery(query, _amplitude_query(query))


def _provenance():
    return V5UncertaintyProvenance("simulated_sigma")


def test_universal_query_enumerates_every_feasible_branch_across_all_34_topologies():
    entries = tuple(_entry(topology, 1000 + index) for index, topology in enumerate(TOPOLOGIES))
    context = build_v5_universal_candidate_context(
        _curve(),
        _provenance(),
        tuple(reversed(entries)),
    )

    assert context.topology_ids == tuple(range(len(TOPOLOGIES)))
    assert context.branch_count == sum(
        len(entry.feasible_wire_pattern_ids) for entry in entries
    )
    expected_keys = tuple(
        f"topology-{entry.topology_id:02d}:wire-{pattern_id:02d}"
        for entry in entries
        for pattern_id in entry.feasible_wire_pattern_ids
    )
    assert context.global_branch_keys == expected_keys
    assert all(
        batch.pattern_ids == entry.feasible_wire_pattern_ids
        for entry, batch in zip(context.topology_queries, context.batches)
    )
    model_inputs = context.for_model()
    assert tuple(model_inputs) == MODEL_V5_INPUT_KEYS
    assert all(value.shape[0] == context.branch_count for value in model_inputs.values())
    assert not any(value.flags.writeable for value in model_inputs.values())
    assert np.unique(model_inputs["branch_topology_id"]).size == len(TOPOLOGIES)
    assert '"all_selected_topology_feasible_branches_enumerated":true' in context.audit_json


def test_universal_query_is_order_independent_and_repeats_exactly_one_observation():
    entries = (
        _entry((SPHERE, CYLINDER), 21),
        _entry((CYLINDER,), 22),
        _entry((SPHERE,), 23),
    )
    first = build_v5_universal_candidate_context(_curve(), _provenance(), entries)
    second = build_v5_universal_candidate_context(
        _curve(), _provenance(), tuple(reversed(entries))
    )

    assert first.audit_json == second.audit_json
    assert first.audit_sha256 == second.audit_sha256
    combined = first.for_model()
    for name in ("x", "point_mask", "global_features", "uncertainty_provenance"):
        expected = np.repeat(combined[name][0:1], first.branch_count, axis=0)
        np.testing.assert_array_equal(combined[name], expected)
    assert json.loads(first.audit_json)["same_observation_repeated_across_topologies"]


def test_topology_filter_is_explicit_and_never_prunes_a_retained_wire_branch():
    entries = (
        _entry((SPHERE,), 31),
        _entry((CYLINDER,), 32),
        _entry((SPHERE, CYLINDER), 33),
    )
    selected_ids = (
        topology_id_for((SPHERE, CYLINDER)),
        topology_id_for((SPHERE,)),
    )
    context = build_v5_universal_candidate_context(
        _curve(),
        _provenance(),
        entries,
        allowed_topology_ids=selected_ids,
    )

    assert context.topology_ids == tuple(sorted(selected_ids))
    assert context.excluded_topology_ids == (topology_id_for((CYLINDER,)),)
    assert all(
        batch.pattern_ids == entry.feasible_wire_pattern_ids
        for entry, batch in zip(context.topology_queries, context.batches)
    )
    audit = json.loads(context.audit_json)
    assert audit["selected_topology_ids"] == list(sorted(selected_ids))
    assert audit["excluded_topology_ids"] == [topology_id_for((CYLINDER,))]


def test_duplicate_topology_and_mismatched_query_pairs_fail_closed():
    first = _entry((SPHERE,), 41)
    duplicate = _entry((SPHERE,), 42)
    with pytest.raises(ValueError, match="at most one query per topology"):
        build_v5_universal_candidate_context(_curve(), _provenance(), (first, duplicate))

    two_component_amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 10.0),
        k=ClosedInterval(1.0, 10.0),
        component_intensities=(ClosedInterval(0.0, 1.0),) * 2,
        resolution_presence_policy="optional",
        int_res=ClosedInterval(0.0, 10.0),
    )
    with pytest.raises(ValueError, match="different topology component counts"):
        V5TopologyQuery(first.geometry, two_component_amplitude)

    no_resolution_amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 10.0),
        k=ClosedInterval(1.0, 10.0),
        component_intensities=(ClosedInterval(0.0, 1.0),),
        resolution_presence_policy="absent",
        int_res=None,
    )
    with pytest.raises(ValueError, match="different Resolution policies"):
        V5TopologyQuery(first.geometry, no_resolution_amplitude)


def test_component_slot_and_amplitude_polytope_stay_with_their_own_topology():
    query = full_range_v5_query((SPHERE, CYLINDER), query_seed=51)
    amplitude = _amplitude_query(
        query,
        intervals=(ClosedInterval(0.10, 0.25), ClosedInterval(0.75, 0.90)),
    )
    entry = V5TopologyQuery(query, amplitude)
    context = build_v5_universal_candidate_context(_curve(), _provenance(), (entry,))

    assert tuple(value.shape for value in entry.component_slots) == (SPHERE, CYLINDER)
    assert tuple(value.component_intensity for value in entry.component_slots) == (
        ClosedInterval(0.10, 0.25),
        ClosedInterval(0.75, 0.90),
    )
    assert len({value.global_slot_key for value in entry.component_slots}) == 2
    for branch in context.branches:
        constraint = context.amplitude_constraint_for(branch.global_key.wire_key)
        assert constraint.component_intensities == amplitude.component_intensities
        codec = context.codec_for(branch.global_key.wire_key)
        assert codec.topology == query.topology
        assert branch.condition.topology_id == query.topology_id


def test_topology_query_uses_amplitude_ranges_in_same_shape_slot_quotient():
    geometry = full_range_v5_query((SPHERE, SPHERE), query_seed=52)
    equal = V5TopologyQuery(geometry, _amplitude_query(geometry))
    unequal = V5TopologyQuery(
        geometry,
        _amplitude_query(
            geometry,
            intervals=(ClosedInterval(0.0, 0.4), ClosedInterval(0.6, 1.0)),
        ),
    )

    assert equal.contextual_branch_catalog.d_equivalence_classes == ((0, 1),)
    assert equal.feasible_wire_pattern_ids == (0, 2, 3, 16, 18, 19)
    assert unequal.contextual_branch_catalog.d_equivalence_classes == ((0,), (1,))
    assert unequal.feasible_wire_pattern_ids == geometry.feasible_wire_pattern_ids


def test_universal_training_contract_forbids_model_selection_claims_from_warmup():
    payload = universal_training_contract_v5_payload()
    warmup = payload["stages"][V5_GENERATING_TOPOLOGY_WARMUP_STAGE]
    verified = payload["stages"][V5_CROSS_TOPOLOGY_VERIFIED_SEARCH_STAGE]

    assert not warmup["search_yield_bce_allowed"]
    assert not warmup["cross_topology_model_selection_claim_allowed"]
    assert verified["search_yield_bce_allowed_only_for_completed_outcomes"]
    assert not verified["generating_mismatch_is_negative"]
    assert validate_universal_training_contract_v5(payload) == payload

    changed = deepcopy(payload)
    changed["stages"][V5_GENERATING_TOPOLOGY_WARMUP_STAGE][
        "cross_topology_model_selection_claim_allowed"
    ] = True
    with pytest.raises(ValueError, match="missing or incompatible"):
        validate_universal_training_contract_v5(changed)
