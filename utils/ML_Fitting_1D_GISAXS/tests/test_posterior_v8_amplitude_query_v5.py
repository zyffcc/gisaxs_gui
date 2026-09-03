from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import (
    AMPLITUDE_AXIS_KEYS,
    AMPLITUDE_AXIS_STRIDE,
    AMPLITUDE_QUERY_EMBEDDING_DIM,
    AMPLITUDE_QUERY_PADDING_VALUE,
    V5_AMPLITUDE_QUERY_SCHEMA,
    V5AmplitudeQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import ClosedInterval
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.sobol_numeric_canonicalization_v5 import (
    v5_numeric_policy_sha256,
)


def _query(
    *,
    component_intensities=(ClosedInterval(1.0, 1.0),),
    policy="absent",
    int_res=None,
):
    return V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 40.0),
        k=ClosedInterval(2.0, 200.0),
        component_intensities=component_intensities,
        resolution_presence_policy=policy,
        int_res=int_res,
    )


def _triplet(embedding: tuple[float, ...], axis_index: int) -> tuple[float, ...]:
    offset = axis_index * AMPLITUDE_AXIS_STRIDE
    return embedding[offset : offset + AMPLITUDE_AXIS_STRIDE]


def test_k1_absent_query_has_explicit_fixed_width_padding_and_exact_constraint():
    query = _query()
    embedding = query.model_embedding(20.0)

    assert len(embedding) == AMPLITUDE_QUERY_EMBEDDING_DIM == 21
    assert len(query.axis_presence_mask) == len(AMPLITUDE_AXIS_KEYS) == 7
    assert query.axis_presence_mask == (True, True, True, False, False, False, False)
    assert _triplet(embedding, 2) == (0.5, 0.5, 1.0)
    assert _triplet(embedding, 3) == (
        AMPLITUDE_QUERY_PADDING_VALUE,
        AMPLITUDE_QUERY_PADDING_VALUE,
        0.0,
    )

    constraint = query.constraint_for_branch(resolution_present=False)
    assert constraint.background == query.background
    assert constraint.k == query.k
    assert constraint.component_intensities == query.component_intensities
    assert constraint.int_res is None
    assert constraint.coefficient_polytope().coefficient_order == ("BG", "a_1")
    with pytest.raises(ValueError, match="not allowed"):
        query.constraint_for_branch(resolution_present=True)


def test_k4_required_query_preserves_independent_intensity_and_resolution_ranges():
    intensities = (
        ClosedInterval(0.05, 0.35),
        ClosedInterval(0.10, 0.45),
        ClosedInterval(0.15, 0.55),
        ClosedInterval(0.05, 2.00),
    )
    int_res = ClosedInterval(0.0, 1.0e12)
    query = _query(component_intensities=intensities, policy="required", int_res=int_res)
    embedding = query.model_embedding(20.0)

    assert query.particle_count == 4
    assert query.axis_presence_mask == (True,) * 7
    assert query.allowed_resolution_states == (True,)
    int4_triplet = _triplet(embedding, 5)
    assert 0.0 < int4_triplet[0] < 0.5 < int4_triplet[1] < 1.0
    assert int4_triplet[2] == 1.0
    assert 0.0 <= _triplet(embedding, 6)[0] <= _triplet(embedding, 6)[1] < 1.0

    constraint = query.constraint_for_branch(resolution_present=True)
    assert constraint.component_intensities == intensities
    assert constraint.int_res == int_res
    assert constraint.coefficient_polytope().coefficient_order == (
        "BG",
        "a_1",
        "a_2",
        "a_3",
        "a_4",
        "a_res",
    )


def test_optional_resolution_query_builds_both_exact_branch_constraints():
    int_res = ClosedInterval(0.25, 3.5)
    query = _query(policy="optional", int_res=int_res)

    assert query.allowed_resolution_states == (False, True)
    absent = query.constraint_for_branch(resolution_present=False)
    present = query.constraint_for_branch(resolution_present=True)
    assert absent.resolution_present is False and absent.int_res is None
    assert present.resolution_present is True and present.int_res == int_res
    assert absent.background == present.background == query.background
    assert absent.k == present.k == query.k
    assert absent.component_intensities == present.component_intensities


def test_component_intensity_queries_are_independent_and_negative_ranges_fail_closed():
    below_unit_sum = _query(
        component_intensities=(ClosedInterval(0.0, 0.2), ClosedInterval(0.0, 0.3))
    )
    above_unit_sum = _query(
        component_intensities=(ClosedInterval(0.6, 0.8), ClosedInterval(0.5, 0.9))
    )
    assert below_unit_sum.particle_count == above_unit_sum.particle_count == 2
    with pytest.raises(ValueError, match="non-negative"):
        _query(component_intensities=(ClosedInterval(-0.1, 1.0),))


def test_k1_intensity_above_one_is_encoded_without_clipping_and_remains_exact():
    query = _query(component_intensities=(ClosedInterval(2.0, 2.0),))
    triplet = _triplet(query.model_embedding(20.0), 2)
    assert 0.5 < triplet[0] == triplet[1] < 1.0
    constraint = query.constraint_for_branch(resolution_present=False)
    assert constraint.contains((1.0, 20.0), k=10.0)


@pytest.mark.parametrize(
    "policy,int_res,match",
    [
        ("absent", ClosedInterval(0.0, 1.0), "must omit"),
        ("optional", None, "ClosedInterval"),
        ("required", None, "ClosedInterval"),
    ],
)
def test_resolution_policy_and_int_res_must_be_coherent(policy, int_res, match):
    with pytest.raises((TypeError, ValueError), match=match):
        _query(policy=policy, int_res=int_res)


def test_intensity_rescaling_is_embedding_equivariant_but_physically_auditable():
    query = _query(policy="optional", int_res=ClosedInterval(0.0, 8.0))
    scaled = query.rescaled_intensity(37.0)

    assert np.allclose(
        query.model_embedding(20.0),
        scaled.model_embedding(740.0),
        rtol=0.0,
        atol=2.0e-16,
    )
    assert scaled.background == ClosedInterval(0.0, 1480.0)
    assert scaled.k == ClosedInterval(74.0, 7400.0)
    assert scaled.component_intensities == query.component_intensities
    assert scaled.int_res == query.int_res
    assert scaled.sha256 != query.sha256


def test_json_hash_replay_audit_and_frozen_derived_fields():
    first = _query(policy="optional", int_res=ClosedInterval(0.1, 9.0))
    replay = _query(policy="optional", int_res=ClosedInterval(0.1, 9.0))
    payload = json.loads(first.canonical_json)

    assert first == replay
    assert first.sha256 == replay.sha256
    assert payload["schema"] == V5_AMPLITUDE_QUERY_SCHEMA
    assert "intensity_reference" not in payload
    assert payload["numeric_policy_sha256"] == v5_numeric_policy_sha256(
        first.numeric_policy_version
    )
    assert first.sha256 == sha256(first.canonical_json.encode("utf-8")).hexdigest()
    audit = first.to_audit_dict()
    assert audit["sha256"] == first.sha256
    with pytest.raises(ValueError, match="does not reproduce"):
        replace(first, sha256="0" * 64)


def test_nonpositive_reference_and_non_boolean_branch_are_rejected():
    query = _query()
    with pytest.raises(ValueError, match="strictly positive"):
        query.model_embedding(0.0)
    with pytest.raises(TypeError, match="must be a bool"):
        query.constraint_for_branch(resolution_present=np.bool_(False))
