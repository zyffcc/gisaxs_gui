from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.formal_label_observation_policy_v5 import (
    V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES,
    V5_FORMAL_LABEL_MAX_CANDIDATE_VIEW_COUNT,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_ID,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256,
    V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION,
    V5FormalLabelObservationSelection,
    formal_label_observation_policy_payload,
    select_v5_formal_label_observation,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    V5_UNCERTAINTY_VIEW_POLICY_VERSION,
    sample_v5_uncertainty_provenance,
)


@pytest.mark.parametrize(
    "recipe_seed",
    (0, 1, 2, 17, 77, 991, 20260903, (1 << 64) - 1),
)
def test_default_adjacent_pool_selects_exactly_one_sigma_view(recipe_seed):
    selection = select_v5_formal_label_observation(recipe_seed)
    states = {
        index: sample_v5_uncertainty_provenance(recipe_seed, index)
        for index in V5_FORMAL_LABEL_DEFAULT_CANDIDATE_VIEW_INDICES
    }

    assert selection.candidate_view_indices == (0, 1)
    assert {value.kind for value in states.values()} == {
        "simulated_sigma",
        "encoder_proxy_missing_sigma",
    }
    expected = [
        index for index, value in states.items() if value.measurement_sigma_available
    ]
    assert expected == [selection.selected_view_index]
    assert selection.audit_payload()["selection_count"] == 1


def test_selection_replays_from_seed_and_pool_and_freezes_complete_audit():
    first = select_v5_formal_label_observation(77, (9, 8))
    replay = V5FormalLabelObservationSelection.create(77, (9, 8))

    assert first == replay
    assert first.schema_version == V5_FORMAL_LABEL_OBSERVATION_POLICY_SCHEMA
    assert first.version == V5_FORMAL_LABEL_OBSERVATION_POLICY_VERSION
    assert first.policy_id == V5_FORMAL_LABEL_OBSERVATION_POLICY_ID
    assert first.policy_sha256 == V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256
    assert first.audit_json == canonical_json(first.audit_payload())
    assert first.audit_sha256 == sha256(first.audit_json.encode("utf-8")).hexdigest()

    audit = json.loads(first.audit_json)
    assert audit["recipe_seed"] == 77
    assert audit["candidate_view_indices"] == [9, 8]
    assert audit["selected_view_index"] == first.selected_view_index
    assert audit["decision_was_curve_blind"] is True
    assert {row["view_index"] for row in audit["candidate_uncertainty"]} == {8, 9}

    policy = formal_label_observation_policy_payload()
    assert policy["uncertainty_view_policy_version"] == V5_UNCERTAINTY_VIEW_POLICY_VERSION
    assert policy["decision_inputs"] == ["recipe_seed", "candidate_view_indices"]
    assert policy["physical_curve_is_decision_input"] is False
    assert V5_FORMAL_LABEL_OBSERVATION_POLICY_SHA256 == sha256(
        canonical_json(policy).encode("utf-8")
    ).hexdigest()


@pytest.mark.parametrize("candidate_pool", ((0, 2), (1, 3), (0, 1, 2, 3)))
def test_policy_fails_closed_when_pool_does_not_select_exactly_one(candidate_pool):
    with pytest.raises(ValueError, match="exactly one.*found (0|2)"):
        select_v5_formal_label_observation(77, candidate_pool)


@pytest.mark.parametrize(
    ("recipe_seed", "candidate_pool", "error_type", "message"),
    (
        (True, (0, 1), TypeError, "recipe_seed"),
        (-1, (0, 1), ValueError, "uint64"),
        (1 << 64, (0, 1), ValueError, "uint64"),
        (77, (), ValueError, "cannot be empty"),
        (77, (0, 0), ValueError, "must be unique"),
        (77, (False, 1), TypeError, "candidate view index"),
        (77, (-1, 0), ValueError, "uint64"),
        (77, "0,1", TypeError, "ordered finite sequence"),
        (77, {0, 1}, TypeError, "ordered finite sequence"),
    ),
)
def test_unsafe_or_ambiguous_inputs_are_rejected(
    recipe_seed,
    candidate_pool,
    error_type,
    message,
):
    with pytest.raises(error_type, match=message):
        select_v5_formal_label_observation(recipe_seed, candidate_pool)


def test_resource_exhausting_candidate_pool_is_rejected_before_sampling():
    with pytest.raises(ValueError, match="safety limit"):
        select_v5_formal_label_observation(
            77,
            tuple(range(V5_FORMAL_LABEL_MAX_CANDIDATE_VIEW_COUNT + 1)),
        )


def test_selection_object_rejects_policy_or_audit_tampering():
    selection = select_v5_formal_label_observation(77)
    other_index = 1 - selection.selected_view_index

    with pytest.raises(ValueError, match="does not reproduce"):
        replace(selection, selected_view_index=other_index)
    with pytest.raises(ValueError, match="audit does not reproduce"):
        replace(selection, audit_sha256="0" * 64)
    with pytest.raises(ValueError, match="policy ID"):
        replace(selection, policy_id="legacy-policy")
    with pytest.raises(ValueError, match="schema"):
        replace(selection, schema_version="gisaxs.posterior_v8.formal_label/v0")


def test_numpy_uint64_inputs_are_canonicalized_without_accepting_bool():
    selection = select_v5_formal_label_observation(
        np.uint64(20260903),
        (np.uint64(0), np.uint64(1)),
    )
    assert type(selection.recipe_seed) is int
    assert all(type(index) is int for index in selection.candidate_view_indices)
