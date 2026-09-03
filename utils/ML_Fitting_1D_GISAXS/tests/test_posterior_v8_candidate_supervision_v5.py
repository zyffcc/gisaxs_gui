from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_supervision_v5 import (
    CANDIDATE_SUPERVISION_TENSOR_KEYS,
    CANDIDATE_SUPERVISION_V5_SCHEMA,
    CANDIDATE_SUPERVISION_V5_VERSION,
    ExactCompatibleProvenance,
    FrozenSearchProvenance,
    KNOWN_TRUTH_ORACLE_PROTOCOL_ID,
    KNOWN_TRUTH_ORACLE_TERMINATION_REASON,
    POSITIVE_TERMINATION_REASON,
    SEARCH_OUTCOME_CODE,
    V5CandidateSupervision,
    candidate_supervision_v5_contract_payload,
    stack_candidate_supervision_v5,
    validate_candidate_supervision_v5_contract,
)


def _masks(*, varying=(0, 1)):
    active = np.zeros(26, dtype=np.float32)
    active[[0, 1, 4, 5]] = 1.0
    density = np.zeros(26, dtype=np.float32)
    density[list(varying)] = 1.0
    return active, density


def _target(*, varying=(0, 1)):
    result = np.full(26, 0.5, dtype=np.float32)
    for index, value in zip(varying, (0.2, 0.8)):
        result[index] = value
    return result


def _search(*, found: int, used: int | None = None, reason: str | None = None):
    negative = found == 0
    return FrozenSearchProvenance(
        search_artifact_id="search/recipe-7/branch-3",
        search_artifact_sha256="b" * 64,
        protocol_id="frozen-search-v5",
        protocol_sha256="c" * 64,
        evaluator_version="forward-v9",
        metric_name="exact_forward_logrmse",
        threshold_name="curve_equivalence_logrmse",
        threshold_value=0.02,
        threshold_source_id="paper-protocol-v2",
        exact_forward_call_budget=64,
        exact_forward_calls_used=64 if used is None else used,
        termination_reason=(
            "exact_forward_budget_exhausted_without_compatible"
            if reason is None and negative
            else (POSITIVE_TERMINATION_REASON if reason is None else reason)
        ),
        completed=True,
        compatible_representative_count=found,
    )


def _exact(*, metric=0.01):
    return ExactCompatibleProvenance(
        artifact_id="exact/representative-1",
        artifact_sha256="a" * 64,
        metric_value=metric,
        bounds_passed=True,
        physics_passed=True,
    )


def _known_truth_search():
    return replace(
        _search(found=1),
        protocol_id=KNOWN_TRUTH_ORACLE_PROTOCOL_ID,
        exact_forward_call_budget=1,
        exact_forward_calls_used=1,
        termination_reason=KNOWN_TRUTH_ORACLE_TERMINATION_REASON,
    )


def _candidate(outcome, *, target=False, search=None, exact=None, generating_match=None):
    active, varying = _masks()
    return V5CandidateSupervision(
        clean_recipe_id="recipe-7",
        candidate_id=f"branch-{outcome}",
        outcome=outcome,
        active_dimension_mask=active,
        varying_dimension_mask=varying,
        search_provenance=search,
        exact_compatible=exact,
        target_local=_target() if target else None,
        generating_candidate_match=generating_match,
    )


def test_operational_states_are_auditable_and_mismatch_stays_unverified():
    positive = _candidate("compatible_found", target=True, search=_search(found=1), exact=_exact())
    negative = _candidate(
        "no_compatible_found_within_frozen_search_budget", search=_search(found=0)
    )
    unknown = _candidate("unverified", generating_match=False)
    assert positive.bce_eligible and positive.local_mdn_eligible
    assert negative.bce_eligible and not negative.local_mdn_eligible
    assert not unknown.bce_eligible and not unknown.local_mdn_eligible
    audit = unknown.audit_payload()
    assert audit["generating_candidate_match"] is False
    assert audit["generating_mismatch_is_automatic_negative"] is False
    assert len(audit["audit_sha256"]) == 64


def test_positive_and_negative_require_the_right_completed_search_evidence():
    active, varying = _masks()
    base = dict(
        clean_recipe_id="r",
        candidate_id="b",
        active_dimension_mask=active,
        varying_dimension_mask=varying,
    )
    with pytest.raises(ValueError, match="frozen search provenance"):
        V5CandidateSupervision(outcome="compatible_found", **base)
    with pytest.raises(ValueError, match="exact-compatible artifact"):
        V5CandidateSupervision(
            outcome="compatible_found", search_provenance=_search(found=1), **base
        )
    with pytest.raises(ValueError, match="disagrees"):
        V5CandidateSupervision(
            outcome="compatible_found",
            search_provenance=_search(found=1),
            exact_compatible=_exact(metric=0.03),
            **base,
        )
    incomplete = replace(_search(found=0), completed=False)
    with pytest.raises(ValueError, match="completed"):
        V5CandidateSupervision(
            outcome="no_compatible_found_within_frozen_search_budget",
            search_provenance=incomplete,
            **base,
        )
    not_exhausted = _search(found=0, used=63)
    with pytest.raises(ValueError, match="equal frozen branch budget"):
        V5CandidateSupervision(
            outcome="no_compatible_found_within_frozen_search_budget",
            search_provenance=not_exhausted,
            **base,
        )
    short_positive = _search(found=1, used=12)
    with pytest.raises(ValueError, match="equal frozen branch budget"):
        V5CandidateSupervision(
            outcome="compatible_found",
            search_provenance=short_positive,
            exact_compatible=_exact(),
            **base,
        )
    wrong_positive_reason = _search(found=1, reason="compatible_target_reached")
    with pytest.raises(ValueError, match="canonical full-budget termination"):
        V5CandidateSupervision(
            outcome="compatible_found",
            search_provenance=wrong_positive_reason,
            exact_compatible=_exact(),
            **base,
        )
    with pytest.raises(ValueError, match="must not carry"):
        V5CandidateSupervision(outcome="unverified", search_provenance=_search(found=0), **base)


def test_local_target_is_positive_only_and_uses_varying_axes():
    active, varying = _masks(varying=(0,))
    valid = _target(varying=(0,))
    candidate = V5CandidateSupervision(
        clean_recipe_id="r",
        candidate_id="b",
        outcome="compatible_found",
        active_dimension_mask=active,
        varying_dimension_mask=varying,
        search_provenance=_search(found=1),
        exact_compatible=_exact(),
        target_local=valid,
    )
    assert candidate.target_local[1] == 0.5
    invalid = valid.copy()
    invalid[1] = 0.6
    with pytest.raises(ValueError, match="fixed/inactive"):
        replace(candidate, target_local=invalid)
    with pytest.raises(ValueError, match="only exact-compatible"):
        _candidate(
            "no_compatible_found_within_frozen_search_budget",
            target=True,
            search=_search(found=0),
        )


def test_known_truth_one_call_remains_local_warmup_evidence_only():
    candidate = _candidate(
        "compatible_found",
        target=True,
        search=_known_truth_search(),
        exact=_exact(),
    )
    assert candidate.local_mdn_eligible
    assert not candidate.bce_eligible
    with pytest.raises(ValueError, match="one-call warmup contract"):
        replace(
            candidate,
            search_provenance=replace(
                _known_truth_search(),
                termination_reason=POSITIVE_TERMINATION_REASON,
            ),
        )


def test_stacked_tensor_contract_keeps_search_and_exact_provenance():
    labels = stack_candidate_supervision_v5(
        (
            _candidate("compatible_found", target=True, search=_search(found=1), exact=_exact()),
            _candidate("unverified", generating_match=False),
        ),
        clean_recipe_indices=(4, 4),
    )
    assert tuple(labels) == CANDIDATE_SUPERVISION_TENSOR_KEYS
    assert labels["search_outcome_code"].tolist() == [
        SEARCH_OUTCOME_CODE["compatible_found"],
        SEARCH_OUTCOME_CODE["unverified"],
    ]
    assert labels["exact_artifact_sha256"].tolist() == ["a" * 64, ""]
    assert labels["search_protocol_sha256"].tolist() == ["c" * 64, ""]
    assert labels["search_exact_forward_call_budget"].tolist() == [64, 0]
    assert labels["has_local_target"].tolist() == [True, False]
    np.testing.assert_array_equal(labels["target_local"][1], np.full(26, 0.5))


def test_contract_is_versioned_exact_and_tensorflow_free():
    payload = candidate_supervision_v5_contract_payload()
    assert payload["schema_version"] == CANDIDATE_SUPERVISION_V5_SCHEMA
    assert payload["version"] == CANDIDATE_SUPERVISION_V5_VERSION
    assert payload["unverified_enters_bce"] is False
    assert payload["negative_is_no_solution_certificate"] is False
    assert validate_candidate_supervision_v5_contract(payload) == payload
    changed = deepcopy(payload)
    changed["generating_mismatch_is_automatic_negative"] = True
    with pytest.raises(ValueError, match="incompatible"):
        validate_candidate_supervision_v5_contract(changed)
    command = (
        "import sys; "
        "import utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_supervision_v5; "
        "assert 'tensorflow' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True)
