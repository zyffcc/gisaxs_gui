from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_budget_evaluator_v5 import (
    V5CandidateEmission,
    V5ExactForwardCall,
    V5FrozenReferenceRepresentative,
    V5FrozenReferenceSet,
    V5MethodExactCallTrace,
    V5PairedPaperBudgetQueryRecord,
    V5PaperBudgetEvaluationConfig,
    V5PaperParameterRepresentativePayload,
    V5_EMITTED_REPRESENTATIVE_ROLE,
    V5_EXACT_COMPATIBLE,
    V5_EXACT_INCOMPATIBLE,
    V5_EXACT_UNVERIFIED,
    V5_PAIRED_BOOTSTRAP_RNG,
    V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE,
    V5_PAIRED_BOOTSTRAP_VERSION,
    V5_REFERENCE_REPRESENTATIVE_ROLE,
    build_v5_paired_bootstrap_auc_input,
    evaluate_v5_paired_paper_budget_query,
    summarize_v5_paired_bootstrap_auc,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    LatentComponentParameters,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    CandidateInput,
    LinearSolutionSnapshot,
    ReferenceMode,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_endpoint_metrics import (
    EXACT_FORWARD_BUDGETS,
    OUTPUT_CAPS,
)


def _sha(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _config() -> V5PaperBudgetEvaluationConfig:
    return V5PaperBudgetEvaluationConfig(
        comparison_protocol_id="paper-comparison-v1",
        comparison_protocol_sha256=_sha("paper-comparison-v1"),
        equivalence_matcher_id="caller-supplied-equivalence-v1",
        equivalence_matcher_sha256=_sha("caller-supplied-equivalence-v1"),
        equivalence_threshold_id="normalized-distance-at-most-one-v1",
        equivalence_threshold_sha256=_sha("normalized-distance-at-most-one-v1"),
        maximum_normalized_distance=1.0,
        output_caps=(1, 2, 4),
        exact_forward_budgets=(2, 4, 8),
    )


def _component(tag: float) -> LatentComponentParameters:
    return LatentComponentParameters(
        shape="sphere",
        log_R=float(np.log(10.0 + tag)),
        sigma_R_fraction=0.1,
    )


def _linear(tag: float) -> LinearSolutionSnapshot:
    return LinearSolutionSnapshot(
        background=tag,
        particle_amplitudes=(1.0 + tag,),
        resolution_amplitude=0.0,
        k=2.0 + tag,
    )


def _reference_payload(representative_id: str, tag: float) -> V5PaperParameterRepresentativePayload:
    return V5PaperParameterRepresentativePayload(
        representative_id=representative_id,
        role=V5_REFERENCE_REPRESENTATIVE_ROLE,
        parameter=ReferenceMode(
            reference_id=representative_id,
            topology_id=0,
            components=(_component(tag),),
            resolution=None,
            linear_solution=_linear(tag),
        ),
        global_branch_key="topology-00:wire-00",
        query_context_sha256=_sha("query-context"),
        source_artifact_sha256=_sha(f"reference-source-{representative_id}"),
    )


def _candidate_payload(
    candidate_id: str,
    tag: float,
    *,
    bounds_pass: bool = True,
    physics_pass: bool = True,
) -> V5PaperParameterRepresentativePayload:
    return V5PaperParameterRepresentativePayload(
        representative_id=candidate_id,
        role=V5_EMITTED_REPRESENTATIVE_ROLE,
        parameter=CandidateInput(
            candidate_id=candidate_id,
            proposal_rank=1,
            topology_id=0,
            components=(_component(tag),),
            resolution=None,
            linear_solution=_linear(tag),
            exact_intensity=np.asarray((1.0 + tag,), dtype=np.float64),
            bounds_pass=bounds_pass,
            physics_pass=physics_pass,
        ),
        global_branch_key="topology-00:wire-00",
        query_context_sha256=_sha("query-context"),
        source_artifact_sha256=_sha(f"candidate-source-{candidate_id}-{tag}"),
    )


_MATCHES_BY_CANDIDATE_ID = {
    "candidate-01": frozenset({"reference-3"}),
    "candidate-02": frozenset({"reference-1", "reference-2"}),
    "candidate-03": frozenset({"reference-2"}),
    "candidate-04": frozenset({"reference-1"}),
    "candidate-05": frozenset({"reference-3"}),
    "candidate-06": frozenset(),
    "candidate-07": frozenset(),
    "candidate-08": frozenset(),
    "only-candidate": frozenset({"reference-1"}),
    "incompatible-first": frozenset(),
    "unverified-second": frozenset({"reference-1"}),
    "compatible-third": frozenset({"reference-1"}),
}


def _references(
    *, query_id: str = "query-001", pairing_unit_id: str = "clean-parent-001"
) -> V5FrozenReferenceSet:
    config = _config()
    return V5FrozenReferenceSet(
        query_id=query_id,
        pairing_unit_id=pairing_unit_id,
        reference_set_id=f"references-{query_id}",
        reference_set_sha256=_sha(f"references-{query_id}"),
        comparison_protocol_id=config.comparison_protocol_id,
        comparison_protocol_sha256=config.comparison_protocol_sha256,
        representatives=tuple(
            V5FrozenReferenceRepresentative(
                representative_id=f"reference-{index}",
                payload=_reference_payload(f"reference-{index}", float(index)),
            )
            for index in range(1, 4)
        ),
    )


def _exact_forward_calls(budget: int = 8) -> tuple[V5ExactForwardCall, ...]:
    return tuple(
        V5ExactForwardCall(exact_call_index=index, elapsed_seconds=index / 10.0)
        for index in range(1, budget + 1)
    )


def _candidate_emissions() -> tuple[V5CandidateEmission, ...]:
    ranks = (4, 2, 5, 1, 3, 6, 7, 8)
    statuses = (
        V5_EXACT_INCOMPATIBLE,
        V5_EXACT_COMPATIBLE,
        V5_EXACT_UNVERIFIED,
        V5_EXACT_COMPATIBLE,
        V5_EXACT_COMPATIBLE,
        V5_EXACT_INCOMPATIBLE,
        V5_EXACT_INCOMPATIBLE,
        V5_EXACT_INCOMPATIBLE,
    )
    return tuple(
        V5CandidateEmission(
            available_after_call=index,
            output_rank=ranks[index - 1],
            candidate_id=f"candidate-{index:02d}",
            compatibility_status=statuses[index - 1],
            elapsed_seconds=index / 10.0,
            payload=_candidate_payload(f"candidate-{index:02d}", float(index + 10)),
        )
        for index in range(1, 9)
    )


def _trace(
    method_id: str,
    *,
    query_id: str = "query-001",
    pairing_unit_id: str = "clean-parent-001",
    exact_forward_calls=None,
    candidate_emissions=None,
    budget: int = 8,
) -> V5MethodExactCallTrace:
    references = _references(query_id=query_id, pairing_unit_id=pairing_unit_id)
    return V5MethodExactCallTrace(
        query_id=query_id,
        pairing_unit_id=pairing_unit_id,
        method_id=method_id,
        method_protocol_id=f"{method_id}-frozen-protocol",
        method_protocol_sha256=_sha(f"{method_id}-frozen-protocol"),
        trace_id=f"{method_id}-{query_id}-trace",
        trace_artifact_sha256=_sha(f"{method_id}-{query_id}-trace"),
        reference_set_id=references.reference_set_id,
        reference_set_sha256=references.reference_set_sha256,
        comparison_protocol_id=references.comparison_protocol_id,
        comparison_protocol_sha256=references.comparison_protocol_sha256,
        exact_forward_call_budget=budget,
        exact_forward_calls=(
            _exact_forward_calls(budget)
            if exact_forward_calls is None
            else tuple(exact_forward_calls)
        ),
        candidate_emissions=(
            _candidate_emissions() if candidate_emissions is None else tuple(candidate_emissions)
        ),
    )


def _matcher(reference_payload, candidate_payload) -> float | None:
    return (
        0.25
        if reference_payload.representative_id
        in _MATCHES_BY_CANDIDATE_ID[candidate_payload.representative_id]
        else None
    )


def _record(traces=None):
    values = (_trace("method-b"), _trace("method-a")) if traces is None else traces
    return evaluate_v5_paired_paper_budget_query(
        _references(), values, config=_config(), equivalence_distance_matcher=_matcher
    )


def test_representative_history_sealed_file_round_trip_and_rejection(tmp_path):
    import os
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_history_v5 import (
        V5RepresentativeHistory, V5RepresentativeSnapshot,
    )
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_history_store_v5 import (
        publish_v5_representative_history, read_v5_representative_history,
    )
    history = V5RepresentativeHistory(trace=_trace("neural"), snapshots=(
        V5RepresentativeSnapshot(available_after_call=2, elapsed_seconds=0.2,
                                 representative_ids=("candidate-02",)),
    ))
    path = tmp_path / "history.json"
    digest = publish_v5_representative_history(path, history)
    kwargs = dict(trace=history.trace, expected_file_sha256=digest,
                  expected_history_sha256=history.sha256)
    restored = read_v5_representative_history(path, **kwargs)
    assert restored.to_payload() == history.to_payload()
    assert path.stat().st_mode & 0o777 == 0o400
    assert path.stat().st_nlink == 1
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        publish_v5_representative_history(path, history)
    assert path.read_bytes() == original
    for field in ("expected_file_sha256", "expected_history_sha256"):
        with pytest.raises(ValueError, match="SHA-256 differs"):
            read_v5_representative_history(path, **{**kwargs, field: "0" * 64})
    link = tmp_path / "symlink.json"
    link.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        read_v5_representative_history(link, **kwargs)
    os.link(path, tmp_path / "hardlink.json")
    with pytest.raises(ValueError, match="one hard link"):
        read_v5_representative_history(path, **kwargs)
    provisional = replace(history, trace=replace(history.trace, trace_artifact_sha256="0" * 64))
    with pytest.raises(ValueError, match="provisional"):
        publish_v5_representative_history(tmp_path / "unsealed.json", provisional)
    assert not (tmp_path / "unsealed.json").exists()


@pytest.mark.parametrize("encoding", ["whitespace", "duplicate", "mode", "other_trace"])
def test_representative_history_store_rejects_noncanonical_or_unbound_file(tmp_path, encoding):
    import json
    from hashlib import sha256
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_history_v5 import (
        V5RepresentativeHistory, V5RepresentativeSnapshot,
    )
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_history_store_v5 import (
        read_v5_representative_history,
    )
    history = V5RepresentativeHistory(trace=_trace("neural"), snapshots=(
        V5RepresentativeSnapshot(available_after_call=2, elapsed_seconds=0.2,
                                 representative_ids=("candidate-02",)),
    ))
    text = canonical_json(history.to_payload())
    if encoding == "whitespace":
        text += "\n"
    if encoding == "duplicate":
        text = '{"query_id":' + json.dumps(history.trace.query_id) + ',' + text[1:]
    raw = text.encode()
    path = tmp_path / "history.json"
    path.write_bytes(raw)
    path.chmod(0o444 if encoding == "mode" else 0o400)
    trace = history.trace
    if encoding == "other_trace":
        trace = replace(trace, trace_artifact_sha256="f" * 64)
    with pytest.raises(ValueError):
        read_v5_representative_history(
            path, trace=trace, expected_file_sha256=sha256(raw).hexdigest(),
            expected_history_sha256=history.sha256,
        )


def test_representative_history_json_replay_binds_trace_and_exact_fields():
    import json
    from hashlib import sha256
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.grouped_artifact_v5 import canonical_json
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_history_v5 import (
        V5RepresentativeHistory, V5RepresentativeSnapshot,
    )
    trace = _trace("neural")
    history = V5RepresentativeHistory(trace=trace, snapshots=(
        V5RepresentativeSnapshot(available_after_call=2, elapsed_seconds=0.2,
                                 representative_ids=("candidate-02",)),
    ))
    payload = json.loads(json.dumps(history.to_payload()))
    assert V5RepresentativeHistory.from_payload(payload, trace=trace) == history
    assert payload["history_sha256"] == history.sha256
    for key in ("query_id", "trace_ledger_sha256", "trace_artifact_sha256",
                "pairing_unit_id", "method_protocol_sha256", "reference_set_sha256",
                "comparison_protocol_sha256", "schema", "method_id"):
        changed = dict(payload)
        changed[key] = "changed"
        changed.pop("history_sha256")
        changed["history_sha256"] = sha256(canonical_json(changed).encode()).hexdigest()
        with pytest.raises(ValueError, match="binding differs"):
            V5RepresentativeHistory.from_payload(changed, trace=trace)
    for changed in ({**payload, "extra": 1}, {**payload, "history_sha256": "bad"}):
        with pytest.raises(ValueError, match="binding differs"):
            V5RepresentativeHistory.from_payload(changed, trace=trace)
    payload["snapshots"][0]["extra"] = 1
    with pytest.raises(ValueError, match="snapshot fields"):
        V5RepresentativeHistory.from_payload(payload, trace=trace)


def test_representative_history_replaces_rather_than_accumulates_visible_modes():
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_history_v5 import (
        V5RepresentativeHistory, V5RepresentativeSnapshot,
    )
    first = V5RepresentativeSnapshot(available_after_call=2, elapsed_seconds=0.2,
                                     representative_ids=("candidate-02",))
    later = V5RepresentativeSnapshot(available_after_call=5, elapsed_seconds=0.5,
                                     representative_ids=("candidate-05", "candidate-04"))
    history = V5RepresentativeHistory(trace=_trace("neural"), snapshots=(first, later))
    assert history.representatives_at(1, 16) == ()
    assert [r.candidate_id for r in history.representatives_at(4, 16)] == ["candidate-02"]
    assert [r.candidate_id for r in history.representatives_at(8, 1)] == ["candidate-05"]
    assert [r.candidate_id for r in history.representatives_at(8, 16)] == ["candidate-05", "candidate-04"]
    withdrawn = replace(history, snapshots=(first, replace(later, representative_ids=())))
    assert withdrawn.representatives_at(8, 16) == ()
    assert history.sha256 != withdrawn.sha256
    assert not isinstance(history, V5MethodExactCallTrace)

    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_history_v5 import (
        evaluate_v5_representative_history,
    )
    evaluated = evaluate_v5_representative_history(
        _references(), history, config=_config(), equivalence_distance_matcher=_matcher,
    )
    assert evaluated.history_sha256 == history.sha256
    assert evaluated.method_result.selected_candidate_ids_matrix[-1] == (
        ("candidate-02",), ("candidate-02",), ("candidate-05", "candidate-04"),
    )
    assert evaluated.method_result.hit_count_matrix[-1] == (1, 1, 2)
    empty = evaluate_v5_representative_history(
        _references(), withdrawn, config=_config(), equivalence_distance_matcher=_matcher,
    )
    assert empty.method_result.hit_count_matrix[-1] == (1, 1, 0)
    assert evaluated.sha256 != empty.sha256
    with pytest.raises(ValueError, match="binding"):
        evaluate_v5_representative_history(
            _references(query_id="another-query"), history,
            config=_config(), equivalence_distance_matcher=_matcher,
        )


@pytest.mark.parametrize("index,elapsed,ids,message", [
    (2, 0.2, ("candidate-04",), "future"),
    (2, 0.2, ("missing",), "unknown"),
    (2, 0.2, ("candidate-01",), "incompatible"),
    (2, 0.31, ("candidate-02",), "next exact call"),
    (9, 0.9, (), "within budget"),
])
def test_representative_history_rejects_unavailable_or_invalid_snapshots(index, elapsed, ids, message):
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_history_v5 import (
        V5RepresentativeHistory, V5RepresentativeSnapshot,
    )
    row = V5RepresentativeSnapshot(available_after_call=index, elapsed_seconds=elapsed,
                                   representative_ids=ids)
    with pytest.raises(ValueError, match=message):
        V5RepresentativeHistory(trace=_trace("neural"), snapshots=(row,))


def test_snapshot_metrics_never_match_hidden_inventory_or_backdate_visibility():
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_representative_history_v5 import (
        V5RepresentativeHistory, V5RepresentativeSnapshot, evaluate_v5_representative_history,
    )
    history = V5RepresentativeHistory(trace=_trace("neural"), snapshots=(
        V5RepresentativeSnapshot(available_after_call=4, elapsed_seconds=0.45,
                                 representative_ids=("candidate-02",)),
    ))
    matched_ids = []

    def only_visible(reference, candidate):
        assert candidate.representative_id == "candidate-02"
        matched_ids.append(candidate.representative_id)
        return _matcher(reference, candidate)

    result = evaluate_v5_representative_history(
        _references(), history, config=_config(), equivalence_distance_matcher=only_visible,
    ).method_result
    assert matched_ids
    assert result.first_compatible_exact_call == 4
    assert result.time_to_first_compatible_seconds == 0.45
    assert result.hit_count_matrix[-1] == (0, 1, 1)
    assert result.selected_candidate_ids_matrix[-1][0] == ()
    with pytest.raises(ValueError, match="exceeds"):
        history.representatives_at(9, 1)
    changed = replace(history, trace=replace(history.trace, pairing_unit_id="another-parent"))
    assert changed.sha256 != history.sha256
    with pytest.raises(TypeError, match="not text"):
        V5RepresentativeSnapshot(available_after_call=1, elapsed_seconds=0.1,
                                 representative_ids="candidate-02")


def test_defaults_reproduce_preregistered_n_by_budget_grid() -> None:
    config = V5PaperBudgetEvaluationConfig(
        comparison_protocol_id="paper",
        comparison_protocol_sha256=_sha("paper"),
        equivalence_matcher_id="matcher",
        equivalence_matcher_sha256=_sha("matcher"),
        equivalence_threshold_id="threshold",
        equivalence_threshold_sha256=_sha("threshold"),
        maximum_normalized_distance=1.0,
    )

    assert config.output_caps == OUTPUT_CAPS == (1, 4, 8, 16)
    assert config.exact_forward_budgets == EXACT_FORWARD_BUDGETS
    assert len(config.sha256) == 64
    assert config.audit_payload()["matching_rule"].startswith("maximum_cardinality_then_")
    assert config.audit_payload()["output_cap_rule"].startswith("filter_exact_compatible_emissions")


def test_primary_payload_is_one_typed_emitted_representative_and_enters_ledger() -> None:
    with pytest.raises(TypeError, match="exactly one CandidateInput"):
        V5PaperParameterRepresentativePayload(
            representative_id="cluster",
            role=V5_EMITTED_REPRESENTATIVE_ROLE,
            parameter=(_candidate_payload("c1", 1.0).parameter,),
            global_branch_key="topology-00:wire-00",
            query_context_sha256=_sha("query-context"),
            source_artifact_sha256=_sha("cluster-source"),
        )
    with pytest.raises(TypeError, match="typed parameter representative"):
        V5CandidateEmission(
            available_after_call=1,
            output_rank=1,
            candidate_id="cluster",
            compatibility_status=V5_EXACT_COMPATIBLE,
            elapsed_seconds=0.1,
            payload=frozenset({_candidate_payload("c1", 1.0)}),
        )
    with pytest.raises(ValueError, match="bounds and physics"):
        V5CandidateEmission(
            available_after_call=1,
            output_rank=1,
            candidate_id="failed-bounds",
            compatibility_status=V5_EXACT_COMPATIBLE,
            elapsed_seconds=0.1,
            payload=_candidate_payload("failed-bounds", 2.0, bounds_pass=False),
        )

    first = replace(_candidate_emissions()[0], output_rank=1)
    changed = replace(first, payload=_candidate_payload(first.candidate_id, 50.0))
    first_trace = _trace("ledger-method", candidate_emissions=(first,))
    changed_trace = _trace("ledger-method", candidate_emissions=(changed,))
    assert first.payload.sha256 != changed.payload.sha256
    assert first_trace.ledger_sha256 != changed_trace.ledger_sha256
    assert len(_references().representative_payload_set_sha256) == 64


def test_reference_and_emitted_payloads_cannot_cross_query_contexts() -> None:
    reference_values = list(_references().representatives)
    reference_values[-1] = replace(
        reference_values[-1],
        payload=replace(
            reference_values[-1].payload,
            query_context_sha256=_sha("different-query-context"),
        ),
    )
    with pytest.raises(ValueError, match="one query context"):
        replace(_references(), representatives=tuple(reference_values))

    emission = replace(
        _candidate_emissions()[0],
        output_rank=1,
        payload=replace(
            _candidate_emissions()[0].payload,
            query_context_sha256=_sha("different-query-context"),
        ),
    )
    with pytest.raises(ValueError, match="frozen query context"):
        evaluate_v5_paired_paper_budget_query(
            _references(),
            (_trace("method", candidate_emissions=(emission,)),),
            config=_config(),
            equivalence_distance_matcher=_matcher,
        )

    assert _record().query_context_sha256 == _sha("query-context")


def test_frozen_threshold_identity_and_value_gate_all_hits() -> None:
    above_threshold = evaluate_v5_paired_paper_budget_query(
        _references(),
        (_trace("method"),),
        config=_config(),
        equivalence_distance_matcher=lambda _reference, _candidate: 1.01,
    )
    relaxed_config = replace(
        _config(),
        equivalence_threshold_id="normalized-distance-at-most-two-v1",
        equivalence_threshold_sha256=_sha("normalized-distance-at-most-two-v1"),
        maximum_normalized_distance=2.0,
    )
    below_threshold = evaluate_v5_paired_paper_budget_query(
        _references(),
        (_trace("method"),),
        config=relaxed_config,
        equivalence_distance_matcher=lambda _reference, _candidate: 1.01,
    )

    assert not any(
        value for row in above_threshold.method_results[0].hit_count_matrix for value in row
    )
    assert below_threshold.method_results[0].hit_count_matrix[-1][-1] == 3
    assert above_threshold.evaluator_config_sha256 != below_threshold.evaluator_config_sha256
    assert below_threshold.equivalence_threshold_id == relaxed_config.equivalence_threshold_id


def test_budget_prefix_matrix_matching_auc_and_time_are_exact() -> None:
    record = _record()

    assert tuple(value.method_id for value in record.method_results) == (
        "method-a",
        "method-b",
    )
    result = record.method_results[0]
    assert result.hit_count_matrix == ((1, 1, 1), (1, 2, 2), (1, 2, 3))
    assert np.asarray(result.recall_matrix) == pytest.approx(
        np.asarray(
            (
                (1 / 3, 1 / 3, 1 / 3),
                (1 / 3, 2 / 3, 2 / 3),
                (1 / 3, 2 / 3, 1.0),
            )
        )
    )
    assert dict(result.auc_by_output_cap) == pytest.approx({1: 1 / 3, 2: 7 / 12, 4: 2 / 3})
    assert dict(result.recall_ceiling_by_output_cap) == pytest.approx({1: 1 / 3, 2: 2 / 3, 4: 1.0})
    assert result.first_compatible_exact_call == 2
    assert result.time_to_first_compatible_seconds == pytest.approx(0.2)
    assert result.exact_forward_calls_evaluated == 8
    assert result.candidate_emissions_evaluated == 8
    assert dict(result.compatibility_status_counts) == {
        V5_EXACT_COMPATIBLE: 3,
        V5_EXACT_INCOMPATIBLE: 4,
        V5_EXACT_UNVERIFIED: 1,
    }
    assert result.selected_candidate_ids_matrix[0] == (
        ("candidate-02",),
        ("candidate-04",),
        ("candidate-04",),
    )
    assert result.matched_pairs_matrix[1][1] == (
        ("reference-1", "candidate-04", 0.25),
        ("reference-2", "candidate-02", 0.25),
    )
    assert result.matched_distance_sum_matrix[1][1] == pytest.approx(0.5)
    assert len(record.sha256) == 64


def test_late_high_rank_and_incompatible_matching_cannot_leak_into_early_budget() -> None:
    result = _record().method_results[0]

    # Call four is globally rank one, but cannot enter B=2. Call one would match
    # r3, but caller marked it incompatible and it therefore never creates a hit.
    assert result.selected_candidate_ids_matrix[0][0] == ("candidate-02",)
    assert result.hit_count_matrix[2][0] == 1
    assert all(
        candidate_id != "candidate-01"
        for row in result.matched_pairs_matrix
        for cell in row
        for _, candidate_id, _ in cell
    )


def test_method_order_is_canonical_and_matcher_must_return_finite_distance_or_none() -> None:
    forward = _record((_trace("method-a"), _trace("method-b")))
    reverse = _record((_trace("method-b"), _trace("method-a")))
    assert forward.sha256 == reverse.sha256
    assert forward.audit_payload() == reverse.audit_payload()
    canonical_emissions = _record((_trace("method-a"),))
    reversed_emissions = _record(
        (_trace("method-a", candidate_emissions=reversed(_candidate_emissions())),)
    )
    assert canonical_emissions.sha256 == reversed_emissions.sha256

    with pytest.raises(TypeError, match="real number"):
        evaluate_v5_paired_paper_budget_query(
            _references(),
            (_trace("method-a"),),
            config=_config(),
            equivalence_distance_matcher=lambda _reference, _candidate: np.bool_(True),
        )


@pytest.mark.parametrize(
    ("exact_forward_calls", "message"),
    [
        (
            (_exact_forward_calls()[0], _exact_forward_calls()[0]),
            "complete, ordered, and contiguous",
        ),
        ((_exact_forward_calls()[0],), "complete, ordered, and contiguous"),
    ],
)
def test_duplicate_and_missing_exact_call_indices_fail_closed(exact_forward_calls, message) -> None:
    with pytest.raises(ValueError, match=message):
        _trace(
            "broken",
            exact_forward_calls=exact_forward_calls,
            candidate_emissions=(),
            budget=2,
        )


def test_duplicate_candidate_ranks_and_budget_leakage_fail_closed() -> None:
    emissions = _candidate_emissions()
    with pytest.raises(ValueError, match="output ranks must be unique"):
        _trace(
            "duplicate-rank",
            candidate_emissions=(
                emissions[0],
                replace(emissions[1], output_rank=emissions[0].output_rank),
            ),
        )
    with pytest.raises(ValueError, match="leaks beyond"):
        _trace(
            "leaked-emission",
            candidate_emissions=(
                V5CandidateEmission(
                    available_after_call=9,
                    output_rank=1,
                    candidate_id="late",
                    compatibility_status=V5_EXACT_COMPATIBLE,
                    elapsed_seconds=0.9,
                    payload=_candidate_payload("late", 30.0),
                ),
            ),
        )


def test_budget_reference_and_protocol_mismatch_fail_closed() -> None:
    emissions = list(_candidate_emissions())

    def must_not_match_past_budget(reference_payload, candidate_payload):
        if candidate_payload.representative_id == "candidate-05":
            raise AssertionError("candidate beyond configured B leaked into matcher")
        return _matcher(reference_payload, candidate_payload)

    truncated = evaluate_v5_paired_paper_budget_query(
        _references(),
        (_trace("method", candidate_emissions=emissions, budget=8),),
        config=replace(_config(), exact_forward_budgets=(1, 2, 4)),
        equivalence_distance_matcher=must_not_match_past_budget,
    )
    truncated_result = truncated.method_results[0]
    assert truncated_result.first_compatible_exact_call == 2
    assert dict(truncated_result.compatibility_status_counts) == {
        V5_EXACT_COMPATIBLE: 2,
        V5_EXACT_INCOMPATIBLE: 1,
        V5_EXACT_UNVERIFIED: 1,
    }
    assert all(
        "candidate-05" not in cell
        for row in truncated_result.selected_candidate_ids_matrix
        for cell in row
    )
    with pytest.raises(ValueError, match="does not cover"):
        evaluate_v5_paired_paper_budget_query(
            _references(),
            (_trace("method", budget=8),),
            config=replace(_config(), exact_forward_budgets=(2, 4, 16)),
            equivalence_distance_matcher=_matcher,
        )

    wrong_reference = replace(_trace("method"), reference_set_sha256="0" * 64)
    with pytest.raises(ValueError, match="reference, query, protocol, or budget mismatch"):
        evaluate_v5_paired_paper_budget_query(
            _references(),
            (wrong_reference,),
            config=_config(),
            equivalence_distance_matcher=_matcher,
        )

    wrong_protocol = replace(_references(), comparison_protocol_sha256="f" * 64)
    with pytest.raises(ValueError, match="reference set escaped"):
        evaluate_v5_paired_paper_budget_query(
            wrong_protocol,
            (_trace("method"),),
            config=_config(),
            equivalence_distance_matcher=_matcher,
        )


def test_many_internal_exact_calls_can_emit_one_candidate_without_consuming_n() -> None:
    emission = V5CandidateEmission(
        available_after_call=4,
        output_rank=1,
        candidate_id="only-candidate",
        compatibility_status=V5_EXACT_COMPATIBLE,
        elapsed_seconds=0.4,
        payload=_candidate_payload("only-candidate", 31.0),
    )
    record = evaluate_v5_paired_paper_budget_query(
        _references(),
        (_trace("sparse", candidate_emissions=(emission,)),),
        config=_config(),
        equivalence_distance_matcher=_matcher,
    )
    result = record.method_results[0]

    assert result.exact_forward_calls_evaluated == 8
    assert result.candidate_emissions_evaluated == 1
    assert result.selected_candidate_ids_matrix[2] == (
        (),
        ("only-candidate",),
        ("only-candidate",),
    )
    assert result.hit_count_matrix[2] == (0, 1, 1)
    assert result.first_compatible_exact_call == 4


def test_incompatible_and_unverified_attempts_consume_budget_but_not_return_cap() -> None:
    emissions = (
        V5CandidateEmission(
            available_after_call=1,
            output_rank=1,
            candidate_id="incompatible-first",
            compatibility_status=V5_EXACT_INCOMPATIBLE,
            elapsed_seconds=0.1,
            payload=_candidate_payload("incompatible-first", 32.0),
        ),
        V5CandidateEmission(
            available_after_call=2,
            output_rank=2,
            candidate_id="unverified-second",
            compatibility_status=V5_EXACT_UNVERIFIED,
            elapsed_seconds=0.2,
            payload=_candidate_payload("unverified-second", 33.0),
        ),
        V5CandidateEmission(
            available_after_call=3,
            output_rank=3,
            candidate_id="compatible-third",
            compatibility_status=V5_EXACT_COMPATIBLE,
            elapsed_seconds=0.3,
            payload=_candidate_payload("compatible-third", 34.0),
        ),
    )
    result = evaluate_v5_paired_paper_budget_query(
        _references(),
        (_trace("returned-cap", candidate_emissions=emissions),),
        config=replace(_config(), output_caps=(1,), exact_forward_budgets=(2, 3, 8)),
        equivalence_distance_matcher=_matcher,
    ).method_results[0]

    assert result.selected_candidate_ids_matrix == (
        ((), ("compatible-third",), ("compatible-third",)),
    )
    assert result.hit_count_matrix == ((0, 1, 1),)
    assert result.exact_forward_calls_evaluated == 8
    assert result.candidate_emissions_evaluated == 3


def test_matching_maximizes_cardinality_then_minimizes_total_distance() -> None:
    references = V5FrozenReferenceSet(
        query_id="distance-query",
        pairing_unit_id="distance-parent",
        reference_set_id="distance-references",
        reference_set_sha256=_sha("distance-references"),
        comparison_protocol_id=_config().comparison_protocol_id,
        comparison_protocol_sha256=_config().comparison_protocol_sha256,
        representatives=(
            V5FrozenReferenceRepresentative(
                representative_id="r1", payload=_reference_payload("r1", 41.0)
            ),
            V5FrozenReferenceRepresentative(
                representative_id="r2", payload=_reference_payload("r2", 42.0)
            ),
        ),
    )
    emissions = (
        V5CandidateEmission(
            available_after_call=1,
            output_rank=1,
            candidate_id="c1",
            compatibility_status=V5_EXACT_COMPATIBLE,
            elapsed_seconds=0.1,
            payload=_candidate_payload("c1", 43.0),
        ),
        V5CandidateEmission(
            available_after_call=2,
            output_rank=2,
            candidate_id="c2",
            compatibility_status=V5_EXACT_COMPATIBLE,
            elapsed_seconds=0.2,
            payload=_candidate_payload("c2", 44.0),
        ),
    )
    trace = V5MethodExactCallTrace(
        query_id=references.query_id,
        pairing_unit_id=references.pairing_unit_id,
        method_id="distance-method",
        method_protocol_id="distance-method-protocol",
        method_protocol_sha256=_sha("distance-method-protocol"),
        trace_id="distance-trace",
        trace_artifact_sha256=_sha("distance-trace"),
        reference_set_id=references.reference_set_id,
        reference_set_sha256=references.reference_set_sha256,
        comparison_protocol_id=references.comparison_protocol_id,
        comparison_protocol_sha256=references.comparison_protocol_sha256,
        exact_forward_call_budget=2,
        exact_forward_calls=_exact_forward_calls(2),
        candidate_emissions=emissions,
    )
    distances = {
        ("r1", "c1"): 0.1,
        ("r1", "c2"): 0.9,
        ("r2", "c1"): 0.2,
        ("r2", "c2"): 0.3,
    }
    config = replace(_config(), output_caps=(1, 2), exact_forward_budgets=(1, 2))
    result = evaluate_v5_paired_paper_budget_query(
        references,
        (trace,),
        config=config,
        equivalence_distance_matcher=lambda reference, candidate: distances[
            (reference.representative_id, candidate.representative_id)
        ],
    ).method_results[0]

    # Both assignments recall two modes. The old cardinality-only augmenting
    # path can choose r1-c2/r2-c1 (1.1); the frozen evaluator must choose 0.4.
    assert result.matched_pairs_matrix[1][1] == (
        ("r1", "c1", 0.1),
        ("r2", "c2", 0.3),
    )
    assert result.matched_distance_sum_matrix[1][1] == pytest.approx(0.4)


@pytest.mark.parametrize("invalid_distance", [np.nan, np.inf, -0.1])
def test_matcher_invalid_distances_fail_closed(invalid_distance) -> None:
    with pytest.raises((TypeError, ValueError), match="equivalence matcher distance"):
        evaluate_v5_paired_paper_budget_query(
            _references(),
            (_trace("method"),),
            config=_config(),
            equivalence_distance_matcher=lambda _reference, _candidate: invalid_distance,
        )


def _record_with_aucs(query_id, pairing_unit_id, baseline, comparison):
    template = _record()
    base_method = replace(
        template.method_results[0],
        method_id="baseline",
        auc_by_output_cap=((1, baseline), (2, baseline), (4, baseline)),
    )
    comparison_method = replace(
        template.method_results[1],
        method_id="comparison",
        auc_by_output_cap=((1, comparison), (2, comparison), (4, comparison)),
    )
    return V5PairedPaperBudgetQueryRecord(
        query_id=query_id,
        query_context_sha256=template.query_context_sha256,
        pairing_unit_id=pairing_unit_id,
        reference_set_id=f"references-{query_id}",
        reference_set_sha256=_sha(f"references-{query_id}"),
        reference_representative_ids=template.reference_representative_ids,
        reference_representative_payload_set_sha256=(
            template.reference_representative_payload_set_sha256
        ),
        evaluator_config_sha256=template.evaluator_config_sha256,
        comparison_protocol_id=template.comparison_protocol_id,
        comparison_protocol_sha256=template.comparison_protocol_sha256,
        equivalence_matcher_id=template.equivalence_matcher_id,
        equivalence_matcher_sha256=template.equivalence_matcher_sha256,
        equivalence_threshold_id=template.equivalence_threshold_id,
        equivalence_threshold_sha256=template.equivalence_threshold_sha256,
        maximum_normalized_distance=template.maximum_normalized_distance,
        output_caps=template.output_caps,
        exact_forward_budgets=template.exact_forward_budgets,
        method_results=(base_method, comparison_method),
    )


def test_paired_bootstrap_clusters_queries_and_records_rng_provenance() -> None:
    records = (
        _record_with_aucs("q-2", "parent-1", 0.4, 0.6),
        _record_with_aucs("q-1", "parent-1", 0.2, 0.4),
        _record_with_aucs("q-3", "parent-2", 0.8, 0.7),
    )
    values = build_v5_paired_bootstrap_auc_input(records, output_cap=4)

    assert values.pairing_unit_ids == ("parent-1", "parent-2")
    assert values.query_ids_by_pairing_unit == (("q-1", "q-2"), ("q-3",))
    assert values.method_ids == ("baseline", "comparison")
    assert np.asarray(values.auc_matrix) == pytest.approx(np.asarray(((0.3, 0.5), (0.8, 0.7))))

    first = summarize_v5_paired_bootstrap_auc(
        values,
        baseline_method_id="baseline",
        comparison_method_id="comparison",
        bootstrap_replicates=2_000,
        rng_seed=20260903,
    )
    second = summarize_v5_paired_bootstrap_auc(
        values,
        baseline_method_id="baseline",
        comparison_method_id="comparison",
        bootstrap_replicates=2_000,
        rng_seed=20260903,
    )
    assert first == second
    assert first.baseline_mean_auc == pytest.approx(0.55)
    assert first.comparison_mean_auc == pytest.approx(0.6)
    assert first.observed_comparison_minus_baseline == pytest.approx(0.05)
    assert first.confidence_interval == pytest.approx((-0.1, 0.2))
    assert first.rng_seed == 20260903
    assert first.rng_algorithm == V5_PAIRED_BOOTSTRAP_RNG
    assert first.bootstrap_version == V5_PAIRED_BOOTSTRAP_VERSION
    assert first.claim_scope == V5_PAIRED_BOOTSTRAP_CLAIM_SCOPE
    assert first.claim_scope.endswith("not_population_inference")
    assert first.numpy_version == np.__version__


def test_bootstrap_rejects_missing_pairs_duplicate_queries_and_unknown_cap() -> None:
    full = _record_with_aucs("q-1", "parent-1", 0.2, 0.4)
    missing = replace(full, query_id="q-2", method_results=full.method_results[:1])
    with pytest.raises(ValueError, match="method contract"):
        build_v5_paired_bootstrap_auc_input((full, missing), output_cap=4)
    with pytest.raises(ValueError, match="query IDs must be unique"):
        build_v5_paired_bootstrap_auc_input((full, full), output_cap=4)
    with pytest.raises(ValueError, match="output cap is absent"):
        build_v5_paired_bootstrap_auc_input((full,), output_cap=16)

    values = build_v5_paired_bootstrap_auc_input((full,), output_cap=4)
    with pytest.raises(ValueError, match="must be distinct members"):
        summarize_v5_paired_bootstrap_auc(
            values,
            baseline_method_id="baseline",
            comparison_method_id="missing",
            bootstrap_replicates=10,
            rng_seed=1,
        )
    with pytest.raises(ValueError, match="rng_seed"):
        summarize_v5_paired_bootstrap_auc(
            values,
            baseline_method_id="baseline",
            comparison_method_id="comparison",
            bootstrap_replicates=10,
            rng_seed=-1,
        )
