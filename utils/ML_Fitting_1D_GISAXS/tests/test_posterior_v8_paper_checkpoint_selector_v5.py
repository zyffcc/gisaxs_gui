from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_budget_evaluator_v5 import (
    V5_EXACT_COMPATIBLE,
    V5_EXACT_INCOMPATIBLE,
    V5_EXACT_UNVERIFIED,
    V5PairedPaperBudgetQueryRecord,
    V5PaperBudgetMethodResult,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_checkpoint_selector_v5 import (
    V5CompletedTuningTrace,
    V5TuningCheckpointEvaluation,
    V5TuningQueryCohort,
    V5TuningQueryCohortMember,
    build_v5_checkpoint_evaluation_method_binding,
    select_v5_paper_checkpoint,
    v5_paper_checkpoint_selection_rule,
    write_v5_paper_checkpoint_selection_receipt,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_endpoint_metrics import (
    EXACT_FORWARD_BUDGETS,
    OUTPUT_CAPS,
    normalized_log2_budget_auc,
)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _cohort(
    *,
    cohort_id: str = "frozen-tuning-cohort",
    units: tuple[str, ...] = ("recipe-01", "recipe-02"),
    selection_split: str = "tuning_validation",
) -> V5TuningQueryCohort:
    return V5TuningQueryCohort(
        cohort_id=cohort_id,
        cohort_artifact_sha256=_digest(f"{cohort_id}-artifact"),
        selection_split=selection_split,
        members=tuple(
            V5TuningQueryCohortMember(
                query_id=f"query-{index:02d}",
                pairing_unit_id=unit,
                reference_set_id=f"references-{index:02d}",
                reference_set_sha256=_digest(f"references-{index:02d}"),
                query_context_sha256=_digest(f"query-context-query-{index:02d}"),
                reference_representative_payload_set_sha256=_digest(
                    f"reference-payload-set-references-{index:02d}"
                ),
            )
            for index, unit in enumerate(units, start=1)
        ),
    )


def _result(
    *,
    trace_id: str,
    primary_hits: int,
    first_call: int | None,
    first_seconds: float | None,
    method_protocol_id: str,
    method_protocol_sha256: str,
    stored_primary_auc: float | None = None,
) -> V5PaperBudgetMethodResult:
    reference_count = 4
    hit_rows = tuple(
        tuple(min(cap, primary_hits) for _ in EXACT_FORWARD_BUDGETS) for cap in OUTPUT_CAPS
    )
    recall_rows = tuple(
        tuple(hit / reference_count for hit in row) for row in hit_rows
    )
    auc_rows = tuple(
        (
            cap,
            normalized_log2_budget_auc(
                dict(zip(EXACT_FORWARD_BUDGETS, recalls)), budgets=EXACT_FORWARD_BUDGETS
            ),
        )
        for cap, recalls in zip(OUTPUT_CAPS, recall_rows)
    )
    if stored_primary_auc is not None:
        auc_rows = tuple(
            (cap, stored_primary_auc if cap == 16 else auc) for cap, auc in auc_rows
        )
    compatible = max(primary_hits, 1) if first_call is not None else 0
    return V5PaperBudgetMethodResult(
        method_id="v5.2-model",
        method_protocol_id=method_protocol_id,
        method_protocol_sha256=method_protocol_sha256,
        trace_id=trace_id,
        trace_artifact_sha256=_digest(f"{trace_id}-artifact"),
        trace_ledger_sha256=_digest(f"{trace_id}-ledger"),
        hit_count_matrix=hit_rows,
        recall_matrix=recall_rows,
        recall_ceiling_by_output_cap=tuple(
            (cap, min(cap, reference_count) / reference_count) for cap in OUTPUT_CAPS
        ),
        selected_candidate_ids_matrix=tuple(
            tuple(tuple() for _ in EXACT_FORWARD_BUDGETS) for _ in OUTPUT_CAPS
        ),
        matched_pairs_matrix=tuple(
            tuple(tuple() for _ in EXACT_FORWARD_BUDGETS) for _ in OUTPUT_CAPS
        ),
        matched_distance_sum_matrix=tuple(
            tuple(0.0 for _ in EXACT_FORWARD_BUDGETS) for _ in OUTPUT_CAPS
        ),
        auc_by_output_cap=auc_rows,
        first_compatible_exact_call=first_call,
        time_to_first_compatible_seconds=first_seconds,
        exact_forward_calls_evaluated=EXACT_FORWARD_BUDGETS[-1],
        candidate_emissions_evaluated=compatible,
        compatibility_status_counts=(
            (V5_EXACT_COMPATIBLE, compatible),
            (V5_EXACT_INCOMPATIBLE, 0),
            (V5_EXACT_UNVERIFIED, 0),
        ),
    )


def _record(
    member,
    *,
    epoch,
    hits,
    first_call,
    first_seconds,
    method_protocol_id,
    method_protocol_sha256,
    stored_auc=None,
):
    trace_id = f"epoch-{epoch}-{member.query_id}"
    result = _result(
        trace_id=trace_id,
        primary_hits=hits,
        first_call=first_call,
        first_seconds=first_seconds,
        method_protocol_id=method_protocol_id,
        method_protocol_sha256=method_protocol_sha256,
        stored_primary_auc=stored_auc,
    )
    return V5PairedPaperBudgetQueryRecord(
        query_id=member.query_id,
        query_context_sha256=_digest(f"query-context-{member.query_id}"),
        pairing_unit_id=member.pairing_unit_id,
        reference_set_id=member.reference_set_id,
        reference_set_sha256=member.reference_set_sha256,
        reference_representative_ids=("ref-1", "ref-2", "ref-3", "ref-4"),
        reference_representative_payload_set_sha256=_digest(
            f"reference-payload-set-{member.reference_set_id}"
        ),
        evaluator_config_sha256=_digest("evaluator-config"),
        comparison_protocol_id="comparison-protocol",
        comparison_protocol_sha256=_digest("comparison-protocol"),
        equivalence_matcher_id="normalized-distance-matcher",
        equivalence_matcher_sha256=_digest("normalized-distance-matcher"),
        equivalence_threshold_id="threshold-v1",
        equivalence_threshold_sha256=_digest("threshold-v1"),
        maximum_normalized_distance=0.1,
        output_caps=OUTPUT_CAPS,
        exact_forward_budgets=EXACT_FORWARD_BUDGETS,
        method_results=(result,),
    )


def _evaluation(
    epoch: int,
    *,
    cohort: V5TuningQueryCohort | None = None,
    hits: int | tuple[int, ...] = 1,
    first_call: int | None = 300,
    first_seconds: float | None = 30.0,
    completion_seconds: float = 60.0,
    inference_seed: int = 20260903,
    source_summary_label: str = "summary-artifact",
) -> V5TuningCheckpointEvaluation:
    selected_cohort = cohort or _cohort()
    hit_values = (hits,) * len(selected_cohort.members) if isinstance(hits, int) else hits
    checkpoint_artifact_sha256 = _digest(f"checkpoint-{epoch}")
    checkpoint_weights_sha256 = _digest(f"weights-{epoch}")
    training_result_sha256 = _digest("training-result")
    source_summary_artifact_sha256 = _digest(source_summary_label)
    base_method_protocol_id = "one-click-frozen-protocol"
    base_method_protocol_sha256 = _digest(base_method_protocol_id)
    binding = build_v5_checkpoint_evaluation_method_binding(
        checkpoint_epoch=epoch,
        checkpoint_artifact_sha256=checkpoint_artifact_sha256,
        checkpoint_weights_sha256=checkpoint_weights_sha256,
        training_result_sha256=training_result_sha256,
        source_summary_artifact_sha256=source_summary_artifact_sha256,
        method_id="v5.2-model",
        base_method_protocol_id=base_method_protocol_id,
        base_method_protocol_sha256=base_method_protocol_sha256,
        inference_seed=inference_seed,
    )
    records = tuple(
        _record(
            member,
            epoch=epoch,
            hits=hit,
            first_call=first_call,
            first_seconds=first_seconds,
            method_protocol_id=binding["bound_method_protocol_id"],
            method_protocol_sha256=binding["bound_method_protocol_sha256"],
        )
        for member, hit in zip(selected_cohort.members, hit_values)
    )
    completions = tuple(
        V5CompletedTuningTrace(
            trace_id=result.trace_id,
            trace_artifact_sha256=result.trace_artifact_sha256,
            trace_ledger_sha256=result.trace_ledger_sha256,
            completion_elapsed_seconds=completion_seconds,
        )
        for record in records
        for result in record.method_results
    )
    return V5TuningCheckpointEvaluation(
        checkpoint_epoch=epoch,
        checkpoint_artifact_sha256=checkpoint_artifact_sha256,
        checkpoint_weights_sha256=checkpoint_weights_sha256,
        training_result_sha256=training_result_sha256,
        source_summary_artifact_sha256=source_summary_artifact_sha256,
        method_id="v5.2-model",
        base_method_protocol_id=base_method_protocol_id,
        base_method_protocol_sha256=base_method_protocol_sha256,
        inference_seed=inference_seed,
        query_cohort=selected_cohort,
        paired_query_records=records,
        trace_completions=completions,
    )


def test_selects_auc_winner_deterministically_and_writes_selection_only_receipt(tmp_path):
    first = _evaluation(1, hits=1)
    second = _evaluation(2, hits=2)

    forward = select_v5_paper_checkpoint(
        (first, second), selection_id="paper-selection", expected_checkpoint_epochs=(1, 2)
    )
    reverse = select_v5_paper_checkpoint(
        (second, first), selection_id="paper-selection", expected_checkpoint_epochs=(1, 2)
    )

    assert forward.to_json() == reverse.to_json()
    assert forward.selected_checkpoint_epoch == 2
    assert forward.selection_complete is True
    assert forward.paper_model_eligibility_asserted is False
    assert forward.qualification_status == "checkpoint_selected_paper_eligibility_not_evaluated"
    assert forward.rule_sha256 == v5_paper_checkpoint_selection_rule()["rule_sha256"]
    assert json.loads(forward.to_json())["receipt_sha256"] == forward.sha256
    with pytest.raises(ValueError, match="selection only"):
        replace(forward, paper_model_eligibility_asserted=True)

    target = tmp_path / "selection.json"
    write_v5_paper_checkpoint_selection_receipt(target, forward)
    assert target.read_text(encoding="utf-8") == forward.to_json() + "\n"
    with pytest.raises(FileExistsError, match="overwrite"):
        write_v5_paper_checkpoint_selection_receipt(target, forward)


def test_uses_exact_then_wall_then_earliest_epoch_tie_breaks():
    candidates = (
        _evaluation(1, first_call=300, first_seconds=10.0),
        _evaluation(2, first_call=200, first_seconds=20.0),
        _evaluation(3, first_call=200, first_seconds=15.0),
        _evaluation(4, first_call=200, first_seconds=15.0),
    )

    receipt = select_v5_paper_checkpoint(
        tuple(reversed(candidates)),
        selection_id="tie-break",
        expected_checkpoint_epochs=(1, 2, 3, 4),
    )

    assert receipt.selected_checkpoint_epoch == 3


def test_missing_hits_use_common_worst_censors():
    hit = _evaluation(1, first_call=4096, first_seconds=59.0, completion_seconds=60.0)
    miss = _evaluation(
        2,
        hits=0,
        first_call=None,
        first_seconds=None,
        completion_seconds=1.0,
    )

    receipt = select_v5_paper_checkpoint(
        (miss, hit), selection_id="censor", expected_checkpoint_epochs=(1, 2)
    )
    scores = {value.checkpoint_epoch: value for value in receipt.candidate_scores}

    assert scores[2].median_exact_calls_to_first_compatible == 4097
    assert scores[2].median_wall_seconds_to_first_compatible == 61.0
    assert receipt.selected_checkpoint_epoch == 1


def test_primary_metric_is_pairing_unit_macro_not_query_weighted_mean():
    cohort = _cohort(units=("recipe-a", "recipe-a", "recipe-b"))
    candidate = _evaluation(1, cohort=cohort, hits=(0, 0, 2))

    receipt = select_v5_paper_checkpoint(
        (candidate,), selection_id="macro", expected_checkpoint_epochs=(1,)
    )

    assert receipt.candidate_scores[0].primary_pairing_unit_macro_auc == pytest.approx(0.25)


@pytest.mark.parametrize("epochs", [(1,), (1, 2, 4)])
def test_rejects_missing_candidate_or_noncontiguous_expected_epoch_inventory(epochs):
    with pytest.raises(ValueError, match="full-epoch|missing"):
        select_v5_paper_checkpoint(
            (_evaluation(1), _evaluation(2)),
            selection_id="missing",
            expected_checkpoint_epochs=epochs,
        )


def test_rejects_wrong_budget_grid_and_forged_auc():
    valid = _evaluation(1)
    bad_budget_record = replace(
        valid.paired_query_records[0], exact_forward_budgets=(256, 512, 1024, 2048)
    )
    with pytest.raises(ValueError, match="budget grid"):
        replace(
            valid,
            paired_query_records=(bad_budget_record, valid.paired_query_records[1]),
        )

    first = valid.paired_query_records[0]
    bad_result = replace(first.method_results[0], auc_by_output_cap=((1, 0.25), (4, 0.25), (8, 0.25), (16, 0.9)))
    with pytest.raises(ValueError, match="does not replay"):
        replace(
            valid,
            paired_query_records=(replace(first, method_results=(bad_result,)), valid.paired_query_records[1]),
        )


def test_rejects_non_tuning_or_mismatched_cohort_and_failed_trace():
    with pytest.raises(ValueError, match="only tuning_validation"):
        _cohort(selection_split="test")

    first = _evaluation(1)
    other = _evaluation(2, cohort=_cohort(cohort_id="other-cohort"))
    with pytest.raises(ValueError, match="one tuning"):
        select_v5_paper_checkpoint(
            (first, other), selection_id="mixed", expected_checkpoint_epochs=(1, 2)
        )

    completion = first.trace_completions[0]
    with pytest.raises(ValueError, match="completed"):
        replace(completion, status="failed")


@pytest.mark.parametrize(
    ("field", "label"),
    (
        ("query_context_sha256", "swapped-query-context"),
        (
            "reference_representative_payload_set_sha256",
            "swapped-reference-payload-set",
        ),
    ),
)
def test_rejects_epoch_record_context_or_reference_payload_identity_swap(field, label):
    valid = _evaluation(1)
    escaped_record = replace(
        valid.paired_query_records[0],
        **{field: _digest(label)},
    )

    with pytest.raises(ValueError, match="context/reference-payload binding"):
        replace(
            valid,
            paired_query_records=(escaped_record, valid.paired_query_records[1]),
        )


@pytest.mark.parametrize(
    "escaped",
    (
        _evaluation(2, inference_seed=20260904),
        _evaluation(2, source_summary_label="another-source-summary"),
    ),
)
def test_rejects_cross_epoch_inference_seed_or_source_summary_swap(escaped):
    with pytest.raises(ValueError, match="one tuning"):
        select_v5_paper_checkpoint(
            (_evaluation(1), escaped),
            selection_id="mixed-checkpoint-condition",
            expected_checkpoint_epochs=(1, 2),
        )


def test_rejects_missing_or_forged_hashes_nan_and_trace_coverage():
    with pytest.raises(ValueError, match="SHA-256"):
        replace(_evaluation(1), checkpoint_weights_sha256="missing")
    with pytest.raises(ValueError, match="finite"):
        replace(_evaluation(1).trace_completions[0], completion_elapsed_seconds=float("nan"))

    valid = _evaluation(1)
    with pytest.raises(ValueError, match="exactly cover"):
        replace(valid, trace_completions=valid.trace_completions[:-1])


def test_rejects_trace_grafted_from_another_checkpoint():
    first = _evaluation(1)
    second = _evaluation(2)
    grafted_record = replace(
        second.paired_query_records[0],
        method_results=first.paired_query_records[0].method_results,
    )

    with pytest.raises(ValueError, match="not bound to this exact checkpoint"):
        replace(
            second,
            paired_query_records=(grafted_record, second.paired_query_records[1]),
        )
