from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import stat

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    LatentComponentParameters,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    CandidateInput,
    LinearSolutionSnapshot,
    ReferenceMode,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_budget_evaluator_v5 import (
    V5CandidateEmission,
    V5ExactForwardCall,
    V5FrozenReferenceRepresentative,
    V5FrozenReferenceSet,
    V5PaperBudgetEvaluationConfig,
    V5PaperParameterRepresentativePayload,
    V5_EMITTED_REPRESENTATIVE_ROLE,
    V5_EXACT_COMPATIBLE,
    V5_REFERENCE_REPRESENTATIVE_ROLE,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_checkpoint_selector_v5 import (
    V5TuningQueryCohort,
    V5TuningQueryCohortMember,
    select_v5_paper_checkpoint,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.paper_endpoint_metrics import (
    EXACT_FORWARD_BUDGETS,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.tuning_checkpoint_runtime_v5 import (
    V5RetainedFullCheckpoint,
    V5TuningExactTraceRunResult,
    evaluate_v5_retained_checkpoints_on_tuning,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.tuning_checkpoint_summary_io_v5 import (
    read_v5_tuning_checkpoint_runtime_result,
)


def _digest(label: str) -> str:
    return sha256(label.encode("utf-8")).hexdigest()


def _linear() -> LinearSolutionSnapshot:
    return LinearSolutionSnapshot(
        background=0.1,
        particle_amplitudes=(1.0,),
        resolution_amplitude=0.0,
        k=1.0,
    )


def _component() -> LatentComponentParameters:
    return LatentComponentParameters(
        shape="sphere",
        log_R=float(np.log(10.0)),
        sigma_R_fraction=0.1,
    )


def _reference_set() -> V5FrozenReferenceSet:
    query_context = _digest("tuning-query-context")
    representative_id = "reference-001"
    payload = V5PaperParameterRepresentativePayload(
        representative_id=representative_id,
        role=V5_REFERENCE_REPRESENTATIVE_ROLE,
        parameter=ReferenceMode(
            reference_id=representative_id,
            topology_id=0,
            components=(_component(),),
            resolution=None,
            linear_solution=_linear(),
        ),
        global_branch_key="topology-00:wire-00",
        query_context_sha256=query_context,
        source_artifact_sha256=_digest("reference-source"),
    )
    return V5FrozenReferenceSet(
        query_id="query-001",
        pairing_unit_id="clean-parent-001",
        reference_set_id="reference-set-001",
        reference_set_sha256=_digest("reference-set-001"),
        comparison_protocol_id="paper-comparison-v1",
        comparison_protocol_sha256=_digest("paper-comparison-v1"),
        representatives=(
            V5FrozenReferenceRepresentative(
                representative_id=representative_id,
                payload=payload,
            ),
        ),
    )


def _cohort(reference: V5FrozenReferenceSet) -> V5TuningQueryCohort:
    return V5TuningQueryCohort(
        cohort_id="frozen-tuning-cohort",
        cohort_artifact_sha256=_digest("frozen-tuning-cohort-artifact"),
        members=(
            V5TuningQueryCohortMember(
                query_id=reference.query_id,
                pairing_unit_id=reference.pairing_unit_id,
                reference_set_id=reference.reference_set_id,
                reference_set_sha256=reference.reference_set_sha256,
                query_context_sha256=reference.query_context_sha256,
                reference_representative_payload_set_sha256=(
                    reference.representative_payload_set_sha256
                ),
            ),
        ),
    )


def _config(reference: V5FrozenReferenceSet) -> V5PaperBudgetEvaluationConfig:
    return V5PaperBudgetEvaluationConfig(
        comparison_protocol_id=reference.comparison_protocol_id,
        comparison_protocol_sha256=reference.comparison_protocol_sha256,
        equivalence_matcher_id="exact-parameter-matcher-v1",
        equivalence_matcher_sha256=_digest("exact-parameter-matcher-v1"),
        equivalence_threshold_id="distance-at-most-one-v1",
        equivalence_threshold_sha256=_digest("distance-at-most-one-v1"),
        maximum_normalized_distance=1.0,
    )


def _checkpoints(tmp_path) -> tuple[V5RetainedFullCheckpoint, ...]:
    values = []
    for epoch in (1, 2):
        path = tmp_path / f"checkpoint-{epoch}.bin"
        path.write_bytes(f"checkpoint-{epoch}".encode("utf-8"))
        values.append(
            V5RetainedFullCheckpoint(
                full_epoch=epoch,
                checkpoint_path=path,
                checkpoint_artifact_sha256=sha256(path.read_bytes()).hexdigest(),
                checkpoint_weights_sha256=_digest(f"checkpoint-{epoch}-weights"),
                training_result_sha256=_digest("training-result"),
            )
        )
    return tuple(values)


def _candidate_emission(reference: V5FrozenReferenceSet) -> V5CandidateEmission:
    candidate_id = "candidate-001"
    payload = V5PaperParameterRepresentativePayload(
        representative_id=candidate_id,
        role=V5_EMITTED_REPRESENTATIVE_ROLE,
        parameter=CandidateInput(
            candidate_id=candidate_id,
            proposal_rank=1,
            topology_id=0,
            components=(_component(),),
            resolution=None,
            linear_solution=_linear(),
            exact_intensity=np.asarray((1.0,), dtype=np.float64),
            bounds_pass=True,
            physics_pass=True,
        ),
        global_branch_key="topology-00:wire-00",
        query_context_sha256=reference.query_context_sha256,
        source_artifact_sha256=_digest("candidate-source"),
    )
    return V5CandidateEmission(
        available_after_call=1,
        output_rank=1,
        candidate_id=candidate_id,
        compatibility_status=V5_EXACT_COMPATIBLE,
        elapsed_seconds=0.015,
        payload=payload,
    )


def _runner(
    checkpoint,
    reference,
    method_binding,
    inference_seed,
    exact_forward_call_budget,
):
    assert exact_forward_call_budget == EXACT_FORWARD_BUDGETS[-1]
    calls = tuple(
        V5ExactForwardCall(
            exact_call_index=index,
            elapsed_seconds=index / 100.0,
        )
        for index in range(1, exact_forward_call_budget + 1)
    )
    return V5TuningExactTraceRunResult(
        trace_id=f"epoch-{checkpoint.full_epoch}-{reference.query_id}",
        checkpoint_artifact_sha256_used=checkpoint.checkpoint_artifact_sha256,
        query_id=reference.query_id,
        method_protocol_sha256_used=method_binding[
            "bound_method_protocol_sha256"
        ],
        inference_seed_used=inference_seed,
        exact_forward_call_budget_used=exact_forward_call_budget,
        exact_forward_calls=calls,
        candidate_emissions=(_candidate_emission(reference),),
        completion_elapsed_seconds=calls[-1].elapsed_seconds,
    )


def _evaluate(tmp_path, *, runner=_runner, output_name="tuning-output"):
    reference = _reference_set()
    source_sha = _digest("source-summary")
    return evaluate_v5_retained_checkpoints_on_tuning(
        checkpoints=_checkpoints(tmp_path),
        query_cohort=_cohort(reference),
        reference_sets=(reference,),
        config=_config(reference),
        equivalence_distance_matcher=lambda _reference, _candidate: 0.0,
        trace_runner=runner,
        method_id="v5.2-model",
        base_method_protocol_id="one-click-frozen-protocol",
        base_method_protocol_sha256=_digest("one-click-frozen-protocol"),
        inference_seed=20260903,
        source_summary_artifact_sha256=source_sha,
        output_root=tmp_path / output_name,
        pre_publish_guard=lambda: source_sha,
    )


def test_runtime_evaluates_every_retained_epoch_and_publishes_complete_inventory(
    tmp_path,
):
    result = _evaluate(tmp_path)

    assert tuple(value.checkpoint_epoch for value in result.evaluations) == (1, 2)
    assert len(result.trace_paths) == len(result.trace_artifact_sha256s) == 2
    assert len(result.lossless_emission_paths) == 2
    assert len(result.lossless_emission_file_sha256s) == 2
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o400
        and path.stat().st_nlink == 1
        for path in result.lossless_emission_paths
    )
    lossless = json.loads(
        result.lossless_emission_paths[0].read_text(encoding="utf-8")
    )
    assert lossless["parameter"]["exact_intensity_dtype"] == "<f8"
    assert lossless["parameter"]["exact_intensity"] == [1.0]
    assert len(result.summary_paths) == len(result.summary_file_sha256s) == 2
    assert all(path.is_file() for path in result.trace_paths + result.summary_paths)
    completion = json.loads(result.completion_path.read_text(encoding="utf-8"))
    assert completion["status"] == "all_retained_full_epochs_exact_budget_evaluated"
    assert completion["expected_exact_trace_count"] == 2
    assert len(completion["exact_trace_artifacts"]) == 2
    assert len(completion["lossless_emission_artifacts"]) == 2
    assert len(completion["checkpoint_summaries"]) == 2
    assert completion["validation_loss_used"] is False
    assert completion["test_calibration_reference_or_ood_used"] is False
    assert result.completion_file_sha256 == sha256(
        result.completion_path.read_bytes()
    ).hexdigest()
    replay = read_v5_tuning_checkpoint_runtime_result(result.output_root)
    assert tuple(value.sha256 for value in replay.evaluations) == tuple(
        value.sha256 for value in result.evaluations
    )

    selection = select_v5_paper_checkpoint(
        result.evaluations,
        selection_id="runtime-selection",
        expected_checkpoint_epochs=(1, 2),
    )
    assert selection.inference_seed == 20260903


def test_runtime_rejects_runner_seed_escape_and_never_writes_completion(tmp_path):
    def wrong_seed_runner(*args, **kwargs):
        return replace(_runner(*args, **kwargs), inference_seed_used=7)

    with pytest.raises(ValueError, match="seed/budget binding"):
        _evaluate(tmp_path, runner=wrong_seed_runner)

    assert not (tmp_path / "tuning-output" / "completion.json").exists()
    with pytest.raises(ValueError, match="wrong filesystem type"):
        read_v5_tuning_checkpoint_runtime_result(tmp_path / "tuning-output")


def test_runtime_detects_checkpoint_toctou_and_never_writes_completion(tmp_path):
    def mutating_runner(checkpoint, *args, **kwargs):
        result = _runner(checkpoint, *args, **kwargs)
        checkpoint.checkpoint_path.write_bytes(b"changed-during-exact-trace")
        return result

    with pytest.raises(RuntimeError, match="changed while"):
        _evaluate(tmp_path, runner=mutating_runner)

    assert not (tmp_path / "tuning-output" / "completion.json").exists()


def test_runtime_reader_rejects_trace_mutation_despite_existing_completion(tmp_path):
    result = _evaluate(tmp_path)
    trace_path = result.trace_paths[0]
    trace_path.write_bytes(trace_path.read_bytes() + b" \n")

    with pytest.raises(ValueError, match="file SHA-256 changed"):
        read_v5_tuning_checkpoint_runtime_result(result.output_root)


def test_runtime_rejects_query_context_or_reference_payload_set_swap(tmp_path):
    reference = _reference_set()
    cohort = _cohort(reference)
    member = cohort.members[0]
    escaped = replace(
        cohort,
        members=(
            replace(
                member,
                reference_representative_payload_set_sha256=_digest(
                    "another-reference-payload-set"
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="context/payload identity"):
        evaluate_v5_retained_checkpoints_on_tuning(
            checkpoints=_checkpoints(tmp_path),
            query_cohort=escaped,
            reference_sets=(reference,),
            config=_config(reference),
            equivalence_distance_matcher=lambda _reference, _candidate: 0.0,
            trace_runner=_runner,
            method_id="v5.2-model",
            base_method_protocol_id="one-click-frozen-protocol",
            base_method_protocol_sha256=_digest("one-click-frozen-protocol"),
            inference_seed=20260903,
            source_summary_artifact_sha256=_digest("source-summary"),
            output_root=tmp_path / "escaped-output",
            pre_publish_guard=lambda: _digest("source-summary"),
        )

    assert not (tmp_path / "escaped-output").exists()
