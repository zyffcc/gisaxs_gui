"""Deterministic external checkpoint selection for the V5.2 paper model.

The selector consumes completed, reference-bound *tuning* budget-evaluation
records.  It deliberately knows nothing about training/validation loss and it
does not promote the selected checkpoint to a paper-qualified model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from math import fsum, isfinite
from numbers import Integral, Real
import os
from pathlib import Path
import re
from statistics import median
from typing import Sequence

from .grouped_artifact_v5 import canonical_json
from .paper_budget_evaluator_v5 import (
    V5_EXACT_COMPATIBILITY_STATUSES,
    V5_EXACT_COMPATIBLE,
    V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
    V5_PAPER_BUDGET_EVALUATOR_VERSION,
    V5PairedPaperBudgetQueryRecord,
    V5PaperBudgetMethodResult,
)
from .paper_endpoint_metrics import (
    EXACT_FORWARD_BUDGETS,
    OUTPUT_CAPS,
    PRIMARY_OUTPUT_CAP,
    normalized_log2_budget_auc,
)
V5_PAPER_CHECKPOINT_SELECTOR_SCHEMA = "gisaxs.posterior_v8.paper_checkpoint_selector/v2"
V5_PAPER_CHECKPOINT_SELECTOR_VERSION = (
    "external_tuning_context_payload_seed_bound_budget_auc_ttfc_selection_v2"
)
V5_PAPER_CHECKPOINT_SELECTION_RULE_ID = (
    "recipe_macro_n16_auc_exact_ttfc_wall_ttfc_epoch_artifact_sha/v1"
)
V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA = (
    "gisaxs.posterior_v8.tuning_checkpoint_budget_summary/v2"
)
V5_TUNING_CHECKPOINT_SUMMARY_VERSION = (
    "complete_exact_trace_same_cohort_method_seed_budget_v2"
)
V5_CHECKPOINT_EVALUATION_METHOD_BINDING_SCHEMA = (
    "gisaxs.posterior_v8.checkpoint_evaluation_method_binding/v2"
)
V5_CHECKPOINT_BOUND_METHOD_PROTOCOL_ID = "v5.2-checkpoint-evaluation-method-binding/v2"
V5_TUNING_QUERY_COHORT_SCHEMA = "gisaxs.posterior_v8.tuning_query_cohort/v2"
V5_TUNING_QUERY_COHORT_VERSION = (
    "query_context_and_reference_payload_set_bound_tuning_cohort_v2"
)
V5_TUNING_SELECTION_SPLIT = "tuning_validation"
V5_TRACE_COMPLETED = "completed_schema_hash_budget_verified"
V5_SELECTION_ONLY_STATUS = "checkpoint_selected_paper_eligibility_not_evaluated"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty stripped string")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _inference_seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("inference_seed must be an integer")
    result = int(value)
    if not 0 <= result < 2**64:
        raise ValueError("inference_seed must be in [0, 2**64)")
    return result


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def v5_paper_checkpoint_selection_rule() -> dict[str, object]:
    """Return the frozen, self-verifying selection rule."""

    # Lazy import lets the methods manifest bind this implementation's owned
    # constants without creating a selector <-> study-protocol import cycle.
    from .study_protocol import (
        STUDY_PROTOCOL_SCHEMA,
        STUDY_PROTOCOL_VERSION,
        protocol_payload,
    )

    study = protocol_payload()
    core = {
        "schema": V5_PAPER_CHECKPOINT_SELECTOR_SCHEMA,
        "version": V5_PAPER_CHECKPOINT_SELECTOR_VERSION,
        "rule_id": V5_PAPER_CHECKPOINT_SELECTION_RULE_ID,
        "study_protocol_schema": STUDY_PROTOCOL_SCHEMA,
        "study_protocol_version": STUDY_PROTOCOL_VERSION,
        "study_protocol_sha256": study["protocol_sha256"],
        "selection_split": V5_TUNING_SELECTION_SPLIT,
        "same_inference_seed_for_every_retained_checkpoint": True,
        "evaluator_schema": V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
        "evaluator_version": V5_PAPER_BUDGET_EVALUATOR_VERSION,
        "output_caps": OUTPUT_CAPS,
        "exact_forward_budgets": EXACT_FORWARD_BUDGETS,
        "primary_output_cap": PRIMARY_OUTPUT_CAP,
        "primary": "maximize_pairing_unit_macro_mean_log2_budget_recall_auc",
        "secondary": "minimize_median_right_censored_exact_calls_to_first_compatible",
        "tertiary": "minimize_median_right_censored_wall_seconds_to_first_compatible",
        "final_tie_break": "earliest_full_epoch_then_lexicographic_checkpoint_artifact_sha256",
        "exact_call_no_hit_censor": EXACT_FORWARD_BUDGETS[-1] + 1,
        "wall_no_hit_censor": "global_max_completed_trace_seconds_plus_one",
        "forbidden_selection_inputs": (
            "training_validation_loss",
            "test_or_reference_holdout_split_truth",
            "calibration_split_truth",
            "real_cut_data",
        ),
        "selection_claim": "selection_complete_only_not_final_paper_qualification",
    }
    return {
        **core,
        "rule_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest(),
    }


def build_v5_checkpoint_evaluation_method_binding(
    *,
    checkpoint_epoch: int,
    checkpoint_artifact_sha256: str,
    checkpoint_weights_sha256: str,
    training_result_sha256: str,
    source_summary_artifact_sha256: str,
    method_id: str,
    base_method_protocol_id: str,
    base_method_protocol_sha256: str,
    inference_seed: int,
) -> dict[str, object]:
    """Bind a trace protocol to one exact checkpoint and its frozen base method."""

    core = {
        "schema": V5_CHECKPOINT_EVALUATION_METHOD_BINDING_SCHEMA,
        "bound_method_protocol_id": V5_CHECKPOINT_BOUND_METHOD_PROTOCOL_ID,
        "checkpoint_epoch": _positive_integer(checkpoint_epoch, "checkpoint_epoch"),
        "checkpoint_artifact_sha256": _digest(
            checkpoint_artifact_sha256, "checkpoint_artifact_sha256"
        ),
        "checkpoint_weights_sha256": _digest(
            checkpoint_weights_sha256, "checkpoint_weights_sha256"
        ),
        "training_result_sha256": _digest(training_result_sha256, "training_result_sha256"),
        "source_summary_artifact_sha256": _digest(
            source_summary_artifact_sha256, "source_summary_artifact_sha256"
        ),
        "method_id": _text(method_id, "method_id"),
        "base_method_protocol_id": _text(base_method_protocol_id, "base_method_protocol_id"),
        "base_method_protocol_sha256": _digest(
            base_method_protocol_sha256, "base_method_protocol_sha256"
        ),
        "inference_seed": _inference_seed(inference_seed),
    }
    return {
        **core,
        "bound_method_protocol_sha256": sha256(
            canonical_json(core).encode("utf-8")
        ).hexdigest(),
    }


@dataclass(frozen=True, kw_only=True)
class V5TuningQueryCohortMember:
    query_id: str
    pairing_unit_id: str
    reference_set_id: str
    reference_set_sha256: str
    query_context_sha256: str
    reference_representative_payload_set_sha256: str

    def __post_init__(self) -> None:
        for name in ("query_id", "pairing_unit_id", "reference_set_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "reference_set_sha256",
            "query_context_sha256",
            "reference_representative_payload_set_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))


@dataclass(frozen=True, kw_only=True)
class V5TuningQueryCohort:
    cohort_id: str
    cohort_artifact_sha256: str
    members: tuple[V5TuningQueryCohortMember, ...]
    selection_split: str = V5_TUNING_SELECTION_SPLIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohort_id", _text(self.cohort_id, "cohort_id"))
        object.__setattr__(
            self,
            "cohort_artifact_sha256",
            _digest(self.cohort_artifact_sha256, "cohort_artifact_sha256"),
        )
        if self.selection_split != V5_TUNING_SELECTION_SPLIT:
            raise ValueError("checkpoint selection may use only tuning_validation")
        members = tuple(self.members)
        if not members or not all(isinstance(value, V5TuningQueryCohortMember) for value in members):
            raise ValueError("members must contain V5TuningQueryCohortMember values")
        query_ids = tuple(value.query_id for value in members)
        if len(query_ids) != len(set(query_ids)):
            raise ValueError("tuning cohort query IDs must be unique")
        object.__setattr__(self, "members", tuple(sorted(members, key=lambda value: value.query_id)))

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": V5_TUNING_QUERY_COHORT_SCHEMA,
            "version": V5_TUNING_QUERY_COHORT_VERSION,
            "selection_split": self.selection_split,
            "cohort_id": self.cohort_id,
            "cohort_artifact_sha256": self.cohort_artifact_sha256,
            "members": [asdict(value) for value in self.members],
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5CompletedTuningTrace:
    trace_id: str
    trace_artifact_sha256: str
    trace_ledger_sha256: str
    completion_elapsed_seconds: float
    status: str = V5_TRACE_COMPLETED

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _text(self.trace_id, "trace_id"))
        for name in ("trace_artifact_sha256", "trace_ledger_sha256"):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "completion_elapsed_seconds",
            _finite_nonnegative(self.completion_elapsed_seconds, "completion_elapsed_seconds"),
        )
        if self.status != V5_TRACE_COMPLETED:
            raise ValueError("every tuning trace must be completed and schema/hash/budget verified")


def _validate_result(
    result: V5PaperBudgetMethodResult,
    record: V5PairedPaperBudgetQueryRecord,
) -> float:
    for name in ("method_id", "method_protocol_id", "trace_id"):
        _text(getattr(result, name), name)
    for name in ("method_protocol_sha256", "trace_artifact_sha256", "trace_ledger_sha256"):
        _digest(getattr(result, name), name)
    if result.exact_forward_calls_evaluated != EXACT_FORWARD_BUDGETS[-1]:
        raise ValueError("tuning trace did not complete the frozen exact-forward budget")
    if isinstance(result.candidate_emissions_evaluated, bool) or not isinstance(
        result.candidate_emissions_evaluated, Integral
    ) or result.candidate_emissions_evaluated < 0:
        raise ValueError("candidate_emissions_evaluated must be a non-negative integer")
    if len(result.hit_count_matrix) != len(OUTPUT_CAPS) or len(result.recall_matrix) != len(
        OUTPUT_CAPS
    ):
        raise ValueError("method result matrix does not cover the frozen output-cap grid")
    reference_count = len(record.reference_representative_ids)
    if reference_count < 1:
        raise ValueError("tuning record requires at least one reference representative")
    if len(set(record.reference_representative_ids)) != reference_count:
        raise ValueError("reference representative IDs must be unique")
    for cap, hits, recalls in zip(OUTPUT_CAPS, result.hit_count_matrix, result.recall_matrix):
        if len(hits) != len(EXACT_FORWARD_BUDGETS) or len(recalls) != len(
            EXACT_FORWARD_BUDGETS
        ):
            raise ValueError("method result matrix does not cover the frozen budget grid")
        prior = -1.0
        for hit, recall in zip(hits, recalls):
            if isinstance(hit, bool) or not isinstance(hit, Integral):
                raise TypeError("hit counts must be integers")
            observed = float(recall)
            if not isfinite(observed) or not 0.0 <= observed <= 1.0:
                raise ValueError("recall values must be finite and in [0, 1]")
            if int(hit) < 0 or int(hit) > min(cap, reference_count):
                raise ValueError("hit count exceeds the reference/output-cap limit")
            if observed != int(hit) / reference_count:
                raise ValueError("recall matrix is inconsistent with hit counts")
            if observed < prior:
                raise ValueError("recall must be non-decreasing with exact-forward budget")
            prior = observed
    aucs = tuple(result.auc_by_output_cap)
    if len(aucs) != len(OUTPUT_CAPS) or tuple(value[0] for value in aucs) != OUTPUT_CAPS:
        raise ValueError("method result AUCs must exactly cover the frozen output-cap grid")
    for cap_index, (cap, stored_auc) in enumerate(aucs):
        observed = float(stored_auc)
        if not isfinite(observed) or not 0.0 <= observed <= 1.0:
            raise ValueError("method AUC must be finite and in [0, 1]")
        expected = normalized_log2_budget_auc(
            dict(zip(EXACT_FORWARD_BUDGETS, result.recall_matrix[cap_index])),
            budgets=EXACT_FORWARD_BUDGETS,
        )
        if observed != expected:
            raise ValueError("stored method AUC does not replay from the recall matrix")
    counts = tuple(result.compatibility_status_counts)
    if tuple(value[0] for value in counts) != V5_EXACT_COMPATIBILITY_STATUSES:
        raise ValueError("compatibility status counts are incomplete or out of order")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) or value < 0
        for _, value in counts
    ):
        raise ValueError("compatibility status counts must be non-negative integers")
    if sum(value for _, value in counts) != result.candidate_emissions_evaluated:
        raise ValueError("compatibility status counts do not cover all candidate emissions")
    compatible_count = dict(counts)[V5_EXACT_COMPATIBLE]
    if any(hit > compatible_count for row in result.hit_count_matrix for hit in row):
        raise ValueError("hit count exceeds the number of exact-compatible emissions")
    if (result.first_compatible_exact_call is None) != (compatible_count == 0):
        raise ValueError("first-compatible call is inconsistent with compatibility counts")
    if (result.time_to_first_compatible_seconds is None) != (compatible_count == 0):
        raise ValueError("first-compatible time is inconsistent with compatibility counts")
    if compatible_count:
        call = _positive_integer(
            result.first_compatible_exact_call, "first_compatible_exact_call"
        )
        if call > EXACT_FORWARD_BUDGETS[-1]:
            raise ValueError("first-compatible call leaks beyond the frozen budget")
        _finite_nonnegative(
            result.time_to_first_compatible_seconds,
            "time_to_first_compatible_seconds",
        )
    return dict(aucs)[PRIMARY_OUTPUT_CAP]


@dataclass(frozen=True, kw_only=True)
class V5TuningCheckpointEvaluation:
    checkpoint_epoch: int
    checkpoint_artifact_sha256: str
    checkpoint_weights_sha256: str
    training_result_sha256: str
    source_summary_artifact_sha256: str
    method_id: str
    base_method_protocol_id: str
    base_method_protocol_sha256: str
    inference_seed: int
    query_cohort: V5TuningQueryCohort
    paired_query_records: tuple[V5PairedPaperBudgetQueryRecord, ...]
    trace_completions: tuple[V5CompletedTuningTrace, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checkpoint_epoch", _positive_integer(self.checkpoint_epoch, "checkpoint_epoch")
        )
        for name in (
            "checkpoint_artifact_sha256",
            "checkpoint_weights_sha256",
            "training_result_sha256",
            "source_summary_artifact_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "method_id", _text(self.method_id, "method_id"))
        object.__setattr__(
            self,
            "base_method_protocol_id",
            _text(self.base_method_protocol_id, "base_method_protocol_id"),
        )
        object.__setattr__(
            self,
            "base_method_protocol_sha256",
            _digest(self.base_method_protocol_sha256, "base_method_protocol_sha256"),
        )
        object.__setattr__(self, "inference_seed", _inference_seed(self.inference_seed))
        if not isinstance(self.query_cohort, V5TuningQueryCohort):
            raise TypeError("query_cohort must be a V5TuningQueryCohort")
        method_binding = build_v5_checkpoint_evaluation_method_binding(
            checkpoint_epoch=self.checkpoint_epoch,
            checkpoint_artifact_sha256=self.checkpoint_artifact_sha256,
            checkpoint_weights_sha256=self.checkpoint_weights_sha256,
            training_result_sha256=self.training_result_sha256,
            source_summary_artifact_sha256=self.source_summary_artifact_sha256,
            method_id=self.method_id,
            base_method_protocol_id=self.base_method_protocol_id,
            base_method_protocol_sha256=self.base_method_protocol_sha256,
            inference_seed=self.inference_seed,
        )
        records = tuple(self.paired_query_records)
        if not records or not all(isinstance(v, V5PairedPaperBudgetQueryRecord) for v in records):
            raise ValueError("paired_query_records must contain evaluated tuning records")
        if len({value.query_id for value in records}) != len(records):
            raise ValueError("paired tuning query records must be unique")
        records = tuple(sorted(records, key=lambda value: value.query_id))
        expected_members = {value.query_id: value for value in self.query_cohort.members}
        if {value.query_id for value in records} != set(expected_members):
            raise ValueError("tuning records do not exactly cover the frozen query cohort")
        contracts = set()
        selected_results = []
        for record in records:
            # This also rejects NaN or non-JSON content anywhere in the upstream record.
            record.sha256
            member = expected_members[record.query_id]
            if (
                record.pairing_unit_id,
                record.reference_set_id,
                record.reference_set_sha256,
                record.query_context_sha256,
                record.reference_representative_payload_set_sha256,
            ) != (
                member.pairing_unit_id,
                member.reference_set_id,
                member.reference_set_sha256,
                member.query_context_sha256,
                member.reference_representative_payload_set_sha256,
            ):
                raise ValueError(
                    "tuning record escaped its frozen cohort/context/reference-payload binding"
                )
            if record.output_caps != OUTPUT_CAPS or record.exact_forward_budgets != EXACT_FORWARD_BUDGETS:
                raise ValueError("tuning record does not use the frozen paper budget grid")
            _digest(record.evaluator_config_sha256, "evaluator_config_sha256")
            for name in (
                "comparison_protocol_id",
                "equivalence_matcher_id",
                "equivalence_threshold_id",
            ):
                _text(getattr(record, name), name)
            for name in (
                "comparison_protocol_sha256",
                "equivalence_matcher_sha256",
                "equivalence_threshold_sha256",
            ):
                _digest(getattr(record, name), name)
            _finite_nonnegative(record.maximum_normalized_distance, "maximum_normalized_distance")
            methods = tuple(value for value in record.method_results if value.method_id == self.method_id)
            if len(methods) != 1 or len({value.method_id for value in record.method_results}) != len(
                record.method_results
            ):
                raise ValueError("each tuning record must contain one unique selected method")
            result = methods[0]
            _validate_result(result, record)
            if (
                result.method_protocol_id,
                result.method_protocol_sha256,
            ) != (
                method_binding["bound_method_protocol_id"],
                method_binding["bound_method_protocol_sha256"],
            ):
                raise ValueError("tuning trace is not bound to this exact checkpoint/source")
            selected_results.append(result)
            contracts.add(
                (
                    record.evaluator_config_sha256,
                    record.comparison_protocol_id,
                    record.comparison_protocol_sha256,
                    record.equivalence_matcher_id,
                    record.equivalence_matcher_sha256,
                    record.equivalence_threshold_id,
                    record.equivalence_threshold_sha256,
                    float(record.maximum_normalized_distance),
                )
            )
        if len(contracts) != 1:
            raise ValueError("tuning records do not share one evaluator/matcher/method contract")
        completions = tuple(self.trace_completions)
        if not all(isinstance(value, V5CompletedTuningTrace) for value in completions):
            raise TypeError("trace_completions must contain V5CompletedTuningTrace values")
        if len({value.trace_id for value in completions}) != len(completions):
            raise ValueError("trace completion IDs must be unique")
        by_trace = {value.trace_id: value for value in completions}
        if set(by_trace) != {value.trace_id for value in selected_results}:
            raise ValueError("trace completions do not exactly cover selected method traces")
        for result in selected_results:
            completion = by_trace[result.trace_id]
            if (
                completion.trace_artifact_sha256,
                completion.trace_ledger_sha256,
            ) != (result.trace_artifact_sha256, result.trace_ledger_sha256):
                raise ValueError("trace completion escaped its artifact/ledger binding")
            if (
                result.time_to_first_compatible_seconds is not None
                and result.time_to_first_compatible_seconds > completion.completion_elapsed_seconds
            ):
                raise ValueError("first-compatible time exceeds trace completion time")
        object.__setattr__(self, "paired_query_records", records)
        object.__setattr__(
            self, "trace_completions", tuple(sorted(completions, key=lambda value: value.trace_id))
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema": V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA,
            "summary_version": V5_TUNING_CHECKPOINT_SUMMARY_VERSION,
            "evaluator_schema": V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
            "evaluator_version": V5_PAPER_BUDGET_EVALUATOR_VERSION,
            "checkpoint_epoch": self.checkpoint_epoch,
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
            "checkpoint_weights_sha256": self.checkpoint_weights_sha256,
            "training_result_sha256": self.training_result_sha256,
            "source_summary_artifact_sha256": self.source_summary_artifact_sha256,
            "method_id": self.method_id,
            "base_method_protocol_id": self.base_method_protocol_id,
            "base_method_protocol_sha256": self.base_method_protocol_sha256,
            "inference_seed": self.inference_seed,
            "checkpoint_evaluation_method_binding": build_v5_checkpoint_evaluation_method_binding(
                checkpoint_epoch=self.checkpoint_epoch,
                checkpoint_artifact_sha256=self.checkpoint_artifact_sha256,
                checkpoint_weights_sha256=self.checkpoint_weights_sha256,
                training_result_sha256=self.training_result_sha256,
                source_summary_artifact_sha256=self.source_summary_artifact_sha256,
                method_id=self.method_id,
                base_method_protocol_id=self.base_method_protocol_id,
                base_method_protocol_sha256=self.base_method_protocol_sha256,
                inference_seed=self.inference_seed,
            ),
            "query_cohort": self.query_cohort.audit_payload(),
            "query_cohort_sha256": self.query_cohort.sha256,
            "paired_query_records": [value.audit_payload() for value in self.paired_query_records],
            "paired_query_record_sha256s": [value.sha256 for value in self.paired_query_records],
            "trace_completions": [asdict(value) for value in self.trace_completions],
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5CheckpointSelectionScore:
    checkpoint_epoch: int
    checkpoint_artifact_sha256: str
    checkpoint_weights_sha256: str
    tuning_summary_sha256: str
    source_summary_artifact_sha256: str
    primary_pairing_unit_macro_auc: float
    median_exact_calls_to_first_compatible: float
    median_wall_seconds_to_first_compatible: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checkpoint_epoch", _positive_integer(self.checkpoint_epoch, "checkpoint_epoch")
        )
        for name in (
            "checkpoint_artifact_sha256",
            "checkpoint_weights_sha256",
            "tuning_summary_sha256",
            "source_summary_artifact_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        auc = float(self.primary_pairing_unit_macro_auc)
        if not isfinite(auc) or not 0.0 <= auc <= 1.0:
            raise ValueError("primary_pairing_unit_macro_auc must be finite and in [0, 1]")
        object.__setattr__(self, "primary_pairing_unit_macro_auc", auc)
        for name in (
            "median_exact_calls_to_first_compatible",
            "median_wall_seconds_to_first_compatible",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))


def _selection_input_payload(
    rule_sha256: str,
    epochs: tuple[int, ...],
    scores: tuple[V5CheckpointSelectionScore, ...],
    inference_seed: int,
) -> dict[str, object]:
    return {
        "rule_sha256": rule_sha256,
        "expected_checkpoint_epochs": epochs,
        "inference_seed": _inference_seed(inference_seed),
        "candidate_tuning_summary_sha256s": [
            value.tuning_summary_sha256 for value in scores
        ],
        "source_summary_artifact_sha256s": [
            value.source_summary_artifact_sha256 for value in scores
        ],
        "checkpoint_artifact_sha256s": [
            value.checkpoint_artifact_sha256 for value in scores
        ],
        "checkpoint_weights_sha256s": [
            value.checkpoint_weights_sha256 for value in scores
        ],
    }


@dataclass(frozen=True, kw_only=True)
class V5PaperCheckpointSelectionReceipt:
    selection_id: str
    rule_sha256: str
    study_protocol_sha256: str
    query_cohort_id: str
    query_cohort_artifact_sha256: str
    query_cohort_sha256: str
    training_result_sha256: str
    inference_seed: int
    expected_checkpoint_epochs: tuple[int, ...]
    exact_call_no_hit_censor: int
    wall_seconds_no_hit_censor: float
    selection_input_sha256: str
    candidate_scores: tuple[V5CheckpointSelectionScore, ...]
    selected_checkpoint_epoch: int
    selected_checkpoint_artifact_sha256: str
    selected_checkpoint_weights_sha256: str
    selected_tuning_summary_sha256: str
    selection_complete: bool = True
    paper_model_eligibility_asserted: bool = False
    qualification_status: str = V5_SELECTION_ONLY_STATUS

    def __post_init__(self) -> None:
        object.__setattr__(self, "selection_id", _text(self.selection_id, "selection_id"))
        for name in (
            "rule_sha256",
            "study_protocol_sha256",
            "query_cohort_artifact_sha256",
            "query_cohort_sha256",
            "training_result_sha256",
            "selection_input_sha256",
            "selected_checkpoint_artifact_sha256",
            "selected_checkpoint_weights_sha256",
            "selected_tuning_summary_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self, "query_cohort_id", _text(self.query_cohort_id, "query_cohort_id")
        )
        object.__setattr__(self, "inference_seed", _inference_seed(self.inference_seed))
        rule = v5_paper_checkpoint_selection_rule()
        if self.rule_sha256 != rule["rule_sha256"]:
            raise ValueError("selection receipt escaped the frozen selection rule")
        if self.study_protocol_sha256 != rule["study_protocol_sha256"]:
            raise ValueError("selection receipt escaped the frozen study protocol")
        epochs = tuple(
            _positive_integer(value, f"expected_checkpoint_epochs[{index}]")
            for index, value in enumerate(self.expected_checkpoint_epochs)
        )
        if not epochs or epochs != tuple(range(1, epochs[-1] + 1)):
            raise ValueError("receipt requires the complete ordered full-epoch range")
        if self.exact_call_no_hit_censor != EXACT_FORWARD_BUDGETS[-1] + 1:
            raise ValueError("receipt exact-call censor is not frozen")
        object.__setattr__(
            self,
            "selected_checkpoint_epoch",
            _positive_integer(self.selected_checkpoint_epoch, "selected_checkpoint_epoch"),
        )
        wall_censor = _finite_nonnegative(
            self.wall_seconds_no_hit_censor, "wall_seconds_no_hit_censor"
        )
        scores = tuple(self.candidate_scores)
        if not scores or not all(isinstance(value, V5CheckpointSelectionScore) for value in scores):
            raise ValueError("candidate_scores must contain checkpoint selection scores")
        if tuple(value.checkpoint_epoch for value in scores) != epochs:
            raise ValueError("receipt scores do not cover the complete epoch inventory")
        for score in scores:
            for name in (
                "checkpoint_artifact_sha256",
                "checkpoint_weights_sha256",
                "tuning_summary_sha256",
                "source_summary_artifact_sha256",
            ):
                _digest(getattr(score, name), name)
            auc = float(score.primary_pairing_unit_macro_auc)
            if not isfinite(auc) or not 0.0 <= auc <= 1.0:
                raise ValueError("checkpoint macro AUC must be finite and in [0, 1]")
            exact = _finite_nonnegative(
                score.median_exact_calls_to_first_compatible,
                "median_exact_calls_to_first_compatible",
            )
            if exact > self.exact_call_no_hit_censor:
                raise ValueError("checkpoint exact-call TTFC exceeds the frozen censor")
            wall = _finite_nonnegative(
                score.median_wall_seconds_to_first_compatible,
                "median_wall_seconds_to_first_compatible",
            )
            if wall > wall_censor:
                raise ValueError("checkpoint wall TTFC exceeds the frozen censor")
        winner = min(
            scores,
            key=lambda value: (
                -value.primary_pairing_unit_macro_auc,
                value.median_exact_calls_to_first_compatible,
                value.median_wall_seconds_to_first_compatible,
                value.checkpoint_epoch,
                value.checkpoint_artifact_sha256,
            ),
        )
        if (
            self.selected_checkpoint_epoch,
            self.selected_checkpoint_artifact_sha256,
            self.selected_checkpoint_weights_sha256,
            self.selected_tuning_summary_sha256,
        ) != (
            winner.checkpoint_epoch,
            winner.checkpoint_artifact_sha256,
            winner.checkpoint_weights_sha256,
            winner.tuning_summary_sha256,
        ):
            raise ValueError("receipt selected checkpoint does not replay from its scores")
        selection_input = _selection_input_payload(
            self.rule_sha256, epochs, scores, self.inference_seed
        )
        expected_input_sha = sha256(
            canonical_json(selection_input).encode("utf-8")
        ).hexdigest()
        if self.selection_input_sha256 != expected_input_sha:
            raise ValueError("receipt selection input SHA does not replay")
        if (
            self.selection_complete is not True
            or self.paper_model_eligibility_asserted is not False
            or self.qualification_status != V5_SELECTION_ONLY_STATUS
        ):
            raise ValueError("receipt may assert checkpoint selection only")
        object.__setattr__(self, "expected_checkpoint_epochs", epochs)
        object.__setattr__(self, "candidate_scores", scores)
        object.__setattr__(self, "wall_seconds_no_hit_censor", wall_censor)

    def _core_payload(self) -> dict[str, object]:
        return {
            "schema": V5_PAPER_CHECKPOINT_SELECTOR_SCHEMA,
            "version": V5_PAPER_CHECKPOINT_SELECTOR_VERSION,
            "selection_id": self.selection_id,
            "selection_rule": v5_paper_checkpoint_selection_rule(),
            **asdict(self),
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self._core_payload()).encode("utf-8")).hexdigest()

    def audit_payload(self) -> dict[str, object]:
        return {**self._core_payload(), "receipt_sha256": self.sha256}

    def to_json(self) -> str:
        return canonical_json(self.audit_payload())


def _candidate_score(
    value: V5TuningCheckpointEvaluation,
    *,
    wall_censor: float,
) -> V5CheckpointSelectionScore:
    completions = {item.trace_id: item for item in value.trace_completions}
    unit_auc: dict[str, list[float]] = {}
    exact_times, wall_times = [], []
    for record in value.paired_query_records:
        result = next(item for item in record.method_results if item.method_id == value.method_id)
        auc = _validate_result(result, record)
        unit_auc.setdefault(record.pairing_unit_id, []).append(auc)
        exact_times.append(
            EXACT_FORWARD_BUDGETS[-1] + 1
            if result.first_compatible_exact_call is None
            else result.first_compatible_exact_call
        )
        wall_times.append(
            wall_censor
            if result.time_to_first_compatible_seconds is None
            else result.time_to_first_compatible_seconds
        )
        # Accessing the completion here keeps the score explicitly trace-bound.
        completions[result.trace_id]
    unit_means = tuple(
        fsum(unit_auc[unit_id]) / len(unit_auc[unit_id]) for unit_id in sorted(unit_auc)
    )
    return V5CheckpointSelectionScore(
        checkpoint_epoch=value.checkpoint_epoch,
        checkpoint_artifact_sha256=value.checkpoint_artifact_sha256,
        checkpoint_weights_sha256=value.checkpoint_weights_sha256,
        tuning_summary_sha256=value.sha256,
        source_summary_artifact_sha256=value.source_summary_artifact_sha256,
        primary_pairing_unit_macro_auc=float(fsum(unit_means) / len(unit_means)),
        median_exact_calls_to_first_compatible=float(median(exact_times)),
        median_wall_seconds_to_first_compatible=float(median(wall_times)),
    )


def select_v5_paper_checkpoint(
    candidates: Sequence[V5TuningCheckpointEvaluation],
    *,
    selection_id: str,
    expected_checkpoint_epochs: Sequence[int],
) -> V5PaperCheckpointSelectionReceipt:
    """Select one checkpoint from the complete frozen full-epoch inventory."""

    selected_id = _text(selection_id, "selection_id")
    values = tuple(candidates)
    if not values or not all(isinstance(value, V5TuningCheckpointEvaluation) for value in values):
        raise ValueError("candidates must contain tuning checkpoint evaluations")
    epochs = tuple(
        _positive_integer(value, f"expected_checkpoint_epochs[{index}]")
        for index, value in enumerate(expected_checkpoint_epochs)
    )
    if not epochs or epochs != tuple(range(1, epochs[-1] + 1)):
        raise ValueError("expected checkpoint epochs must be the complete ordered full-epoch range")
    if len({value.checkpoint_epoch for value in values}) != len(values):
        raise ValueError("checkpoint epochs must be unique")
    if {value.checkpoint_epoch for value in values} != set(epochs):
        raise ValueError("candidate inventory is missing or adds a full epoch")
    if len({value.checkpoint_artifact_sha256 for value in values}) != len(values):
        raise ValueError("checkpoint artifact hashes must be unique")
    contracts = {
        (
            value.training_result_sha256,
            value.source_summary_artifact_sha256,
            value.query_cohort.cohort_id,
            value.query_cohort.cohort_artifact_sha256,
            value.query_cohort.sha256,
            value.method_id,
            value.base_method_protocol_id,
            value.base_method_protocol_sha256,
            value.inference_seed,
            tuple(
                (
                    record.evaluator_config_sha256,
                    record.comparison_protocol_id,
                    record.comparison_protocol_sha256,
                    record.equivalence_matcher_id,
                    record.equivalence_matcher_sha256,
                    record.equivalence_threshold_id,
                    record.equivalence_threshold_sha256,
                    record.maximum_normalized_distance,
                )
                for record in value.paired_query_records
            ),
        )
        for value in values
    }
    if len(contracts) != 1:
        raise ValueError("checkpoint candidates do not share one tuning/evaluator/method contract")
    completion_times = tuple(
        item.completion_elapsed_seconds for value in values for item in value.trace_completions
    )
    wall_censor = max(completion_times) + 1.0
    if not isfinite(wall_censor) or wall_censor <= max(completion_times):
        raise ValueError("cannot construct a finite common wall-clock censor")
    scores = tuple(
        sorted(
            (_candidate_score(value, wall_censor=wall_censor) for value in values),
            key=lambda value: value.checkpoint_epoch,
        )
    )
    winner = min(
        scores,
        key=lambda value: (
            -value.primary_pairing_unit_macro_auc,
            value.median_exact_calls_to_first_compatible,
            value.median_wall_seconds_to_first_compatible,
            value.checkpoint_epoch,
            value.checkpoint_artifact_sha256,
        ),
    )
    rule = v5_paper_checkpoint_selection_rule()
    inference_seed = values[0].inference_seed
    selection_input = _selection_input_payload(
        str(rule["rule_sha256"]), epochs, scores, inference_seed
    )
    input_sha = sha256(canonical_json(selection_input).encode("utf-8")).hexdigest()
    cohort = values[0].query_cohort
    return V5PaperCheckpointSelectionReceipt(
        selection_id=selected_id,
        rule_sha256=str(rule["rule_sha256"]),
        study_protocol_sha256=str(rule["study_protocol_sha256"]),
        query_cohort_id=cohort.cohort_id,
        query_cohort_artifact_sha256=cohort.cohort_artifact_sha256,
        query_cohort_sha256=cohort.sha256,
        training_result_sha256=values[0].training_result_sha256,
        inference_seed=inference_seed,
        expected_checkpoint_epochs=epochs,
        exact_call_no_hit_censor=EXACT_FORWARD_BUDGETS[-1] + 1,
        wall_seconds_no_hit_censor=wall_censor,
        selection_input_sha256=input_sha,
        candidate_scores=scores,
        selected_checkpoint_epoch=winner.checkpoint_epoch,
        selected_checkpoint_artifact_sha256=winner.checkpoint_artifact_sha256,
        selected_checkpoint_weights_sha256=winner.checkpoint_weights_sha256,
        selected_tuning_summary_sha256=winner.tuning_summary_sha256,
    )


def write_v5_paper_checkpoint_selection_receipt(
    path: str | os.PathLike[str], receipt: V5PaperCheckpointSelectionReceipt
) -> Path:
    """Exclusively publish a durable canonical selection receipt."""

    if not isinstance(receipt, V5PaperCheckpointSelectionReceipt):
        raise TypeError("receipt must be a V5PaperCheckpointSelectionReceipt")
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"receipt parent directory does not exist: {target.parent}")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint selection receipt: {target}")
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(receipt.to_json())
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    directory_fd = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


__all__ = [
    "V5CheckpointSelectionScore",
    "V5CompletedTuningTrace",
    "V5PaperCheckpointSelectionReceipt",
    "V5TuningCheckpointEvaluation",
    "V5TuningQueryCohort",
    "V5TuningQueryCohortMember",
    "V5_CHECKPOINT_BOUND_METHOD_PROTOCOL_ID",
    "V5_CHECKPOINT_EVALUATION_METHOD_BINDING_SCHEMA",
    "V5_PAPER_CHECKPOINT_SELECTION_RULE_ID",
    "V5_PAPER_CHECKPOINT_SELECTOR_SCHEMA",
    "V5_PAPER_CHECKPOINT_SELECTOR_VERSION",
    "V5_SELECTION_ONLY_STATUS",
    "V5_TRACE_COMPLETED",
    "V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA",
    "V5_TUNING_CHECKPOINT_SUMMARY_VERSION",
    "V5_TUNING_QUERY_COHORT_SCHEMA",
    "V5_TUNING_QUERY_COHORT_VERSION",
    "V5_TUNING_SELECTION_SPLIT",
    "build_v5_checkpoint_evaluation_method_binding",
    "select_v5_paper_checkpoint",
    "v5_paper_checkpoint_selection_rule",
    "write_v5_paper_checkpoint_selection_receipt",
]
