"""Exact-forward tuning evaluation for every retained full-epoch checkpoint.

The runtime owns the common cohort/method/seed/budget loop and durable trace
receipts.  A model-specific adapter supplies only the exact-call runner.  No
training or validation loss enters this API, so a selector-ready summary can
exist only after a complete contiguous exact-forward trace has been returned
for every query and every retained full epoch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from numbers import Integral, Real
import os
from pathlib import Path
import re
from typing import Callable, Mapping, Sequence

from .grouped_artifact_v5 import canonical_json
from .k1_staging_files_v5 import read_only_bytes_identity
from .paper_budget_evaluator_v5 import (
    EquivalenceDistanceMatcher,
    V5CandidateEmission,
    V5ExactForwardCall,
    V5FrozenReferenceSet,
    V5MethodExactCallTrace,
    V5PaperBudgetEvaluationConfig,
    evaluate_v5_paired_paper_budget_query,
    _paired_record_from_results,
)
from .paper_representative_history_v5 import (
    V5RepresentativeHistory, V5RepresentativeSnapshot, evaluate_v5_representative_history,
)
from .paper_checkpoint_selector_v5 import (
    V5CompletedTuningTrace,
    V5TuningCheckpointEvaluation,
    V5TuningQueryCohort,
    V5_CHECKPOINT_BOUND_METHOD_PROTOCOL_ID,
    V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA,
    V5_TUNING_CHECKPOINT_SUMMARY_VERSION,
    build_v5_checkpoint_evaluation_method_binding,
)
from .paper_endpoint_metrics import EXACT_FORWARD_BUDGETS, OUTPUT_CAPS
from .tuning_lossless_emission_store_v5 import publish_v5_tuning_lossless_emissions
from .tuning_trace_artifact_v5 import (
    V5_TUNING_EXACT_TRACE_ARTIFACT_SCHEMA,
    V5_TUNING_EXACT_TRACE_ARTIFACT_VERSION,
    build_v5_tuning_exact_trace_artifact,
    write_v5_tuning_json_exclusive,
)


V5_TUNING_CHECKPOINT_RUNTIME_SCHEMA = (
    "gisaxs.posterior_v8.tuning_checkpoint_exact_budget_runtime/v2"
)
V5_TUNING_CHECKPOINT_RUNTIME_VERSION = (
    "all_retained_epochs_complete_trace_explicit_representative_history_v3"
)
V5_TUNING_CHECKPOINT_RUNTIME_COMPLETE = (
    "all_retained_full_epochs_exact_budget_evaluated"
)
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
    if not 0.0 <= result < float("inf"):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_regular_file(path: Path, expected_sha256: str, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file")
    if _file_sha256(path) != _digest(expected_sha256, f"{name} SHA-256"):
        raise RuntimeError(f"{name} changed or disagrees with its frozen SHA-256")
    return path.resolve(strict=True)


def _checked_sealed_file(path: Path, expected_sha256: str, name: str) -> None:
    _, identity = read_only_bytes_identity(path, name)
    if identity["mode_octal"] != "0400" or identity["sha256"] != expected_sha256:
        raise RuntimeError(f"{name} is not the expected sealed artifact")


@dataclass(frozen=True, eq=False, kw_only=True)
class V5RetainedFullCheckpoint:
    full_epoch: int
    checkpoint_path: Path
    checkpoint_artifact_sha256: str
    checkpoint_weights_sha256: str
    training_result_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "full_epoch", _positive_integer(self.full_epoch, "full_epoch")
        )
        for name in (
            "checkpoint_artifact_sha256",
            "checkpoint_weights_sha256",
            "training_result_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(
            self,
            "checkpoint_path",
            _checked_regular_file(
                Path(self.checkpoint_path),
                self.checkpoint_artifact_sha256,
                "checkpoint artifact",
            ),
        )


@dataclass(frozen=True, eq=False, kw_only=True)
class V5TuningExactTraceRunResult:
    """Identity echoes plus the exact calls/emissions returned by an adapter."""

    trace_id: str
    checkpoint_artifact_sha256_used: str
    query_id: str
    method_protocol_sha256_used: str
    inference_seed_used: int
    exact_forward_call_budget_used: int
    exact_forward_calls: tuple[V5ExactForwardCall, ...]
    candidate_emissions: tuple[V5CandidateEmission, ...]
    completion_elapsed_seconds: float
    representative_snapshots: tuple[V5RepresentativeSnapshot, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace_id", _text(self.trace_id, "trace_id"))
        for name in (
            "checkpoint_artifact_sha256_used",
            "method_protocol_sha256_used",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        object.__setattr__(self, "query_id", _text(self.query_id, "query_id"))
        object.__setattr__(
            self, "inference_seed_used", _inference_seed(self.inference_seed_used)
        )
        object.__setattr__(
            self,
            "exact_forward_call_budget_used",
            _positive_integer(
                self.exact_forward_call_budget_used,
                "exact_forward_call_budget_used",
            ),
        )
        calls = tuple(self.exact_forward_calls)
        emissions = tuple(self.candidate_emissions)
        if not all(isinstance(value, V5ExactForwardCall) for value in calls):
            raise TypeError("exact_forward_calls contain an invalid value")
        if not all(isinstance(value, V5CandidateEmission) for value in emissions):
            raise TypeError("candidate_emissions contain an invalid value")
        object.__setattr__(self, "exact_forward_calls", calls)
        object.__setattr__(self, "candidate_emissions", emissions)
        if self.representative_snapshots is not None:
            snapshots = tuple(self.representative_snapshots)
            if not snapshots or any(type(row) is not V5RepresentativeSnapshot for row in snapshots):
                raise ValueError("snapshot runner requires explicit typed representative snapshots")
            object.__setattr__(self, "representative_snapshots", snapshots)
        object.__setattr__(
            self,
            "completion_elapsed_seconds",
            _finite_nonnegative(
                self.completion_elapsed_seconds, "completion_elapsed_seconds"
            ),
        )


TuningExactTraceRunner = Callable[
    [
        V5RetainedFullCheckpoint,
        V5FrozenReferenceSet,
        Mapping[str, object],
        int,
        int,
    ],
    V5TuningExactTraceRunResult,
]


@dataclass(frozen=True, eq=False, kw_only=True)
class V5TuningCheckpointRuntimeResult:
    output_root: Path
    evaluations: tuple[V5TuningCheckpointEvaluation, ...]
    trace_paths: tuple[Path, ...]
    trace_artifact_sha256s: tuple[str, ...]
    lossless_emission_paths: tuple[Path, ...]
    lossless_emission_file_sha256s: tuple[str, ...]
    summary_paths: tuple[Path, ...]
    summary_file_sha256s: tuple[str, ...]
    completion_path: Path
    completion_sha256: str
    completion_file_sha256: str


def _validate_cohort_references(
    cohort: V5TuningQueryCohort,
    references: Sequence[V5FrozenReferenceSet],
) -> tuple[V5FrozenReferenceSet, ...]:
    if not isinstance(cohort, V5TuningQueryCohort):
        raise TypeError("query_cohort must be a V5TuningQueryCohort")
    values = tuple(references)
    if not values or not all(isinstance(value, V5FrozenReferenceSet) for value in values):
        raise ValueError("reference_sets must contain frozen reference sets")
    if len({value.query_id for value in values}) != len(values):
        raise ValueError("reference-set query IDs must be unique")
    by_query = {value.query_id: value for value in values}
    if set(by_query) != {value.query_id for value in cohort.members}:
        raise ValueError("reference sets do not exactly cover the frozen tuning cohort")
    ordered = tuple(by_query[value.query_id] for value in cohort.members)
    for member, reference in zip(cohort.members, ordered):
        if (
            reference.pairing_unit_id,
            reference.reference_set_id,
            reference.reference_set_sha256,
            reference.query_context_sha256,
            reference.representative_payload_set_sha256,
        ) != (
            member.pairing_unit_id,
            member.reference_set_id,
            member.reference_set_sha256,
            member.query_context_sha256,
            member.reference_representative_payload_set_sha256,
        ):
            raise ValueError("reference set escaped the frozen cohort/context/payload identity")
    return ordered


def evaluate_v5_retained_checkpoints_on_tuning(
    *,
    checkpoints: Sequence[V5RetainedFullCheckpoint],
    query_cohort: V5TuningQueryCohort,
    reference_sets: Sequence[V5FrozenReferenceSet],
    config: V5PaperBudgetEvaluationConfig,
    equivalence_distance_matcher: EquivalenceDistanceMatcher,
    trace_runner: TuningExactTraceRunner,
    method_id: str,
    base_method_protocol_id: str,
    base_method_protocol_sha256: str,
    inference_seed: int,
    source_summary_artifact_sha256: str,
    output_root: str | os.PathLike[str],
    pre_publish_guard: Callable[[], str],
    representative_selection_policy: str = "append_only",
) -> V5TuningCheckpointRuntimeResult:
    """Evaluate the complete retained full-epoch inventory on one tuning cohort."""

    if representative_selection_policy not in ("append_only", "budget_snapshot"):
        raise ValueError("unknown representative selection policy")
    values = tuple(checkpoints)
    if not values or not all(isinstance(value, V5RetainedFullCheckpoint) for value in values):
        raise ValueError("checkpoints must contain retained full checkpoints")
    values = tuple(sorted(values, key=lambda value: value.full_epoch))
    epochs = tuple(value.full_epoch for value in values)
    if epochs != tuple(range(1, epochs[-1] + 1)):
        raise ValueError("retained checkpoints must cover every full epoch from one")
    if len({value.checkpoint_artifact_sha256 for value in values}) != len(values):
        raise ValueError("retained checkpoint artifact hashes must be unique")
    if len({value.training_result_sha256 for value in values}) != 1:
        raise ValueError("retained checkpoints must share one training result")
    if not isinstance(config, V5PaperBudgetEvaluationConfig):
        raise TypeError("config must be a V5PaperBudgetEvaluationConfig")
    if config.exact_forward_budgets != EXACT_FORWARD_BUDGETS or config.output_caps != OUTPUT_CAPS:
        raise ValueError("tuning runtime requires the frozen paper budget/output-cap grid")
    if not callable(equivalence_distance_matcher) or not callable(trace_runner):
        raise TypeError("matcher and trace_runner must be callable")
    if not callable(pre_publish_guard):
        raise TypeError("pre_publish_guard must be callable")
    method = _text(method_id, "method_id")
    base_protocol = _text(base_method_protocol_id, "base_method_protocol_id")
    base_protocol_sha = _digest(
        base_method_protocol_sha256, "base_method_protocol_sha256"
    )
    seed = _inference_seed(inference_seed)
    source_sha = _digest(
        source_summary_artifact_sha256, "source_summary_artifact_sha256"
    )
    references = _validate_cohort_references(query_cohort, reference_sets)
    if pre_publish_guard() != source_sha:
        raise RuntimeError("source summary changed before tuning evaluation")
    root = Path(output_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError("refusing to reuse a tuning evaluation output root")
    if not root.parent.is_dir():
        raise FileNotFoundError("tuning evaluation output parent does not exist")
    root.mkdir(mode=0o700)
    trace_root = root / "exact-traces"
    summary_root = root / "checkpoint-summaries"
    trace_root.mkdir(mode=0o700)
    summary_root.mkdir(mode=0o700)

    evaluations = []
    trace_paths = []
    trace_artifact_sha256s = []
    trace_inventory = []
    lossless_emission_paths = []
    lossless_emission_file_sha256s = []
    lossless_emission_inventory = []
    summary_paths = []
    summary_file_sha256s = []
    summary_inventory = []
    trace_ids: set[str] = set()
    for checkpoint in values:
        _checked_regular_file(
            checkpoint.checkpoint_path,
            checkpoint.checkpoint_artifact_sha256,
            "checkpoint artifact",
        )
        binding = build_v5_checkpoint_evaluation_method_binding(
            representative_selection_policy=representative_selection_policy,
            checkpoint_epoch=checkpoint.full_epoch,
            checkpoint_artifact_sha256=checkpoint.checkpoint_artifact_sha256,
            checkpoint_weights_sha256=checkpoint.checkpoint_weights_sha256,
            training_result_sha256=checkpoint.training_result_sha256,
            source_summary_artifact_sha256=source_sha,
            method_id=method,
            base_method_protocol_id=base_protocol,
            base_method_protocol_sha256=base_protocol_sha,
            inference_seed=seed,
        )
        records = []
        completions = []
        for reference in references:
            before = _file_sha256(checkpoint.checkpoint_path)
            result = trace_runner(
                checkpoint,
                reference,
                binding,
                seed,
                EXACT_FORWARD_BUDGETS[-1],
            )
            if not isinstance(result, V5TuningExactTraceRunResult):
                raise TypeError("trace_runner must return V5TuningExactTraceRunResult")
            if (result.representative_snapshots is not None) != (representative_selection_policy == "budget_snapshot"):
                raise ValueError("runner escaped frozen representative selection policy")
            after = _file_sha256(checkpoint.checkpoint_path)
            if before != checkpoint.checkpoint_artifact_sha256 or after != before:
                raise RuntimeError("checkpoint changed while its exact trace was running")
            if (
                result.checkpoint_artifact_sha256_used
                != checkpoint.checkpoint_artifact_sha256
                or result.query_id != reference.query_id
                or result.method_protocol_sha256_used
                != binding["bound_method_protocol_sha256"]
                or result.inference_seed_used != seed
                or result.exact_forward_call_budget_used != EXACT_FORWARD_BUDGETS[-1]
                or len(result.exact_forward_calls) != EXACT_FORWARD_BUDGETS[-1]
            ):
                raise ValueError("exact trace escaped checkpoint/query/method/seed/budget binding")
            if result.trace_id in trace_ids:
                raise ValueError("exact trace IDs must be globally unique")
            trace_ids.add(result.trace_id)
            provisional = V5MethodExactCallTrace(
                query_id=reference.query_id,
                pairing_unit_id=reference.pairing_unit_id,
                method_id=method,
                method_protocol_id=V5_CHECKPOINT_BOUND_METHOD_PROTOCOL_ID,
                method_protocol_sha256=binding["bound_method_protocol_sha256"],
                trace_id=result.trace_id,
                trace_artifact_sha256="0" * 64,
                reference_set_id=reference.reference_set_id,
                reference_set_sha256=reference.reference_set_sha256,
                comparison_protocol_id=config.comparison_protocol_id,
                comparison_protocol_sha256=config.comparison_protocol_sha256,
                exact_forward_call_budget=EXACT_FORWARD_BUDGETS[-1],
                exact_forward_calls=result.exact_forward_calls,
                candidate_emissions=result.candidate_emissions,
            )
            last_elapsed = provisional.exact_forward_calls[-1].elapsed_seconds
            if result.representative_snapshots is not None:
                history = V5RepresentativeHistory(trace=provisional, snapshots=result.representative_snapshots)
                if history.snapshots[-1].elapsed_seconds > result.completion_elapsed_seconds:
                    raise ValueError("trace completion precedes its final representative snapshot")
            if result.completion_elapsed_seconds < last_elapsed:
                raise ValueError("trace completion time precedes its final exact call")
            lossless = publish_v5_tuning_lossless_emissions(
                output_root=root,
                full_epoch=checkpoint.full_epoch,
                query_id=reference.query_id,
                emissions=result.candidate_emissions,
            )
            lossless_emission_paths.extend(
                root / value.relative_path for value in lossless
            )
            lossless_emission_file_sha256s.extend(
                value.file_sha256 for value in lossless
            )
            lossless_emission_inventory.extend(
                {
                    "full_epoch": checkpoint.full_epoch,
                    "query_id": reference.query_id,
                    **value.audit_payload(),
                }
                for value in lossless
            )
            trace_core = build_v5_tuning_exact_trace_artifact(
                trace_id=result.trace_id,
                checkpoint_binding={
                    "full_epoch": checkpoint.full_epoch,
                    "artifact_sha256": checkpoint.checkpoint_artifact_sha256,
                    "weights_sha256": checkpoint.checkpoint_weights_sha256,
                    "training_result_sha256": checkpoint.training_result_sha256,
                },
                query_binding={
                    "query_id": reference.query_id,
                    "query_context_sha256": reference.query_context_sha256,
                    "pairing_unit_id": reference.pairing_unit_id,
                    "reference_set_id": reference.reference_set_id,
                    "reference_set_sha256": reference.reference_set_sha256,
                    "reference_representative_payload_set_sha256": (
                        reference.representative_payload_set_sha256
                    ),
                },
                method_id=method,
                method_binding=binding,
                inference_seed=seed,
                source_summary_artifact_sha256=source_sha,
                exact_forward_calls=result.exact_forward_calls,
                candidate_emissions=result.candidate_emissions,
                ledger_sha256=provisional.ledger_sha256,
                lossless_emissions=lossless,
                completion_elapsed_seconds=result.completion_elapsed_seconds,
                representative_snapshots=result.representative_snapshots,
            )
            trace_path = trace_root / (
                f"epoch-{checkpoint.full_epoch:06d}-"
                f"{sha256(reference.query_id.encode('utf-8')).hexdigest()}.json"
            )
            trace_sha = write_v5_tuning_json_exclusive(trace_path, trace_core)
            trace_paths.append(trace_path)
            trace_artifact_sha256s.append(trace_sha)
            trace_inventory.append(
                {
                    "full_epoch": checkpoint.full_epoch,
                    "query_id": reference.query_id,
                    "trace_id": result.trace_id,
                    "relative_path": trace_path.relative_to(root).as_posix(),
                    "artifact_sha256": trace_sha,
                    "ledger_sha256": provisional.ledger_sha256,
                }
            )
            trace = replace(
                provisional,
                trace_artifact_sha256=trace_sha,
            )
            if result.representative_snapshots is None:
                record = evaluate_v5_paired_paper_budget_query(
                    reference, (trace,), config=config,
                    equivalence_distance_matcher=equivalence_distance_matcher,
                )
            else:
                history = V5RepresentativeHistory(trace=trace, snapshots=result.representative_snapshots)
                snapshot_result = evaluate_v5_representative_history(
                    reference, history, config=config,
                    equivalence_distance_matcher=equivalence_distance_matcher,
                )
                record = _paired_record_from_results(reference, config, (snapshot_result.method_result,))
            records.append(record)
            completions.append(
                V5CompletedTuningTrace(
                    trace_id=trace.trace_id,
                    trace_artifact_sha256=trace.trace_artifact_sha256,
                    trace_ledger_sha256=trace.ledger_sha256,
                    completion_elapsed_seconds=result.completion_elapsed_seconds,
                )
            )
        evaluation = V5TuningCheckpointEvaluation(
            representative_selection_policy=representative_selection_policy,
            checkpoint_epoch=checkpoint.full_epoch,
            checkpoint_artifact_sha256=checkpoint.checkpoint_artifact_sha256,
            checkpoint_weights_sha256=checkpoint.checkpoint_weights_sha256,
            training_result_sha256=checkpoint.training_result_sha256,
            source_summary_artifact_sha256=source_sha,
            method_id=method,
            base_method_protocol_id=base_protocol,
            base_method_protocol_sha256=base_protocol_sha,
            inference_seed=seed,
            query_cohort=query_cohort,
            paired_query_records=tuple(records),
            trace_completions=tuple(completions),
        )
        if pre_publish_guard() != source_sha:
            raise RuntimeError("source summary changed before checkpoint summary publication")
        _checked_regular_file(
            checkpoint.checkpoint_path,
            checkpoint.checkpoint_artifact_sha256,
            "checkpoint artifact",
        )
        summary_path = summary_root / f"full-epoch-{checkpoint.full_epoch:06d}.json"
        summary_file_sha = write_v5_tuning_json_exclusive(
            summary_path,
            {**evaluation.audit_payload(), "summary_sha256": evaluation.sha256},
        )
        evaluations.append(evaluation)
        summary_paths.append(summary_path)
        summary_file_sha256s.append(summary_file_sha)
        summary_inventory.append(
            {
                "full_epoch": checkpoint.full_epoch,
                "relative_path": summary_path.relative_to(root).as_posix(),
                "summary_sha256": evaluation.sha256,
                "file_sha256": summary_file_sha,
            }
        )

    completion_core = {
        "schema": V5_TUNING_CHECKPOINT_RUNTIME_SCHEMA,
        "version": V5_TUNING_CHECKPOINT_RUNTIME_VERSION,
        "status": V5_TUNING_CHECKPOINT_RUNTIME_COMPLETE,
        "training_result_sha256": values[0].training_result_sha256,
        "source_summary_artifact_sha256": source_sha,
        "query_cohort_sha256": query_cohort.sha256,
        "method_id": method,
        "base_method_protocol_id": base_protocol,
        "base_method_protocol_sha256": base_protocol_sha,
        "inference_seed": seed,
        "exact_forward_budgets": list(EXACT_FORWARD_BUDGETS),
        "output_caps": list(OUTPUT_CAPS),
        "retained_full_epochs": list(epochs),
        "representative_selection_policy": representative_selection_policy,
        "expected_tuning_query_count": len(references),
        "expected_exact_trace_count": len(values) * len(references),
        "expected_lossless_emission_count": len(lossless_emission_inventory),
        "tuning_summary_schema": V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA,
        "tuning_summary_version": V5_TUNING_CHECKPOINT_SUMMARY_VERSION,
        "checkpoint_summaries": summary_inventory,
        "exact_trace_artifacts": trace_inventory,
        "lossless_emission_artifacts": lossless_emission_inventory,
        "validation_loss_used": False,
        "test_calibration_reference_or_ood_used": False,
        "publication_rule": (
            "completion_written_exclusively_after_every_trace_summary_and_lossless_emission"
        ),
    }
    completion_sha = sha256(
        canonical_json(completion_core).encode("utf-8")
    ).hexdigest()
    if pre_publish_guard() != source_sha:
        raise RuntimeError("source summary changed before runtime completion publication")
    for checkpoint in values:
        _checked_regular_file(
            checkpoint.checkpoint_path,
            checkpoint.checkpoint_artifact_sha256,
            "checkpoint artifact",
        )
    for path, expected_sha in zip(trace_paths, trace_artifact_sha256s):
        _checked_sealed_file(path, expected_sha, "exact trace artifact")
    for path, expected_sha in zip(
        lossless_emission_paths, lossless_emission_file_sha256s
    ):
        _checked_sealed_file(path, expected_sha, "lossless emission artifact")
    for path, expected_sha in zip(summary_paths, summary_file_sha256s):
        _checked_sealed_file(path, expected_sha, "checkpoint summary")
    completion_path = root / "completion.json"
    completion_file_sha = write_v5_tuning_json_exclusive(
        completion_path,
        {**completion_core, "completion_sha256": completion_sha},
    )
    return V5TuningCheckpointRuntimeResult(
        output_root=root.resolve(strict=True),
        evaluations=tuple(evaluations),
        trace_paths=tuple(trace_paths),
        trace_artifact_sha256s=tuple(trace_artifact_sha256s),
        lossless_emission_paths=tuple(lossless_emission_paths),
        lossless_emission_file_sha256s=tuple(
            lossless_emission_file_sha256s
        ),
        summary_paths=tuple(summary_paths),
        summary_file_sha256s=tuple(summary_file_sha256s),
        completion_path=completion_path,
        completion_sha256=completion_sha,
        completion_file_sha256=completion_file_sha,
    )


__all__ = [
    "TuningExactTraceRunner",
    "V5RetainedFullCheckpoint",
    "V5TuningCheckpointRuntimeResult",
    "V5TuningExactTraceRunResult",
    "V5_TUNING_CHECKPOINT_RUNTIME_COMPLETE",
    "V5_TUNING_CHECKPOINT_RUNTIME_SCHEMA",
    "V5_TUNING_CHECKPOINT_RUNTIME_VERSION",
    "V5_TUNING_EXACT_TRACE_ARTIFACT_SCHEMA",
    "V5_TUNING_EXACT_TRACE_ARTIFACT_VERSION",
    "evaluate_v5_retained_checkpoints_on_tuning",
]
