"""Strict readers for V5 tuning checkpoint summaries and runtime completion."""

from __future__ import annotations

from hashlib import sha256
import json
from numbers import Integral, Real
import os
from pathlib import Path, PurePosixPath
import re
from typing import Mapping, Sequence

from .grouped_artifact_v5 import canonical_json
from .paper_budget_evaluator_v5 import (
    V5PairedPaperBudgetQueryRecord,
    V5PaperBudgetMethodResult,
    V5_EXACT_COMPATIBILITY_STATUSES,
    V5_PAPER_BUDGET_EVALUATOR_SCHEMA,
    V5_PAPER_BUDGET_EVALUATOR_VERSION,
)
from .paper_checkpoint_selector_v5 import (
    V5CompletedTuningTrace,
    V5TuningCheckpointEvaluation,
    V5TuningQueryCohort,
    V5TuningQueryCohortMember,
    V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA,
    V5_TUNING_CHECKPOINT_SUMMARY_VERSION,
    V5_TUNING_QUERY_COHORT_SCHEMA,
    V5_TUNING_QUERY_COHORT_VERSION,
)
from .paper_endpoint_metrics import EXACT_FORWARD_BUDGETS, OUTPUT_CAPS
from .paper_representative_payload_v5 import (
    V5_EMITTED_REPRESENTATIVE_ROLE,
    V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA,
    V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION,
)
from .tuning_checkpoint_runtime_v5 import (
    V5TuningCheckpointRuntimeResult,
    V5_TUNING_CHECKPOINT_RUNTIME_COMPLETE,
    V5_TUNING_CHECKPOINT_RUNTIME_SCHEMA,
    V5_TUNING_CHECKPOINT_RUNTIME_VERSION,
    V5_TUNING_EXACT_TRACE_ARTIFACT_SCHEMA,
    V5_TUNING_EXACT_TRACE_ARTIFACT_VERSION,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SUMMARY_FIELDS = frozenset(
    {
        "schema",
        "summary_version",
        "evaluator_schema",
        "evaluator_version",
        "checkpoint_epoch",
        "checkpoint_artifact_sha256",
        "checkpoint_weights_sha256",
        "training_result_sha256",
        "source_summary_artifact_sha256",
        "method_id",
        "base_method_protocol_id",
        "base_method_protocol_sha256",
        "inference_seed",
        "checkpoint_evaluation_method_binding",
        "query_cohort",
        "query_cohort_sha256",
        "paired_query_records",
        "paired_query_record_sha256s",
        "trace_completions",
        "summary_sha256",
    }
)
_COHORT_FIELDS = frozenset(
    {
        "schema",
        "version",
        "selection_split",
        "cohort_id",
        "cohort_artifact_sha256",
        "members",
    }
)
_RECORD_FIELDS = frozenset(V5PairedPaperBudgetQueryRecord.__dataclass_fields__) | {
    "schema",
    "version",
}
_COMPLETION_FIELDS = frozenset(
    {
        "schema",
        "version",
        "status",
        "training_result_sha256",
        "source_summary_artifact_sha256",
        "query_cohort_sha256",
        "method_id",
        "base_method_protocol_id",
        "base_method_protocol_sha256",
        "inference_seed",
        "exact_forward_budgets",
        "output_caps",
        "retained_full_epochs",
        "expected_tuning_query_count",
        "expected_exact_trace_count",
        "tuning_summary_schema",
        "tuning_summary_version",
        "checkpoint_summaries",
        "exact_trace_artifacts",
        "validation_loss_used",
        "test_calibration_reference_or_ood_used",
        "publication_rule",
        "completion_sha256",
    }
)
_TRACE_FIELDS = frozenset(
    {
        "schema",
        "version",
        "trace_id",
        "checkpoint",
        "source_summary_artifact_sha256",
        "query",
        "method_id",
        "method_binding",
        "inference_seed",
        "exact_forward_call_budget",
        "exact_forward_calls",
        "candidate_emissions",
        "trace_ledger_sha256",
        "completion_elapsed_seconds",
        "complete_contiguous_exact_call_trace",
        "validation_loss_used",
    }
)


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_path(value: str | os.PathLike[str], name: str, *, directory: bool) -> Path:
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"{name} must not contain parent traversal")
    lexical = path if path.is_absolute() else Path.cwd() / path
    probe = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        probe /= part
        if probe.is_symlink():
            raise ValueError(f"{name} must not contain symbolic-link components")
    if directory is not lexical.is_dir() or (not directory and not lexical.is_file()):
        raise ValueError(f"{name} has the wrong filesystem type")
    return lexical.resolve(strict=True)


def _load_json_file(
    value: str | os.PathLike[str],
    name: str,
    *,
    expected_file_sha256: str | None = None,
) -> tuple[Path, dict[str, object], str]:
    path = _checked_path(value, name, directory=False)
    raw = path.read_bytes()
    if len(raw) > 128 * 1024 * 1024:
        raise ValueError(f"{name} is unexpectedly large")
    file_sha = sha256(raw).hexdigest()
    if expected_file_sha256 is not None and file_sha != _digest(
        expected_file_sha256, f"{name} file SHA-256"
    ):
        raise ValueError(f"{name} file SHA-256 changed")
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain one JSON object")
    return path, payload, file_sha


def _nested_tuple(value: object, depth: int, name: str):
    if depth == 0:
        return value
    if not isinstance(value, list):
        raise ValueError(f"{name} has an invalid nested sequence")
    return tuple(_nested_tuple(item, depth - 1, name) for item in value)


def _method_result(payload: object) -> V5PaperBudgetMethodResult:
    if not isinstance(payload, Mapping) or set(payload) != set(
        V5PaperBudgetMethodResult.__dataclass_fields__
    ):
        raise ValueError("tuning method result fields are incomplete")
    values = dict(payload)
    for name, depth in (
        ("hit_count_matrix", 2),
        ("recall_matrix", 2),
        ("recall_ceiling_by_output_cap", 2),
        ("selected_candidate_ids_matrix", 3),
        ("matched_pairs_matrix", 4),
        ("matched_distance_sum_matrix", 2),
        ("auc_by_output_cap", 2),
        ("compatibility_status_counts", 2),
    ):
        values[name] = _nested_tuple(values[name], depth, name)
    return V5PaperBudgetMethodResult(**values)


def _paired_record(payload: object) -> V5PairedPaperBudgetQueryRecord:
    if not isinstance(payload, Mapping) or frozenset(payload) != _RECORD_FIELDS:
        raise ValueError("paired tuning record fields are incomplete")
    values = dict(payload)
    if values.pop("schema") != V5_PAPER_BUDGET_EVALUATOR_SCHEMA or values.pop(
        "version"
    ) != V5_PAPER_BUDGET_EVALUATOR_VERSION:
        raise ValueError("paired tuning record evaluator identity changed")
    methods = values.get("method_results")
    if not isinstance(methods, list):
        raise ValueError("paired tuning record method_results must be a list")
    values["method_results"] = tuple(_method_result(value) for value in methods)
    for name in ("reference_representative_ids", "output_caps", "exact_forward_budgets"):
        values[name] = _nested_tuple(values[name], 1, name)
    return V5PairedPaperBudgetQueryRecord(**values)


def _query_cohort(payload: object) -> V5TuningQueryCohort:
    if not isinstance(payload, Mapping) or frozenset(payload) != _COHORT_FIELDS:
        raise ValueError("tuning query-cohort fields are incomplete")
    if payload["schema"] != V5_TUNING_QUERY_COHORT_SCHEMA or payload[
        "version"
    ] != V5_TUNING_QUERY_COHORT_VERSION:
        raise ValueError("tuning query-cohort contract changed")
    members = payload["members"]
    if not isinstance(members, list):
        raise ValueError("tuning query-cohort members must be a list")
    return V5TuningQueryCohort(
        cohort_id=payload["cohort_id"],
        cohort_artifact_sha256=payload["cohort_artifact_sha256"],
        selection_split=payload["selection_split"],
        members=tuple(V5TuningQueryCohortMember(**dict(value)) for value in members),
    )


def read_v5_tuning_checkpoint_evaluation(
    path: str | os.PathLike[str],
    *,
    expected_file_sha256: str | None = None,
    expected_summary_sha256: str | None = None,
) -> V5TuningCheckpointEvaluation:
    """Read and fully reconstruct one selector-ready checkpoint evaluation."""

    _, payload, _ = _load_json_file(
        path,
        "tuning checkpoint summary",
        expected_file_sha256=expected_file_sha256,
    )
    if frozenset(payload) != _SUMMARY_FIELDS:
        raise ValueError("tuning checkpoint summary fields are incomplete")
    supplied_sha = _digest(payload["summary_sha256"], "summary_sha256")
    if expected_summary_sha256 is not None and supplied_sha != _digest(
        expected_summary_sha256, "expected_summary_sha256"
    ):
        raise ValueError("tuning checkpoint semantic SHA-256 changed")
    core = {key: value for key, value in payload.items() if key != "summary_sha256"}
    if (
        core["schema"] != V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA
        or core["summary_version"] != V5_TUNING_CHECKPOINT_SUMMARY_VERSION
        or core["evaluator_schema"] != V5_PAPER_BUDGET_EVALUATOR_SCHEMA
        or core["evaluator_version"] != V5_PAPER_BUDGET_EVALUATOR_VERSION
    ):
        raise ValueError("tuning checkpoint summary contract changed")
    cohort = _query_cohort(core["query_cohort"])
    if core["query_cohort_sha256"] != cohort.sha256:
        raise ValueError("tuning checkpoint cohort SHA-256 does not reproduce")
    records_payload = core["paired_query_records"]
    if not isinstance(records_payload, list):
        raise ValueError("paired_query_records must be a list")
    records = tuple(_paired_record(value) for value in records_payload)
    if core["paired_query_record_sha256s"] != [value.sha256 for value in records]:
        raise ValueError("paired tuning record SHA-256 inventory does not reproduce")
    completions_payload = core["trace_completions"]
    if not isinstance(completions_payload, list):
        raise ValueError("trace_completions must be a list")
    evaluation = V5TuningCheckpointEvaluation(
        checkpoint_epoch=core["checkpoint_epoch"],
        checkpoint_artifact_sha256=core["checkpoint_artifact_sha256"],
        checkpoint_weights_sha256=core["checkpoint_weights_sha256"],
        training_result_sha256=core["training_result_sha256"],
        source_summary_artifact_sha256=core["source_summary_artifact_sha256"],
        method_id=core["method_id"],
        base_method_protocol_id=core["base_method_protocol_id"],
        base_method_protocol_sha256=core["base_method_protocol_sha256"],
        inference_seed=core["inference_seed"],
        query_cohort=cohort,
        paired_query_records=records,
        trace_completions=tuple(
            V5CompletedTuningTrace(**dict(value)) for value in completions_payload
        ),
    )
    if (
        canonical_json(evaluation.audit_payload()) != canonical_json(core)
        or evaluation.sha256 != supplied_sha
    ):
        raise ValueError("tuning checkpoint summary does not exactly replay")
    if core["checkpoint_evaluation_method_binding"] != evaluation.audit_payload()[
        "checkpoint_evaluation_method_binding"
    ]:
        raise ValueError("tuning checkpoint method binding changed")
    return evaluation


def _safe_member(root: Path, relative: object, expected: str) -> Path:
    if not isinstance(relative, str):
        raise ValueError("runtime artifact relative path must be text")
    member = PurePosixPath(relative)
    if member.is_absolute() or "." in member.parts or ".." in member.parts:
        raise ValueError("runtime artifact relative path is unsafe")
    if member.as_posix() != expected:
        raise ValueError("runtime artifact is outside its canonical location")
    path = root / Path(*member.parts)
    selected = _checked_path(path, "runtime artifact", directory=False)
    if root not in selected.parents:
        raise ValueError("runtime artifact escaped its output root")
    return selected


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _elapsed(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and non-negative")
    result = float(value)
    if not 0.0 <= result < float("inf"):
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _verify_trace_payload(
    payload: Mapping[str, object],
    *,
    inventory: Mapping[str, object],
    evaluation: V5TuningCheckpointEvaluation,
) -> None:
    if frozenset(payload) != _TRACE_FIELDS:
        raise ValueError("exact trace artifact fields are incomplete")
    if (
        payload["schema"] != V5_TUNING_EXACT_TRACE_ARTIFACT_SCHEMA
        or payload["version"] != V5_TUNING_EXACT_TRACE_ARTIFACT_VERSION
        or payload["trace_id"] != inventory["trace_id"]
        or payload["trace_ledger_sha256"] != inventory["ledger_sha256"]
        or payload["method_id"] != evaluation.method_id
        or payload["method_binding"]
        != evaluation.audit_payload()["checkpoint_evaluation_method_binding"]
        or payload["inference_seed"] != evaluation.inference_seed
        or payload["source_summary_artifact_sha256"]
        != evaluation.source_summary_artifact_sha256
        or payload["exact_forward_call_budget"] != EXACT_FORWARD_BUDGETS[-1]
        or payload["complete_contiguous_exact_call_trace"] is not True
        or payload["validation_loss_used"] is not False
    ):
        raise ValueError("exact trace escaped its checkpoint/method/seed/source contract")
    checkpoint = payload["checkpoint"]
    query = payload["query"]
    if not isinstance(checkpoint, Mapping) or not isinstance(query, Mapping):
        raise ValueError("exact trace checkpoint/query binding is missing")
    member = next(
        value
        for value in evaluation.query_cohort.members
        if value.query_id == inventory["query_id"]
    )
    if checkpoint != {
        "full_epoch": evaluation.checkpoint_epoch,
        "artifact_sha256": evaluation.checkpoint_artifact_sha256,
        "weights_sha256": evaluation.checkpoint_weights_sha256,
        "training_result_sha256": evaluation.training_result_sha256,
    } or query != {
        "query_id": member.query_id,
        "query_context_sha256": member.query_context_sha256,
        "pairing_unit_id": member.pairing_unit_id,
        "reference_set_id": member.reference_set_id,
        "reference_set_sha256": member.reference_set_sha256,
        "reference_representative_payload_set_sha256": (
            member.reference_representative_payload_set_sha256
        ),
    }:
        raise ValueError("exact trace escaped its checkpoint or frozen query cohort")
    calls = payload["exact_forward_calls"]
    emissions = payload["candidate_emissions"]
    if not isinstance(calls, list) or len(calls) != EXACT_FORWARD_BUDGETS[-1]:
        raise ValueError("exact trace call ledger is incomplete")
    call_times = []
    for index, call in enumerate(calls, start=1):
        if not isinstance(call, Mapping) or call.get("exact_call_index") != index:
            raise ValueError("exact trace call indices are not contiguous")
        call_times.append(_elapsed(call.get("elapsed_seconds"), "exact-call time"))
    if any(left > right for left, right in zip(call_times, call_times[1:])):
        raise ValueError("exact trace call times are not monotonic")
    if not isinstance(emissions, list):
        raise ValueError("exact trace candidate emissions must be a list")
    ledger_emissions = []
    seen_ids, seen_ranks = set(), set()
    for emission in emissions:
        if not isinstance(emission, Mapping):
            raise ValueError("exact trace candidate emission must be an object")
        call = _positive_integer(emission.get("available_after_call"), "emission call")
        rank = _positive_integer(emission.get("output_rank"), "emission rank")
        candidate_id = emission.get("candidate_id")
        status = emission.get("compatibility_status")
        elapsed = _elapsed(emission.get("elapsed_seconds"), "emission time")
        representative = emission.get("representative_payload")
        if (
            call > len(calls)
            or not isinstance(candidate_id, str)
            or candidate_id in seen_ids
            or rank in seen_ranks
            or status not in V5_EXACT_COMPATIBILITY_STATUSES
            or elapsed < call_times[call - 1]
            or (call < len(calls) and elapsed > call_times[call])
            or not isinstance(representative, Mapping)
            or representative.get("schema") != V5_PAPER_REPRESENTATIVE_PAYLOAD_SCHEMA
            or representative.get("version") != V5_PAPER_REPRESENTATIVE_PAYLOAD_VERSION
            or representative.get("role") != V5_EMITTED_REPRESENTATIVE_ROLE
            or representative.get("representative_id") != candidate_id
            or representative.get("query_context_sha256") != member.query_context_sha256
        ):
            raise ValueError("exact trace candidate emission identity is invalid")
        payload_sha = sha256(canonical_json(dict(representative)).encode("utf-8")).hexdigest()
        if emission.get("representative_payload_sha256") != payload_sha:
            raise ValueError("exact trace representative payload SHA-256 does not reproduce")
        seen_ids.add(candidate_id)
        seen_ranks.add(rank)
        ledger_emissions.append(
            {
                "available_after_call": call,
                "output_rank": rank,
                "candidate_id": candidate_id,
                "compatibility_status": status,
                "elapsed_seconds": elapsed,
                "emitted_representative_payload_sha256": payload_sha,
            }
        )
    if seen_ranks != set(range(1, len(emissions) + 1)):
        raise ValueError("exact trace candidate ranks are not contiguous")
    ledger = {
        "exact_forward_calls": calls,
        "candidate_emissions": sorted(ledger_emissions, key=lambda value: value["output_rank"]),
    }
    if sha256(canonical_json(ledger).encode("utf-8")).hexdigest() != inventory[
        "ledger_sha256"
    ]:
        raise ValueError("exact trace ledger SHA-256 does not reproduce")
    completion = _elapsed(payload["completion_elapsed_seconds"], "trace completion time")
    if completion < call_times[-1]:
        raise ValueError("exact trace completion precedes its final call")


def read_v5_tuning_checkpoint_runtime_result(
    output_root: str | os.PathLike[str],
) -> V5TuningCheckpointRuntimeResult:
    """Accept only a complete runtime root whose trace and summary files replay."""

    root = _checked_path(output_root, "tuning runtime root", directory=True)
    completion_path, completion, completion_file_sha = _load_json_file(
        root / "completion.json", "tuning runtime completion"
    )
    if frozenset(completion) != _COMPLETION_FIELDS:
        raise ValueError("tuning runtime completion fields are incomplete")
    core = dict(completion)
    supplied_sha = _digest(core.pop("completion_sha256"), "completion_sha256")
    if supplied_sha != sha256(canonical_json(core).encode("utf-8")).hexdigest():
        raise ValueError("tuning runtime completion SHA-256 does not reproduce")
    if (
        core["schema"] != V5_TUNING_CHECKPOINT_RUNTIME_SCHEMA
        or core["version"] != V5_TUNING_CHECKPOINT_RUNTIME_VERSION
        or core["status"] != V5_TUNING_CHECKPOINT_RUNTIME_COMPLETE
        or core["exact_forward_budgets"] != list(EXACT_FORWARD_BUDGETS)
        or core["output_caps"] != list(OUTPUT_CAPS)
        or core["tuning_summary_schema"] != V5_TUNING_CHECKPOINT_SUMMARY_SCHEMA
        or core["tuning_summary_version"] != V5_TUNING_CHECKPOINT_SUMMARY_VERSION
        or core["validation_loss_used"] is not False
        or core["test_calibration_reference_or_ood_used"] is not False
        or core["publication_rule"]
        != "completion_written_exclusively_after_every_trace_and_summary_file"
    ):
        raise ValueError("tuning runtime completion contract changed")
    epochs = tuple(
        _positive_integer(value, "retained full epoch")
        for value in core["retained_full_epochs"]
    )
    if not epochs or epochs != tuple(range(1, epochs[-1] + 1)):
        raise ValueError("tuning runtime completion has an incomplete epoch inventory")
    summaries = core["checkpoint_summaries"]
    traces = core["exact_trace_artifacts"]
    if not isinstance(summaries, list) or not isinstance(traces, list):
        raise ValueError("tuning runtime artifact inventories must be lists")
    if len(summaries) != len(epochs) or len(traces) != core[
        "expected_exact_trace_count"
    ]:
        raise ValueError("tuning runtime artifact inventory counts are incomplete")
    evaluations, summary_paths, summary_file_shas = [], [], []
    for epoch, item in zip(epochs, summaries):
        if not isinstance(item, Mapping) or set(item) != {
            "full_epoch",
            "relative_path",
            "summary_sha256",
            "file_sha256",
        } or item["full_epoch"] != epoch:
            raise ValueError("tuning checkpoint summary inventory is invalid")
        path = _safe_member(
            root,
            item["relative_path"],
            f"checkpoint-summaries/full-epoch-{epoch:06d}.json",
        )
        evaluation = read_v5_tuning_checkpoint_evaluation(
            path,
            expected_file_sha256=item["file_sha256"],
            expected_summary_sha256=item["summary_sha256"],
        )
        evaluations.append(evaluation)
        summary_paths.append(path)
        summary_file_shas.append(item["file_sha256"])
    if any(
        (
            value.checkpoint_epoch != epoch
            or value.training_result_sha256 != core["training_result_sha256"]
            or value.source_summary_artifact_sha256
            != core["source_summary_artifact_sha256"]
            or value.query_cohort.sha256 != core["query_cohort_sha256"]
            or value.method_id != core["method_id"]
            or value.base_method_protocol_id != core["base_method_protocol_id"]
            or value.base_method_protocol_sha256
            != core["base_method_protocol_sha256"]
            or value.inference_seed != core["inference_seed"]
        )
        for epoch, value in zip(epochs, evaluations)
    ):
        raise ValueError("tuning summaries escaped their common completion contract")
    expected_trace_bindings = {
        result.trace_id: (
            evaluation.checkpoint_epoch,
            record.query_id,
            result.trace_artifact_sha256,
            result.trace_ledger_sha256,
        )
        for evaluation in evaluations
        for record in evaluation.paired_query_records
        for result in record.method_results
        if result.method_id == evaluation.method_id
    }
    trace_paths, trace_shas, seen = [], [], set()
    by_epoch = {value.checkpoint_epoch: value for value in evaluations}
    for item in traces:
        if not isinstance(item, Mapping) or set(item) != {
            "full_epoch",
            "query_id",
            "trace_id",
            "relative_path",
            "artifact_sha256",
            "ledger_sha256",
        }:
            raise ValueError("exact trace inventory entry is invalid")
        trace_id = item["trace_id"]
        binding = expected_trace_bindings.get(trace_id)
        if binding != (
            item["full_epoch"],
            item["query_id"],
            item["artifact_sha256"],
            item["ledger_sha256"],
        ) or trace_id in seen:
            raise ValueError("exact trace inventory escaped its tuning summary")
        seen.add(trace_id)
        relative = (
            f"exact-traces/epoch-{item['full_epoch']:06d}-"
            f"{sha256(item['query_id'].encode('utf-8')).hexdigest()}.json"
        )
        path = _safe_member(root, item["relative_path"], relative)
        _, payload, file_sha = _load_json_file(
            path,
            "exact trace artifact",
            expected_file_sha256=item["artifact_sha256"],
        )
        _verify_trace_payload(
            payload,
            inventory=item,
            evaluation=by_epoch[item["full_epoch"]],
        )
        trace_paths.append(path)
        trace_shas.append(file_sha)
    if set(expected_trace_bindings) != seen or len(traces) != len(epochs) * core[
        "expected_tuning_query_count"
    ]:
        raise ValueError("exact trace inventory does not exactly cover every epoch/query")
    return V5TuningCheckpointRuntimeResult(
        output_root=root,
        evaluations=tuple(evaluations),
        trace_paths=tuple(trace_paths),
        trace_artifact_sha256s=tuple(trace_shas),
        summary_paths=tuple(summary_paths),
        summary_file_sha256s=tuple(summary_file_shas),
        completion_path=completion_path,
        completion_sha256=supplied_sha,
        completion_file_sha256=completion_file_sha,
    )


__all__ = [
    "read_v5_tuning_checkpoint_evaluation",
    "read_v5_tuning_checkpoint_runtime_result",
]
