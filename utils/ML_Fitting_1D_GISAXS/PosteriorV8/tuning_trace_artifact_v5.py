"""Canonical audit trace that binds separate lossless tuning emissions."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from .paper_budget_evaluator_v5 import V5CandidateEmission, V5ExactForwardCall
from .paper_endpoint_metrics import EXACT_FORWARD_BUDGETS
from .read_only_json_publication_v5 import publish_read_only_canonical_json
from .k1_staging_files_v5 import lexical_no_symlinks
from .tuning_lossless_emission_store_v5 import V5TuningLosslessEmissionArtifact


V5_TUNING_EXACT_TRACE_ARTIFACT_SCHEMA = "gisaxs.posterior_v8.tuning_checkpoint_exact_call_trace/v2"
V5_TUNING_EXACT_TRACE_ARTIFACT_VERSION = (
    "checkpoint_query_method_seed_complete_ledger_explicit_snapshots_v3"
)


def write_v5_tuning_json_exclusive(path: Path, payload: Mapping[str, object]) -> str:
    target = lexical_no_symlinks(path, "tuning publication")
    return publish_read_only_canonical_json(target, payload)


def build_v5_tuning_exact_trace_artifact(
    *,
    trace_id: str,
    checkpoint_binding: Mapping[str, object],
    source_summary_artifact_sha256: str,
    query_binding: Mapping[str, object],
    method_id: str,
    method_binding: Mapping[str, object],
    inference_seed: int,
    exact_forward_calls: tuple[V5ExactForwardCall, ...],
    candidate_emissions: tuple[V5CandidateEmission, ...],
    lossless_emissions: tuple[V5TuningLosslessEmissionArtifact, ...],
    ledger_sha256: str,
    completion_elapsed_seconds: float,
    representative_snapshots: tuple | None = None,
) -> dict[str, object]:
    lossless_by_rank = {value.output_rank: value for value in lossless_emissions}
    if set(lossless_by_rank) != {value.output_rank for value in candidate_emissions}:
        raise ValueError("lossless emission files do not cover the exact emitted ranks")
    return {
        "schema": V5_TUNING_EXACT_TRACE_ARTIFACT_SCHEMA,
        "version": V5_TUNING_EXACT_TRACE_ARTIFACT_VERSION,
        "trace_id": trace_id,
        "checkpoint": dict(checkpoint_binding),
        "source_summary_artifact_sha256": source_summary_artifact_sha256,
        "query": dict(query_binding),
        "method_id": method_id,
        "method_binding": dict(method_binding),
        "inference_seed": inference_seed,
        "exact_forward_call_budget": EXACT_FORWARD_BUDGETS[-1],
        "exact_forward_calls": [asdict(value) for value in exact_forward_calls],
        "candidate_emissions": [
            {
                "available_after_call": value.available_after_call,
                "output_rank": value.output_rank,
                "candidate_id": value.candidate_id,
                "compatibility_status": value.compatibility_status,
                "elapsed_seconds": value.elapsed_seconds,
                "representative_payload": value.payload.audit_payload(),
                "representative_payload_sha256": value.payload.sha256,
                "lossless_payload_artifact": lossless_by_rank[value.output_rank].audit_payload(),
            }
            for value in candidate_emissions
        ],
        "trace_ledger_sha256": ledger_sha256,
        "completion_elapsed_seconds": completion_elapsed_seconds,
        "complete_contiguous_exact_call_trace": True,
        "validation_loss_used": False,
        "representative_selection_policy": (
            "append_only" if representative_snapshots is None else "budget_snapshot"
        ),
        "representative_snapshots": (
            None if representative_snapshots is None
            else [asdict(row) for row in representative_snapshots]
        ),
    }


__all__ = [
    "V5_TUNING_EXACT_TRACE_ARTIFACT_SCHEMA",
    "V5_TUNING_EXACT_TRACE_ARTIFACT_VERSION",
    "build_v5_tuning_exact_trace_artifact",
    "write_v5_tuning_json_exclusive",
]
