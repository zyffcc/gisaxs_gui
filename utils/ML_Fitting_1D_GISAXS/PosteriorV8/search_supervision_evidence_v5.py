"""Build one immutable supervision/search record from a checked runner result."""

from __future__ import annotations

from hashlib import sha256

from .candidate_supervision_v5 import (
    ExactCompatibleProvenance,
    FrozenSearchProvenance,
    V5CandidateSupervision,
)
from .grouped_artifact_v5 import canonical_json
from .search_supervision_contract_v5 import (
    V5FrozenBranchSearchResult,
    V5FrozenSearchTask,
)

V5_SEARCH_RECORD_SCHEMA = "gisaxs.posterior_v8.cross_topology_branch_search_record/v3"


def _digest_text(*values: str) -> str:
    digest = sha256()
    for value in values:
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def build_v5_search_record(
    task: V5FrozenSearchTask,
    result: V5FrozenBranchSearchResult,
) -> tuple[str, str, str]:
    identity = _digest_text(
        task.clean_group_id,
        task.observation_id,
        task.exact_curve_sha256,
        task.universal_context.audit_sha256,
        task.query_catalog_artifact_sha256,
        task.branch.global_key.wire_key,
        task.branch.context_sha256,
        task.protocol.sha256,
        task.audit_sha256,
    )
    artifact_id = f"v5-search-{identity}"
    payload = {
        "schema": V5_SEARCH_RECORD_SCHEMA,
        "artifact_id": artifact_id,
        "protocol_id": task.protocol.protocol_id,
        "protocol_sha256": task.protocol.sha256,
        "protocol_tier": task.protocol.protocol_tier,
        "task_audit_sha256": task.audit_sha256,
        "calibrated_threshold_sha256": (
            None
            if task.calibrated_threshold is None
            else task.calibrated_threshold.sha256
        ),
        "full_training_label_permitted": task.full_training_label_permitted,
        "clean_group_id": task.clean_group_id,
        "recipe_id": task.recipe_id,
        "observation_id": task.observation_id,
        "exact_curve_sha256": task.exact_curve_sha256,
        "exact_observation_audit_sha256": task.exact_observation.audit_sha256,
        "acceptance_sigma_log_available": task.observed_curve.sigma_log is not None,
        "acceptance_sigma_source_id": (
            task.exact_observation.acceptance_sigma_source_id
        ),
        "exact_metric_name": task.selected_metric_name,
        "exact_threshold_name": task.selected_threshold_name,
        "exact_threshold_value": task.selected_threshold_value,
        "exact_threshold_source_id": task.selected_threshold_source_id,
        "query_catalog_artifact_id": task.query_catalog_artifact_id,
        "query_catalog_artifact_sha256": task.query_catalog_artifact_sha256,
        "universal_query_sha256": task.universal_context.audit_sha256,
        "global_branch_key": task.branch.global_key.wire_key,
        "context_sha256": task.branch.context_sha256,
        "outcome": result.outcome,
        "completed": result.completed,
        "exact_forward_call_budget": task.protocol.exact_forward_call_budget,
        "exact_forward_calls_used": result.exact_forward_calls_used,
        "termination_reason": result.termination_reason,
        "executor_artifact_id": result.executor_artifact_id,
        "executor_artifact_sha256": result.executor_artifact_sha256,
        "compatible_representatives": [
            value.audit_payload() for value in result.representatives
        ],
        "negative_is_no_solution_certificate": False,
    }
    encoded = canonical_json(payload)
    return artifact_id, encoded, sha256(encoded.encode("utf-8")).hexdigest()


def build_v5_candidate_supervision(
    task: V5FrozenSearchTask,
    result: V5FrozenBranchSearchResult,
    search_artifact_id: str,
    search_sha256: str,
) -> V5CandidateSupervision:
    search = None
    exact = None
    target = None
    if result.completed:
        search = FrozenSearchProvenance(
            search_artifact_id=search_artifact_id,
            search_artifact_sha256=search_sha256,
            protocol_id=task.protocol.protocol_id,
            protocol_sha256=task.protocol.sha256,
            evaluator_version=task.protocol.evaluator_version,
            metric_name=task.selected_metric_name,
            threshold_name=task.selected_threshold_name,
            threshold_value=task.selected_threshold_value,
            threshold_source_id=task.selected_threshold_source_id,
            exact_forward_call_budget=task.protocol.exact_forward_call_budget,
            exact_forward_calls_used=result.exact_forward_calls_used,
            termination_reason=result.termination_reason,
            completed=True,
            compatible_representative_count=len(result.representatives),
        )
    if result.representatives:
        representative = result.representatives[0]
        exact = ExactCompatibleProvenance(
            artifact_id=representative.artifact_id,
            artifact_sha256=representative.artifact_sha256,
            metric_value=representative.metric_value,
            bounds_passed=representative.bounds_passed,
            physics_passed=representative.physics_passed,
        )
        target = representative.target_local
    candidate_id = (
        f"{task.observation_id}:{task.branch.global_key.wire_key}:"
        f"context-{task.branch.context_sha256}"
    )
    return V5CandidateSupervision(
        clean_recipe_id=task.recipe_id,
        candidate_id=candidate_id,
        outcome=result.outcome,
        active_dimension_mask=task.branch.condition.active_dimension_mask,
        varying_dimension_mask=task.branch.condition.varying_dimension_mask,
        search_provenance=search,
        exact_compatible=exact,
        target_local=target,
        generating_candidate_match=None,
    )


__all__ = [
    "V5_SEARCH_RECORD_SCHEMA",
    "build_v5_candidate_supervision",
    "build_v5_search_record",
]
