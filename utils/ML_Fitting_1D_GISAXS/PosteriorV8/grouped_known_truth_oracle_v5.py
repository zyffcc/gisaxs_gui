"""Frozen one-call known-truth oracle for V5.1 solution-stage shards."""

from __future__ import annotations

from hashlib import sha256

import numpy as np

from .candidate_supervision_v5 import ExactCompatibleProvenance, FrozenSearchProvenance
from .clean_recipe_forward_v5 import (
    V5_CLEAN_EXACT_FORWARD_PATH,
    V5_CLEAN_RECIPE_PROTOCOL_VERSION,
    V5CleanRecipeLike,
    authoritative_v5_gui_parameters,
    evaluate_v5_clean_recipe_forward,
)
from .grouped_artifact_v5 import array_sha256, canonical_json
from .paper_endpoint_metrics import CURVE_EQUIVALENCE_LOG_RMSE_MAX
from .study_protocol import STUDY_PROTOCOL_SCHEMA, STUDY_PROTOCOL_VERSION


V5_ORACLE_EXACT_ARTIFACT_SCHEMA = "gisaxs.posterior_v8.known_truth_exact_forward/v1"
V5_ORACLE_SEARCH_ARTIFACT_SCHEMA = "gisaxs.posterior_v8.known_truth_one_call_search/v1"
V5_ORACLE_PROTOCOL_ID = "posterior-v8-v5-known-truth-one-exact-forward-call/v1"
V5_ORACLE_PROTOCOL_PAYLOAD = {
    "schema": "gisaxs.posterior_v8.known_truth_search_protocol/v1",
    "protocol_id": V5_ORACLE_PROTOCOL_ID,
    "stage": "solution_target_pretraining_only",
    "candidate_source": "frozen_generating_parameter_vector",
    "exact_forward_path": V5_CLEAN_EXACT_FORWARD_PATH,
    "clean_recipe_protocol_version": V5_CLEAN_RECIPE_PROTOCOL_VERSION,
    "metric_name": "exact_forward_logrmse",
    "threshold_name": "raw_curve_equivalence_logrmse",
    "threshold_value": CURVE_EQUIVALENCE_LOG_RMSE_MAX,
    "threshold_source_id": f"{STUDY_PROTOCOL_SCHEMA}|{STUDY_PROTOCOL_VERSION}",
    "exact_forward_call_budget": 1,
    "negative_labels_produced": False,
    "calibrated_search_yield_evidence": False,
}
V5_ORACLE_PROTOCOL_SHA256 = sha256(
    canonical_json(V5_ORACLE_PROTOCOL_PAYLOAD).encode("utf-8")
).hexdigest()


def run_v5_known_truth_exact_oracle(
    recipe: V5CleanRecipeLike,
    recipe_id: str,
) -> tuple[FrozenSearchProvenance, ExactCompatibleProvenance, str, str]:
    """Evaluate the generating vector exactly once and freeze its proof."""

    q = recipe.grid.values()
    exact_curve = evaluate_v5_clean_recipe_forward(recipe, q)
    parameters = np.asarray(authoritative_v5_gui_parameters(recipe), dtype=np.float64)
    target = np.asarray(recipe.target.local_target_unit, dtype=np.float32)
    amplitude_constraint = recipe.amplitude_query.constraint_for_branch(
        resolution_present=recipe.amplitude.resolution_present
    )
    if not amplitude_constraint.contains(
        recipe.amplitude.coefficient_vector,
        k=recipe.amplitude.k,
        atol=2.0e-9,
    ):
        raise RuntimeError("generating coefficients escaped their query before exact self-check")

    exact_id = f"{recipe_id}:known-truth:branch-{recipe.target.pattern_id}"
    candidate_id = f"{recipe_id}:wire-{recipe.target.pattern_id}"
    exact_payload = {
        "schema": V5_ORACLE_EXACT_ARTIFACT_SCHEMA,
        "artifact_id": exact_id,
        "protocol_sha256": V5_ORACLE_PROTOCOL_SHA256,
        "recipe_sha256": recipe.sha256,
        "geometry_query_sha256": recipe.query.sha256,
        "amplitude_query_sha256": recipe.amplitude_query.sha256,
        "branch_pattern_id": recipe.target.pattern_id,
        "target_local": target.tolist(),
        "target_local_sha256": array_sha256("oracle_target_local", target),
        "gui_parameters_sha256": array_sha256("oracle_gui_parameters", parameters),
        "q_sha256": array_sha256("oracle_q", q),
        "exact_curve_sha256": array_sha256("oracle_exact_curve", exact_curve),
        "metric_name": V5_ORACLE_PROTOCOL_PAYLOAD["metric_name"],
        "metric_value": 0.0,
        "geometry_bounds_passed": True,
        "amplitude_bounds_passed": True,
        "physics_passed": True,
        "exact_forward_calls_used": 1,
    }
    exact_json = canonical_json(exact_payload)
    exact_hash = sha256(exact_json.encode("utf-8")).hexdigest()

    search_id = f"{recipe_id}:known-truth-search:branch-{recipe.target.pattern_id}"
    search_payload = {
        "schema": V5_ORACLE_SEARCH_ARTIFACT_SCHEMA,
        "artifact_id": search_id,
        "protocol_id": V5_ORACLE_PROTOCOL_ID,
        "protocol_sha256": V5_ORACLE_PROTOCOL_SHA256,
        "candidate_id": candidate_id,
        "exact_artifact_sha256": exact_hash,
        "exact_forward_call_budget": 1,
        "exact_forward_calls_used": 1,
        "termination_reason": "compatible_known_truth_exact_one_call",
        "compatible_representative_count": 1,
        "completed": True,
    }
    search_json = canonical_json(search_payload)
    search_hash = sha256(search_json.encode("utf-8")).hexdigest()
    search = FrozenSearchProvenance(
        search_artifact_id=search_id,
        search_artifact_sha256=search_hash,
        protocol_id=V5_ORACLE_PROTOCOL_ID,
        protocol_sha256=V5_ORACLE_PROTOCOL_SHA256,
        evaluator_version=(f"{V5_CLEAN_EXACT_FORWARD_PATH}|{V5_ORACLE_EXACT_ARTIFACT_SCHEMA}"),
        metric_name=str(V5_ORACLE_PROTOCOL_PAYLOAD["metric_name"]),
        threshold_name=str(V5_ORACLE_PROTOCOL_PAYLOAD["threshold_name"]),
        threshold_value=float(V5_ORACLE_PROTOCOL_PAYLOAD["threshold_value"]),
        threshold_source_id=str(V5_ORACLE_PROTOCOL_PAYLOAD["threshold_source_id"]),
        exact_forward_call_budget=1,
        exact_forward_calls_used=1,
        termination_reason="compatible_known_truth_exact_one_call",
        completed=True,
        compatible_representative_count=1,
    )
    exact = ExactCompatibleProvenance(
        artifact_id=exact_id,
        artifact_sha256=exact_hash,
        metric_value=0.0,
        bounds_passed=True,
        physics_passed=True,
    )
    return search, exact, exact_json, search_json


__all__ = [
    "V5_ORACLE_EXACT_ARTIFACT_SCHEMA",
    "V5_ORACLE_PROTOCOL_ID",
    "V5_ORACLE_PROTOCOL_PAYLOAD",
    "V5_ORACLE_PROTOCOL_SHA256",
    "V5_ORACLE_SEARCH_ARTIFACT_SCHEMA",
    "run_v5_known_truth_exact_oracle",
]
