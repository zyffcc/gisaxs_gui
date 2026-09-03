"""Optional small-sample exact verification for the bounds-first V3 audit."""

from __future__ import annotations

import numpy as np

from .bounds_first_shards import BoundsFirstShardConfig, reconstruct_clean_recipe
from .bounds_holdout_metrics import (
    BoundsHoldoutAuditConfig,
    HoldoutRowReference,
    quantiles,
)
from .bounds_local_verified import run_verified_bounds_local_inference
from .build_bounds_first_shards import load_shard
from .contract import gui_component_to_latent
from .evaluation import (
    LinearSolutionSnapshot,
    ObservedCurve,
    ReferenceMode,
    evaluate_candidates,
)
from .preprocessing import preprocess_curve
from .production_bridge import (
    ProductionBranchFactory,
    ResolutionSearchPolicy,
    UserSearchSpace,
)
from .simulation import sample_observation_view, simulate_recipe


OBSERVABILITY_NOT_RUN = "not_run_exact_subset_disabled"
OBSERVABILITY_NOT_APPLICABLE = "not_assessed_no_compatible_candidate"
OBSERVABILITY_ASSESSED = "assessed_without_unknown"
OBSERVABILITY_UNKNOWN = "provisional_or_unknown"


def reconstruct_observed(reference: HoldoutRowReference, *, shard=None):
    """Recreate one immutable synthetic view and verify its stored encoder tensors."""

    shard = load_shard(reference.path) if shard is None else shard
    if shard.path.resolve() != reference.path.resolve():
        raise ValueError("provided V4 shard does not match the exact row reference")
    config = BoundsFirstShardConfig(**shard.metadata["config"])
    clean = reconstruct_clean_recipe(config, reference.recipe_index)
    view = sample_observation_view(
        clean.recipe_seed, reference.view_index, max_points=config.max_raw_points
    )
    simulated = simulate_recipe(view.simulation_recipe(clean.simulation_recipe))
    preprocessed = preprocess_curve(
        simulated.q,
        simulated.intensity,
        simulated.sigma,
        mask=view.selection_mask(simulated.q),
        q_range=view.preprocess_q_range,
    )
    for name, expected in (
        ("x", preprocessed.x),
        ("point_mask", preprocessed.point_mask),
        ("global_features", preprocessed.global_features),
    ):
        if not np.array_equal(shard.arrays[name][reference.row], expected):
            raise ValueError(f"reconstructed observation disagrees with shard {name}")
    q, intensity, sigma = preprocessed.valid_arrays()
    observed = ObservedCurve(
        curve_id=f"v4-r{reference.recipe_index}-v{reference.view_index}",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
        sigma_log=sigma / intensity,
    )
    return shard, clean, observed


def exact_record(reference, model, config: BoundsHoldoutAuditConfig, *, shard=None):
    shard, clean, observed = reconstruct_observed(reference, shard=shard)
    bounds = clean.label.bounds
    resolution_policy = (
        ResolutionSearchPolicy(presence="required", bounds=bounds.resolution_bounds)
        if bounds.resolution_bounds is not None
        else ResolutionSearchPolicy(presence="absent")
    )
    factory = ProductionBranchFactory(
        UserSearchSpace.for_components(bounds.component_bounds, resolution=resolution_policy)
    )
    curve_inputs = {
        name: shard.arrays[name][reference.row] for name in ("x", "point_mask", "global_features")
    }
    result = run_verified_bounds_local_inference(
        curve_inputs,
        observed,
        model=model,
        branch_factory=factory,
        thresholds=config.thresholds,
        target_parameter_mode_count=config.target_parameter_mode_count,
        inference_budget=config.inference_budget,
        rescue_policy=config.rescue_policy,
        seed=int(
            np.random.SeedSequence(
                [config.seed, reference.recipe_index, reference.view_index, 0x45584143]
            ).generate_state(1, dtype=np.uint32)[0]
        ),
    )
    report = result.evaluation_report
    truth_recall = None
    parameter_mode_count = 0
    accepted_count = 0
    if report is not None:
        truth = ReferenceMode(
            reference_id="synthetic_generation_truth",
            topology_id=clean.simulation_recipe.topology_id,
            components=tuple(
                gui_component_to_latent(item) for item in clean.label.truth_components
            ),
            resolution=clean.label.truth_resolution,
            linear_solution=LinearSolutionSnapshot(
                background=clean.simulation_recipe.background,
                particle_amplitudes=clean.simulation_recipe.effective_amplitudes,
                resolution_amplitude=clean.simulation_recipe.resolution_effective_amplitude,
            ),
        )
        with_reference = evaluate_candidates(
            observed,
            result.raw_candidates,
            thresholds=config.thresholds,
            best_of_n=(len(result.raw_candidates),),
            reference_modes=(truth,),
        )
        truth_recall = with_reference.mode_recall
        parameter_mode_count = with_reference.predicted_parameter_mode_count
        accepted_count = with_reference.accepted_count
    assessment_statuses = [item.status for item in result.observability_assessments]
    assessment_status_distribution = {
        str(value): int(count)
        for value, count in zip(*np.unique(assessment_statuses, return_counts=True))
    }
    unknown_count = assessment_status_distribution.get(OBSERVABILITY_UNKNOWN, 0)
    if unknown_count:
        observability_status = OBSERVABILITY_UNKNOWN
    elif assessment_statuses:
        observability_status = OBSERVABILITY_ASSESSED
    else:
        observability_status = OBSERVABILITY_NOT_APPLICABLE
    return {
        "recipe_index": reference.recipe_index,
        "view_index": reference.view_index,
        "bounds_sha256": bounds.sha256,
        "verified_status": result.status,
        "raw_status": result.raw_generation.status,
        "finite_search_failure_is_no_solution": False,
        "candidate_count": len(result.raw_candidates),
        "accepted_candidate_count": accepted_count,
        "compatible_candidate_found": accepted_count > 0,
        "compatibility_threshold_source": "unfrozen_engineering_thresholds",
        "formal_paper_compatibility_claim_allowed": False,
        "primary_success_semantics": (
            "exact_compatible_candidate_and_pre_observability_parameter_modes"
        ),
        "observability_metrics_role": "secondary_diagnostic_not_success_gate",
        "parameter_mode_count_before_observability": parameter_mode_count,
        "confirmed_effective_parameter_mode_count": result.effective_parameter_mode_count,
        "truth_parameter_mode_recalled": truth_recall,
        "observability_status": observability_status,
        "observability_assessment_status_distribution": assessment_status_distribution,
        "observability_unknown_count": unknown_count,
        "forward_budget": {
            "limit": result.total_forward_evaluation_limit,
            "used": result.total_forward_evaluations,
            "remaining": result.total_forward_evaluation_limit - result.total_forward_evaluations,
            "raw_refinement_used": result.raw_refinement_forward_evaluations,
            "observability_used": result.observability_exact_forward_evaluations,
            "observability_limit": result.observability_forward_evaluation_limit,
            "per_candidate_limit": config.inference_budget.per_candidate_forward_evaluation_limit,
            "source_usage": {
                item.source: item.forward_evaluations for item in result.audit.sources
            },
        },
        "evaluation_audit_schema": None if report is None else report.audit_schema,
    }


def _distribution(records, field):
    values, counts = np.unique([item[field] for item in records], return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts)}


def _aggregate_nested_distribution(records, field):
    aggregate: dict[str, int] = {}
    for item in records:
        for name, count in item[field].items():
            aggregate[name] = aggregate.get(name, 0) + int(count)
    return dict(sorted(aggregate.items()))


def exact_summary(records, config: BoundsHoldoutAuditConfig):
    if not records:
        return {
            "enabled": False,
            "example_count": 0,
            "reason": "exact_maximum_examples_is_zero",
            "compatibility_threshold_source": "unfrozen_engineering_thresholds",
            "compatibility_calibration_artifact_sha256": None,
            "formal_paper_compatibility_claim_allowed": False,
            "primary_success_semantics": (
                "exact_compatible_candidate_and_pre_observability_parameter_modes"
            ),
            "observability_metrics_role": "secondary_diagnostic_not_success_gate",
            "finite_search_failure_is_no_solution": False,
            "budget_ledger": {
                "scope": "all_refinement_and_observability_exact_forward_calls",
                "configured_per_curve_limit": config.inference_budget.forward_evaluation_limit,
                "configured_total_limit": 0,
                "used": 0,
                "remaining": 0,
                "observed_curve_reconstruction_calls": 0,
                "observed_curve_reconstruction_in_search_budget": False,
                "observability_calls": 0,
                "observability_assessment_count": 0,
                "observability_unknown_assessment_count": 0,
                "observability_unknown_status_preserved": True,
                "observability_status": OBSERVABILITY_NOT_RUN,
            },
        }
    used = sum(item["forward_budget"]["used"] for item in records)
    raw_used = sum(item["forward_budget"]["raw_refinement_used"] for item in records)
    observability_used = sum(item["forward_budget"]["observability_used"] for item in records)
    remaining = sum(item["forward_budget"]["remaining"] for item in records)
    observability_assessment_distribution = _aggregate_nested_distribution(
        records, "observability_assessment_status_distribution"
    )
    observability_assessment_count = sum(observability_assessment_distribution.values())
    observability_unknown_count = observability_assessment_distribution.get(
        OBSERVABILITY_UNKNOWN, 0
    )
    return {
        "enabled": True,
        "example_count": len(records),
        "compatible_candidate_success_rate": float(
            np.mean([item["compatible_candidate_found"] for item in records])
        ),
        "compatibility_threshold_source": "unfrozen_engineering_thresholds",
        "compatibility_calibration_artifact_sha256": None,
        "formal_paper_compatibility_claim_allowed": False,
        "primary_success_semantics": (
            "exact_compatible_candidate_and_pre_observability_parameter_modes"
        ),
        "observability_metrics_role": "secondary_diagnostic_not_success_gate",
        "truth_parameter_mode_recall_rate": float(
            np.mean([item["truth_parameter_mode_recalled"] == 1.0 for item in records])
        ),
        "candidate_count": quantiles([item["candidate_count"] for item in records]),
        "parameter_mode_count_before_observability": quantiles(
            [item["parameter_mode_count_before_observability"] for item in records]
        ),
        "confirmed_effective_parameter_mode_count": quantiles(
            [item["confirmed_effective_parameter_mode_count"] for item in records]
        ),
        "verified_status_distribution": _distribution(records, "verified_status"),
        "raw_status_distribution": _distribution(records, "raw_status"),
        "observability_status_distribution": _distribution(records, "observability_status"),
        "observability_assessment_status_distribution": observability_assessment_distribution,
        "finite_search_failure_is_no_solution": False,
        "budget_ledger": {
            "scope": "all_refinement_and_observability_exact_forward_calls",
            "configured_per_curve_limit": config.inference_budget.forward_evaluation_limit,
            "configured_total_limit": sum(item["forward_budget"]["limit"] for item in records),
            "used": used,
            "remaining": remaining,
            "raw_refinement_calls": raw_used,
            "observability_calls": observability_used,
            "nominal_observability_sub_limits": sum(
                item["forward_budget"]["observability_limit"] for item in records
            ),
            "observed_curve_reconstruction_calls": len(records),
            "observed_curve_reconstruction_in_search_budget": False,
            "observability_assessment_count": observability_assessment_count,
            "observability_unknown_assessment_count": observability_unknown_count,
            "observability_unknown_status_preserved": True,
        },
    }


__all__ = [
    "OBSERVABILITY_ASSESSED",
    "OBSERVABILITY_NOT_APPLICABLE",
    "OBSERVABILITY_NOT_RUN",
    "OBSERVABILITY_UNKNOWN",
    "exact_record",
    "exact_summary",
    "reconstruct_observed",
]
