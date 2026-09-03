"""Sobol + exact-refinement competing-model search for a reference bank.

The default CLI is deliberately a K1--K2 pilot.  The same API enumerates the
418 permutation-quotiented physical V3/V4 branches when
``maximum_components`` is four.  Heavy runs belong on a Slurm worker.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import platform
import sys
import time
from typing import Callable, Mapping, Sequence

import numpy as np
import scipy

from .branch_catalog import BRANCH_CATALOG_VERSION
from .branch_codec import BRANCH_CODEC_VERSION, ResolutionBounds
from .canonical_branch_catalog import (
    CANONICAL_BRANCH_COUNT,
    CANONICAL_BRANCH_CATALOG_VERSION,
    canonical_branch_pattern_is_valid,
)
from .compatibility_calibration import (
    CALIBRATION_SCHEMA,
    CALIBRATION_VERSION,
    COMPATIBILITY_STRATUM_FIELDS,
    COMPATIBILITY_STRATUM_VERSION,
    STRATIFICATION_SEMANTICS,
    CompatibilityCalibrationArtifact,
    load_compatibility_calibration,
)
from .component_observability import (
    COMPONENT_OBSERVABILITY_VERSION,
    OBSERVABILITY_POLICY_VERSION,
)
from .contract import (
    CODEC_VERSION,
    CONTRACT_VERSION,
    FORWARD_MODEL_VERSION,
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    GuiComponentBounds,
    full_component_bounds,
)
from .evaluation import (
    CLUSTERING_LINKAGE,
    PARAMETER_DISTANCE_SCOPE,
    PARAMETER_NORMALIZATION_VERSION,
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
    CandidateInput,
    LinearSolutionSnapshot,
    ObservedCurve,
    ReferenceMode,
)
from .profiled_refinement import refine_profiled_branch
from .proposal_sampling import PROPOSAL_SAMPLING_VERSION, generate_profiled_branch_seeds
from .reduced_model_search import (
    REDUCED_MODEL_SEARCH_VERSION,
    BoundedReducedModelSearcher,
)
from .reference_bank import (
    BRANCH_ENUMERATION_VERSION,
    COMPATIBILITY_SEMANTICS,
    DISCOVERY_CLAIM,
    PRIMARY_REFERENCE_GATES,
    PRIMARY_REFERENCE_SEARCH_SOURCE,
    PRIMARY_REFERENCE_SEMANTICS,
    REFERENCE_BANK_SCHEMA,
    REFERENCE_BANK_VERSION,
    REFERENCE_MODE_CLUSTERING_VERSION,
    STRICT_MINIMAL_SECONDARY_SEMANTICS,
    CalibrationObservationProvenance,
    CandidateDiscovery,
    CompatibilityPolicy,
    CompetingBranch,
    assess_candidate,
    enumerate_competing_branches,
    generating_diagnostic,
    primary_reference_groups,
    reference_bank_scientific_sha256,
    strict_minimal_secondary_groups,
    write_reference_bank_atomic,
)


SEARCH_VERSION = "posterior_v8_network_free_reference_search_v5"


@dataclass(frozen=True, kw_only=True)
class CompetingSearchConfig:
    seed: int
    starts_per_branch: int
    max_nfev: int
    minimum_components: int = 1
    maximum_components: int = 2
    saturation_rounds: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

    def __post_init__(self) -> None:
        for name in ("seed", "starts_per_branch", "max_nfev"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            if int(value) < int(name != "seed"):
                raise ValueError(
                    f"{name} must be {'positive' if name != 'seed' else 'non-negative'}"
                )
            object.__setattr__(self, name, int(value))
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in (self.minimum_components, self.maximum_components)
        ):
            raise TypeError("component limits must be integers")
        minimum, maximum = int(self.minimum_components), int(self.maximum_components)
        if not 1 <= minimum <= maximum <= 4:
            raise ValueError("component limits must satisfy 1 <= minimum <= maximum <= 4")
        object.__setattr__(self, "minimum_components", minimum)
        object.__setattr__(self, "maximum_components", maximum)
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
            for value in self.saturation_rounds
        ):
            raise TypeError("saturation_rounds must contain integers")
        rounds = tuple(sorted({int(value) for value in self.saturation_rounds}))
        if not rounds or rounds[0] < 1:
            raise ValueError("saturation_rounds must contain positive integers")
        object.__setattr__(self, "saturation_rounds", rounds)


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _provenance(
    config: CompetingSearchConfig,
    policy: CompatibilityPolicy,
) -> dict[str, object]:
    root = Path(__file__).resolve().parent
    repository = Path(__file__).resolve().parents[3]
    sources = {
        name: root / name
        for name in (
            "competing_model_search.py",
            "reference_bank.py",
            "component_observability.py",
            "component_observability_diagnostics.py",
            "observability_assessment.py",
            "reduced_model_search.py",
            "compatibility_calibration.py",
            "contract.py",
            "branch_catalog.py",
            "canonical_branch_catalog.py",
            "branch_codec.py",
            "profiled_forward.py",
            "profiled_refinement.py",
            "proposal_sampling.py",
            "evaluation.py",
        )
    }
    sources.update(
        {
            "src/gimap/features/fitting/domain/scattering_model.py": repository
            / "src/gimap/features/fitting/domain/scattering_model.py",
            "src/gimap/features/fitting/domain/physical_constraints.py": repository
            / "src/gimap/features/fitting/domain/physical_constraints.py",
        }
    )
    return {
        "search_version": SEARCH_VERSION,
        "contract_version": CONTRACT_VERSION,
        "codec_version": CODEC_VERSION,
        "branch_codec_version": BRANCH_CODEC_VERSION,
        "branch_catalog_version": BRANCH_CATALOG_VERSION,
        "canonical_branch_catalog_version": CANONICAL_BRANCH_CATALOG_VERSION,
        "branch_enumeration_version": BRANCH_ENUMERATION_VERSION,
        "forward_model_version": FORWARD_MODEL_VERSION,
        "proposal_sampling_version": PROPOSAL_SAMPLING_VERSION,
        "component_observability_version": COMPONENT_OBSERVABILITY_VERSION,
        "observability_policy_version": OBSERVABILITY_POLICY_VERSION,
        "reduced_model_search_version": REDUCED_MODEL_SEARCH_VERSION,
        "calibration_version": (None if policy.calibration is None else CALIBRATION_VERSION),
        "calibration_schema": (None if policy.calibration is None else CALIBRATION_SCHEMA),
        "calibration_stratum_version": (
            None if policy.calibration is None else COMPATIBILITY_STRATUM_VERSION
        ),
        "calibration_stratum_fields": (
            None if policy.calibration is None else list(COMPATIBILITY_STRATUM_FIELDS)
        ),
        "calibration_stratification_semantics": (
            None if policy.calibration is None else STRATIFICATION_SEMANTICS
        ),
        "calibration_input_sha256": (
            None if policy.calibration is None else policy.calibration.input_sha256
        ),
        "calibration_dataset_manifest_sha256": (
            None if policy.calibration is None else policy.calibration.dataset_manifest_sha256
        ),
        "calibration_split_sha256": (
            None if policy.calibration is None else policy.calibration.calibration_split_sha256
        ),
        "config": asdict(config),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_sha256": {name: _source_sha256(path) for name, path in sorted(sources.items())},
    }


def _bounds_for(
    branches: Sequence[CompetingBranch],
    overrides: Mapping[int, Sequence[GuiComponentBounds]] | None,
) -> dict[int, tuple[GuiComponentBounds, ...]]:
    selected_topologies = sorted({branch.topology_id for branch in branches})
    unknown = set() if overrides is None else set(overrides) - set(selected_topologies)
    if unknown:
        raise ValueError(f"component-bound overrides contain unsearched topology IDs: {unknown}")
    result = {}
    for topology_id in selected_topologies:
        topology = branches[
            next(index for index, value in enumerate(branches) if value.topology_id == topology_id)
        ].topology
        values = (
            tuple(full_component_bounds(shape, d_policy="optional") for shape in topology)
            if overrides is None or topology_id not in overrides
            else tuple(overrides[topology_id])
        )
        if len(values) != len(topology) or tuple(value.shape for value in values) != topology:
            raise ValueError(
                f"component bounds for topology {topology_id} do not match canonical order"
            )
        result[topology_id] = values
    return result


def _policy_payload(policy: CompatibilityPolicy) -> dict[str, object]:
    return {
        "raw_log_rmse_max": policy.raw_log_rmse_max,
        "standardized_log_rmse_max": policy.standardized_log_rmse_max,
        "calibration_schema": (None if policy.calibration is None else policy.calibration.schema),
        "calibration_version": (
            None if policy.calibration is None else policy.calibration.calibration_version
        ),
        "calibration_stratum_version": (
            None if policy.calibration is None else policy.calibration.compatibility_stratum_version
        ),
        "calibration_stratum_fields": (
            None
            if policy.calibration is None
            else list(policy.calibration.compatibility_stratum_fields)
        ),
        "calibration_stratification_semantics": (
            None if policy.calibration is None else policy.calibration.stratification_semantics
        ),
        "calibration_input_sha256": (
            None if policy.calibration is None else policy.calibration.input_sha256
        ),
        "calibration_dataset_manifest_sha256": (
            None if policy.calibration is None else policy.calibration.dataset_manifest_sha256
        ),
        "calibration_split_sha256": (
            None if policy.calibration is None else policy.calibration.calibration_split_sha256
        ),
        "calibration_observation": (
            None
            if policy.calibration_observation is None
            else asdict(policy.calibration_observation)
        ),
        "observability": asdict(policy.observability),
        "parameter_mode_distance_max": policy.parameter_mode_distance_max,
        "curve_equivalence_log_rmse_max": policy.curve_equivalence_log_rmse_max,
        "generating_mode_distance_max": policy.generating_mode_distance_max,
    }


def _saturation(
    discoveries: Sequence[CandidateDiscovery],
    attempts: Sequence[Mapping[str, object]],
    branches: Sequence[CompetingBranch],
    config: CompetingSearchConfig,
    policy: CompatibilityPolicy,
) -> list[dict[str, object]]:
    points = []
    previous_modes = 0
    rounds = sorted(
        {min(value, config.starts_per_branch) for value in config.saturation_rounds}
        | {config.starts_per_branch}
    )
    for round_count in rounds:
        budget = round_count * len(branches)
        prefix = [value for value in discoveries if value.attempt_rank <= budget]
        modes, curve_groups = primary_reference_groups(prefix, policy)
        eligible = [value for value in prefix if value.primary_reference_eligible]
        attempt_prefix = [value for value in attempts if int(value["attempt_rank"]) <= budget]
        points.append(
            {
                "rounds_per_branch": round_count,
                "scheduled_attempt_budget": budget,
                "attempted_refinements": sum(
                    value["status"] != "branch_unavailable" for value in attempt_prefix
                ),
                "returned_candidates": len(prefix),
                "primary_reference_candidate_count": len(eligible),
                "primary_reference_topology_count": len(
                    {value.branch.topology_id for value in eligible}
                ),
                "primary_reference_branch_count": len({value.branch.key for value in eligible}),
                "primary_reference_mode_count": len(modes),
                "delta_primary_reference_mode_count": len(modes) - previous_modes,
                "primary_reference_curve_group_count": len(curve_groups),
                "refinement_nfev": sum(int(value.get("nfev", 0)) for value in attempt_prefix),
                "residual_calls": sum(
                    int(value.get("residual_calls", 0)) for value in attempt_prefix
                ),
                "observability_budget_affects_this_saturation_curve": False,
            }
        )
        previous_modes = len(modes)
    return points


def _primary_reference_qualification(
    primary_mode_count: int,
    saturation: Sequence[Mapping[str, object]],
    branch_audit: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    unavailable_branches = [
        str(value["key"]) for value in branch_audit if value.get("preparation_status") != "ready"
    ]
    plateau_observed = (
        len(saturation) >= 2 and int(saturation[-1]["delta_primary_reference_mode_count"]) == 0
    )
    if unavailable_branches:
        status = "INCOMPLETE_BRANCH_COVERAGE"
        reason = "one_or_more_scheduled_branches_could_not_be_prepared"
    elif primary_mode_count == 0:
        status = "EMPTY_WITHIN_FINITE_SEARCH"
        reason = "no_candidate_passed_primary_gates_within_the_executed_budget"
    elif not plateau_observed:
        status = "UNSATURATED_WITHIN_FINITE_SEARCH"
        reason = "final_scheduled_round_added_modes_or_too_few_rounds_to_assess_plateau"
    else:
        status = "SCHEDULE_PLATEAU_OBSERVED"
        reason = "final_scheduled_round_added_no_primary_modes_but_this_is_not_completeness_proof"
    return {
        "qualification_status": status,
        "qualification_reason": reason,
        "schedule_plateau_observed": plateau_observed,
        "completeness_certificate_status": "NOT_ESTABLISHED",
        "paper_freeze_status": "NOT_QUALIFIED_NO_INDEPENDENT_CERTIFICATE",
        "paper_freeze_reason": "cross_optimizer_or_equivalent_independent_certificate_not_run",
        "unavailable_branch_keys": unavailable_branches,
        "finite_search_only": True,
        "does_not_establish_all_solutions_or_no_solution": True,
    }


def run_competing_model_search(
    curve: ObservedCurve,
    *,
    config: CompetingSearchConfig,
    policy: CompatibilityPolicy,
    branches: Sequence[CompetingBranch] | None = None,
    component_bounds_by_topology: Mapping[int, Sequence[GuiComponentBounds]] | None = None,
    resolution_bounds: ResolutionBounds | None = None,
    generating_mode: ReferenceMode | None = None,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run a finite competing-branch search and return one audit payload."""

    if not isinstance(curve, ObservedCurve):
        raise TypeError("curve must be an ObservedCurve")
    if not isinstance(config, CompetingSearchConfig):
        raise TypeError("config must be a CompetingSearchConfig")
    if not isinstance(policy, CompatibilityPolicy):
        raise TypeError("policy must be a CompatibilityPolicy")
    if branches is None:
        selected = enumerate_competing_branches(
            minimum_components=config.minimum_components,
            maximum_components=config.maximum_components,
        )
    else:
        selected = tuple(branches)
        if not selected or not all(isinstance(value, CompetingBranch) for value in selected):
            raise ValueError("branches must contain CompetingBranch values")
        if len(set(selected)) != len(selected):
            raise ValueError("branches must be unique")
        selected = tuple(sorted(selected))
    if any(
        not canonical_branch_pattern_is_valid(value.topology_id, value.pattern_id)
        for value in selected
    ):
        raise ValueError("reference-bank search requires the canonical 418-branch catalog")
    bounds = _bounds_for(selected, component_bounds_by_topology)
    resolution_bounds = resolution_bounds or ResolutionBounds(
        RESOLUTION_SIGMA_DOMAIN, RESOLUTION_NU_DOMAIN
    )
    compatibility_threshold, threshold_source, used_fallback = policy.threshold_for(curve)
    compatibility_metric = (
        RAW_LOG_RMSE_METRIC if curve.sigma_log is None else STANDARDIZED_LOG_RMSE_METRIC
    )

    started = time.monotonic()
    prepared = {}
    branch_audit = []
    for branch_index, branch in enumerate(selected):
        try:
            seeds = generate_profiled_branch_seeds(
                branch.topology,
                bounds[branch.topology_id],
                branch.d_present,
                resolution_bounds=resolution_bounds if branch.resolution_present else None,
                seed=config.seed + branch_index,
                count=config.starts_per_branch,
            )
            prepared[branch] = seeds
            branch_audit.append({**branch.to_payload(), "preparation_status": "ready"})
        except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
            prepared[branch] = None
            branch_audit.append(
                {
                    **branch.to_payload(),
                    "preparation_status": "unavailable",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    attempts: list[dict[str, object]] = []
    refined_candidates: list[tuple[CandidateInput, CompetingBranch, int]] = []
    proposal_rank = 0
    for sequence_index in range(config.starts_per_branch):
        for branch_index, branch in enumerate(selected):
            attempt_rank = sequence_index * len(selected) + branch_index + 1
            seeds = prepared[branch]
            if seeds is None:
                attempts.append(
                    {
                        "attempt_rank": attempt_rank,
                        "branch_key": branch.key,
                        "sequence_index": sequence_index,
                        "status": "branch_unavailable",
                    }
                )
                continue
            call_started = time.monotonic()
            try:
                result = refine_profiled_branch(
                    curve.q,
                    curve.intensity,
                    sigma_log=curve.sigma_log,
                    max_nfev=config.max_nfev,
                    ftol=1.0e-7,
                    xtol=1.0e-7,
                    gtol=1.0e-7,
                    **seeds[sequence_index].refinement_kwargs(),
                )
                proposal_rank += 1
                candidate_id = f"candidate_{proposal_rank:06d}"
                profile = result.final_profile
                candidate = CandidateInput(
                    candidate_id=candidate_id,
                    proposal_rank=proposal_rank,
                    topology_id=branch.topology_id,
                    components=result.final_latent_components,
                    resolution=result.final_resolution,
                    linear_solution=LinearSolutionSnapshot(
                        background=profile.background,
                        particle_amplitudes=profile.particle_amplitudes,
                        resolution_amplitude=profile.resolution_amplitude,
                        k=profile.k,
                    ),
                    exact_intensity=result.exact_forward_intensity,
                    bounds_pass=bool(result.bounds_satisfied),
                    physics_pass=True,
                )
                refined_candidates.append((candidate, branch, attempt_rank))
                attempts.append(
                    {
                        "attempt_rank": attempt_rank,
                        "branch_key": branch.key,
                        "sequence_index": sequence_index,
                        "status": "returned",
                        "candidate_id": candidate_id,
                        "optimizer_converged": bool(result.success),
                        "nfev": int(result.nfev),
                        "residual_calls": int(result.residual_calls),
                        "initial_raw_log_rmse": float(result.initial_log_rmse),
                        "final_raw_log_rmse": float(result.final_log_rmse),
                        "elapsed_seconds": time.monotonic() - call_started,
                    }
                )
            except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
                attempts.append(
                    {
                        "attempt_rank": attempt_rank,
                        "branch_key": branch.key,
                        "sequence_index": sequence_index,
                        "status": "exception",
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": time.monotonic() - call_started,
                    }
                )
            if progress is not None:
                progress(attempts[-1].copy())

    primary_search_elapsed = time.monotonic() - started
    observability_started = time.monotonic()
    discoveries: list[CandidateDiscovery] = []
    observability_attempts = []
    for candidate, branch, attempt_rank in refined_candidates:
        observation_started = time.monotonic()
        discovery = assess_candidate(
            curve,
            candidate,
            branch,
            attempt_rank=attempt_rank,
            search_source=PRIMARY_REFERENCE_SEARCH_SOURCE,
            policy=policy,
            reduced_model_searcher=BoundedReducedModelSearcher(
                component_bounds=bounds[branch.topology_id],
                resolution_bounds=(resolution_bounds if branch.resolution_present else None),
                seed=config.seed + attempt_rank,
            ),
        )
        discoveries.append(discovery)
        observability_attempts.append(
            {
                "candidate_id": candidate.candidate_id,
                "status": discovery.observability.status,
                "diagnostic_error": discovery.observability.diagnostic_error,
                "exact_forward_calls": discovery.observability.exact_forward_calls,
                "exact_forward_call_limit": discovery.observability.exact_forward_call_limit,
                "budget_exhausted": discovery.observability.budget_exhausted,
                "elapsed_seconds": time.monotonic() - observation_started,
            }
        )
    observability_elapsed = time.monotonic() - observability_started

    primary_modes, primary_curve_groups = primary_reference_groups(discoveries, policy)
    strict_modes, strict_curve_groups = strict_minimal_secondary_groups(discoveries, policy)
    saturation = _saturation(discoveries, attempts, selected, config, policy)
    primary_eligible = [value for value in discoveries if value.primary_reference_eligible]
    strict_eligible = [value for value in discoveries if value.strict_minimal_secondary_eligible]
    statuses = ("confirmed_effective", "confirmed_redundant", "provisional_or_unknown")
    all_observability_counts = {
        status: sum(value.observability.status == status for value in discoveries)
        for status in statuses
    }
    primary_observability_counts = {
        status: sum(value.observability.status == status for value in primary_eligible)
        for status in statuses
    }
    qualification = _primary_reference_qualification(len(primary_modes), saturation, branch_audit)
    payload: dict[str, object] = {
        "schema": REFERENCE_BANK_SCHEMA,
        "reference_bank_version": REFERENCE_BANK_VERSION,
        "branch_enumeration_version": BRANCH_ENUMERATION_VERSION,
        "discovery_claim": DISCOVERY_CLAIM,
        "compatibility_semantics": COMPATIBILITY_SEMANTICS,
        "primary_reference_semantics": PRIMARY_REFERENCE_SEMANTICS,
        "strict_minimal_secondary_semantics": STRICT_MINIMAL_SECONDARY_SEMANTICS,
        "primary_reference_source_policy": {
            "required_search_source": PRIMARY_REFERENCE_SEARCH_SOURCE,
            "denominator_is_network_free": True,
            "evaluated_network_proposals_allowed_in_primary_denominator": False,
            "network_proposal_union_policy": "secondary_or_leave_one_out_sensitivity_only",
        },
        "primary_reference_gates": list(PRIMARY_REFERENCE_GATES),
        "state": "COMPUTATION_COMPLETE",
        "curve": {
            "curve_id": curve.curve_id,
            "source_kind": curve.source_kind,
            "point_count": int(curve.q.size),
            "q_min": float(curve.q[0]),
            "q_max": float(curve.q[-1]),
            "has_sigma_log": curve.sigma_log is not None,
            "q_sha256": hashlib.sha256(np.asarray(curve.q, dtype="<f8").tobytes()).hexdigest(),
            "intensity_sha256": hashlib.sha256(
                np.asarray(curve.intensity, dtype="<f8").tobytes()
            ).hexdigest(),
        },
        "search_space": {
            "branch_enumeration_version": BRANCH_ENUMERATION_VERSION,
            "full_canonical_catalog_branch_count": CANONICAL_BRANCH_COUNT,
            "branch_count": len(selected),
            "topology_ids": sorted({value.topology_id for value in selected}),
            "branches": branch_audit,
            "component_bounds_by_topology": {
                str(key): [asdict(value) for value in values]
                for key, values in sorted(bounds.items())
            },
            "resolution_bounds": asdict(resolution_bounds),
        },
        "compatibility_policy": _policy_payload(policy),
        "primary_compatibility_gate": {
            "metric": compatibility_metric,
            "comparison": "less_than_or_equal",
            "threshold": compatibility_threshold,
            "threshold_source": threshold_source,
            "used_descriptive_fallback": used_fallback,
            "exact_gui_forward_required": True,
        },
        "parameter_mode_clustering": {
            "distance_metric_version": PARAMETER_NORMALIZATION_VERSION,
            "distance_scope": PARAMETER_DISTANCE_SCOPE,
            "distance_threshold": policy.parameter_mode_distance_max,
            "linkage": REFERENCE_MODE_CLUSTERING_VERSION,
            "hard_branches_clustered_separately": True,
        },
        "curve_equivalence_clustering": {
            "distance_metric": RAW_LOG_RMSE_METRIC,
            "distance_threshold": policy.curve_equivalence_log_rmse_max,
            "linkage": CLUSTERING_LINKAGE,
            "representative_curve_source": "authoritative_exact_gui_forward",
        },
        "provenance": _provenance(config, policy),
        "primary_search_budget": {
            "search_source": PRIMARY_REFERENCE_SEARCH_SOURCE,
            "scheduled_attempts": len(selected) * config.starts_per_branch,
            "attempted_refinements": sum(
                value["status"] != "branch_unavailable" for value in attempts
            ),
            "returned_candidates": len(discoveries),
            "optimizer_converged": sum(
                bool(value.get("optimizer_converged")) for value in attempts
            ),
            "refinement_nfev": sum(int(value.get("nfev", 0)) for value in attempts),
            "residual_calls": sum(int(value.get("residual_calls", 0)) for value in attempts),
            "observability_calls_included": False,
        },
        "post_search_observability_budget": {
            "candidate_count": len(discoveries),
            "configured_exact_forward_call_limit_per_candidate": (
                policy.observability.exact_forward_call_limit
            ),
            "total_exact_forward_call_limit": sum(
                value.observability.exact_forward_call_limit for value in discoveries
            ),
            "total_exact_forward_calls": sum(
                value.observability.exact_forward_calls for value in discoveries
            ),
            "budget_exhausted_candidate_count": sum(
                value.observability.budget_exhausted for value in discoveries
            ),
            "diagnostic_error_candidate_count": sum(
                value.observability.diagnostic_error is not None for value in discoveries
            ),
            "affects_primary_search_budget_or_saturation": False,
        },
        "runtime_seconds": {
            "primary_search": primary_search_elapsed,
            "post_search_observability": observability_elapsed,
            "total": time.monotonic() - started,
        },
        "primary_search_attempts": attempts,
        "post_search_observability_attempts": observability_attempts,
        "candidates": [value.to_payload() for value in discoveries],
        "primary_reference_modes": list(primary_modes),
        "primary_reference_curve_equivalence_groups": list(primary_curve_groups),
        "strict_minimal_secondary_modes": list(strict_modes),
        "strict_minimal_secondary_curve_equivalence_groups": list(strict_curve_groups),
        "primary_reference_inventory": {
            "candidate_ids": [value.candidate.candidate_id for value in primary_eligible],
            "topology_ids": sorted({value.branch.topology_id for value in primary_eligible}),
            "branch_keys": sorted({value.branch.key for value in primary_eligible}),
            "mode_ids": [value["mode_id"] for value in primary_modes],
            "curve_group_ids": [value["curve_group_id"] for value in primary_curve_groups],
            "strict_minimal_secondary_candidate_ids": [
                value.candidate.candidate_id for value in strict_eligible
            ],
            "strict_minimal_secondary_mode_ids": [value["mode_id"] for value in strict_modes],
        },
        "observability_summary": {
            "all_candidate_status_counts": all_observability_counts,
            "primary_reference_status_counts": primary_observability_counts,
            "primary_reference_unknown_candidate_count": primary_observability_counts[
                "provisional_or_unknown"
            ],
            "primary_reference_confirmed_redundant_candidate_count": (
                primary_observability_counts["confirmed_redundant"]
            ),
            "primary_reference_confirmed_effective_candidate_count": (
                primary_observability_counts["confirmed_effective"]
            ),
            "observability_is_not_a_primary_eligibility_gate": True,
        },
        "primary_reference_qualification": qualification,
        "primary_mode_discovery_saturation": saturation,
        "generating_diagnostic": generating_diagnostic(
            discoveries,
            primary_modes,
            strict_modes,
            generating_mode,
            distance_max=policy.generating_mode_distance_max,
        ),
        "cross_optimizer_validation": {
            "status": "NOT_RUN",
            "required_before_paper_reference_freeze": True,
            "absence_does_not_invalidate_finite_discovered_set": True,
        },
    }
    payload["scientific_payload_sha256"] = reference_bank_scientific_sha256(payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="NPZ with q/intensity[/sigma_log]"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--curve-id", default="reference_bank_curve")
    parser.add_argument(
        "--source-kind", choices=("synthetic", "real_cut_data"), default="synthetic"
    )
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--starts-per-branch", type=int, default=4)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--minimum-components", type=int, default=1)
    parser.add_argument("--maximum-components", type=int, default=2)
    parser.add_argument("--raw-threshold", type=float, default=0.05)
    parser.add_argument("--standardized-threshold", type=float)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--noise-id")
    parser.add_argument("--q-window-id")
    parser.add_argument(
        "--calibration-point-count",
        type=int,
        help="pre-mask point count from the dataset observation provenance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    with np.load(args.input, allow_pickle=False) as archive:
        keys = set(archive.files)
        if not {"q", "intensity"}.issubset(keys):
            raise ValueError("input NPZ must contain q and intensity")
        sigma_log = archive["sigma_log"] if "sigma_log" in keys else None
        curve = ObservedCurve(
            curve_id=args.curve_id,
            source_kind=args.source_kind,
            q=archive["q"],
            intensity=archive["intensity"],
            sigma_log=sigma_log,
        )
    calibration: CompatibilityCalibrationArtifact | None = (
        None if args.calibration is None else load_compatibility_calibration(args.calibration)
    )
    policy = CompatibilityPolicy(
        raw_log_rmse_max=args.raw_threshold,
        standardized_log_rmse_max=args.standardized_threshold,
        calibration=calibration,
        calibration_observation=(
            None
            if calibration is None
            else CalibrationObservationProvenance(
                point_count=args.calibration_point_count,
                noise_id=args.noise_id,
                q_window_id=args.q_window_id,
            )
        ),
    )
    config = CompetingSearchConfig(
        seed=args.seed,
        starts_per_branch=args.starts_per_branch,
        max_nfev=args.max_nfev,
        minimum_components=args.minimum_components,
        maximum_components=args.maximum_components,
    )
    payload = run_competing_model_search(
        curve,
        config=config,
        policy=policy,
        progress=lambda value: print(json.dumps(value, allow_nan=False), flush=True),
    )
    write_reference_bank_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "state": payload["state"],
                "scientific_payload_sha256": payload["scientific_payload_sha256"],
                "primary_reference_mode_count": len(payload["primary_reference_modes"]),
                "strict_minimal_secondary_mode_count": len(
                    payload["strict_minimal_secondary_modes"]
                ),
                "qualification_status": payload["primary_reference_qualification"][
                    "qualification_status"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SEARCH_VERSION",
    "CompetingSearchConfig",
    "main",
    "run_competing_model_search",
]
