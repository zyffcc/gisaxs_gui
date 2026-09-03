"""Paper-auditable discovered-mode reference bank for Posterior V8.

This module owns the *interpretation* of competing-model search results.  It
does not use the simulation label as an acceptance gate: every candidate is
scored against the observed curve, then audited separately for physical gates
and observability.  The primary bank contains every network-free, exact-curve
compatible physical discovery; strict minimal/effective modes are a secondary
subset.  Both remain finite, budget-qualified discoveries, never a claim of
all mathematical solutions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import hmac
import json
from numbers import Integral
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np

from src.gimap.features.fitting.domain.physical_constraints import ConstraintSet

from .branch_catalog import (
    BRANCH_PATTERN_COUNT,
    branch_pattern_id,
    branch_pattern_is_valid,
    decode_branch_pattern,
)
from .compatibility_calibration import (
    CALIBRATION_SCHEMA,
    CALIBRATION_VERSION,
    COMPATIBILITY_STRATUM_FIELDS,
    COMPATIBILITY_STRATUM_VERSION,
    STRATIFICATION_SEMANTICS,
    CompatibilityCalibrationArtifact,
    CompatibilityStratum,
)
from .canonical_branch_catalog import (
    CANONICAL_BRANCH_CATALOG_VERSION,
    canonical_branch_pattern_is_valid,
)
from .contract import (
    NUM_TOPOLOGIES,
    TOPOLOGIES,
    LatentComponentParameters,
    latent_component_to_gui,
    topology_from_id,
)
from .component_observability import (
    COMPONENT_OBSERVABILITY_VERSION,
    DELETION_PROFILE_FAILED,
    OBSERVABILITY_POLICY_VERSION,
    CandidateObservabilityAssessment,
    FeatureObservabilityAssessment,
    ObservabilityPolicy,
    ReducedModelSearchPort,
    assess_candidate_observability,
)
from .evaluation import (
    CLUSTERING_LINKAGE,
    PARAMETER_DISTANCE_SCOPE,
    PARAMETER_NORMALIZATION_VERSION,
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
    CandidateInput,
    ObservedCurve,
    ReferenceMode,
    complete_linkage_groups,
    natural_log_rmse,
    normalized_latent_parameter_distance,
)


REFERENCE_BANK_SCHEMA = "gisaxs.posterior_v8.discovered_mode_reference_bank/v3"
REFERENCE_BANK_VERSION = "posterior_v8_network_free_primary_reference_bank_v3"
REFERENCE_MODE_CLUSTERING_VERSION = CLUSTERING_LINKAGE
BRANCH_ENUMERATION_VERSION = (
    f"posterior_v8_canonical_418_joint_branches_v2;catalog={CANONICAL_BRANCH_CATALOG_VERSION}"
)
LEGACY_BRANCH_ENUMERATION_VERSION = "posterior_v8_all_valid_700_joint_branches_v1"
COMPATIBILITY_SEMANTICS = (
    "exact_gui_forward_curve_compatibility_separate_from_physics_and_visibility"
)
DISCOVERY_CLAIM = "finite_budget_discovered_modes_not_all_mathematical_solutions"
PRIMARY_REFERENCE_SEMANTICS = (
    "network_free_exact_curve_compatible_bounds_and_physics_valid_discovered_modes;"
    " observability_unknown_and_confirmed_redundant_are_included"
)
STRICT_MINIMAL_SECONDARY_SEMANTICS = (
    "secondary_subset_all_declared_terms_confirmed_needed_by_current_observability_contract"
)
PRIMARY_REFERENCE_SEARCH_SOURCE = "network_free_sobol_multistart_exact_refinement"
EVALUATED_NETWORK_PROPOSAL_SOURCE = "evaluated_network_proposal_secondary_or_loo_only"
ALLOWED_SEARCH_SOURCES = (
    PRIMARY_REFERENCE_SEARCH_SOURCE,
    EVALUATED_NETWORK_PROPOSAL_SOURCE,
)
PRIMARY_REFERENCE_GATES = (
    "exact_curve_compatible",
    "bounds_pass",
    "physics_pass",
    "no_physical_violations",
    "network_free_reference_search_source",
)


_SCIENTIFIC_PAYLOAD_FIELDS = (
    "curve",
    "search_space",
    "primary_reference_semantics",
    "strict_minimal_secondary_semantics",
    "primary_reference_source_policy",
    "primary_reference_gates",
    "compatibility_policy",
    "primary_compatibility_gate",
    "branch_enumeration_version",
    "parameter_mode_clustering",
    "curve_equivalence_clustering",
    "primary_search_budget",
    "post_search_observability_budget",
    "candidates",
    "primary_reference_modes",
    "primary_reference_curve_equivalence_groups",
    "strict_minimal_secondary_modes",
    "strict_minimal_secondary_curve_equivalence_groups",
    "primary_reference_inventory",
    "observability_summary",
    "primary_reference_qualification",
    "primary_mode_discovery_saturation",
    "generating_diagnostic",
)


def reference_bank_scientific_sha256(payload: Mapping[str, object]) -> str:
    """Hash the scientific result fields independently of runtime timing."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    missing = [name for name in _SCIENTIFIC_PAYLOAD_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"reference-bank payload is missing scientific fields: {missing}")
    scientific = {name: payload[name] for name in _SCIENTIFIC_PAYLOAD_FIELDS}
    encoded = json.dumps(
        scientific,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_nonnegative(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True, order=True, kw_only=True)
class CompetingBranch:
    """One canonical topology × D-pattern × Resolution hard branch."""

    topology_id: int
    pattern_id: int

    def __post_init__(self) -> None:
        if isinstance(self.topology_id, bool) or not isinstance(self.topology_id, Integral):
            raise TypeError("topology_id must be an integer")
        if isinstance(self.pattern_id, bool) or not isinstance(self.pattern_id, Integral):
            raise TypeError("pattern_id must be an integer")
        topology_id, pattern_id = int(self.topology_id), int(self.pattern_id)
        if not 0 <= topology_id < NUM_TOPOLOGIES:
            raise ValueError("topology_id must be in [0, 33]")
        if not branch_pattern_is_valid(topology_id, pattern_id):
            raise ValueError("pattern_id is invalid for this topology")
        object.__setattr__(self, "topology_id", topology_id)
        object.__setattr__(self, "pattern_id", pattern_id)

    @property
    def topology(self) -> tuple[str, ...]:
        return topology_from_id(self.topology_id)

    @property
    def d_present(self) -> tuple[bool, ...]:
        return decode_branch_pattern(self.pattern_id)[0][: len(self.topology)]

    @property
    def resolution_present(self) -> bool:
        return decode_branch_pattern(self.pattern_id)[1]

    @property
    def key(self) -> str:
        return f"topology_{self.topology_id:02d}:branch_{self.pattern_id:02d}"

    def to_payload(self) -> dict[str, object]:
        return {
            "key": self.key,
            "topology_id": self.topology_id,
            "topology": list(self.topology),
            "component_count": len(self.topology),
            "pattern_id": self.pattern_id,
            "d_present": list(self.d_present),
            "resolution_present": self.resolution_present,
        }


def _enumerate_branches(
    *,
    topology_ids: Sequence[int] | None = None,
    minimum_components: int = 1,
    maximum_components: int = 4,
    canonical_only: bool,
) -> tuple[CompetingBranch, ...]:
    minimum = _positive_int(minimum_components, "minimum_components")
    maximum = _positive_int(maximum_components, "maximum_components")
    if minimum > maximum or maximum > 4:
        raise ValueError("component limits must satisfy 1 <= minimum <= maximum <= 4")
    if topology_ids is None:
        selected = tuple(range(NUM_TOPOLOGIES))
    else:
        raw = tuple(topology_ids)
        if not raw:
            raise ValueError("topology_ids must not be empty")
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in raw):
            raise TypeError("topology_ids must contain integers")
        selected = tuple(sorted(int(value) for value in raw))
        if len(set(selected)) != len(selected):
            raise ValueError("topology_ids must be unique")
        if selected[0] < 0 or selected[-1] >= NUM_TOPOLOGIES:
            raise ValueError("topology_ids must lie in [0, 33]")
    return tuple(
        CompetingBranch(topology_id=topology_id, pattern_id=pattern_id)
        for topology_id in selected
        if minimum <= len(TOPOLOGIES[topology_id]) <= maximum
        for pattern_id in range(BRANCH_PATTERN_COUNT)
        if branch_pattern_is_valid(topology_id, pattern_id)
        and (not canonical_only or canonical_branch_pattern_is_valid(topology_id, pattern_id))
    )


def enumerate_competing_branches(
    *,
    topology_ids: Sequence[int] | None = None,
    minimum_components: int = 1,
    maximum_components: int = 4,
) -> tuple[CompetingBranch, ...]:
    """Enumerate the 418 permutation-quotiented V3/V4 physical branches."""

    return _enumerate_branches(
        topology_ids=topology_ids,
        minimum_components=minimum_components,
        maximum_components=maximum_components,
        canonical_only=True,
    )


def enumerate_legacy_competing_branches(
    *,
    topology_ids: Sequence[int] | None = None,
    minimum_components: int = 1,
    maximum_components: int = 4,
) -> tuple[CompetingBranch, ...]:
    """Enumerate the legacy 700 wire branches for reproducibility baselines."""

    return _enumerate_branches(
        topology_ids=topology_ids,
        minimum_components=minimum_components,
        maximum_components=maximum_components,
        canonical_only=False,
    )


def branch_for_reference(reference: ReferenceMode) -> CompetingBranch:
    if not isinstance(reference, ReferenceMode):
        raise TypeError("reference must be a ReferenceMode")
    flags = tuple(component.log_D is not None for component in reference.components)
    padded = flags + (False,) * (4 - len(flags))
    pattern_id = branch_pattern_id(padded, reference.resolution is not None)
    return CompetingBranch(topology_id=reference.topology_id, pattern_id=pattern_id)


@dataclass(frozen=True, kw_only=True)
class CalibrationObservationProvenance:
    """Pre-mask observation stratum identity supplied by the dataset manifest."""

    point_count: int
    noise_id: str
    q_window_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_count", _positive_int(self.point_count, "point_count"))
        for name in ("noise_id", "q_window_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value.strip())


@dataclass(frozen=True, kw_only=True)
class CompatibilityPolicy:
    """Frozen gates used by one reference-bank search.

    Noisy observations require either a fixed engineering threshold or a
    split-conformal artifact. As a deliberate cross-K comparison policy, the
    primary threshold has marginal acquisition-conditional coverage and is
    independent of both the generating label and a candidate's inferred K.
    """

    raw_log_rmse_max: float
    standardized_log_rmse_max: float | None = None
    calibration: CompatibilityCalibrationArtifact | None = None
    calibration_observation: CalibrationObservationProvenance | None = None
    observability: ObservabilityPolicy = ObservabilityPolicy()
    parameter_mode_distance_max: float = 0.03
    curve_equivalence_log_rmse_max: float = 0.01
    generating_mode_distance_max: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "raw_log_rmse_max",
            "parameter_mode_distance_max",
            "curve_equivalence_log_rmse_max",
            "generating_mode_distance_max",
        ):
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))
        if self.standardized_log_rmse_max is not None:
            object.__setattr__(
                self,
                "standardized_log_rmse_max",
                _finite_nonnegative(self.standardized_log_rmse_max, "standardized_log_rmse_max"),
            )
        if self.calibration is not None:
            if not isinstance(self.calibration, CompatibilityCalibrationArtifact):
                raise TypeError("calibration must be a CompatibilityCalibrationArtifact")
            if self.standardized_log_rmse_max is not None:
                raise ValueError("choose calibrated or fixed standardized threshold, not both")
            if not isinstance(self.calibration_observation, CalibrationObservationProvenance):
                raise ValueError(
                    "calibrated scoring requires explicit pre-mask observation provenance"
                )
        elif self.calibration_observation is not None:
            raise ValueError("calibration_observation requires a calibration artifact")
        if not isinstance(self.observability, ObservabilityPolicy):
            raise TypeError("observability must be an ObservabilityPolicy")

    def threshold_for(self, curve: ObservedCurve) -> tuple[float, str, bool]:
        if curve.sigma_log is None:
            return self.raw_log_rmse_max, "fixed_raw", False
        if self.calibration is None:
            if self.standardized_log_rmse_max is None:
                raise ValueError("noisy scoring requires standardized_log_rmse_max or calibration")
            return self.standardized_log_rmse_max, "fixed_standardized", False
        observation = self.calibration_observation
        assert observation is not None
        stratum = CompatibilityStratum(
            point_count=observation.point_count,
            noise_id=observation.noise_id,
            q_window_id=observation.q_window_id,
        )
        # Formal reference-bank acceptance never uses the artifact's pooled
        # descriptive fallback because it carries no conditional coverage claim.
        calibrated, fallback = self.calibration.threshold_for(stratum)
        if fallback:  # pragma: no cover - fail-closed artifact API invariant
            raise RuntimeError("formal compatibility unexpectedly received a fallback")
        return calibrated.threshold, "calibrated_stratum", False


@dataclass(frozen=True, kw_only=True)
class CandidateDiscovery:
    candidate: CandidateInput
    branch: CompetingBranch
    attempt_rank: int
    search_source: str
    raw_log_rmse: float
    standardized_log_rmse: float | None
    compatibility_metric: str
    primary_score: float
    compatibility_threshold: float
    threshold_source: str
    used_calibration_fallback: bool
    curve_compatible: bool
    physical_violations: tuple[str, ...]
    effective_component_indices: tuple[int, ...]
    effective_topology: tuple[str, ...]
    effective_resolution_present: bool
    declared_components_observable: bool
    primary_reference_eligible: bool
    primary_reference_qualification_reason: str
    strict_minimal_secondary_eligible: bool
    strict_minimal_secondary_qualification_reason: str
    observability: CandidateObservabilityAssessment

    @property
    def reference_eligible(self) -> bool:
        """Deprecated read-only alias for the former strict-minimal meaning."""

        return self.strict_minimal_secondary_eligible

    def to_payload(self) -> dict[str, object]:
        candidate = self.candidate
        linear = candidate.linear_solution
        return {
            "candidate_id": candidate.candidate_id,
            "proposal_rank": candidate.proposal_rank,
            "attempt_rank": self.attempt_rank,
            "search_source": self.search_source,
            "branch": self.branch.to_payload(),
            "latent_components": [asdict(value) for value in candidate.components],
            "gui_components": [
                asdict(latent_component_to_gui(value)) for value in candidate.components
            ],
            "resolution": None if candidate.resolution is None else asdict(candidate.resolution),
            "linear_solution": {
                "background": linear.background,
                "particle_amplitudes": list(linear.particle_amplitudes),
                "resolution_amplitude": linear.resolution_amplitude,
                "k": linear.k,
                "component_weights": list(linear.component_weights),
                "int_res": linear.int_res,
            },
            "raw_log_rmse": self.raw_log_rmse,
            "standardized_log_rmse": self.standardized_log_rmse,
            "compatibility_metric": self.compatibility_metric,
            "primary_score": self.primary_score,
            "compatibility_threshold": self.compatibility_threshold,
            "threshold_source": self.threshold_source,
            "used_calibration_fallback": self.used_calibration_fallback,
            "curve_compatible": self.curve_compatible,
            "bounds_pass": candidate.bounds_pass,
            "physics_pass": candidate.physics_pass,
            "physical_violations": list(self.physical_violations),
            "effective_component_indices": list(self.effective_component_indices),
            "effective_component_semantics": "confirmed_needed_only",
            "effective_topology": list(self.effective_topology),
            "effective_resolution_present": self.effective_resolution_present,
            "declared_components_observable": self.declared_components_observable,
            "primary_reference_eligible": self.primary_reference_eligible,
            "primary_reference_qualification_status": (
                "ELIGIBLE" if self.primary_reference_eligible else "INELIGIBLE"
            ),
            "primary_reference_qualification_reason": (self.primary_reference_qualification_reason),
            "strict_minimal_secondary_eligible": self.strict_minimal_secondary_eligible,
            "strict_minimal_secondary_qualification_status": (
                "ELIGIBLE" if self.strict_minimal_secondary_eligible else "INELIGIBLE"
            ),
            "strict_minimal_secondary_qualification_reason": (
                self.strict_minimal_secondary_qualification_reason
            ),
            "observability": self.observability.to_payload(),
            "exact_intensity_sha256": hashlib.sha256(
                np.asarray(candidate.exact_intensity, dtype="<f8").tobytes()
            ).hexdigest(),
        }


def _constraint_violations(components: Sequence[LatentComponentParameters]) -> tuple[str, ...]:
    payload = []
    for latent in components:
        value = latent_component_to_gui(latent)
        payload.append(
            {
                "type": value.shape,
                "params": {
                    "R": value.R,
                    "sigma_R": value.sigma_R,
                    "h": value.h,
                    "sigma_h": value.sigma_h,
                    "D": 0.0 if value.D is None else value.D,
                    "sigma_D": 0.0 if value.sigma_D is None else value.sigma_D,
                },
            }
        )
    return tuple(
        f"{value.constraint_id}:{value.component_index}"
        for value in ConstraintSet.defaults().validate_components(payload)
    )


def _unknown_observability_after_error(
    candidate: CandidateInput,
    *,
    metric_name: str,
    score: float,
    threshold: float,
    policy: ObservabilityPolicy,
    error: Exception,
) -> CandidateObservabilityAssessment:
    """Keep primary compatibility evidence when optional observability fails."""

    def unknown_feature(label: str, *, present: bool) -> FeatureObservabilityAssessment:
        return FeatureObservabilityAssessment(
            label=label,
            decision="unknown" if present else "not_present",
            deletion_status=DELETION_PROFILE_FAILED if present else "not_present",
            reduced_primary_score=None,
            primary_compatibility_threshold=threshold,
            evidence_scope="post_search_observability_failed" if present else "not_present",
        )

    return CandidateObservabilityAssessment(
        candidate_id=candidate.candidate_id,
        status="provisional_or_unknown",
        primary_metric_name=metric_name,
        primary_compatibility_threshold=threshold,
        full_primary_score=score,
        full_curve_compatible=score <= threshold,
        particles=tuple(
            unknown_feature(f"particle_{index}", present=True)
            for index in range(len(candidate.components))
        ),
        resolution=unknown_feature("resolution", present=candidate.resolution is not None),
        d_terms=tuple(
            unknown_feature(f"d_{index}", present=component.log_D is not None)
            for index, component in enumerate(candidate.components)
        ),
        exact_forward_calls=0,
        exact_forward_call_limit=policy.exact_forward_call_limit,
        budget_exhausted=False,
        diagnostic_error=f"{type(error).__name__}: {error}",
        policy_version=policy.version,
        diagnostic_version=COMPONENT_OBSERVABILITY_VERSION,
    )


def assess_candidate(
    curve: ObservedCurve,
    candidate: CandidateInput,
    branch: CompetingBranch,
    *,
    attempt_rank: int,
    search_source: str,
    policy: CompatibilityPolicy,
    reduced_model_searcher: ReducedModelSearchPort | None = None,
) -> CandidateDiscovery:
    """Score one exact-forward candidate without consulting a generating label."""

    if candidate.topology_id != branch.topology_id:
        raise ValueError("candidate topology does not match its competing branch")
    if search_source not in ALLOWED_SEARCH_SOURCES:
        raise ValueError(f"search_source must be one of {ALLOWED_SEARCH_SOURCES}")
    if (
        branch_for_reference(
            ReferenceMode(
                reference_id="candidate_branch_check",
                topology_id=candidate.topology_id,
                components=candidate.components,
                resolution=candidate.resolution,
                linear_solution=candidate.linear_solution,
            )
        )
        != branch
    ):
        raise ValueError("candidate D/Resolution presence does not match its branch")
    rank = _positive_int(attempt_rank, "attempt_rank")
    raw = natural_log_rmse(candidate.exact_intensity, curve.intensity)
    standardized = (
        None
        if curve.sigma_log is None
        else natural_log_rmse(candidate.exact_intensity, curve.intensity, sigma_log=curve.sigma_log)
    )
    score = raw if standardized is None else standardized
    threshold, source, fallback = policy.threshold_for(curve)
    compatible = score <= threshold
    violations = _constraint_violations(candidate.components)
    metric_name = RAW_LOG_RMSE_METRIC if standardized is None else STANDARDIZED_LOG_RMSE_METRIC
    try:
        observability = assess_candidate_observability(
            curve,
            candidate,
            primary_metric_name=metric_name,
            primary_compatibility_threshold=threshold,
            policy=policy.observability,
            reduced_model_searcher=reduced_model_searcher,
        )
    except (FloatingPointError, RuntimeError, TypeError, ValueError) as exc:
        observability = _unknown_observability_after_error(
            candidate,
            metric_name=metric_name,
            score=score,
            threshold=threshold,
            policy=policy.observability,
            error=exc,
        )
    effective_indices = tuple(
        index
        for index, assessment in enumerate(observability.particles)
        if assessment.decision == "needed"
    )
    effective_topology = tuple(candidate.components[index].shape for index in effective_indices)
    effective_resolution = observability.resolution.decision == "needed"
    declared_observable = observability.confirmed_effective
    scientific_failures = []
    if not compatible:
        scientific_failures.append("curve_incompatible")
    if not candidate.bounds_pass:
        scientific_failures.append("bounds_failed")
    if not candidate.physics_pass:
        scientific_failures.append("physics_flag_failed")
    if violations:
        scientific_failures.append("physical_constraint_violations")
    primary_failures = list(scientific_failures)
    if search_source != PRIMARY_REFERENCE_SEARCH_SOURCE:
        primary_failures.append("non_primary_or_network_dependent_search_source")
    strict_failures = list(scientific_failures)
    if not declared_observable:
        strict_failures.append("declared_terms_not_all_confirmed_needed")
    primary_eligible = not primary_failures
    strict_eligible = not strict_failures
    return CandidateDiscovery(
        candidate=candidate,
        branch=branch,
        attempt_rank=rank,
        search_source=search_source,
        raw_log_rmse=raw,
        standardized_log_rmse=standardized,
        compatibility_metric=metric_name,
        primary_score=score,
        compatibility_threshold=threshold,
        threshold_source=source,
        used_calibration_fallback=fallback,
        curve_compatible=compatible,
        physical_violations=violations,
        effective_component_indices=effective_indices,
        effective_topology=effective_topology,
        effective_resolution_present=effective_resolution,
        declared_components_observable=declared_observable,
        primary_reference_eligible=primary_eligible,
        primary_reference_qualification_reason=(
            "all_primary_reference_gates_passed" if primary_eligible else ";".join(primary_failures)
        ),
        strict_minimal_secondary_eligible=strict_eligible,
        strict_minimal_secondary_qualification_reason=(
            "all_strict_minimal_secondary_gates_passed"
            if strict_eligible
            else ";".join(strict_failures)
        ),
        observability=observability,
    )


_OBSERVABILITY_STATUSES = (
    "confirmed_effective",
    "confirmed_redundant",
    "provisional_or_unknown",
)


def _observability_counts(
    discoveries: Sequence[CandidateDiscovery],
) -> dict[str, int]:
    values = tuple(discoveries)
    return {
        status: sum(value.observability.status == status for value in values)
        for status in _OBSERVABILITY_STATUSES
    }


def _group_discoveries(
    discoveries: Sequence[CandidateDiscovery],
    policy: CompatibilityPolicy,
    *,
    include,
    mode_prefix: str,
    curve_group_prefix: str,
    interpretation: str,
    reference_set_role: str,
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    eligible = sorted(
        (value for value in discoveries if include(value)),
        key=lambda value: value.candidate.proposal_rank,
    )
    # Hard branches have infinite cross-branch parameter distance.  Cluster
    # each branch separately so scipy receives finite condensed matrices and
    # a full 418-branch search remains quadratic instead of cubic.
    parameter_groups = []
    for branch_key in sorted({value.branch.key for value in eligible}):
        branch_values = [value for value in eligible if value.branch.key == branch_key]
        parameter_groups.extend(
            complete_linkage_groups(
                branch_values,
                lambda left, right: normalized_latent_parameter_distance(
                    left.candidate, right.candidate
                ),
                policy.parameter_mode_distance_max,
            )
        )
    parameter_groups.sort(key=lambda group: min(value.candidate.proposal_rank for value in group))
    modes = []
    for number, group in enumerate(parameter_groups, 1):
        representative = min(
            group, key=lambda value: (value.primary_score, value.candidate.proposal_rank)
        )
        distances = [
            normalized_latent_parameter_distance(left.candidate, right.candidate)
            for index, left in enumerate(group)
            for right in group[index + 1 :]
        ]
        observability_by_candidate = {
            value.candidate.candidate_id: value.observability.status for value in group
        }
        modes.append(
            {
                "mode_id": f"{mode_prefix}_{number:03d}",
                "branch_key": representative.branch.key,
                "topology_id": representative.branch.topology_id,
                "pattern_id": representative.branch.pattern_id,
                "member_candidate_ids": [value.candidate.candidate_id for value in group],
                "representative_candidate_id": representative.candidate.candidate_id,
                "best_primary_score": representative.primary_score,
                "pairwise_parameter_diameter": max(distances, default=0.0),
                "reference_set_role": reference_set_role,
                "all_members_eligible_for_this_set": True,
                "qualification_status": "ELIGIBLE",
                "qualification_reason": interpretation,
                "observability_status_counts": _observability_counts(group),
                "observability_by_candidate_id": observability_by_candidate,
                "strict_minimal_secondary_member_candidate_ids": [
                    value.candidate.candidate_id
                    for value in group
                    if value.strict_minimal_secondary_eligible
                ],
                "interpretation": interpretation,
            }
        )
    candidate_by_id = {value.candidate.candidate_id: value for value in eligible}
    curve_groups_raw = complete_linkage_groups(
        modes,
        lambda left, right: natural_log_rmse(
            candidate_by_id[left["representative_candidate_id"]].candidate.exact_intensity,
            candidate_by_id[right["representative_candidate_id"]].candidate.exact_intensity,
        ),
        policy.curve_equivalence_log_rmse_max,
    )
    curve_groups = []
    for number, group in enumerate(curve_groups_raw, 1):
        representative = min(group, key=lambda value: value["best_primary_score"])
        member_candidate_ids = [
            candidate_id for mode in group for candidate_id in mode["member_candidate_ids"]
        ]
        member_discoveries = [
            candidate_by_id[candidate_id] for candidate_id in member_candidate_ids
        ]
        curve_groups.append(
            {
                "curve_group_id": f"{curve_group_prefix}_{number:03d}",
                "mode_ids": [value["mode_id"] for value in group],
                "member_candidate_ids": member_candidate_ids,
                "representative_candidate_id": representative["representative_candidate_id"],
                "best_primary_score": representative["best_primary_score"],
                "reference_set_role": reference_set_role,
                "qualification_status": "ELIGIBLE",
                "qualification_reason": interpretation,
                "observability_status_counts": _observability_counts(member_discoveries),
            }
        )
    return tuple(modes), tuple(curve_groups)


def primary_reference_groups(
    discoveries: Sequence[CandidateDiscovery], policy: CompatibilityPolicy
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Cluster every network-free exact-compatible physical discovery."""

    return _group_discoveries(
        discoveries,
        policy,
        include=lambda value: value.primary_reference_eligible,
        mode_prefix="primary_reference_mode",
        curve_group_prefix="primary_reference_curve_group",
        interpretation="primary_network_free_compatible_mode_observability_is_descriptive",
        reference_set_role="primary_reference_denominator",
    )


def strict_minimal_secondary_groups(
    discoveries: Sequence[CandidateDiscovery], policy: CompatibilityPolicy
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Cluster the secondary subset whose declared terms are confirmed needed."""

    return _group_discoveries(
        discoveries,
        policy,
        include=lambda value: value.strict_minimal_secondary_eligible,
        mode_prefix="strict_minimal_secondary_mode",
        curve_group_prefix="strict_minimal_secondary_curve_group",
        interpretation="secondary_strict_minimal_effective_subset_not_primary_denominator",
        reference_set_role="strict_minimal_secondary",
    )


def discovered_groups(
    discoveries: Sequence[CandidateDiscovery], policy: CompatibilityPolicy
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Deprecated read-only alias for :func:`primary_reference_groups`."""

    return primary_reference_groups(discoveries, policy)


def compatible_discovered_groups(
    discoveries: Sequence[CandidateDiscovery], policy: CompatibilityPolicy
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    """Deprecated read-only alias for :func:`primary_reference_groups`."""

    return primary_reference_groups(discoveries, policy)


def generating_diagnostic(
    discoveries: Sequence[CandidateDiscovery],
    primary_modes: Sequence[Mapping[str, object]],
    strict_minimal_secondary_modes: Sequence[Mapping[str, object]],
    generating_mode: ReferenceMode | None,
    *,
    distance_max: float,
) -> dict[str, object] | None:
    """Separate generating-branch discovery from generating-parameter recovery."""

    if generating_mode is None:
        return None
    branch = branch_for_reference(generating_mode)
    same_branch_compatible = [
        value for value in discoveries if value.branch == branch and value.curve_compatible
    ]
    same_branch_primary = [
        value
        for value in discoveries
        if value.branch == branch and value.primary_reference_eligible
    ]
    same_branch_strict = [
        value
        for value in discoveries
        if value.branch == branch and value.strict_minimal_secondary_eligible
    ]
    primary = [value for value in discoveries if value.primary_reference_eligible]
    distances = [
        (normalized_latent_parameter_distance(generating_mode, value.candidate), value)
        for value in primary
    ]
    finite = [value for value in distances if np.isfinite(value[0])]
    nearest = min(
        finite,
        key=lambda value: (value[0], value[1].candidate.proposal_rank),
        default=None,
    )
    alternative_branches = sorted({value.branch.key for value in primary if value.branch != branch})
    candidate_to_mode = {
        str(candidate_id): str(mode["mode_id"])
        for mode in primary_modes
        for candidate_id in mode["member_candidate_ids"]
    }
    recovered = nearest is not None and nearest[0] <= float(distance_max)
    generating_mode_id = (
        None if not recovered else candidate_to_mode.get(nearest[1].candidate.candidate_id)
    )
    alternative_modes = sorted(
        str(mode["mode_id"]) for mode in primary_modes if mode["branch_key"] != branch.key
    )
    same_branch_other_modes = sorted(
        str(mode["mode_id"])
        for mode in primary_modes
        if mode["branch_key"] == branch.key and mode["mode_id"] != generating_mode_id
    )
    strict_mode_ids = sorted(
        str(mode["mode_id"])
        for mode in strict_minimal_secondary_modes
        if mode["branch_key"] == branch.key
    )
    if alternative_modes:
        label = "COMPETING_HARD_BRANCH_AMBIGUITY_DISCOVERED"
    elif same_branch_other_modes:
        label = "WITHIN_BRANCH_PARAMETER_MULTIMODALITY_DISCOVERED"
    elif recovered:
        label = "ONLY_GENERATING_MODE_DISCOVERED_WITHIN_BUDGET"
    elif same_branch_primary:
        label = "GENERATING_BRANCH_COMPATIBLE_BUT_GENERATING_MODE_NOT_RECOVERED"
    else:
        label = "GENERATING_BRANCH_NOT_PRIMARY_REFERENCE_ELIGIBLE_WITHIN_BUDGET"
    return {
        "generating_reference_id": generating_mode.reference_id,
        "generating_branch_key": branch.key,
        "generating_branch_curve_compatible": bool(same_branch_compatible),
        "generating_branch_primary_reference_eligible": bool(same_branch_primary),
        "generating_branch_strict_minimal_secondary_eligible": bool(same_branch_strict),
        "generating_mode_recovered": recovered,
        "generating_reference_mode_id": generating_mode_id,
        "generating_mode_distance_max": float(distance_max),
        "nearest_candidate_id": (None if nearest is None else nearest[1].candidate.candidate_id),
        "nearest_parameter_distance": None if nearest is None else float(nearest[0]),
        "alternative_primary_reference_branch_keys": alternative_branches,
        "alternative_primary_reference_mode_ids": alternative_modes,
        "generating_branch_strict_minimal_secondary_mode_ids": strict_mode_ids,
        "same_branch_alternative_parameter_mode_ids": same_branch_other_modes,
        "competing_model_ambiguity_discovered": bool(alternative_branches),
        "within_branch_parameter_multimodality_discovered": bool(same_branch_other_modes),
        "budget_qualified_label": label,
        "label_is_not_mathematical_identifiability_proof": True,
    }


def _validate_reference_bank_v3_payload(payload: Mapping[str, object]) -> None:
    if payload.get("reference_bank_version") != REFERENCE_BANK_VERSION:
        raise ValueError("payload has an incompatible reference-bank version")
    if payload.get("branch_enumeration_version") != BRANCH_ENUMERATION_VERSION:
        raise ValueError("payload does not use the canonical 418-branch enumeration")
    legacy_top_level = {
        "reference_mode_clustering_version",
        "reference_modes",
        "curve_equivalence_groups",
        "compatible_parameter_modes",
        "compatible_curve_equivalence_groups",
        "reference_eligible_topology_ids",
        "reference_eligible_branch_keys",
        "budget_qualified_interpretation",
        "mode_discovery_saturation",
        "search_budget",
        "attempts",
    } & set(payload)
    if legacy_top_level:
        raise ValueError(f"v3 payload mixes legacy reference semantics: {legacy_top_level}")
    if payload.get("primary_reference_semantics") != PRIMARY_REFERENCE_SEMANTICS:
        raise ValueError("payload has incompatible primary-reference semantics")
    if payload.get("strict_minimal_secondary_semantics") != STRICT_MINIMAL_SECONDARY_SEMANTICS:
        raise ValueError("payload has incompatible strict-minimal secondary semantics")
    if payload.get("primary_reference_gates") != list(PRIMARY_REFERENCE_GATES):
        raise ValueError("payload has incompatible primary-reference gates")
    source_policy = payload.get("primary_reference_source_policy")
    if not isinstance(source_policy, Mapping) or not (
        source_policy.get("required_search_source") == PRIMARY_REFERENCE_SEARCH_SOURCE
        and source_policy.get("denominator_is_network_free") is True
        and source_policy.get("evaluated_network_proposals_allowed_in_primary_denominator") is False
        and source_policy.get("network_proposal_union_policy")
        == "secondary_or_leave_one_out_sensitivity_only"
    ):
        raise ValueError("payload primary denominator is not explicitly network-free")
    parameter_clustering = payload.get("parameter_mode_clustering")
    if not isinstance(parameter_clustering, Mapping) or not (
        parameter_clustering.get("distance_metric_version") == PARAMETER_NORMALIZATION_VERSION
        and parameter_clustering.get("distance_scope") == PARAMETER_DISTANCE_SCOPE
        and parameter_clustering.get("linkage") == CLUSTERING_LINKAGE
    ):
        raise ValueError("payload has an incompatible parameter-mode clustering contract")
    curve_clustering = payload.get("curve_equivalence_clustering")
    if not isinstance(curve_clustering, Mapping) or not (
        curve_clustering.get("distance_metric") == RAW_LOG_RMSE_METRIC
        and curve_clustering.get("linkage") == CLUSTERING_LINKAGE
    ):
        raise ValueError("payload has an incompatible curve-equivalence clustering contract")
    policy = payload.get("compatibility_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("payload requires a compatibility_policy mapping")
    legacy = {"minimum_particle_weight", "minimum_resolution_ratio"} & set(policy)
    if legacy:
        raise ValueError(f"legacy coefficient observability policy is unsupported: {legacy}")
    calibration_digest = policy.get("calibration_input_sha256")
    if calibration_digest is not None:
        expected_calibration_contract = {
            "calibration_schema": CALIBRATION_SCHEMA,
            "calibration_version": CALIBRATION_VERSION,
            "calibration_stratum_version": COMPATIBILITY_STRATUM_VERSION,
            "calibration_stratum_fields": list(COMPATIBILITY_STRATUM_FIELDS),
            "calibration_stratification_semantics": STRATIFICATION_SEMANTICS,
        }
        if any(policy.get(key) != value for key, value in expected_calibration_contract.items()):
            raise ValueError(
                "payload does not use the observable-acquisition compatibility calibration"
            )
        calibration_observation = policy.get("calibration_observation")
        if not isinstance(calibration_observation, Mapping) or set(calibration_observation) != set(
            COMPATIBILITY_STRATUM_FIELDS
        ):
            raise ValueError(
                "calibration observation must contain only pre-fit acquisition metadata"
            )
    observability_policy = policy.get("observability")
    if (
        not isinstance(observability_policy, Mapping)
        or observability_policy.get("version") != OBSERVABILITY_POLICY_VERSION
    ):
        raise ValueError("payload requires the current observability policy")
    compatibility_gate = payload.get("primary_compatibility_gate")
    if not isinstance(compatibility_gate, Mapping) or not (
        compatibility_gate.get("metric") in {RAW_LOG_RMSE_METRIC, STANDARDIZED_LOG_RMSE_METRIC}
        and compatibility_gate.get("comparison") == "less_than_or_equal"
        and compatibility_gate.get("exact_gui_forward_required") is True
        and compatibility_gate.get("used_descriptive_fallback") is False
    ):
        raise ValueError("payload has an invalid primary compatibility gate")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("payload candidates must be a list")
    candidate_by_id: dict[str, Mapping[str, object]] = {}
    for value in candidates:
        if not isinstance(value, Mapping):
            raise ValueError("payload candidate audit must be a mapping")
        if "reference_eligible" in value:
            raise ValueError("v3 candidate mixes legacy reference_eligible semantics")
        candidate_id = value.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in candidate_by_id:
            raise ValueError("payload candidate IDs must be unique strings")
        observability = value.get("observability")
        if not isinstance(observability, Mapping):
            raise ValueError("every candidate requires a full observability audit")
        if observability.get("policy_version") != OBSERVABILITY_POLICY_VERSION:
            raise ValueError("candidate observability policy version is unsupported")
        if observability.get("diagnostic_version") != COMPONENT_OBSERVABILITY_VERSION:
            raise ValueError("candidate observability diagnostic version is unsupported")
        calls = observability.get("exact_forward_calls")
        if isinstance(calls, bool) or not isinstance(calls, Integral) or int(calls) < 0:
            raise ValueError("candidate observability exact-forward ledger is invalid")
        scientific_gates_pass = (
            value.get("curve_compatible") is True
            and value.get("bounds_pass") is True
            and value.get("physics_pass") is True
            and value.get("physical_violations") == []
        )
        primary = value.get("primary_reference_eligible") is True
        strict = value.get("strict_minimal_secondary_eligible") is True
        if value.get("primary_reference_qualification_status") != (
            "ELIGIBLE" if primary else "INELIGIBLE"
        ):
            raise ValueError("candidate primary qualification status is inconsistent")
        if value.get("strict_minimal_secondary_qualification_status") != (
            "ELIGIBLE" if strict else "INELIGIBLE"
        ):
            raise ValueError("candidate strict-secondary qualification status is inconsistent")
        if primary and not (
            scientific_gates_pass and value.get("search_source") == PRIMARY_REFERENCE_SEARCH_SOURCE
        ):
            raise ValueError("primary-reference candidate fails a scientific or source gate")
        if strict and not (
            scientific_gates_pass and observability.get("status") == "confirmed_effective"
        ):
            raise ValueError("strict-minimal secondary candidate lacks confirmed evidence")
        if value.get("compatibility_metric") != compatibility_gate.get("metric"):
            raise ValueError("candidate compatibility metric is inconsistent")
        if value.get("compatibility_threshold") != compatibility_gate.get("threshold"):
            raise ValueError("candidate compatibility threshold is inconsistent")
        if value.get("threshold_source") != compatibility_gate.get("threshold_source"):
            raise ValueError("candidate threshold source is inconsistent")
        candidate_by_id[candidate_id] = value

    def validate_modes(
        field: str, *, role: str, eligibility_field: str
    ) -> list[Mapping[str, object]]:
        modes = payload.get(field)
        if not isinstance(modes, list):
            raise ValueError(f"payload {field} must be a list")
        for mode in modes:
            if not isinstance(mode, Mapping) or "paper_reference_eligible" in mode:
                raise ValueError(f"{field} mixes legacy paper-reference semantics")
            if mode.get("reference_set_role") != role:
                raise ValueError(f"{field} has an incompatible reference-set role")
            member_ids = mode.get("member_candidate_ids")
            if not isinstance(member_ids, list) or not member_ids:
                raise ValueError(f"{field} mode requires candidate members")
            if any(
                candidate_id not in candidate_by_id
                or candidate_by_id[candidate_id].get(eligibility_field) is not True
                for candidate_id in member_ids
            ):
                raise ValueError(f"{field} contains a non-eligible candidate")
        return modes

    primary_modes = validate_modes(
        "primary_reference_modes",
        role="primary_reference_denominator",
        eligibility_field="primary_reference_eligible",
    )
    strict_modes = validate_modes(
        "strict_minimal_secondary_modes",
        role="strict_minimal_secondary",
        eligibility_field="strict_minimal_secondary_eligible",
    )
    inventory = payload.get("primary_reference_inventory")
    if not isinstance(inventory, Mapping):
        raise ValueError("payload requires a primary-reference inventory")
    expected_primary_ids = [
        candidate_id
        for candidate_id, value in candidate_by_id.items()
        if value.get("primary_reference_eligible") is True
    ]
    expected_strict_ids = [
        candidate_id
        for candidate_id, value in candidate_by_id.items()
        if value.get("strict_minimal_secondary_eligible") is True
    ]
    if inventory.get("candidate_ids") != expected_primary_ids:
        raise ValueError("primary-reference candidate inventory is inconsistent")
    if inventory.get("strict_minimal_secondary_candidate_ids") != expected_strict_ids:
        raise ValueError("strict-minimal candidate inventory is inconsistent")
    if inventory.get("mode_ids") != [mode.get("mode_id") for mode in primary_modes]:
        raise ValueError("primary-reference mode inventory is inconsistent")
    if inventory.get("strict_minimal_secondary_mode_ids") != [
        mode.get("mode_id") for mode in strict_modes
    ]:
        raise ValueError("strict-minimal mode inventory is inconsistent")
    search_budget = payload.get("primary_search_budget")
    if not isinstance(search_budget, Mapping) or not (
        search_budget.get("search_source") == PRIMARY_REFERENCE_SEARCH_SOURCE
        and search_budget.get("observability_calls_included") is False
    ):
        raise ValueError("primary-search budget is not independent of observability")
    saturation = payload.get("primary_mode_discovery_saturation")
    if not isinstance(saturation, list) or any(
        not isinstance(value, Mapping)
        or value.get("observability_budget_affects_this_saturation_curve") is not False
        for value in saturation
    ):
        raise ValueError("primary saturation audit is missing or observability-dependent")
    qualification = payload.get("primary_reference_qualification")
    if not isinstance(qualification, Mapping) or not (
        isinstance(qualification.get("qualification_status"), str)
        and isinstance(qualification.get("qualification_reason"), str)
        and qualification.get("completeness_certificate_status") == "NOT_ESTABLISHED"
        and qualification.get("does_not_establish_all_solutions_or_no_solution") is True
    ):
        raise ValueError("primary-reference qualification is not fail-honest")


def write_reference_bank_atomic(path: Path | str, payload: Mapping[str, object]) -> None:
    """Write a completed audit without overwriting prior reference evidence."""

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if payload.get("schema") != REFERENCE_BANK_SCHEMA:
        raise ValueError("payload has an incompatible reference-bank schema")
    if payload.get("state") != "COMPUTATION_COMPLETE":
        raise ValueError("only a COMPUTATION_COMPLETE reference-bank payload can be published")
    _validate_reference_bank_v3_payload(payload)
    digest = payload.get("scientific_payload_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("payload requires a lowercase scientific SHA-256 digest")
    expected_digest = reference_bank_scientific_sha256(payload)
    if not hmac.compare_digest(digest, expected_digest):
        raise ValueError("reference-bank scientific SHA-256 digest does not match payload")
    output = Path(path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite reference bank: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(f"refusing to overwrite reference bank: {output}") from None
        try:
            directory_descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "BRANCH_ENUMERATION_VERSION",
    "COMPATIBILITY_SEMANTICS",
    "DISCOVERY_CLAIM",
    "EVALUATED_NETWORK_PROPOSAL_SOURCE",
    "LEGACY_BRANCH_ENUMERATION_VERSION",
    "PRIMARY_REFERENCE_GATES",
    "PRIMARY_REFERENCE_SEARCH_SOURCE",
    "PRIMARY_REFERENCE_SEMANTICS",
    "REFERENCE_BANK_SCHEMA",
    "REFERENCE_BANK_VERSION",
    "REFERENCE_MODE_CLUSTERING_VERSION",
    "STRICT_MINIMAL_SECONDARY_SEMANTICS",
    "CandidateDiscovery",
    "CalibrationObservationProvenance",
    "CompatibilityPolicy",
    "CompetingBranch",
    "assess_candidate",
    "branch_for_reference",
    "complete_linkage_groups",
    "compatible_discovered_groups",
    "discovered_groups",
    "enumerate_competing_branches",
    "enumerate_legacy_competing_branches",
    "generating_diagnostic",
    "primary_reference_groups",
    "reference_bank_scientific_sha256",
    "strict_minimal_secondary_groups",
    "write_reference_bank_atomic",
]
