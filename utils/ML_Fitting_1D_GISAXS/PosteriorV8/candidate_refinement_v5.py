"""Budgeted V5.1 local-seed refinement with exact GUI-amplitude constraints.

This module is the scientific boundary between proposal generation and the
verified multi-candidate evaluator.  It consumes user-local neural, retrieval,
or Sobol seeds; every successful output is revalidated by the same branch
codec and the exact ``GuiAmplitudeConstraint`` instance carried by the
``V5CandidateContextBatch``.  It does not decide measurement compatibility,
mathematical solvability, or posterior probability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Literal, Sequence

import numpy as np

from src.gimap.features.fitting.domain.physical_constraints import ConstraintSet

from .branch_codec import BRANCH_CODEC_VERSION, UNIT_CUBE_DIMENSIONS, ProfiledBranchCodec
from .candidate_batch_v5 import V5CandidateContextBatch
from .candidate_proposals_v5 import V5LocalProposal
from .candidate_refinement_contract_v5 import (
    EXACT_FORWARD_PHASES,
    V5_EXACT_REFINEMENT_SCHEMA,
    V5_EXACT_REFINEMENT_SCOPE,
    V5_EXACT_REFINEMENT_VERSION,
    V5_EXACT_FORWARD_BUDGET_UNIT,
    V5AttemptStatus,
    V5CandidatePrerequisiteAudit,
    V5ExactBatchStatus,
    V5ExactForwardLedger,
    V5ExactRefinementAttempt,
    V5ExactRefinementBatchResult,
    V5SeedSource,
    _integer,
    exact_forward_phase_counts,
)
from .canonical_component_slots import canonicalize_component_slots
from .contract import (
    LatentComponentParameters,
    latent_component_to_gui,
    topology_from_id,
)
from .evaluation import CandidateInput, LinearSolutionSnapshot, ObservedCurve
from .gui_amplitude_constraints import GuiAmplitudeConstraint
from .profiled_forward import (
    ProfiledForwardResult,
    ResolutionShape,
    evaluate_profiled_forward,
    profile_linear_amplitudes,
)
from .profiled_refinement import refine_profiled_branch


_CANDIDATE_ERRORS = (FloatingPointError, OverflowError, RuntimeError, TypeError, ValueError)


def _local_vector(value: Sequence[float]) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (UNIT_CUBE_DIMENSIONS,):
        raise ValueError(f"local_unit must have shape ({UNIT_CUBE_DIMENSIONS},)")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("local_unit must contain finite values in [0, 1]")
    return tuple(float(item) for item in array)


def _constraint_sha256(value: GuiAmplitudeConstraint) -> str:
    encoded = json.dumps(
        value.to_audit_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5LocalRefinementSeed:
    """Source-neutral physical seed expressed in one V5 user-local codec."""

    source: V5SeedSource
    source_id: str
    query_sha256: str
    topology_id: int
    pattern_id: int
    branch_batch_index: int
    local_unit: tuple[float, ...]
    latent_components: tuple[LatentComponentParameters, ...]
    resolution: ResolutionShape | None
    search_yield_logit: float | None = None
    mixture_log_weight: float | None = None

    def __post_init__(self) -> None:
        if self.source not in {"neural", "retrieval", "sobol"}:
            raise ValueError("source must be neural, retrieval, or sobol")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.query_sha256, str) or not self.query_sha256:
            raise ValueError("query_sha256 must be a non-empty string")
        topology = topology_from_id(self.topology_id)
        components = tuple(self.latent_components)
        if not all(isinstance(item, LatentComponentParameters) for item in components):
            raise TypeError("latent_components must contain LatentComponentParameters")
        if tuple(item.shape for item in components) != topology:
            raise ValueError("latent_components do not match topology_id")
        if self.resolution is not None and not isinstance(self.resolution, ResolutionShape):
            raise TypeError("resolution must be ResolutionShape or None")
        for name in ("pattern_id", "branch_batch_index"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=0))
        for name in ("search_yield_logit", "mixture_log_weight"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite or None")
            object.__setattr__(self, name, None if value is None else float(value))
        object.__setattr__(self, "source_id", self.source_id.strip())
        object.__setattr__(self, "local_unit", _local_vector(self.local_unit))
        object.__setattr__(self, "latent_components", components)


def v5_refinement_seed_from_proposal(proposal: V5LocalProposal) -> V5LocalRefinementSeed:
    """Retain a neural proposal's local physics and its two non-probability scores."""

    if not isinstance(proposal, V5LocalProposal):
        raise TypeError("proposal must be a V5LocalProposal")
    return V5LocalRefinementSeed(
        source="neural",
        source_id=(
            f"pattern_{proposal.pattern_id:02d}:mixture_{proposal.mixture_index:02d}:"
            f"{proposal.source}_{proposal.draw_index:03d}"
        ),
        query_sha256=proposal.query_sha256,
        topology_id=proposal.topology_id,
        pattern_id=proposal.pattern_id,
        branch_batch_index=proposal.branch_batch_index,
        local_unit=proposal.local_unit,
        latent_components=proposal.latent_components,
        resolution=proposal.resolution,
        search_yield_logit=proposal.search_yield_logit,
        mixture_log_weight=proposal.mixture_log_weight,
    )


def v5_external_local_refinement_seed(
    batch: V5CandidateContextBatch,
    *,
    source: Literal["retrieval", "sobol"],
    source_id: str,
    pattern_id: int,
    local_unit: Sequence[float],
) -> V5LocalRefinementSeed:
    """Create a future retrieval/Sobol seed through the same local codec."""

    if source not in {"retrieval", "sobol"}:
        raise ValueError("external source must be retrieval or sobol")
    if not isinstance(batch, V5CandidateContextBatch):
        raise TypeError("batch must be a V5CandidateContextBatch")
    try:
        batch_index = batch.pattern_ids.index(int(pattern_id))
    except (ValueError, TypeError) as exc:
        raise ValueError("pattern_id is not present in the candidate batch") from exc
    codec = batch.query.codec_for(int(pattern_id))
    components, resolution = codec.decode(_local_vector(local_unit))
    canonical = canonicalize_component_slots(
        codec,
        components,
        resolution,
        component_intensity_bounds=batch.amplitude_query.component_intensities,
    )
    return V5LocalRefinementSeed(
        source=source,
        source_id=source_id,
        query_sha256=batch.query_sha256,
        topology_id=batch.query.topology_id,
        pattern_id=int(pattern_id),
        branch_batch_index=batch_index,
        local_unit=canonical.coordinates.unit_cube,
        latent_components=canonical.components,
        resolution=canonical.resolution,
    )


def v5_external_physical_refinement_seed(
    batch: V5CandidateContextBatch,
    *,
    source: Literal["retrieval", "sobol"],
    source_id: str,
    pattern_id: int,
    components: Sequence[LatentComponentParameters],
    resolution: ResolutionShape | None,
) -> V5LocalRefinementSeed:
    """Validate a physical fallback seed and encode it only in the user codec."""

    if source not in {"retrieval", "sobol"}:
        raise ValueError("external source must be retrieval or sobol")
    if not isinstance(batch, V5CandidateContextBatch):
        raise TypeError("batch must be a V5CandidateContextBatch")
    try:
        batch_index = batch.pattern_ids.index(int(pattern_id))
    except (ValueError, TypeError) as exc:
        raise ValueError("pattern_id is not present in the candidate batch") from exc
    codec = batch.query.codec_for(int(pattern_id))
    values = tuple(components)
    canonical = canonicalize_component_slots(
        codec,
        values,
        resolution,
        component_intensity_bounds=batch.amplitude_query.component_intensities,
    )
    return V5LocalRefinementSeed(
        source=source,
        source_id=source_id,
        query_sha256=batch.query_sha256,
        topology_id=batch.query.topology_id,
        pattern_id=int(pattern_id),
        branch_batch_index=batch_index,
        local_unit=canonical.coordinates.unit_cube,
        latent_components=canonical.components,
        resolution=canonical.resolution,
    )


class _ForwardBudgetReached(Exception):
    pass


def _validate_seed(
    batch: V5CandidateContextBatch,
    seed: V5LocalRefinementSeed,
) -> tuple[ProfiledBranchCodec, GuiAmplitudeConstraint]:
    if seed.query_sha256 != batch.query_sha256:
        raise ValueError("seed belongs to a different geometry/amplitude user query")
    if seed.branch_batch_index >= batch.branch_count:
        raise ValueError("seed branch_batch_index is outside the candidate batch")
    condition = batch.branch_conditions[seed.branch_batch_index]
    if (seed.topology_id, seed.pattern_id) != (condition.topology_id, condition.pattern_id):
        raise ValueError("seed branch identity disagrees with its candidate-batch row")
    codec = batch.query.codec_for(seed.pattern_id)
    if codec.d_present != condition.d_present[: len(codec.topology)]:
        raise ValueError("seed branch D presence disagrees with the user codec")
    if (seed.resolution is not None) != condition.resolution_present:
        raise ValueError("seed Resolution presence disagrees with the hard branch")
    coordinates = codec.encode(seed.latent_components, seed.resolution)
    if not np.allclose(coordinates.unit_cube, seed.local_unit, rtol=0.0, atol=5.0e-12):
        raise ValueError("seed physical values do not round-trip through its user-local codec")
    constraint = batch.amplitude_constraints[seed.branch_batch_index]
    constraint.validate_branch(len(seed.latent_components), seed.resolution is not None)
    return codec, constraint


def _physics_violations(
    codec: ProfiledBranchCodec,
    components: Sequence[LatentComponentParameters],
) -> tuple[str, ...]:
    gui = codec.latent_components_to_gui(components)
    violations = ConstraintSet.defaults().validate_components(
        tuple({"type": item.shape, "params": asdict(item)} for item in gui)
    )
    return tuple(
        f"{item.constraint_id}:component[{item.component_index}]:{item.message}"
        for item in violations
    )


def _profile_once(
    curve: ObservedCurve,
    seed: V5LocalRefinementSeed,
    codec: ProfiledBranchCodec,
    constraint: GuiAmplitudeConstraint,
    note_exact_call,
) -> tuple[
    ProfiledForwardResult,
    np.ndarray,
    tuple[LatentComponentParameters, ...],
    ResolutionShape | None,
    str,
]:
    gui = codec.latent_components_to_gui(seed.latent_components)
    profile_sigma = curve.intensity * (1.0 if curve.sigma_log is None else curve.sigma_log)
    note_exact_call("seed_profile_verification")
    profile = profile_linear_amplitudes(
        curve.q,
        curve.intensity,
        gui,
        resolution=seed.resolution,
        sigma=profile_sigma,
        amplitude_constraint=constraint,
    )
    exact = evaluate_profiled_forward(curve.q, profile)
    return (
        profile,
        exact,
        seed.latent_components,
        seed.resolution,
        "exact seed profile; "
        f"amplitude_solver_status={profile.solver_status}; "
        f"amplitude_solver_message={profile.solver_message}",
    )


def run_v5_exact_refinement(
    batch: V5CandidateContextBatch,
    curve: ObservedCurve,
    proposals: Sequence[V5LocalProposal | V5LocalRefinementSeed],
    *,
    per_candidate_forward_evaluation_limit: int = 128,
    forward_evaluation_limit: int = 4096,
    ftol: float = 1.0e-8,
    xtol: float = 1.0e-8,
    gtol: float = 1.0e-8,
) -> V5ExactRefinementBatchResult:
    """Refine all supplied seeds in order; one candidate failure never aborts later seeds."""

    if not isinstance(batch, V5CandidateContextBatch):
        raise TypeError("batch must be a V5CandidateContextBatch")
    if not isinstance(curve, ObservedCurve):
        raise TypeError("curve must be an ObservedCurve")
    per_limit = _integer(
        per_candidate_forward_evaluation_limit,
        "per_candidate_forward_evaluation_limit",
        minimum=1,
    )
    total_limit = _integer(forward_evaluation_limit, "forward_evaluation_limit", minimum=0)
    raw = tuple(proposals)
    if not raw:
        raise ValueError("proposals must contain at least one V5 local seed")
    seeds = tuple(
        v5_refinement_seed_from_proposal(value) if isinstance(value, V5LocalProposal) else value
        for value in raw
    )
    if not all(isinstance(value, V5LocalRefinementSeed) for value in seeds):
        raise TypeError("proposals must contain V5LocalProposal or V5LocalRefinementSeed values")

    attempts: list[V5ExactRefinementAttempt] = []
    candidates: list[CandidateInput] = []
    total_phases: list[str] = []
    all_processed = True

    for seed in seeds:
        assert isinstance(seed, V5LocalRefinementSeed)
        before = len(total_phases)
        remaining = total_limit - before
        if remaining <= 0:
            all_processed = False
            attempts.append(
                V5ExactRefinementAttempt(
                    attempt_rank=len(attempts) + 1,
                    source=seed.source,
                    source_id=seed.source_id,
                    topology_id=seed.topology_id,
                    pattern_id=seed.pattern_id,
                    branch_batch_index=seed.branch_batch_index,
                    status="total_forward_budget_exhausted_before_seed",
                    candidate_id=None,
                    candidate_rank=None,
                    forward_call_limit=0,
                    exact_forward_calls=0,
                    exact_forward_calls_by_phase=exact_forward_phase_counts(()),
                    cumulative_calls_before=before,
                    cumulative_calls_after=before,
                    message="total exact-forward budget exhausted before this seed",
                    search_yield_logit=seed.search_yield_logit,
                    mixture_log_weight=seed.mixture_log_weight,
                )
            )
            break
        allowance = min(per_limit, remaining)
        local_phases: list[str] = []

        def note_exact_call(phase: str) -> None:
            if phase not in EXACT_FORWARD_PHASES:
                raise ValueError("unknown exact-forward call phase")
            if len(local_phases) >= allowance:
                raise _ForwardBudgetReached
            local_phases.append(phase)
            total_phases.append(phase)

        try:
            codec, constraint = _validate_seed(batch, seed)
        except _CANDIDATE_ERRORS as exc:
            attempts.append(
                V5ExactRefinementAttempt(
                    attempt_rank=len(attempts) + 1,
                    source=seed.source,
                    source_id=seed.source_id,
                    topology_id=seed.topology_id,
                    pattern_id=seed.pattern_id,
                    branch_batch_index=seed.branch_batch_index,
                    status="validation_failed",
                    candidate_id=None,
                    candidate_rank=None,
                    forward_call_limit=allowance,
                    exact_forward_calls=0,
                    exact_forward_calls_by_phase=exact_forward_phase_counts(()),
                    cumulative_calls_before=before,
                    cumulative_calls_after=before,
                    message=f"{type(exc).__name__}: {exc}",
                    search_yield_logit=seed.search_yield_logit,
                    mixture_log_weight=seed.mixture_log_weight,
                )
            )
            continue

        try:
            condition_varying = tuple(
                batch.branch_conditions[seed.branch_batch_index].varying_dimension_mask
            )
            if condition_varying != codec.varying_mask:
                raise RuntimeError("candidate branch condition disagrees with codec varying axes")
            varying_count = len(codec.varying_indices)
            solver_nfev = (allowance - 3) // (varying_count + 1)
            if varying_count == 0 or solver_nfev < 1:
                profile, exact, final_components, final_resolution, message = _profile_once(
                    curve, seed, codec, constraint, note_exact_call
                )
                initial_profile = profile
            else:
                refined = refine_profiled_branch(
                    curve.q,
                    curve.intensity,
                    codec.component_bounds,
                    seed.latent_components,
                    resolution_bounds=codec.resolution_bounds,
                    resolution_seed=seed.resolution,
                    sigma_log=curve.sigma_log,
                    amplitude_constraint=constraint,
                    max_nfev=solver_nfev,
                    ftol=ftol,
                    xtol=xtol,
                    gtol=gtol,
                    exact_forward_call_hook=note_exact_call,
                    numeric_policy_version=codec.numeric_policy_version,
                )
                if refined.gui_amplitude_constraint is not constraint:
                    raise RuntimeError(
                        "refinement replaced the candidate-batch amplitude constraint"
                    )
                if refined.exact_forward_call_phases != tuple(local_phases):
                    raise RuntimeError("refinement exact-forward ledger disagrees with its hook")
                profile = refined.final_profile
                initial_profile = refined.initial_profile
                exact = refined.exact_forward_intensity
                final_components = refined.final_latent_components
                final_resolution = refined.final_resolution
                message = (
                    "nonlinear optimizer converged"
                    if refined.success
                    else "best exact point retained after optimizer termination"
                )
                message += (
                    "; final_amplitude_solver_status="
                    f"{profile.solver_status}; final_amplitude_solver_message="
                    f"{profile.solver_message}"
                )

            final_coordinates = codec.encode(final_components, final_resolution)
            bounds_satisfied = all(
                bounds.contains(value, codec.numeric_policy_version)
                for bounds, value in zip(codec.latent_bounds, final_components)
            ) and (
                codec.resolution_bounds is None
                or codec.resolution_bounds.contains(final_resolution)
            )
            resolution_satisfied = (final_resolution is not None) == (
                codec.resolution_bounds is not None
            )
            violations = _physics_violations(codec, final_components)
            initial_amplitude = initial_profile.amplitude_constraint_audit
            final_amplitude = profile.amplitude_constraint_audit
            if initial_amplitude is None or final_amplitude is None:
                raise RuntimeError("V5 exact refinement lost its GUI amplitude audit")
            exact_consistent = bool(
                np.allclose(exact, profile.fitted_intensity, rtol=2.0e-10, atol=1.0e-12)
            )
            prerequisites = V5CandidatePrerequisiteAudit(
                query_sha256=batch.query_sha256,
                geometry_query_sha256=batch.query.sha256,
                amplitude_query_sha256=batch.amplitude_query.sha256,
                pattern_id=seed.pattern_id,
                initial_local_unit=seed.local_unit,
                final_local_unit=final_coordinates.unit_cube,
                user_local_codec_version=BRANCH_CODEC_VERSION,
                amplitude_constraint_version=constraint.version,
                amplitude_constraint_sha256=_constraint_sha256(constraint),
                amplitude_constraint_identity_preserved=(
                    constraint is batch.amplitude_constraints[seed.branch_batch_index]
                ),
                initial_amplitude_audit=initial_amplitude,
                final_amplitude_audit=final_amplitude,
                geometry_bounds_satisfied=bounds_satisfied,
                physics_satisfied=not violations,
                physics_violations=violations,
                resolution_presence_satisfied=resolution_satisfied,
                exact_forward_consistency_satisfied=exact_consistent,
                all_prerequisites_satisfied=bool(
                    bounds_satisfied
                    and not violations
                    and resolution_satisfied
                    and exact_consistent
                    and initial_amplitude.all_constraints_satisfied
                    and final_amplitude.all_constraints_satisfied
                ),
            )
            if not prerequisites.all_prerequisites_satisfied:
                raise RuntimeError("refined V5 candidate failed range/physics prerequisites")
            candidate_id = f"candidate_{len(candidates) + 1:05d}"
            candidate_rank = len(candidates) + 1
            candidate = CandidateInput(
                candidate_id=candidate_id,
                proposal_rank=candidate_rank,
                topology_id=seed.topology_id,
                components=tuple(final_components),
                resolution=final_resolution,
                linear_solution=LinearSolutionSnapshot.from_profiled_forward(profile),
                exact_intensity=exact,
                bounds_pass=True,
                physics_pass=True,
                proposal_score_raw=None,
            )
            candidates.append(candidate)
            status: V5AttemptStatus = "refined"
            candidate_id_value: str | None = candidate_id
            candidate_rank_value: int | None = candidate_rank
            audit: V5CandidatePrerequisiteAudit | None = prerequisites
        except _ForwardBudgetReached:
            status = "per_candidate_forward_budget_exhausted"
            candidate_id_value = candidate_rank_value = audit = None
            message = "per-candidate exact-forward budget exhausted during refinement"
        except _CANDIDATE_ERRORS as exc:
            status = "refinement_failed"
            candidate_id_value = candidate_rank_value = audit = None
            message = f"{type(exc).__name__}: {exc}"

        used = len(local_phases)
        attempts.append(
            V5ExactRefinementAttempt(
                attempt_rank=len(attempts) + 1,
                source=seed.source,
                source_id=seed.source_id,
                topology_id=seed.topology_id,
                pattern_id=seed.pattern_id,
                branch_batch_index=seed.branch_batch_index,
                status=status,
                candidate_id=candidate_id_value,
                candidate_rank=candidate_rank_value,
                forward_call_limit=allowance,
                exact_forward_calls=used,
                exact_forward_calls_by_phase=exact_forward_phase_counts(local_phases),
                cumulative_calls_before=before,
                cumulative_calls_after=before + used,
                message=message,
                prerequisite_audit=audit,
                search_yield_logit=seed.search_yield_logit,
                mixture_log_weight=seed.mixture_log_weight,
            )
        )

    calls_used = len(total_phases)
    ledger = V5ExactForwardLedger(
        budget_unit=V5_EXACT_FORWARD_BUDGET_UNIT,
        configured_total_limit=total_limit,
        configured_per_candidate_limit=per_limit,
        calls_used=calls_used,
        calls_remaining=total_limit - calls_used,
        calls_by_phase=exact_forward_phase_counts(total_phases),
        input_seed_count=len(seeds),
        attempts_recorded=len(attempts),
        refinement_successes=sum(item.status == "refined" for item in attempts),
        validation_failures=sum(item.status == "validation_failed" for item in attempts),
        refinement_failures=sum(item.status == "refinement_failed" for item in attempts),
        per_candidate_budget_exhausted_attempts=sum(
            item.status == "per_candidate_forward_budget_exhausted" for item in attempts
        ),
        total_budget_exhausted_before_seed_attempts=sum(
            item.status == "total_forward_budget_exhausted_before_seed" for item in attempts
        ),
    )
    if candidates:
        status_value: V5ExactBatchStatus = (
            "exact_candidates_ready_for_verification"
            if all_processed
            else "exact_candidates_partial_budget_exhausted"
        )
    else:
        status_value = "no_candidate_found_within_budget"
    return V5ExactRefinementBatchResult(
        status=status_value,
        query_sha256=batch.query_sha256,
        candidates=tuple(candidates),
        attempts=tuple(attempts),
        ledger=ledger,
        all_input_seeds_processed=all_processed,
    )


__all__ = [
    "V5_EXACT_REFINEMENT_SCHEMA",
    "V5_EXACT_REFINEMENT_SCOPE",
    "V5_EXACT_REFINEMENT_VERSION",
    "V5_EXACT_FORWARD_BUDGET_UNIT",
    "V5CandidatePrerequisiteAudit",
    "V5ExactForwardLedger",
    "V5ExactRefinementAttempt",
    "V5ExactRefinementBatchResult",
    "V5LocalRefinementSeed",
    "run_v5_exact_refinement",
    "v5_external_local_refinement_seed",
    "v5_external_physical_refinement_seed",
    "v5_refinement_seed_from_proposal",
]
