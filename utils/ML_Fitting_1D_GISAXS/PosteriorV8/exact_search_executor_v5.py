"""CPU-safe frozen exact-search executor for V5.1 supervision labels.

This runner deliberately has no model input and never imports a learned model.
It searches one universal-query branch from direct Sobol local-unit starts,
profiles the exact GUI amplitude polytope, and spends the complete frozen
per-branch forward budget before producing either a positive or a completed
operational negative.  Any incomplete or inconsistent execution fails closed
to ``unverified`` and is still preserved in a checked audit artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import scipy

from .branch_codec import BRANCH_CODEC_VERSION, UNIT_CUBE_DIMENSIONS
from .calibrated_search_threshold_v5 import V5CalibrationArtifactIdentity
from .candidate_refinement_contract_v5 import (
    EXACT_FORWARD_PHASES,
    V5_EXACT_REFINEMENT_VERSION,
)
from .candidate_refinement_v5 import (
    V5LocalRefinementSeed,
    run_v5_exact_refinement,
    v5_external_local_refinement_seed,
)
from .contract import FORWARD_MODEL_VERSION, MAX_COMPONENTS, latent_component_to_gui
from .evaluation import (
    CLUSTERING_LINKAGE,
    EVALUATION_AUDIT_SCHEMA,
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
    CandidateInput,
    LinearSolutionSnapshot,
    complete_linkage_groups,
    natural_log_rmse,
)
from .exact_search_schedule_v5 import (
    V5_EXACT_SEARCH_TERMINATION_POLICY_ID,
    V5_EXACT_SEARCH_TERMINATION_REASONS,
    V5FrozenExactOptimizerSchedule,
    V5FrozenLocalSobolSchedule,
)
from .grouped_artifact_v5 import (
    V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
    V5_CHECKED_ARRAY_ARTIFACT_VERSION,
    V5ArtifactReceipt,
    array_manifest,
    array_sha256,
    canonical_json,
    read_checked_array_artifact,
    write_checked_array_artifact,
)
from .gui_amplitude_constraints import GUI_AMPLITUDE_CONSTRAINT_VERSION
from .profiled_forward import (
    PROFILED_AMPLITUDE_SOLVER_VERSION,
    evaluate_gui_forward_snapshot,
)
from .query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_PAYLOAD,
    V5_QUERY_PARAMETER_DISTANCE_SCOPE,
    V5_QUERY_PARAMETER_DISTANCE_SHA256,
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
    query_local_parameter_distance,
)
from .search_supervision_contract_v5 import (
    V5_ENGINEERING_PILOT_CLAIM_LIMITS,
    V5_PAPER_FULL_CALIBRATED_CLAIM_LIMITS,
    V5CompatibleRepresentativeReference,
    V5FrozenBranchSearchResult,
    V5FrozenExactSearchProtocol,
    V5FrozenSearchTask,
    V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)
from .sobol_numeric_canonicalization_v5 import (
    v5_numeric_policy_sha256,
    validate_v5_numeric_policy,
)
from .universal_query_contract_v5 import amplitude_constraint_sha256


V5_EXACT_SEARCH_EXECUTOR_SCHEMA = "gisaxs.posterior_v8.exact_search_executor/v1"
V5_EXACT_SEARCH_EXECUTOR_VERSION = (
    "posterior_v8_full_source_numeric_policy_bound_exact_multistart_executor_v5"
)
V5_EXACT_SEARCH_EVALUATOR_VERSION = "posterior_v8_selected_exact_metric_bounds_physics_gate_v1"
V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID = V5_QUERY_PARAMETER_DISTANCE_VERSION
V5_EXACT_SEARCH_ARTIFACT_SUFFIX = ".gvd5"
V5_EXACT_SEARCH_COMPLETION_RULE = (
    "all_direct_scouts_complete_and_exact_forward_calls_equal_frozen_branch_budget_"
    "with_no_execution_error_v1"
)

_REPRESENTATIVE_DISTANCE_PAYLOAD = {
    "distance_id": V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID,
    "distance_scope": V5_QUERY_PARAMETER_DISTANCE_SCOPE,
    "query_distance_sha256": V5_QUERY_PARAMETER_DISTANCE_SHA256,
    "query_distance_contract": V5_QUERY_PARAMETER_DISTANCE_PAYLOAD,
    "implementation": "query_local_parameter_distance",
    "clustering_linkage": CLUSTERING_LINKAGE,
    "cross_topology_distance": "infinity",
    "cluster_interpretation": "complete_linkage_maximum_within_cluster_diameter_at_most_delta",
}
V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_SHA256 = sha256(
    canonical_json(_REPRESENTATIVE_DISTANCE_PAYLOAD).encode("utf-8")
).hexdigest()


@dataclass(frozen=True, eq=False)
class _CandidateRecord:
    candidate: CandidateInput
    local_unit: tuple[float, ...]
    seed_index: int
    stage: str
    raw_log_rmse: float
    standardized_log_rmse: float | None
    selected_metric_value: float
    exact_intensity_sha256: str


@dataclass(frozen=True)
class _AttemptRecord:
    seed_index: int
    stage: str
    source_id: str
    status: str
    message: str
    exact_forward_calls: int
    cumulative_calls_before: int
    cumulative_calls_after: int
    calls_by_phase: tuple[int, ...]
    candidate_index: int
    initial_local_unit: tuple[float, ...]


@dataclass(frozen=True)
class V5ExactSearchExecutorArtifact:
    manifest: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]
    receipt: V5ArtifactReceipt


def _candidate_exact_sha256(candidate: CandidateInput) -> str:
    return array_sha256("candidate_exact_intensity", candidate.exact_intensity)


def _selected_metrics(
    task: V5FrozenSearchTask, candidate: CandidateInput
) -> tuple[float, float | None, float]:
    raw = natural_log_rmse(candidate.exact_intensity, task.observed_curve.intensity)
    standardized = None
    if task.observed_curve.sigma_log is not None:
        standardized = natural_log_rmse(
            candidate.exact_intensity,
            task.observed_curve.intensity,
            sigma_log=task.observed_curve.sigma_log,
        )
    selected = raw if standardized is None else standardized
    return raw, standardized, selected


def _representative_distance_contract() -> tuple[str, str]:
    return (
        V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID,
        V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_SHA256,
    )


def build_v5_frozen_exact_search_protocol(
    *,
    protocol_id: str,
    seed_schedule: V5FrozenLocalSobolSchedule,
    optimizer_schedule: V5FrozenExactOptimizerSchedule,
    standardized_threshold_name: str | None = None,
    standardized_threshold_value: float | None = None,
    raw_threshold_name: str | None = None,
    raw_threshold_value: float | None = None,
    threshold_source_id: str | None = None,
    protocol_tier: str = V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT,
    calibration_identity: V5CalibrationArtifactIdentity | None = None,
    delta_separation_threshold: float = 0.08,
) -> V5FrozenExactSearchProtocol:
    """Build a protocol whose hashes reproduce this concrete executor."""

    if not isinstance(seed_schedule, V5FrozenLocalSobolSchedule):
        raise TypeError("seed_schedule must be a V5FrozenLocalSobolSchedule")
    if not isinstance(optimizer_schedule, V5FrozenExactOptimizerSchedule):
        raise TypeError("optimizer_schedule must be a V5FrozenExactOptimizerSchedule")
    distance_id, distance_sha = _representative_distance_contract()
    if protocol_tier == V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT:
        claim_limits = V5_ENGINEERING_PILOT_CLAIM_LIMITS
    elif protocol_tier == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED:
        claim_limits = V5_PAPER_FULL_CALIBRATED_CLAIM_LIMITS
    else:
        raise ValueError("unsupported frozen-search protocol tier")
    return V5FrozenExactSearchProtocol(
        protocol_id=protocol_id,
        evaluator_version=V5_EXACT_SEARCH_EVALUATOR_VERSION,
        authoritative_forward_id=FORWARD_MODEL_VERSION,
        metric_name=STANDARDIZED_LOG_RMSE_METRIC,
        threshold_name=standardized_threshold_name,
        threshold_value=standardized_threshold_value,
        threshold_source_id=threshold_source_id,
        missing_acceptance_sigma_metric_name=RAW_LOG_RMSE_METRIC,
        missing_acceptance_sigma_threshold_name=raw_threshold_name,
        missing_acceptance_sigma_threshold_value=raw_threshold_value,
        exact_forward_call_budget=seed_schedule.point_count,
        seed_schedule_id=seed_schedule.schedule_id,
        seed_schedule_sha256=seed_schedule.sha256,
        optimizer_schedule_id=optimizer_schedule.schedule_id,
        optimizer_schedule_sha256=optimizer_schedule.sha256,
        termination_policy_id=optimizer_schedule.termination_policy_id,
        representative_distance_id=distance_id,
        representative_distance_sha256=distance_sha,
        delta_separation_threshold=delta_separation_threshold,
        protocol_tier=protocol_tier,
        claim_limits=claim_limits,
        calibration_identity=calibration_identity,
    )


def _validate_executor_binding(
    task: V5FrozenSearchTask,
    seed_schedule: V5FrozenLocalSobolSchedule,
    optimizer_schedule: V5FrozenExactOptimizerSchedule,
) -> None:
    if not isinstance(task, V5FrozenSearchTask):
        raise TypeError("task must be a V5FrozenSearchTask")
    if not isinstance(seed_schedule, V5FrozenLocalSobolSchedule):
        raise TypeError("seed_schedule must be a V5FrozenLocalSobolSchedule")
    if not isinstance(optimizer_schedule, V5FrozenExactOptimizerSchedule):
        raise TypeError("optimizer_schedule must be a V5FrozenExactOptimizerSchedule")
    seed_schedule.verify_runtime_replay()
    protocol = task.protocol
    if protocol.exact_forward_call_budget != seed_schedule.point_count:
        raise ValueError("frozen Sobol point count must equal the per-branch budget")
    if optimizer_schedule.direct_scout_seed_count > seed_schedule.point_count:
        raise ValueError("direct scout prefix exceeds the frozen Sobol schedule")
    expected = (
        (protocol.seed_schedule_id, seed_schedule.schedule_id, "seed schedule ID"),
        (protocol.seed_schedule_sha256, seed_schedule.sha256, "seed schedule digest"),
        (protocol.optimizer_schedule_id, optimizer_schedule.schedule_id, "optimizer schedule ID"),
        (
            protocol.optimizer_schedule_sha256,
            optimizer_schedule.sha256,
            "optimizer schedule digest",
        ),
        (
            protocol.termination_policy_id,
            V5_EXACT_SEARCH_TERMINATION_POLICY_ID,
            "termination policy",
        ),
        (
            optimizer_schedule.termination_policy_id,
            protocol.termination_policy_id,
            "optimizer termination policy",
        ),
        (
            protocol.representative_distance_id,
            V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID,
            "representative distance ID",
        ),
        (
            protocol.representative_distance_sha256,
            V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_SHA256,
            "representative distance digest",
        ),
        (
            protocol.evaluator_version,
            V5_EXACT_SEARCH_EVALUATOR_VERSION,
            "evaluator version",
        ),
        (
            protocol.authoritative_forward_id,
            FORWARD_MODEL_VERSION,
            "authoritative forward ID",
        ),
    )
    for actual, wanted, label in expected:
        if actual != wanted:
            raise ValueError(f"{label} does not reproduce the concrete executor")
    branch = task.branch
    batch = task.universal_context.batches[branch.topology_batch_index]
    if batch.branch_conditions[branch.branch_batch_index] != branch.condition:
        raise ValueError("universal branch escaped its owning topology batch")
    if batch.amplitude_constraints[branch.branch_batch_index] is not task.amplitude_constraint:
        raise ValueError("task did not preserve the owning GUI amplitude constraint object")
    if amplitude_constraint_sha256(task.amplitude_constraint) != branch.amplitude_constraint_sha256:
        raise ValueError("task GUI amplitude constraint digest does not reproduce")
    if task.selected_metric_name not in {
        RAW_LOG_RMSE_METRIC,
        STANDARDIZED_LOG_RMSE_METRIC,
    }:
        raise ValueError("task selected an unsupported exact-search metric")


def _artifact_identity(
    task: V5FrozenSearchTask,
    seed_schedule: V5FrozenLocalSobolSchedule,
    optimizer_schedule: V5FrozenExactOptimizerSchedule,
) -> str:
    payload = {
        "schema": V5_EXACT_SEARCH_EXECUTOR_SCHEMA,
        "clean_group_id": task.clean_group_id,
        "recipe_id": task.recipe_id,
        "observation_id": task.observation_id,
        "exact_curve_sha256": task.exact_curve_sha256,
        "universal_query_sha256": task.universal_context.audit_sha256,
        "global_branch_key": task.branch.global_key.wire_key,
        "context_sha256": task.branch.context_sha256,
        "protocol_sha256": task.protocol.sha256,
        "task_audit_sha256": task.audit_sha256,
        "seed_schedule_sha256": seed_schedule.sha256,
        "optimizer_schedule_sha256": optimizer_schedule.sha256,
    }
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def v5_exact_search_artifact_path(
    output_directory: str | Path,
    task: V5FrozenSearchTask,
    seed_schedule: V5FrozenLocalSobolSchedule,
    optimizer_schedule: V5FrozenExactOptimizerSchedule,
) -> Path:
    root = Path(output_directory)
    identity = _artifact_identity(task, seed_schedule, optimizer_schedule)
    return root / f"exact_search_{identity}{V5_EXACT_SEARCH_ARTIFACT_SUFFIX}"


def _run_one_seed(
    task: V5FrozenSearchTask,
    seed_schedule: V5FrozenLocalSobolSchedule,
    optimizer_schedule: V5FrozenExactOptimizerSchedule,
    *,
    seed_index: int,
    stage: str,
    allowance: int,
    cumulative_calls: int,
    candidate_count: int,
) -> tuple[_AttemptRecord, _CandidateRecord | None, str | None]:
    branch = task.branch
    batch = task.universal_context.batches[branch.topology_batch_index]
    local = seed_schedule.points[seed_index]
    seed: V5LocalRefinementSeed = v5_external_local_refinement_seed(
        batch,
        source="sobol",
        source_id=f"{seed_schedule.schedule_id}:{stage}:seed-{seed_index:06d}",
        pattern_id=branch.global_key.pattern_id,
        local_unit=local,
    )
    if seed.branch_batch_index != branch.branch_batch_index:
        raise RuntimeError("Sobol seed resolved to the wrong owning topology branch")
    result = run_v5_exact_refinement(
        batch,
        task.observed_curve,
        (seed,),
        per_candidate_forward_evaluation_limit=allowance,
        forward_evaluation_limit=allowance,
        ftol=optimizer_schedule.ftol,
        xtol=optimizer_schedule.xtol,
        gtol=optimizer_schedule.gtol,
    )
    if result.query_sha256 != batch.query_sha256 or len(result.attempts) != 1:
        raise RuntimeError("exact refiner returned an inconsistent single-seed result")
    attempt = result.attempts[0]
    used = result.ledger.calls_used
    if used < 0 or used > allowance or attempt.exact_forward_calls != used:
        raise RuntimeError("exact refiner did not produce a bounded call ledger")
    if (
        attempt.source != "sobol"
        or attempt.search_yield_logit is not None
        or attempt.mixture_log_weight is not None
    ):
        raise RuntimeError("frozen exact search received learned proposal scores")
    seed_failure = None
    if attempt.status in {"validation_failed", "refinement_failed"}:
        seed_failure = f"exact refiner failed a frozen Sobol seed: {attempt.message}"
    elif used == 0:
        seed_failure = "exact refiner made no progress on a frozen Sobol seed"
    if len(result.candidates) > 1:
        raise RuntimeError("single Sobol start produced multiple untracked candidates")
    phase_counts = dict(result.ledger.calls_by_phase)
    candidate_record = None
    candidate_index = -1
    if result.candidates:
        candidate_index = candidate_count
        candidate = replace(
            result.candidates[0],
            candidate_id=f"search_candidate_{candidate_count + 1:06d}",
            proposal_rank=candidate_count + 1,
        )
        if candidate.proposal_score_raw is not None:
            raise RuntimeError("frozen exact candidate unexpectedly carries a model score")
        coordinates = task.codec.encode(candidate.components, candidate.resolution)
        raw, standardized, selected = _selected_metrics(task, candidate)
        candidate_record = _CandidateRecord(
            candidate=candidate,
            local_unit=coordinates.unit_cube,
            seed_index=seed_index,
            stage=stage,
            raw_log_rmse=raw,
            standardized_log_rmse=standardized,
            selected_metric_value=selected,
            exact_intensity_sha256=_candidate_exact_sha256(candidate),
        )
    record = _AttemptRecord(
        seed_index=seed_index,
        stage=stage,
        source_id=seed.source_id,
        status=attempt.status,
        message=attempt.message,
        exact_forward_calls=used,
        cumulative_calls_before=cumulative_calls,
        cumulative_calls_after=cumulative_calls + used,
        calls_by_phase=tuple(phase_counts[name] for name in EXACT_FORWARD_PHASES),
        candidate_index=candidate_index,
        initial_local_unit=seed.local_unit,
    )
    return record, candidate_record, seed_failure


def _compatible(record: _CandidateRecord, threshold: float) -> bool:
    return bool(
        record.selected_metric_value <= threshold
        and record.candidate.bounds_pass
        and record.candidate.physics_pass
    )


def _representatives(
    task: V5FrozenSearchTask,
    artifact_id: str,
    candidates: Sequence[_CandidateRecord],
) -> tuple[
    tuple[V5CompatibleRepresentativeReference, ...], tuple[int, ...], list[dict[str, object]]
]:
    compatible = [
        (index, record)
        for index, record in enumerate(candidates)
        if _compatible(record, task.selected_threshold_value)
    ]
    groups = complete_linkage_groups(
        compatible,
        lambda left, right: query_local_parameter_distance(
            task, left[1].candidate, right[1].candidate
        ),
        task.protocol.delta_separation_threshold,
    )
    groups.sort(key=lambda group: min(value[1].candidate.proposal_rank for value in group))
    selected: list[tuple[int, _CandidateRecord, str, str]] = []
    payloads: list[dict[str, object]] = []
    for cluster_number, group in enumerate(groups, 1):
        index, record = min(
            group,
            key=lambda value: (
                value[1].selected_metric_value,
                value[1].candidate.proposal_rank,
            ),
        )
        cluster_id = f"cluster-{cluster_number:04d}"
        candidate = record.candidate
        payload = {
            "artifact_id": f"{artifact_id}#representative-{cluster_number:04d}",
            "cluster_id": cluster_id,
            "candidate_index": index,
            "candidate_id": candidate.candidate_id,
            "target_local": list(record.local_unit),
            "metric_name": task.selected_metric_name,
            "metric_value": record.selected_metric_value,
            "bounds_passed": candidate.bounds_pass,
            "physics_passed": candidate.physics_pass,
            "exact_intensity_sha256": record.exact_intensity_sha256,
            "components": [asdict(value) for value in candidate.components],
            "resolution": None if candidate.resolution is None else asdict(candidate.resolution),
            "linear_solution": asdict(candidate.linear_solution),
        }
        payload_sha = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        payloads.append({**payload, "artifact_sha256": payload_sha})
        selected.append((index, record, cluster_id, payload_sha))
    set_id = f"{artifact_id}#complete-linkage-diameter-cluster-set"
    set_payload = {
        "representative_set_id": set_id,
        "distance_id": task.protocol.representative_distance_id,
        "distance_sha256": task.protocol.representative_distance_sha256,
        "delta": task.protocol.delta_separation_threshold,
        "linkage": task.protocol.clustering_linkage,
        "representative_artifact_sha256": [value[3] for value in selected],
    }
    set_sha = sha256(canonical_json(set_payload).encode("utf-8")).hexdigest()
    references = tuple(
        V5CompatibleRepresentativeReference(
            artifact_id=payloads[number]["artifact_id"],
            artifact_sha256=payload_sha,
            representative_set_id=set_id,
            representative_set_sha256=set_sha,
            cluster_id=cluster_id,
            metric_value=record.selected_metric_value,
            bounds_passed=record.candidate.bounds_pass,
            physics_passed=record.candidate.physics_pass,
            target_local=record.local_unit,
        )
        for number, (_, record, cluster_id, payload_sha) in enumerate(selected)
    )
    return references, tuple(value[0] for value in selected), payloads


def _arrays(
    schedule: V5FrozenLocalSobolSchedule,
    attempts: Sequence[_AttemptRecord],
    candidates: Sequence[_CandidateRecord],
    representative_indices: Sequence[int],
    *,
    threshold: float,
    q_count: int,
) -> Mapping[str, np.ndarray]:
    particle_amplitudes = np.zeros((len(candidates), MAX_COMPONENTS), dtype=np.float64)
    for row, record in enumerate(candidates):
        values = record.candidate.linear_solution.particle_amplitudes
        particle_amplitudes[row, : len(values)] = values
    representative_curves = (
        np.stack([candidates[index].candidate.exact_intensity for index in representative_indices])
        if representative_indices
        else np.empty((0, q_count), dtype=np.float64)
    )
    values = {
        "sobol_local_unit": schedule.points,
        "attempt_seed_index": np.asarray([value.seed_index for value in attempts], dtype=np.int64),
        "attempt_stage": np.asarray([value.stage for value in attempts], dtype=np.str_),
        "attempt_source_id": np.asarray([value.source_id for value in attempts], dtype=np.str_),
        "attempt_status": np.asarray([value.status for value in attempts], dtype=np.str_),
        "attempt_message": np.asarray([value.message for value in attempts], dtype=np.str_),
        "attempt_exact_forward_calls": np.asarray(
            [value.exact_forward_calls for value in attempts], dtype=np.int64
        ),
        "attempt_cumulative_calls_before": np.asarray(
            [value.cumulative_calls_before for value in attempts], dtype=np.int64
        ),
        "attempt_cumulative_calls_after": np.asarray(
            [value.cumulative_calls_after for value in attempts], dtype=np.int64
        ),
        "attempt_calls_by_phase": np.asarray(
            [value.calls_by_phase for value in attempts], dtype=np.int64
        ).reshape((-1, len(EXACT_FORWARD_PHASES))),
        "attempt_candidate_index": np.asarray(
            [value.candidate_index for value in attempts], dtype=np.int64
        ),
        "attempt_initial_local_unit": np.asarray(
            [value.initial_local_unit for value in attempts], dtype=np.float64
        ).reshape((-1, UNIT_CUBE_DIMENSIONS)),
        "candidate_seed_index": np.asarray(
            [value.seed_index for value in candidates], dtype=np.int64
        ),
        "candidate_stage": np.asarray([value.stage for value in candidates], dtype=np.str_),
        "candidate_local_unit": np.asarray(
            [value.local_unit for value in candidates], dtype=np.float64
        ).reshape((-1, UNIT_CUBE_DIMENSIONS)),
        "candidate_raw_log_rmse": np.asarray(
            [value.raw_log_rmse for value in candidates], dtype=np.float64
        ),
        "candidate_standardized_metric_available": np.asarray(
            [value.standardized_log_rmse is not None for value in candidates], dtype=np.bool_
        ),
        "candidate_standardized_log_rmse": np.asarray(
            [
                0.0 if value.standardized_log_rmse is None else value.standardized_log_rmse
                for value in candidates
            ],
            dtype=np.float64,
        ),
        "candidate_selected_metric_value": np.asarray(
            [value.selected_metric_value for value in candidates], dtype=np.float64
        ),
        "candidate_bounds_pass": np.asarray(
            [value.candidate.bounds_pass for value in candidates], dtype=np.bool_
        ),
        "candidate_physics_pass": np.asarray(
            [value.candidate.physics_pass for value in candidates], dtype=np.bool_
        ),
        "candidate_compatible": np.asarray(
            [_compatible(value, threshold) for value in candidates], dtype=np.bool_
        ),
        "candidate_exact_intensity_sha256": np.asarray(
            [value.exact_intensity_sha256 for value in candidates], dtype=np.str_
        ),
        "candidate_background": np.asarray(
            [value.candidate.linear_solution.background for value in candidates], dtype=np.float64
        ),
        "candidate_k": np.asarray(
            [value.candidate.linear_solution.k for value in candidates], dtype=np.float64
        ),
        "candidate_particle_count": np.asarray(
            [len(value.candidate.components) for value in candidates], dtype=np.int8
        ),
        "candidate_particle_amplitudes": particle_amplitudes,
        "candidate_resolution_present": np.asarray(
            [value.candidate.resolution is not None for value in candidates], dtype=np.bool_
        ),
        "candidate_resolution_sigma": np.asarray(
            [
                0.0 if value.candidate.resolution is None else value.candidate.resolution.sigma_res
                for value in candidates
            ],
            dtype=np.float64,
        ),
        "candidate_resolution_nu": np.asarray(
            [
                0.0 if value.candidate.resolution is None else value.candidate.resolution.nu_res
                for value in candidates
            ],
            dtype=np.float64,
        ),
        "candidate_resolution_amplitude": np.asarray(
            [value.candidate.linear_solution.resolution_amplitude for value in candidates],
            dtype=np.float64,
        ),
        "representative_candidate_index": np.asarray(representative_indices, dtype=np.int64),
        "representative_exact_intensity": representative_curves,
    }
    return MappingProxyType(values)


def _source_sha256() -> dict[str, str]:
    posterior_root = Path(__file__).resolve().parent
    repository_root = posterior_root.parents[2]
    sources = sorted(posterior_root.glob("*.py"))
    sources.extend(
        (
            repository_root / "src/gimap/features/fitting/domain/scattering_model.py",
            repository_root / "src/gimap/features/fitting/domain/physical_constraints.py",
        )
    )
    return {
        path.relative_to(repository_root).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sources
    }


def _build_manifest(
    task: V5FrozenSearchTask,
    schedule: V5FrozenLocalSobolSchedule,
    optimizer: V5FrozenExactOptimizerSchedule,
    *,
    artifact_id: str,
    outcome: str,
    completed: bool,
    termination_reason: str,
    failure: Mapping[str, object] | None,
    attempts: Sequence[_AttemptRecord],
    candidates: Sequence[_CandidateRecord],
    representatives: Sequence[V5CompatibleRepresentativeReference],
    representative_payloads: Sequence[Mapping[str, object]],
    arrays: Mapping[str, np.ndarray],
    calls_used: int,
    scout_completed: bool,
) -> dict[str, object]:
    phase_totals = [
        sum(value.calls_by_phase[index] for value in attempts)
        for index in range(len(EXACT_FORWARD_PHASES))
    ]
    core = {
        "container_schema": V5_CHECKED_ARRAY_ARTIFACT_SCHEMA,
        "container_version": V5_CHECKED_ARRAY_ARTIFACT_VERSION,
        "artifact_schema": V5_EXACT_SEARCH_EXECUTOR_SCHEMA,
        "artifact_version": V5_EXACT_SEARCH_EXECUTOR_VERSION,
        "artifact_id": artifact_id,
        "task": {
            "query_index": task.query_index,
            "clean_group_id": task.clean_group_id,
            "recipe_id": task.recipe_id,
            "observation_id": task.observation_id,
            "exact_curve_sha256": task.exact_curve_sha256,
            "exact_observation_audit_sha256": task.exact_observation.audit_sha256,
            "query_catalog_artifact_id": task.query_catalog_artifact_id,
            "query_catalog_artifact_sha256": task.query_catalog_artifact_sha256,
            "universal_query_sha256": task.universal_context.audit_sha256,
            "global_branch_key": task.branch.global_key.wire_key,
            "context_sha256": task.branch.context_sha256,
            "candidate_query_sha256": task.universal_context.batches[
                task.branch.topology_batch_index
            ].query_sha256,
            "amplitude_constraint_sha256": task.branch.amplitude_constraint_sha256,
            "task_audit_sha256": task.audit_sha256,
            "calibrated_threshold_sha256": (
                None if task.calibrated_threshold is None else task.calibrated_threshold.sha256
            ),
        },
        "protocol": task.protocol.audit_payload(),
        "protocol_sha256": task.protocol.sha256,
        "seed_schedule": schedule.audit_payload(),
        "seed_schedule_sha256": schedule.sha256,
        "optimizer_schedule": optimizer.audit_payload(),
        "optimizer_schedule_sha256": optimizer.sha256,
        "selected_metric_name": task.selected_metric_name,
        "selected_threshold_name": task.selected_threshold_name,
        "selected_threshold_value": task.selected_threshold_value,
        "selected_threshold_source_id": task.selected_threshold_source_id,
        "outcome": outcome,
        "completed": completed,
        "termination_reason": termination_reason,
        "completion_rule": V5_EXACT_SEARCH_COMPLETION_RULE,
        "direct_scout_schedule_completed": scout_completed,
        "positive_early_stop_used": False,
        "uses_neural_proposal_scores": False,
        "failure": None if failure is None else dict(failure),
        "ledger": {
            "exact_forward_call_budget": task.protocol.exact_forward_call_budget,
            "exact_forward_calls_used": calls_used,
            "exact_forward_calls_remaining": task.protocol.exact_forward_call_budget - calls_used,
            "attempt_count": len(attempts),
            "candidate_count": len(candidates),
            "compatible_candidate_count": sum(
                _compatible(value, task.selected_threshold_value) for value in candidates
            ),
            "representative_count": len(representatives),
            "phase_order": list(EXACT_FORWARD_PHASES),
            "calls_by_phase": phase_totals,
        },
        "representative_distance_contract": _REPRESENTATIVE_DISTANCE_PAYLOAD,
        "representatives": [value.audit_payload() for value in representatives],
        "representative_payloads": list(representative_payloads),
        "versions": {
            "branch_codec": BRANCH_CODEC_VERSION,
            "gui_amplitude_constraint": GUI_AMPLITUDE_CONSTRAINT_VERSION,
            "exact_refinement": V5_EXACT_REFINEMENT_VERSION,
            "profiled_amplitude_solver": PROFILED_AMPLITUDE_SOLVER_VERSION,
            "query_parameter_distance": V5_QUERY_PARAMETER_DISTANCE_VERSION,
            "evaluation": EVALUATION_AUDIT_SCHEMA,
            "forward_model": FORWARD_MODEL_VERSION,
            "numeric_policy": task.codec.numeric_policy_version,
            "numeric_policy_sha256": v5_numeric_policy_sha256(
                task.codec.numeric_policy_version
            ),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "source_sha256": _source_sha256(),
        "arrays": array_manifest(arrays),
    }
    return {**core, "manifest_sha256": sha256(canonical_json(core).encode("utf-8")).hexdigest()}


def execute_v5_frozen_branch_search(
    task: V5FrozenSearchTask,
    *,
    seed_schedule: V5FrozenLocalSobolSchedule,
    optimizer_schedule: V5FrozenExactOptimizerSchedule,
    artifact_path: str | Path,
) -> V5FrozenBranchSearchResult:
    """Run one full-budget branch search and exclusively publish its evidence."""

    _validate_executor_binding(task, seed_schedule, optimizer_schedule)
    artifact_id = f"v5-exact-search/{_artifact_identity(task, seed_schedule, optimizer_schedule)}"
    attempts: list[_AttemptRecord] = []
    candidates: list[_CandidateRecord] = []
    calls_used = 0
    scout_completed = False
    failure: dict[str, object] | None = None

    try:
        scout_candidates: dict[int, _CandidateRecord] = {}
        for seed_index in range(optimizer_schedule.direct_scout_seed_count):
            attempt, candidate, seed_failure = _run_one_seed(
                task,
                seed_schedule,
                optimizer_schedule,
                seed_index=seed_index,
                stage="direct_scout",
                allowance=1,
                cumulative_calls=calls_used,
                candidate_count=len(candidates),
            )
            attempts.append(attempt)
            calls_used = attempt.cumulative_calls_after
            if seed_failure is not None:
                raise RuntimeError(seed_failure)
            if candidate is None:
                raise RuntimeError("direct Sobol scout did not return an exact candidate")
            candidates.append(candidate)
            scout_candidates[seed_index] = candidate
        scout_completed = True

        ranked_scouts = sorted(
            scout_candidates,
            key=lambda index: (scout_candidates[index].selected_metric_value, index),
        )
        execution_order = tuple(ranked_scouts) + tuple(
            range(optimizer_schedule.direct_scout_seed_count, seed_schedule.point_count)
        )
        for position, seed_index in enumerate(execution_order):
            if calls_used == task.protocol.exact_forward_call_budget:
                break
            remaining = task.protocol.exact_forward_call_budget - calls_used
            allowance = min(
                optimizer_schedule.per_seed_forward_evaluation_limit,
                remaining,
            )
            stage = (
                "exact_ranked_scout_refinement"
                if position < len(ranked_scouts)
                else "sobol_continuation_refinement"
            )
            attempt, candidate, seed_failure = _run_one_seed(
                task,
                seed_schedule,
                optimizer_schedule,
                seed_index=seed_index,
                stage=stage,
                allowance=allowance,
                cumulative_calls=calls_used,
                candidate_count=len(candidates),
            )
            attempts.append(attempt)
            calls_used = attempt.cumulative_calls_after
            if seed_failure is not None:
                raise RuntimeError(seed_failure)
            if candidate is not None:
                candidates.append(candidate)
        if calls_used != task.protocol.exact_forward_call_budget:
            raise RuntimeError("frozen seed/optimizer schedule ended before its exact budget")
    except Exception as exc:  # fail closed while preserving a finite audit artifact
        failure = {
            "exception_type": type(exc).__name__,
            "message": str(exc)[:1000],
        }

    completed = bool(
        failure is None
        and scout_completed
        and calls_used == task.protocol.exact_forward_call_budget
    )
    if completed:
        has_compatible = any(
            _compatible(value, task.selected_threshold_value) for value in candidates
        )
        outcome = (
            "compatible_found"
            if has_compatible
            else "no_compatible_found_within_frozen_search_budget"
        )
        termination_reason = V5_EXACT_SEARCH_TERMINATION_REASONS[outcome]
        references, representative_indices, representative_payloads = _representatives(
            task, artifact_id, candidates
        )
    else:
        outcome = "unverified"
        termination_reason = "executor_failed_before_frozen_protocol_completion"
        references = ()
        representative_indices = ()
        representative_payloads = []
    arrays = _arrays(
        seed_schedule,
        attempts,
        candidates,
        representative_indices,
        threshold=task.selected_threshold_value,
        q_count=task.observed_curve.q.size,
    )
    manifest = _build_manifest(
        task,
        seed_schedule,
        optimizer_schedule,
        artifact_id=artifact_id,
        outcome=outcome,
        completed=completed,
        termination_reason=termination_reason,
        failure=failure,
        attempts=attempts,
        candidates=candidates,
        representatives=references,
        representative_payloads=representative_payloads,
        arrays=arrays,
        calls_used=calls_used,
        scout_completed=scout_completed,
    )
    receipt = write_checked_array_artifact(
        artifact_path,
        manifest=manifest,
        arrays=arrays,
    )
    verified = read_v5_exact_search_executor_artifact(artifact_path, task=task)
    if verified.receipt.artifact_sha256 != receipt.artifact_sha256:
        raise RuntimeError("published exact-search artifact receipt did not round-trip")
    return V5FrozenBranchSearchResult(
        universal_query_sha256=task.universal_context.audit_sha256,
        exact_curve_sha256=task.exact_curve_sha256,
        global_branch_key=task.branch.global_key,
        context_sha256=task.branch.context_sha256,
        outcome=outcome,
        completed=completed,
        exact_forward_calls_used=calls_used,
        termination_reason=termination_reason,
        executor_artifact_id=artifact_id,
        executor_artifact_sha256=receipt.artifact_sha256,
        representatives=tuple(references),
    )


def _expect_array_shapes(manifest: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> None:
    ledger = manifest["ledger"]
    if not isinstance(ledger, Mapping):
        raise ValueError("executor ledger is missing")
    attempts = int(ledger["attempt_count"])
    candidates = int(ledger["candidate_count"])
    representatives = int(ledger["representative_count"])
    expected_first = {
        "attempt_seed_index": attempts,
        "attempt_stage": attempts,
        "attempt_source_id": attempts,
        "attempt_status": attempts,
        "attempt_message": attempts,
        "attempt_exact_forward_calls": attempts,
        "attempt_cumulative_calls_before": attempts,
        "attempt_cumulative_calls_after": attempts,
        "attempt_calls_by_phase": attempts,
        "attempt_candidate_index": attempts,
        "attempt_initial_local_unit": attempts,
        "candidate_seed_index": candidates,
        "candidate_stage": candidates,
        "candidate_local_unit": candidates,
        "candidate_raw_log_rmse": candidates,
        "candidate_standardized_metric_available": candidates,
        "candidate_standardized_log_rmse": candidates,
        "candidate_selected_metric_value": candidates,
        "candidate_bounds_pass": candidates,
        "candidate_physics_pass": candidates,
        "candidate_compatible": candidates,
        "candidate_exact_intensity_sha256": candidates,
        "candidate_background": candidates,
        "candidate_k": candidates,
        "candidate_particle_count": candidates,
        "candidate_particle_amplitudes": candidates,
        "candidate_resolution_present": candidates,
        "candidate_resolution_sigma": candidates,
        "candidate_resolution_nu": candidates,
        "candidate_resolution_amplitude": candidates,
        "representative_candidate_index": representatives,
        "representative_exact_intensity": representatives,
    }
    if set(arrays) != {"sobol_local_unit", *expected_first}:
        raise ValueError("executor artifact array inventory is incomplete or unsupported")
    if any(arrays[name].shape[0] != count for name, count in expected_first.items()):
        raise ValueError("executor artifact table dimensions disagree with its ledger")
    if arrays["attempt_calls_by_phase"].shape != (attempts, len(EXACT_FORWARD_PHASES)):
        raise ValueError("executor attempt phase ledger has the wrong shape")
    if arrays["attempt_initial_local_unit"].shape != (attempts, UNIT_CUBE_DIMENSIONS):
        raise ValueError("executor attempt local-unit table has the wrong shape")
    if arrays["candidate_local_unit"].shape != (candidates, UNIT_CUBE_DIMENSIONS):
        raise ValueError("executor candidate local-unit table has the wrong shape")
    if arrays["candidate_particle_amplitudes"].shape != (candidates, MAX_COMPONENTS):
        raise ValueError("executor particle-amplitude table has the wrong shape")


def _replay_candidate_evidence(
    task: V5FrozenSearchTask,
    arrays: Mapping[str, np.ndarray],
) -> tuple[_CandidateRecord, ...]:
    """Rebuild every candidate curve and metric from its compact snapshot."""

    sigma_available = task.observed_curve.sigma_log is not None
    expects_standardized = task.selected_metric_name == STANDARDIZED_LOG_RMSE_METRIC
    if sigma_available != expects_standardized:
        raise ValueError("task selected metric disagrees with acceptance-sigma presence")
    if not np.all(arrays["candidate_bounds_pass"]):
        raise ValueError("executor persisted a candidate that failed its owning codec bounds")
    if not np.all(arrays["candidate_physics_pass"]):
        raise ValueError("executor persisted a candidate that failed its physics prerequisites")

    reconstructed: list[_CandidateRecord] = []
    selected_values: list[float] = []
    coefficients = arrays["candidate_particle_amplitudes"]
    for index, local in enumerate(arrays["candidate_local_unit"]):
        components, resolution = task.codec.decode(local)
        count = len(components)
        if int(arrays["candidate_particle_count"][index]) != count:
            raise ValueError("executor candidate particle count disagrees with its local codec")
        if np.any(coefficients[index, count:] != 0.0):
            raise ValueError("executor candidate has nonzero amplitudes in inactive slots")
        resolution_present = resolution is not None
        if bool(arrays["candidate_resolution_present"][index]) != resolution_present:
            raise ValueError(
                "executor candidate resolution presence disagrees with its local codec"
            )
        expected_sigma = 0.0 if resolution is None else resolution.sigma_res
        expected_nu = 0.0 if resolution is None else resolution.nu_res
        if not np.isclose(
            arrays["candidate_resolution_sigma"][index],
            expected_sigma,
            rtol=1.0e-12,
            atol=1.0e-12,
        ) or not np.isclose(
            arrays["candidate_resolution_nu"][index],
            expected_nu,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("executor candidate resolution snapshot does not replay")

        linear = LinearSolutionSnapshot(
            background=arrays["candidate_background"][index],
            particle_amplitudes=tuple(coefficients[index, :count]),
            resolution_amplitude=arrays["candidate_resolution_amplitude"][index],
            k=arrays["candidate_k"][index],
        )
        if resolution is None and linear.resolution_amplitude != 0.0:
            raise ValueError("resolution-absent candidate has a nonzero resolution amplitude")
        coefficient_vector = (
            linear.background,
            *linear.particle_amplitudes,
            *(() if resolution is None else (linear.resolution_amplitude,)),
        )
        if not task.amplitude_constraint.contains(coefficient_vector, k=linear.k):
            raise ValueError("executor candidate escaped its concrete frozen GUI ranges")

        exact = evaluate_gui_forward_snapshot(
            task.observed_curve.q,
            task.codec.latent_components_to_gui(components),
            resolution=resolution,
            background=linear.background,
            particle_amplitudes=linear.particle_amplitudes,
            resolution_amplitude=linear.resolution_amplitude,
            gui_k=linear.k,
        )
        if (
            array_sha256("candidate_exact_intensity", exact)
            != arrays["candidate_exact_intensity_sha256"][index]
        ):
            raise ValueError("executor candidate exact GUI curve digest does not replay")

        raw = natural_log_rmse(exact, task.observed_curve.intensity)
        standardized = (
            natural_log_rmse(
                exact,
                task.observed_curve.intensity,
                sigma_log=task.observed_curve.sigma_log,
            )
            if sigma_available
            else None
        )
        available = bool(arrays["candidate_standardized_metric_available"][index])
        if available != sigma_available:
            raise ValueError("executor standardized-metric availability does not replay")
        if not np.isclose(arrays["candidate_raw_log_rmse"][index], raw, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("executor candidate raw exact metric does not replay")
        stored_standardized = arrays["candidate_standardized_log_rmse"][index]
        if standardized is None:
            if stored_standardized != 0.0:
                raise ValueError("executor absent standardized metric must use zero sentinel")
            selected = raw
        else:
            if not np.isclose(
                stored_standardized,
                standardized,
                rtol=1.0e-12,
                atol=1.0e-12,
            ):
                raise ValueError("executor candidate standardized exact metric does not replay")
            selected = standardized
        if not np.isclose(
            arrays["candidate_selected_metric_value"][index],
            selected,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError("executor candidate selected exact metric does not replay")
        candidate = CandidateInput(
            candidate_id=f"search_candidate_{index + 1:06d}",
            proposal_rank=index + 1,
            topology_id=task.codec.topology_id,
            components=components,
            resolution=resolution,
            linear_solution=linear,
            exact_intensity=exact,
            bounds_pass=True,
            physics_pass=True,
        )
        reconstructed.append(
            _CandidateRecord(
                candidate=candidate,
                local_unit=tuple(float(value) for value in local),
                seed_index=int(arrays["candidate_seed_index"][index]),
                stage=str(arrays["candidate_stage"][index]),
                raw_log_rmse=raw,
                standardized_log_rmse=standardized,
                selected_metric_value=selected,
                exact_intensity_sha256=array_sha256("candidate_exact_intensity", exact),
            )
        )
        selected_values.append(selected)

    compatible = np.asarray(selected_values, dtype=np.float64) <= task.selected_threshold_value
    if not np.array_equal(arrays["candidate_compatible"], compatible):
        raise ValueError("executor compatibility flags disagree with replayed exact evidence")
    return tuple(reconstructed)


def read_v5_exact_search_executor_artifact(
    path: str | Path,
    *,
    task: V5FrozenSearchTask | None = None,
) -> V5ExactSearchExecutorArtifact:
    """Read, hash-check, and scientifically replay the compact branch evidence."""

    manifest, arrays, receipt = read_checked_array_artifact(path)
    if manifest.get("artifact_schema") != V5_EXACT_SEARCH_EXECUTOR_SCHEMA:
        raise ValueError("unsupported exact-search executor artifact schema")
    if manifest.get("artifact_version") != V5_EXACT_SEARCH_EXECUTOR_VERSION:
        raise ValueError("unsupported exact-search executor artifact version")
    schedule_payload = manifest.get("seed_schedule")
    optimizer_payload = manifest.get("optimizer_schedule")
    protocol_payload = manifest.get("protocol")
    if not all(
        isinstance(value, Mapping)
        for value in (schedule_payload, optimizer_payload, protocol_payload)
    ):
        raise ValueError("executor artifact contracts are incomplete")
    schedule = V5FrozenLocalSobolSchedule(
        schedule_id=schedule_payload["schedule_id"],
        base_seed=schedule_payload["base_seed"],
        points=arrays["sobol_local_unit"],
        scipy_version=schedule_payload["scipy_version"],
        engine=schedule_payload["engine"],
        scramble=schedule_payload["scramble"],
        ordering_policy=schedule_payload["ordering_policy"],
        schema_version=schedule_payload["schema_version"],
        version=schedule_payload["version"],
    )
    optimizer = V5FrozenExactOptimizerSchedule(**dict(optimizer_payload))
    protocol = V5FrozenExactSearchProtocol(**dict(protocol_payload))
    if schedule.audit_payload() != dict(schedule_payload) or schedule.sha256 != manifest.get(
        "seed_schedule_sha256"
    ):
        raise ValueError("executor Sobol schedule does not reproduce")
    if optimizer.sha256 != manifest.get("optimizer_schedule_sha256"):
        raise ValueError("executor optimizer schedule does not reproduce")
    if protocol.sha256 != manifest.get("protocol_sha256"):
        raise ValueError("executor frozen protocol does not reproduce")
    persisted_versions = manifest.get("versions")
    if not isinstance(persisted_versions, Mapping):
        raise ValueError("executor scientific runtime versions are incomplete")
    try:
        numeric_policy = validate_v5_numeric_policy(
            persisted_versions.get("numeric_policy")
        )
    except ValueError as exc:
        raise ValueError("executor numeric policy is unsupported") from exc
    if task is not None and numeric_policy != task.codec.numeric_policy_version:
        raise ValueError("executor numeric policy disagrees with the replay task")
    expected_versions = {
        "branch_codec": BRANCH_CODEC_VERSION,
        "gui_amplitude_constraint": GUI_AMPLITUDE_CONSTRAINT_VERSION,
        "exact_refinement": V5_EXACT_REFINEMENT_VERSION,
        "profiled_amplitude_solver": PROFILED_AMPLITUDE_SOLVER_VERSION,
        "query_parameter_distance": V5_QUERY_PARAMETER_DISTANCE_VERSION,
        "evaluation": EVALUATION_AUDIT_SCHEMA,
        "forward_model": FORWARD_MODEL_VERSION,
        "numeric_policy": numeric_policy,
        "numeric_policy_sha256": v5_numeric_policy_sha256(numeric_policy),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if persisted_versions != expected_versions:
        raise ValueError("executor scientific runtime versions do not reproduce")
    if manifest.get("source_sha256") != _source_sha256():
        raise ValueError("executor scientific source digests do not reproduce")
    _expect_array_shapes(manifest, arrays)
    ledger = manifest["ledger"]
    calls = arrays["attempt_exact_forward_calls"]
    if int(np.sum(calls)) != ledger["exact_forward_calls_used"]:
        raise ValueError("executor exact-forward call ledger does not add up")
    before = arrays["attempt_cumulative_calls_before"]
    after = arrays["attempt_cumulative_calls_after"]
    if before.size and (
        before[0] != 0
        or not np.array_equal(after - before, calls)
        or not np.array_equal(before[1:], after[:-1])
        or after[-1] != ledger["exact_forward_calls_used"]
    ):
        raise ValueError("executor cumulative exact-forward ledger is not contiguous")
    phase_totals = np.sum(arrays["attempt_calls_by_phase"], axis=0, dtype=np.int64)
    if phase_totals.tolist() != ledger["calls_by_phase"] or int(np.sum(phase_totals)) != int(
        np.sum(calls)
    ):
        raise ValueError("executor phase and total exact-forward ledgers disagree")
    completed = manifest.get("completed")
    if type(completed) is not bool:
        raise ValueError("executor completion flag must be boolean")
    if completed:
        if (
            ledger["exact_forward_calls_used"] != protocol.exact_forward_call_budget
            or not manifest.get("direct_scout_schedule_completed")
            or manifest.get("failure") is not None
        ):
            raise ValueError("completed executor artifact did not finish the frozen budget")
        expected_reason = V5_EXACT_SEARCH_TERMINATION_REASONS.get(manifest.get("outcome"))
        if manifest.get("termination_reason") != expected_reason:
            raise ValueError("executor outcome and termination policy disagree")
        compatible_count = int(ledger["compatible_candidate_count"])
        representative_count = int(ledger["representative_count"])
        if manifest.get("outcome") == "compatible_found":
            if compatible_count < 1 or representative_count < 1:
                raise ValueError("positive executor artifact has no compatible representative")
        elif compatible_count or representative_count:
            raise ValueError("completed negative carries compatible candidate evidence")
    elif manifest.get("outcome") != "unverified":
        raise ValueError("incomplete executor artifact cannot carry a training label")
    if (
        manifest.get("uses_neural_proposal_scores") is not False
        or manifest.get("positive_early_stop_used") is not False
    ):
        raise ValueError("executor artifact used a forbidden learned score or early stop")
    references = manifest.get("representatives")
    payloads = manifest.get("representative_payloads")
    if (
        not isinstance(references, list)
        or not isinstance(payloads, list)
        or len(references) != len(payloads)
    ):
        raise ValueError("executor representative subartifacts are incomplete")
    payload_hashes = []
    for reference, payload_with_hash in zip(references, payloads):
        if not isinstance(reference, Mapping) or not isinstance(payload_with_hash, Mapping):
            raise ValueError("executor representative subartifact is malformed")
        payload = dict(payload_with_hash)
        declared = payload.pop("artifact_sha256", None)
        observed = sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        if declared != observed or reference.get("artifact_sha256") != observed:
            raise ValueError("executor representative subartifact digest does not reproduce")
        for name in (
            "artifact_id",
            "cluster_id",
            "metric_value",
            "bounds_passed",
            "physics_passed",
            "target_local",
        ):
            if reference.get(name) != payload.get(name):
                raise ValueError("executor representative reference and payload disagree")
        payload_hashes.append(observed)
    if references:
        set_ids = {value.get("representative_set_id") for value in references}
        set_hashes = {value.get("representative_set_sha256") for value in references}
        if len(set_ids) != 1 or len(set_hashes) != 1:
            raise ValueError(
                "executor representatives do not share one complete-linkage diameter-cluster set"
            )
        set_payload = {
            "representative_set_id": next(iter(set_ids)),
            "distance_id": protocol.representative_distance_id,
            "distance_sha256": protocol.representative_distance_sha256,
            "delta": protocol.delta_separation_threshold,
            "linkage": protocol.clustering_linkage,
            "representative_artifact_sha256": payload_hashes,
        }
        if sha256(canonical_json(set_payload).encode("utf-8")).hexdigest() != next(
            iter(set_hashes)
        ):
            raise ValueError("executor representative-set digest does not reproduce")
    if task is not None:
        _validate_executor_binding(task, schedule, optimizer)
        task_payload = manifest.get("task")
        if not isinstance(task_payload, Mapping):
            raise ValueError("executor task identity is missing")
        expected_task = {
            "query_index": task.query_index,
            "clean_group_id": task.clean_group_id,
            "recipe_id": task.recipe_id,
            "observation_id": task.observation_id,
            "exact_curve_sha256": task.exact_curve_sha256,
            "exact_observation_audit_sha256": task.exact_observation.audit_sha256,
            "query_catalog_artifact_id": task.query_catalog_artifact_id,
            "query_catalog_artifact_sha256": task.query_catalog_artifact_sha256,
            "universal_query_sha256": task.universal_context.audit_sha256,
            "global_branch_key": task.branch.global_key.wire_key,
            "context_sha256": task.branch.context_sha256,
            "candidate_query_sha256": task.universal_context.batches[
                task.branch.topology_batch_index
            ].query_sha256,
            "amplitude_constraint_sha256": task.branch.amplitude_constraint_sha256,
            "task_audit_sha256": task.audit_sha256,
            "calibrated_threshold_sha256": (
                None if task.calibrated_threshold is None else task.calibrated_threshold.sha256
            ),
        }
        if any(task_payload.get(name) != value for name, value in expected_task.items()):
            raise ValueError("executor artifact belongs to a different task")
        if protocol.sha256 != task.protocol.sha256:
            raise ValueError("executor artifact belongs to a different frozen protocol")
        expected_selection = {
            "selected_metric_name": task.selected_metric_name,
            "selected_threshold_name": task.selected_threshold_name,
            "selected_threshold_value": task.selected_threshold_value,
            "selected_threshold_source_id": task.selected_threshold_source_id,
        }
        if any(manifest.get(name) != value for name, value in expected_selection.items()):
            raise ValueError("executor artifact threshold binding does not reproduce")
        reconstructed = _replay_candidate_evidence(task, arrays)
        compatible = arrays["candidate_compatible"]
        if int(np.count_nonzero(compatible)) != ledger["compatible_candidate_count"]:
            raise ValueError("executor compatible-candidate ledger does not replay")
        representative_indices = arrays["representative_candidate_index"]
        representative_curves = arrays["representative_exact_intensity"]
        if len(references) != len(representative_indices):
            raise ValueError("executor representative references are incomplete")
        artifact_id = manifest.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError("executor artifact identity is missing")
        if completed:
            replay_references, replay_indices, replay_payloads = _representatives(
                task, artifact_id, reconstructed
            )
            if [value.audit_payload() for value in replay_references] != references:
                raise ValueError("executor representative references do not replay")
            if canonical_json(replay_payloads) != canonical_json(payloads) or not np.array_equal(
                representative_indices, np.asarray(replay_indices, dtype=np.int64)
            ):
                raise ValueError("executor complete-linkage representative set does not replay")
        for row, candidate_index in enumerate(representative_indices):
            candidate_index = int(candidate_index)
            if not 0 <= candidate_index < len(reconstructed):
                raise ValueError("executor representative candidate index is outside its table")
            if not compatible[candidate_index]:
                raise ValueError(
                    "executor representative does not reference a compatible candidate"
                )
            curve = representative_curves[row]
            if not np.array_equal(curve, reconstructed[candidate_index].candidate.exact_intensity):
                raise ValueError("executor representative curve disagrees with candidate replay")
            metric = arrays["candidate_selected_metric_value"][candidate_index]
            if not np.isclose(metric, references[row]["metric_value"], rtol=1e-12, atol=1e-12):
                raise ValueError("executor representative metric does not replay")
    return V5ExactSearchExecutorArtifact(manifest, arrays, receipt)


@dataclass(frozen=True, eq=False, kw_only=True)
class V5FrozenExactSearchExecutor:
    """Callable adapter accepted directly by the search-sidecar collector."""

    output_directory: Path
    seed_schedule: V5FrozenLocalSobolSchedule
    optimizer_schedule: V5FrozenExactOptimizerSchedule

    def __post_init__(self) -> None:
        root = Path(self.output_directory)
        if not root.is_dir():
            raise FileNotFoundError(f"exact-search output directory does not exist: {root}")
        object.__setattr__(self, "output_directory", root)
        if not isinstance(self.seed_schedule, V5FrozenLocalSobolSchedule):
            raise TypeError("seed_schedule must be a V5FrozenLocalSobolSchedule")
        if not isinstance(self.optimizer_schedule, V5FrozenExactOptimizerSchedule):
            raise TypeError("optimizer_schedule must be a V5FrozenExactOptimizerSchedule")

    def __call__(self, task: V5FrozenSearchTask) -> V5FrozenBranchSearchResult:
        path = v5_exact_search_artifact_path(
            self.output_directory,
            task,
            self.seed_schedule,
            self.optimizer_schedule,
        )
        return execute_v5_frozen_branch_search(
            task,
            seed_schedule=self.seed_schedule,
            optimizer_schedule=self.optimizer_schedule,
            artifact_path=path,
        )

    def evidence_binding(
        self,
        task: V5FrozenSearchTask,
        result: V5FrozenBranchSearchResult,
    ) -> Mapping[str, object]:
        """Return a replayed, portable index entry for the sidecar/receipt chain."""

        path = v5_exact_search_artifact_path(
            self.output_directory,
            task,
            self.seed_schedule,
            self.optimizer_schedule,
        )
        artifact = read_v5_exact_search_executor_artifact(path, task=task)
        if (
            artifact.manifest["artifact_id"] != result.executor_artifact_id
            or artifact.receipt.artifact_sha256 != result.executor_artifact_sha256
        ):
            raise ValueError("executor evidence binding disagrees with its runner result")
        source_hashes = artifact.manifest["source_sha256"]
        return {
            "relative_path": path.relative_to(self.output_directory).as_posix(),
            "artifact_schema": artifact.manifest["artifact_schema"],
            "artifact_version": artifact.manifest["artifact_version"],
            "source_bundle_sha256": sha256(
                canonical_json(source_hashes).encode("utf-8")
            ).hexdigest(),
        }


__all__ = [
    "V5_EXACT_SEARCH_ARTIFACT_SUFFIX",
    "V5_EXACT_SEARCH_COMPLETION_RULE",
    "V5_EXACT_SEARCH_EVALUATOR_VERSION",
    "V5_EXACT_SEARCH_EXECUTOR_SCHEMA",
    "V5_EXACT_SEARCH_EXECUTOR_VERSION",
    "V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_ID",
    "V5_EXACT_SEARCH_REPRESENTATIVE_DISTANCE_SHA256",
    "V5ExactSearchExecutorArtifact",
    "V5FrozenExactSearchExecutor",
    "build_v5_frozen_exact_search_protocol",
    "execute_v5_frozen_branch_search",
    "read_v5_exact_search_executor_artifact",
    "v5_exact_search_artifact_path",
]
