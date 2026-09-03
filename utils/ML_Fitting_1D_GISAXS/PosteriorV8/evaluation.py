"""Versioned acceptance, clustering, and audit metrics for Posterior V8.

Scores stay opaque; exact curves plus explicit bounds and physics flags gate acceptance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import permutations
import json
from numbers import Integral
from typing import Callable, Literal, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contract import (
    CODEC_VERSION,
    CONTRACT_VERSION,
    CYLINDER,
    D_DOMAIN,
    D_WIDTH_FRACTION_DOMAIN,
    H_DOMAIN,
    R_DOMAIN,
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    SIZE_WIDTH_FRACTION_DOMAIN,
    FORWARD_MODEL_VERSION,
    LatentComponentParameters,
    topology_from_id,
)
from .profiled_forward import ProfiledForwardResult, ResolutionShape


EVALUATION_AUDIT_SCHEMA = "gisaxs.posterior_v8.evaluation/v3"
PARAMETER_NORMALIZATION_VERSION = "posterior_v8_full_geometry_and_profiled_composition_rms/v4"
CLUSTERING_LINKAGE = "deterministic_agglomerative_complete_linkage/v2"
REFERENCE_MATCHING_VERSION = (
    "maximum_cardinality_minimum_distance_bipartite_parameter_mode_matching/v1"
)
REFERENCE_SET_DISTANCE_VERSION = "symmetric_chamfer_and_hausdorff_with_unit_unmatched_penalty/v1"
REFERENCE_SET_UNMATCHED_DISTANCE = 1.0
PROPOSAL_SCORE_SEMANTICS = "opaque_raw_score_not_probability"
RAW_LOG_RMSE_METRIC = "raw_natural_log_rmse"
STANDARDIZED_LOG_RMSE_METRIC = "sigma_log_standardized_natural_log_rmse"
PARAMETER_DISTANCE_SCOPE = (
    "canonical topology + unordered nonlinear component geometry/D presence + "
    "resolution presence/sigma_res/nu_res + mandatory scale-invariant profiled "
    "BG/particle/resolution composition; excludes overall intensity scale"
)


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_nonnegative(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _readonly_positive_1d(value: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain finite, strictly positive values")
    result.setflags(write=False)
    return result


def _validated_components(
    topology_id: int,
    components: Sequence[LatentComponentParameters],
) -> tuple[int, tuple[LatentComponentParameters, ...]]:
    topology = topology_from_id(topology_id)
    result = tuple(components)
    if not all(isinstance(item, LatentComponentParameters) for item in result):
        raise TypeError("components must contain only LatentComponentParameters")
    if tuple(item.shape for item in result) != topology:
        raise ValueError("component shapes/order do not match the canonical topology_id")
    return int(topology_id), result


@dataclass(frozen=True, eq=False, kw_only=True)
class ObservedCurve:
    curve_id: str
    source_kind: str
    q: np.ndarray
    intensity: np.ndarray
    sigma_log: np.ndarray | None = None

    def __post_init__(self) -> None:
        curve_id = _text(self.curve_id, "curve_id")
        source_kind = str(self.source_kind).strip().lower()
        if source_kind not in {"synthetic", "real_cut_data"}:
            raise ValueError("source_kind must be synthetic or real_cut_data")
        q = _readonly_positive_1d(self.q, "q")
        if np.any(np.diff(q) <= 0.0):
            raise ValueError("q must be strictly increasing")
        intensity = _readonly_positive_1d(self.intensity, "intensity")
        if intensity.shape != q.shape:
            raise ValueError("intensity must have the same shape as q")
        sigma_log = None
        if self.sigma_log is not None:
            sigma_log = _readonly_positive_1d(self.sigma_log, "sigma_log")
            if sigma_log.shape != q.shape:
                raise ValueError("sigma_log must have the same shape as q")
        object.__setattr__(self, "curve_id", curve_id)
        object.__setattr__(self, "source_kind", source_kind)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "sigma_log", sigma_log)


@dataclass(frozen=True, eq=False, kw_only=True)
class CandidateInput:
    candidate_id: str
    proposal_rank: int
    topology_id: int
    components: tuple[LatentComponentParameters, ...]
    resolution: ResolutionShape | None
    linear_solution: LinearSolutionSnapshot
    exact_intensity: np.ndarray
    bounds_pass: bool
    physics_pass: bool
    proposal_score_raw: float | None = None

    def __post_init__(self) -> None:
        candidate_id = _text(self.candidate_id, "candidate_id")
        if isinstance(self.proposal_rank, (bool, np.bool_)) or not isinstance(
            self.proposal_rank, Integral
        ):
            raise TypeError("proposal_rank must be a positive integer")
        proposal_rank = int(self.proposal_rank)
        if proposal_rank < 1:
            raise ValueError("proposal_rank must be a positive integer")
        topology_id, components = _validated_components(self.topology_id, self.components)
        if self.resolution is not None and not isinstance(self.resolution, ResolutionShape):
            raise TypeError("resolution must be None or ResolutionShape")
        if not isinstance(self.linear_solution, LinearSolutionSnapshot):
            raise TypeError("linear_solution must be a LinearSolutionSnapshot")
        if len(self.linear_solution.particle_amplitudes) != len(components):
            raise ValueError("linear_solution requires one particle amplitude per component")
        if self.resolution is None and self.linear_solution.resolution_amplitude != 0.0:
            raise ValueError("absent resolution requires zero resolution_amplitude")
        exact_intensity = _readonly_positive_1d(self.exact_intensity, "exact_intensity")
        if type(self.bounds_pass) is not bool or type(self.physics_pass) is not bool:
            raise TypeError("bounds_pass and physics_pass must be explicit booleans")
        score = self.proposal_score_raw
        if score is not None:
            try:
                score = float(score)
            except (TypeError, ValueError) as exc:
                raise ValueError("proposal_score_raw must be finite or None") from exc
            if not np.isfinite(score):
                raise ValueError("proposal_score_raw must be finite or None")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "proposal_rank", proposal_rank)
        object.__setattr__(self, "topology_id", topology_id)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "exact_intensity", exact_intensity)
        object.__setattr__(self, "proposal_score_raw", score)


@dataclass(frozen=True, kw_only=True)
class ReferenceMode:
    reference_id: str
    topology_id: int
    components: tuple[LatentComponentParameters, ...]
    resolution: ResolutionShape | None
    linear_solution: LinearSolutionSnapshot

    def __post_init__(self) -> None:
        topology_id, components = _validated_components(self.topology_id, self.components)
        if self.resolution is not None and not isinstance(self.resolution, ResolutionShape):
            raise TypeError("resolution must be None or ResolutionShape")
        if not isinstance(self.linear_solution, LinearSolutionSnapshot):
            raise TypeError("linear_solution must be a LinearSolutionSnapshot")
        if len(self.linear_solution.particle_amplitudes) != len(components):
            raise ValueError("linear_solution requires one particle amplitude per component")
        if self.resolution is None and self.linear_solution.resolution_amplitude != 0.0:
            raise ValueError("absent resolution requires zero resolution_amplitude")
        object.__setattr__(self, "reference_id", _text(self.reference_id, "reference_id"))
        object.__setattr__(self, "topology_id", topology_id)
        object.__setattr__(self, "components", components)


@dataclass(frozen=True, kw_only=True)
class LinearSolutionSnapshot:
    """Effective amplitudes plus one lossless current GUI parameterization."""

    background: float
    particle_amplitudes: tuple[float, ...]
    resolution_amplitude: float
    k: float | None = None
    component_weights: tuple[float, ...] = field(init=False)
    int_res: float = field(init=False)

    def __post_init__(self) -> None:
        background = _finite_nonnegative(self.background, "background")
        amplitudes = tuple(
            _finite_nonnegative(value, f"particle_amplitudes[{index}]")
            for index, value in enumerate(self.particle_amplitudes)
        )
        if not amplitudes or sum(amplitudes) <= 0.0:
            raise ValueError("particle_amplitudes must have a strictly positive total")
        resolution_amplitude = _finite_nonnegative(
            self.resolution_amplitude, "resolution_amplitude"
        )
        k = float(sum(amplitudes)) if self.k is None else _finite_nonnegative(self.k, "k")
        if k <= 0.0:
            raise ValueError("k must be strictly positive")
        object.__setattr__(self, "background", background)
        object.__setattr__(self, "particle_amplitudes", amplitudes)
        object.__setattr__(self, "resolution_amplitude", resolution_amplitude)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "component_weights", tuple(value / k for value in amplitudes))
        object.__setattr__(self, "int_res", resolution_amplitude / k)

    @property
    def effective_particle_total(self) -> float:
        return float(sum(self.particle_amplitudes))

    @property
    def effective_component_fractions(self) -> tuple[float, ...]:
        total = self.effective_particle_total
        return tuple(float(value / total) for value in self.particle_amplitudes)

    @property
    def effective_resolution_ratio(self) -> float:
        return float(self.resolution_amplitude / self.effective_particle_total)

    @classmethod
    def from_profiled_forward(cls, result: ProfiledForwardResult) -> LinearSolutionSnapshot:
        if not isinstance(result, ProfiledForwardResult):
            raise TypeError("result must be a ProfiledForwardResult")
        return cls(
            background=result.background,
            particle_amplitudes=result.particle_amplitudes,
            resolution_amplitude=result.resolution_amplitude,
            k=result.k,
        )


@dataclass(frozen=True, kw_only=True)
class EvaluationThresholds:
    raw_exact_log_rmse_max: float
    standardized_exact_log_rmse_max: float
    parameter_mode_distance_max: float
    raw_curve_equivalence_log_rmse_max: float
    reference_mode_distance_max: float

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _finite_nonnegative(getattr(self, name), name))


def natural_log_rmse(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
    *,
    sigma_log: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Return RMS natural-log residual, optionally standardized by sigma_log."""
    left = _readonly_positive_1d(first, "first")
    right = _readonly_positive_1d(second, "second")
    if left.shape != right.shape:
        raise ValueError("first and second must have the same shape")
    residual = np.log(left) - np.log(right)
    if sigma_log is not None:
        sigma = _readonly_positive_1d(sigma_log, "sigma_log")
        if sigma.shape != left.shape:
            raise ValueError("sigma_log must have the same shape as the curves")
        residual = residual / sigma
    return float(np.sqrt(np.mean(np.square(residual))))


_LOG_R_SPAN = float(np.log(R_DOMAIN.high / R_DOMAIN.low))
_LOG_H_SPAN = float(np.log(H_DOMAIN.high / H_DOMAIN.low))
_LOG_D_SPAN = float(np.log(D_DOMAIN.high / D_DOMAIN.low))
_SIZE_WIDTH_SPAN = SIZE_WIDTH_FRACTION_DOMAIN.high - SIZE_WIDTH_FRACTION_DOMAIN.low
_D_WIDTH_SPAN = D_WIDTH_FRACTION_DOMAIN.high - D_WIDTH_FRACTION_DOMAIN.low
_LOG_RESOLUTION_SIGMA_SPAN = float(
    np.log(RESOLUTION_SIGMA_DOMAIN.high / RESOLUTION_SIGMA_DOMAIN.low)
)
_RESOLUTION_NU_SPAN = RESOLUTION_NU_DOMAIN.high - RESOLUTION_NU_DOMAIN.low


def _profiled_composition(
    value: CandidateInput | ReferenceMode,
) -> tuple[float, tuple[float, ...], float]:
    solution = value.linear_solution
    if not isinstance(solution, LinearSolutionSnapshot):
        raise TypeError(
            "scale-quotiented parameter distance requires a LinearSolutionSnapshot for both modes"
        )
    total = solution.background + sum(solution.particle_amplitudes) + solution.resolution_amplitude
    return (
        solution.background / total,
        tuple(amplitude / total for amplitude in solution.particle_amplitudes),
        solution.resolution_amplitude / total,
    )


def _component_cost(
    left: LatentComponentParameters,
    right: LatentComponentParameters,
) -> tuple[float, int] | None:
    if left.shape != right.shape or (left.log_D is None) != (right.log_D is None):
        return None
    squared = [
        ((left.log_R - right.log_R) / _LOG_R_SPAN) ** 2,
        ((left.sigma_R_fraction - right.sigma_R_fraction) / _SIZE_WIDTH_SPAN) ** 2,
    ]
    if left.shape == CYLINDER:
        squared.extend(
            [
                ((left.log_h - right.log_h) / _LOG_H_SPAN) ** 2,
                ((left.sigma_h_fraction - right.sigma_h_fraction) / _SIZE_WIDTH_SPAN) ** 2,
            ]
        )
    if left.log_D is not None:
        squared.extend(
            [
                ((left.log_D - right.log_D) / _LOG_D_SPAN) ** 2,
                ((left.sigma_D_fraction - right.sigma_D_fraction) / _D_WIDTH_SPAN) ** 2,
            ]
        )
    return float(np.sum(squared)), len(squared)


def normalized_latent_parameter_distance(
    left: CandidateInput | ReferenceMode,
    right: CandidateInput | ReferenceMode,
) -> float:
    """Global geometry-plus-composition RMS distance in contract-domain units.

    This legacy/global metric intentionally quotients overall intensity scale
    and permits every same-shape slot permutation. Query-bound paper endpoints
    must use ``query_local_parameter_distance`` instead. Missing linear
    solutions fail closed instead of silently switching to geometry only.
    """
    if not isinstance(left, (CandidateInput, ReferenceMode)) or not isinstance(
        right, (CandidateInput, ReferenceMode)
    ):
        raise TypeError("left and right must be CandidateInput or ReferenceMode")
    left_composition = _profiled_composition(left)
    right_composition = _profiled_composition(right)
    if left.topology_id != right.topology_id:
        return float("inf")
    if (left.resolution is None) != (right.resolution is None):
        return float("inf")
    resolution_cost = 0.0
    resolution_dimensions = 0
    if left.resolution is not None:
        resolution_cost = (
            np.log(left.resolution.sigma_res / right.resolution.sigma_res)
            / _LOG_RESOLUTION_SIGMA_SPAN
        ) ** 2 + ((left.resolution.nu_res - right.resolution.nu_res) / _RESOLUTION_NU_SPAN) ** 2
        resolution_dimensions = 2
    best = float("inf")
    for order in permutations(range(len(right.components))):
        if any(
            left.components[index].shape != right.components[other].shape
            for index, other in enumerate(order)
        ):
            continue
        total = resolution_cost + (left_composition[0] - right_composition[0]) ** 2
        dimensions = resolution_dimensions + 1
        if left.resolution is not None:
            total += (left_composition[2] - right_composition[2]) ** 2
            dimensions += 1
        for index, other in enumerate(order):
            cost = _component_cost(left.components[index], right.components[other])
            if cost is None:
                break
            total += cost[0]
            dimensions += cost[1]
            total += (left_composition[1][index] - right_composition[1][other]) ** 2
            dimensions += 1
        else:
            best = min(best, float(np.sqrt(total / dimensions)))
    return best


@dataclass(frozen=True, kw_only=True)
class CandidateAssessment:
    candidate_id: str
    proposal_rank: int
    topology_id: int
    components: tuple[LatentComponentParameters, ...]
    resolution: ResolutionShape | None
    linear_solution: LinearSolutionSnapshot
    proposal_score_raw: float | None
    raw_exact_log_rmse: float
    standardized_exact_log_rmse: float | None
    primary_gate_error: float
    exact_valid: bool
    bounds_pass: bool
    physics_pass: bool
    accepted: bool
    parameter_mode_id: str | None
    curve_group_id: str | None


@dataclass(frozen=True, kw_only=True)
class ParameterMode:
    mode_id: str
    topology_id: int
    topology: tuple[str, ...]
    member_candidate_ids: tuple[str, ...]
    representative_candidate_id: str
    best_primary_gate_error: float


@dataclass(frozen=True, kw_only=True)
class CurveEquivalenceGroup:
    group_id: str
    parameter_mode_ids: tuple[str, ...]
    member_candidate_ids: tuple[str, ...]
    representative_candidate_id: str
    best_primary_gate_error: float


@dataclass(frozen=True, kw_only=True)
class BestOfNMetric:
    n: int
    best_candidate_id: str
    best_primary_gate_error: float
    exact_valid_found: bool
    accepted_found: bool
    best_accepted_candidate_id: str | None
    best_accepted_primary_gate_error: float | None


@dataclass(frozen=True, kw_only=True)
class ReferenceModeMatch:
    reference_id: str
    recalled: bool
    nearest_candidate_id: str | None
    nearest_parameter_distance: float | None
    matched_candidate_id: str | None
    matched_parameter_distance: float | None


@dataclass(frozen=True, kw_only=True)
class EvaluationReport:
    audit_schema: str
    contract_version: str
    codec_version: str
    forward_model_version: str
    curve_id: str
    source_kind: str
    n_points: int
    q_min: float
    q_max: float
    raw_metric_name: str
    standardized_metric_name: str
    primary_gate_metric_name: str
    primary_gate_threshold: float
    curve_equivalence_metric_name: str
    curve_equivalence_threshold: float
    thresholds: EvaluationThresholds
    parameter_normalization_version: str
    parameter_distance_scope: str
    clustering_linkage: str
    reference_matching_version: str
    reference_set_distance_version: str
    reference_set_unmatched_distance: float
    proposal_score_semantics: str
    candidates: tuple[CandidateAssessment, ...]
    parameter_modes: tuple[ParameterMode, ...]
    curve_groups: tuple[CurveEquivalenceGroup, ...]
    representative_candidate_ids: tuple[str, ...]
    best_of_n: tuple[BestOfNMetric, ...]
    reference_matches: tuple[ReferenceModeMatch, ...]
    proposal_count: int
    accepted_count: int
    valid_precision: float
    duplicate_ratio: float
    curve_equivalence_redundancy_ratio: float
    bounds_pass_rate: float
    physics_pass_rate: float
    mode_recall: float | None
    reference_mode_count: int
    predicted_parameter_mode_count: int
    reference_match_count: int
    unmatched_compatible_mode_count: int
    prediction_reference_match_fraction: float | None
    symmetric_chamfer_parameter_distance: float | None
    hausdorff_parameter_distance: float | None

    def to_audit_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_audit_dict(), allow_nan=False, ensure_ascii=False, indent=indent, sort_keys=True
        )


@dataclass(frozen=True)
class _ScoredCandidate:
    value: CandidateInput
    raw_log_rmse: float
    standardized_log_rmse: float | None
    primary_gate_error: float
    exact_valid: bool
    accepted: bool


def complete_linkage_groups(nodes, distance, threshold):
    """Cluster deterministically without single-linkage chaining.

    A merge is legal only when every cross-cluster pair is within the fixed
    threshold. The closest legal pair is merged first, with stable index
    tie-breaking, so every final cluster has a bounded pairwise diameter.
    """

    groups = [[node] for node in nodes]
    while True:
        best = None
        for left_index in range(len(groups)):
            for right_index in range(left_index + 1, len(groups)):
                linkage = max(
                    distance(left, right)
                    for left in groups[left_index]
                    for right in groups[right_index]
                )
                if linkage <= threshold:
                    key = (linkage, left_index, right_index)
                    if best is None or key < best[0]:
                        best = (key, left_index, right_index)
        if best is None:
            return groups
        _, left_index, right_index = best
        groups[left_index].extend(groups[right_index])
        del groups[right_index]


def _best(scored: Sequence[_ScoredCandidate]) -> _ScoredCandidate:
    return min(scored, key=lambda item: (item.primary_gate_error, item.value.proposal_rank))


def _reference_mode_matching(
    references: Sequence[ReferenceMode],
    prediction_modes: Sequence[Sequence[CandidateInput]],
    *,
    distance_max: float,
    parameter_distance: Callable[[ReferenceMode, CandidateInput], float],
):
    """Match reference and predicted modes one-to-one without recall inflation.

    The assignment first maximizes the number of pairs within ``distance_max``
    and then minimizes their total normalized parameter distance. Independent
    nearest-neighbour diagnostics are retained, but only assigned in-threshold
    pairs count as recalled modes.
    """

    reference_count = len(references)
    prediction_count = len(prediction_modes)
    if reference_count == 0:
        return (), 0, None, None

    distances = np.full((reference_count, prediction_count), np.inf, dtype=np.float64)
    closest_members: dict[tuple[int, int], CandidateInput] = {}
    for reference_index, reference in enumerate(references):
        for prediction_index, members in enumerate(prediction_modes):
            member_distances = tuple(
                (parameter_distance(reference, member), member)
                for member in members
            )
            finite = tuple(item for item in member_distances if np.isfinite(item[0]))
            if finite:
                distance, member = min(
                    finite,
                    key=lambda item: (item[0], item[1].proposal_rank),
                )
                distances[reference_index, prediction_index] = distance
                closest_members[(reference_index, prediction_index)] = member

    assigned: dict[int, tuple[int, float]] = {}
    if prediction_count:
        # A single invalid edge must cost more than every possible valid edge
        # combined. Consequently Hungarian minimization first maximizes valid
        # cardinality and only then minimizes total distance.
        assignment_size = min(reference_count, prediction_count)
        invalid_cost = (assignment_size + 1.0) * (float(distance_max) + 1.0)
        costs = np.where(
            np.isfinite(distances) & (distances <= distance_max),
            distances,
            invalid_cost,
        )
        reference_indices, prediction_indices = linear_sum_assignment(costs)
        for reference_index, prediction_index in zip(
            reference_indices.tolist(), prediction_indices.tolist()
        ):
            distance = float(distances[reference_index, prediction_index])
            if np.isfinite(distance) and distance <= distance_max:
                assigned[reference_index] = (prediction_index, distance)

    matches = []
    for reference_index, reference in enumerate(references):
        finite = np.flatnonzero(np.isfinite(distances[reference_index]))
        if finite.size:
            nearest_index = min(
                finite.tolist(),
                key=lambda index: (
                    distances[reference_index, index],
                    closest_members[(reference_index, index)].proposal_rank,
                ),
            )
            nearest_id = closest_members[(reference_index, nearest_index)].candidate_id
            nearest_distance = float(distances[reference_index, nearest_index])
        else:
            nearest_id = None
            nearest_distance = None
        assignment = assigned.get(reference_index)
        matches.append(
            ReferenceModeMatch(
                reference_id=reference.reference_id,
                recalled=assignment is not None,
                nearest_candidate_id=nearest_id,
                nearest_parameter_distance=nearest_distance,
                matched_candidate_id=(
                    None
                    if assignment is None
                    else closest_members[(reference_index, assignment[0])].candidate_id
                ),
                matched_parameter_distance=(None if assignment is None else assignment[1]),
            )
        )

    if prediction_count:
        capped = np.where(
            np.isfinite(distances),
            np.minimum(distances, REFERENCE_SET_UNMATCHED_DISTANCE),
            REFERENCE_SET_UNMATCHED_DISTANCE,
        )
        reference_to_prediction = np.min(capped, axis=1)
        prediction_to_reference = np.min(capped, axis=0)
        chamfer = float(0.5 * (np.mean(reference_to_prediction) + np.mean(prediction_to_reference)))
        hausdorff = float(max(np.max(reference_to_prediction), np.max(prediction_to_reference)))
    else:
        chamfer = REFERENCE_SET_UNMATCHED_DISTANCE
        hausdorff = REFERENCE_SET_UNMATCHED_DISTANCE
    return tuple(matches), len(assigned), chamfer, hausdorff


def _evaluate_candidates(
    curve: ObservedCurve,
    candidates: Sequence[CandidateInput],
    *,
    thresholds: EvaluationThresholds,
    best_of_n: Sequence[int],
    reference_modes: Sequence[ReferenceMode] = (),
    candidate_parameter_distance: Callable[[CandidateInput, CandidateInput], float],
    reference_parameter_distance: Callable[[ReferenceMode, CandidateInput], float],
    audit_schema: str,
    parameter_normalization_version: str,
    parameter_distance_scope: str,
    reference_matching_version: str,
    reference_member_policy: Literal["all_mode_members", "actual_representative"],
) -> EvaluationReport:
    """Shared exact-gate engine with an explicitly injected distance contract."""
    if not isinstance(curve, ObservedCurve):
        raise TypeError("curve must be an ObservedCurve")
    if not isinstance(thresholds, EvaluationThresholds):
        raise TypeError("thresholds must be EvaluationThresholds")
    values = tuple(candidates)
    if not values or not all(isinstance(item, CandidateInput) for item in values):
        raise ValueError("candidates must contain at least one CandidateInput")
    ids = [item.candidate_id for item in values]
    ranks = [item.proposal_rank for item in values]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id values must be unique")
    if set(ranks) != set(range(1, len(values) + 1)):
        raise ValueError("proposal_rank values must be unique and contiguous from one")
    ordered = tuple(sorted(values, key=lambda item: item.proposal_rank))
    for item in ordered:
        if item.exact_intensity.shape != curve.intensity.shape:
            raise ValueError(
                f"candidate {item.candidate_id!r} exact_intensity does not match the curve"
            )

    ns = tuple(best_of_n)
    if not ns or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in ns
    ):
        raise ValueError("best_of_n must contain positive integer cutoffs")
    ns = tuple(int(value) for value in ns)
    if any(value < 1 or value > len(ordered) for value in ns) or len(set(ns)) != len(ns):
        raise ValueError("best_of_n cutoffs must be unique and within the proposal count")
    references = tuple(reference_modes)
    if not all(isinstance(item, ReferenceMode) for item in references):
        raise TypeError("reference_modes must contain only ReferenceMode values")
    if len({item.reference_id for item in references}) != len(references):
        raise ValueError("reference_id values must be unique")
    if reference_member_policy not in {"all_mode_members", "actual_representative"}:
        raise ValueError("unknown reference-member policy")

    if curve.sigma_log is None:
        primary_metric_name = RAW_LOG_RMSE_METRIC
        primary_threshold = thresholds.raw_exact_log_rmse_max
    else:
        primary_metric_name = STANDARDIZED_LOG_RMSE_METRIC
        primary_threshold = thresholds.standardized_exact_log_rmse_max

    scored = []
    for item in ordered:
        raw = natural_log_rmse(item.exact_intensity, curve.intensity)
        standardized = None
        if curve.sigma_log is not None:
            standardized = natural_log_rmse(
                item.exact_intensity, curve.intensity, sigma_log=curve.sigma_log
            )
        primary_error = raw if standardized is None else standardized
        exact_valid = primary_error <= primary_threshold
        accepted = exact_valid and item.bounds_pass and item.physics_pass
        scored.append(
            _ScoredCandidate(item, raw, standardized, primary_error, exact_valid, accepted)
        )

    accepted = [item for item in scored if item.accepted]
    parameter_groups = complete_linkage_groups(
        accepted,
        lambda left, right: candidate_parameter_distance(left.value, right.value),
        thresholds.parameter_mode_distance_max,
    )
    parameter_groups.sort(key=lambda group: min(item.value.proposal_rank for item in group))
    modes = []
    candidate_to_mode = {}
    mode_members = {}
    for number, group in enumerate(parameter_groups, 1):
        group.sort(key=lambda item: item.value.proposal_rank)
        representative = _best(group)
        mode_id = f"parameter_mode_{number:03d}"
        member_ids = tuple(item.value.candidate_id for item in group)
        mode = ParameterMode(
            mode_id=mode_id,
            topology_id=representative.value.topology_id,
            topology=topology_from_id(representative.value.topology_id),
            member_candidate_ids=member_ids,
            representative_candidate_id=representative.value.candidate_id,
            best_primary_gate_error=representative.primary_gate_error,
        )
        modes.append(mode)
        mode_members[mode_id] = group
        candidate_to_mode.update({candidate_id: mode_id for candidate_id in member_ids})

    scored_by_id = {item.value.candidate_id: item for item in scored}
    curve_components = complete_linkage_groups(
        modes,
        lambda left, right: natural_log_rmse(
            scored_by_id[left.representative_candidate_id].value.exact_intensity,
            scored_by_id[right.representative_candidate_id].value.exact_intensity,
        ),
        thresholds.raw_curve_equivalence_log_rmse_max,
    )
    curve_components.sort(
        key=lambda group: min(
            scored_by_id[mode.representative_candidate_id].value.proposal_rank for mode in group
        )
    )
    curve_groups = []
    candidate_to_curve = {}
    for number, grouped_modes in enumerate(curve_components, 1):
        members = [item for mode in grouped_modes for item in mode_members[mode.mode_id]]
        members.sort(key=lambda item: item.value.proposal_rank)
        representative = _best(members)
        group_id = f"curve_group_{number:03d}"
        member_ids = tuple(item.value.candidate_id for item in members)
        curve_groups.append(
            CurveEquivalenceGroup(
                group_id=group_id,
                parameter_mode_ids=tuple(mode.mode_id for mode in grouped_modes),
                member_candidate_ids=member_ids,
                representative_candidate_id=representative.value.candidate_id,
                best_primary_gate_error=representative.primary_gate_error,
            )
        )
        candidate_to_curve.update({candidate_id: group_id for candidate_id in member_ids})

    assessments = tuple(
        CandidateAssessment(
            candidate_id=item.value.candidate_id,
            proposal_rank=item.value.proposal_rank,
            topology_id=item.value.topology_id,
            components=item.value.components,
            resolution=item.value.resolution,
            linear_solution=item.value.linear_solution,
            proposal_score_raw=item.value.proposal_score_raw,
            raw_exact_log_rmse=item.raw_log_rmse,
            standardized_exact_log_rmse=item.standardized_log_rmse,
            primary_gate_error=item.primary_gate_error,
            exact_valid=item.exact_valid,
            bounds_pass=item.value.bounds_pass,
            physics_pass=item.value.physics_pass,
            accepted=item.accepted,
            parameter_mode_id=candidate_to_mode.get(item.value.candidate_id),
            curve_group_id=candidate_to_curve.get(item.value.candidate_id),
        )
        for item in scored
    )

    best_metrics = []
    for cutoff in ns:
        prefix = scored[:cutoff]
        best = _best(prefix)
        accepted_prefix = [item for item in prefix if item.accepted]
        best_accepted = _best(accepted_prefix) if accepted_prefix else None
        best_metrics.append(
            BestOfNMetric(
                n=cutoff,
                best_candidate_id=best.value.candidate_id,
                best_primary_gate_error=best.primary_gate_error,
                exact_valid_found=any(item.exact_valid for item in prefix),
                accepted_found=best_accepted is not None,
                best_accepted_candidate_id=(
                    None if best_accepted is None else best_accepted.value.candidate_id
                ),
                best_accepted_primary_gate_error=(
                    None if best_accepted is None else best_accepted.primary_gate_error
                ),
            )
        )

    if reference_member_policy == "actual_representative":
        predicted_mode_members = tuple(
            (scored_by_id[mode.representative_candidate_id].value,) for mode in modes
        )
    else:
        predicted_mode_members = tuple(
            tuple(item.value for item in mode_members[mode.mode_id]) for mode in modes
        )
    (
        reference_matches,
        reference_match_count,
        symmetric_chamfer,
        hausdorff,
    ) = _reference_mode_matching(
        references,
        predicted_mode_members,
        distance_max=thresholds.reference_mode_distance_max,
        parameter_distance=reference_parameter_distance,
    )

    count = len(scored)
    accepted_count = len(accepted)
    representatives = tuple(
        group.representative_candidate_id
        for group in sorted(curve_groups, key=lambda item: item.best_primary_gate_error)
    )
    return EvaluationReport(
        audit_schema=audit_schema,
        contract_version=CONTRACT_VERSION,
        codec_version=CODEC_VERSION,
        forward_model_version=FORWARD_MODEL_VERSION,
        curve_id=curve.curve_id,
        source_kind=curve.source_kind,
        n_points=curve.q.size,
        q_min=float(curve.q[0]),
        q_max=float(curve.q[-1]),
        raw_metric_name=RAW_LOG_RMSE_METRIC,
        standardized_metric_name=STANDARDIZED_LOG_RMSE_METRIC,
        primary_gate_metric_name=primary_metric_name,
        primary_gate_threshold=primary_threshold,
        curve_equivalence_metric_name=RAW_LOG_RMSE_METRIC,
        curve_equivalence_threshold=thresholds.raw_curve_equivalence_log_rmse_max,
        thresholds=thresholds,
        parameter_normalization_version=parameter_normalization_version,
        parameter_distance_scope=parameter_distance_scope,
        clustering_linkage=CLUSTERING_LINKAGE,
        reference_matching_version=reference_matching_version,
        reference_set_distance_version=REFERENCE_SET_DISTANCE_VERSION,
        reference_set_unmatched_distance=REFERENCE_SET_UNMATCHED_DISTANCE,
        proposal_score_semantics=PROPOSAL_SCORE_SEMANTICS,
        candidates=assessments,
        parameter_modes=tuple(modes),
        curve_groups=tuple(curve_groups),
        representative_candidate_ids=representatives,
        best_of_n=tuple(best_metrics),
        reference_matches=tuple(reference_matches),
        proposal_count=count,
        accepted_count=accepted_count,
        valid_precision=accepted_count / count,
        duplicate_ratio=0.0 if accepted_count == 0 else 1.0 - len(modes) / accepted_count,
        curve_equivalence_redundancy_ratio=(
            0.0 if not modes else 1.0 - len(curve_groups) / len(modes)
        ),
        bounds_pass_rate=sum(item.value.bounds_pass for item in scored) / count,
        physics_pass_rate=sum(item.value.physics_pass for item in scored) / count,
        mode_recall=(None if not references else reference_match_count / len(references)),
        reference_mode_count=len(references),
        predicted_parameter_mode_count=len(modes),
        reference_match_count=reference_match_count,
        unmatched_compatible_mode_count=len(modes) - reference_match_count,
        prediction_reference_match_fraction=(
            None if not references or not modes else reference_match_count / len(modes)
        ),
        symmetric_chamfer_parameter_distance=symmetric_chamfer,
        hausdorff_parameter_distance=hausdorff,
    )


def evaluate_candidates(
    curve: ObservedCurve,
    candidates: Sequence[CandidateInput],
    *,
    thresholds: EvaluationThresholds,
    best_of_n: Sequence[int],
    reference_modes: Sequence[ReferenceMode] = (),
) -> EvaluationReport:
    """Apply the legacy/global exact gates, clustering, and reference matching.

    This API remains the frozen baseline for older single-query workflows.
    V5.2 universal inference must instead use its query-bound evaluator.
    """

    return _evaluate_candidates(
        curve,
        candidates,
        thresholds=thresholds,
        best_of_n=best_of_n,
        reference_modes=reference_modes,
        candidate_parameter_distance=normalized_latent_parameter_distance,
        reference_parameter_distance=normalized_latent_parameter_distance,
        audit_schema=EVALUATION_AUDIT_SCHEMA,
        parameter_normalization_version=PARAMETER_NORMALIZATION_VERSION,
        parameter_distance_scope=PARAMETER_DISTANCE_SCOPE,
        reference_matching_version=REFERENCE_MATCHING_VERSION,
        reference_member_policy="all_mode_members",
    )
