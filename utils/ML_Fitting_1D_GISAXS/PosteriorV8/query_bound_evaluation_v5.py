"""V5.2 query-bound evaluation for universal one-click inference.

The legacy evaluator has no access to the complete GUI query or to the hard
branch that produced each candidate.  This adapter binds both before any
parameter clustering or reference matching, then delegates the shared exact
curve gates and report construction to :mod:`evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, Sequence

import numpy as np

from .amplitude_query_v5 import V5_AMPLITUDE_QUERY_SCHEMA, V5_AMPLITUDE_QUERY_VERSION
from .bounds_query_v5 import V5_BOUNDS_QUERY_SCHEMA, V5_BOUNDS_QUERY_VERSION
from .branch_catalog import branch_pattern_id
from .contract import MAX_COMPONENTS
from .evaluation import (
    CLUSTERING_LINKAGE,
    CandidateInput,
    EvaluationReport,
    EvaluationThresholds,
    ObservedCurve,
    ReferenceMode,
    _evaluate_candidates,
)
from .grouped_artifact_v5 import canonical_json
from .model_v5_contract import (
    MODEL_V5_NAME,
    MODEL_V5_NUMERIC_POLICY_CONTRACT,
    MODEL_V5_SCHEMA,
    MODEL_V5_VERSION,
)
from .query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
    V5_QUERY_PARAMETER_DISTANCE_SCOPE,
    V5_QUERY_PARAMETER_DISTANCE_SHA256,
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
    query_local_parameter_distance,
)
from .universal_query_v5 import (
    V5_UNIVERSAL_QUERY_SCHEMA,
    V5_UNIVERSAL_QUERY_VERSION,
    V5UniversalCandidateContext,
    validate_v5_universal_curve_alignment,
)


V5_QUERY_BOUND_EVALUATION_SCHEMA = "gisaxs.posterior_v8.query_bound_evaluation/v2"
V5_QUERY_BOUND_EVALUATION_VERSION = (
    "posterior_v8_v5_2_r2_numeric_query_context_branch_bound_"
    "clustering_and_emission_matching_v2"
)
V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION = (
    "maximum_cardinality_minimum_query_local_distance_bipartite_actual_representative_matching/v1"
)
V5_QUERY_BOUND_EVALUATION_PAYLOAD = {
    "schema": V5_QUERY_BOUND_EVALUATION_SCHEMA,
    "version": V5_QUERY_BOUND_EVALUATION_VERSION,
    "parameter_distance": {
        "schema": V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
        "version": V5_QUERY_PARAMETER_DISTANCE_VERSION,
        "sha256": V5_QUERY_PARAMETER_DISTANCE_SHA256,
    },
    "query_contract": {
        "universal_schema": V5_UNIVERSAL_QUERY_SCHEMA,
        "universal_version": V5_UNIVERSAL_QUERY_VERSION,
        "geometry_schema": V5_BOUNDS_QUERY_SCHEMA,
        "geometry_version": V5_BOUNDS_QUERY_VERSION,
        "amplitude_schema": V5_AMPLITUDE_QUERY_SCHEMA,
        "amplitude_version": V5_AMPLITUDE_QUERY_VERSION,
    },
    "model_contract": {
        "schema": MODEL_V5_SCHEMA,
        "version": MODEL_V5_VERSION,
        "name": MODEL_V5_NAME,
        "numeric_policy_contract": {
            name: dict(value)
            for name, value in MODEL_V5_NUMERIC_POLICY_CONTRACT.items()
        },
    },
    "candidate_branch_binding": (
        "V5UniversalCandidateContext_plus_candidate_provenance_global_branch_key"
    ),
    "curve_context_binding": (
        "same_preprocessed_q_intensity_uncertainty_provenance_and_acceptance_sigma"
    ),
    "reference_branch_binding": (
        "unique_context_branch_from_reference_topology_D_presence_resolution_pattern"
    ),
    "cross_contextual_branch_distance": "infinity",
    "clustering_linkage": CLUSTERING_LINKAGE,
    "reference_matching": V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION,
    "reference_matching_population": "actual_emitted_parameter_mode_representatives_only",
    "invalid_or_ambiguous_binding": "fail_closed",
    "legacy_global_evaluator": "retained_as_a_separate_baseline",
}
V5_QUERY_BOUND_EVALUATION_SHA256 = sha256(
    canonical_json(V5_QUERY_BOUND_EVALUATION_PAYLOAD).encode("utf-8")
).hexdigest()


class V5CandidateProvenanceLike(Protocol):
    candidate_id: str
    proposal_rank: int
    global_branch_key: str
    topology_id: int
    pattern_id: int
    model_global_index: int


@dataclass(frozen=True)
class _QueryBoundTask:
    universal_context: V5UniversalCandidateContext
    branch: object

    @property
    def codec(self):
        return self.universal_context.codec_for(self.branch.global_key.wire_key)

    @property
    def amplitude_constraint(self):
        return self.branch.amplitude_constraint


def _physical_pattern(value: CandidateInput | ReferenceMode) -> int:
    d_present = tuple(component.log_D is not None for component in value.components)
    padded = d_present + (False,) * (MAX_COMPONENTS - len(d_present))
    return branch_pattern_id(padded, value.resolution is not None)


def _validated_task_value(
    task: _QueryBoundTask,
    value: CandidateInput | ReferenceMode,
    *,
    label: str,
) -> None:
    expected = task.branch.global_key
    if value.topology_id != expected.topology_id or _physical_pattern(value) != expected.pattern_id:
        raise ValueError(
            f"{label} physical topology/D/Resolution pattern disagrees with its branch"
        )
    try:
        self_distance = query_local_parameter_distance(task, value, value)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} escaped its query-bound geometry/amplitude contract") from exc
    if not np.isfinite(self_distance) or self_distance > 1.0e-12:
        raise ValueError(f"{label} could not be validated in its contextual branch")


def _candidate_tasks(
    context: V5UniversalCandidateContext,
    candidates: tuple[CandidateInput, ...],
    provenance: Sequence[V5CandidateProvenanceLike],
) -> dict[str, _QueryBoundTask]:
    if isinstance(provenance, (str, bytes)):
        raise TypeError("candidate_provenance must be a sequence of provenance rows")
    rows = tuple(provenance)
    if len(rows) != len(candidates):
        raise ValueError("candidate_provenance must contain exactly one row per candidate")
    tasks: dict[str, _QueryBoundTask] = {}
    for index, (candidate, row) in enumerate(zip(candidates, rows)):
        try:
            row_identity = (
                row.candidate_id,
                row.proposal_rank,
                row.global_branch_key,
                row.topology_id,
                row.pattern_id,
                row.model_global_index,
            )
        except AttributeError as exc:
            raise TypeError(
                f"candidate_provenance[{index}] does not implement the V5 provenance contract"
            ) from exc
        try:
            branch = context.branch_for(row.global_branch_key)
        except KeyError as exc:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} provenance names an unknown branch"
            ) from exc
        expected_identity = (
            candidate.candidate_id,
            candidate.proposal_rank,
            branch.global_key.wire_key,
            branch.global_key.topology_id,
            branch.global_key.pattern_id,
            branch.global_index,
        )
        if row_identity != expected_identity:
            raise ValueError(
                f"candidate {candidate.candidate_id!r} provenance disagrees with the universal context"
            )
        task = _QueryBoundTask(context, branch)
        _validated_task_value(task, candidate, label=f"candidate {candidate.candidate_id!r}")
        tasks[candidate.candidate_id] = task
    return tasks


def _reference_tasks(
    context: V5UniversalCandidateContext,
    references: tuple[ReferenceMode, ...],
) -> dict[str, _QueryBoundTask]:
    tasks: dict[str, _QueryBoundTask] = {}
    for reference in references:
        if not isinstance(reference, ReferenceMode):
            raise TypeError("reference_modes must contain only ReferenceMode values")
        pattern_id = _physical_pattern(reference)
        matches = tuple(
            branch
            for branch in context.branches
            if branch.global_key.topology_id == reference.topology_id
            and branch.global_key.pattern_id == pattern_id
        )
        if len(matches) != 1:
            raise ValueError(
                f"reference {reference.reference_id!r} does not map to exactly one "
                "selected contextual topology/D/Resolution branch"
            )
        task = _QueryBoundTask(context, matches[0])
        _validated_task_value(task, reference, label=f"reference {reference.reference_id!r}")
        tasks[reference.reference_id] = task
    return tasks


def evaluate_v5_query_bound_candidates(
    context: V5UniversalCandidateContext,
    curve: ObservedCurve,
    candidates: Sequence[CandidateInput],
    *,
    candidate_provenance: Sequence[V5CandidateProvenanceLike],
    thresholds: EvaluationThresholds,
    best_of_n: Sequence[int],
    reference_modes: Sequence[ReferenceMode] = (),
) -> EvaluationReport:
    """Evaluate candidates under the exact V5.2 user query and branch provenance.

    Every candidate and reference is bound and validated before scoring.  Two
    values from different contextual branches have infinite parameter
    distance.  Reference recall uses only the representative that the product
    actually emits, never another accepted member hidden inside its cluster.
    """

    if not isinstance(context, V5UniversalCandidateContext):
        raise TypeError("context must be a V5UniversalCandidateContext")
    validate_v5_universal_curve_alignment(context, curve)
    values = tuple(candidates)
    if not values or not all(isinstance(value, CandidateInput) for value in values):
        raise ValueError("candidates must contain at least one CandidateInput")
    references = tuple(reference_modes)
    candidate_tasks = _candidate_tasks(context, values, candidate_provenance)
    reference_tasks = _reference_tasks(context, references)

    def candidate_distance(left: CandidateInput, right: CandidateInput) -> float:
        left_task = candidate_tasks[left.candidate_id]
        right_task = candidate_tasks[right.candidate_id]
        if left_task.branch.context_sha256 != right_task.branch.context_sha256:
            return float("inf")
        return query_local_parameter_distance(left_task, left, right)

    def reference_distance(reference: ReferenceMode, candidate: CandidateInput) -> float:
        reference_task = reference_tasks[reference.reference_id]
        candidate_task = candidate_tasks[candidate.candidate_id]
        if reference_task.branch.context_sha256 != candidate_task.branch.context_sha256:
            return float("inf")
        return query_local_parameter_distance(reference_task, reference, candidate)

    return _evaluate_candidates(
        curve,
        values,
        thresholds=thresholds,
        best_of_n=best_of_n,
        reference_modes=references,
        candidate_parameter_distance=candidate_distance,
        reference_parameter_distance=reference_distance,
        audit_schema=V5_QUERY_BOUND_EVALUATION_SCHEMA,
        parameter_normalization_version=V5_QUERY_PARAMETER_DISTANCE_VERSION,
        parameter_distance_scope=V5_QUERY_PARAMETER_DISTANCE_SCOPE,
        reference_matching_version=V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION,
        reference_member_policy="actual_representative",
    )


__all__ = [
    "V5CandidateProvenanceLike",
    "V5_QUERY_BOUND_EVALUATION_PAYLOAD",
    "V5_QUERY_BOUND_EVALUATION_SCHEMA",
    "V5_QUERY_BOUND_EVALUATION_SHA256",
    "V5_QUERY_BOUND_EVALUATION_VERSION",
    "V5_QUERY_BOUND_REFERENCE_MATCHING_VERSION",
    "evaluate_v5_query_bound_candidates",
]
