"""Query-bound operational reference representatives for V5.1.

The legacy reference bank uses a global parameter distance and therefore may
exchange repeated shape slots that carry different user ranges.  This module
defines the stricter paper endpoint: every compatible candidate is tied to one
observation, the complete geometry/amplitude query, the exact GUI forward,
the calibrated protocol, and the scientific source bundle.  Component slots
are exchangeable only when their complete geometry bounds, D policy, and GUI
``Int_i`` range are identical.

The result is a finite-search operational reference set.  It is deliberately
not a posterior, a proof of identifiability, or a claim that all mathematical
solutions were found.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import re
from typing import Mapping, Sequence

import numpy as np

from .contract import FORWARD_MODEL_VERSION, latent_component_to_gui
from .evaluation import (
    CandidateInput,
    LinearSolutionSnapshot,
    complete_linkage_groups,
    natural_log_rmse,
)
from .grouped_artifact_v5 import array_sha256, canonical_json
from .profiled_forward import evaluate_gui_forward_snapshot
from .query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_PAYLOAD,
    V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
    V5_QUERY_PARAMETER_DISTANCE_SCOPE,
    V5_QUERY_PARAMETER_DISTANCE_SHA256,
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
    legal_query_slot_permutations,
    query_local_parameter_distance,
)
from .search_supervision_contract_v5 import (
    V5FrozenSearchTask,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)


V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA = "gisaxs.posterior_v8.contextual_reference_bank/v3"
V5_CONTEXTUAL_REFERENCE_BANK_VERSION = (
    "posterior_v8_query_bound_scale_sensitive_complete_linkage_diameter_representatives_v3"
)
V5_CONTEXTUAL_REFERENCE_DISTANCE_VERSION = V5_QUERY_PARAMETER_DISTANCE_VERSION
V5_CONTEXTUAL_SLOT_EQUIVALENCE_VERSION = (
    "posterior_v8_contextual_catalog_class_plus_actual_d_presence_slots_v2"
)
V5_CONTEXTUAL_DEDUPLICATION_VERSION = (
    "posterior_v8_metric_ordered_agglomerative_complete_linkage_diameter_delta_v3"
)
V5_CONTEXTUAL_REFERENCE_CLAIM = (
    "finite_search_exact_compatible_complete_linkage_diameter_cluster_representatives_"
    "not_all_solutions_or_posterior_modes"
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EVIDENCE_FIELDS = (
    "exact_curve_sha256",
    "exact_observation_audit_sha256",
    "universal_query_sha256",
    "global_branch_key",
    "branch_context_sha256",
    "topology_query_sha256",
    "geometry_query_sha256",
    "amplitude_query_sha256",
    "amplitude_constraint_sha256",
    "authoritative_forward_id",
    "protocol_sha256",
    "calibration_identity_sha256",
    "calibrated_threshold_sha256",
    "source_bundle_sha256",
)


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_finite(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _source_items(source_sha256: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(source_sha256, Mapping) or not source_sha256:
        raise ValueError("source_sha256 must be a non-empty mapping")
    items = []
    for raw_name, raw_digest in source_sha256.items():
        name = _text(raw_name, "source name")
        items.append((name, _digest(raw_digest, f"source_sha256[{name}]")))
    if len({name for name, _ in items}) != len(items):
        raise ValueError("source_sha256 names must be unique")
    return tuple(sorted(items))


def _source_bundle_sha256(items: tuple[tuple[str, str], ...]) -> str:
    return sha256(canonical_json(dict(items)).encode("utf-8")).hexdigest()


def _expected_identity_order(values: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if set(values) != set(_EVIDENCE_FIELDS):
        raise ValueError("candidate evidence identity fields are incomplete or unsupported")
    return tuple((name, values[name]) for name in _EVIDENCE_FIELDS)


def _observation_matches_task_context(task: V5FrozenSearchTask) -> None:
    view = task.exact_observation.observation_view
    expected = {
        "x": view.preprocessed.x.astype(np.float32),
        "point_mask": view.preprocessed.point_mask,
        "global_features": view.preprocessed.global_features.astype(np.float32),
        "uncertainty_provenance": np.asarray(view.uncertainty.feature_vector, dtype=np.float32),
    }
    first = task.universal_context.batches[0].model_inputs
    if any(not np.array_equal(first[name][0], value) for name, value in expected.items()):
        raise ValueError("universal query does not encode the bound exact observation")


def _full_query_payload(task: V5FrozenSearchTask) -> list[dict[str, object]]:
    result = []
    for topology_index, (entry, batch) in enumerate(
        zip(task.universal_context.topology_queries, task.universal_context.batches)
    ):
        result.append(
            {
                "topology_query_sha256": entry.sha256,
                "candidate_query_sha256": batch.query_sha256,
                "geometry_query": json.loads(entry.geometry.canonical_json),
                "amplitude_query": json.loads(entry.amplitude.canonical_json),
                "branches": [
                    {
                        "global_branch_key": branch.global_key.wire_key,
                        "context_sha256": branch.context_sha256,
                        "amplitude_constraint_sha256": branch.amplitude_constraint_sha256,
                        "amplitude_constraint": branch.amplitude_constraint.to_audit_dict(),
                    }
                    for branch in task.universal_context.branches
                    if branch.topology_batch_index == topology_index
                ],
            }
        )
    return result


def _scope_payload(
    task: V5FrozenSearchTask,
    source_items: tuple[tuple[str, str], ...],
    reference_delta: float,
) -> dict[str, object]:
    threshold = task.calibrated_threshold
    assert threshold is not None
    return {
        "schema": V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
        "version": V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
        "claim": V5_CONTEXTUAL_REFERENCE_CLAIM,
        "exact_curve_sha256": task.exact_curve_sha256,
        "exact_observation_audit": task.exact_observation.audit_payload(),
        "universal_query_sha256": task.universal_context.audit_sha256,
        "universal_query": json.loads(task.universal_context.audit_json),
        "full_gui_query_contracts": _full_query_payload(task),
        "query_catalog_artifact_id": task.query_catalog_artifact_id,
        "query_catalog_artifact_sha256": task.query_catalog_artifact_sha256,
        "authoritative_forward_id": task.protocol.authoritative_forward_id,
        "protocol_sha256": task.protocol.sha256,
        "protocol": task.protocol.audit_payload(),
        "calibration_identity_sha256": threshold.calibration_identity.sha256,
        "calibration_identity": threshold.calibration_identity.audit_payload(),
        "calibrated_threshold_sha256": threshold.sha256,
        "calibrated_threshold": threshold.audit_payload(),
        "source_sha256": dict(source_items),
        "source_bundle_sha256": _source_bundle_sha256(source_items),
        "distance_contract": {
            "schema": V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
            "version": V5_CONTEXTUAL_REFERENCE_DISTANCE_VERSION,
            "scope": V5_QUERY_PARAMETER_DISTANCE_SCOPE,
            "sha256": V5_QUERY_PARAMETER_DISTANCE_SHA256,
            "payload": V5_QUERY_PARAMETER_DISTANCE_PAYLOAD,
        },
        "slot_equivalence_contract": V5_CONTEXTUAL_SLOT_EQUIVALENCE_VERSION,
        "deduplication_contract": V5_CONTEXTUAL_DEDUPLICATION_VERSION,
        "reference_delta": reference_delta,
        "executor_protocol_delta": task.protocol.delta_separation_threshold,
    }


@dataclass(frozen=True, eq=False)
class V5ContextualReferenceScope:
    """One immutable paper-reference population for one query/observation."""

    task_template: V5FrozenSearchTask
    source_sha256: tuple[tuple[str, str], ...]
    reference_delta: float

    def __post_init__(self) -> None:
        _validate_scope_task(self.task_template)
        items = _source_items(dict(self.source_sha256))
        if items != self.source_sha256:
            raise ValueError("source_sha256 must use unique canonical name order")
        delta = _positive_finite(self.reference_delta, "reference_delta")
        object.__setattr__(self, "reference_delta", delta)

    @property
    def source_bundle_sha256(self) -> str:
        return _source_bundle_sha256(self.source_sha256)

    @property
    def delta(self) -> float:
        return self.reference_delta

    @property
    def audit_json(self) -> str:
        return canonical_json(
            _scope_payload(self.task_template, self.source_sha256, self.reference_delta)
        )

    @property
    def sha256(self) -> str:
        return sha256(self.audit_json.encode("utf-8")).hexdigest()


def _validate_scope_task(task: V5FrozenSearchTask) -> None:
    if not isinstance(task, V5FrozenSearchTask):
        raise TypeError("task_template must be a V5FrozenSearchTask")
    if task.protocol.protocol_tier != V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED:
        raise ValueError("contextual paper reference scope requires a formal calibrated task")
    if task.calibrated_threshold is None or task.observed_curve.sigma_log is None:
        raise ValueError("contextual paper reference scope requires calibrated acceptance sigma")
    if task.protocol.authoritative_forward_id != FORWARD_MODEL_VERSION:
        raise ValueError("formal reference scope selected a different authoritative forward")
    if task.observation_id != task.observed_curve.curve_id:
        raise ValueError("task observation_id does not match the exact observation")
    if task.calibrated_threshold.calibration_identity != task.protocol.calibration_identity:
        raise ValueError("task threshold escaped the protocol calibration identity")
    _observation_matches_task_context(task)


def build_v5_contextual_reference_scope(
    task: V5FrozenSearchTask,
    *,
    source_sha256: Mapping[str, str],
    reference_delta: float,
) -> V5ContextualReferenceScope:
    """Freeze the full query, observation, calibration, protocol, and sources."""

    _validate_scope_task(task)
    items = _source_items(source_sha256)
    delta = _positive_finite(reference_delta, "reference_delta")
    return V5ContextualReferenceScope(task, items, delta)


@dataclass(frozen=True)
class V5ContextualCandidateEvidence:
    """Declared external evidence that must replay against the live task."""

    identity: tuple[tuple[str, str], ...]
    executor_artifact_id: str
    executor_artifact_sha256: str

    def __post_init__(self) -> None:
        try:
            identity = tuple((str(name), str(value)) for name, value in self.identity)
        except (TypeError, ValueError) as exc:
            raise TypeError("identity must contain name/value pairs") from exc
        if identity != tuple(_expected_identity_order(dict(identity))):
            raise ValueError("candidate evidence identity fields are incomplete or unordered")
        for name, value in identity:
            if name in {"global_branch_key", "authoritative_forward_id"}:
                _text(value, name)
            else:
                _digest(value, name)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(
            self, "executor_artifact_id", _text(self.executor_artifact_id, "executor_artifact_id")
        )
        object.__setattr__(
            self,
            "executor_artifact_sha256",
            _digest(self.executor_artifact_sha256, "executor_artifact_sha256"),
        )

    def value(self, name: str) -> str:
        try:
            return dict(self.identity)[name]
        except KeyError as exc:
            raise KeyError(f"unknown candidate evidence identity {name!r}") from exc

    def audit_payload(self) -> dict[str, object]:
        return {
            **dict(self.identity),
            "executor_artifact_id": self.executor_artifact_id,
            "executor_artifact_sha256": self.executor_artifact_sha256,
        }


def contextual_candidate_evidence_from_task(
    task: V5FrozenSearchTask,
    *,
    source_bundle_sha256: str,
    executor_artifact_id: str,
    executor_artifact_sha256: str,
) -> V5ContextualCandidateEvidence:
    """Copy every identity a checked executor candidate must declare."""

    threshold = task.calibrated_threshold
    if threshold is None:
        raise ValueError("contextual paper candidate requires calibrated threshold evidence")
    return V5ContextualCandidateEvidence(
        identity=tuple(
            _expected_identity_order(_task_evidence_identity(task, source_bundle_sha256))
        ),
        executor_artifact_id=executor_artifact_id,
        executor_artifact_sha256=executor_artifact_sha256,
    )


def legal_contextual_slot_permutations(task: V5FrozenSearchTask) -> tuple[tuple[int, ...], ...]:
    """Return exactly the slot permutations authorized by the complete query."""

    if not isinstance(task, V5FrozenSearchTask):
        raise TypeError("task must be a V5FrozenSearchTask")
    return legal_query_slot_permutations(task)


def _candidate_payload(candidate: CandidateInput, pattern_id: int) -> dict[str, object]:
    return {
        "topology_id": candidate.topology_id,
        "pattern_id": pattern_id,
        "components": [asdict(value) for value in candidate.components],
        "resolution": None if candidate.resolution is None else asdict(candidate.resolution),
        "linear_solution": asdict(candidate.linear_solution),
    }


def _canonical_candidate(task: V5FrozenSearchTask, candidate: CandidateInput) -> CandidateInput:
    choices = []
    for order in legal_contextual_slot_permutations(task):
        components = tuple(candidate.components[index] for index in order)
        amplitudes = tuple(candidate.linear_solution.particle_amplitudes[index] for index in order)
        linear = LinearSolutionSnapshot(
            background=candidate.linear_solution.background,
            particle_amplitudes=amplitudes,
            resolution_amplitude=candidate.linear_solution.resolution_amplitude,
            k=candidate.linear_solution.k,
        )
        task.codec.encode(components, candidate.resolution)
        coefficients = (
            linear.background,
            *linear.particle_amplitudes,
            *(() if candidate.resolution is None else (linear.resolution_amplitude,)),
        )
        if not task.amplitude_constraint.contains(coefficients, k=linear.k):
            continue
        value = replace(candidate, components=components, linear_solution=linear)
        choices.append(
            (canonical_json(_candidate_payload(value, task.branch.global_key.pattern_id)), value)
        )
    if not choices:
        raise ValueError("candidate has no full-query-authorized slot assignment")
    return min(choices, key=lambda value: value[0])[1]


def _task_evidence_identity(task: V5FrozenSearchTask, source_bundle_sha256: str) -> dict[str, str]:
    threshold = task.calibrated_threshold
    if threshold is None:
        raise ValueError("candidate task lacks calibrated threshold evidence")
    branch = task.branch
    batch = task.universal_context.batches[branch.topology_batch_index]
    entry = task.universal_context.topology_queries[branch.topology_batch_index]
    return {
        "exact_curve_sha256": task.exact_curve_sha256,
        "exact_observation_audit_sha256": task.exact_observation.audit_sha256,
        "universal_query_sha256": task.universal_context.audit_sha256,
        "global_branch_key": branch.global_key.wire_key,
        "branch_context_sha256": branch.context_sha256,
        "topology_query_sha256": entry.sha256,
        "geometry_query_sha256": batch.query.sha256,
        "amplitude_query_sha256": batch.amplitude_query.sha256,
        "amplitude_constraint_sha256": branch.amplitude_constraint_sha256,
        "authoritative_forward_id": task.protocol.authoritative_forward_id,
        "protocol_sha256": task.protocol.sha256,
        "calibration_identity_sha256": threshold.calibration_identity.sha256,
        "calibrated_threshold_sha256": threshold.sha256,
        "source_bundle_sha256": source_bundle_sha256,
    }


@dataclass(frozen=True, eq=False)
class V5BoundReferenceCandidate:
    """An exact-compatible candidate canonically bound to one reference scope."""

    scope_sha256: str
    candidate: CandidateInput
    evidence: V5ContextualCandidateEvidence
    selected_metric_value: float
    canonical_parameter_sha256: str
    binding_sha256: str

    def __post_init__(self) -> None:
        _digest(self.scope_sha256, "scope_sha256")
        if not isinstance(self.candidate, CandidateInput):
            raise TypeError("candidate must be a CandidateInput")
        if not isinstance(self.evidence, V5ContextualCandidateEvidence):
            raise TypeError("evidence must be V5ContextualCandidateEvidence")
        metric = float(self.selected_metric_value)
        if not np.isfinite(metric) or metric < 0.0:
            raise ValueError("selected_metric_value must be finite and non-negative")
        object.__setattr__(self, "selected_metric_value", metric)
        _digest(self.canonical_parameter_sha256, "canonical_parameter_sha256")
        _digest(self.binding_sha256, "binding_sha256")

    def audit_payload(self) -> dict[str, object]:
        pattern_id = int(self.evidence.value("global_branch_key").rsplit("-", 1)[1])
        return {
            "scope_sha256": self.scope_sha256,
            "candidate_id": self.candidate.candidate_id,
            "proposal_rank": self.candidate.proposal_rank,
            "evidence": self.evidence.audit_payload(),
            "selected_metric_value": self.selected_metric_value,
            "canonical_parameter": _candidate_payload(self.candidate, pattern_id),
            "canonical_parameter_sha256": self.canonical_parameter_sha256,
            "exact_intensity_sha256": array_sha256(
                "candidate_exact_intensity", self.candidate.exact_intensity
            ),
            "binding_sha256": self.binding_sha256,
        }


def bind_v5_contextual_reference_candidate(
    scope: V5ContextualReferenceScope,
    task: V5FrozenSearchTask,
    candidate: CandidateInput,
    evidence: V5ContextualCandidateEvidence,
) -> V5BoundReferenceCandidate:
    """Replay all scientific gates and bind one candidate fail-closed."""

    if not isinstance(scope, V5ContextualReferenceScope):
        raise TypeError("scope must be a V5ContextualReferenceScope")
    if not isinstance(task, V5FrozenSearchTask):
        raise TypeError("task must be a V5FrozenSearchTask")
    if not isinstance(candidate, CandidateInput):
        raise TypeError("candidate must be a CandidateInput")
    if not isinstance(evidence, V5ContextualCandidateEvidence):
        raise TypeError("evidence must be V5ContextualCandidateEvidence")
    template = scope.task_template
    shared = (
        task.exact_curve_sha256 == template.exact_curve_sha256
        and task.exact_observation.audit_sha256 == template.exact_observation.audit_sha256
        and task.universal_context.audit_sha256 == template.universal_context.audit_sha256
        and task.query_catalog_artifact_id == template.query_catalog_artifact_id
        and task.query_catalog_artifact_sha256 == template.query_catalog_artifact_sha256
        and task.protocol.sha256 == template.protocol.sha256
        and task.calibrated_threshold is not None
        and template.calibrated_threshold is not None
        and task.calibrated_threshold.sha256 == template.calibrated_threshold.sha256
    )
    if not shared:
        raise ValueError("candidate task escaped the query/observation/protocol/calibration scope")
    expected = _task_evidence_identity(task, scope.source_bundle_sha256)
    for name, value in expected.items():
        if evidence.value(name) != value:
            raise ValueError(f"candidate evidence {name} does not match its bound scope")
    branch = task.branch
    if candidate.topology_id != branch.global_key.topology_id:
        raise ValueError("candidate topology does not match its contextual branch")
    if (candidate.resolution is not None) != branch.condition.resolution_present:
        raise ValueError("candidate Resolution presence does not match its contextual branch")
    if tuple(value.log_D is not None for value in candidate.components) != tuple(
        branch.condition.d_present[: len(candidate.components)]
    ):
        raise ValueError("candidate D presence does not match its contextual branch")
    if candidate.proposal_score_raw is not None:
        raise ValueError("paper reference candidates must be model-independent")
    if not candidate.bounds_pass or not candidate.physics_pass:
        raise ValueError("paper reference candidate must pass bounds and physics gates")
    task.codec.encode(candidate.components, candidate.resolution)
    coefficients = (
        candidate.linear_solution.background,
        *candidate.linear_solution.particle_amplitudes,
        *(
            ()
            if candidate.resolution is None
            else (candidate.linear_solution.resolution_amplitude,)
        ),
    )
    if not task.amplitude_constraint.contains(
        coefficients,
        k=candidate.linear_solution.k,
    ):
        raise ValueError("candidate escaped its concrete complete-query GUI ranges")
    replay = evaluate_gui_forward_snapshot(
        task.observed_curve.q,
        task.codec.latent_components_to_gui(candidate.components),
        resolution=candidate.resolution,
        background=candidate.linear_solution.background,
        particle_amplitudes=candidate.linear_solution.particle_amplitudes,
        resolution_amplitude=candidate.linear_solution.resolution_amplitude,
        gui_k=candidate.linear_solution.k,
    )
    if array_sha256("candidate_exact_intensity", replay) != array_sha256(
        "candidate_exact_intensity", candidate.exact_intensity
    ):
        raise ValueError(
            "candidate exact intensity does not replay through the authoritative forward"
        )
    sigma = task.observed_curve.sigma_log
    metric = natural_log_rmse(replay, task.observed_curve.intensity, sigma_log=sigma)
    if metric > task.selected_threshold_value:
        raise ValueError("candidate is not exact-compatible under the calibrated threshold")
    canonical = _canonical_candidate(task, candidate)
    parameter_payload = _candidate_payload(canonical, branch.global_key.pattern_id)
    parameter_sha = sha256(canonical_json(parameter_payload).encode("utf-8")).hexdigest()
    binding_payload = {
        "scope_sha256": scope.sha256,
        "candidate_id": canonical.candidate_id,
        "evidence": evidence.audit_payload(),
        "selected_metric_value": metric,
        "canonical_parameter_sha256": parameter_sha,
        "branch_query": {
            "geometry_query": json.loads(
                task.universal_context.topology_queries[
                    branch.topology_batch_index
                ].geometry.canonical_json
            ),
            "amplitude_query": json.loads(
                task.universal_context.topology_queries[
                    branch.topology_batch_index
                ].amplitude.canonical_json
            ),
            "amplitude_constraint": task.amplitude_constraint.to_audit_dict(),
        },
    }
    return V5BoundReferenceCandidate(
        scope_sha256=scope.sha256,
        candidate=canonical,
        evidence=evidence,
        selected_metric_value=metric,
        canonical_parameter_sha256=parameter_sha,
        binding_sha256=sha256(canonical_json(binding_payload).encode("utf-8")).hexdigest(),
    )


def _task_for_branch(
    scope: V5ContextualReferenceScope, global_branch_key: str
) -> V5FrozenSearchTask:
    try:
        branch_index = scope.task_template.universal_context.global_branch_keys.index(
            global_branch_key
        )
    except ValueError as exc:
        raise ValueError("candidate branch is absent from the bound universal query") from exc
    return replace(scope.task_template, branch_index=branch_index)


def contextual_reference_parameter_distance(
    scope: V5ContextualReferenceScope,
    left: V5BoundReferenceCandidate,
    right: V5BoundReferenceCandidate,
) -> float:
    """Return query-local RMS distance; different contextual branches are infinite."""

    if not all(isinstance(value, V5BoundReferenceCandidate) for value in (left, right)):
        raise TypeError("left and right must be V5BoundReferenceCandidate values")
    if left.scope_sha256 != scope.sha256 or right.scope_sha256 != scope.sha256:
        raise ValueError("candidate escaped the supplied contextual reference scope")
    left_branch = left.evidence.value("global_branch_key")
    right_branch = right.evidence.value("global_branch_key")
    if left_branch != right_branch:
        return float("inf")
    task = _task_for_branch(scope, left_branch)
    return query_local_parameter_distance(task, left.candidate, right.candidate)


@dataclass(frozen=True)
class V5OperationalReferenceRepresentative:
    representative_id: str
    representative_candidate_id: str
    representative_binding_sha256: str
    member_candidate_ids: tuple[str, ...]
    best_selected_metric_value: float


@dataclass(frozen=True, eq=False)
class V5ContextualReferenceBank:
    scope: V5ContextualReferenceScope
    candidates: tuple[V5BoundReferenceCandidate, ...]
    representatives: tuple[V5OperationalReferenceRepresentative, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scope, V5ContextualReferenceScope):
            raise TypeError("scope must be a V5ContextualReferenceScope")

    def audit_payload(self) -> dict[str, object]:
        return _bank_payload(self.scope, self.candidates, self.representatives)

    @property
    def audit_sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


def _bank_payload(
    scope: V5ContextualReferenceScope,
    candidates: Sequence[V5BoundReferenceCandidate],
    representatives: Sequence[V5OperationalReferenceRepresentative],
) -> dict[str, object]:
    return {
        "schema": V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
        "version": V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
        "claim": V5_CONTEXTUAL_REFERENCE_CLAIM,
        "scope_sha256": scope.sha256,
        "scope": json.loads(scope.audit_json),
        "delta": scope.delta,
        "candidate_count": len(candidates),
        "representative_count": len(representatives),
        "candidates": [value.audit_payload() for value in candidates],
        "representatives": [asdict(value) for value in representatives],
    }


def build_v5_contextual_reference_bank(
    scope: V5ContextualReferenceScope,
    candidates: Sequence[V5BoundReferenceCandidate],
) -> V5ContextualReferenceBank:
    """Build deterministic complete-linkage δ clusters and their representatives."""

    if not isinstance(scope, V5ContextualReferenceScope):
        raise TypeError("scope must be a V5ContextualReferenceScope")
    values = tuple(candidates)
    if not values or not all(isinstance(value, V5BoundReferenceCandidate) for value in values):
        raise ValueError("candidates must contain at least one bound reference candidate")
    if any(value.scope_sha256 != scope.sha256 for value in values):
        raise ValueError("reference-bank candidates do not share one contextual scope")
    if len({value.candidate.candidate_id for value in values}) != len(values):
        raise ValueError("reference-bank candidate IDs must be unique")
    for value in values:
        task = _task_for_branch(scope, value.evidence.value("global_branch_key"))
        replay = bind_v5_contextual_reference_candidate(
            scope, task, value.candidate, value.evidence
        )
        if (
            replay.binding_sha256 != value.binding_sha256
            or replay.canonical_parameter_sha256 != value.canonical_parameter_sha256
            or replay.selected_metric_value != value.selected_metric_value
        ):
            raise ValueError("reference-bank candidate binding does not reproduce")

    def order_key(value: V5BoundReferenceCandidate) -> tuple[float, str, str, str]:
        return (
            value.selected_metric_value,
            value.canonical_parameter_sha256,
            value.candidate.candidate_id,
            value.binding_sha256,
        )

    ordered = tuple(sorted(values, key=order_key))
    groups = complete_linkage_groups(
        ordered,
        lambda left, right: contextual_reference_parameter_distance(scope, left, right),
        scope.delta,
    )
    canonical_groups = [tuple(sorted(group, key=order_key)) for group in groups]
    canonical_groups.sort(key=lambda group: order_key(group[0]))
    representatives = tuple(
        V5OperationalReferenceRepresentative(
            representative_id=f"contextual_reference_{index:04d}",
            representative_candidate_id=group[0].candidate.candidate_id,
            representative_binding_sha256=group[0].binding_sha256,
            member_candidate_ids=tuple(value.candidate.candidate_id for value in group),
            best_selected_metric_value=group[0].selected_metric_value,
        )
        for index, group in enumerate(canonical_groups, 1)
    )
    return V5ContextualReferenceBank(scope, ordered, representatives)


__all__ = [
    "V5_CONTEXTUAL_DEDUPLICATION_VERSION",
    "V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA",
    "V5_CONTEXTUAL_REFERENCE_BANK_VERSION",
    "V5_CONTEXTUAL_REFERENCE_CLAIM",
    "V5_CONTEXTUAL_REFERENCE_DISTANCE_VERSION",
    "V5_CONTEXTUAL_SLOT_EQUIVALENCE_VERSION",
    "V5BoundReferenceCandidate",
    "V5ContextualCandidateEvidence",
    "V5ContextualReferenceBank",
    "V5ContextualReferenceScope",
    "V5OperationalReferenceRepresentative",
    "bind_v5_contextual_reference_candidate",
    "build_v5_contextual_reference_bank",
    "build_v5_contextual_reference_scope",
    "contextual_candidate_evidence_from_task",
    "contextual_reference_parameter_distance",
    "legal_contextual_slot_permutations",
]
