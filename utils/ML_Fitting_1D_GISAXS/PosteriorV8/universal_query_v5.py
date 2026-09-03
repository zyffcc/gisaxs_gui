"""TensorFlow-free cross-topology query bundle for V5.1 one-click inference.

``V5BoundsQuery`` intentionally owns one canonical topology.  A user-facing
one-click search, however, normally compares several K/shape hypotheses for
the same observation.  This module composes those single-topology contracts
without inventing a second codec or flattening topology-specific component
slots into a shared physical parameter vector.

Every selected topology contributes every continuously feasible contextual
wire branch.  The stable global branch key is the pair
``(topology_id, pattern_id)``; the separate context digest binds that branch
to the exact geometry and amplitude query used for one run.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Mapping, Sequence

import numpy as np

from .candidate_batch_v5 import V5CandidateContextBatch, build_v5_candidate_context_batch
from .evaluation import ObservedCurve
from .gui_amplitude_constraints import GuiAmplitudeConstraint
from .model_v5_contract import MODEL_V5_INPUT_KEYS
from .preprocessing import DEFAULT_CONTRACT, PreprocessedCurve
from .uncertainty_provenance_v5 import V5UncertaintyProvenance
from .universal_query_contract_v5 import (
    V5_CONTEXT_BRANCH_KEY_VERSION,
    V5_GLOBAL_BRANCH_KEY_VERSION,
    V5_TOPOLOGY_QUERY_VERSION,
    V5GlobalBranchKey,
    V5TopologyComponentSlot,
    V5TopologyQuery,
    V5UniversalBranchContext,
    amplitude_constraint_sha256,
    branch_context_sha256,
    canonical_universal_json,
    validated_topology_id,
)


V5_UNIVERSAL_QUERY_SCHEMA = "gisaxs.posterior_v8.universal_candidate_context/v2"
V5_UNIVERSAL_QUERY_VERSION = (
    "posterior_v8_same_observation_complete_slot_contract_branches_v2"
)
V5_UNIVERSAL_BRANCH_ORDER = "ascending_topology_id_then_ascending_wire_pattern_id"
V5_UNIVERSAL_DEDUPLICATION_SEMANTICS = (
    "branch_keys_only_deduplicate_search_contexts;after_exact_forward_verification_"
    "parameter_distance_deduplication_is_within_one_declared_topology_and_legal_slot_"
    "equivalence_class;different_topology_slot_layouts_are_never_aligned_and_only_"
    "curve_equivalence_grouping_may_span_topologies"
)
V5_UNIVERSAL_BUDGET_SEMANTICS = (
    "global_branch_key_is_the_atomic_branch_budget_unit;explicit_topology_filtering_"
    "is_allowed_but_no_feasible_wire_branch_inside_a_selected_topology_is_pruned"
)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result

def _selected_topology_ids(
    entries: tuple[V5TopologyQuery, ...],
    allowed_topology_ids: Sequence[int] | None,
) -> tuple[int, ...]:
    available = tuple(value.topology_id for value in entries)
    if len(set(available)) != len(available):
        raise ValueError("topology_queries must contain at most one query per topology")
    if allowed_topology_ids is None:
        return tuple(sorted(available))
    if isinstance(allowed_topology_ids, (str, bytes)):
        raise TypeError("allowed_topology_ids must be a sequence of integer topology IDs")
    try:
        selected = tuple(
            validated_topology_id(value, f"allowed_topology_ids[{index}]")
            for index, value in enumerate(allowed_topology_ids)
        )
    except TypeError as exc:
        if "must be an integer" in str(exc):
            raise
        raise TypeError(
            "allowed_topology_ids must be a sequence of integer topology IDs"
        ) from exc
    if not selected:
        raise ValueError("allowed_topology_ids must select at least one topology")
    if len(set(selected)) != len(selected):
        raise ValueError("allowed_topology_ids must not contain duplicates")
    missing = tuple(sorted(set(selected) - set(available)))
    if missing:
        raise ValueError(f"allowed topology queries were not supplied: {missing}")
    return tuple(sorted(selected))


def _same_observation(batches: Sequence[V5CandidateContextBatch]) -> bool:
    observation_keys = (
        "x",
        "point_mask",
        "global_features",
        "uncertainty_provenance",
    )
    first = batches[0]
    for batch in batches[1:]:
        for name in observation_keys:
            if not np.array_equal(first.model_inputs[name][0], batch.model_inputs[name][0]):
                return False
    return True


def validate_v5_universal_curve_alignment(
    context: V5UniversalCandidateContext,
    curve: ObservedCurve,
) -> None:
    """Fail closed unless encoder context and exact-physics curve are identical."""

    if not isinstance(context, V5UniversalCandidateContext):
        raise TypeError("context must be a V5UniversalCandidateContext")
    if not isinstance(curve, ObservedCurve):
        raise TypeError("curve must be an ObservedCurve")
    first = context.batches[0]
    mask = np.asarray(first.model_inputs["point_mask"][0], dtype=bool)
    encoded = np.asarray(first.model_inputs["x"][0][mask], dtype=np.float64)
    if encoded.shape[0] != curve.q.size:
        raise ValueError("universal context and ObservedCurve have different point counts")
    audit = json.loads(first.audit_json)
    if audit.get("preprocessing_version") != DEFAULT_CONTRACT.version:
        raise ValueError("universal context uses an unsupported preprocessing contract")
    uncertainty_payload = audit.get("uncertainty")
    if not isinstance(uncertainty_payload, Mapping):
        raise ValueError("universal context is missing uncertainty provenance")
    try:
        uncertainty = V5UncertaintyProvenance(
            kind=uncertainty_payload["kind"],
            encoder_relative_sigma_proxy=uncertainty_payload["encoder_relative_sigma_proxy"],
            schema_version=uncertainty_payload["schema_version"],
            version=uncertainty_payload["version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("universal context has invalid uncertainty provenance") from exc
    model_provenance = np.asarray(
        first.model_inputs["uncertainty_provenance"][0], dtype=np.float64
    )
    if not np.array_equal(model_provenance, uncertainty.feature_vector):
        raise ValueError("uncertainty provenance audit and model condition disagree")
    if uncertainty.measurement_sigma_available and curve.sigma_log is None:
        raise ValueError("measured/simulated sigma context requires acceptance sigma_log")
    if not uncertainty.measurement_sigma_available and curve.sigma_log is not None:
        raise ValueError("encoder proxy context must not become acceptance sigma_log evidence")
    intensity_reference = float(audit["intensity_reference"])
    log_q_min = np.log(DEFAULT_CONTRACT.q_min)
    expected = np.column_stack(
        (
            (np.log(curve.q) - log_q_min)
            / (np.log(DEFAULT_CONTRACT.q_max) - log_q_min),
            np.log(curve.intensity / intensity_reference),
        )
    )
    if not np.allclose(encoded[:, :2], expected, rtol=2.0e-6, atol=2.0e-6):
        raise ValueError("universal context was not built from this ObservedCurve")
    relative_sigma = (
        curve.sigma_log
        if uncertainty.measurement_sigma_available
        else np.full(curve.q.shape, float(uncertainty.encoder_relative_sigma_proxy))
    )
    expected_sigma = np.log(curve.intensity * relative_sigma / intensity_reference)
    if not np.allclose(encoded[:, 2], expected_sigma, rtol=2.0e-6, atol=2.0e-6):
        raise ValueError("context encoder uncertainty disagrees with its provenance")


@dataclass(frozen=True)
class V5UniversalCandidateContext:
    """One observation enumerated over a user-selected topology subset."""

    topology_queries: tuple[V5TopologyQuery, ...]
    batches: tuple[V5CandidateContextBatch, ...]
    branches: tuple[V5UniversalBranchContext, ...]
    supplied_topology_ids: tuple[int, ...]
    excluded_topology_ids: tuple[int, ...]
    audit_json: str
    audit_sha256: str
    schema_version: str = V5_UNIVERSAL_QUERY_SCHEMA
    version: str = V5_UNIVERSAL_QUERY_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != V5_UNIVERSAL_QUERY_SCHEMA:
            raise ValueError("unsupported V5 universal-query schema")
        if self.version != V5_UNIVERSAL_QUERY_VERSION:
            raise ValueError("unsupported V5 universal-query version")
        if not self.topology_queries or len(self.topology_queries) != len(self.batches):
            raise ValueError("universal context needs one batch per selected topology query")
        if not all(isinstance(value, V5TopologyQuery) for value in self.topology_queries):
            raise TypeError("topology_queries must contain only V5TopologyQuery values")
        if not all(isinstance(value, V5CandidateContextBatch) for value in self.batches):
            raise TypeError("batches must contain only V5CandidateContextBatch values")
        if not all(isinstance(value, V5UniversalBranchContext) for value in self.branches):
            raise TypeError("branches must contain only V5UniversalBranchContext values")
        topology_ids = tuple(value.topology_id for value in self.topology_queries)
        if topology_ids != tuple(sorted(topology_ids)) or len(set(topology_ids)) != len(
            topology_ids
        ):
            raise ValueError("selected topology queries must use unique stable topology order")
        if tuple(sorted(self.supplied_topology_ids)) != self.supplied_topology_ids or len(
            set(self.supplied_topology_ids)
        ) != len(self.supplied_topology_ids):
            raise ValueError("supplied_topology_ids must be unique and sorted")
        if any(
            validated_topology_id(value, f"supplied_topology_ids[{index}]") != value
            for index, value in enumerate(self.supplied_topology_ids)
        ):
            raise ValueError("supplied_topology_ids are not canonical integers")
        if self.excluded_topology_ids != tuple(sorted(set(self.excluded_topology_ids))):
            raise ValueError("excluded_topology_ids must be unique and sorted")
        if set(topology_ids) | set(self.excluded_topology_ids) != set(
            self.supplied_topology_ids
        ) or set(topology_ids) & set(self.excluded_topology_ids):
            raise ValueError("selected/excluded topology audit does not partition supplied queries")
        if not _same_observation(self.batches):
            raise ValueError("all topology batches must repeat the same observation")

        expected_branches: list[tuple[V5GlobalBranchKey, int, int]] = []
        for topology_batch_index, (entry, batch) in enumerate(
            zip(self.topology_queries, self.batches)
        ):
            if batch.query != entry.geometry or batch.amplitude_query != entry.amplitude:
                raise ValueError("topology batch escaped its paired geometry/amplitude query")
            if batch.pattern_ids != entry.feasible_wire_pattern_ids:
                raise ValueError("selected topology silently lost or reordered a feasible branch")
            expected_branches.extend(
                (
                    V5GlobalBranchKey(entry.topology_id, pattern_id),
                    topology_batch_index,
                    branch_batch_index,
                )
                for branch_batch_index, pattern_id in enumerate(batch.pattern_ids)
            )
        if len(expected_branches) != len(self.branches):
            raise ValueError("universal branch table has the wrong size")
        for global_index, (actual, expected) in enumerate(zip(self.branches, expected_branches)):
            global_key, topology_batch_index, branch_batch_index = expected
            batch = self.batches[topology_batch_index]
            constraint = batch.amplitude_constraints[branch_batch_index]
            constraint_sha = amplitude_constraint_sha256(constraint)
            context_sha = branch_context_sha256(
                global_key, batch.query_sha256, constraint_sha
            )
            if (
                actual.global_index != global_index
                or actual.topology_batch_index != topology_batch_index
                or actual.branch_batch_index != branch_batch_index
                or actual.global_key != global_key
                or actual.topology_query_sha256 != batch.query_sha256
                or actual.condition != batch.branch_conditions[branch_batch_index]
                or actual.amplitude_constraint != constraint
                or actual.amplitude_constraint_sha256 != constraint_sha
                or actual.context_sha256 != context_sha
            ):
                raise ValueError("universal branch row does not reproduce its source batch")
        keys = tuple(value.global_key for value in self.branches)
        if keys != tuple(sorted(keys)) or len(set(keys)) != len(keys):
            raise ValueError("global branch keys must be unique and stable-sorted")
        expected_audit_json = canonical_universal_json(
            _audit_payload(
                entries=self.topology_queries,
                batches=self.batches,
                branches=self.branches,
                supplied_topology_ids=self.supplied_topology_ids,
                excluded_topology_ids=self.excluded_topology_ids,
            )
        )
        if self.audit_json != expected_audit_json:
            raise ValueError("universal query audit does not reproduce its context")
        if sha256(self.audit_json.encode("utf-8")).hexdigest() != self.audit_sha256:
            raise ValueError("universal query audit digest mismatch")

    @property
    def topology_ids(self) -> tuple[int, ...]:
        return tuple(value.topology_id for value in self.topology_queries)

    @property
    def branch_count(self) -> int:
        return len(self.branches)

    @property
    def global_branch_keys(self) -> tuple[str, ...]:
        return tuple(value.global_key.wire_key for value in self.branches)

    @property
    def batch_slices(self) -> tuple[slice, ...]:
        result = []
        start = 0
        for batch in self.batches:
            stop = start + batch.branch_count
            result.append(slice(start, stop))
            start = stop
        return tuple(result)

    def for_model(self) -> dict[str, np.ndarray]:
        """Concatenate stable topology batches for one framework-neutral model call."""

        values = {
            name: _readonly(
                np.concatenate([batch.model_inputs[name] for batch in self.batches], axis=0)
            )
            for name in MODEL_V5_INPUT_KEYS
        }
        return values

    def branch_for(self, global_branch_key: str) -> V5UniversalBranchContext:
        matches = tuple(
            value for value in self.branches if value.global_key.wire_key == global_branch_key
        )
        if len(matches) != 1:
            raise KeyError(f"unknown global branch key {global_branch_key!r}")
        return matches[0]

    def amplitude_constraint_for(self, global_branch_key: str) -> GuiAmplitudeConstraint:
        return self.branch_for(global_branch_key).amplitude_constraint

    def codec_for(self, global_branch_key: str):
        """Resolve through the owning topology query, never through another slot layout."""

        branch = self.branch_for(global_branch_key)
        query = self.topology_queries[branch.topology_batch_index].geometry
        return query.codec_for(branch.global_key.pattern_id)


def _audit_payload(
    *,
    entries: tuple[V5TopologyQuery, ...],
    batches: tuple[V5CandidateContextBatch, ...],
    branches: tuple[V5UniversalBranchContext, ...],
    supplied_topology_ids: tuple[int, ...],
    excluded_topology_ids: tuple[int, ...],
) -> dict[str, object]:
    slices = []
    start = 0
    topology_payloads = []
    for entry, batch in zip(entries, batches):
        stop = start + batch.branch_count
        slices.append([start, stop])
        topology_payloads.append(
            {
                **entry.audit_payload(),
                "topology_query_sha256": entry.sha256,
                "candidate_query_sha256": batch.query_sha256,
                "candidate_batch_audit_sha256": batch.audit_sha256,
                "global_row_slice": [start, stop],
                "global_branch_keys": [
                    f"topology-{entry.topology_id:02d}:wire-{value:02d}"
                    for value in batch.pattern_ids
                ],
                "all_feasible_wire_branches_enumerated": (
                    batch.pattern_ids == entry.feasible_wire_pattern_ids
                ),
                "amplitude_polytopes": [
                    {
                        "global_branch_key": (
                            f"topology-{entry.topology_id:02d}:wire-{condition.pattern_id:02d}"
                        ),
                        "sha256": amplitude_constraint_sha256(constraint),
                        "constraint": constraint.to_audit_dict(),
                    }
                    for condition, constraint in zip(
                        batch.branch_conditions, batch.amplitude_constraints
                    )
                ],
            }
        )
        start = stop
    return {
        "schema_version": V5_UNIVERSAL_QUERY_SCHEMA,
        "version": V5_UNIVERSAL_QUERY_VERSION,
        "branch_key_version": V5_GLOBAL_BRANCH_KEY_VERSION,
        "context_key_version": V5_CONTEXT_BRANCH_KEY_VERSION,
        "branch_order": V5_UNIVERSAL_BRANCH_ORDER,
        "supplied_topology_ids": list(supplied_topology_ids),
        "selected_topology_ids": [value.topology_id for value in entries],
        "excluded_topology_ids": list(excluded_topology_ids),
        "topology_query_count": len(entries),
        "branch_count": len(branches),
        "same_observation_repeated_across_topologies": _same_observation(batches),
        "all_selected_topology_feasible_branches_enumerated": all(
            batch.pattern_ids == entry.feasible_wire_pattern_ids
            for entry, batch in zip(entries, batches)
        ),
        "global_row_slices": slices,
        "topology_queries": topology_payloads,
        "branches": [value.audit_payload() for value in branches],
        "deduplication_semantics": V5_UNIVERSAL_DEDUPLICATION_SEMANTICS,
        "budget_allocation_semantics": V5_UNIVERSAL_BUDGET_SEMANTICS,
        "claim_limits": {
            "branch_enumeration_is_model_selection": False,
            "neural_score_is_exact_compatibility": False,
            "zero_candidates_is_no_solution_certificate": False,
            "final_candidates_require_exact_forward_verification": True,
        },
    }


def build_v5_universal_candidate_context(
    curve: PreprocessedCurve,
    uncertainty: V5UncertaintyProvenance,
    topology_queries: Sequence[V5TopologyQuery],
    *,
    allowed_topology_ids: Sequence[int] | None = None,
) -> V5UniversalCandidateContext:
    """Enumerate every feasible branch for each explicitly selected topology.

    ``allowed_topology_ids`` is the only pruning control.  It applies at the
    topology level and is recorded in the audit; branches inside a retained
    topology can never be silently dropped here.
    """

    if not isinstance(curve, PreprocessedCurve):
        raise TypeError("curve must be a PreprocessedCurve")
    if not isinstance(uncertainty, V5UncertaintyProvenance):
        raise TypeError("uncertainty must be V5UncertaintyProvenance")
    if isinstance(topology_queries, (str, bytes)):
        raise TypeError("topology_queries must be a sequence of V5TopologyQuery values")
    try:
        supplied = tuple(topology_queries)
    except TypeError as exc:
        raise TypeError(
            "topology_queries must be a sequence of V5TopologyQuery values"
        ) from exc
    if not supplied:
        raise ValueError("topology_queries must contain at least one topology")
    if not all(isinstance(value, V5TopologyQuery) for value in supplied):
        raise TypeError("topology_queries must contain only V5TopologyQuery values")
    selected_ids = _selected_topology_ids(supplied, allowed_topology_ids)
    supplied_ids = tuple(sorted(value.topology_id for value in supplied))
    by_topology = {value.topology_id: value for value in supplied}
    selected = tuple(by_topology[value] for value in selected_ids)
    excluded = tuple(value for value in supplied_ids if value not in set(selected_ids))

    batches = tuple(
        build_v5_candidate_context_batch(
            curve,
            uncertainty,
            entry.geometry,
            entry.amplitude,
        )
        for entry in selected
    )
    branches = []
    global_index = 0
    for topology_batch_index, batch in enumerate(batches):
        for branch_batch_index, (condition, constraint) in enumerate(
            zip(batch.branch_conditions, batch.amplitude_constraints)
        ):
            global_key = V5GlobalBranchKey(condition.topology_id, condition.pattern_id)
            constraint_sha = amplitude_constraint_sha256(constraint)
            branches.append(
                V5UniversalBranchContext(
                    global_index=global_index,
                    topology_batch_index=topology_batch_index,
                    branch_batch_index=branch_batch_index,
                    global_key=global_key,
                    topology_query_sha256=batch.query_sha256,
                    condition=condition,
                    amplitude_constraint=constraint,
                    amplitude_constraint_sha256=constraint_sha,
                    context_sha256=branch_context_sha256(
                        global_key,
                        batch.query_sha256,
                        constraint_sha,
                    ),
                )
            )
            global_index += 1
    frozen_branches = tuple(branches)
    audit = _audit_payload(
        entries=selected,
        batches=batches,
        branches=frozen_branches,
        supplied_topology_ids=supplied_ids,
        excluded_topology_ids=excluded,
    )
    audit_json = canonical_universal_json(audit)
    return V5UniversalCandidateContext(
        topology_queries=selected,
        batches=batches,
        branches=frozen_branches,
        supplied_topology_ids=supplied_ids,
        excluded_topology_ids=excluded,
        audit_json=audit_json,
        audit_sha256=sha256(audit_json.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "V5_CONTEXT_BRANCH_KEY_VERSION",
    "V5_GLOBAL_BRANCH_KEY_VERSION",
    "V5_TOPOLOGY_QUERY_VERSION",
    "V5_UNIVERSAL_BRANCH_ORDER",
    "V5_UNIVERSAL_BUDGET_SEMANTICS",
    "V5_UNIVERSAL_DEDUPLICATION_SEMANTICS",
    "V5_UNIVERSAL_QUERY_SCHEMA",
    "V5_UNIVERSAL_QUERY_VERSION",
    "V5GlobalBranchKey",
    "V5TopologyComponentSlot",
    "V5TopologyQuery",
    "V5UniversalBranchContext",
    "V5UniversalCandidateContext",
    "build_v5_universal_candidate_context",
    "validate_v5_universal_curve_alignment",
]
