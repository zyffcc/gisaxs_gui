"""Model-ready enumeration of every feasible V5 branch in one user query.

The same immutable adapter is shared by training, evaluation, and production
inference.  It deliberately performs no neural scoring and no physical
acceptance decision: it only repeats one preprocessed observation across the
requested branch contexts without changing their order or semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .amplitude_query_v5 import V5AmplitudeQuery
from .bounds_query_v5 import V5BoundsQuery, V5BranchCondition, branch_condition
from .gui_amplitude_constraints import GuiAmplitudeConstraint
from .model_v5_contract import MODEL_V5_INPUT_KEYS
from .preprocessing import PreprocessedCurve
from .uncertainty_provenance_v5 import V5UncertaintyProvenance
from .universal_query_contract_v5 import V5TopologyQuery


V5_CANDIDATE_BATCH_SCHEMA = "gisaxs.posterior_v8.candidate_context_batch/v3"
V5_CANDIDATE_BATCH_VERSION = (
    "posterior_v8_complete_slot_contract_feasible_query_branches_batch_v3"
)
V5_USER_QUERY_DIGEST_VERSION = "posterior_v8_geometry_plus_amplitude_user_query_digest_v1"


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def _array_digest(values: Mapping[str, np.ndarray]) -> str:
    digest = sha256()
    for name in MODEL_V5_INPUT_KEYS:
        array = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _user_query_digest(query: V5BoundsQuery, amplitude_query: V5AmplitudeQuery) -> str:
    digest = sha256()
    for value in (V5_USER_QUERY_DIGEST_VERSION, query.sha256, amplitude_query.sha256):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class V5CandidateContextBatch:
    """One observation repeated over an ordered feasible branch selection."""

    query: V5BoundsQuery
    amplitude_query: V5AmplitudeQuery
    query_sha256: str
    branch_conditions: tuple[V5BranchCondition, ...]
    amplitude_constraints: tuple[GuiAmplitudeConstraint, ...]
    model_inputs: Mapping[str, np.ndarray]
    model_inputs_sha256: str
    audit_json: str
    audit_sha256: str
    schema_version: str = V5_CANDIDATE_BATCH_SCHEMA
    version: str = V5_CANDIDATE_BATCH_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != V5_CANDIDATE_BATCH_SCHEMA:
            raise ValueError("unsupported V5 candidate-batch schema")
        if self.version != V5_CANDIDATE_BATCH_VERSION:
            raise ValueError("unsupported V5 candidate-batch version")
        if not isinstance(self.query, V5BoundsQuery):
            raise TypeError("query must be a V5BoundsQuery")
        if not isinstance(self.amplitude_query, V5AmplitudeQuery):
            raise TypeError("amplitude_query must be a V5AmplitudeQuery")
        if self.query_sha256 != _user_query_digest(self.query, self.amplitude_query):
            raise ValueError("candidate user-query digest mismatch")
        if self.amplitude_query.particle_count != len(self.query.topology):
            raise ValueError("geometry and amplitude queries have different component counts")
        if (
            self.amplitude_query.resolution_presence_policy
            != self.query.resolution_presence_policy
        ):
            raise ValueError("geometry and amplitude Resolution policies disagree")
        paired_query = V5TopologyQuery(self.query, self.amplitude_query)
        if not self.branch_conditions:
            raise ValueError("candidate batch must contain at least one branch")
        if any(value.query_sha256 != self.query.sha256 for value in self.branch_conditions):
            raise ValueError("candidate branch escaped the selected user query")
        invalid_patterns = tuple(
            value.pattern_id
            for value in self.branch_conditions
            if value.pattern_id not in paired_query.feasible_wire_pattern_ids
        )
        if invalid_patterns:
            raise ValueError(
                "candidate branch escaped the paired geometry/amplitude quotient: "
                f"{invalid_patterns}"
            )
        expected_constraints = tuple(
            self.amplitude_query.constraint_for_branch(
                resolution_present=value.resolution_present
            )
            for value in self.branch_conditions
        )
        if self.amplitude_constraints != expected_constraints:
            raise ValueError("candidate amplitude constraints escaped the user query")
        values = dict(self.model_inputs)
        if tuple(values) != MODEL_V5_INPUT_KEYS:
            raise ValueError("candidate batch inputs do not match the V5 model contract order")
        count = len(self.branch_conditions)
        if any(not isinstance(value, np.ndarray) for value in values.values()):
            raise TypeError("candidate batch model inputs must be NumPy arrays")
        if any(value.shape[0] != count for value in values.values()):
            raise ValueError("every candidate input must have the branch batch dimension")
        if _array_digest(values) != self.model_inputs_sha256:
            raise ValueError("candidate model-input digest mismatch")
        if sha256(self.audit_json.encode("utf-8")).hexdigest() != self.audit_sha256:
            raise ValueError("candidate audit digest mismatch")

    @property
    def branch_count(self) -> int:
        return len(self.branch_conditions)

    @property
    def pattern_ids(self) -> tuple[int, ...]:
        return tuple(value.pattern_id for value in self.branch_conditions)

    def for_model(self) -> dict[str, np.ndarray]:
        """Return a plain ordered dictionary accepted by the Keras graph."""

        return {name: self.model_inputs[name] for name in MODEL_V5_INPUT_KEYS}


def _selected_patterns(
    query: V5BoundsQuery,
    amplitude_query: V5AmplitudeQuery,
    pattern_ids: Sequence[int] | None,
) -> tuple[int, ...]:
    paired_query = V5TopologyQuery(query, amplitude_query)
    selected = (
        paired_query.feasible_wire_pattern_ids
        if pattern_ids is None
        else tuple(int(value) for value in pattern_ids)
    )
    if not selected:
        raise ValueError("pattern_ids must select at least one branch")
    if len(set(selected)) != len(selected):
        raise ValueError("pattern_ids must not contain duplicates")
    invalid = tuple(
        value for value in selected if value not in paired_query.feasible_wire_pattern_ids
    )
    if invalid:
        raise ValueError(f"pattern_ids are not feasible in this query: {invalid}")
    return selected


def build_v5_candidate_context_batch(
    curve: PreprocessedCurve,
    uncertainty: V5UncertaintyProvenance,
    query: V5BoundsQuery,
    amplitude_query: V5AmplitudeQuery,
    *,
    pattern_ids: Sequence[int] | None = None,
) -> V5CandidateContextBatch:
    """Enumerate one query without silently pruning any feasible branch."""

    if not isinstance(curve, PreprocessedCurve):
        raise TypeError("curve must be a PreprocessedCurve")
    if not isinstance(uncertainty, V5UncertaintyProvenance):
        raise TypeError("uncertainty must be V5UncertaintyProvenance")
    if not isinstance(query, V5BoundsQuery):
        raise TypeError("query must be a V5BoundsQuery")
    if not isinstance(amplitude_query, V5AmplitudeQuery):
        raise TypeError("amplitude_query must be a V5AmplitudeQuery")
    if amplitude_query.particle_count != len(query.topology):
        raise ValueError("geometry and amplitude queries have different component counts")
    if amplitude_query.resolution_presence_policy != query.resolution_presence_policy:
        raise ValueError("geometry and amplitude Resolution policies disagree")
    paired_query = V5TopologyQuery(query, amplitude_query)
    selected = _selected_patterns(query, amplitude_query, pattern_ids)
    conditions = tuple(branch_condition(query, value) for value in selected)
    amplitude_constraints = tuple(
        amplitude_query.constraint_for_branch(resolution_present=value.resolution_present)
        for value in conditions
    )
    count = len(conditions)

    x = _readonly(np.repeat(curve.x[np.newaxis, ...], count, axis=0).astype(np.float32))
    point_mask = _readonly(np.repeat(curve.point_mask[np.newaxis, ...], count, axis=0))
    global_features = _readonly(
        np.repeat(curve.global_features[np.newaxis, ...], count, axis=0).astype(np.float32)
    )
    provenance = _readonly(
        np.repeat(
            np.asarray(uncertainty.feature_vector, dtype=np.float32)[np.newaxis, ...],
            count,
            axis=0,
        )
    )
    topology_ids = _readonly(
        np.asarray([[value.topology_id] for value in conditions], dtype=np.int32)
    )
    patterns = _readonly(np.asarray([[value.pattern_id] for value in conditions], dtype=np.int32))
    geometry_bounds = _readonly(
        np.asarray([value.bounds_embedding for value in conditions], dtype=np.float32)
    )
    try:
        intensity_reference = float(curve.stats["intensity_reference"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("preprocessed curve is missing its positive intensity reference") from exc
    amplitude_embedding = amplitude_query.model_embedding(intensity_reference)
    amplitude_bounds = _readonly(
        np.repeat(
            np.asarray(amplitude_embedding, dtype=np.float32)[np.newaxis, ...],
            count,
            axis=0,
        )
    )
    available = _readonly(
        np.asarray([value.available_dimension_mask for value in conditions], dtype=np.float32)
    )
    active = _readonly(
        np.asarray([value.active_dimension_mask for value in conditions], dtype=np.float32)
    )
    varying = _readonly(
        np.asarray([value.varying_dimension_mask for value in conditions], dtype=np.float32)
    )
    values = {
        "x": x,
        "point_mask": point_mask,
        "global_features": global_features,
        "uncertainty_provenance": provenance,
        "branch_topology_id": topology_ids,
        "branch_pattern_id": patterns,
        "geometry_bounds_embedding": geometry_bounds,
        "amplitude_bounds_embedding": amplitude_bounds,
        "available_dimension_mask": available,
        "active_dimension_mask": active,
        "varying_dimension_mask": varying,
    }
    if tuple(values) != MODEL_V5_INPUT_KEYS:  # pragma: no cover - import-time contract guard
        raise RuntimeError("candidate adapter disagrees with V5 model input order")
    input_digest = _array_digest(values)
    user_query_digest = _user_query_digest(query, amplitude_query)
    audit = {
        "schema_version": V5_CANDIDATE_BATCH_SCHEMA,
        "version": V5_CANDIDATE_BATCH_VERSION,
        "query_sha256": user_query_digest,
        "geometry_query_sha256": query.sha256,
        "amplitude_query_sha256": amplitude_query.sha256,
        "topology_id": query.topology_id,
        "pattern_ids": list(selected),
        "contextual_branch_catalog": paired_query.contextual_branch_catalog.audit_payload(),
        "all_feasible_branches_enumerated": (
            selected == paired_query.feasible_wire_pattern_ids
        ),
        "branch_count": count,
        "curve_valid_count": curve.valid_count,
        "preprocessing_version": curve.stats.get("contract", {}).get("version"),
        "intensity_reference": intensity_reference,
        "amplitude_constraints": [
            value.to_audit_dict() for value in amplitude_constraints
        ],
        "uncertainty": uncertainty.audit_payload(),
        "model_input_keys": list(MODEL_V5_INPUT_KEYS),
        "model_inputs_sha256": input_digest,
        "scientific_scope": "context_enumeration_only_no_score_or_acceptance_decision",
    }
    audit_json = json.dumps(audit, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return V5CandidateContextBatch(
        query=query,
        amplitude_query=amplitude_query,
        query_sha256=user_query_digest,
        branch_conditions=conditions,
        amplitude_constraints=amplitude_constraints,
        model_inputs=MappingProxyType(values),
        model_inputs_sha256=input_digest,
        audit_json=audit_json,
        audit_sha256=sha256(audit_json.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "V5_CANDIDATE_BATCH_SCHEMA",
    "V5_CANDIDATE_BATCH_VERSION",
    "V5_USER_QUERY_DIGEST_VERSION",
    "V5CandidateContextBatch",
    "build_v5_candidate_context_batch",
]
