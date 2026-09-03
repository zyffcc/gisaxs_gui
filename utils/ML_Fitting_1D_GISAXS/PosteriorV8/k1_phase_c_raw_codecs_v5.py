"""Primitive codecs for lossless K1 Phase-C filesystem artifacts.

These codecs restore query geometry, typed representative payloads, provenance,
sidecars, and shared receipts.  They only read already hash-checked JSON; the
filesystem adapter owns byte and identity verification.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
import json
from types import SimpleNamespace
from typing import Mapping

import numpy as np

from .amplitude_query_v5 import (
    V5_AMPLITUDE_EMBEDDING_VERSION,
    V5_AMPLITUDE_QUERY_SCHEMA,
    V5_AMPLITUDE_QUERY_VERSION,
    V5AmplitudeQuery,
)
from .bounds_query_v5 import branch_condition, bounds_query_from_json
from .contract import ClosedInterval, LatentComponentParameters
from .evaluation import CandidateInput, LinearSolutionSnapshot, ReferenceMode
from .grouped_artifact_v5 import array_sha256, canonical_json
from .k1_phase_c_contract_v5 import digest
from .k1_phase_c_replay_contract_v5 import (
    V5K1PhaseCBranchSearchReplay,
    V5K1PhaseCCandidateJudgement,
    V5K1PhaseCParentProvenance,
    V5K1PhaseCSplitReplayReceipt,
)
from .k1_phase_c_replay_receipt_v5 import V5K1PhaseCArtifactBinding
from .paper_budget_evaluator_v5 import (
    V5ExactForwardCall,
    V5PaperBudgetEvaluationConfig,
)
from .paper_representative_payload_v5 import (
    V5_EMITTED_REPRESENTATIVE_ROLE,
    V5_REFERENCE_REPRESENTATIVE_ROLE,
    V5PaperParameterRepresentativePayload,
)
from .profiled_forward import ResolutionShape
from .query_parameter_distance_v5 import query_local_parameter_distance
from .search_supervision_sidecar_v5 import V5_SEARCH_SIDECAR_SCHEMA, V5_SEARCH_SIDECAR_VERSION
from .sobol_numeric_canonicalization_v5 import v5_numeric_policy_sha256
from .universal_query_contract_v5 import V5GlobalBranchKey, V5TopologyQuery


V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_raw_manifest/v1"
V5_K1_PHASE_C_RAW_MANIFEST_VERSION = (
    "lossless_per_file_sha_bound_phase_c_replay_inputs_v1"
)
V5_K1_PHASE_C_RAW_ARTIFACT_VERSION = "lossless_typed_phase_c_raw_json_v1"
V5_K1_PHASE_C_RAW_ARTIFACT_SHA_SEMANTICS = (
    "sha256_of_exact_canonical_json_bytes_of_the_lossless_artifact;"
    "never_an_upstream_audit_digest_or_an_unopened_source_path"
)

RAW_ARTIFACT_BINDING_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_artifact_binding_raw/v1"
RAW_PARENT_PROVENANCE_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_parent_provenance_raw/v1"
RAW_SPLIT_RECEIPT_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_split_receipt_raw/v1"
RAW_EVALUATOR_CONFIG_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_evaluator_config_raw/v1"
RAW_DISTANCE_CONTEXT_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_distance_context_raw/v1"
RAW_BRANCH_SIDECAR_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_branch_search_lossless/v1"
RAW_REFERENCE_BANK_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_reference_bank_raw/v1"
RAW_REFERENCE_TRACE_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_reference_trace_raw/v1"
RAW_METHOD_TRACE_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_method_trace_raw/v1"
RAW_REPRESENTATIVE_PAYLOAD_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_c_representative_payload_raw/v1"
)


def _field_names(cls: type) -> set[str]:
    return {value.name for value in fields(cls) if value.init}


def _object(value: object, expected: set[str], name: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{name} fields are incomplete or unsupported")
    return dict(value)


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty stripped string")
    return value


def _raw_envelope(value: object, schema: str, fields_: set[str], name: str) -> dict[str, object]:
    row = _object(value, {"schema", "version", *fields_}, name)
    if (row["schema"], row["version"]) != (schema, V5_K1_PHASE_C_RAW_ARTIFACT_VERSION):
        raise ValueError(f"unsupported {name} schema/version")
    return row


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCRawFile:
    file_id: str
    role: str
    relative_path: str
    file_sha256: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "file_id", _text(self.file_id, "file_id"))
        object.__setattr__(self, "role", _text(self.role, "role"))
        object.__setattr__(self, "relative_path", _text(self.relative_path, "relative_path"))
        object.__setattr__(self, "file_sha256", digest(self.file_sha256, "file_sha256"))
        if not isinstance(self.payload, Mapping):
            raise TypeError("raw artifact payload must be an object")


class V5K1PhaseCQueryDistanceMatcher:
    """Recomputed query-local distance over losslessly restored GUI queries."""

    def __init__(self, tasks: Mapping[tuple[str, str], object]) -> None:
        if not tasks:
            raise ValueError("distance matcher requires at least one query-bound branch")
        self._tasks = dict(tasks)
        identity = [list(value) for value in sorted(self._tasks)]
        self.identity_sha256 = sha256(canonical_json(identity).encode("utf-8")).hexdigest()

    def __call__(
        self,
        left: V5PaperParameterRepresentativePayload,
        right: V5PaperParameterRepresentativePayload,
    ) -> float | None:
        if not isinstance(left, V5PaperParameterRepresentativePayload) or not isinstance(
            right, V5PaperParameterRepresentativePayload
        ):
            raise TypeError("distance operands must be typed representative payloads")
        if left.query_context_sha256 != right.query_context_sha256:
            raise ValueError("distance operands escaped their one query context")
        if left.global_branch_key != right.global_branch_key:
            return float("inf")
        key = (left.query_context_sha256, left.global_branch_key)
        try:
            task = self._tasks[key]
        except KeyError as exc:
            raise ValueError("representative branch is absent from its raw distance context") from exc
        return query_local_parameter_distance(task, left.parameter, right.parameter)


def _parse_amplitude_query(encoded: object, expected_sha256: object) -> V5AmplitudeQuery:
    text = _text(encoded, "amplitude_query_canonical_json")
    expected = digest(expected_sha256, "amplitude_query_sha256")
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("amplitude query is not valid JSON") from exc
    expected_fields = {
        "schema",
        "version",
        "embedding_version",
        "gui_amplitude_constraint_version",
        "canonical_gauge",
        "numeric_policy_version",
        "numeric_policy_sha256",
        "background",
        "k",
        "component_intensities",
        "resolution_presence_policy",
        "int_res",
    }
    row = _object(payload, expected_fields, "amplitude query")
    if canonical_json(row) != text:
        raise ValueError("amplitude query must be canonical JSON")
    if (
        row["schema"] != V5_AMPLITUDE_QUERY_SCHEMA
        or row["version"] != V5_AMPLITUDE_QUERY_VERSION
        or row["embedding_version"] != V5_AMPLITUDE_EMBEDDING_VERSION
        or row["numeric_policy_sha256"]
        != v5_numeric_policy_sha256(row["numeric_policy_version"])
    ):
        raise ValueError("amplitude query contract drifted")

    def interval(value: object, name: str) -> ClosedInterval:
        item = _object(value, {"low", "high"}, name)
        return ClosedInterval(low=item["low"], high=item["high"])

    components = tuple(
        interval(value, f"component_intensities[{index}]")
        for index, value in enumerate(_sequence(row["component_intensities"], "component_intensities"))
    )
    int_res = None if row["int_res"] is None else interval(row["int_res"], "int_res")
    result = V5AmplitudeQuery.create(
        background=interval(row["background"], "background"),
        k=interval(row["k"], "k"),
        component_intensities=components,
        resolution_presence_policy=row["resolution_presence_policy"],
        int_res=int_res,
        numeric_policy_version=row["numeric_policy_version"],
    )
    if result.canonical_json != text or result.sha256 != expected:
        raise ValueError("amplitude query JSON/SHA-256 does not reproduce")
    return result


def _parse_distance_context(
    raw: V5K1PhaseCRawFile,
    provenance: V5K1PhaseCParentProvenance,
) -> dict[tuple[str, str], object]:
    row = _raw_envelope(
        raw.payload,
        RAW_DISTANCE_CONTEXT_SCHEMA,
        {"query_context_sha256", "universal_query_sha256", "topology_queries"},
        "distance-context artifact",
    )
    if (
        digest(row["query_context_sha256"], "query_context_sha256")
        != provenance.evaluation_query_context_sha256
        or digest(row["universal_query_sha256"], "universal_query_sha256")
        != provenance.universal_query_sha256
    ):
        raise ValueError("raw distance context escaped the parent query identity")
    query_fields = {
        "topology_id",
        "topology_query_sha256",
        "geometry_query_canonical_json",
        "geometry_query_sha256",
        "amplitude_query_canonical_json",
        "amplitude_query_sha256",
    }
    tasks: dict[tuple[str, str], object] = {}
    topology_ids = []
    for index, value in enumerate(_sequence(row["topology_queries"], "topology_queries")):
        item = _object(value, query_fields, f"topology_queries[{index}]")
        geometry = bounds_query_from_json(
            _text(item["geometry_query_canonical_json"], "geometry query JSON"),
            digest(item["geometry_query_sha256"], "geometry_query_sha256"),
        )
        amplitude = _parse_amplitude_query(
            item["amplitude_query_canonical_json"], item["amplitude_query_sha256"]
        )
        query = V5TopologyQuery(geometry=geometry, amplitude=amplitude)
        if query.topology_id != item["topology_id"] or query.sha256 != digest(
            item["topology_query_sha256"], "topology_query_sha256"
        ):
            raise ValueError("topology query identity does not reproduce")
        topology_ids.append(query.topology_id)
        for pattern_id in query.feasible_wire_pattern_ids:
            key = V5GlobalBranchKey(query.topology_id, pattern_id)
            condition = branch_condition(geometry, pattern_id)
            constraint = amplitude.constraint_for_branch(
                resolution_present=condition.resolution_present
            )
            task = SimpleNamespace(
                branch=SimpleNamespace(
                    topology_batch_index=0,
                    global_key=key,
                    condition=condition,
                ),
                codec=geometry.codec_for(pattern_id),
                amplitude_constraint=constraint,
                universal_context=SimpleNamespace(topology_queries=(query,)),
            )
            tasks[(provenance.evaluation_query_context_sha256, key.wire_key)] = task
    if not topology_ids or topology_ids != sorted(set(topology_ids)):
        raise ValueError("distance-context topology queries must be unique and sorted")
    return tasks


def _linear(value: object) -> LinearSolutionSnapshot:
    row = _object(
        value,
        {"background", "particle_amplitudes", "resolution_amplitude", "k"},
        "linear_solution",
    )
    return LinearSolutionSnapshot(
        background=row["background"],
        particle_amplitudes=tuple(_sequence(row["particle_amplitudes"], "particle_amplitudes")),
        resolution_amplitude=row["resolution_amplitude"],
        k=row["k"],
    )


def _components(value: object) -> tuple[LatentComponentParameters, ...]:
    expected = _field_names(LatentComponentParameters)
    return tuple(
        LatentComponentParameters(**_object(item, expected, f"components[{index}]"))
        for index, item in enumerate(_sequence(value, "components"))
    )


def _resolution(value: object) -> ResolutionShape | None:
    if value is None:
        return None
    return ResolutionShape(**_object(value, {"sigma_res", "nu_res"}, "resolution"))


def _parse_representative(raw: V5K1PhaseCRawFile) -> V5PaperParameterRepresentativePayload:
    row = _raw_envelope(
        raw.payload,
        RAW_REPRESENTATIVE_PAYLOAD_SCHEMA,
        {"representative_id", "role", "global_branch_key", "query_context_sha256", "parameter"},
        "representative-payload artifact",
    )
    role = row["role"]
    common = {"topology_id", "components", "resolution", "linear_solution"}
    if role == V5_REFERENCE_REPRESENTATIVE_ROLE:
        parameter_row = _object(row["parameter"], {"reference_id", *common}, "reference parameter")
        parameter = ReferenceMode(
            reference_id=parameter_row["reference_id"],
            topology_id=parameter_row["topology_id"],
            components=_components(parameter_row["components"]),
            resolution=_resolution(parameter_row["resolution"]),
            linear_solution=_linear(parameter_row["linear_solution"]),
        )
    elif role == V5_EMITTED_REPRESENTATIVE_ROLE:
        candidate_fields = {
            "candidate_id",
            "proposal_rank",
            "exact_intensity",
            "exact_intensity_dtype",
            "exact_intensity_shape",
            "exact_intensity_order",
            "exact_intensity_sha256",
            "bounds_pass",
            "physics_pass",
            "proposal_score_raw",
            *common,
        }
        parameter_row = _object(row["parameter"], candidate_fields, "candidate parameter")
        if (
            parameter_row["exact_intensity_dtype"] != "<f8"
            or parameter_row["exact_intensity_order"] != "C"
            or not isinstance(parameter_row["exact_intensity_shape"], list)
            or len(parameter_row["exact_intensity_shape"]) != 1
            or isinstance(parameter_row["exact_intensity_shape"][0], bool)
            or not isinstance(parameter_row["exact_intensity_shape"][0], int)
            or parameter_row["exact_intensity_shape"][0] < 1
        ):
            raise ValueError("candidate exact intensity must be non-empty 1-D little-endian f8 C-order")
        exact_values = _sequence(parameter_row["exact_intensity"], "exact_intensity")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in exact_values
        ):
            raise ValueError("candidate exact intensity requires JSON numeric scalars, not bools")
        exact = np.ascontiguousarray(np.asarray(exact_values, dtype=np.dtype("<f8")))
        if not np.all(np.isfinite(exact)) or np.any(exact <= 0.0):
            raise ValueError("candidate exact intensity values must be finite and strictly positive")
        if list(exact.shape) != parameter_row["exact_intensity_shape"]:
            raise ValueError("candidate exact-intensity shape does not reproduce")
        expected_intensity_sha = digest(
            parameter_row["exact_intensity_sha256"], "exact_intensity_sha256"
        )
        if array_sha256("candidate_exact_intensity", exact) != expected_intensity_sha:
            raise ValueError("candidate exact-intensity SHA-256 does not reproduce")
        parameter = CandidateInput(
            candidate_id=parameter_row["candidate_id"],
            proposal_rank=parameter_row["proposal_rank"],
            topology_id=parameter_row["topology_id"],
            components=_components(parameter_row["components"]),
            resolution=_resolution(parameter_row["resolution"]),
            linear_solution=_linear(parameter_row["linear_solution"]),
            exact_intensity=exact,
            bounds_pass=parameter_row["bounds_pass"],
            physics_pass=parameter_row["physics_pass"],
            proposal_score_raw=parameter_row["proposal_score_raw"],
        )
    else:
        raise ValueError("representative role is unsupported")
    return V5PaperParameterRepresentativePayload(
        representative_id=row["representative_id"],
        role=role,
        parameter=parameter,
        global_branch_key=row["global_branch_key"],
        query_context_sha256=row["query_context_sha256"],
        source_artifact_sha256=raw.file_sha256,
    )


def _judgement(value: object, name: str) -> V5K1PhaseCCandidateJudgement:
    return V5K1PhaseCCandidateJudgement(
        **_object(value, _field_names(V5K1PhaseCCandidateJudgement), name)
    )


def _exact_calls(value: object, name: str) -> tuple[V5ExactForwardCall, ...]:
    return tuple(
        V5ExactForwardCall(**_object(item, _field_names(V5ExactForwardCall), f"{name}[{index}]"))
        for index, item in enumerate(_sequence(value, name))
    )


def _parse_parent_provenance(raw: V5K1PhaseCRawFile) -> V5K1PhaseCParentProvenance:
    row = _raw_envelope(
        raw.payload,
        RAW_PARENT_PROVENANCE_SCHEMA,
        {"provenance"},
        "parent-provenance artifact",
    )
    values = _object(
        row["provenance"], _field_names(V5K1PhaseCParentProvenance), "parent provenance"
    )
    tuple_fields = {
        "observation_effects",
        "geometry_axis_regimes",
        "geometry_axis_placements",
        "geometry_axis_coordinate_sha256s",
        "amplitude_axis_regimes",
        "amplitude_axis_coordinate_sha256s",
    }
    for name in tuple_fields:
        values[name] = tuple(_sequence(values[name], name))
    return V5K1PhaseCParentProvenance(**values)


def _parse_branch_sidecar(raw: V5K1PhaseCRawFile) -> V5K1PhaseCBranchSearchReplay:
    row = _raw_envelope(
        raw.payload,
        RAW_BRANCH_SIDECAR_SCHEMA,
        {
            "search_evidence_schema",
            "search_evidence_version",
            "branch_id",
            "frozen_search_yield_rank",
            "completed",
            "candidates",
        },
        "branch-search sidecar",
    )
    if (row["search_evidence_schema"], row["search_evidence_version"]) != (
        V5_SEARCH_SIDECAR_SCHEMA,
        V5_SEARCH_SIDECAR_VERSION,
    ):
        raise ValueError("branch sidecar did not originate from the current search contract")
    return V5K1PhaseCBranchSearchReplay(
        branch_id=row["branch_id"],
        search_sidecar_sha256=raw.file_sha256,
        frozen_search_yield_rank=row["frozen_search_yield_rank"],
        completed=row["completed"],
        candidates=tuple(
            _judgement(value, f"branch candidates[{index}]")
            for index, value in enumerate(_sequence(row["candidates"], "candidates"))
        ),
    )


def _artifact_binding(raw: V5K1PhaseCRawFile) -> V5K1PhaseCArtifactBinding:
    row = _raw_envelope(
        raw.payload,
        RAW_ARTIFACT_BINDING_SCHEMA,
        {"artifact_binding", "source_provenance", "model_provenance"},
        "artifact-binding provenance",
    )
    binding_values = _object(
        row["artifact_binding"], _field_names(V5K1PhaseCArtifactBinding), "artifact_binding"
    )
    binding = V5K1PhaseCArtifactBinding(**binding_values)
    source = _object(
        row["source_provenance"],
        {"source_archive_sha256", "source_manifest_sha256", "source_tree_sha256", "source_bundle_sha256"},
        "source_provenance",
    )
    model = _object(
        row["model_provenance"],
        {
            "cross_platform_gate_claim_sha256",
            "phase_a_launch_receipt_sha256",
            "model_artifact_sha256",
            "model_weights_sha256",
            "model_training_result_sha256",
            "proposal_execution_policy_sha256",
        },
        "model_provenance",
    )
    expected_source = {name: getattr(binding, name) for name in source}
    expected_model = {name: getattr(binding, name) for name in model}
    if source != expected_source or model != expected_model:
        raise ValueError("source/model provenance disagrees with the artifact binding")
    return binding


def _split_receipt(raw: V5K1PhaseCRawFile) -> V5K1PhaseCSplitReplayReceipt:
    row = _raw_envelope(
        raw.payload,
        RAW_SPLIT_RECEIPT_SCHEMA,
        {
            "split_id",
            "plan_sha256",
            "included_clean_parent_sha256s",
            "excluded_population_sha256s",
            "disjointness_verified",
        },
        "split receipt",
    )
    return V5K1PhaseCSplitReplayReceipt(
        split_id=row["split_id"],
        artifact_sha256=raw.file_sha256,
        plan_sha256=row["plan_sha256"],
        included_clean_parent_sha256s=tuple(
            _sequence(row["included_clean_parent_sha256s"], "included clean parents")
        ),
        excluded_population_sha256s=tuple(
            _sequence(row["excluded_population_sha256s"], "excluded populations")
        ),
        disjointness_verified=row["disjointness_verified"],
    )


def _evaluator_config(raw: V5K1PhaseCRawFile) -> V5PaperBudgetEvaluationConfig:
    row = _raw_envelope(
        raw.payload,
        RAW_EVALUATOR_CONFIG_SCHEMA,
        {"config"},
        "evaluator config",
    )
    values = _object(
        row["config"], _field_names(V5PaperBudgetEvaluationConfig), "evaluator config"
    )
    values["output_caps"] = tuple(_sequence(values["output_caps"], "output_caps"))
    values["exact_forward_budgets"] = tuple(
        _sequence(values["exact_forward_budgets"], "exact_forward_budgets")
    )
    return V5PaperBudgetEvaluationConfig(**values)


__all__ = [
    "RAW_ARTIFACT_BINDING_SCHEMA",
    "RAW_BRANCH_SIDECAR_SCHEMA",
    "RAW_DISTANCE_CONTEXT_SCHEMA",
    "RAW_EVALUATOR_CONFIG_SCHEMA",
    "RAW_METHOD_TRACE_SCHEMA",
    "RAW_PARENT_PROVENANCE_SCHEMA",
    "RAW_REFERENCE_BANK_SCHEMA",
    "RAW_REFERENCE_TRACE_SCHEMA",
    "RAW_REPRESENTATIVE_PAYLOAD_SCHEMA",
    "RAW_SPLIT_RECEIPT_SCHEMA",
    "V5_K1_PHASE_C_RAW_ARTIFACT_VERSION",
    "V5_K1_PHASE_C_RAW_ARTIFACT_SHA_SEMANTICS",
    "V5_K1_PHASE_C_RAW_MANIFEST_SCHEMA",
    "V5_K1_PHASE_C_RAW_MANIFEST_VERSION",
    "V5K1PhaseCQueryDistanceMatcher",
    "V5K1PhaseCRawFile",
]
