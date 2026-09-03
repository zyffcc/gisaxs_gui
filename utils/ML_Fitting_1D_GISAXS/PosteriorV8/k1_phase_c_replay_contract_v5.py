"""Typed raw evidence contract for replaying the K1 Phase-C evaluation.

The objects in this module deliberately contain observations, traces, and
candidate judgements rather than the aggregate counts consumed by the gate.
The runner is the only owner of the raw-evidence-to-parent-record reduction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Protocol

from .contextual_reference_bank_v5 import (
    V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
    V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
)
from .grouped_artifact_v5 import canonical_json
from .k1_phase_c_contract_v5 import digest
from .k1_phase_c_plan_v5 import V5K1PhaseCPlan
from .k1_phase_c_replay_receipt_v5 import V5K1PhaseCArtifactBinding
from .paper_budget_evaluator_v5 import (
    EquivalenceDistanceMatcher,
    V5_EXACT_COMPATIBILITY_STATUSES,
    V5_EXACT_COMPATIBLE,
    V5ExactForwardCall,
    V5FrozenReferenceSet,
    V5MethodExactCallTrace,
    V5PaperBudgetEvaluationConfig,
)
from .query_parameter_distance_v5 import (
    V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
    V5_QUERY_PARAMETER_DISTANCE_SHA256,
    V5_QUERY_PARAMETER_DISTANCE_VERSION,
)
from .proposal_execution_policy_v5 import V5_PROPOSAL_EXECUTION_POLICY_SHA256
from .search_supervision_sidecar_v5 import V5_SEARCH_SIDECAR_SCHEMA, V5_SEARCH_SIDECAR_VERSION


V5_K1_PHASE_C_REPLAY_INPUT_SCHEMA = "gisaxs.posterior_v8.k1_phase_c_replay_input/v3"
V5_K1_PHASE_C_REPLAY_INPUT_VERSION = (
    "policy_and_full_source_bound_raw_emission_contiguous_exact_call_evidence_v3"
)
V5_K1_PHASE_C_SPLIT_REPLAY_RECEIPT_SCHEMA = (
    "gisaxs.posterior_v8.k1_phase_c_split_replay_receipt/v1"
)
V5_K1_PHASE_C_SPLIT_REPLAY_RECEIPT_VERSION = (
    "exact_included_cohort_and_excluded_population_binding_v1"
)
V5_K1_PHASE_C_METHOD_COMPLETED = "completed"
V5_K1_PHASE_C_METHOD_CRASHED_ZERO = "crashed_zero_scored"
V5_K1_PHASE_C_METHOD_STATUSES = (
    V5_K1_PHASE_C_METHOD_COMPLETED,
    V5_K1_PHASE_C_METHOD_CRASHED_ZERO,
)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a non-empty stripped string")
    return value


def _explicit_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be an explicit bool")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCSplitReplayReceipt:
    split_id: str
    artifact_sha256: str
    plan_sha256: str
    included_clean_parent_sha256s: tuple[str, ...]
    excluded_population_sha256s: tuple[str, ...]
    disjointness_verified: bool
    schema: str = V5_K1_PHASE_C_SPLIT_REPLAY_RECEIPT_SCHEMA
    version: str = V5_K1_PHASE_C_SPLIT_REPLAY_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if (self.schema, self.version) != (
            V5_K1_PHASE_C_SPLIT_REPLAY_RECEIPT_SCHEMA,
            V5_K1_PHASE_C_SPLIT_REPLAY_RECEIPT_VERSION,
        ):
            raise ValueError("unsupported K1 Phase-C split-replay receipt schema/version")
        object.__setattr__(self, "split_id", _text(self.split_id, "split_id"))
        for name in ("artifact_sha256", "plan_sha256"):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        included = tuple(
            digest(value, f"included_clean_parent_sha256s[{index}]")
            for index, value in enumerate(self.included_clean_parent_sha256s)
        )
        excluded = tuple(
            digest(value, f"excluded_population_sha256s[{index}]")
            for index, value in enumerate(self.excluded_population_sha256s)
        )
        if not included or len(included) != len(set(included)):
            raise ValueError("split receipt requires unique included clean parents")
        if not excluded or len(excluded) != len(set(excluded)):
            raise ValueError("split receipt requires explicit unique excluded populations")
        if set(included) & set(excluded):
            raise ValueError("split receipt included and excluded identities overlap")
        if not _explicit_bool(self.disjointness_verified, "disjointness_verified"):
            raise ValueError("split receipt must verify disjointness")
        object.__setattr__(self, "included_clean_parent_sha256s", tuple(sorted(included)))
        object.__setattr__(self, "excluded_population_sha256s", tuple(sorted(excluded)))

    def audit_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCParentProvenance:
    clean_parent_sha256: str
    clean_recipe_artifact_sha256: str
    universal_query_sha256: str
    query_ranges_artifact_sha256: str
    observation_artifact_sha256: str
    evaluation_query_context_sha256: str
    protocol_sha256: str
    model_artifact_sha256: str
    source_bundle_sha256: str
    calibration_identity_sha256: str
    calibrated_threshold_sha256: str
    exact_judge_sha256: str
    generating_branch_id: str
    topology_id: int
    pattern_id: int
    sobol_block_sha256: str
    sobol_design_sha256: str
    sobol_index: int
    range_stress_stratum: str
    observation_stress_stratum: str
    observation_effects: tuple[str, ...]
    geometry_axis_count: int
    geometry_axis_regimes: tuple[str, ...]
    geometry_axis_placements: tuple[str, ...]
    geometry_axis_coordinate_sha256s: tuple[str, ...]
    amplitude_axis_count: int
    amplitude_axis_regimes: tuple[str, ...]
    amplitude_axis_coordinate_sha256s: tuple[str, ...]
    amplitude_range_assignment_schema: str
    amplitude_range_assignment_version: str
    range_coordinate_contract_sha256: str
    range_coordinate_contract_schema: str
    range_coordinate_contract_version: str
    range_coordinate_contract_dimension: int
    axis_independent_range_coordinates: bool
    range_generated_before_truth: bool
    observation_generated_before_curve: bool

    def __post_init__(self) -> None:
        digest_fields = (
            "clean_parent_sha256",
            "clean_recipe_artifact_sha256",
            "universal_query_sha256",
            "query_ranges_artifact_sha256",
            "observation_artifact_sha256",
            "evaluation_query_context_sha256",
            "protocol_sha256",
            "model_artifact_sha256",
            "source_bundle_sha256",
            "calibration_identity_sha256",
            "calibrated_threshold_sha256",
            "exact_judge_sha256",
            "sobol_block_sha256",
            "sobol_design_sha256",
            "range_coordinate_contract_sha256",
        )
        for name in digest_fields:
            object.__setattr__(self, name, digest(getattr(self, name), name))
        for name in (
            "generating_branch_id",
            "range_stress_stratum",
            "observation_stress_stratum",
            "amplitude_range_assignment_schema",
            "amplitude_range_assignment_version",
            "range_coordinate_contract_schema",
            "range_coordinate_contract_version",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "topology_id",
            "pattern_id",
            "sobol_index",
            "geometry_axis_count",
            "amplitude_axis_count",
            "range_coordinate_contract_dimension",
        ):
            object.__setattr__(self, name, _nonnegative_int(getattr(self, name), name))
        for name in (
            "axis_independent_range_coordinates",
            "range_generated_before_truth",
            "observation_generated_before_curve",
        ):
            _explicit_bool(getattr(self, name), name)
        object.__setattr__(
            self,
            "geometry_axis_coordinate_sha256s",
            tuple(
                digest(value, f"geometry_axis_coordinate_sha256s[{index}]")
                for index, value in enumerate(self.geometry_axis_coordinate_sha256s)
            ),
        )
        object.__setattr__(
            self,
            "amplitude_axis_coordinate_sha256s",
            tuple(
                digest(value, f"amplitude_axis_coordinate_sha256s[{index}]")
                for index, value in enumerate(self.amplitude_axis_coordinate_sha256s)
            ),
        )
        object.__setattr__(
            self,
            "observation_effects",
            tuple(_text(value, "observation_effect") for value in self.observation_effects),
        )
        object.__setattr__(
            self,
            "geometry_axis_regimes",
            tuple(_text(value, "geometry_axis_regime") for value in self.geometry_axis_regimes),
        )
        object.__setattr__(
            self,
            "geometry_axis_placements",
            tuple(
                _text(value, "geometry_axis_placement")
                for value in self.geometry_axis_placements
            ),
        )
        object.__setattr__(
            self,
            "amplitude_axis_regimes",
            tuple(_text(value, "amplitude_axis_regime") for value in self.amplitude_axis_regimes),
        )

    def record_payload(self) -> dict[str, object]:
        payload = asdict(self)
        for name in (
            "clean_recipe_artifact_sha256",
            "query_ranges_artifact_sha256",
            "observation_artifact_sha256",
            "evaluation_query_context_sha256",
            "calibration_identity_sha256",
            "calibrated_threshold_sha256",
            "exact_judge_sha256",
        ):
            payload.pop(name)
        return payload


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCCandidateJudgement:
    candidate_id: str
    exact_call_index: int
    compatibility_status: str
    bounds_compliant: bool
    physics_compliant: bool
    amplitude_compliant: bool
    dedup_cluster_id: str | None
    exact_judge_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        call_index = _nonnegative_int(self.exact_call_index, "exact_call_index")
        if call_index < 1:
            raise ValueError("exact_call_index must be positive")
        object.__setattr__(self, "exact_call_index", call_index)
        status = _text(self.compatibility_status, "compatibility_status")
        if status not in V5_EXACT_COMPATIBILITY_STATUSES:
            raise ValueError("candidate compatibility status is unsupported")
        object.__setattr__(self, "compatibility_status", status)
        for name in ("bounds_compliant", "physics_compliant", "amplitude_compliant"):
            _explicit_bool(getattr(self, name), name)
        cluster = self.dedup_cluster_id
        if status == V5_EXACT_COMPATIBLE and cluster is None:
            raise ValueError("exact-compatible candidate requires a dedup cluster")
        if cluster is not None:
            object.__setattr__(self, "dedup_cluster_id", _text(cluster, "dedup_cluster_id"))
        object.__setattr__(
            self, "exact_judge_sha256", digest(self.exact_judge_sha256, "exact_judge_sha256")
        )


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCBranchSearchReplay:
    branch_id: str
    search_sidecar_sha256: str
    frozen_search_yield_rank: int
    completed: bool
    candidates: tuple[V5K1PhaseCCandidateJudgement, ...]
    sidecar_schema: str = V5_SEARCH_SIDECAR_SCHEMA
    sidecar_version: str = V5_SEARCH_SIDECAR_VERSION

    def __post_init__(self) -> None:
        if (self.sidecar_schema, self.sidecar_version) != (
            V5_SEARCH_SIDECAR_SCHEMA,
            V5_SEARCH_SIDECAR_VERSION,
        ):
            raise ValueError("branch search replay requires the current search-sidecar contract")
        object.__setattr__(self, "branch_id", _text(self.branch_id, "branch_id"))
        object.__setattr__(
            self,
            "search_sidecar_sha256",
            digest(self.search_sidecar_sha256, "search_sidecar_sha256"),
        )
        rank = _nonnegative_int(self.frozen_search_yield_rank, "frozen_search_yield_rank")
        if rank < 1:
            raise ValueError("frozen_search_yield_rank must be positive")
        object.__setattr__(self, "frozen_search_yield_rank", rank)
        if not _explicit_bool(self.completed, "completed"):
            raise ValueError("branch search sidecar is not complete")
        values = tuple(self.candidates)
        if not all(isinstance(value, V5K1PhaseCCandidateJudgement) for value in values):
            raise TypeError("branch candidates must contain typed judgements")
        ids = [value.candidate_id for value in values]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate IDs must be unique within a branch sidecar")
        object.__setattr__(self, "candidates", values)


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCReferenceCluster:
    representative_id: str
    member_candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        representative = _text(self.representative_id, "representative_id")
        members = tuple(_text(value, "member_candidate_id") for value in self.member_candidate_ids)
        if not members or len(members) != len(set(members)):
            raise ValueError("reference cluster members must be non-empty and unique")
        if representative not in members:
            raise ValueError("reference representative must be an explicit cluster member")
        object.__setattr__(self, "representative_id", representative)
        object.__setattr__(self, "member_candidate_ids", tuple(sorted(members)))


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCReferenceBankReplay:
    bank_artifact_sha256: str
    search_trace_artifact_sha256: str
    calibration_identity_sha256: str
    calibrated_threshold_sha256: str
    exact_judge_sha256: str
    source_bundle_sha256: str
    query_id: str
    pairing_unit_id: str
    query_context_sha256: str
    candidate_ids: tuple[str, ...]
    exact_compatible_candidate_ids: tuple[str, ...]
    candidate_judgements: tuple[V5K1PhaseCCandidateJudgement, ...]
    representative_clusters: tuple[V5K1PhaseCReferenceCluster, ...]
    representative_payload_sha256s: tuple[tuple[str, str], ...]
    configured_exact_call_budget: int
    consumed_exact_calls: int
    exact_forward_calls: tuple[V5ExactForwardCall, ...]
    enumeration_complete: bool
    network_free: bool
    schema: str = V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA
    version: str = V5_CONTEXTUAL_REFERENCE_BANK_VERSION
    distance_schema: str = V5_QUERY_PARAMETER_DISTANCE_SCHEMA
    distance_version: str = V5_QUERY_PARAMETER_DISTANCE_VERSION
    distance_sha256: str = V5_QUERY_PARAMETER_DISTANCE_SHA256

    def __post_init__(self) -> None:
        if (self.schema, self.version) != (
            V5_CONTEXTUAL_REFERENCE_BANK_SCHEMA,
            V5_CONTEXTUAL_REFERENCE_BANK_VERSION,
        ):
            raise ValueError("legacy or unsupported contextual reference-bank contract")
        if (self.distance_schema, self.distance_version, self.distance_sha256) != (
            V5_QUERY_PARAMETER_DISTANCE_SCHEMA,
            V5_QUERY_PARAMETER_DISTANCE_VERSION,
            V5_QUERY_PARAMETER_DISTANCE_SHA256,
        ):
            raise ValueError("legacy global or unsupported parameter distance is forbidden")
        object.__setattr__(
            self, "bank_artifact_sha256", digest(self.bank_artifact_sha256, "bank_artifact_sha256")
        )
        object.__setattr__(
            self,
            "search_trace_artifact_sha256",
            digest(self.search_trace_artifact_sha256, "search_trace_artifact_sha256"),
        )
        for name in (
            "calibration_identity_sha256",
            "calibrated_threshold_sha256",
            "exact_judge_sha256",
            "source_bundle_sha256",
            "query_id",
            "pairing_unit_id",
            "query_context_sha256",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        candidates = tuple(_text(value, "candidate_id") for value in self.candidate_ids)
        compatible = tuple(
            _text(value, "exact_compatible_candidate_id")
            for value in self.exact_compatible_candidate_ids
        )
        if not candidates or len(candidates) != len(set(candidates)):
            raise ValueError("reference-bank candidate IDs must be non-empty and unique")
        if set(compatible) != set(candidates) or len(compatible) != len(candidates):
            raise ValueError("every reference-bank candidate must be exact-compatible")
        judgements = tuple(self.candidate_judgements)
        if not all(isinstance(value, V5K1PhaseCCandidateJudgement) for value in judgements):
            raise TypeError("reference-bank judgements must be typed")
        if {value.candidate_id for value in judgements} != set(candidates) or len(
            judgements
        ) != len(candidates):
            raise ValueError("reference-bank judgements must bind every candidate exactly once")
        if any(
            value.compatibility_status != V5_EXACT_COMPATIBLE
            or not value.bounds_compliant
            or not value.physics_compliant
            or not value.amplitude_compliant
            or value.exact_judge_sha256 != self.exact_judge_sha256
            for value in judgements
        ):
            raise ValueError("reference-bank candidates require raw compatible gate judgements")
        clusters = tuple(self.representative_clusters)
        if not clusters or not all(isinstance(value, V5K1PhaseCReferenceCluster) for value in clusters):
            raise ValueError("reference bank requires typed representative clusters")
        members = [member for cluster in clusters for member in cluster.member_candidate_ids]
        if len(members) != len(set(members)) or set(members) != set(candidates):
            raise ValueError("reference clusters must expose the exact candidate partition")
        representatives = [value.representative_id for value in clusters]
        if len(representatives) != len(set(representatives)):
            raise ValueError("reference representative IDs must be unique")
        cluster_by_member = {
            member: cluster.representative_id
            for cluster in clusters
            for member in cluster.member_candidate_ids
        }
        if any(
            value.dedup_cluster_id != cluster_by_member[value.candidate_id]
            for value in judgements
        ):
            raise ValueError("raw reference judgements disagree with representative clusters")
        payloads = []
        for index, value in enumerate(self.representative_payload_sha256s):
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise TypeError("representative payload digest entries must be ID/SHA pairs")
            payloads.append(
                (
                    _text(value[0], f"representative_payload_sha256s[{index}].id"),
                    digest(value[1], f"representative_payload_sha256s[{index}].sha256"),
                )
            )
        payloads = tuple(payloads)
        if len(payloads) != len(set(payloads)) or {value[0] for value in payloads} != set(
            representatives
        ):
            raise ValueError("reference payload digests must bind every representative exactly once")
        budget = _nonnegative_int(self.configured_exact_call_budget, "configured_exact_call_budget")
        consumed = _nonnegative_int(self.consumed_exact_calls, "consumed_exact_calls")
        if budget < 1 or consumed > budget:
            raise ValueError("reference-bank exact-call accounting is invalid")
        if any(value.exact_call_index > consumed for value in judgements):
            raise ValueError("reference-bank judgement escaped the consumed exact-call prefix")
        calls = tuple(self.exact_forward_calls)
        if not all(isinstance(value, V5ExactForwardCall) for value in calls) or tuple(
            value.exact_call_index for value in calls
        ) != tuple(range(1, consumed + 1)):
            raise ValueError("reference-bank exact calls must be complete and contiguous")
        if any(
            left.elapsed_seconds > right.elapsed_seconds for left, right in zip(calls, calls[1:])
        ):
            raise ValueError("reference-bank exact-call elapsed time must be non-decreasing")
        complete = _explicit_bool(self.enumeration_complete, "enumeration_complete")
        if not complete and consumed != budget:
            raise ValueError("reference bank is not saturated")
        if not _explicit_bool(self.network_free, "network_free"):
            raise ValueError("reference-bank replay must be network-free")
        object.__setattr__(self, "candidate_ids", tuple(sorted(candidates)))
        object.__setattr__(self, "exact_compatible_candidate_ids", tuple(sorted(compatible)))
        object.__setattr__(
            self, "candidate_judgements", tuple(sorted(judgements, key=lambda value: value.candidate_id))
        )
        object.__setattr__(
            self,
            "representative_clusters",
            tuple(sorted(clusters, key=lambda value: value.representative_id)),
        )
        object.__setattr__(self, "configured_exact_call_budget", budget)
        object.__setattr__(self, "consumed_exact_calls", consumed)
        object.__setattr__(self, "exact_forward_calls", calls)
        object.__setattr__(self, "representative_payload_sha256s", tuple(sorted(payloads)))

    @property
    def saturated(self) -> bool:
        return self.enumeration_complete or (
            self.consumed_exact_calls == self.configured_exact_call_budget
        )

    @property
    def exact_call_ledger_sha256(self) -> str:
        return sha256(
            canonical_json([asdict(value) for value in self.exact_forward_calls]).encode("utf-8")
        ).hexdigest()

    def audit_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("exact_forward_calls")
        payload["exact_call_ledger_sha256"] = self.exact_call_ledger_sha256
        return payload


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCMethodReplay:
    status: str
    trace: V5MethodExactCallTrace
    source_bundle_sha256_used: str
    model_artifact_sha256_used: str | None
    proposal_execution_policy_sha256_used: str | None

    def __post_init__(self) -> None:
        status = _text(self.status, "method status")
        if status not in V5_K1_PHASE_C_METHOD_STATUSES:
            raise ValueError("method status is unsupported")
        if not isinstance(self.trace, V5MethodExactCallTrace):
            raise TypeError("method replay requires a typed exact-call trace")
        if status == V5_K1_PHASE_C_METHOD_CRASHED_ZERO and self.trace.candidate_emissions:
            raise ValueError("crashed method must be represented by a zero-emission trace")
        object.__setattr__(
            self,
            "source_bundle_sha256_used",
            digest(self.source_bundle_sha256_used, "source_bundle_sha256_used"),
        )
        if self.model_artifact_sha256_used is not None:
            object.__setattr__(
                self,
                "model_artifact_sha256_used",
                digest(self.model_artifact_sha256_used, "model_artifact_sha256_used"),
            )
        if self.proposal_execution_policy_sha256_used is not None:
            object.__setattr__(
                self,
                "proposal_execution_policy_sha256_used",
                digest(
                    self.proposal_execution_policy_sha256_used,
                    "proposal_execution_policy_sha256_used",
                ),
            )
        object.__setattr__(self, "status", status)

    def audit_payload(self) -> dict[str, object]:
        trace = self.trace
        return {
            "status": self.status,
            "source_bundle_sha256_used": self.source_bundle_sha256_used,
            "model_artifact_sha256_used": self.model_artifact_sha256_used,
            "proposal_execution_policy_sha256_used": (
                self.proposal_execution_policy_sha256_used
            ),
            "trace_identity": {
                "query_id": trace.query_id,
                "pairing_unit_id": trace.pairing_unit_id,
                "method_id": trace.method_id,
                "method_protocol_id": trace.method_protocol_id,
                "method_protocol_sha256": trace.method_protocol_sha256,
                "trace_id": trace.trace_id,
                "trace_artifact_sha256": trace.trace_artifact_sha256,
                "reference_set_id": trace.reference_set_id,
                "reference_set_sha256": trace.reference_set_sha256,
                "comparison_protocol_id": trace.comparison_protocol_id,
                "comparison_protocol_sha256": trace.comparison_protocol_sha256,
                "exact_forward_call_budget": trace.exact_forward_call_budget,
                "exact_call_count": len(trace.exact_forward_calls),
                "exact_call_ledger_sha256": trace.ledger_sha256,
            },
            "raw_typed_candidate_emissions": [
                {
                    "available_after_call": value.available_after_call,
                    "output_rank": value.output_rank,
                    "candidate_id": value.candidate_id,
                    "compatibility_status": value.compatibility_status,
                    "elapsed_seconds": value.elapsed_seconds,
                    "payload": value.payload.audit_payload(),
                    "payload_sha256": value.payload.sha256,
                }
                for value in trace.candidate_emissions
            ],
        }


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCParentReplayEvidence:
    provenance: V5K1PhaseCParentProvenance
    branch_searches: tuple[V5K1PhaseCBranchSearchReplay, ...]
    product_candidate_judgements: tuple[V5K1PhaseCCandidateJudgement, ...]
    reference_bank: V5K1PhaseCReferenceBankReplay
    reference_set: V5FrozenReferenceSet
    methods: tuple[V5K1PhaseCMethodReplay, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, V5K1PhaseCParentProvenance):
            raise TypeError("provenance must be typed")
        branches = tuple(self.branch_searches)
        candidates = tuple(self.product_candidate_judgements)
        methods = tuple(self.methods)
        if not branches or not all(isinstance(value, V5K1PhaseCBranchSearchReplay) for value in branches):
            raise ValueError("branch_searches must contain typed branch sidecars")
        if not all(isinstance(value, V5K1PhaseCCandidateJudgement) for value in candidates):
            raise TypeError("product_candidate_judgements must contain typed judgements")
        if not isinstance(self.reference_bank, V5K1PhaseCReferenceBankReplay):
            raise TypeError("reference_bank must be typed")
        if not isinstance(self.reference_set, V5FrozenReferenceSet):
            raise TypeError("reference_set must be typed")
        if not methods or not all(isinstance(value, V5K1PhaseCMethodReplay) for value in methods):
            raise ValueError("methods must contain typed method replays")
        object.__setattr__(self, "branch_searches", branches)
        object.__setattr__(self, "product_candidate_judgements", candidates)
        object.__setattr__(self, "methods", methods)


@dataclass(frozen=True, kw_only=True)
class V5K1PhaseCReplayBundle:
    plan_sha256: str
    contract_sha256: str
    artifact_binding: V5K1PhaseCArtifactBinding
    split_receipt: V5K1PhaseCSplitReplayReceipt
    evaluator_config: V5PaperBudgetEvaluationConfig
    parents: tuple[V5K1PhaseCParentReplayEvidence, ...]
    schema: str = V5_K1_PHASE_C_REPLAY_INPUT_SCHEMA
    version: str = V5_K1_PHASE_C_REPLAY_INPUT_VERSION

    def __post_init__(self) -> None:
        if (self.schema, self.version) != (
            V5_K1_PHASE_C_REPLAY_INPUT_SCHEMA,
            V5_K1_PHASE_C_REPLAY_INPUT_VERSION,
        ):
            raise ValueError("unsupported K1 Phase-C replay-input schema/version")
        for name in (
            "plan_sha256",
            "contract_sha256",
        ):
            object.__setattr__(self, name, digest(getattr(self, name), name))
        if not isinstance(self.artifact_binding, V5K1PhaseCArtifactBinding):
            raise TypeError("artifact_binding must be typed")
        if not isinstance(self.split_receipt, V5K1PhaseCSplitReplayReceipt):
            raise TypeError("split_receipt must be typed")
        if not isinstance(self.evaluator_config, V5PaperBudgetEvaluationConfig):
            raise TypeError("evaluator_config must be typed")
        values = tuple(self.parents)
        if not values or not all(isinstance(value, V5K1PhaseCParentReplayEvidence) for value in values):
            raise ValueError("parents must contain typed replay evidence")
        object.__setattr__(self, "parents", values)

    def audit_payload(self) -> dict[str, object]:
        parent_payloads = []
        for parent in self.parents:
            parent_payloads.append(
                {
                    "provenance": asdict(parent.provenance),
                    "branch_searches": [asdict(value) for value in parent.branch_searches],
                    "product_candidate_judgements": [
                        asdict(value) for value in parent.product_candidate_judgements
                    ],
                    "reference_bank": parent.reference_bank.audit_payload(),
                    "reference_set": {
                        "query_id": parent.reference_set.query_id,
                        "pairing_unit_id": parent.reference_set.pairing_unit_id,
                        "reference_set_id": parent.reference_set.reference_set_id,
                        "reference_set_sha256": parent.reference_set.reference_set_sha256,
                        "comparison_protocol_id": parent.reference_set.comparison_protocol_id,
                        "comparison_protocol_sha256": (
                            parent.reference_set.comparison_protocol_sha256
                        ),
                        "raw_typed_representatives": [
                            {
                                "representative_id": value.representative_id,
                                "payload": value.payload.audit_payload(),
                                "payload_sha256": value.payload.sha256,
                            }
                            for value in parent.reference_set.representatives
                        ],
                        "representative_payload_set_sha256": (
                            parent.reference_set.representative_payload_set_sha256
                        ),
                    },
                    "methods": [value.audit_payload() for value in parent.methods],
                }
            )
        return {
            "schema": self.schema,
            "version": self.version,
            "plan_sha256": self.plan_sha256,
            "contract_sha256": self.contract_sha256,
            "artifact_binding": self.artifact_binding.audit_payload(),
            "split_receipt": self.split_receipt.audit_payload(),
            "evaluator_config": self.evaluator_config.audit_payload(),
            "parents": parent_payloads,
        }

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


class V5K1PhaseCReplayPort(Protocol):
    """Adapter boundary for loading immutable evidence and query-local distance."""

    adapter_id: str
    adapter_version: str

    def load_bundle(
        self, *, plan: V5K1PhaseCPlan, contract: dict[str, object]
    ) -> V5K1PhaseCReplayBundle: ...

    def revalidate_bundle(
        self,
        *,
        bundle: V5K1PhaseCReplayBundle,
        plan: V5K1PhaseCPlan,
        contract: dict[str, object],
    ) -> None: ...

    @property
    def equivalence_distance_matcher(self) -> EquivalenceDistanceMatcher: ...


__all__ = [
    "V5_K1_PHASE_C_METHOD_COMPLETED",
    "V5_K1_PHASE_C_METHOD_CRASHED_ZERO",
    "V5_K1_PHASE_C_REPLAY_INPUT_SCHEMA",
    "V5_K1_PHASE_C_REPLAY_INPUT_VERSION",
    "V5_K1_PHASE_C_SPLIT_REPLAY_RECEIPT_SCHEMA",
    "V5_K1_PHASE_C_SPLIT_REPLAY_RECEIPT_VERSION",
    "V5K1PhaseCArtifactBinding",
    "V5K1PhaseCBranchSearchReplay",
    "V5K1PhaseCCandidateJudgement",
    "V5K1PhaseCMethodReplay",
    "V5K1PhaseCParentProvenance",
    "V5K1PhaseCParentReplayEvidence",
    "V5K1PhaseCReferenceBankReplay",
    "V5K1PhaseCReferenceCluster",
    "V5K1PhaseCReplayBundle",
    "V5K1PhaseCReplayPort",
    "V5K1PhaseCSplitReplayReceipt",
]
