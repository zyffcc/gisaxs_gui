"""Frozen exact-search protocol and branch-result contracts for V5.1.

These values describe an operational search experiment.  A completed
negative means only that this exact protocol exhausted or completed its
declared budget without finding a compatible representative.  It is never a
certificate that a mathematical solution does not exist.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral
from typing import Protocol

import numpy as np

from .branch_codec import UNIT_CUBE_DIMENSIONS
from .calibrated_search_threshold_v5 import (
    V5CalibrationArtifactIdentity,
    V5CalibratedObservationThreshold,
    compatibility_stratum_from_v5_observation,
)
from .candidate_supervision_v5 import NEGATIVE_TERMINATION_REASONS, SEARCH_OUTCOME_STATES
from .evaluation import (
    RAW_LOG_RMSE_METRIC,
    STANDARDIZED_LOG_RMSE_METRIC,
    ObservedCurve,
)
from .grouped_artifact_v5 import array_sha256, canonical_json
from .observation_v5 import V5ObservationDataView
from .universal_query_contract_v5 import V5GlobalBranchKey
from .universal_query_v5 import V5UniversalCandidateContext

V5_FROZEN_SEARCH_PROTOCOL_SCHEMA = "gisaxs.posterior_v8.frozen_exact_search_protocol/v3"
V5_FROZEN_SEARCH_PROTOCOL_VERSION = (
    "posterior_v8_tiered_calibrated_or_pilot_equal_branch_exact_search_v3"
)
V5_SEARCH_RESULT_SCHEMA = "gisaxs.posterior_v8.frozen_branch_search_result/v2"
V5_SEARCH_RESULT_VERSION = "posterior_v8_curve_bound_completed_positive_negative_or_unverified_v2"
V5_EXACT_SEARCH_OBSERVATION_SCHEMA = "gisaxs.posterior_v8.exact_search_observation/v1"
V5_EXACT_SEARCH_OBSERVATION_VERSION = (
    "posterior_v8_preprocessed_valid_curve_acceptance_sigma_separation_v1"
)
V5_SEARCH_METRIC_SELECTION_POLICY = (
    "acceptance_sigma_log_available_uses_standardized_metric_else_raw_metric_v1"
)
V5_SEARCH_CATALOG_POLICY = (
    "every_codec_feasible_branch_for_every_selected_topology_in_the_universal_query"
)
V5_SEARCH_NEGATIVE_CLAIM = (
    "no_compatible_representative_found_by_this_frozen_protocol_within_its_budget_"
    "not_a_no_solution_certificate"
)
V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT = "engineering_pilot"
V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED = "paper_full_calibrated"
V5_SEARCH_PROTOCOL_TIERS = (
    V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT,
    V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED,
)
V5_ENGINEERING_PILOT_CLAIM_LIMITS = (
    "throughput_and_contract_plumbing_only_not_formal_compatibility_or_full_training_labels"
)
V5_PAPER_FULL_CALIBRATED_CLAIM_LIMITS = (
    "reserved_calibration_split_acquisition_stratum_compatibility_labels"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: object, name: str) -> str:
    result = _text(value, name).lower()
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return result


def _count(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


@dataclass(frozen=True, kw_only=True)
class V5FrozenExactSearchProtocol:
    """Immutable identity of the model-independent per-branch search."""

    protocol_id: str
    evaluator_version: str
    authoritative_forward_id: str
    metric_name: str
    threshold_name: str | None
    threshold_value: float | None
    threshold_source_id: str | None
    missing_acceptance_sigma_metric_name: str
    missing_acceptance_sigma_threshold_name: str | None
    missing_acceptance_sigma_threshold_value: float | None
    metric_selection_policy: str = V5_SEARCH_METRIC_SELECTION_POLICY
    exact_forward_call_budget: int
    seed_schedule_id: str
    seed_schedule_sha256: str
    optimizer_schedule_id: str
    optimizer_schedule_sha256: str
    termination_policy_id: str
    representative_distance_id: str
    representative_distance_sha256: str
    # Historical wire name. Under complete linkage this is a maximum
    # within-cluster diameter, not a claim that representatives are pairwise
    # delta-separated.
    delta_separation_threshold: float
    clustering_linkage: str = "complete"
    uses_neural_proposal_scores: bool = False
    same_budget_for_every_branch: bool = True
    catalog_policy: str = V5_SEARCH_CATALOG_POLICY
    negative_claim: str = V5_SEARCH_NEGATIVE_CLAIM
    protocol_tier: str = V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT
    claim_limits: str = V5_ENGINEERING_PILOT_CLAIM_LIMITS
    calibration_identity: V5CalibrationArtifactIdentity | None = None
    schema_version: str = V5_FROZEN_SEARCH_PROTOCOL_SCHEMA
    version: str = V5_FROZEN_SEARCH_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        for name in (
            "protocol_id",
            "evaluator_version",
            "authoritative_forward_id",
            "metric_name",
            "seed_schedule_id",
            "optimizer_schedule_id",
            "termination_policy_id",
            "representative_distance_id",
            "missing_acceptance_sigma_metric_name",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        for name in (
            "seed_schedule_sha256",
            "optimizer_schedule_sha256",
            "representative_distance_sha256",
        ):
            object.__setattr__(self, name, _digest(getattr(self, name), name))
        budget = _count(self.exact_forward_call_budget, "exact_forward_call_budget")
        if budget < 1:
            raise ValueError("exact_forward_call_budget must be positive")
        object.__setattr__(self, "exact_forward_call_budget", budget)
        delta = float(self.delta_separation_threshold)
        if not np.isfinite(delta) or delta <= 0.0:
            raise ValueError("delta_separation_threshold must be finite and positive")
        object.__setattr__(self, "delta_separation_threshold", delta)
        if self.clustering_linkage != "complete":
            raise ValueError("representative clustering must use complete linkage")
        for name in ("uses_neural_proposal_scores", "same_budget_for_every_branch"):
            object.__setattr__(self, name, _strict_bool(getattr(self, name), name))
        if self.uses_neural_proposal_scores:
            raise ValueError("frozen supervision search cannot use neural proposal scores")
        if not self.same_budget_for_every_branch:
            raise ValueError("frozen supervision search requires one equal per-branch budget")
        if self.catalog_policy != V5_SEARCH_CATALOG_POLICY:
            raise ValueError("frozen search catalog policy is incompatible")
        if self.negative_claim != V5_SEARCH_NEGATIVE_CLAIM:
            raise ValueError("frozen search negative-claim boundary is incompatible")
        if self.metric_selection_policy != V5_SEARCH_METRIC_SELECTION_POLICY:
            raise ValueError("frozen search metric-selection policy is incompatible")
        if self.metric_name != STANDARDIZED_LOG_RMSE_METRIC:
            raise ValueError(
                "acceptance-sigma search metric must be the standardized natural-log RMSE"
            )
        if self.missing_acceptance_sigma_metric_name != RAW_LOG_RMSE_METRIC:
            raise ValueError("missing acceptance sigma must select raw natural-log RMSE")
        tier = _text(self.protocol_tier, "protocol_tier")
        if tier not in V5_SEARCH_PROTOCOL_TIERS:
            raise ValueError(f"protocol_tier must be one of {V5_SEARCH_PROTOCOL_TIERS}")
        calibration_identity = self.calibration_identity
        if isinstance(calibration_identity, dict):
            calibration_identity = V5CalibrationArtifactIdentity.from_payload(calibration_identity)
        scalar_fields = (
            self.threshold_name,
            self.threshold_value,
            self.threshold_source_id,
            self.missing_acceptance_sigma_threshold_name,
            self.missing_acceptance_sigma_threshold_value,
        )
        if tier == V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT:
            if calibration_identity is not None:
                raise ValueError("engineering pilot cannot bind a formal calibration artifact")
            if any(value is None for value in scalar_fields):
                raise ValueError("engineering pilot requires explicit scalar thresholds")
            for name in (
                "threshold_name",
                "threshold_source_id",
                "missing_acceptance_sigma_threshold_name",
            ):
                object.__setattr__(self, name, _text(getattr(self, name), name))
            for name in (
                "threshold_value",
                "missing_acceptance_sigma_threshold_value",
            ):
                value = float(getattr(self, name))
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be finite and non-negative")
                object.__setattr__(self, name, value)
            if self.claim_limits != V5_ENGINEERING_PILOT_CLAIM_LIMITS:
                raise ValueError("engineering-pilot claim limits are incompatible")
        else:
            if not isinstance(calibration_identity, V5CalibrationArtifactIdentity):
                raise ValueError("paper/full search requires a checked calibration identity")
            if any(value is not None for value in scalar_fields):
                raise ValueError("paper/full search forbids command-line scalar thresholds")
            if self.claim_limits != V5_PAPER_FULL_CALIBRATED_CLAIM_LIMITS:
                raise ValueError("paper/full calibrated claim limits are incompatible")
        object.__setattr__(self, "protocol_tier", tier)
        object.__setattr__(self, "calibration_identity", calibration_identity)
        if self.schema_version != V5_FROZEN_SEARCH_PROTOCOL_SCHEMA:
            raise ValueError("unsupported frozen-search protocol schema")
        if self.version != V5_FROZEN_SEARCH_PROTOCOL_VERSION:
            raise ValueError("unsupported frozen-search protocol version")

    def audit_payload(self) -> dict[str, object]:
        payload = dict(self.__dict__)
        if self.calibration_identity is not None:
            payload["calibration_identity"] = self.calibration_identity.audit_payload()
        return payload

    @property
    def sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


def observed_curve_sha256(curve: ObservedCurve) -> str:
    """Hash exact-search samples without relying on encoder normalization."""

    if not isinstance(curve, ObservedCurve):
        raise TypeError("curve must be an ObservedCurve")
    payload = {
        "curve_id": curve.curve_id,
        "source_kind": curve.source_kind,
        "point_count": int(curve.q.size),
        "q_sha256": array_sha256("exact_curve_q", curve.q),
        "intensity_sha256": array_sha256("exact_curve_intensity", curve.intensity),
        "sigma_log_sha256": (
            None
            if curve.sigma_log is None
            else array_sha256("exact_curve_sigma_log", curve.sigma_log)
        ),
    }
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def preprocessed_valid_arrays_sha256(view: V5ObservationDataView) -> str:
    """Bind the exact curve to the physical arrays behind the parent encoder row."""

    if not isinstance(view, V5ObservationDataView):
        raise TypeError("view must be a V5ObservationDataView")
    q, intensity, encoder_sigma = view.preprocessed.valid_arrays()
    payload = {
        "q_sha256": array_sha256("preprocessed_valid_q", q),
        "intensity_sha256": array_sha256("preprocessed_valid_intensity", intensity),
        "encoder_sigma_sha256": array_sha256("preprocessed_valid_encoder_sigma", encoder_sigma),
        "valid_count": int(q.size),
        "preprocessing_contract": dict(view.preprocessed.stats["contract"]),
    }
    return sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5ExactSearchObservation:
    """Explicit physical curve and acceptance evidence handed to every runner call.

    The V5 observation view is retained deliberately: it separates measured/simulated
    acceptance sigma from the encoder-only proxy and lets collection prove that the
    unnormalised curve produced the exact parent observation tensors.
    """

    observed_curve: ObservedCurve
    observation_view: V5ObservationDataView
    acceptance_sigma_source_id: str
    schema_version: str = V5_EXACT_SEARCH_OBSERVATION_SCHEMA
    version: str = V5_EXACT_SEARCH_OBSERVATION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.observed_curve, ObservedCurve):
            raise TypeError("observed_curve must be an ObservedCurve")
        if not isinstance(self.observation_view, V5ObservationDataView):
            raise TypeError("observation_view must be a V5ObservationDataView")
        source = _text(self.acceptance_sigma_source_id, "acceptance_sigma_source_id")
        q, intensity, _ = self.observation_view.preprocessed.valid_arrays()
        curve = self.observed_curve
        if not np.array_equal(curve.q, q) or not np.array_equal(curve.intensity, intensity):
            raise ValueError(
                "exact-search curve must equal PreprocessedCurve.valid_arrays q/intensity"
            )
        acceptance = self.observation_view.acceptance_sigma_log
        if acceptance is None:
            if curve.sigma_log is not None:
                raise ValueError(
                    "encoder-only uncertainty cannot become exact-search acceptance sigma"
                )
            if not source.endswith("|acceptance_sigma_absent_use_raw_metric"):
                raise ValueError("missing acceptance sigma must declare raw-metric fallback")
        else:
            if curve.sigma_log is None or not np.array_equal(curve.sigma_log, acceptance):
                raise ValueError("ObservedCurve.sigma_log must be the V5 acceptance evidence")
            if not source.endswith("|acceptance_sigma_log"):
                raise ValueError("acceptance sigma source must be explicit")
        if not source.startswith(f"{self.observation_view.acquisition_policy_id}|"):
            raise ValueError("acceptance sigma source escaped observation provenance")
        if self.schema_version != V5_EXACT_SEARCH_OBSERVATION_SCHEMA:
            raise ValueError("unsupported exact-search observation schema")
        if self.version != V5_EXACT_SEARCH_OBSERVATION_VERSION:
            raise ValueError("unsupported exact-search observation version")
        object.__setattr__(self, "acceptance_sigma_source_id", source)

    @classmethod
    def from_observation_view(
        cls,
        view: V5ObservationDataView,
        *,
        curve_id: str,
    ) -> "V5ExactSearchObservation":
        if not isinstance(view, V5ObservationDataView):
            raise TypeError("view must be a V5ObservationDataView")
        q, intensity, _ = view.preprocessed.valid_arrays()
        suffix = (
            "acceptance_sigma_log"
            if view.acceptance_sigma_log is not None
            else "acceptance_sigma_absent_use_raw_metric"
        )
        return cls(
            observed_curve=ObservedCurve(
                curve_id=curve_id,
                source_kind="synthetic",
                q=q,
                intensity=intensity,
                sigma_log=view.acceptance_sigma_log,
            ),
            observation_view=view,
            acceptance_sigma_source_id=f"{view.acquisition_policy_id}|{suffix}",
        )

    @property
    def curve_sha256(self) -> str:
        return observed_curve_sha256(self.observed_curve)

    @property
    def parent_observation_audit_json(self) -> str:
        return canonical_json(self.observation_view.audit_payload())

    @property
    def parent_observation_audit_sha256(self) -> str:
        return sha256(self.parent_observation_audit_json.encode("utf-8")).hexdigest()

    @property
    def preprocessed_valid_arrays_sha256(self) -> str:
        return preprocessed_valid_arrays_sha256(self.observation_view)

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "exact_curve_sha256": self.curve_sha256,
            "exact_curve_id": self.observed_curve.curve_id,
            "exact_curve_source_kind": self.observed_curve.source_kind,
            "exact_curve_point_count": int(self.observed_curve.q.size),
            "acceptance_sigma_log_available": self.observed_curve.sigma_log is not None,
            "acceptance_sigma_source_id": self.acceptance_sigma_source_id,
            "parent_observation_audit_sha256": self.parent_observation_audit_sha256,
            "preprocessed_valid_arrays_sha256": self.preprocessed_valid_arrays_sha256,
        }

    @property
    def audit_json(self) -> str:
        return canonical_json(self.audit_payload())

    @property
    def audit_sha256(self) -> str:
        return sha256(self.audit_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True, kw_only=True)
class V5CompatibleRepresentativeReference:
    """External exact-compatible representative referenced by immutable hash."""

    artifact_id: str
    artifact_sha256: str
    representative_set_id: str
    representative_set_sha256: str
    cluster_id: str
    metric_value: float
    bounds_passed: bool
    physics_passed: bool
    target_local: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self, "artifact_sha256", _digest(self.artifact_sha256, "artifact_sha256")
        )
        object.__setattr__(
            self,
            "representative_set_id",
            _text(self.representative_set_id, "representative_set_id"),
        )
        object.__setattr__(
            self,
            "representative_set_sha256",
            _digest(self.representative_set_sha256, "representative_set_sha256"),
        )
        object.__setattr__(self, "cluster_id", _text(self.cluster_id, "cluster_id"))
        metric = float(self.metric_value)
        if not np.isfinite(metric) or metric < 0.0:
            raise ValueError("metric_value must be finite and non-negative")
        object.__setattr__(self, "metric_value", metric)
        object.__setattr__(self, "bounds_passed", _strict_bool(self.bounds_passed, "bounds_passed"))
        object.__setattr__(
            self, "physics_passed", _strict_bool(self.physics_passed, "physics_passed")
        )
        if self.target_local is not None:
            target = np.asarray(self.target_local, dtype=np.float64)
            if (
                target.shape != (UNIT_CUBE_DIMENSIONS,)
                or not np.all(np.isfinite(target))
                or np.any(target < 0.0)
                or np.any(target > 1.0)
            ):
                raise ValueError("target_local must contain 26 finite values in [0,1]")
            object.__setattr__(self, "target_local", tuple(float(value) for value in target))

    def audit_payload(self) -> dict[str, object]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_sha256": self.artifact_sha256,
            "representative_set_id": self.representative_set_id,
            "representative_set_sha256": self.representative_set_sha256,
            "cluster_id": self.cluster_id,
            "metric_value": self.metric_value,
            "bounds_passed": self.bounds_passed,
            "physics_passed": self.physics_passed,
            "target_local": None if self.target_local is None else list(self.target_local),
        }


@dataclass(frozen=True, kw_only=True)
class V5FrozenBranchSearchResult:
    """Runner output for exactly one universal-query branch task."""

    universal_query_sha256: str
    exact_curve_sha256: str
    global_branch_key: V5GlobalBranchKey
    context_sha256: str
    outcome: str
    completed: bool
    exact_forward_calls_used: int
    termination_reason: str
    executor_artifact_id: str | None = None
    executor_artifact_sha256: str | None = None
    representatives: tuple[V5CompatibleRepresentativeReference, ...] = ()
    schema_version: str = V5_SEARCH_RESULT_SCHEMA
    version: str = V5_SEARCH_RESULT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "universal_query_sha256",
            _digest(self.universal_query_sha256, "universal_query_sha256"),
        )
        object.__setattr__(
            self,
            "exact_curve_sha256",
            _digest(self.exact_curve_sha256, "exact_curve_sha256"),
        )
        if not isinstance(self.global_branch_key, V5GlobalBranchKey):
            raise TypeError("global_branch_key must be V5GlobalBranchKey")
        object.__setattr__(self, "context_sha256", _digest(self.context_sha256, "context_sha256"))
        if self.outcome not in SEARCH_OUTCOME_STATES:
            raise ValueError(f"outcome must be one of {SEARCH_OUTCOME_STATES}")
        object.__setattr__(self, "completed", _strict_bool(self.completed, "completed"))
        object.__setattr__(
            self,
            "exact_forward_calls_used",
            _count(self.exact_forward_calls_used, "exact_forward_calls_used"),
        )
        object.__setattr__(
            self, "termination_reason", _text(self.termination_reason, "termination_reason")
        )
        references = tuple(self.representatives)
        if not all(isinstance(value, V5CompatibleRepresentativeReference) for value in references):
            raise TypeError("representatives must contain compatible representative references")
        keys = tuple(
            (value.metric_value, value.artifact_id, value.artifact_sha256) for value in references
        )
        if len(set(keys)) != len(keys):
            raise ValueError("representative references must be unique")
        if len({value.cluster_id for value in references}) != len(references):
            raise ValueError("representative references must use unique cluster IDs")
        if (
            len(
                {
                    (value.representative_set_id, value.representative_set_sha256)
                    for value in references
                }
            )
            > 1
        ):
            raise ValueError(
                "one branch result must reference one complete-linkage diameter-cluster set artifact"
            )
        local_targets = tuple(
            value.target_local for value in references if value.target_local is not None
        )
        if len(set(local_targets)) != len(local_targets):
            raise ValueError("representative local targets must be unique")
        object.__setattr__(
            self,
            "representatives",
            tuple(
                sorted(
                    references,
                    key=lambda value: (
                        value.metric_value,
                        value.artifact_id,
                        value.artifact_sha256,
                    ),
                )
            ),
        )
        paired = (self.executor_artifact_id is None, self.executor_artifact_sha256 is None)
        if paired[0] != paired[1]:
            raise ValueError("executor artifact ID and SHA-256 must be supplied together")
        if self.executor_artifact_id is not None:
            object.__setattr__(
                self,
                "executor_artifact_id",
                _text(self.executor_artifact_id, "executor_artifact_id"),
            )
            object.__setattr__(
                self,
                "executor_artifact_sha256",
                _digest(self.executor_artifact_sha256, "executor_artifact_sha256"),
            )
        if self.outcome == "unverified":
            if self.completed or references:
                raise ValueError("unverified search cannot be completed or carry representatives")
        else:
            if not self.completed or self.executor_artifact_id is None:
                raise ValueError("completed outcomes require a completed executor artifact")
        if self.outcome == "compatible_found" and not references:
            raise ValueError("compatible_found requires at least one representative")
        if self.outcome == "no_compatible_found_within_frozen_search_budget":
            if references or self.termination_reason not in NEGATIVE_TERMINATION_REASONS:
                raise ValueError("negative outcome requires a valid completed negative termination")
        if self.schema_version != V5_SEARCH_RESULT_SCHEMA:
            raise ValueError("unsupported frozen branch-result schema")
        if self.version != V5_SEARCH_RESULT_VERSION:
            raise ValueError("unsupported frozen branch-result version")


@dataclass(frozen=True)
class V5UniversalSearchSpec:
    """Bind one universal query context to one parent observation row."""

    parent_observation_index: int
    context: V5UniversalCandidateContext
    exact_observation: V5ExactSearchObservation
    query_catalog_artifact_id: str
    query_catalog_artifact_sha256: str
    calibrated_threshold: V5CalibratedObservationThreshold | None = None

    def __post_init__(self) -> None:
        if isinstance(self.parent_observation_index, (bool, np.bool_)) or not isinstance(
            self.parent_observation_index, Integral
        ):
            raise TypeError("parent_observation_index must be an integer")
        if int(self.parent_observation_index) < 0:
            raise ValueError("parent_observation_index must be non-negative")
        if not isinstance(self.context, V5UniversalCandidateContext):
            raise TypeError("context must be V5UniversalCandidateContext")
        if not isinstance(self.exact_observation, V5ExactSearchObservation):
            raise TypeError("exact_observation must be V5ExactSearchObservation")
        if self.calibrated_threshold is not None and not isinstance(
            self.calibrated_threshold, V5CalibratedObservationThreshold
        ):
            raise TypeError("calibrated_threshold must be a V5CalibratedObservationThreshold")
        object.__setattr__(
            self,
            "query_catalog_artifact_id",
            _text(self.query_catalog_artifact_id, "query_catalog_artifact_id"),
        )
        object.__setattr__(
            self,
            "query_catalog_artifact_sha256",
            _digest(
                self.query_catalog_artifact_sha256,
                "query_catalog_artifact_sha256",
            ),
        )
        object.__setattr__(self, "parent_observation_index", int(self.parent_observation_index))


@dataclass(frozen=True)
class V5FrozenSearchTask:
    """Framework-neutral task handed to the exact-search runner."""

    query_index: int
    clean_group_id: str
    recipe_id: str
    observation_id: str
    universal_context: V5UniversalCandidateContext
    exact_observation: V5ExactSearchObservation
    query_catalog_artifact_id: str
    query_catalog_artifact_sha256: str
    branch_index: int
    protocol: V5FrozenExactSearchProtocol
    calibrated_threshold: V5CalibratedObservationThreshold | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.protocol, V5FrozenExactSearchProtocol):
            raise TypeError("protocol must be a V5FrozenExactSearchProtocol")
        if not isinstance(self.exact_observation, V5ExactSearchObservation):
            raise TypeError("exact_observation must be a V5ExactSearchObservation")
        binding = self.calibrated_threshold
        formal = self.protocol.protocol_tier == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED
        if formal:
            if not isinstance(binding, V5CalibratedObservationThreshold):
                raise ValueError(
                    "paper/full search requires one calibrated threshold per observation"
                )
            if self.observed_curve.sigma_log is None:
                raise ValueError(
                    "paper/full calibrated search cannot label missing-sigma observations"
                )
            if binding.calibration_identity != self.protocol.calibration_identity:
                raise ValueError("observation threshold escaped the protocol calibration artifact")
            expected_stratum = compatibility_stratum_from_v5_observation(
                self.exact_observation.observation_view
            )
            if binding.stratum != expected_stratum:
                raise ValueError(
                    "observation threshold does not match pre-fit acquisition metadata"
                )
            expected_policy_sha = sha256(
                self.exact_observation.observation_view.acquisition_policy_id.encode("utf-8")
            ).hexdigest()
            if binding.acquisition_policy_id_sha256 != expected_policy_sha:
                raise ValueError("observation threshold does not bind the acquisition policy")
        elif binding is not None:
            raise ValueError("engineering-pilot search cannot carry a formal calibrated threshold")

    @property
    def branch(self):
        return self.universal_context.branches[self.branch_index]

    @property
    def codec(self):
        return self.universal_context.codec_for(self.branch.global_key.wire_key)

    @property
    def amplitude_constraint(self):
        return self.branch.amplitude_constraint

    @property
    def observed_curve(self) -> ObservedCurve:
        return self.exact_observation.observed_curve

    @property
    def exact_curve_sha256(self) -> str:
        return self.exact_observation.curve_sha256

    @property
    def selected_metric_name(self) -> str:
        if self.calibrated_threshold is not None:
            return self.calibrated_threshold.metric_name
        if self.observed_curve.sigma_log is None:
            return self.protocol.missing_acceptance_sigma_metric_name
        return self.protocol.metric_name

    @property
    def selected_threshold_name(self) -> str:
        if self.calibrated_threshold is not None:
            return self.calibrated_threshold.threshold_name
        if self.observed_curve.sigma_log is None:
            value = self.protocol.missing_acceptance_sigma_threshold_name
        else:
            value = self.protocol.threshold_name
        assert value is not None
        return value

    @property
    def selected_threshold_value(self) -> float:
        if self.calibrated_threshold is not None:
            return self.calibrated_threshold.threshold_value
        if self.observed_curve.sigma_log is None:
            value = self.protocol.missing_acceptance_sigma_threshold_value
        else:
            value = self.protocol.threshold_value
        assert value is not None
        return value

    @property
    def selected_threshold_source_id(self) -> str:
        if self.calibrated_threshold is not None:
            return self.calibrated_threshold.threshold_source_id
        value = self.protocol.threshold_source_id
        assert value is not None
        return value

    @property
    def full_training_label_permitted(self) -> bool:
        return self.protocol.protocol_tier == V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED

    def audit_payload(self) -> dict[str, object]:
        return {
            "query_index": self.query_index,
            "clean_group_id": self.clean_group_id,
            "recipe_id": self.recipe_id,
            "observation_id": self.observation_id,
            "exact_observation_audit_sha256": self.exact_observation.audit_sha256,
            "query_catalog_artifact_sha256": self.query_catalog_artifact_sha256,
            "universal_query_sha256": self.universal_context.audit_sha256,
            "global_branch_key": self.branch.global_key.wire_key,
            "context_sha256": self.branch.context_sha256,
            "protocol_sha256": self.protocol.sha256,
            "calibrated_threshold_sha256": (
                None if self.calibrated_threshold is None else self.calibrated_threshold.sha256
            ),
        }

    @property
    def audit_sha256(self) -> str:
        return sha256(canonical_json(self.audit_payload()).encode("utf-8")).hexdigest()


def unverified_v5_search_result(
    task: V5FrozenSearchTask,
    *,
    termination_reason: str = "search_not_run",
) -> V5FrozenBranchSearchResult:
    """Create an explicit non-label for plumbing; it is never a negative."""

    return V5FrozenBranchSearchResult(
        universal_query_sha256=task.universal_context.audit_sha256,
        exact_curve_sha256=task.exact_curve_sha256,
        global_branch_key=task.branch.global_key,
        context_sha256=task.branch.context_sha256,
        outcome="unverified",
        completed=False,
        exact_forward_calls_used=0,
        termination_reason=termination_reason,
    )


class V5FrozenBranchSearchRunner(Protocol):
    """Injectable executor; implementations must not read learned model scores."""

    def __call__(self, task: V5FrozenSearchTask) -> V5FrozenBranchSearchResult: ...


__all__ = [
    "V5CompatibleRepresentativeReference",
    "V5ExactSearchObservation",
    "V5FrozenBranchSearchResult",
    "V5FrozenBranchSearchRunner",
    "V5FrozenExactSearchProtocol",
    "V5FrozenSearchTask",
    "V5UniversalSearchSpec",
    "V5_FROZEN_SEARCH_PROTOCOL_SCHEMA",
    "V5_FROZEN_SEARCH_PROTOCOL_VERSION",
    "V5_EXACT_SEARCH_OBSERVATION_SCHEMA",
    "V5_EXACT_SEARCH_OBSERVATION_VERSION",
    "V5_ENGINEERING_PILOT_CLAIM_LIMITS",
    "V5_PAPER_FULL_CALIBRATED_CLAIM_LIMITS",
    "V5_SEARCH_CATALOG_POLICY",
    "V5_SEARCH_METRIC_SELECTION_POLICY",
    "V5_SEARCH_NEGATIVE_CLAIM",
    "V5_SEARCH_PROTOCOL_TIERS",
    "V5_SEARCH_PROTOCOL_TIER_ENGINEERING_PILOT",
    "V5_SEARCH_PROTOCOL_TIER_PAPER_FULL_CALIBRATED",
    "V5_SEARCH_RESULT_SCHEMA",
    "V5_SEARCH_RESULT_VERSION",
    "observed_curve_sha256",
    "preprocessed_valid_arrays_sha256",
    "unverified_v5_search_result",
]
