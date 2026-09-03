"""TensorFlow-free operational supervision for V5 candidate branches.

The network scores whether a frozen search protocol finds a compatible
representative in one query/branch. It does not classify mathematical branch
solvability, and a generating-parameter mismatch is never a negative label.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral
import re
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from .branch_codec import INACTIVE_UNIT_VALUE, UNIT_CUBE_DIMENSIONS


CANDIDATE_SUPERVISION_V5_SCHEMA = "gisaxs.posterior_v8.candidate_supervision/v5"
CANDIDATE_SUPERVISION_V5_VERSION = "posterior_v8_frozen_search_yield_three_state_supervision_v2"
SEARCH_OUTCOME_STATES = (
    "compatible_found",
    "no_compatible_found_within_frozen_search_budget",
    "unverified",
)
SEARCH_OUTCOME_CODE = MappingProxyType(
    {
        "unverified": 0,
        "no_compatible_found_within_frozen_search_budget": 1,
        "compatible_found": 2,
    }
)
SEARCH_OUTCOME_FROM_CODE = MappingProxyType(
    {value: key for key, value in SEARCH_OUTCOME_CODE.items()}
)
NEGATIVE_TERMINATION_REASONS = (
    "exact_forward_budget_exhausted_without_compatible",
    "frozen_protocol_completed_without_compatible",
)
POSITIVE_TERMINATION_REASON = (
    "frozen_full_budget_completed_with_compatible_representatives"
)
KNOWN_TRUTH_ORACLE_PROTOCOL_ID = (
    "posterior-v8-v5-known-truth-one-exact-forward-call/v1"
)
KNOWN_TRUTH_ORACLE_TERMINATION_REASON = "compatible_known_truth_exact_one_call"
EXACT_COMPARISON_SEMANTICS = (
    "compatible_iff_metric_lte_frozen_threshold_and_bounds_pass_and_physics_pass_v1"
)
RECIPE_WEIGHTING_SEMANTICS = (
    "candidate_weights_normalized_within_each_eligible_clean_recipe_then_"
    "clean_recipe_weighted_macro_mean_v1"
)

CANDIDATE_SUPERVISION_TENSOR_KEYS = (
    "search_outcome_code",
    "exact_artifact_id",
    "exact_artifact_sha256",
    "exact_metric_value",
    "exact_bounds_passed",
    "exact_physics_passed",
    "search_artifact_id",
    "search_artifact_sha256",
    "search_protocol_id",
    "search_protocol_sha256",
    "search_evaluator_version",
    "search_metric_name",
    "search_threshold_name",
    "search_threshold_value",
    "search_threshold_source_id",
    "search_exact_forward_call_budget",
    "search_exact_forward_calls_used",
    "search_termination_reason",
    "search_completed",
    "search_compatible_representative_count",
    "target_local",
    "has_local_target",
    "active_dimension_mask",
    "varying_dimension_mask",
    "clean_recipe_index",
    "clean_recipe_weight",
    "candidate_weight",
)

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _digest(value: object, name: str) -> str:
    result = _text(value, name).lower()
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must contain 64 hexadecimal characters")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _mask(value: object, name: str) -> tuple[bool, ...]:
    array = np.asarray(value)
    if array.shape != (UNIT_CUBE_DIMENSIONS,):
        raise ValueError(f"{name} must contain {UNIT_CUBE_DIMENSIONS} values")
    if array.dtype.kind not in "biuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite binary values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(f"{name} must contain only zero or one")
    return tuple(bool(item) for item in array)


@dataclass(frozen=True)
class FrozenSearchProvenance:
    """Reproducible fixed-protocol search outcome for one query/branch."""

    search_artifact_id: str
    search_artifact_sha256: str
    protocol_id: str
    protocol_sha256: str
    evaluator_version: str
    metric_name: str
    threshold_name: str
    threshold_value: float
    threshold_source_id: str
    exact_forward_call_budget: int
    exact_forward_calls_used: int
    termination_reason: str
    completed: bool
    compatible_representative_count: int

    def __post_init__(self) -> None:
        for name in (
            "search_artifact_id",
            "protocol_id",
            "evaluator_version",
            "metric_name",
            "threshold_name",
            "threshold_source_id",
            "termination_reason",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        object.__setattr__(
            self,
            "search_artifact_sha256",
            _digest(self.search_artifact_sha256, "search_artifact_sha256"),
        )
        object.__setattr__(
            self, "protocol_sha256", _digest(self.protocol_sha256, "protocol_sha256")
        )
        threshold = float(self.threshold_value)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold_value must be finite and non-negative")
        budget = _nonnegative_int(self.exact_forward_call_budget, "exact_forward_call_budget")
        used = _nonnegative_int(self.exact_forward_calls_used, "exact_forward_calls_used")
        count = _nonnegative_int(
            self.compatible_representative_count, "compatible_representative_count"
        )
        if budget < 1 or used > budget:
            raise ValueError("exact forward calls must satisfy 0 <= used <= positive budget")
        object.__setattr__(self, "threshold_value", threshold)
        object.__setattr__(self, "exact_forward_call_budget", budget)
        object.__setattr__(self, "exact_forward_calls_used", used)
        object.__setattr__(self, "compatible_representative_count", count)
        object.__setattr__(self, "completed", _boolean(self.completed, "completed"))

    def audit_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ExactCompatibleProvenance:
    """Exact artifact for the representative used as a positive local target."""

    artifact_id: str
    artifact_sha256: str
    metric_value: float
    bounds_passed: bool
    physics_passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text(self.artifact_id, "artifact_id"))
        object.__setattr__(
            self, "artifact_sha256", _digest(self.artifact_sha256, "artifact_sha256")
        )
        metric = float(self.metric_value)
        if not np.isfinite(metric) or metric < 0.0:
            raise ValueError("metric_value must be finite and non-negative")
        object.__setattr__(self, "metric_value", metric)
        object.__setattr__(self, "bounds_passed", _boolean(self.bounds_passed, "bounds_passed"))
        object.__setattr__(self, "physics_passed", _boolean(self.physics_passed, "physics_passed"))

    def audit_payload(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class V5CandidateSupervision:
    """One operational search-yield label and optional compatible local target."""

    clean_recipe_id: str
    candidate_id: str
    outcome: str
    active_dimension_mask: tuple[bool, ...]
    varying_dimension_mask: tuple[bool, ...]
    search_provenance: FrozenSearchProvenance | None = None
    exact_compatible: ExactCompatibleProvenance | None = None
    target_local: tuple[float, ...] | None = None
    clean_recipe_weight: float = 1.0
    candidate_weight: float = 1.0
    generating_candidate_match: bool | None = None
    schema_version: str = CANDIDATE_SUPERVISION_V5_SCHEMA
    version: str = CANDIDATE_SUPERVISION_V5_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "clean_recipe_id", _text(self.clean_recipe_id, "clean_recipe_id"))
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        if self.outcome not in SEARCH_OUTCOME_STATES:
            raise ValueError(f"outcome must be one of {SEARCH_OUTCOME_STATES}")
        active = _mask(self.active_dimension_mask, "active_dimension_mask")
        varying = _mask(self.varying_dimension_mask, "varying_dimension_mask")
        if any(vary and not act for vary, act in zip(varying, active)):
            raise ValueError("varying dimensions must be active")
        object.__setattr__(self, "active_dimension_mask", active)
        object.__setattr__(self, "varying_dimension_mask", varying)
        self._validate_evidence()
        if self.target_local is not None:
            self._validate_target(varying)
        object.__setattr__(
            self,
            "clean_recipe_weight",
            _positive(self.clean_recipe_weight, "clean_recipe_weight"),
        )
        object.__setattr__(
            self, "candidate_weight", _positive(self.candidate_weight, "candidate_weight")
        )
        if self.generating_candidate_match is not None:
            object.__setattr__(
                self,
                "generating_candidate_match",
                _boolean(self.generating_candidate_match, "generating_candidate_match"),
            )
        if self.schema_version != CANDIDATE_SUPERVISION_V5_SCHEMA:
            raise ValueError("unsupported V5 candidate-supervision schema")
        if self.version != CANDIDATE_SUPERVISION_V5_VERSION:
            raise ValueError("unsupported V5 candidate-supervision version")

    def _validate_evidence(self) -> None:
        verified = self.outcome != "unverified"
        if verified and not isinstance(self.search_provenance, FrozenSearchProvenance):
            raise ValueError("verified search outcomes require frozen search provenance")
        if not verified:
            if self.search_provenance is not None or self.exact_compatible is not None:
                raise ValueError("unverified outcomes must not carry completed-search evidence")
            return
        search = self.search_provenance
        if not search.completed:
            raise ValueError("verified search outcomes require a completed frozen protocol")
        known_truth_oracle = search.protocol_id == KNOWN_TRUTH_ORACLE_PROTOCOL_ID
        if known_truth_oracle and (
            self.outcome != "compatible_found"
            or search.exact_forward_call_budget != 1
            or search.exact_forward_calls_used != 1
            or search.compatible_representative_count != 1
            or search.termination_reason != KNOWN_TRUTH_ORACLE_TERMINATION_REASON
        ):
            raise ValueError("known-truth oracle evidence violates its one-call warmup contract")
        if (
            not known_truth_oracle
            and search.exact_forward_calls_used != search.exact_forward_call_budget
        ):
            raise ValueError("verified search outcomes require the equal frozen branch budget")
        if self.outcome == "compatible_found":
            if not isinstance(self.exact_compatible, ExactCompatibleProvenance):
                raise ValueError("compatible_found requires exact-compatible artifact evidence")
            exact = self.exact_compatible
            compatible = (
                exact.metric_value <= search.threshold_value
                and exact.bounds_passed
                and exact.physics_passed
            )
            if not compatible or search.compatible_representative_count < 1:
                raise ValueError("positive outcome disagrees with exact compatibility evidence")
            if (
                not known_truth_oracle
                and search.termination_reason != POSITIVE_TERMINATION_REASON
            ):
                raise ValueError("positive outcome requires the canonical full-budget termination")
            return
        if self.exact_compatible is not None:
            raise ValueError("no-compatible-found outcome cannot carry a positive artifact")
        if search.compatible_representative_count != 0:
            raise ValueError("negative search outcome must report zero compatible representatives")
        if search.termination_reason not in NEGATIVE_TERMINATION_REASONS:
            raise ValueError("negative search outcome requires frozen protocol exhaustion")

    def _validate_target(self, varying: Sequence[bool]) -> None:
        if self.outcome != "compatible_found" or self.exact_compatible is None:
            raise ValueError("only exact-compatible found representatives may be local targets")
        target = np.asarray(self.target_local)
        if target.shape != (UNIT_CUBE_DIMENSIONS,) or target.dtype.kind not in "iuf":
            raise ValueError("target_local must contain 26 numeric values")
        target = np.asarray(target, dtype=np.float64)
        fixed = np.logical_not(np.asarray(varying, dtype=np.bool_))
        if (
            not np.all(np.isfinite(target))
            or np.any(target < 0.0)
            or np.any(target > 1.0)
            or np.any(target[fixed] != INACTIVE_UNIT_VALUE)
        ):
            raise ValueError("target_local must be in [0,1] and fixed/inactive axes equal 0.5")
        object.__setattr__(self, "target_local", tuple(float(value) for value in target))

    @property
    def outcome_code(self) -> int:
        return SEARCH_OUTCOME_CODE[self.outcome]

    @property
    def bce_eligible(self) -> bool:
        return self.outcome != "unverified" and not (
            self.search_provenance is not None
            and self.search_provenance.protocol_id == KNOWN_TRUTH_ORACLE_PROTOCOL_ID
        )

    @property
    def local_mdn_eligible(self) -> bool:
        return self.outcome == "compatible_found" and self.target_local is not None

    def audit_payload(self) -> dict[str, object]:
        payload = {
            "schema_version": self.schema_version,
            "version": self.version,
            "clean_recipe_id": self.clean_recipe_id,
            "candidate_id": self.candidate_id,
            "outcome": self.outcome,
            "outcome_code": self.outcome_code,
            "search_provenance": (
                None if self.search_provenance is None else self.search_provenance.audit_payload()
            ),
            "exact_compatible": (
                None if self.exact_compatible is None else self.exact_compatible.audit_payload()
            ),
            "target_local": None if self.target_local is None else list(self.target_local),
            "bce_eligible": self.bce_eligible,
            "local_mdn_eligible": self.local_mdn_eligible,
            "generating_candidate_match": self.generating_candidate_match,
            "generating_mismatch_is_automatic_negative": False,
        }
        canonical = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        payload["audit_sha256"] = sha256(canonical.encode()).hexdigest()
        return payload

    def tensor_record(self, *, clean_recipe_index: int) -> dict[str, object]:
        index = _nonnegative_int(clean_recipe_index, "clean_recipe_index")
        search, exact = self.search_provenance, self.exact_compatible
        target = self.target_local or (INACTIVE_UNIT_VALUE,) * UNIT_CUBE_DIMENSIONS
        return {
            "search_outcome_code": self.outcome_code,
            "exact_artifact_id": "" if exact is None else exact.artifact_id,
            "exact_artifact_sha256": "" if exact is None else exact.artifact_sha256,
            "exact_metric_value": 0.0 if exact is None else exact.metric_value,
            "exact_bounds_passed": False if exact is None else exact.bounds_passed,
            "exact_physics_passed": False if exact is None else exact.physics_passed,
            "search_artifact_id": "" if search is None else search.search_artifact_id,
            "search_artifact_sha256": "" if search is None else search.search_artifact_sha256,
            "search_protocol_id": "" if search is None else search.protocol_id,
            "search_protocol_sha256": "" if search is None else search.protocol_sha256,
            "search_evaluator_version": "" if search is None else search.evaluator_version,
            "search_metric_name": "" if search is None else search.metric_name,
            "search_threshold_name": "" if search is None else search.threshold_name,
            "search_threshold_value": 0.0 if search is None else search.threshold_value,
            "search_threshold_source_id": "" if search is None else search.threshold_source_id,
            "search_exact_forward_call_budget": (
                0 if search is None else search.exact_forward_call_budget
            ),
            "search_exact_forward_calls_used": (
                0 if search is None else search.exact_forward_calls_used
            ),
            "search_termination_reason": "" if search is None else search.termination_reason,
            "search_completed": False if search is None else search.completed,
            "search_compatible_representative_count": (
                0 if search is None else search.compatible_representative_count
            ),
            "target_local": target,
            "has_local_target": self.target_local is not None,
            "active_dimension_mask": self.active_dimension_mask,
            "varying_dimension_mask": self.varying_dimension_mask,
            "clean_recipe_index": index,
            "clean_recipe_weight": self.clean_recipe_weight,
            "candidate_weight": self.candidate_weight,
        }


def candidate_supervision_v5_contract_payload() -> dict[str, object]:
    return {
        "schema_version": CANDIDATE_SUPERVISION_V5_SCHEMA,
        "version": CANDIDATE_SUPERVISION_V5_VERSION,
        "search_outcomes": list(SEARCH_OUTCOME_STATES),
        "outcome_codes": dict(SEARCH_OUTCOME_CODE),
        "tensor_keys": list(CANDIDATE_SUPERVISION_TENSOR_KEYS),
        "exact_comparison_semantics": EXACT_COMPARISON_SEMANTICS,
        "recipe_weighting_semantics": RECIPE_WEIGHTING_SEMANTICS,
        "logit_estimand": ("at_least_one_exact_compatible_representative_found_by_frozen_protocol"),
        "unverified_enters_bce": False,
        "generating_mismatch_is_automatic_negative": False,
        "negative_is_no_solution_certificate": False,
        "known_truth_oracle_protocol_id": KNOWN_TRUTH_ORACLE_PROTOCOL_ID,
        "known_truth_oracle_termination_reason": KNOWN_TRUTH_ORACLE_TERMINATION_REASON,
        "known_truth_oracle_enters_search_yield_bce": False,
        "frozen_search_positive_termination_reason": POSITIVE_TERMINATION_REASON,
        "frozen_search_verified_requires_equal_branch_budget": True,
    }


def validate_candidate_supervision_v5_contract(
    payload: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        raise TypeError("candidate-supervision contract must be a mapping")
    value = deepcopy(dict(payload))
    if value != candidate_supervision_v5_contract_payload():
        raise ValueError("candidate-supervision contract is missing or incompatible")
    return value


def stack_candidate_supervision_v5(
    candidates: Sequence[V5CandidateSupervision],
    *,
    clean_recipe_indices: Sequence[int],
) -> dict[str, np.ndarray]:
    """Stack validated labels into framework-neutral array tensors."""

    items, indices = tuple(candidates), tuple(clean_recipe_indices)
    if not items or len(items) != len(indices):
        raise ValueError("candidates and clean_recipe_indices must have the same non-zero length")
    if not all(isinstance(item, V5CandidateSupervision) for item in items):
        raise TypeError("all candidates must be V5CandidateSupervision instances")
    records = [item.tensor_record(clean_recipe_index=index) for item, index in zip(items, indices)]
    string_keys = {
        key for key in CANDIDATE_SUPERVISION_TENSOR_KEYS if "_id" in key or "sha256" in key
    }
    string_keys.update(
        {
            "search_evaluator_version",
            "search_metric_name",
            "search_threshold_name",
            "search_termination_reason",
        }
    )
    bool_keys = {
        "exact_bounds_passed",
        "exact_physics_passed",
        "search_completed",
        "has_local_target",
    }
    integer_keys = {
        "search_outcome_code",
        "search_exact_forward_call_budget",
        "search_exact_forward_calls_used",
        "search_compatible_representative_count",
        "clean_recipe_index",
    }
    result = {}
    for key in CANDIDATE_SUPERVISION_TENSOR_KEYS:
        if key in string_keys:
            dtype = np.str_
        elif key in bool_keys:
            dtype = np.bool_
        elif key in integer_keys:
            dtype = np.int32
        else:
            dtype = np.float32
        result[key] = np.asarray([record[key] for record in records], dtype=dtype)
    return result


__all__ = [
    "CANDIDATE_SUPERVISION_TENSOR_KEYS",
    "CANDIDATE_SUPERVISION_V5_SCHEMA",
    "CANDIDATE_SUPERVISION_V5_VERSION",
    "EXACT_COMPARISON_SEMANTICS",
    "ExactCompatibleProvenance",
    "FrozenSearchProvenance",
    "KNOWN_TRUTH_ORACLE_PROTOCOL_ID",
    "KNOWN_TRUTH_ORACLE_TERMINATION_REASON",
    "NEGATIVE_TERMINATION_REASONS",
    "POSITIVE_TERMINATION_REASON",
    "RECIPE_WEIGHTING_SEMANTICS",
    "SEARCH_OUTCOME_CODE",
    "SEARCH_OUTCOME_FROM_CODE",
    "SEARCH_OUTCOME_STATES",
    "V5CandidateSupervision",
    "candidate_supervision_v5_contract_payload",
    "stack_candidate_supervision_v5",
    "validate_candidate_supervision_v5_contract",
]
