"""Strict record and artifact contracts for the V5 split leakage audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping, Sequence

import numpy as np

from .split_design_v5 import MAIN_SPLITS, OOD_LABELS
from .split_leakage_features_v5 import (
    V5_CLEAN_PHYSICS_LEAKAGE_DIM,
    V5_QUERY_BOUNDS_LEAKAGE_DIM,
)


V5_LEAKAGE_AUDIT_SCHEMA = "gisaxs.posterior_v8.split_leakage_audit/v1"
V5_LEAKAGE_AUDIT_VERSION = (
    "posterior_v8_clean_group_cross_split_exact_and_nearest_neighbour_audit_v1"
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _finite(value: float, name: str, *, low: float, high: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite in [{low}, {high}]") from exc
    if not np.isfinite(result) or not low <= result <= high:
        raise ValueError(f"{name} must be finite in [{low}, {high}]")
    return result


def _normalized_vector(
    value: Sequence[float],
    name: str,
    *,
    expected_dimension: int,
) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (expected_dimension,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {expected_dimension}D vector")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} must be normalized to [0, 1]")
    return tuple(float(item) for item in array)


def _sha256_id(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest") from exc
    if len(decoded) != 32 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, kw_only=True)
class V5OODDefinition:
    held_out_topology_ids: tuple[int, ...]
    held_out_range_width_labels: tuple[str, ...]
    weak_component_max_fraction: float
    held_out_acquisition_policy_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        topologies = tuple(
            _integer(value, f"held_out_topology_ids[{index}]")
            for index, value in enumerate(self.held_out_topology_ids)
        )
        if not topologies or len(set(topologies)) != len(topologies) or max(topologies) > 33:
            raise ValueError("held_out_topology_ids must be unique IDs in [0, 33]")
        ranges = tuple(self.held_out_range_width_labels)
        policies = tuple(self.held_out_acquisition_policy_ids)
        for values, name in (
            (ranges, "held_out_range_width_labels"),
            (policies, "held_out_acquisition_policy_ids"),
        ):
            if not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"{name} must contain non-empty strings")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates")
        threshold = _finite(
            self.weak_component_max_fraction,
            "weak_component_max_fraction",
            low=0.0,
            high=1.0,
        )
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("weak_component_max_fraction must lie strictly inside (0, 1)")
        object.__setattr__(self, "held_out_topology_ids", topologies)
        object.__setattr__(self, "held_out_range_width_labels", ranges)
        object.__setattr__(self, "weak_component_max_fraction", threshold)
        object.__setattr__(self, "held_out_acquisition_policy_ids", policies)


@dataclass(frozen=True, kw_only=True)
class V5LeakageAuditConfig:
    physics_near_duplicate_rms_threshold: float = 1.0e-10
    observation_near_duplicate_rms_threshold: float = 1.0e-10
    require_complete_plan: bool = True
    maximum_reported_violations: int = 100

    def __post_init__(self) -> None:
        for name in (
            "physics_near_duplicate_rms_threshold",
            "observation_near_duplicate_rms_threshold",
        ):
            value = _finite(getattr(self, name), name, low=0.0, high=1.0)
            object.__setattr__(self, name, value)
        if not isinstance(self.require_complete_plan, (bool, np.bool_)):
            raise TypeError("require_complete_plan must be boolean")
        object.__setattr__(self, "require_complete_plan", bool(self.require_complete_plan))
        object.__setattr__(
            self,
            "maximum_reported_violations",
            _integer(
                self.maximum_reported_violations,
                "maximum_reported_violations",
                minimum=1,
            ),
        )


@dataclass(frozen=True, kw_only=True)
class V5LeakageAuditRow:
    design_index: int
    clean_group_id: str
    assigned_split: str
    ood_label: str | None
    view_index: int
    normalized_continuous_parameters: tuple[float, ...]
    normalized_query_bounds: tuple[float, ...]
    topology_id: int
    branch_pattern_id: int
    acquisition_policy_id: str
    range_width_label: str
    weakest_particle_fraction: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "design_index", _integer(self.design_index, "design_index"))
        object.__setattr__(
            self, "clean_group_id", _sha256_id(self.clean_group_id, "clean_group_id")
        )
        if self.assigned_split not in MAIN_SPLITS:
            raise ValueError(f"assigned_split must be one of {MAIN_SPLITS}")
        if self.ood_label is not None and self.ood_label not in OOD_LABELS:
            raise ValueError(f"ood_label must be None or one of {OOD_LABELS}")
        object.__setattr__(self, "view_index", _integer(self.view_index, "view_index"))
        object.__setattr__(
            self,
            "normalized_continuous_parameters",
            _normalized_vector(
                self.normalized_continuous_parameters,
                "normalized_continuous_parameters",
                expected_dimension=V5_CLEAN_PHYSICS_LEAKAGE_DIM,
            ),
        )
        object.__setattr__(
            self,
            "normalized_query_bounds",
            _normalized_vector(
                self.normalized_query_bounds,
                "normalized_query_bounds",
                expected_dimension=V5_QUERY_BOUNDS_LEAKAGE_DIM,
            ),
        )
        topology = _integer(self.topology_id, "topology_id")
        branch = _integer(self.branch_pattern_id, "branch_pattern_id")
        if topology > 33 or branch > 31:
            raise ValueError("topology_id/branch_pattern_id exceed the frozen wire ranges")
        object.__setattr__(self, "topology_id", topology)
        object.__setattr__(self, "branch_pattern_id", branch)
        for name in ("acquisition_policy_id", "range_width_label"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        object.__setattr__(
            self,
            "weakest_particle_fraction",
            _finite(
                self.weakest_particle_fraction,
                "weakest_particle_fraction",
                low=0.0,
                high=1.0,
            ),
        )


@dataclass(frozen=True)
class V5LeakageAudit:
    payload: Mapping[str, object]
    sha256: str

    @classmethod
    def create(cls, payload: Mapping[str, object]) -> "V5LeakageAudit":
        value = json.loads(canonical_json(dict(payload)))
        if value.get("schema") != V5_LEAKAGE_AUDIT_SCHEMA:
            raise ValueError("unsupported V5 leakage-audit schema")
        canonical = canonical_json(value)
        return cls(value, sha256(canonical.encode("utf-8")).hexdigest())

    @property
    def passed(self) -> bool:
        return self.payload.get("status") == "passed"

    def to_json(self) -> str:
        value = dict(self.payload)
        value["audit_sha256"] = self.sha256
        return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, encoded: str) -> "V5LeakageAudit":
        def reject_duplicates(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON field {key!r}")
                result[key] = value
            return result

        try:
            value = json.loads(encoded, object_pairs_hook=reject_duplicates)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid V5 leakage-audit JSON") from exc
        if not isinstance(value, dict) or value.get("schema") != V5_LEAKAGE_AUDIT_SCHEMA:
            raise ValueError("unsupported V5 leakage-audit schema")
        supplied = value.pop("audit_sha256", None)
        result = cls.create(value)
        if supplied != result.sha256:
            raise ValueError("V5 leakage-audit hash does not reproduce")
        return result


class V5LeakageViolation(ValueError):
    def __init__(self, audit: V5LeakageAudit):
        self.audit = audit
        count = audit.payload["violations"]["total_count"]
        super().__init__(f"V5 leakage audit failed with {count} violation(s)")


class ViolationCollector:
    def __init__(self, maximum: int):
        self.maximum = maximum
        self.total = 0
        self.records: list[dict[str, object]] = []

    def add(self, code: str, **details: object) -> None:
        self.total += 1
        if len(self.records) < self.maximum:
            self.records.append({"code": code, **details})


def row_fingerprint(row: V5LeakageAuditRow) -> dict[str, object]:
    return {
        **asdict(row),
        "normalized_continuous_parameters": [
            value.hex() for value in row.normalized_continuous_parameters
        ],
        "normalized_query_bounds": [value.hex() for value in row.normalized_query_bounds],
        "weakest_particle_fraction": row.weakest_particle_fraction.hex(),
    }


def parent_signature(row: V5LeakageAuditRow) -> tuple[object, ...]:
    return (
        row.design_index,
        row.assigned_split,
        row.ood_label,
        row.normalized_continuous_parameters,
        row.normalized_query_bounds,
        row.topology_id,
        row.branch_pattern_id,
        row.range_width_label,
        row.weakest_particle_fraction,
    )


def support_flags(row: V5LeakageAuditRow, definition: V5OODDefinition) -> dict[str, bool]:
    return {
        "topology_holdout": row.topology_id in definition.held_out_topology_ids,
        "range_width_holdout": row.range_width_label in definition.held_out_range_width_labels,
        "weak_component_holdout": row.weakest_particle_fraction
        <= definition.weak_component_max_fraction,
        "acquisition_policy_holdout": row.acquisition_policy_id
        in definition.held_out_acquisition_policy_ids,
    }


__all__ = [
    "V5_LEAKAGE_AUDIT_SCHEMA",
    "V5_LEAKAGE_AUDIT_VERSION",
    "V5LeakageAudit",
    "V5LeakageAuditConfig",
    "V5LeakageAuditRow",
    "V5LeakageViolation",
    "V5OODDefinition",
]
