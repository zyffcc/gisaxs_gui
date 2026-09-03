"""Stable identities and slot bindings for cross-topology V5.1 queries."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from numbers import Integral

import numpy as np

from .amplitude_query_v5 import V5AmplitudeQuery
from .bounds_query_v5 import V5BoundsQuery, V5BranchCondition
from .branch_catalog import BRANCH_PATTERN_COUNT
from .contextual_branch_catalog import (
    ContextualBranchCatalog,
    build_contextual_branch_catalog,
)
from .contract import ClosedInterval, NUM_TOPOLOGIES, topology_from_id
from .gui_amplitude_constraints import GuiAmplitudeConstraint


V5_TOPOLOGY_QUERY_VERSION = (
    "posterior_v8_numeric_contract_bound_topology_complete_slot_query_pair_v4"
)
V5_GLOBAL_BRANCH_KEY_VERSION = "posterior_v8_global_topology_wire_branch_key_v1"
V5_CONTEXT_BRANCH_KEY_VERSION = "posterior_v8_query_bound_global_branch_context_v1"


def canonical_universal_json(payload: object) -> str:
    return json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)


def universal_json_sha256(payload: object) -> str:
    return sha256(canonical_universal_json(payload).encode("utf-8")).hexdigest()


def validated_topology_id(value: int, name: str = "topology_id") -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result < NUM_TOPOLOGIES:
        raise ValueError(f"{name} must be in [0, {NUM_TOPOLOGIES - 1}]")
    return result


def _interval_payload(value: ClosedInterval) -> dict[str, float]:
    return {"low": float(value.low), "high": float(value.high)}


@dataclass(frozen=True)
class V5TopologyComponentSlot:
    """Lossless binding between one geometry slot and its amplitude slot."""

    topology_id: int
    slot_index: int
    shape: str
    component_intensity: ClosedInterval

    def __post_init__(self) -> None:
        topology_id = validated_topology_id(self.topology_id)
        topology = topology_from_id(topology_id)
        if isinstance(self.slot_index, (bool, np.bool_)) or not isinstance(
            self.slot_index, Integral
        ):
            raise TypeError("slot_index must be an integer")
        slot = int(self.slot_index)
        if not 0 <= slot < len(topology):
            raise ValueError("slot_index is outside its topology")
        if self.shape != topology[slot]:
            raise ValueError("component slot shape disagrees with the stable topology catalog")
        if not isinstance(self.component_intensity, ClosedInterval):
            raise TypeError("component_intensity must be a ClosedInterval")
        object.__setattr__(self, "topology_id", topology_id)
        object.__setattr__(self, "slot_index", slot)

    @property
    def global_slot_key(self) -> str:
        return f"topology-{self.topology_id:02d}:slot-{self.slot_index}:{self.shape}"

    def audit_payload(self) -> dict[str, object]:
        return {
            "global_slot_key": self.global_slot_key,
            "topology_id": self.topology_id,
            "slot_index": self.slot_index,
            "shape": self.shape,
            "amplitude_axis": f"Int_{self.slot_index + 1}",
            "component_intensity": _interval_payload(self.component_intensity),
        }


@dataclass(frozen=True)
class V5TopologyQuery:
    """One explicitly paired geometry/amplitude query for one topology."""

    geometry: V5BoundsQuery
    amplitude: V5AmplitudeQuery
    version: str = V5_TOPOLOGY_QUERY_VERSION

    def __post_init__(self) -> None:
        if self.version != V5_TOPOLOGY_QUERY_VERSION:
            raise ValueError("unsupported V5 topology-query version")
        if not isinstance(self.geometry, V5BoundsQuery):
            raise TypeError("geometry must be a V5BoundsQuery")
        if not isinstance(self.amplitude, V5AmplitudeQuery):
            raise TypeError("amplitude must be a V5AmplitudeQuery")
        if self.amplitude.particle_count != len(self.geometry.topology):
            raise ValueError(
                "geometry and amplitude queries have different topology component counts"
            )
        if (
            self.amplitude.resolution_presence_policy
            != self.geometry.resolution_presence_policy
        ):
            raise ValueError("geometry and amplitude queries have different Resolution policies")
        if self.amplitude.numeric_policy_version != self.geometry.numeric_policy_version:
            raise ValueError("geometry and amplitude queries have different numeric policies")
        if not self.feasible_wire_pattern_ids:
            raise ValueError("paired geometry/amplitude query has no feasible wire branch")

    @property
    def topology_id(self) -> int:
        return self.geometry.topology_id

    @property
    def topology(self) -> tuple[str, ...]:
        return self.geometry.topology

    @property
    def component_slots(self) -> tuple[V5TopologyComponentSlot, ...]:
        return tuple(
            V5TopologyComponentSlot(
                topology_id=self.topology_id,
                slot_index=slot,
                shape=bounds.shape,
                component_intensity=self.amplitude.component_intensities[slot],
            )
            for slot, bounds in enumerate(self.geometry.component_bounds)
        )

    @property
    def contextual_branch_catalog(self) -> ContextualBranchCatalog:
        """Return the quotient defined by the complete paired GUI contract."""

        return build_contextual_branch_catalog(
            self.geometry.component_bounds,
            component_intensity_bounds=self.amplitude.component_intensities,
            resolution_presence_policy=self.geometry.resolution_presence_policy,
        )

    @property
    def policy_wire_pattern_ids(self) -> tuple[int, ...]:
        return self.contextual_branch_catalog.wire_pattern_ids

    @property
    def feasible_wire_pattern_ids(self) -> tuple[int, ...]:
        geometry_feasible = set(self.geometry.feasible_wire_pattern_ids)
        return tuple(
            pattern_id
            for pattern_id in self.policy_wire_pattern_ids
            if pattern_id in geometry_feasible
        )

    def audit_payload(self) -> dict[str, object]:
        return {
            "version": self.version,
            "topology_id": self.topology_id,
            "topology": list(self.topology),
            "geometry_query_sha256": self.geometry.sha256,
            "amplitude_query_sha256": self.amplitude.sha256,
            "resolution_presence_policy": self.geometry.resolution_presence_policy,
            "numeric_policy_version": self.geometry.numeric_policy_version,
            "component_slots": [value.audit_payload() for value in self.component_slots],
            "contextual_branch_catalog": self.contextual_branch_catalog.audit_payload(),
            "feasible_wire_pattern_ids": list(self.feasible_wire_pattern_ids),
        }

    @property
    def sha256(self) -> str:
        return universal_json_sha256(self.audit_payload())


@dataclass(frozen=True, order=True)
class V5GlobalBranchKey:
    """Query-independent identity of one topology/wire branch."""

    topology_id: int
    pattern_id: int

    def __post_init__(self) -> None:
        topology_id = validated_topology_id(self.topology_id)
        if isinstance(self.pattern_id, (bool, np.bool_)) or not isinstance(
            self.pattern_id, Integral
        ):
            raise TypeError("pattern_id must be an integer")
        pattern = int(self.pattern_id)
        if not 0 <= pattern < BRANCH_PATTERN_COUNT:
            raise ValueError(f"pattern_id must be in [0, {BRANCH_PATTERN_COUNT - 1}]")
        object.__setattr__(self, "topology_id", topology_id)
        object.__setattr__(self, "pattern_id", pattern)

    @property
    def wire_key(self) -> str:
        return f"topology-{self.topology_id:02d}:wire-{self.pattern_id:02d}"

    @property
    def topology(self) -> tuple[str, ...]:
        return topology_from_id(self.topology_id)

    def audit_payload(self) -> dict[str, object]:
        return {
            "version": V5_GLOBAL_BRANCH_KEY_VERSION,
            "wire_key": self.wire_key,
            "topology_id": self.topology_id,
            "topology": list(self.topology),
            "pattern_id": self.pattern_id,
        }


@dataclass(frozen=True)
class V5UniversalBranchContext:
    """Stable row mapping and exact amplitude domain for one global branch."""

    global_index: int
    topology_batch_index: int
    branch_batch_index: int
    global_key: V5GlobalBranchKey
    topology_query_sha256: str
    condition: V5BranchCondition
    amplitude_constraint: GuiAmplitudeConstraint
    amplitude_constraint_sha256: str
    context_sha256: str

    def audit_payload(self) -> dict[str, object]:
        return {
            "global_index": self.global_index,
            "topology_batch_index": self.topology_batch_index,
            "branch_batch_index": self.branch_batch_index,
            "global_branch_key": self.global_key.wire_key,
            "topology_query_sha256": self.topology_query_sha256,
            "context_sha256": self.context_sha256,
            "amplitude_constraint_sha256": self.amplitude_constraint_sha256,
        }


def amplitude_constraint_sha256(value: GuiAmplitudeConstraint) -> str:
    return universal_json_sha256(value.to_audit_dict())


def branch_context_sha256(
    global_key: V5GlobalBranchKey,
    topology_query_sha256: str,
    amplitude_constraint_digest: str,
) -> str:
    return universal_json_sha256(
        {
            "version": V5_CONTEXT_BRANCH_KEY_VERSION,
            "global_branch_key": global_key.wire_key,
            "topology_query_sha256": topology_query_sha256,
            "amplitude_constraint_sha256": amplitude_constraint_digest,
        }
    )


__all__ = [
    "V5_CONTEXT_BRANCH_KEY_VERSION",
    "V5_GLOBAL_BRANCH_KEY_VERSION",
    "V5_TOPOLOGY_QUERY_VERSION",
    "V5GlobalBranchKey",
    "V5TopologyComponentSlot",
    "V5TopologyQuery",
    "V5UniversalBranchContext",
    "amplitude_constraint_sha256",
    "branch_context_sha256",
    "canonical_universal_json",
    "universal_json_sha256",
    "validated_topology_id",
]
