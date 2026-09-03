"""User-contract-aware symmetry catalog for Posterior V8 wire branches.

The five-bit wire representation remains unchanged: component D-presence uses
bits 0..3 and Resolution presence uses bit 4.  Only component slots with the
same shape, exactly equal complete GUI geometry bounds (including D policy),
and exactly equal ``Int_i`` bounds are exchangeable.  In particular,
overlapping or otherwise heterogeneous ranges never authorize a
slot-permutation quotient.  A geometry-only caller has incomplete context and
therefore receives the fail-closed, no-permutation-quotient catalog.

This module applies discrete presence policies and the contextual symmetry
quotient.  It deliberately does not test continuous hard-core feasibility;
that remains the branch codec/refinement layer's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Literal, Sequence

import numpy as np

from .branch_catalog import (
    BRANCH_PATTERN_COUNT,
    branch_pattern_id,
    decode_branch_pattern,
)
from .contract import (
    MAX_COMPONENTS,
    ClosedInterval,
    GuiComponentBounds,
    canonical_topology,
    topology_id_for,
)


CONTEXTUAL_BRANCH_CATALOG_SCHEMA = "gisaxs.posterior_v8.contextual_branch_catalog/v2"
CONTEXTUAL_BRANCH_CATALOG_VERSION = (
    "posterior_v8_complete_geometry_d_policy_amplitude_exchangeability_branch_v2"
)
CONTEXTUAL_BRANCH_CATALOG_SEMANTICS = (
    "policy_allowed_five_bit_wire_patterns_quotiented_only_by_exactly_equal_"
    "complete_gui_geometry_bounds_d_policy_and_component_intensity_bounds;"
    "missing_amplitude_context_disables_slot_permutation_quotient;without_"
    "continuous_feasibility_filtering"
)
CONTEXTUAL_D_ORDER = (
    "d_absent_before_d_present_within_exact_complete_slot_contract_equivalence_class"
)

PresencePolicy = Literal["absent", "optional", "required"]
PRESENCE_POLICIES: tuple[PresencePolicy, ...] = (
    "absent",
    "optional",
    "required",
)


def _component_context(
    component_bounds: Sequence[GuiComponentBounds],
) -> tuple[tuple[GuiComponentBounds, ...], tuple[str, ...], int]:
    if isinstance(component_bounds, (str, bytes)):
        raise TypeError("component_bounds must be a sequence of GuiComponentBounds")
    try:
        bounds = tuple(component_bounds)
    except TypeError as exc:
        raise TypeError("component_bounds must be a sequence of GuiComponentBounds") from exc
    if not 1 <= len(bounds) <= MAX_COMPONENTS:
        raise ValueError("component_bounds must contain between one and four components")
    if not all(isinstance(value, GuiComponentBounds) for value in bounds):
        raise TypeError("component_bounds must contain only GuiComponentBounds")
    topology = tuple(value.shape for value in bounds)
    expected = canonical_topology(topology)
    if topology != expected:
        raise ValueError(
            "component_bounds must use canonical topology order; "
            f"expected {expected!r}, got {topology!r}"
        )
    return bounds, topology, topology_id_for(topology)


def _presence_policy(value: object, name: str) -> PresencePolicy:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value not in PRESENCE_POLICIES:
        raise ValueError(f"{name} must be absent, optional, or required")
    return value


def _component_d_policy(bounds: GuiComponentBounds) -> PresencePolicy:
    if bounds.D is None:
        return "absent"
    return "optional" if bounds.allow_D_absent else "required"


def _validated_d_flags(d_present: Sequence[bool], component_count: int) -> tuple[bool, ...]:
    if isinstance(d_present, (str, bytes)):
        raise TypeError("d_present must be a sequence of four booleans")
    try:
        flags = tuple(d_present)
    except TypeError as exc:
        raise TypeError("d_present must be a sequence of four booleans") from exc
    if len(flags) != MAX_COMPONENTS:
        raise ValueError("d_present must contain exactly four boolean wire flags")
    if not all(isinstance(value, (bool, np.bool_)) for value in flags):
        raise TypeError("d_present must contain only boolean values")
    result = tuple(bool(value) for value in flags)
    if any(result[component_count:]):
        raise ValueError("unused component D flags must be false")
    return result


def _component_intensity_context(
    component_intensity_bounds: Sequence[ClosedInterval] | None,
    component_count: int,
) -> tuple[ClosedInterval, ...] | None:
    if component_intensity_bounds is None:
        return None
    if isinstance(component_intensity_bounds, (str, bytes)):
        raise TypeError(
            "component_intensity_bounds must be a sequence of ClosedInterval values"
        )
    try:
        intervals = tuple(component_intensity_bounds)
    except TypeError as exc:
        raise TypeError(
            "component_intensity_bounds must be a sequence of ClosedInterval values"
        ) from exc
    if len(intervals) != component_count:
        raise ValueError(
            "component_intensity_bounds must contain one interval per component slot"
        )
    if not all(isinstance(value, ClosedInterval) for value in intervals):
        raise TypeError("component_intensity_bounds must contain only ClosedInterval values")
    return intervals


def _equivalence_classes_from_validated_context(
    bounds: tuple[GuiComponentBounds, ...],
    component_intensity_bounds: tuple[ClosedInterval, ...] | None,
) -> tuple[tuple[int, ...], ...]:
    # Without the amplitude half of the GUI contract, treating two slots as
    # exchangeable could erase a legal wire branch once Int_i ranges arrive.
    if component_intensity_bounds is None:
        return tuple((slot,) for slot in range(len(bounds)))
    remaining = set(range(len(bounds)))
    classes = []
    for slot in range(len(bounds)):
        if slot not in remaining:
            continue
        equivalent = tuple(
            candidate
            for candidate in range(slot, len(bounds))
            if bounds[candidate] == bounds[slot]
            and component_intensity_bounds[candidate] == component_intensity_bounds[slot]
        )
        remaining.difference_update(equivalent)
        classes.append(equivalent)
    return tuple(classes)


def component_slot_contract_equivalence_classes(
    component_bounds: Sequence[GuiComponentBounds],
    *,
    component_intensity_bounds: Sequence[ClosedInterval] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return exchangeable slots under the complete paired GUI contract.

    Exact dataclass equality is intentional: overlap is not identity.  When
    amplitude ranges are unavailable, each labelled slot remains a singleton.
    Branch-specific D presence is not part of this query-level helper; fixed
    branch consumers must refine these classes by their D-present flags.
    """

    bounds, _, _ = _component_context(component_bounds)
    intensity_bounds = _component_intensity_context(
        component_intensity_bounds,
        len(bounds),
    )
    return _equivalence_classes_from_validated_context(bounds, intensity_bounds)


def _policy_allowed(policy: PresencePolicy, present: bool) -> bool:
    return policy == "optional" or present == (policy == "required")


def contextual_canonicalize_d_flags(
    component_bounds: Sequence[GuiComponentBounds],
    d_present: Sequence[bool],
    *,
    component_intensity_bounds: Sequence[ClosedInterval] | None = None,
) -> tuple[bool, ...]:
    """Return the policy-valid contextual representative of four D wire bits.

    A policy conflict or a set bit for an unused component is invalid input,
    rather than a branch that can be silently canonicalized into validity.
    """

    bounds, _, _ = _component_context(component_bounds)
    intensity_bounds = _component_intensity_context(
        component_intensity_bounds, len(bounds)
    )
    flags = _validated_d_flags(d_present, len(bounds))
    policies = tuple(_component_d_policy(value) for value in bounds)
    for slot, (policy, present) in enumerate(zip(policies, flags)):
        if not _policy_allowed(policy, present):
            raise ValueError(
                f"component slot {slot} D-present={present} conflicts with {policy} D policy"
            )

    result = list(flags)
    for equivalent_slots in component_slot_contract_equivalence_classes(
        bounds,
        component_intensity_bounds=intensity_bounds,
    ):
        present_count = sum(result[slot] for slot in equivalent_slots)
        ordered = (False,) * (len(equivalent_slots) - present_count) + (True,) * present_count
        for slot, present in zip(equivalent_slots, ordered):
            result[slot] = present
    return tuple(result)


def contextual_canonical_branch_pattern_id(
    component_bounds: Sequence[GuiComponentBounds],
    pattern_id: int,
    *,
    component_intensity_bounds: Sequence[ClosedInterval] | None = None,
    resolution_presence_policy: PresencePolicy,
) -> int:
    """Map one policy-valid 0..31 wire ID to its contextual representative."""

    resolution_policy = _presence_policy(resolution_presence_policy, "resolution_presence_policy")
    d_present, resolution_present = decode_branch_pattern(pattern_id)
    if not _policy_allowed(resolution_policy, resolution_present):
        raise ValueError(
            f"Resolution-present={resolution_present} conflicts with "
            f"{resolution_policy} Resolution policy"
        )
    canonical_flags = contextual_canonicalize_d_flags(
        component_bounds,
        d_present,
        component_intensity_bounds=component_intensity_bounds,
    )
    return branch_pattern_id(canonical_flags, resolution_present)


@dataclass(frozen=True)
class ContextualBranchCatalog:
    """One auditable policy-filtered contextual quotient on 0..31 wire IDs."""

    component_bounds: tuple[GuiComponentBounds, ...]
    component_intensity_bounds: tuple[ClosedInterval, ...] | None
    topology_id: int
    topology: tuple[str, ...]
    d_policies: tuple[PresencePolicy, ...]
    resolution_presence_policy: PresencePolicy
    d_equivalence_classes: tuple[tuple[int, ...], ...]
    wire_pattern_ids: tuple[int, ...]
    wire_pattern_mask: tuple[bool, ...]
    catalog_schema: str = field(default=CONTEXTUAL_BRANCH_CATALOG_SCHEMA, init=False)
    catalog_version: str = field(default=CONTEXTUAL_BRANCH_CATALOG_VERSION, init=False)
    semantics: str = field(default=CONTEXTUAL_BRANCH_CATALOG_SEMANTICS, init=False)

    def audit_payload(self) -> dict[str, object]:
        """Return stable catalog identity and derived discrete context metadata."""

        return {
            "catalog_schema": self.catalog_schema,
            "catalog_version": self.catalog_version,
            "semantics": self.semantics,
            "canonical_d_order": CONTEXTUAL_D_ORDER,
            "topology_id": self.topology_id,
            "topology": list(self.topology),
            "component_intensity_bounds": (
                None
                if self.component_intensity_bounds is None
                else [
                    {"low": value.low, "high": value.high}
                    for value in self.component_intensity_bounds
                ]
            ),
            "exchangeability_context_complete": self.component_intensity_bounds is not None,
            "d_policies": list(self.d_policies),
            "resolution_presence_policy": self.resolution_presence_policy,
            "d_equivalence_classes": [list(value) for value in self.d_equivalence_classes],
            "wire_pattern_ids": list(self.wire_pattern_ids),
        }


def build_contextual_branch_catalog(
    component_bounds: Sequence[GuiComponentBounds],
    *,
    component_intensity_bounds: Sequence[ClosedInterval] | None = None,
    resolution_presence_policy: PresencePolicy,
) -> ContextualBranchCatalog:
    """Enumerate policy-allowed contextual-canonical wire IDs and a 32-mask.

    ``component_intensity_bounds=None`` is deliberately fail closed: no slot
    permutation quotient is applied until the complete geometry/amplitude GUI
    contract is available.
    """

    bounds, topology, topology_id = _component_context(component_bounds)
    intensity_bounds = _component_intensity_context(
        component_intensity_bounds, len(bounds)
    )
    resolution_policy = _presence_policy(resolution_presence_policy, "resolution_presence_policy")
    policies = tuple(_component_d_policy(value) for value in bounds)
    d_options = tuple(
        (False, True) if policy == "optional" else (policy == "required",) for policy in policies
    )
    contextual_d_flags = {
        contextual_canonicalize_d_flags(
            bounds,
            tuple(active_flags) + (False,) * (MAX_COMPONENTS - len(bounds)),
            component_intensity_bounds=intensity_bounds,
        )
        for active_flags in product(*d_options)
    }
    resolution_options = (
        (False, True) if resolution_policy == "optional" else (resolution_policy == "required",)
    )
    pattern_ids = tuple(
        sorted(
            branch_pattern_id(d_flags, resolution_present)
            for d_flags in contextual_d_flags
            for resolution_present in resolution_options
        )
    )
    selected = set(pattern_ids)
    pattern_mask = tuple(pattern_id in selected for pattern_id in range(BRANCH_PATTERN_COUNT))
    return ContextualBranchCatalog(
        component_bounds=bounds,
        component_intensity_bounds=intensity_bounds,
        topology_id=topology_id,
        topology=topology,
        d_policies=policies,
        resolution_presence_policy=resolution_policy,
        d_equivalence_classes=component_slot_contract_equivalence_classes(
            bounds,
            component_intensity_bounds=intensity_bounds,
        ),
        wire_pattern_ids=pattern_ids,
        wire_pattern_mask=pattern_mask,
    )


__all__ = [
    "CONTEXTUAL_BRANCH_CATALOG_SCHEMA",
    "CONTEXTUAL_BRANCH_CATALOG_SEMANTICS",
    "CONTEXTUAL_BRANCH_CATALOG_VERSION",
    "CONTEXTUAL_D_ORDER",
    "PRESENCE_POLICIES",
    "ContextualBranchCatalog",
    "PresencePolicy",
    "build_contextual_branch_catalog",
    "component_slot_contract_equivalence_classes",
    "contextual_canonical_branch_pattern_id",
    "contextual_canonicalize_d_flags",
]
