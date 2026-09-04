"""Deterministic component-slot quotient under the complete user contract.

A slot permutation is a symmetry only when the corresponding slots have
exactly equal GUI geometry bounds (including the D policy), equal branch
D-presence, and exactly equal per-slot ``Int_i`` bounds.  Geometry overlap is
not equivalence.  Missing amplitude context is treated conservatively: every
slot is its own labelled class and no permutation quotient is applied.

At most ``4!`` assignments are inspected.  Feasibility is decided exclusively
by the supplied authoritative user-bounds codec, and the lexicographically
smallest physical assignment *inside each proven equivalence class* is
returned.  Labeled order is preserved across non-equivalent classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from typing import Sequence

import numpy as np

from .branch_codec import BranchCoordinates, ProfiledBranchCodec
from .contract import ClosedInterval, LatentComponentParameters, latent_component_to_gui
from .contextual_branch_catalog import component_slot_contract_equivalence_classes
from .profiled_forward import ResolutionShape
from .sobol_numeric_canonicalization_v5 import V5_FAST_NUMERIC_POLICY_VERSION


CANONICAL_COMPONENT_SLOTS_VERSION = (
    "posterior_v8_policy_bound_decode_fixed_point_exact_endpoint_slots_v5"
)
CANONICAL_COMPONENT_ROUNDTRIP_ATOL = 5.0e-12
MAX_CANONICAL_ASSIGNMENTS = 24
_FEASIBILITY_ERRORS = (
    FloatingPointError,
    OverflowError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _signature(component: LatentComponentParameters) -> tuple[str, bool]:
    return component.shape, component.log_D is not None


def component_slot_equivalence_classes(
    codec: ProfiledBranchCodec,
    *,
    component_intensity_bounds: Sequence[ClosedInterval] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Return the only slot classes authorized for physical relabeling.

    ``None`` means the amplitude half of the GUI contract is unavailable.  It
    does *not* mean that intensity bounds are unconstrained or interchangeable.
    """

    if not isinstance(codec, ProfiledBranchCodec):
        raise TypeError("codec must be ProfiledBranchCodec")
    query_classes = component_slot_contract_equivalence_classes(
        codec.component_bounds,
        component_intensity_bounds=component_intensity_bounds,
    )
    classes: list[tuple[int, ...]] = []
    for query_class in query_classes:
        remaining = set(query_class)
        for slot in query_class:
            if slot not in remaining:
                continue
            equivalent = tuple(
                candidate
                for candidate in query_class
                if candidate in remaining
                and codec.d_present[candidate] == codec.d_present[slot]
            )
            remaining.difference_update(equivalent)
            classes.append(equivalent)
    return tuple(classes)


def _class_local_orders(
    component_count: int,
    equivalence_classes: tuple[tuple[int, ...], ...],
):
    """Yield source-index orders without crossing a labelled class boundary."""

    class_permutations = tuple(permutations(slots) for slots in equivalence_classes)
    for selected in product(*class_permutations):
        order = list(range(component_count))
        for destination_slots, source_slots in zip(equivalence_classes, selected):
            for destination, source in zip(destination_slots, source_slots):
                order[destination] = source
        yield tuple(order)


def component_physical_dictionary_key(
    component: LatentComponentParameters,
    *,
    numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION,
) -> tuple[object, ...]:
    """Return the stable GUI-physical field ordering used by the quotient."""

    if not isinstance(component, LatentComponentParameters):
        raise TypeError("component must be LatentComponentParameters")
    gui = latent_component_to_gui(
        component,
        numeric_policy_version=numeric_policy_version,
    )
    return (
        gui.shape,
        float(gui.R),
        float(gui.sigma_R),
        gui.h is not None,
        0.0 if gui.h is None else float(gui.h),
        0.0 if gui.sigma_h is None else float(gui.sigma_h),
        gui.D is not None,
        0.0 if gui.D is None else float(gui.D),
        0.0 if gui.sigma_D is None else float(gui.sigma_D),
    )


def _component_numeric_values(component: LatentComponentParameters) -> np.ndarray:
    return np.asarray(
        [
            component.log_R,
            component.sigma_R_fraction,
            0.0 if component.log_h is None else component.log_h,
            0.0 if component.sigma_h_fraction is None else component.sigma_h_fraction,
            0.0 if component.log_D is None else component.log_D,
            0.0 if component.sigma_D_fraction is None else component.sigma_D_fraction,
        ],
        dtype=np.float64,
    )


def _physical_roundtrip_equal(
    expected: Sequence[LatentComponentParameters],
    actual: Sequence[LatentComponentParameters],
) -> bool:
    if len(expected) != len(actual):
        return False
    for left, right in zip(expected, actual):
        if _signature(left) != _signature(right) or not np.allclose(
            _component_numeric_values(left),
            _component_numeric_values(right),
            rtol=0.0,
            atol=CANONICAL_COMPONENT_ROUNDTRIP_ATOL,
        ):
            return False
    return True


def _resolution_roundtrip_equal(
    expected: ResolutionShape | None,
    actual: ResolutionShape | None,
) -> bool:
    if expected is None or actual is None:
        return expected is actual
    return bool(
        np.allclose(
            (expected.sigma_res, expected.nu_res),
            (actual.sigma_res, actual.nu_res),
            rtol=0.0,
            atol=CANONICAL_COMPONENT_ROUNDTRIP_ATOL,
        )
    )


@dataclass(frozen=True, kw_only=True)
class CanonicalComponentSlotAssignment:
    components: tuple[LatentComponentParameters, ...]
    resolution: ResolutionShape | None
    coordinates: BranchCoordinates
    physical_dictionary_key: tuple[tuple[object, ...], ...]
    slot_equivalence_classes: tuple[tuple[int, ...], ...]
    exchangeability_context_complete: bool
    numeric_policy_version: str = V5_FAST_NUMERIC_POLICY_VERSION
    version: str = CANONICAL_COMPONENT_SLOTS_VERSION

    def __post_init__(self) -> None:
        if self.version != CANONICAL_COMPONENT_SLOTS_VERSION:
            raise ValueError("unsupported component-slot canonicalization version")
        if not self.components or len(self.components) > 4:
            raise ValueError("canonical assignment requires one to four components")
        if not all(isinstance(item, LatentComponentParameters) for item in self.components):
            raise TypeError("components must contain LatentComponentParameters")
        if self.resolution is not None and not isinstance(self.resolution, ResolutionShape):
            raise TypeError("resolution must be ResolutionShape or None")
        if not isinstance(self.coordinates, BranchCoordinates):
            raise TypeError("coordinates must be BranchCoordinates")
        expected_key = tuple(
            component_physical_dictionary_key(
                item,
                numeric_policy_version=self.numeric_policy_version,
            )
            for item in self.components
        )
        if self.physical_dictionary_key != expected_key:
            raise ValueError("physical_dictionary_key does not match components")
        flattened = tuple(slot for values in self.slot_equivalence_classes for slot in values)
        if sorted(flattened) != list(range(len(self.components))) or len(flattened) != len(
            self.components
        ):
            raise ValueError("slot_equivalence_classes must partition every component slot")
        if type(self.exchangeability_context_complete) is not bool:
            raise TypeError("exchangeability_context_complete must be a bool")
        if not self.exchangeability_context_complete and any(
            len(values) != 1 for values in self.slot_equivalence_classes
        ):
            raise ValueError("incomplete amplitude context cannot authorize slot exchange")


def canonicalize_component_slots(
    codec: ProfiledBranchCodec,
    components: Sequence[LatentComponentParameters],
    resolution: ResolutionShape | None,
    *,
    component_intensity_bounds: Sequence[ClosedInterval] | None = None,
) -> CanonicalComponentSlotAssignment:
    """Choose the stable representative inside proven slot-equivalence classes.

    The function never clips or changes a physical component.  An assignment
    is considered feasible only if codec encode -> decode -> encode preserves
    both its physical parameters and local coordinates within the versioned
    round-trip tolerance.  Components are never moved across non-equivalent
    labelled slots, even when their numerical geometry ranges overlap.
    """

    if not isinstance(codec, ProfiledBranchCodec):
        raise TypeError("codec must be ProfiledBranchCodec")
    try:
        values = tuple(components)
    except TypeError as exc:
        raise TypeError("components must be a sequence") from exc
    if len(values) != len(codec.topology) or not all(
        isinstance(item, LatentComponentParameters) for item in values
    ):
        raise ValueError("components must match the codec component count")
    expected_signatures = sorted(zip(codec.topology, codec.d_present))
    actual_signatures = sorted(_signature(item) for item in values)
    if actual_signatures != expected_signatures:
        raise ValueError("component shape/D multiset does not match the user codec")
    equivalence_classes = component_slot_equivalence_classes(
        codec,
        component_intensity_bounds=component_intensity_bounds,
    )

    feasible: list[
        tuple[
            tuple[tuple[object, ...], ...],
            tuple[tuple[object, ...], ...],
            tuple[LatentComponentParameters, ...],
            BranchCoordinates,
        ]
    ] = []
    assignment_count = 0
    for order in _class_local_orders(len(values), equivalence_classes):
        assigned = tuple(values[index] for index in order)
        if any(
            _signature(component) != (codec.topology[slot], codec.d_present[slot])
            for slot, component in enumerate(assigned)
        ):
            continue
        assignment_count += 1
        if assignment_count > MAX_CANONICAL_ASSIGNMENTS:  # pragma: no cover
            raise RuntimeError("component-slot enumeration exceeded 4!")
        try:
            encoded = codec.encode(assigned, resolution)
            decoded, decoded_resolution = codec.decode(encoded)
            reencoded = codec.encode(decoded, decoded_resolution)
        except _FEASIBILITY_ERRORS:
            continue
        if not _physical_roundtrip_equal(assigned, decoded):
            continue
        if not _resolution_roundtrip_equal(resolution, decoded_resolution):
            continue
        if not np.allclose(
            encoded.unit_cube,
            reencoded.unit_cube,
            rtol=0.0,
            atol=CANONICAL_COMPONENT_ROUNDTRIP_ATOL,
        ):
            continue
        selection_key = tuple(
            component_physical_dictionary_key(
                item,
                numeric_policy_version=codec.numeric_policy_version,
            )
            for item in assigned
        )
        canonical_key = tuple(
            component_physical_dictionary_key(
                item,
                numeric_policy_version=codec.numeric_policy_version,
            )
            for item in decoded
        )
        # Persist the exact decode representative of the stored coordinates.
        # The original assignment remains the lexicographic selection key, but
        # returning it could differ from a replayed decode by one binary64 ULP.
        feasible.append((selection_key, canonical_key, decoded, encoded))
    if not feasible:
        raise ValueError("no shape/D-compatible component assignment is feasible in the user codec")
    _, key, assigned, coordinates = min(feasible, key=lambda item: item[0])
    return CanonicalComponentSlotAssignment(
        components=assigned,
        resolution=resolution,
        coordinates=coordinates,
        physical_dictionary_key=key,
        slot_equivalence_classes=equivalence_classes,
        exchangeability_context_complete=component_intensity_bounds is not None,
        numeric_policy_version=codec.numeric_policy_version,
    )


__all__ = [
    "CANONICAL_COMPONENT_ROUNDTRIP_ATOL",
    "CANONICAL_COMPONENT_SLOTS_VERSION",
    "MAX_CANONICAL_ASSIGNMENTS",
    "CanonicalComponentSlotAssignment",
    "canonicalize_component_slots",
    "component_slot_equivalence_classes",
    "component_physical_dictionary_key",
]
