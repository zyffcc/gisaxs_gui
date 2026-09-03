"""TensorFlow-free catalog for Posterior V8 joint discrete branches."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

from .contract import MAX_COMPONENTS, NUM_TOPOLOGIES, TOPOLOGIES


BRANCH_CATALOG_VERSION = "posterior_v8_joint_branch_5bit_v1"
D_PATTERN_BITS = tuple(range(MAX_COMPONENTS))
RESOLUTION_PATTERN_BIT = MAX_COMPONENTS
BRANCH_PATTERN_COUNT = 1 << (MAX_COMPONENTS + 1)


def _pattern_is_valid(topology_id: int, pattern_id: int) -> bool:
    component_count = len(TOPOLOGIES[topology_id])
    unused_d_bits = sum(1 << bit for bit in range(component_count, MAX_COMPONENTS))
    return pattern_id & unused_d_bits == 0


VALID_BRANCH_PATTERN_MASK = tuple(
    tuple(
        _pattern_is_valid(topology_id, pattern_id)
        for pattern_id in range(BRANCH_PATTERN_COUNT)
    )
    for topology_id in range(NUM_TOPOLOGIES)
)
TOPOLOGY_SLOT_MASK = tuple(
    tuple(slot < len(topology) for slot in range(MAX_COMPONENTS))
    for topology in TOPOLOGIES
)

if BRANCH_PATTERN_COUNT != 32:
    raise RuntimeError("Posterior V8 branch pattern must remain a five-bit integer")
if sum(sum(row) for row in VALID_BRANCH_PATTERN_MASK) != 700:
    raise RuntimeError("Posterior V8 branch catalog must contain exactly 700 valid branches")


def branch_pattern_id(
    d_present: Sequence[bool],
    resolution_present: bool,
) -> int:
    """Encode four D flags (low bits) and Resolution (bit 4)."""

    flags = tuple(d_present)
    if len(flags) != MAX_COMPONENTS or not all(isinstance(value, bool) for value in flags):
        raise ValueError("d_present must contain exactly four boolean flags")
    if not isinstance(resolution_present, bool):
        raise ValueError("resolution_present must be boolean")
    result = sum(int(value) << bit for bit, value in enumerate(flags))
    return result | (int(resolution_present) << RESOLUTION_PATTERN_BIT)


def decode_branch_pattern(pattern_id: int) -> tuple[tuple[bool, ...], bool]:
    """Decode the stable five-bit branch-pattern representation."""

    if isinstance(pattern_id, bool) or not isinstance(pattern_id, Integral):
        raise TypeError("pattern_id must be an integer")
    pattern_id = int(pattern_id)
    if not 0 <= pattern_id < BRANCH_PATTERN_COUNT:
        raise ValueError("pattern_id must be in [0, 31]")
    d_present = tuple(bool(pattern_id & (1 << bit)) for bit in D_PATTERN_BITS)
    resolution_present = bool(pattern_id & (1 << RESOLUTION_PATTERN_BIT))
    return d_present, resolution_present


def branch_pattern_is_valid(topology_id: int, pattern_id: int) -> bool:
    """Return whether nonexistent component D bits are all disabled."""

    if isinstance(topology_id, bool) or not isinstance(topology_id, Integral):
        raise TypeError("topology_id must be an integer")
    if not 0 <= int(topology_id) < NUM_TOPOLOGIES:
        raise ValueError("topology_id must be in [0, 33]")
    if isinstance(pattern_id, bool) or not isinstance(pattern_id, Integral):
        raise TypeError("pattern_id must be an integer")
    if not 0 <= int(pattern_id) < BRANCH_PATTERN_COUNT:
        raise ValueError("pattern_id must be in [0, 31]")
    return VALID_BRANCH_PATTERN_MASK[int(topology_id)][int(pattern_id)]


# Explicit verb form for consumers that serialize this codec operation.
encode_branch_pattern = branch_pattern_id


__all__ = [
    "BRANCH_CATALOG_VERSION",
    "BRANCH_PATTERN_COUNT",
    "D_PATTERN_BITS",
    "RESOLUTION_PATTERN_BIT",
    "TOPOLOGY_SLOT_MASK",
    "VALID_BRANCH_PATTERN_MASK",
    "branch_pattern_id",
    "branch_pattern_is_valid",
    "decode_branch_pattern",
    "encode_branch_pattern",
]
