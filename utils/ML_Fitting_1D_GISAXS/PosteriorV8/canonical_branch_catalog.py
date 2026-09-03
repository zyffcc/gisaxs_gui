"""Permutation-quotiented D/Resolution branches for bounds-first V3.

The legacy five-bit representation assigns one D bit to every component slot.
For repeated instances of the same shape, permuting those bits does not create
a new physical branch.  This module keeps the stable 0..31 wire IDs while
marking exactly one representative per physical permutation class.
"""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np

from .branch_catalog import (
    BRANCH_PATTERN_COUNT,
    branch_pattern_id,
    branch_pattern_is_valid,
    decode_branch_pattern,
)
from .contract import MAX_COMPONENTS, NUM_TOPOLOGIES, TOPOLOGIES


CANONICAL_BRANCH_CATALOG_VERSION = (
    "posterior_v8_permutation_quotiented_d_resolution_branch_v1"
)
CANONICAL_D_ORDER = "d_absent_before_d_present_within_equal_shape_group"


def _topology_id(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("topology_id must be an integer")
    result = int(value)
    if not 0 <= result < NUM_TOPOLOGIES:
        raise ValueError("topology_id must be in [0, 33]")
    return result


def canonicalize_d_flags(
    topology_id: int, d_present: Sequence[bool]
) -> tuple[bool, ...]:
    """Return the unique D-bit ordering for one unordered shape multiset."""

    index = _topology_id(topology_id)
    flags = tuple(d_present)
    if len(flags) != MAX_COMPONENTS or not all(
        isinstance(value, (bool, np.bool_)) for value in flags
    ):
        raise ValueError("d_present must contain exactly four boolean flags")
    component_count = len(TOPOLOGIES[index])
    if any(flags[component_count:]):
        raise ValueError("unused component D flags must be false")
    result = list(bool(value) for value in flags)
    topology = TOPOLOGIES[index]
    start = 0
    while start < component_count:
        stop = start + 1
        while stop < component_count and topology[stop] == topology[start]:
            stop += 1
        present_count = sum(result[start:stop])
        result[start:stop] = [False] * (stop - start - present_count) + [
            True
        ] * present_count
        start = stop
    return tuple(result)


def canonical_branch_pattern_id(topology_id: int, pattern_id: int) -> int:
    """Map any legacy-valid pattern to its physical permutation representative."""

    index = _topology_id(topology_id)
    if not branch_pattern_is_valid(index, pattern_id):
        raise ValueError("pattern_id is invalid for this topology")
    flags, resolution_present = decode_branch_pattern(pattern_id)
    return branch_pattern_id(
        canonicalize_d_flags(index, flags), resolution_present
    )


CANONICAL_VALID_BRANCH_PATTERN_MASK = tuple(
    tuple(
        branch_pattern_is_valid(topology_id, pattern_id)
        and canonical_branch_pattern_id(topology_id, pattern_id) == pattern_id
        for pattern_id in range(BRANCH_PATTERN_COUNT)
    )
    for topology_id in range(NUM_TOPOLOGIES)
)
CANONICAL_PATTERN_IDS_BY_TOPOLOGY = tuple(
    tuple(
        pattern_id
        for pattern_id, valid in enumerate(row)
        if valid
    )
    for row in CANONICAL_VALID_BRANCH_PATTERN_MASK
)
CANONICAL_BRANCH_COUNT = sum(
    sum(row) for row in CANONICAL_VALID_BRANCH_PATTERN_MASK
)

if CANONICAL_BRANCH_COUNT != 418:  # pragma: no cover - contract invariant
    raise RuntimeError(
        "permutation-quotiented Posterior V8 catalog must contain 418 branches"
    )


def canonical_branch_pattern_is_valid(topology_id: int, pattern_id: int) -> bool:
    index = _topology_id(topology_id)
    if isinstance(pattern_id, (bool, np.bool_)) or not isinstance(
        pattern_id, Integral
    ):
        raise TypeError("pattern_id must be an integer")
    pattern = int(pattern_id)
    if not 0 <= pattern < BRANCH_PATTERN_COUNT:
        raise ValueError("pattern_id must be in [0, 31]")
    return CANONICAL_VALID_BRANCH_PATTERN_MASK[index][pattern]


__all__ = [
    "CANONICAL_BRANCH_CATALOG_VERSION",
    "CANONICAL_BRANCH_COUNT",
    "CANONICAL_D_ORDER",
    "CANONICAL_PATTERN_IDS_BY_TOPOLOGY",
    "CANONICAL_VALID_BRANCH_PATTERN_MASK",
    "canonical_branch_pattern_id",
    "canonical_branch_pattern_is_valid",
    "canonicalize_d_flags",
]
