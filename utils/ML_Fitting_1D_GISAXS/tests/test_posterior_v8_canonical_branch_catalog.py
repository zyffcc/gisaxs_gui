from __future__ import annotations

from itertools import permutations

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import (
    branch_pattern_id,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.canonical_branch_catalog import (
    CANONICAL_BRANCH_COUNT,
    CANONICAL_PATTERN_IDS_BY_TOPOLOGY,
    CANONICAL_VALID_BRANCH_PATTERN_MASK,
    canonical_branch_pattern_id,
    canonical_branch_pattern_is_valid,
    canonicalize_d_flags,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    TOPOLOGIES,
    topology_id_for,
)


def test_catalog_quotients_repeated_shape_d_bit_permutations():
    assert CANONICAL_BRANCH_COUNT == 418
    same_spheres = topology_id_for(("sphere", "sphere"))
    mixed = topology_id_for(("sphere", "cylinder"))
    four_spheres = topology_id_for(("sphere",) * 4)

    assert len(CANONICAL_PATTERN_IDS_BY_TOPOLOGY[same_spheres]) == 6
    assert len(CANONICAL_PATTERN_IDS_BY_TOPOLOGY[mixed]) == 8
    assert len(CANONICAL_PATTERN_IDS_BY_TOPOLOGY[four_spheres]) == 10
    assert sum(sum(row) for row in CANONICAL_VALID_BRANCH_PATTERN_MASK) == 418


def test_every_equal_shape_bit_permutation_has_one_stable_representative():
    topology_id = topology_id_for(("sphere",) * 4)
    representatives = set()
    for positions in permutations((True, True, False, False)):
        pattern = branch_pattern_id(positions, True)
        representatives.add(canonical_branch_pattern_id(topology_id, pattern))
    assert representatives == {branch_pattern_id((False, False, True, True), True)}
    representative = representatives.pop()
    assert canonical_branch_pattern_is_valid(topology_id, representative)


def test_distinct_shape_groups_keep_independent_d_presence_counts():
    topology_id = topology_id_for(("sphere", "sphere", "cylinder", "cylinder"))
    flags = (True, False, True, False)
    assert canonicalize_d_flags(topology_id, flags) == (
        False,
        True,
        False,
        True,
    )
    canonical = branch_pattern_id((False, True, False, True), False)
    assert canonical_branch_pattern_id(
        topology_id, branch_pattern_id(flags, False)
    ) == canonical


def test_canonical_helpers_reject_unused_or_invalid_slots():
    with pytest.raises(ValueError, match="unused"):
        canonicalize_d_flags(0, (False, True, False, False))
    with pytest.raises(ValueError, match="invalid"):
        canonical_branch_pattern_id(0, branch_pattern_id((False, True, False, False), False))
    with pytest.raises(TypeError, match="integer"):
        canonical_branch_pattern_is_valid(True, 0)


def test_catalog_count_matches_shape_multiplicity_formula():
    expected = 0
    for topology in TOPOLOGIES:
        multiplicities = {shape: topology.count(shape) for shape in set(topology)}
        branch_count = 2  # Resolution absent/present.
        for count in multiplicities.values():
            branch_count *= count + 1
        expected += branch_count
    assert expected == CANONICAL_BRANCH_COUNT
