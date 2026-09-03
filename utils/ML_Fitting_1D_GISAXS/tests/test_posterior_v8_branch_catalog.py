from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import (
    BRANCH_PATTERN_COUNT,
    VALID_BRANCH_PATTERN_MASK,
    branch_pattern_id,
    branch_pattern_is_valid,
    decode_branch_pattern,
    encode_branch_pattern,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import TOPOLOGIES


def test_catalog_import_does_not_load_tensorflow():
    root = Path(__file__).resolve().parents[3]
    code = """
import sys
assert 'tensorflow' not in sys.modules
import utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog as catalog
assert catalog.BRANCH_PATTERN_COUNT == 32
assert 'tensorflow' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_catalog_contains_exactly_700_valid_topology_patterns():
    assert BRANCH_PATTERN_COUNT == 32
    assert len(VALID_BRANCH_PATTERN_MASK) == 34
    assert all(len(row) == 32 for row in VALID_BRANCH_PATTERN_MASK)
    assert sum(sum(row) for row in VALID_BRANCH_PATTERN_MASK) == 700
    for topology_id, topology in enumerate(TOPOLOGIES):
        assert sum(VALID_BRANCH_PATTERN_MASK[topology_id]) == 2 ** (len(topology) + 1)


def test_five_bit_encode_decode_and_topology_validity_contract():
    flags = (True, False, True, False)
    pattern = encode_branch_pattern(flags, True)
    assert pattern == branch_pattern_id(flags, True) == 1 + 4 + 16
    assert decode_branch_pattern(pattern) == (flags, True)

    valid_k1 = branch_pattern_id((True, False, False, False), True)
    assert branch_pattern_is_valid(0, valid_k1)
    assert not branch_pattern_is_valid(0, pattern)
    assert branch_pattern_is_valid(33, pattern)


def test_catalog_helpers_reject_ambiguous_or_out_of_range_values():
    with pytest.raises(ValueError, match="four boolean"):
        branch_pattern_id((True, False), False)
    with pytest.raises(ValueError, match="boolean"):
        branch_pattern_id((True, False, False, False), 1)
    with pytest.raises(TypeError, match="integer"):
        decode_branch_pattern(True)
    with pytest.raises(ValueError, match=r"\[0, 31\]"):
        decode_branch_pattern(32)
    with pytest.raises(ValueError, match=r"\[0, 33\]"):
        branch_pattern_is_valid(34, 0)
