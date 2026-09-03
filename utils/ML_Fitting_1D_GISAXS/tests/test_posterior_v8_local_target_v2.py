from __future__ import annotations

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import (
    INACTIVE_UNIT_VALUE,
    ProfiledBranchCodec,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import full_component_bounds
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.local_target import (
    BOUNDS_FIRST_LOCAL_TARGET_SEMANTICS,
    GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS,
    global_to_local_target,
    local_labels_from_bounds_first,
    local_to_global_target,
)


def test_same_physical_point_round_trips_global_box_and_local_coordinates():
    codec = ProfiledBranchCodec.build(
        ("sphere",),
        (full_component_bounds("sphere", d_policy="absent"),),
        (False,),
    )
    target_global = np.full(26, INACTIVE_UNIT_VALUE, dtype=np.float64)
    target_global[np.asarray(codec.active_mask)] = (0.3, 0.7)
    original_components, original_resolution = codec.decode(target_global)
    assert original_resolution is None

    low = np.full(26, INACTIVE_UNIT_VALUE, dtype=np.float64)
    high = np.full(26, INACTIVE_UNIT_VALUE, dtype=np.float64)
    active_indices = np.flatnonzero(codec.active_mask)
    low[active_indices[0]], high[active_indices[0]] = 0.1, 0.9
    # A fixed active dimension has the unique local representation 0.5.
    low[active_indices[1]] = high[active_indices[1]] = target_global[active_indices[1]]
    local = global_to_local_target(
        target_global,
        low,
        high,
        codec.active_mask,
        branch_box_coordinate_semantics=GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS,
    )
    assert local.target_local[active_indices[0]] == pytest.approx(0.25)
    assert local.target_local[active_indices[1]] == INACTIVE_UNIT_VALUE
    assert local.varying_dimension_mask[active_indices[0]]
    assert not local.varying_dimension_mask[active_indices[1]]

    restored_global = local_to_global_target(
        local.target_local,
        low,
        high,
        codec.active_mask,
        branch_box_coordinate_semantics=GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS,
    )
    restored_components, restored_resolution = codec.decode(restored_global)
    assert restored_resolution is None
    assert np.allclose(restored_global, target_global, rtol=0.0, atol=5e-8)
    assert restored_components[0].log_R == pytest.approx(original_components[0].log_R)
    assert restored_components[0].sigma_R_fraction == pytest.approx(
        original_components[0].sigma_R_fraction
    )


def test_global_box_transform_refuses_gui_physical_bound_semantics():
    target = np.full(26, 0.5, dtype=np.float32)
    with pytest.raises(ValueError, match="GUI physical bounds"):
        global_to_local_target(
            target,
            target,
            target,
            np.zeros(26, dtype=np.bool_),
            branch_box_coordinate_semantics="gui_physical_bounds",
        )


def test_bounds_first_native_local_labels_use_a_separate_direct_adapter():
    active = np.zeros((2, 26), dtype=np.bool_)
    varying = np.zeros((2, 26), dtype=np.bool_)
    active[:, :2] = True
    varying[:, 0] = True
    target = np.full((2, 26), INACTIVE_UNIT_VALUE, dtype=np.float32)
    target[:, 0] = (0.1, 0.9)
    adapted = local_labels_from_bounds_first(
        {
            "topology_id": np.asarray([0, 0], dtype=np.int32),
            "branch_pattern_id": np.asarray([0, 0], dtype=np.int32),
            "target_local_unit": target,
            "active_dimension_mask": active,
            "local_varying_mask": varying,
        },
        target_coordinate_semantics=BOUNDS_FIRST_LOCAL_TARGET_SEMANTICS,
    )

    assert np.array_equal(adapted["target_local"], target)
    assert np.array_equal(adapted["varying_dimension_mask"], varying.astype(np.float32))
    assert "branch_low" not in adapted and "branch_high" not in adapted
    with pytest.raises(ValueError, match="semantics"):
        local_labels_from_bounds_first(
            {
                "topology_id": np.asarray([0, 0]),
                "branch_pattern_id": np.asarray([0, 0]),
                "target_local_unit": target,
                "active_dimension_mask": active,
                "local_varying_mask": varying,
            },
            target_coordinate_semantics=GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS,
        )
