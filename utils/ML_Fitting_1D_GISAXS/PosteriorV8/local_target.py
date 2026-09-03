"""Explicit local-target coordinates for Posterior V8 training.

The Phase-2 pilot stores targets and range boxes in the *global* branch-codec
cube.  A local-target model instead learns the affine coordinate inside one
already validated global-coordinate box.  This module deliberately does not
convert GUI physical bounds into such a box: coupled widths and hard-core
constraints make that a separate scientific operation owned by the branch
codec or a future bounds-first dataset builder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .bounds_first_contract import (
    LOCAL_TARGET_SEMANTICS as BOUNDS_FIRST_LOCAL_TARGET_SEMANTICS,
)
from .branch_codec import INACTIVE_UNIT_VALUE, UNIT_CUBE_DIMENSIONS


LOCAL_TARGET_TRANSFORM_VERSION = "posterior_v8_global_box_to_local_target_v2"
GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS = (
    "axis_aligned_box_in_global_full_domain_branch_codec_unit_cube_v1"
)
LOCAL_TARGET_COORDINATE_SEMANTICS = (
    "local_unit_of_declared_bounds_conditioning_codec_fixed_and_inactive_at_0_5_v2"
)
VARYING_DIMENSION_SEMANTICS = (
    "effective_varying_degrees_of_freedom_subset_of_semantic_active_v2"
)
PHASE2_VARYING_DIMENSION_DERIVATION_SEMANTICS = (
    "semantic_active_and_global_branch_low_strictly_less_than_high_v1"
)
BOUNDS_FIRST_RANGE_CONSTRUCTION_SEMANTICS = (
    "presampled_gui_physical_bounds_then_native_local_codec_target_v1"
)
PHASE2_GLOBAL_BOX_BOUNDS_FIRST_RANGE_CONSTRUCTION_SEMANTICS = (
    "bounds_first_global_branch_box_then_target_sampled_inside_v1"
)
TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS = (
    "target_first_then_containing_global_branch_box_engineering_pilot_v1"
)
SUPPORTED_RANGE_CONSTRUCTION_SEMANTICS = (
    BOUNDS_FIRST_RANGE_CONSTRUCTION_SEMANTICS,
    PHASE2_GLOBAL_BOX_BOUNDS_FIRST_RANGE_CONSTRUCTION_SEMANTICS,
    TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS,
)
SUPPORTED_GLOBAL_BOX_RANGE_CONSTRUCTION_SEMANTICS = (
    PHASE2_GLOBAL_BOX_BOUNDS_FIRST_RANGE_CONSTRUCTION_SEMANTICS,
    TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS,
)

LOCAL_TARGET_CONTRACT = {
    "transform_version": LOCAL_TARGET_TRANSFORM_VERSION,
    "branch_box_coordinates": GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS,
    "model_target_coordinates": LOCAL_TARGET_COORDINATE_SEMANTICS,
    "varying_dimension_mask": VARYING_DIMENSION_SEMANTICS,
    "phase2_varying_dimension_derivation": (
        PHASE2_VARYING_DIMENSION_DERIVATION_SEMANTICS
    ),
    "fixed_coordinate_value": INACTIVE_UNIT_VALUE,
    "inactive_coordinate_value": INACTIVE_UNIT_VALUE,
}


def _numeric_array(value, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if array.ndim < 1 or array.shape[-1] != UNIT_CUBE_DIMENSIONS:
        raise ValueError(
            f"{name} must end in {UNIT_CUBE_DIMENSIONS} branch coordinates"
        )
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    return np.asarray(array, dtype=np.float64)


def _binary_mask(value, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError("active_dimension_mask must match the coordinate tensor shape")
    if array.dtype.kind == "b":
        return array.astype(np.bool_, copy=False)
    if array.dtype.kind not in "iuf" or not np.all(np.isfinite(array)):
        raise ValueError("active_dimension_mask must contain finite binary values")
    if not np.all((array == 0) | (array == 1)):
        raise ValueError("active_dimension_mask must contain only zero or one")
    return array.astype(np.bool_)


def _validate_box_semantics(value: str) -> None:
    if value != GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS:
        raise ValueError(
            "branch_low/high must already be an axis-aligned box in the global "
            "full-domain branch-codec cube; GUI physical bounds require a separate "
            "validated conversion"
        )


def _validated_global_box(target_global, branch_low, branch_high, active_dimension_mask):
    target = _numeric_array(target_global, "target_global")
    low = _numeric_array(branch_low, "branch_low")
    high = _numeric_array(branch_high, "branch_high")
    if low.shape != target.shape or high.shape != target.shape:
        raise ValueError("target_global, branch_low and branch_high must have identical shapes")
    active = _binary_mask(active_dimension_mask, target.shape)
    if np.any(low < 0.0) or np.any(high > 1.0) or np.any(low > high):
        raise ValueError("global branch boxes must satisfy 0 <= low <= high <= 1")
    canonical_inactive = ~active
    if (
        np.any(target[canonical_inactive] != INACTIVE_UNIT_VALUE)
        or np.any(low[canonical_inactive] != INACTIVE_UNIT_VALUE)
        or np.any(high[canonical_inactive] != INACTIVE_UNIT_VALUE)
    ):
        raise ValueError("inactive global target and box coordinates must equal 0.5")
    if np.any(target[active] < low[active]) or np.any(target[active] > high[active]):
        raise ValueError("global target must lie inside its declared branch box")
    fixed = active & (low == high)
    if np.any(target[fixed] != low[fixed]):
        raise ValueError("fixed active coordinates must equal their branch-box endpoint")
    return target, low, high, active


@dataclass(frozen=True, eq=False)
class LocalTargetBatch:
    """One validated batch of local targets and its density mask."""

    target_local: np.ndarray
    varying_dimension_mask: np.ndarray

    def __post_init__(self) -> None:
        target = _numeric_array(self.target_local, "target_local").astype(np.float32)
        varying = _binary_mask(self.varying_dimension_mask, target.shape)
        if np.any(target < 0.0) or np.any(target > 1.0):
            raise ValueError("target_local must lie in [0, 1]")
        if np.any(target[~varying] != INACTIVE_UNIT_VALUE):
            raise ValueError("fixed and inactive local target coordinates must equal 0.5")
        target.setflags(write=False)
        varying = varying.copy()
        varying.setflags(write=False)
        object.__setattr__(self, "target_local", target)
        object.__setattr__(self, "varying_dimension_mask", varying)


def global_to_local_target(
    target_global,
    branch_low,
    branch_high,
    active_dimension_mask,
    *,
    branch_box_coordinate_semantics: str,
) -> LocalTargetBatch:
    """Map a global-codec target into one declared global-coordinate box.

    The affine formula is evaluated only on semantically active, non-fixed
    dimensions.  Fixed and inactive coordinates have the unique local value
    ``0.5`` and are excluded from density training.
    """

    _validate_box_semantics(branch_box_coordinate_semantics)
    target, low, high, active = _validated_global_box(
        target_global, branch_low, branch_high, active_dimension_mask
    )
    varying = active & (low < high)
    local = np.full(target.shape, INACTIVE_UNIT_VALUE, dtype=np.float64)
    local[varying] = (target[varying] - low[varying]) / (high[varying] - low[varying])
    if np.any(local[varying] < 0.0) or np.any(local[varying] > 1.0):
        raise RuntimeError("local target escaped [0, 1]")
    return LocalTargetBatch(local.astype(np.float32), varying)


def local_to_global_target(
    target_local,
    branch_low,
    branch_high,
    active_dimension_mask,
    *,
    branch_box_coordinate_semantics: str,
) -> np.ndarray:
    """Invert :func:`global_to_local_target` for an already validated box."""

    _validate_box_semantics(branch_box_coordinate_semantics)
    local = _numeric_array(target_local, "target_local")
    low = _numeric_array(branch_low, "branch_low")
    high = _numeric_array(branch_high, "branch_high")
    if low.shape != local.shape or high.shape != local.shape:
        raise ValueError("target_local, branch_low and branch_high must have identical shapes")
    active = _binary_mask(active_dimension_mask, local.shape)
    if np.any(low < 0.0) or np.any(high > 1.0) or np.any(low > high):
        raise ValueError("global branch boxes must satisfy 0 <= low <= high <= 1")
    varying = active & (low < high)
    if np.any(local < 0.0) or np.any(local > 1.0):
        raise ValueError("target_local must lie in [0, 1]")
    if np.any(local[~varying] != INACTIVE_UNIT_VALUE):
        raise ValueError("fixed and inactive local target coordinates must equal 0.5")
    if (
        np.any(low[~active] != INACTIVE_UNIT_VALUE)
        or np.any(high[~active] != INACTIVE_UNIT_VALUE)
    ):
        raise ValueError("inactive global branch-box coordinates must equal 0.5")
    global_target = np.full(local.shape, INACTIVE_UNIT_VALUE, dtype=np.float64)
    global_target[active] = low[active]
    global_target[varying] = low[varying] + (high[varying] - low[varying]) * local[varying]
    return global_target.astype(np.float32)


def local_labels_from_global_box(
    inputs: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    *,
    range_construction_semantics: str,
) -> dict[str, np.ndarray]:
    """Adapt a global-box dataset batch to the v2 objective label contract.

    ``range_construction_semantics`` is carried by the caller into its manifest.
    Requiring it here prevents a truth-centered pilot from being silently
    described as bounds-first data.
    """

    if range_construction_semantics not in (
        SUPPORTED_GLOBAL_BOX_RANGE_CONSTRUCTION_SEMANTICS
    ):
        raise ValueError("unsupported or non-global-box range construction semantics")
    required_inputs = {"branch_low", "branch_high", "active_dimension_mask"}
    required_labels = {"topology_id", "branch_pattern_id", "target_unit"}
    if not required_inputs.issubset(inputs) or not required_labels.issubset(labels):
        raise ValueError("global-box batch is missing fields required by local-target v2")
    transformed = global_to_local_target(
        labels["target_unit"],
        inputs["branch_low"],
        inputs["branch_high"],
        inputs["active_dimension_mask"],
        branch_box_coordinate_semantics=GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS,
    )
    return {
        "topology_id": np.asarray(labels["topology_id"], dtype=np.int32),
        "branch_pattern_id": np.asarray(labels["branch_pattern_id"], dtype=np.int32),
        "target_local": transformed.target_local,
        "active_dimension_mask": np.asarray(
            inputs["active_dimension_mask"], dtype=np.float32
        ),
        "varying_dimension_mask": transformed.varying_dimension_mask.astype(np.float32),
        "branch_low": np.asarray(inputs["branch_low"], dtype=np.float32),
        "branch_high": np.asarray(inputs["branch_high"], dtype=np.float32),
    }


def local_labels_from_bounds_first(
    labels: Mapping[str, np.ndarray],
    *,
    target_coordinate_semantics: str,
) -> dict[str, np.ndarray]:
    """Adapt native bounds-first labels without inventing a global 26-D box.

    Bounds-first data is encoded by the ``ProfiledBranchCodec`` built from the
    presampled GUI physical bounds.  Its local target and codec-effective
    varying mask are therefore already authoritative and must not pass through
    :func:`global_to_local_target`.
    """

    if target_coordinate_semantics != BOUNDS_FIRST_LOCAL_TARGET_SEMANTICS:
        raise ValueError("bounds-first local target semantics are missing or unsupported")
    required = {
        "topology_id",
        "branch_pattern_id",
        "target_local_unit",
        "active_dimension_mask",
        "local_varying_mask",
    }
    if not required.issubset(labels):
        raise ValueError("bounds-first batch is missing local objective fields")
    target = _numeric_array(labels["target_local_unit"], "target_local_unit")
    active = _binary_mask(labels["active_dimension_mask"], target.shape)
    varying = _binary_mask(labels["local_varying_mask"], target.shape)
    if np.any(varying & ~active):
        raise ValueError("bounds-first varying dimensions must be semantically active")
    if np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("bounds-first local targets must lie in [0, 1]")
    if np.any(target[~varying] != INACTIVE_UNIT_VALUE):
        raise ValueError("fixed and inactive bounds-first targets must equal 0.5")
    return {
        "topology_id": np.asarray(labels["topology_id"], dtype=np.int32),
        "branch_pattern_id": np.asarray(labels["branch_pattern_id"], dtype=np.int32),
        "target_local": target.astype(np.float32),
        "active_dimension_mask": active.astype(np.float32),
        "varying_dimension_mask": varying.astype(np.float32),
    }


__all__ = [
    "BOUNDS_FIRST_RANGE_CONSTRUCTION_SEMANTICS",
    "BOUNDS_FIRST_LOCAL_TARGET_SEMANTICS",
    "GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS",
    "LOCAL_TARGET_CONTRACT",
    "LOCAL_TARGET_COORDINATE_SEMANTICS",
    "LOCAL_TARGET_TRANSFORM_VERSION",
    "PHASE2_GLOBAL_BOX_BOUNDS_FIRST_RANGE_CONSTRUCTION_SEMANTICS",
    "PHASE2_VARYING_DIMENSION_DERIVATION_SEMANTICS",
    "LocalTargetBatch",
    "SUPPORTED_RANGE_CONSTRUCTION_SEMANTICS",
    "SUPPORTED_GLOBAL_BOX_RANGE_CONSTRUCTION_SEMANTICS",
    "TRUTH_CENTERED_RANGE_CONSTRUCTION_SEMANTICS",
    "VARYING_DIMENSION_SEMANTICS",
    "global_to_local_target",
    "local_labels_from_bounds_first",
    "local_labels_from_global_box",
    "local_to_global_target",
]
