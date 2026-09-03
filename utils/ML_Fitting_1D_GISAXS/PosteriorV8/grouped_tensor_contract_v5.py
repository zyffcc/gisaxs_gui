"""Exact dtype and trailing-shape checks for V5.1 grouped tensor tables."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .bounds_first_contract import BOUNDS_EMBEDDING_DIM
from .branch_codec import UNIT_CUBE_DIMENSIONS
from .candidate_supervision_v5 import CANDIDATE_SUPERVISION_TENSOR_KEYS
from .model_v5_contract import (
    CURVE_GLOBAL_FEATURE_DIM,
    CURVE_POINT_FEATURE_DIM,
    MODEL_V5_INPUT_KEYS,
)
from .uncertainty_provenance_v5 import UNCERTAINTY_FEATURE_DIM


_STRING_LABELS = {
    name for name in CANDIDATE_SUPERVISION_TENSOR_KEYS if "_id" in name or "sha256" in name
} | {
    "search_evaluator_version",
    "search_metric_name",
    "search_threshold_name",
    "search_termination_reason",
}
_BOOL_LABELS = {
    "exact_bounds_passed",
    "exact_physics_passed",
    "search_completed",
    "has_local_target",
}
_INT_LABELS = {
    "search_outcome_code",
    "search_exact_forward_call_budget",
    "search_exact_forward_calls_used",
    "search_compatible_representative_count",
    "clean_recipe_index",
}


def _dtype(array: np.ndarray, expected: np.dtype | type, name: str) -> None:
    if array.dtype != np.dtype(expected):
        raise ValueError(f"{name} must use exact dtype {np.dtype(expected)}")


def _unicode(array: np.ndarray, name: str) -> None:
    if array.dtype.kind != "U":
        raise ValueError(f"{name} must use a non-pickle Unicode dtype")


def _shape(array: np.ndarray, expected: tuple[int | None, ...], name: str) -> None:
    if array.ndim != len(expected) or any(
        value is not None and array.shape[index] != value for index, value in enumerate(expected)
    ):
        raise ValueError(f"{name} has incompatible shape {array.shape}; expected {expected}")


def validate_grouped_tensor_contract(
    arrays: Mapping[str, np.ndarray],
    *,
    recipe_count: int,
    observation_count: int,
    candidate_count: int,
) -> None:
    """Validate every persisted training tensor without importing TensorFlow."""

    for name in (
        "clean__recipe_id",
        "clean__clean_group_id",
        "clean__recipe_sha256",
        "clean__recipe_canonical_json",
        "clean__recipe_schema_version",
        "clean__recipe_generator_version",
        "clean__geometry_query_sha256",
        "clean__amplitude_query_sha256",
        "clean__amplitude_query_canonical_json",
        "clean__split_id",
        "clean__split_plan_sha256",
        "clean__sobol_design_sha256",
    ):
        _unicode(arrays[name], name)
        _shape(arrays[name], (recipe_count,), name)
    for name, dtype in (
        ("clean__recipe_seed", np.uint64),
        ("clean__target_pattern_id", np.int32),
        ("clean__sobol_index", np.int64),
    ):
        _dtype(arrays[name], dtype, name)
        _shape(arrays[name], (recipe_count,), name)
    _dtype(arrays["clean__target_local"], np.float32, "clean__target_local")
    _shape(
        arrays["clean__target_local"],
        (recipe_count, UNIT_CUBE_DIMENSIONS),
        "clean__target_local",
    )

    for name in (
        "observation__observation_id",
        "observation__split_id",
        "observation__acquisition_policy_id",
        "observation__audit_json",
    ):
        _unicode(arrays[name], name)
        _shape(arrays[name], (observation_count,), name)
    for name, dtype in (
        ("observation__recipe_index", np.int32),
        ("observation__view_index", np.uint64),
        ("observation__intensity_reference", np.float64),
    ):
        _dtype(arrays[name], dtype, name)
        _shape(arrays[name], (observation_count,), name)

    for name in (
        "candidate_context__candidate_id",
        "candidate_context__geometry_query_sha256",
        "candidate_context__amplitude_query_sha256",
        "candidate_context__amplitude_constraint_json",
        "candidate_context__amplitude_constraint_sha256",
    ):
        _unicode(arrays[name], name)
        _shape(arrays[name], (candidate_count,), name)
    _dtype(arrays["candidate_context__recipe_index"], np.int32, "candidate recipe index")
    _shape(arrays["candidate_context__recipe_index"], (candidate_count,), "candidate recipe index")

    input_contract = {
        "x": (np.float32, (observation_count, None, CURVE_POINT_FEATURE_DIM)),
        "point_mask": (np.bool_, (observation_count, None)),
        "global_features": (np.float32, (observation_count, CURVE_GLOBAL_FEATURE_DIM)),
        "uncertainty_provenance": (
            np.float32,
            (observation_count, UNCERTAINTY_FEATURE_DIM),
        ),
        "branch_topology_id": (np.int32, (candidate_count, 1)),
        "branch_pattern_id": (np.int32, (candidate_count, 1)),
        "geometry_bounds_embedding": (
            np.float32,
            (candidate_count, BOUNDS_EMBEDDING_DIM),
        ),
        "available_dimension_mask": (
            np.float32,
            (candidate_count, UNIT_CUBE_DIMENSIONS),
        ),
        "active_dimension_mask": (
            np.float32,
            (candidate_count, UNIT_CUBE_DIMENSIONS),
        ),
        "varying_dimension_mask": (
            np.float32,
            (candidate_count, UNIT_CUBE_DIMENSIONS),
        ),
    }
    if set(input_contract) | {"amplitude_bounds_embedding"} != set(MODEL_V5_INPUT_KEYS):
        raise RuntimeError("grouped tensor validator has not been upgraded to current V5 inputs")
    observation_inputs = {
        "x",
        "point_mask",
        "global_features",
        "uncertainty_provenance",
    }
    for name, (dtype, shape) in input_contract.items():
        prefix = "observation" if name in observation_inputs else "candidate_context"
        full_name = f"{prefix}__input__{name}"
        _dtype(arrays[full_name], dtype, full_name)
        _shape(arrays[full_name], shape, full_name)
    x = arrays["observation__input__x"]
    point_mask = arrays["observation__input__point_mask"]
    if x.shape[:2] != point_mask.shape:
        raise ValueError("curve x and point_mask point dimensions disagree")
    intensity_reference = arrays["observation__intensity_reference"]
    if not np.all(np.isfinite(intensity_reference)) or np.any(intensity_reference <= 0.0):
        raise ValueError("observation intensity_reference must be finite and positive")

    for name in CANDIDATE_SUPERVISION_TENSOR_KEYS:
        full_name = f"candidate_label__{name}"
        value = arrays[full_name]
        if name in _STRING_LABELS:
            _unicode(value, full_name)
            expected_shape = (candidate_count,)
        elif name in _BOOL_LABELS:
            _dtype(value, np.bool_, full_name)
            expected_shape = (candidate_count,)
        elif name in _INT_LABELS:
            _dtype(value, np.int32, full_name)
            expected_shape = (candidate_count,)
        else:
            _dtype(value, np.float32, full_name)
            expected_shape = (
                (candidate_count, UNIT_CUBE_DIMENSIONS)
                if name in {"target_local", "active_dimension_mask", "varying_dimension_mask"}
                else (candidate_count,)
            )
        _shape(value, expected_shape, full_name)
    for name in (
        "candidate_label__supervision_audit_json",
        "candidate_label__oracle_exact_artifact_json",
        "candidate_label__oracle_search_artifact_json",
    ):
        _unicode(arrays[name], name)
        _shape(arrays[name], (candidate_count,), name)
    _dtype(
        arrays["candidate_label__generating_candidate_match"],
        np.bool_,
        "generating_candidate_match",
    )
    _shape(
        arrays["candidate_label__generating_candidate_match"],
        (candidate_count,),
        "generating_candidate_match",
    )


__all__ = ["validate_grouped_tensor_contract"]
