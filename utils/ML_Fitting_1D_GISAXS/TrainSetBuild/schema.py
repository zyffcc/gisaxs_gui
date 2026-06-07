"""Shared schema, physical ranges, and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

TYPE_EMPTY = 0
TYPE_SPHERE = 1
TYPE_CYLINDER = 2
TYPE_VERTICAL_CYLINDER = 3

TYPE_NAMES = {
    TYPE_EMPTY: "empty",
    TYPE_SPHERE: "sphere",
    TYPE_CYLINDER: "cylinder",
    TYPE_VERTICAL_CYLINDER: "vertical_cylinder",
}
NAME_TO_TYPE = {v: k for k, v in TYPE_NAMES.items()}

MAX_SLOTS = 4
NUM_TYPES = 4
MAX_POINTS = 1000
P_MAX = 6
G_MAX = 5

PARAM_NAMES = ["R", "sigma_R", "h", "sigma_h", "D", "sigma_D"]
GLOBAL_PARAM_NAMES = ["BG", "sigma_Res", "nu_Res", "int_Res", "k"]

Q_MIN_GLOBAL = 0.001
Q_MAX_GLOBAL = 10.0
Q_SPACING = "linear"

TYPE_PROBS = {
    TYPE_SPHERE: 1.0 / 3.0,
    TYPE_CYLINDER: 1.0 / 3.0,
    TYPE_VERTICAL_CYLINDER: 1.0 / 3.0,
}


@dataclass(frozen=True)
class RangeSpec:
    low: float
    high: float
    transform: str = "log"


PARAM_RANGES: Dict[str, RangeSpec] = {
    "R": RangeSpec(1.0, 100.0, "log"),
    "sigma_R_frac": RangeSpec(0.02, 0.9, "linear"),
    "h": RangeSpec(2.0, 500.0, "log"),
    "sigma_h_frac": RangeSpec(0.02, 0.9, "linear"),
    "D": RangeSpec(3.0, 500.0, "log"),
    "sigma_D_frac": RangeSpec(0.02, 0.9, "linear"),
    "vertical_sigma_R": RangeSpec(0.02, 0.9, "linear"),
}

PARAM_NORM_RANGES: Dict[str, RangeSpec] = {
    "R": RangeSpec(1.0, 100.0, "log"),
    "sigma_R": RangeSpec(0.02, 90.0, "log"),
    "h": RangeSpec(2.0, 500.0, "log"),
    "sigma_h": RangeSpec(0.04, 400.0, "log"),
    "D": RangeSpec(3.0, 500.0, "log"),
    "sigma_D": RangeSpec(0.06, 400.0, "log"),
}

GLOBAL_NORM_RANGES: Dict[str, RangeSpec] = {
    "BG": RangeSpec(1e-18, 1e8, "log"),
    "sigma_Res": RangeSpec(0.002, 0.3, "log"),
    "nu_Res": RangeSpec(1.0, 10.0, "linear"),
    "int_Res": RangeSpec(1e-18, 1e8, "log"),
    "k": RangeSpec(1e-2, 1e6, "log"),
}


def type_param_mask(type_id: int) -> np.ndarray:
    mask = np.zeros(P_MAX, dtype=np.float32)
    if type_id in (TYPE_SPHERE, TYPE_VERTICAL_CYLINDER):
        mask[[0, 1, 4, 5]] = 1.0
    elif type_id == TYPE_CYLINDER:
        mask[:] = 1.0
    return mask


def effective_param_mask(type_id: int, params_phys: np.ndarray | None = None) -> np.ndarray:
    mask = type_param_mask(type_id)
    if params_phys is not None:
        params_phys = np.asarray(params_phys, dtype=np.float64)
        # D/sigma_D are optional structure-factor parameters. When generation
        # stores them as physical zero, keep them inactive through norm->phys.
        if params_phys.shape[0] >= 6 and (params_phys[4] <= 0.0 or params_phys[5] <= 0.0):
            mask[[4, 5]] = 0.0
    return mask.astype(np.float32)


def apply_param_mask(params_phys: np.ndarray, param_mask: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(param_mask, dtype=np.float32) > 0.5, np.asarray(params_phys), 0.0).astype(np.float32)


def apply_type_param_mask(params_phys: np.ndarray, type_id: int) -> np.ndarray:
    return apply_param_mask(params_phys, type_param_mask(type_id))


def normalize_value(x: float, spec: RangeSpec) -> float:
    if x <= 0 and spec.transform == "log":
        return 0.0
    if spec.transform == "log":
        val = (np.log(max(float(x), spec.low)) - np.log(spec.low)) / (np.log(spec.high) - np.log(spec.low))
    else:
        val = (float(x) - spec.low) / (spec.high - spec.low)
    return float(np.clip(val, 0.0, 1.0))


def denormalize_value(x_norm: float, spec: RangeSpec) -> float:
    x_norm = float(np.clip(x_norm, 0.0, 1.0))
    if spec.transform == "log":
        return float(np.exp(np.log(spec.low) + x_norm * (np.log(spec.high) - np.log(spec.low))))
    return float(spec.low + x_norm * (spec.high - spec.low))


def normalize_params(params_phys: np.ndarray, type_id: int) -> np.ndarray:
    out = np.zeros(P_MAX, dtype=np.float32)
    for i, name in enumerate(PARAM_NAMES):
        if type_param_mask(type_id)[i] > 0:
            out[i] = normalize_value(float(params_phys[i]), PARAM_NORM_RANGES[name])
    return out


def denormalize_params(params_norm: np.ndarray, type_id: int) -> np.ndarray:
    out = np.zeros(P_MAX, dtype=np.float32)
    for i, name in enumerate(PARAM_NAMES):
        if type_param_mask(type_id)[i] > 0:
            out[i] = denormalize_value(float(params_norm[i]), PARAM_NORM_RANGES[name])
    return out


def denormalize_params_with_mask(params_norm: np.ndarray, type_id: int, param_mask: np.ndarray | None = None) -> np.ndarray:
    params_phys = denormalize_params(params_norm, type_id)
    if param_mask is None:
        param_mask = type_param_mask(type_id)
    return apply_param_mask(params_phys, param_mask)


def normalize_global(global_phys: np.ndarray) -> np.ndarray:
    out = np.zeros(G_MAX, dtype=np.float32)
    for i, name in enumerate(GLOBAL_PARAM_NAMES):
        out[i] = normalize_value(float(global_phys[i]), GLOBAL_NORM_RANGES[name])
    return out


def denormalize_global(global_norm: np.ndarray) -> np.ndarray:
    out = np.zeros(G_MAX, dtype=np.float32)
    for i, name in enumerate(GLOBAL_PARAM_NAMES):
        out[i] = denormalize_value(float(global_norm[i]), GLOBAL_NORM_RANGES[name])
    return out


def denormalize_global_with_optional_zero(global_norm: np.ndarray) -> np.ndarray:
    out = denormalize_global(global_norm)
    global_norm = np.asarray(global_norm, dtype=np.float32)
    int_res_idx = GLOBAL_PARAM_NAMES.index("int_Res")
    if global_norm[int_res_idx] <= 0.0:
        out[int_res_idx] = 0.0
    return out


def normalize_logq(q: np.ndarray) -> np.ndarray:
    return ((np.log(q) - np.log(Q_MIN_GLOBAL)) / (np.log(Q_MAX_GLOBAL) - np.log(Q_MIN_GLOBAL))).astype(np.float32)


def metadata_dict() -> dict:
    return {
        "type_ids": TYPE_NAMES,
        "max_slots": MAX_SLOTS,
        "num_types": NUM_TYPES,
        "max_points": MAX_POINTS,
        "component_param_names": PARAM_NAMES,
        "global_param_names": GLOBAL_PARAM_NAMES,
        "q_range": [Q_MIN_GLOBAL, Q_MAX_GLOBAL],
        "q_spacing": Q_SPACING,
        "param_norm_ranges": {k: vars(v) for k, v in PARAM_NORM_RANGES.items()},
        "global_norm_ranges": {k: vars(v) for k, v in GLOBAL_NORM_RANGES.items()},
    }


def global_feature_from_curve(q: np.ndarray, log_i: np.ndarray) -> Tuple[np.ndarray, float, float]:
    q_min_norm = normalize_logq(np.array([np.min(q)], dtype=np.float64))[0]
    q_max_norm = normalize_logq(np.array([np.max(q)], dtype=np.float64))[0]
    offset = float(np.median(log_i))
    iqr = float(np.percentile(log_i, 75) - np.percentile(log_i, 25))
    scale = max(iqr, 1e-6)
    return np.array([q_min_norm, q_max_norm, len(q) / MAX_POINTS, offset, scale], dtype=np.float32), offset, scale
