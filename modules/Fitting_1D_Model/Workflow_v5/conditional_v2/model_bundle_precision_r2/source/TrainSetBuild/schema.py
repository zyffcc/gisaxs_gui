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

D_RULE_FREE = 0
D_RULE_MAX = 1
D_RULE_MEAN = 2
D_RULE_COMPONENT = 3
D_RULE_NAMES = {
    D_RULE_FREE: "free",
    D_RULE_MAX: "max_diameter",
    D_RULE_MEAN: "mean_diameter",
    D_RULE_COMPONENT: "component_diameter",
}
NAME_TO_D_RULE = {v: k for k, v in D_RULE_NAMES.items()}
NUM_D_RULES = len(D_RULE_NAMES)

PARAM_NAMES = ["R", "sigma_R", "h", "sigma_h", "D", "sigma_D"]
GLOBAL_PARAM_NAMES = ["BG", "sigma_Res", "nu_Res", "int_Res", "k"]

Q_MIN_GLOBAL = 0.001
Q_MAX_GLOBAL = 10.0
V5_Q_MIN_GLOBAL = 0.00009
V5_Q_MAX_GLOBAL = 6.0
Q_SPACING = "linear"

SCHEMA_VERSION_LEGACY_V3 = 3
SCHEMA_VERSION_UNIVERSAL_V4 = 4
SCHEMA_VERSION_UNIVERSAL_V5 = 5
DATASET_PROFILE_LEGACY_V3 = "legacy_v3"
DATASET_PROFILE_UNIVERSAL_V4 = "universal_v4"
DATASET_PROFILE_UNIVERSAL_V5 = "universal_v5"
FORWARD_VERSION_LEGACY = "legacy_v3_amplitude"
# The vertical-cylinder expression is intentionally retained until its absolute
# GISAXS normalization can be validated against a reference implementation.
FORWARD_VERSION_UNIVERSAL_V4 = "universal_v4_legacy_vertical_amplitude"
FORWARD_VERSION_UNIVERSAL_V5 = "universal_v5_normalized_components_k1_experimental_q_resolution_v2"

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
    # Finite-width structure peaks: narrower peaks are not supported by the
    # intended experimental resolution and are easy for optimizers to abuse.
    "sigma_D_frac": RangeSpec(0.05, 0.9, "linear"),
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

# universal_v4 stores all component distribution widths as dimensionless
# fractions.  The adapter converts them to the legacy NumPy forward units at
# one explicit boundary (vertical-cylinder sigma_R was already fractional).
V4_PARAM_NORM_RANGES: Dict[str, RangeSpec] = {
    "R": PARAM_RANGES["R"],
    "sigma_R": PARAM_RANGES["sigma_R_frac"],
    "h": PARAM_RANGES["h"],
    "sigma_h": PARAM_RANGES["sigma_h_frac"],
    "D": PARAM_RANGES["D"],
    "sigma_D": PARAM_RANGES["sigma_D_frac"],
}
V5_PARAM_NORM_RANGES = V4_PARAM_NORM_RANGES

V5_GLOBAL_TARGET_NAMES = ["rho_BG", "sigma_Res", "nu_Res", "rho_Res", "unused_k"]
V5_GLOBAL_NORM_RANGES: Dict[str, RangeSpec] = {
    "rho_BG": RangeSpec(1e-6, 1e-2, "log"),
    "sigma_Res": RangeSpec(0.007, 0.013, "log"),
    "nu_Res": RangeSpec(5.0, 10.0, "linear"),
    "rho_Res": RangeSpec(10.0, 1000.0, "log"),
    "unused_k": RangeSpec(0.0, 1.0, "linear"),
}

# V5.1 keeps the V5 forward equation unchanged, but deliberately broadens the
# nuisance-parameter proposal used by the 20K ablation.  Keeping this as a
# separate normalization version is essential: a normalized value from the
# production V5 range must never silently be decoded with this wider range.
V51_GLOBAL_NORM_RANGES: Dict[str, RangeSpec] = {
    "rho_BG": RangeSpec(1e-6, 1e-2, "log"),
    "sigma_Res": RangeSpec(0.004, 0.04, "log"),
    "nu_Res": RangeSpec(3.0, 10.0, "linear"),
    "rho_Res": RangeSpec(1.0, 1e4, "log"),
    "unused_k": RangeSpec(0.0, 1.0, "linear"),
}
V5_GLOBAL_NORM_VERSION = "v5_qres3"
V51_GLOBAL_NORM_VERSION = "v5_1_extended_resolution"

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


def param_norm_ranges(dataset_profile: str = DATASET_PROFILE_LEGACY_V3) -> Dict[str, RangeSpec]:
    if dataset_profile == DATASET_PROFILE_LEGACY_V3:
        return PARAM_NORM_RANGES
    if dataset_profile == DATASET_PROFILE_UNIVERSAL_V4:
        return V4_PARAM_NORM_RANGES
    if dataset_profile == DATASET_PROFILE_UNIVERSAL_V5:
        return V5_PARAM_NORM_RANGES
    raise ValueError(f"Unknown dataset_profile: {dataset_profile!r}")


def normalize_params_for_profile(params_phys: np.ndarray, type_id: int, dataset_profile: str) -> np.ndarray:
    ranges = param_norm_ranges(dataset_profile)
    out = np.zeros(P_MAX, dtype=np.float32)
    mask = effective_param_mask(type_id, params_phys)
    for i, name in enumerate(PARAM_NAMES):
        if mask[i] > 0:
            value = float(params_phys[i])
            spec = ranges[name]
            if not spec.low <= value <= spec.high:
                raise ValueError(f"{dataset_profile} {TYPE_NAMES[type_id]} {name}={value} outside [{spec.low}, {spec.high}]")
            out[i] = normalize_value(value, spec)
    return out


def denormalize_params_for_profile(
    params_norm: np.ndarray,
    type_id: int,
    dataset_profile: str,
    param_mask: np.ndarray | None = None,
) -> np.ndarray:
    ranges = param_norm_ranges(dataset_profile)
    out = np.zeros(P_MAX, dtype=np.float32)
    mask = type_param_mask(type_id) if param_mask is None else np.asarray(param_mask, dtype=np.float32)
    for i, name in enumerate(PARAM_NAMES):
        if mask[i] > 0:
            out[i] = denormalize_value(float(params_norm[i]), ranges[name])
    return apply_param_mask(out, mask)


def params_to_legacy_forward_units(params_phys: np.ndarray, type_id: int, dataset_profile: str) -> np.ndarray:
    """Convert stored profile units to the units expected by utils.fitting."""
    out = np.asarray(params_phys, dtype=np.float64).copy()
    if dataset_profile == DATASET_PROFILE_LEGACY_V3:
        return out
    if dataset_profile not in (DATASET_PROFILE_UNIVERSAL_V4, DATASET_PROFILE_UNIVERSAL_V5):
        raise ValueError(f"Unknown dataset_profile: {dataset_profile!r}")
    if type_id in (TYPE_SPHERE, TYPE_CYLINDER):
        out[1] *= out[0]
    # vertical cylinder already expects sigma_R/R in utils.fitting.
    if type_id == TYPE_CYLINDER:
        out[3] *= out[2]
    if out[4] > 0.0:
        out[5] *= out[4]
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


def v5_global_norm_ranges(version: str = V5_GLOBAL_NORM_VERSION) -> Dict[str, RangeSpec]:
    if version == V5_GLOBAL_NORM_VERSION:
        return V5_GLOBAL_NORM_RANGES
    if version == V51_GLOBAL_NORM_VERSION:
        return V51_GLOBAL_NORM_RANGES
    raise ValueError(f"Unknown V5 global normalization version: {version!r}")


def normalize_global_v5(
    rho_bg: float,
    sigma_res: float,
    nu_res: float,
    rho_res: float,
    version: str = V5_GLOBAL_NORM_VERSION,
) -> np.ndarray:
    values = [rho_bg, sigma_res, nu_res, rho_res, 0.0]
    out = np.zeros(G_MAX, dtype=np.float32)
    ranges = v5_global_norm_ranges(version)
    for i, name in enumerate(V5_GLOBAL_TARGET_NAMES):
        if name == "rho_Res" and rho_res <= 0.0:
            out[i] = 0.0
        else:
            out[i] = normalize_value(float(values[i]), ranges[name])
    return out


def denormalize_global_v5(
    global_norm: np.ndarray,
    version: str = V5_GLOBAL_NORM_VERSION,
    resolution_present: bool = True,
) -> np.ndarray:
    """Decode V5 nuisance labels as [rho_BG, sigma, nu, rho_Res, unused]."""
    ranges = v5_global_norm_ranges(version)
    out = np.zeros(G_MAX, dtype=np.float32)
    for i, name in enumerate(V5_GLOBAL_TARGET_NAMES):
        out[i] = denormalize_value(float(global_norm[i]), ranges[name])
    if not resolution_present:
        out[3] = 0.0
    out[4] = 0.0
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


def normalize_logq_for_profile(q: np.ndarray, dataset_profile: str) -> np.ndarray:
    if dataset_profile == DATASET_PROFILE_UNIVERSAL_V5:
        return ((np.log(q) - np.log(V5_Q_MIN_GLOBAL)) / (np.log(V5_Q_MAX_GLOBAL) - np.log(V5_Q_MIN_GLOBAL))).astype(np.float32)
    return normalize_logq(q)


def metadata_dict(dataset_profile: str = DATASET_PROFILE_LEGACY_V3) -> dict:
    versions = {
        DATASET_PROFILE_LEGACY_V3: (SCHEMA_VERSION_LEGACY_V3, FORWARD_VERSION_LEGACY),
        DATASET_PROFILE_UNIVERSAL_V4: (SCHEMA_VERSION_UNIVERSAL_V4, FORWARD_VERSION_UNIVERSAL_V4),
        DATASET_PROFILE_UNIVERSAL_V5: (SCHEMA_VERSION_UNIVERSAL_V5, FORWARD_VERSION_UNIVERSAL_V5),
    }
    if dataset_profile not in versions:
        raise ValueError(f"Unknown dataset_profile: {dataset_profile!r}")
    version, forward = versions[dataset_profile]
    return {
        "schema_version": version,
        "dataset_profile": dataset_profile,
        "forward_version": forward,
        "type_ids": TYPE_NAMES,
        "max_slots": MAX_SLOTS,
        "num_types": NUM_TYPES,
        "max_points": MAX_POINTS,
        "component_param_names": PARAM_NAMES,
        "global_param_names": GLOBAL_PARAM_NAMES,
        "d_spacing_rule_ids": D_RULE_NAMES,
        "q_range": [V5_Q_MIN_GLOBAL, V5_Q_MAX_GLOBAL] if dataset_profile == DATASET_PROFILE_UNIVERSAL_V5 else [Q_MIN_GLOBAL, Q_MAX_GLOBAL],
        "q_spacing": Q_SPACING,
        "param_norm_ranges": {k: vars(v) for k, v in param_norm_ranges(dataset_profile).items()},
        "global_norm_ranges": {k: vars(v) for k, v in (V5_GLOBAL_NORM_RANGES if dataset_profile == DATASET_PROFILE_UNIVERSAL_V5 else GLOBAL_NORM_RANGES).items()},
    }


def global_feature_from_curve(q: np.ndarray, log_i: np.ndarray) -> Tuple[np.ndarray, float, float]:
    q_min_norm = normalize_logq(np.array([np.min(q)], dtype=np.float64))[0]
    q_max_norm = normalize_logq(np.array([np.max(q)], dtype=np.float64))[0]
    offset = float(np.median(log_i))
    iqr = float(np.percentile(log_i, 75) - np.percentile(log_i, 25))
    scale = max(iqr, 1e-6)
    return np.array([q_min_norm, q_max_norm, len(q) / MAX_POINTS, offset, scale], dtype=np.float32), offset, scale
