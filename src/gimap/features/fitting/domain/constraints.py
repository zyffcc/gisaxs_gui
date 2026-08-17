"""Manual fitting 参数选择与 bounds 规则。"""

from __future__ import annotations

import re

import numpy as np


def parameter_base_name(name: str) -> str:
    return re.sub(r"\d+$", "", str(name))


def default_refine_selected(name: str) -> bool:
    return parameter_base_name(name) in {"Int", "BG", "int_Res", "k"}


def default_refine_bounds(name: str, value: float) -> tuple[float, float]:
    base = parameter_base_name(name)
    value = float(value)
    if base == "BG":
        return 0.0, max(abs(value) * 10.0, 1.0)
    if base in {"sigma_R", "sigma_h", "sigma_D"}:
        return 0.0, max(abs(value) * 5.0, 1.0)
    if base == "nu_Res":
        return 0.1, max(abs(value) * 4.0, 50.0)
    return 0.0, max(abs(value) * 10.0, 1.0)


def clamp_to_open_bounds(values, lower, upper, epsilon: float = 1e-15):
    values_array = np.asarray(values, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if values_array.shape != lower_array.shape or values_array.shape != upper_array.shape:
        raise ValueError("Values and bounds must have the same shape")
    if np.any(lower_array > upper_array):
        raise ValueError("Lower bounds cannot exceed upper bounds")
    return np.minimum(np.maximum(values_array, lower_array + epsilon), upper_array - epsilon)
