"""Manual fitting 参数选择与 bounds 规则。"""

from __future__ import annotations

import re

import numpy as np


def parameter_base_name(name: str) -> str:
    return re.sub(r"\d+$", "", str(name))


def default_refine_selected(name: str) -> bool:
    # Refine the main component geometry and amplitude by default.  Resolution
    # terms and the global scale remain opt-in because they are strongly
    # correlated with component intensities and can make a local refinement
    # under-determined.
    return parameter_base_name(name) in {"Int", "R", "h", "BG"}


def default_global_search_selected(name: str) -> bool:
    """Select identifiable geometry and amplitudes, but not redundant ``k``."""

    return parameter_base_name(name) in {
        "Int",
        "R",
        "sigma_R",
        "h",
        "sigma_h",
        "D",
        "sigma_D",
        "BG",
        "sigma_Res",
        "nu_Res",
        "int_Res",
    }


def default_refine_bounds(name: str, value: float) -> tuple[float, float]:
    """Return a conservative, positive local window around ``value``.

    Auto Refine is a polishing step from the current manual fit, not a global
    parameter search.  Shape lengths therefore get a +/-20% window, while the
    more weakly determined amplitudes and widths get a +/-50% window.
    """

    base = parameter_base_name(name)
    value = max(0.0, float(value))

    relative_span = {
        "Int": 0.5,
        "R": 0.2,
        "sigma_R": 0.5,
        "h": 0.2,
        "sigma_h": 0.5,
        "D": 0.2,
        "sigma_D": 0.5,
        "BG": 0.5,
        "sigma_Res": 0.5,
        "nu_Res": 0.25,
        "int_Res": 0.5,
        "k": 0.2,
    }.get(base, 0.2)
    zero_upper = {
        "sigma_R": 0.1,
        "sigma_h": 0.1,
        "sigma_D": 0.1,
        "sigma_Res": 0.01,
        "nu_Res": 1.0,
    }.get(base, 1.0)

    if value == 0.0:
        lower, upper = 0.0, zero_upper
    else:
        lower = max(0.0, value * (1.0 - relative_span))
        upper = value * (1.0 + relative_span)
    if base == "nu_Res":
        lower = max(0.1, lower)
        upper = max(upper, 0.1 + 1e-6)
    return lower, upper


def default_global_search_bounds(
    name: str,
    value: float,
    observed=None,
    q_values=None,
) -> tuple[float, float]:
    """Return broad, positive bounds for global exploration.

    These ranges deliberately remain anchored to the current model.  Global
    search is therefore useful for escaping a nearby basin without pretending
    that one generic interval is a scientifically valid prior for every
    experiment.  Background is the exception: it shares the measured intensity
    unit, so the lower quartile of the active curve provides a useful scale.
    """

    base = parameter_base_name(name)
    value = max(0.0, float(value))

    positive = np.asarray([] if observed is None else observed, dtype=float).reshape(-1)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    data_floor = float(np.percentile(positive, 25)) if positive.size else 0.0
    data_peak = float(np.max(positive)) if positive.size else 1.0

    q_abs = np.abs(np.asarray([] if q_values is None else q_values, dtype=float).reshape(-1))
    q_abs = q_abs[np.isfinite(q_abs) & (q_abs > 0)]
    q_min = float(np.min(q_abs)) if q_abs.size else 0.0
    q_max = float(np.max(q_abs)) if q_abs.size else 0.0
    small_length = max(1e-6, 0.05 / q_max) if q_max > 0 else 1e-3
    broad_length = max(1.0, 20.0 / q_min) if q_min > 0 else 100.0

    if base == "BG":
        return 0.0, max(value * 20.0, data_floor * 2.0, 1e-12)

    if base == "nu_Res":
        return 0.5, max(30.0, value * 2.0)

    if base in {"Int", "int_Res", "k"}:
        if value == 0.0:
            return 0.0, max(data_peak * 10.0, 1.0)
        return (0.0 if base != "k" else value * 1e-4), value * 1e4

    if base in {"R", "h"}:
        if value == 0.0:
            return small_length, broad_length
        return min(value * 0.25, small_length), max(value * 4.0, broad_length)

    if base == "D":
        upper_from_q = max(broad_length, np.pi * broad_length)
        if value == 0.0:
            return small_length, upper_from_q
        return min(value * 0.1, small_length), max(value * 10.0, upper_from_q)

    if base in {"sigma_R", "sigma_h", "sigma_D"}:
        width_floor = max(1e-8, small_length * 0.01)
        if value == 0.0:
            return width_floor, broad_length
        return min(value * 0.1, width_floor), max(value * 3.0, broad_length)

    if base == "sigma_Res":
        resolution_floor = max(1e-8, q_min * 1e-3) if q_min > 0 else 1e-4
        resolution_ceiling = max(1.0, 2.0 * q_max) if q_max > 0 else 10.0
        if value == 0.0:
            return resolution_floor, resolution_ceiling
        return min(value * 0.1, resolution_floor), max(
            value * 5.0,
            resolution_ceiling,
        )

    if value == 0.0:
        return 0.0, 1.0
    return value * 0.25, value * 4.0


def clamp_to_open_bounds(values, lower, upper, epsilon: float = 1e-15):
    values_array = np.asarray(values, dtype=float)
    lower_array = np.asarray(lower, dtype=float)
    upper_array = np.asarray(upper, dtype=float)
    if values_array.shape != lower_array.shape or values_array.shape != upper_array.shape:
        raise ValueError("Values and bounds must have the same shape")
    if np.any(lower_array > upper_array):
        raise ValueError("Lower bounds cannot exceed upper bounds")
    return np.minimum(np.maximum(values_array, lower_array + epsilon), upper_array - epsilon)
