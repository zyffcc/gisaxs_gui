"""1D curve 的排序、过滤、插值、单位和显示变换。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .models import AxisFilterMode, QUnit


Diagnostic = Callable[[str], None]


def interpolate_series(x, y, x_new, method: str):
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    target = np.asarray(x_new, dtype=float)
    normalized = (method or "Linear").lower()
    if normalized == "linear":
        return np.interp(target, x_array, y_array)
    if normalized == "quadratic":
        try:
            from scipy.interpolate import interp1d

            function = interp1d(
                x_array,
                y_array,
                kind="quadratic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            return function(target)
        except Exception:
            coefficients = np.polyfit(
                x_array,
                y_array,
                deg=min(2, max(1, len(x_array) - 1)),
            )
            return np.polyval(coefficients, target)
    if normalized == "spline":
        try:
            from scipy.interpolate import CubicSpline

            return CubicSpline(x_array, y_array, extrapolate=True)(target)
        except Exception:
            return np.interp(target, x_array, y_array)
    return np.interp(target, x_array, y_array)


def sort_filter_pairs(
    x_values,
    intensity_values,
    *,
    context: str = "cut",
    pixel_rows=None,
    on_diagnostic: Diagnostic | None = None,
):
    """保持 q/intensity 配对，去除无效值，稳定排序并平均重复 q。"""
    report = on_diagnostic or (lambda _message: None)
    x_array = np.asarray(x_values, dtype=float).reshape(-1)
    y_array = np.asarray(intensity_values, dtype=float).reshape(-1)
    count = min(x_array.size, y_array.size)
    if x_array.size != y_array.size:
        report(
            f"{context}: length mismatch x={x_array.size}, intensity={y_array.size}; "
            f"truncating to {count}."
        )
    x_array = x_array[:count]
    y_array = y_array[:count]
    rows = None if pixel_rows is None else np.asarray(pixel_rows).reshape(-1)[:count]

    finite = np.isfinite(x_array) & np.isfinite(y_array)
    removed = int(count - np.sum(finite))
    if removed:
        report(f"{context}: removed {removed} non-finite q/intensity point(s).")
    x_array = x_array[finite]
    y_array = y_array[finite]
    if rows is not None:
        rows = rows[finite]
    if x_array.size == 0:
        raise ValueError(f"{context}: no finite q/intensity pairs")

    order = np.argsort(x_array, kind="mergesort")
    x_array = x_array[order]
    y_array = y_array[order]
    if rows is not None:
        rows = rows[order]

    if x_array.size > 1:
        unique_x, inverse, duplicate_counts = np.unique(
            x_array,
            return_inverse=True,
            return_counts=True,
        )
        if unique_x.size != x_array.size:
            summed_y = np.zeros(unique_x.size, dtype=float)
            np.add.at(summed_y, inverse, y_array)
            y_array = summed_y / duplicate_counts
            if rows is not None:
                summed_rows = np.zeros(unique_x.size, dtype=float)
                np.add.at(summed_rows, inverse, rows.astype(float))
                rows = summed_rows / duplicate_counts
            duplicate_count = int(x_array.size - unique_x.size)
            report(
                f"{context}: merged {duplicate_count} duplicate q coordinate(s) "
                "before interpolation."
            )
            x_array = unique_x
    return x_array, y_array, rows


def filter_axis(q_values, intensity_values, mode: AxisFilterMode = "all", *, context="cut"):
    q_array = np.asarray(q_values, dtype=float)
    intensity = np.asarray(intensity_values, dtype=float)
    if mode == "positive":
        keep = q_array > 0
    elif mode == "negative":
        keep = q_array < 0
    elif mode == "all":
        keep = np.ones(q_array.shape, dtype=bool)
    else:
        raise ValueError(f"Unsupported axis filter mode: {mode}")
    q_array = q_array[keep]
    intensity = intensity[keep]
    if q_array.size < 2:
        raise ValueError(f"{context}: not enough points remain after {mode} axis filtering")
    return q_array, intensity


def filter_for_display(q_data, y_data=None, mode: AxisFilterMode = "all"):
    q_array = np.asarray([] if q_data is None else q_data)
    y_array = None if y_data is None else np.asarray(y_data)
    finite = np.isfinite(q_array)
    if y_array is not None:
        finite &= np.isfinite(y_array)
    q_array = q_array[finite]
    if y_array is not None:
        y_array = y_array[finite]

    if mode == "positive":
        axis_mask = q_array > 0
    elif mode == "negative":
        axis_mask = q_array < 0
    elif mode == "all":
        axis_mask = np.ones(q_array.shape, dtype=bool)
    else:
        raise ValueError(f"Unsupported axis filter mode: {mode}")
    q_raw = q_array[axis_mask]
    if y_array is not None:
        y_array = y_array[axis_mask]
    q_plot = np.abs(q_raw) if mode == "negative" else np.array(q_raw, copy=True)
    if q_plot.size > 0 and mode == "negative":
        order = np.argsort(q_plot)
        q_raw = q_raw[order]
        q_plot = q_plot[order]
        if y_array is not None:
            y_array = y_array[order]
    return q_raw, q_plot, y_array, mode


def q_values_for_model(q_values, source_unit: QUnit):
    array = np.asarray([] if q_values is None else q_values, dtype=float)
    if array.size == 0:
        return array
    return array * 10.0 if source_unit == "angstrom" else array


def q_values_for_display(q_values, source_unit: QUnit, display_unit: QUnit):
    values_nm = q_values_for_model(q_values, source_unit)
    if values_nm.size == 0:
        return values_nm
    return values_nm * (0.1 if display_unit == "angstrom" else 1.0)


def valid_y_values_for_limits(y_values, log_y=False):
    array = np.asarray([] if y_values is None else y_values, dtype=float).ravel()
    if array.size == 0:
        return array
    mask = np.isfinite(array)
    if log_y:
        mask &= array > 0
    return array[mask]


def normalize_intensity(intensity):
    if len(intensity) == 0:
        return intensity
    maximum = np.max(intensity)
    if maximum > 0:
        return intensity / maximum
    return intensity
