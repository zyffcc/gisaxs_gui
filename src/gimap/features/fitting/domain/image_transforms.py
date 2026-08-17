"""Detector image 的无 UI 数组变换。"""

from __future__ import annotations

import numpy as np


def apply_threshold_mask(image, enabled=False, lower=-1e12, upper=1e12):
    """以 NaN 表示无效或超出阈值范围的 detector pixel。"""
    array = np.asarray(image, dtype=np.float32)
    if not enabled:
        return array
    low, high = sorted((float(lower), float(upper)))
    masked = array.copy()
    invalid = ~np.isfinite(masked) | (masked < low) | (masked > high)
    masked[invalid] = np.nan
    return masked


def apply_input_image_options(
    image,
    *,
    flip_ud=False,
    threshold_enabled=False,
    threshold_min=-1e12,
    threshold_max=1e12,
):
    """执行 legacy read-level transforms，保持 flip 只应用一次。"""
    transformed = np.asarray(image, dtype=np.float32)
    if flip_ud:
        transformed = np.ascontiguousarray(np.flipud(transformed))
    return apply_threshold_mask(
        transformed,
        enabled=threshold_enabled,
        lower=threshold_min,
        upper=threshold_max,
    )


def finite_mean_axis(data, axis):
    """沿指定轴求均值，无效 pixel 权重为零。"""
    array = np.asarray(data, dtype=float)
    finite = np.isfinite(array)
    counts = np.sum(finite, axis=axis)
    totals = np.sum(np.where(finite, array, 0.0), axis=axis)
    result = np.full(np.shape(totals), np.nan, dtype=float)
    np.divide(totals, counts, out=result, where=counts > 0)
    return result


def finite_log_profiles(data):
    """建立 center-finding log profiles，不给 masked pixel 统计权重。"""
    array = np.asarray(data, dtype=float)
    finite = np.isfinite(array)
    if not np.any(finite):
        raise ValueError("No valid detector pixels remain after masking")
    log_data = np.full(array.shape, np.nan, dtype=float)
    log_data[finite] = np.log10(np.maximum(array[finite], 1.0))
    vertical = np.nansum(log_data, axis=1)
    vertical[~np.any(finite, axis=1)] = -np.inf
    horizontal = np.nansum(log_data, axis=0)
    horizontal[~np.any(finite, axis=0)] = 0.0
    return vertical, horizontal


def mirror_fill_detector_gaps(image, center_x=None, gap_value=-1, gap_margin_px=0):
    """使用束心左右镜像补全 detector gap，保持 legacy index 规则。"""
    if center_x is None:
        raise ValueError("center_x is required for mirror gap fill")
    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError("mirror gap fill expects a 2D image")

    filled = array.copy()
    source = array.copy()
    gap_mask = array == gap_value
    if not np.any(gap_mask):
        return filled

    margin = max(0, int(gap_margin_px or 0))
    replace_mask = gap_mask.copy()
    if margin > 0:
        for dx in range(-margin, margin + 1):
            if dx == 0:
                continue
            shifted = np.zeros_like(gap_mask, dtype=bool)
            if dx < 0:
                shifted[:, :dx] = gap_mask[:, -dx:]
            else:
                shifted[:, dx:] = gap_mask[:, :-dx]
            replace_mask |= shifted

    gap_y, gap_x = np.where(replace_mask)
    if gap_x.size == 0:
        return filled
    x_mirror = np.rint((2.0 * float(center_x)) - gap_x).astype(int)
    in_bounds = (x_mirror >= 0) & (x_mirror < array.shape[1])
    if not np.any(in_bounds):
        return filled

    target_y = gap_y[in_bounds]
    target_x = gap_x[in_bounds]
    source_x = x_mirror[in_bounds]
    source_values = source[target_y, source_x]
    valid_source = (source_values != gap_value) & np.isfinite(source_values)
    if np.any(valid_source):
        filled[target_y[valid_source], target_x[valid_source]] = source_values[valid_source]
    return filled
