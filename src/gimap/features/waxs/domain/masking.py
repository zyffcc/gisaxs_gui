"""WAXS mask/display pure calculations。"""

from __future__ import annotations

import numpy as np


def prepare_display_array(
    image: np.ndarray,
    *,
    log_scale: bool,
    mask_min: float,
    mask_max: float,
    flip_vertical: bool,
) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32).copy()
    if log_scale:
        valid = (
            np.isfinite(arr)
            & (arr >= mask_min)
            & (arr <= mask_max)
            & (arr > 0)
        )
        arr[~valid] = np.nan
        np.log10(arr, out=arr, where=valid)
    else:
        invalid = ~np.isfinite(arr) | (arr < mask_min) | (arr > mask_max)
        arr[invalid] = np.nan
    if flip_vertical:
        arr = np.flipud(arr)
    return arr


def percentile_limits(arr: np.ndarray) -> tuple[float, float] | None:
    values = np.asarray(arr, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return None
    values = values.ravel() if finite.all() else values[finite]
    low, high = np.nanpercentile(values, [0.5, 99.5])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(np.nanmin(values))
        high = float(np.nanmax(values))
    if low == high:
        high = low + 1e-9
    return float(low), float(high)


def estimate_display_limits(
    image: np.ndarray,
    *,
    log_scale: bool,
    mask_min: float,
    mask_max: float,
    max_samples: int = 200_000,
    stride_hint: int = 20,
) -> tuple[float, float] | None:
    flat = np.asarray(image).ravel()
    if flat.size == 0:
        return None
    stride = max(int(stride_hint), int(np.ceil(flat.size / max_samples)))
    sample = np.asarray(flat[::stride], dtype=np.float32)
    if sample.size == 0:
        return None
    valid = np.isfinite(sample) & (sample >= mask_min) & (sample <= mask_max)
    if log_scale:
        valid &= sample > 0
    sample = sample[valid]
    if sample.size == 0:
        return None
    if log_scale:
        np.log10(sample, out=sample)
    return percentile_limits(sample)
