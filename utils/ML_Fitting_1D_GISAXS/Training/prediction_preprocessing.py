"""Focused helpers for TOP-K prediction: prediction preprocessing."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TrainSetBuild import schema


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    expv = np.exp(logits)
    return expv / np.maximum(expv.sum(), 1e-300)


def drop_log_outliers(q, I, sigma_arr, outlier_sigma=6.0, window_size=21, max_run=10):
    if len(q) < 16:
        return q, I, sigma_arr, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    window_size = min(31, max(11, int(window_size)))
    if window_size % 2 == 0:
        window_size += 1
    log_i = np.log(np.maximum(I, 1e-30))
    local_median = rolling_median(log_i, window_size)
    residual = log_i - local_median
    center = np.median(residual)
    mad = np.median(np.abs(residual - center))
    robust_sigma = 1.4826 * mad
    if not np.isfinite(robust_sigma) or robust_sigma <= 1e-12:
        print("Outlier filtering skipped: MAD is too small.", flush=True)
        return q, I, sigma_arr, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    raw_outlier = np.abs(residual) > float(outlier_sigma) * robust_sigma
    isolated_outlier = short_true_runs(raw_outlier, max_run=max_run)
    keep = ~isolated_outlier
    if np.sum(keep) < 16:
        print("Outlier filtering skipped: it would leave too few points.", flush=True)
        return q, I, sigma_arr, np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    removed = int(np.sum(isolated_outlier))
    broad = int(np.sum(raw_outlier & ~isolated_outlier))
    print(
        f"Outlier filtering removed {removed} isolated points "
        f"(raw outliers={int(np.sum(raw_outlier))}, broad-run kept={broad}, "
        f"window={window_size}, max_run={max_run}, MAD={mad:.4g}).",
        flush=True,
    )
    return q[keep], I[keep], sigma_arr[keep], q[isolated_outlier], I[isolated_outlier]


def downsample_curve(q, I, sigma_arr):
    if len(q) <= schema.MAX_POINTS:
        return q, I, sigma_arr
    idx = np.linspace(0, len(q) - 1, schema.MAX_POINTS).astype(int)
    return q[idx], I[idx], sigma_arr[idx]


def effective_range(low, high, mask):
    low_eff = np.where(mask > 0.0, low, 0.0)
    high_eff = np.where(mask > 0.0, high, 1.0)
    return low_eff, high_eff


def sigmoid_stable(x):
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-np.minimum(x[pos], 80.0)))
    exp_x = np.exp(np.maximum(x[~pos], -80.0))
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def apply_q_range(q, I, sigma_arr, q_min=None, q_max=None):
    keep = np.ones_like(q, dtype=bool)
    if q_min is not None:
        keep &= q >= float(q_min)
    if q_max is not None:
        keep &= q <= float(q_max)
    removed = int(np.sum(~keep))
    if np.sum(keep) < 16:
        raise ValueError(f"q range leaves too few points: kept {np.sum(keep)}, removed {removed}")
    if removed:
        print(f"q range mask removed {removed} points; kept {np.sum(keep)}", flush=True)
    return q[keep], I[keep], sigma_arr[keep]


def rolling_median(values, window_size):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 3:
        return values.copy()
    window_size = int(window_size)
    window_size = max(3, min(window_size, values.size if values.size % 2 == 1 else values.size - 1))
    if window_size % 2 == 0:
        window_size -= 1
    pad = window_size // 2
    padded = np.pad(values, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_size)
    return np.median(windows, axis=1)


def short_true_runs(mask, max_run=2):
    mask = np.asarray(mask, dtype=bool)
    out = np.zeros_like(mask, dtype=bool)
    idx = 0
    while idx < mask.size:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < mask.size and mask[idx]:
            idx += 1
        if idx - start <= max_run:
            out[start:idx] = True
    return out
