"""Fitting score 和解析 scale 优化。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def log_residuals(observed, predicted, epsilon: float = 1e-30) -> np.ndarray:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if observed_array.shape != predicted_array.shape:
        raise ValueError("Observed and predicted arrays must have the same shape")
    return np.log10(np.maximum(predicted_array, epsilon)) - np.log10(
        np.maximum(observed_array, epsilon)
    )


def log_rmse(observed, predicted, epsilon: float = 1e-30) -> float:
    residual = log_residuals(observed, predicted, epsilon=epsilon)
    return float(np.sqrt(np.mean(residual * residual)))


def chi_square(observed, predicted) -> float:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    count = min(observed_array.size, predicted_array.size)
    if count == 0:
        return float("nan")
    observed_array = observed_array.reshape(-1)[:count]
    predicted_array = predicted_array.reshape(-1)[:count]
    finite = np.isfinite(observed_array) & np.isfinite(predicted_array)
    if not np.any(finite):
        return float("nan")
    delta = observed_array[finite] - predicted_array[finite]
    return float(np.mean(delta * delta))


@dataclass(frozen=True)
class ScaleOptimization:
    scale: float
    method: str
    residual_before: float
    residual_after: float
    base: np.ndarray
    observed: np.ndarray


def optimize_scale_factor(observed, fitted, current_scale: float) -> ScaleOptimization:
    """保持 controller 的 analytical/NNLS K-value 定义。"""
    observed_array = np.asarray(observed, dtype=float).reshape(-1)
    fitted_array = np.asarray(fitted, dtype=float).reshape(-1)
    if observed_array.shape != fitted_array.shape:
        raise ValueError("Observed and fitted arrays must have the same shape")
    if np.any(~np.isfinite(observed_array)) or np.any(~np.isfinite(fitted_array)):
        raise ValueError("Data contains NaN or infinite values")

    safe_scale = max(abs(float(current_scale)), 1e-12)
    base = fitted_array / safe_scale
    valid = np.isfinite(observed_array) & np.isfinite(base) & (base != 0)
    if not np.any(valid):
        raise ValueError("No valid data points for K optimization")
    observed_valid = observed_array[valid]
    base_valid = base[valid]
    denominator = np.dot(base_valid, base_valid)
    if denominator <= 1e-12:
        raise ValueError("Base function has zero norm, cannot optimize K")

    scale = float(np.dot(base_valid, observed_valid) / denominator)
    method = "Analytical"
    if scale <= 0:
        try:
            from scipy.optimize import nnls

            result, _residual = nnls(base_valid.reshape(-1, 1), observed_valid)
            scale = float(result[0]) if len(result) > 0 else 1.0
            method = "NNLS"
        except ImportError:
            scale = abs(scale)
            method = "Analytical (abs)"
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid optimized K-value: {scale}")

    residual_before = float(np.sum((float(current_scale) * base_valid - observed_valid) ** 2))
    residual_after = float(np.sum((scale * base_valid - observed_valid) ** 2))
    return ScaleOptimization(
        scale=scale,
        method=method,
        residual_before=residual_before,
        residual_after=residual_after,
        base=base,
        observed=observed_array,
    )
