"""AI fitting 输入曲线的纯清洗规则。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import AxisFilterMode


@dataclass(frozen=True)
class AiCurve:
    q: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=np.float64).reshape(-1)
        intensity = np.asarray(self.intensity, dtype=np.float64).reshape(-1)
        sigma = np.asarray(self.sigma, dtype=np.float64).reshape(-1)
        if not (q.size == intensity.size == sigma.size):
            raise ValueError("AI curve arrays must have the same length")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "sigma", sigma)


def ai_q_key(q_value) -> str:
    try:
        return f"{float(q_value):.12g}"
    except Exception:
        return str(q_value)


def prepare_ai_curve(
    q,
    intensity,
    sigma=None,
    *,
    axis_filter: AxisFilterMode = "all",
    roi: tuple[float, float] | None = None,
    excluded_q: set[str] | None = None,
    minimum_points: int = 16,
) -> AiCurve:
    q_array = np.asarray(q, dtype=np.float64).reshape(-1)
    intensity_array = np.asarray(intensity, dtype=np.float64).reshape(-1)
    count = min(q_array.size, intensity_array.size)
    q_array = q_array[:count]
    intensity_array = intensity_array[:count]
    sigma_array = None if sigma is None else np.asarray(sigma, dtype=np.float64).reshape(-1)[:count]

    if axis_filter == "positive":
        keep = q_array > 0
        q_array = q_array[keep]
        intensity_array = intensity_array[keep]
        if sigma_array is not None:
            sigma_array = sigma_array[keep]
    elif axis_filter == "negative":
        keep = q_array < 0
        q_array = np.abs(q_array[keep])
        intensity_array = intensity_array[keep]
        if sigma_array is not None:
            sigma_array = sigma_array[keep]
        order = np.argsort(q_array)
        q_array = q_array[order]
        intensity_array = intensity_array[order]
        if sigma_array is not None:
            sigma_array = sigma_array[order]
    elif axis_filter != "all":
        raise ValueError(f"Unsupported axis filter mode: {axis_filter}")

    if roi is not None:
        lower, upper = sorted((float(roi[0]), float(roi[1])))
        region = np.isfinite(q_array) & (q_array >= lower) & (q_array <= upper)
        if int(np.sum(region)) >= minimum_points:
            q_array = q_array[region]
            intensity_array = intensity_array[region]
            if sigma_array is not None:
                sigma_array = sigma_array[region]

    if sigma_array is None:
        sigma_array = np.maximum(0.05 * np.maximum(intensity_array, 1e-30), 1e-30)
    finite = (
        np.isfinite(q_array)
        & np.isfinite(intensity_array)
        & np.isfinite(sigma_array)
        & (q_array > 0)
        & (intensity_array > 0)
        & (sigma_array > 0)
    )
    if int(np.sum(finite)) < minimum_points:
        raise ValueError(f"AI fitting requires at least {minimum_points} valid positive points")
    q_array = q_array[finite]
    intensity_array = intensity_array[finite]
    sigma_array = sigma_array[finite]

    excluded = excluded_q or set()
    if excluded:
        keep = np.asarray([ai_q_key(value) not in excluded for value in q_array], dtype=bool)
        if int(np.sum(keep)) >= minimum_points:
            q_array = q_array[keep]
            intensity_array = intensity_array[keep]
            sigma_array = sigma_array[keep]
    return AiCurve(q_array, intensity_array, sigma_array)
