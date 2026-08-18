"""Focused Trainset detector-data and generation behavior."""

from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np


def _range_value(
    step: Dict[str, Any],
    key: str,
    rng: np.random.Generator,
    overrides: Dict[str, float],
    plugin: str,
    fallback_min: float,
    fallback_max: float,
) -> float:
    override_key = f"{plugin}.{key}"
    if override_key in overrides:
        return float(overrides[override_key])
    minimum = float(step.get(f"{key}_min", fallback_min))
    maximum = float(step.get(f"{key}_max", fallback_max))
    if maximum < minimum:
        minimum, maximum = maximum, minimum
    return minimum if maximum == minimum else float(rng.uniform(minimum, maximum))


def generate_physical_background(
    image: np.ndarray,
    config: Dict[str, Any],
    step: Dict[str, Any],
    rng: np.random.Generator,
    overrides: Optional[Dict[str, float]] = None,
    trace: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Generate the configurable Yuxin-style physical GISAXS background.

    Coordinates are normalized to the selected ROI, which makes the controls
    meaningful for both small local previews and full-resolution datasets.
    """
    from ...domain.geometry import q_vectors

    overrides = overrides or {}
    if "target_fraction_min" not in step and "fraction_min" in step:
        step = {
            **step,
            "target_fraction_min": step["fraction_min"],
            "target_fraction_max": step.get("fraction_max", step["fraction_min"]),
        }
    height, width = image.shape[:2]
    q = q_vectors(config)
    roi = config.get("roi", {})
    x, y = int(roi.get("x", 0)), int(roi.get("y", 0))
    qy = np.asarray(q["qy"][y : y + height, x : x + width], dtype=np.float64)
    qz = np.asarray(q["qz"][y : y + height, x : x + width], dtype=np.float64)
    if qy.shape != image.shape or qz.shape != image.shape:
        qz, qy = np.mgrid[0.0 : 1.0 : complex(height), -1.0 : 1.0 : complex(width)]
    else:
        qy_mid = float(np.nanmedian(qy))
        qy_span = max(float(np.nanmax(qy) - np.nanmin(qy)), 1e-12)
        qz_min = float(np.nanmin(qz))
        qz_span = max(float(np.nanmax(qz) - qz_min), 1e-12)
        qy = (qy - qy_mid) / qy_span
        qz = (qz - qz_min) / qz_span

    def value(key: str, low: float, high: float) -> float:
        selected = _range_value(step, key, rng, overrides, "physical_background", low, high)
        if trace is not None:
            trace[key] = float(selected)
        return selected

    target_fraction = max(0.0, value("target_fraction", 0.05, 0.30))
    constant = max(0.0, value("constant_fraction", 0.0, 0.03))
    spec_amplitude = max(0.0, value("specular_amplitude", 0.2, 1.0))
    spec_width = max(1e-4, value("specular_width_fraction", 0.01, 0.04))
    spec_widening = max(0.0, value("specular_widening", 0.0, 0.12))
    spec_decay = max(1e-4, value("specular_decay_fraction", 0.2, 0.8))
    local_width = spec_width * (1.0 + spec_widening * np.clip(qz, 0.0, 1.0))
    specular = (
        spec_amplitude
        * np.exp(-0.5 * (qy / local_width) ** 2)
        * np.exp(-np.clip(qz, 0.0, None) / spec_decay)
    )

    yoneda_amplitude = max(0.0, value("yoneda_amplitude", 0.1, 0.7))
    yoneda_center = value("yoneda_center_fraction", 0.50, 0.72)
    yoneda_width = max(1e-4, value("yoneda_width_fraction", 0.02, 0.08))
    yoneda_hole = float(np.clip(value("yoneda_center_hole", 0.4, 0.95), 0.0, 1.0))
    center_suppression = 1.0 - yoneda_hole * np.exp(-0.5 * (qy / max(2.5 * spec_width, 1e-4)) ** 2)
    yoneda = (
        yoneda_amplitude
        * np.exp(-0.5 * ((qz - yoneda_center) / yoneda_width) ** 2)
        * center_suppression
    )

    wedge_amplitude = max(0.0, value("wedge_amplitude", 0.05, 0.40))
    anisotropy = max(1e-3, value("wedge_anisotropy", 0.6, 2.0))
    porod = max(0.1, value("wedge_porod_exponent", 2.0, 3.8))
    rg_fraction = max(1e-3, value("wedge_rg_fraction", 0.05, 0.25))
    radial = np.sqrt((qy / anisotropy) ** 2 + np.clip(qz, 0.0, None) ** 2)
    wedge = wedge_amplitude * np.power(1.0 + (radial / rg_fraction) ** 2, -0.5 * porod)

    plane = (
        constant
        + value("plane_qy_slope", -0.08, 0.08) * qy
        + value("plane_qz_slope", -0.08, 0.08) * (qz - 0.5)
    )
    background = np.maximum(specular + yoneda + wedge + plane, 0.0)
    qz_cut = max(0.0, value("low_qz_cut_fraction", 0.0, 0.08))
    if qz_cut > 0:
        transition = max(0.005, qz_cut * 0.25)
        background *= 1.0 / (1.0 + np.exp(-(qz - qz_cut) / transition))
    blur_sigma = max(0.0, value("blur_sigma_px", 0.0, 0.6))
    if blur_sigma > 1e-6:
        background = cv2.GaussianBlur(background.astype(np.float32), (0, 0), blur_sigma)

    positive = np.asarray(image)[np.isfinite(image) & (np.asarray(image) > 0)]
    signal_reference = float(np.percentile(positive, 75)) if positive.size else 1.0
    bg_reference = max(float(np.percentile(background, 95)), 1e-12)
    background = background / bg_reference * signal_reference * target_fraction
    return np.asarray(background, dtype=np.float32)
