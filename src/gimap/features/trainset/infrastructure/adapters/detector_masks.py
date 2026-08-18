"""Focused Trainset detector-data and generation behavior."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


from .detector_images import crop_roi, load_scattering_image


def build_roi_shape_mask(shape: tuple[int, int], config: Dict[str, Any]) -> np.ndarray:
    """Return geometry-only masks that apply in both fixed and random modes."""
    mask = np.zeros(shape, dtype=bool)
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    for region in config.get("mask", {}).get("fixed_shapes", []):
        if region.get("type") != "roi_ellipse_exterior":
            continue
        cx, cy = float(region.get("cx", 0)), float(region.get("cy", 0))
        radius_x = max(1e-6, float(region.get("radius_x", region.get("radius", 0))))
        radius_y = max(1e-6, float(region.get("radius_y", region.get("radius", 0))))
        inside = ((xx - cx) / radius_x) ** 2 + ((yy - cy) / radius_y) ** 2 <= 1.0
        mask |= ~inside
    return mask


@lru_cache(maxsize=16)
def _cached_reference_roi(
    path: str,
    modified_ns: int,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    del modified_ns  # Included in the cache key so changed files are reloaded.
    return crop_roi(
        load_scattering_image(path),
        {"x": x, "y": y, "width": width, "height": height},
    ).copy()


def build_reference_threshold_mask(
    image: np.ndarray,
    config: Dict[str, Any],
    reference_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a spatial bad-pixel mask from the experimental reference ROI.

    Experimental intensity thresholds describe detector locations (module gaps,
    saturated/hot pixels and non-finite values). They must not be evaluated on
    a BornAgain intensity field, otherwise those detector defects disappear
    from simulated training images.

    ``image`` supplies the target shape. ``reference_image`` may be either the
    full detector or an already-cropped ROI. If no reference file is configured
    we retain the old direct-image behaviour for standalone callers and tests.
    """
    target = np.asarray(image)
    threshold = config.get("mask", {}).get("threshold", {})
    if not threshold.get("enabled", False):
        return np.zeros(target.shape, dtype=bool)

    roi = config.get("roi", {})
    source: Optional[np.ndarray] = None
    if reference_image is not None:
        candidate = np.asarray(reference_image)
        if candidate.shape == target.shape:
            source = candidate
        else:
            source = crop_roi(candidate, roi)
    else:
        path_text = str(config.get("project", {}).get("reference_file", "")).strip()
        path = Path(path_text) if path_text else None
        if path is not None and path.exists():
            source = _cached_reference_roi(
                str(path.resolve()),
                int(path.stat().st_mtime_ns),
                int(roi.get("x", 0)),
                int(roi.get("y", 0)),
                int(roi.get("width", target.shape[1])),
                int(roi.get("height", target.shape[0])),
            )

    if source is None:
        source = target
    mask = build_threshold_mask(source, config)
    if mask.shape != target.shape:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (int(target.shape[1]), int(target.shape[0])),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return np.asarray(mask, dtype=bool)


def build_fixed_mask(
    image: np.ndarray,
    config: Dict[str, Any],
    reference_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    mask_cfg = config.get("mask", {})
    mask = build_roi_shape_mask(image.shape, config)
    mask |= build_reference_threshold_mask(image, config, reference_image)
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    for shape in mask_cfg.get("fixed_shapes", []):
        shape_type = shape.get("type")
        if shape_type == "roi_ellipse_exterior":
            continue
        if shape_type == "rectangle":
            x, y = int(shape.get("x", 0)), int(shape.get("y", 0))
            width, height = int(shape.get("width", 0)), int(shape.get("height", 0))
            x0, x1 = max(0, x), min(image.shape[1], x + max(0, width))
            y0, y1 = max(0, y), min(image.shape[0], y + max(0, height))
            if x1 > x0 and y1 > y0:
                mask[y0:y1, x0:x1] = True
        elif shape_type == "circle":
            cx, cy = float(shape.get("cx", 0)), float(shape.get("cy", 0))
            radius = max(0.0, float(shape.get("radius", 0)))
            mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        elif shape_type == "ellipse":
            cx, cy = float(shape.get("cx", 0)), float(shape.get("cy", 0))
            radius_x = max(1e-6, float(shape.get("radius_x", shape.get("radius", 0))))
            radius_y = max(1e-6, float(shape.get("radius_y", shape.get("radius", 0))))
            mask |= ((xx - cx) / radius_x) ** 2 + ((yy - cy) / radius_y) ** 2 <= 1.0
    return mask


def build_threshold_mask(image: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Mask non-finite or out-of-range intensity in every mask mode."""
    threshold = config.get("mask", {}).get("threshold", {})
    if not threshold.get("enabled", False):
        return np.zeros(np.asarray(image).shape, dtype=bool)
    low = float(threshold.get("minimum", -np.inf))
    high = float(threshold.get("maximum", np.inf))
    data = np.asarray(image)
    return ~np.isfinite(data) | (data < low) | (data > high)


def merge_threshold_mask(
    image: np.ndarray,
    mask: np.ndarray,
    config: Dict[str, Any],
    reference_image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Merge random/fixed geometry with the experimental spatial threshold mask."""
    return np.asarray(mask, dtype=bool) | build_reference_threshold_mask(
        image,
        config,
        reference_image,
    )


def build_random_mask(
    shape: tuple[int, int], config: Dict[str, Any], rng: np.random.Generator
) -> np.ndarray:
    height, width = shape
    spec = config.get("mask", {}).get("random", {})
    mask = build_roi_shape_mask(shape, config)
    min_width = max(1, int(spec.get("bar_width_min", 2)))
    max_width = max(min_width, int(spec.get("bar_width_max", 6)))
    for _ in range(max(0, int(spec.get("vertical_bars", 0)))):
        bar_width = int(rng.integers(min_width, max_width + 1))
        x = int(rng.integers(0, max(1, width - bar_width + 1)))
        mask[:, x : x + bar_width] = True
    for _ in range(max(0, int(spec.get("horizontal_bars", 0)))):
        bar_width = int(rng.integers(min_width, max_width + 1))
        y = int(rng.integers(0, max(1, height - bar_width + 1)))
        mask[y : y + bar_width, :] = True
    yy, xx = np.ogrid[:height, :width]
    r_min = max(1, int(spec.get("circle_radius_min", 4)))
    r_max = max(r_min, int(spec.get("circle_radius_max", 12)))
    for _ in range(max(0, int(spec.get("circles", 0)))):
        radius = int(rng.integers(r_min, r_max + 1))
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    if spec.get("beamstop", True):
        radius = int(rng.integers(r_min, r_max + 1))
        cx, cy = width // 2, height // 2
        mask |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        direction = -1 if rng.random() < 0.5 else 1
        x0, x1 = sorted((cx, int(np.clip(cx + direction * width, 0, width))))
        mask[max(0, cy - 2) : min(height, cy + 3), x0:x1] = True
    return mask
