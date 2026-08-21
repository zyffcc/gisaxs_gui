"""Framework-neutral detector image state and preprocessing pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .image_transforms import apply_input_image_options, mirror_fill_detector_gaps


@dataclass(frozen=True)
class DetectorPreprocessing:
    """Scientific transforms that define one analysis-image revision."""

    flip_ud: bool = False
    threshold_enabled: bool = False
    threshold_min: float = -1e12
    threshold_max: float = 1e12
    mirror_fill_gaps: bool = False
    mirror_center_x: float | None = None
    mirror_gap_margin_px: int = 0
    mirror_gap_value: float = -1.0


@dataclass(frozen=True)
class DetectorImageState:
    """Immutable raw input and the single derived scientific analysis array."""

    raw_image: np.ndarray
    analysis_image: np.ndarray
    preprocessing: DetectorPreprocessing
    revision: int
    mirror_filled_gap_pixels: int = 0
    mirror_replaced_pixels: int = 0


def prepare_detector_image(
    image,
    preprocessing: DetectorPreprocessing,
    *,
    revision: int,
) -> DetectorImageState:
    """Build an analysis image from raw input without accumulating transforms."""

    raw = _immutable_raw_image(image)
    analysis_before_mirror = apply_input_image_options(
        raw,
        flip_ud=bool(preprocessing.flip_ud),
        threshold_enabled=bool(preprocessing.threshold_enabled),
        threshold_min=float(preprocessing.threshold_min),
        threshold_max=float(preprocessing.threshold_max),
    )

    filled_gap_pixels = 0
    replaced_pixels = 0
    if preprocessing.mirror_fill_gaps:
        analysis = mirror_fill_detector_gaps(
            analysis_before_mirror,
            center_x=preprocessing.mirror_center_x,
            gap_value=float(preprocessing.mirror_gap_value),
            gap_margin_px=max(0, int(preprocessing.mirror_gap_margin_px)),
        )
        original = np.asarray(analysis_before_mirror)
        filled = np.asarray(analysis)
        gap_value = float(preprocessing.mirror_gap_value)
        filled_gap_pixels = int(
            np.count_nonzero((original == gap_value) & (filled != gap_value))
        )
        changed = original != filled
        changed &= ~(np.isnan(original) & np.isnan(filled))
        replaced_pixels = int(np.count_nonzero(changed))
    else:
        analysis = analysis_before_mirror

    analysis = np.array(analysis, dtype=np.float32, copy=True, order="C")
    analysis.setflags(write=False)
    return DetectorImageState(
        raw_image=raw,
        analysis_image=analysis,
        preprocessing=preprocessing,
        revision=max(0, int(revision)),
        mirror_filled_gap_pixels=filled_gap_pixels,
        mirror_replaced_pixels=replaced_pixels,
    )


def _immutable_raw_image(image) -> np.ndarray:
    raw = np.asarray(image, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("detector preprocessing expects a 2D image")
    if raw.flags.writeable or not raw.flags.c_contiguous:
        raw = np.array(raw, dtype=np.float32, copy=True, order="C")
    raw.setflags(write=False)
    return raw


__all__ = [
    "DetectorImageState",
    "DetectorPreprocessing",
    "prepare_detector_image",
]
