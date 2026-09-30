"""Frame corrections applied before any reduction: frame sum, background, gap
guard and valid range.

All work on the raw frame, so every curve, the q map and the exports use the
corrected values.  Nothing here changes the geometry or the array layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

MAX_GAP_GUARD_PX = 20


@dataclass(frozen=True)
class Corrections:
    """What to correct; the defaults change nothing."""

    background_path: Optional[str] = None
    """Frame subtracted from every analysed frame (same detector and shape)."""
    background_frame: int = 0
    background_scale: float = 1.0
    """``image − scale × background`` (for example the ratio of exposure times)."""
    minimum: Optional[float] = None
    """Pixels below this raw value are treated as invalid (``None``: no limit)."""
    maximum: Optional[float] = None
    """Pixels above this raw value are treated as invalid, e.g. saturated or hot pixels."""
    gap_guard_px: int = 0
    """Also ignore pixels this close to a detector gap or bad pixel (see :func:`guard_invalid`)."""
    mask_path: Optional[str] = None
    """An image of the same size whose non-zero pixels are masked (pyFAI / Xeuss mask files)."""
    mask_shapes: tuple = ()
    """Rectangles and polygons drawn on the image (``MaskShape``): their pixels are left out."""
    mirror_fill: bool = False
    """GIWAXS: fill pixels without data from the mirror side (``preprocess.mirror_fill``)."""
    bad_pixels: bool = True
    """Leave out isolated hot and dead pixels found in the frame (see ``bad_pixels.py``). Part of the
    frame's validity, found before the corrections, so it does not count for ``is_identity``."""
    solid_angle: bool = False
    """GIWAXS: divide by the solid angle of each pixel, cos³2θ (``intensity.py``)."""
    polarization: Optional[float] = None
    """GIWAXS: divide by the polarisation factor with this ``f`` (0.95–0.99 synchrotron, 0 laboratory)."""
    film_thickness_nm: Optional[float] = None
    attenuation_length_um: Optional[float] = None
    """GIWAXS: divide by the absorption in a film of this thickness and attenuation length (both needed).

    The intensity corrections need the geometry; they are applied after the others, in the analysis,
    and are not part of ``is_identity``."""

    @property
    def is_identity(self) -> bool:
        return (
            not self.mirror_fill
            and self.background_path is None
            and self.minimum is None
            and self.maximum is None
            and self.gap_guard_px <= 0
        )


def guard_invalid(valid: np.ndarray, margin_px: int) -> np.ndarray:
    """``valid`` without the pixels within ``margin_px`` (square) of an invalid pixel.

    Pixels next to module gaps and dead pixels of Pilatus/Eiger detectors often
    read wrong (edge pixels, charge sharing) and show up as spikes in cuts.
    The border of the frame does not count as invalid.
    """
    margin = int(margin_px)
    if margin <= 0:
        return valid
    if margin > MAX_GAP_GUARD_PX:
        raise ValueError(f"The gap guard must be at most {MAX_GAP_GUARD_PX} pixels.")
    valid = np.asarray(valid, dtype=bool)
    invalid = ~valid
    if not invalid.any():
        return valid
    from scipy.ndimage import maximum_filter

    grown = maximum_filter(invalid.view(np.uint8), size=2 * margin + 1, mode="constant", cval=0)
    return valid & (grown == 0)


def sum_frames(frames: Sequence[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Pixel-wise sum of ``(data, valid)`` frames; a pixel is valid only where all are.

    Summing improves the statistics of short exposures (an in-situ series)
    without changing the geometry, so the result is reduced like one frame.
    """
    if not frames:
        raise ValueError("No frames to sum.")
    first = np.asarray(frames[0][0])
    total = np.zeros(first.shape, dtype=np.float64)
    valid = np.ones(first.shape, dtype=bool)
    for data, frame_valid in frames:
        data = np.asarray(data)
        if data.shape != first.shape:
            raise ValueError(
                f"Cannot sum a {data.shape[0]}×{data.shape[1]} frame with "
                f"{first.shape[0]}×{first.shape[1]} frames."
            )
        valid &= np.asarray(frame_valid, dtype=bool)
        total += np.where(frame_valid, data, 0.0)
    summed = total.astype(np.float32)
    summed[~valid] = np.nan
    return summed, valid


def apply_valid_range(
    data: np.ndarray, valid: np.ndarray, minimum: Optional[float], maximum: Optional[float]
) -> np.ndarray:
    """``valid`` restricted to ``minimum <= data <= maximum`` (limits are inclusive)."""
    if minimum is None and maximum is None:
        return valid
    limited = np.array(valid, dtype=bool, copy=True)
    with np.errstate(invalid="ignore"):
        if minimum is not None:
            limited &= data >= float(minimum)
        if maximum is not None:
            limited &= data <= float(maximum)
    return limited


def subtract_background(
    data: np.ndarray,
    valid: np.ndarray,
    background: np.ndarray,
    background_valid: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``data − scale × background``; a pixel is valid only where both frames are.

    The subtracted frame may be negative (over-subtraction, noise); such
    pixels stay valid because negative values are real results here.
    """
    data = np.asarray(data)
    background = np.asarray(background)
    if background.shape != data.shape:
        raise ValueError(
            f"The background frame is {background.shape[0]}×{background.shape[1]}, "
            f"the frame is {data.shape[0]}×{data.shape[1]}."
        )
    corrected = np.asarray(data, dtype=np.float32) - np.float32(scale) * np.asarray(
        background, dtype=np.float32
    )
    both = np.asarray(valid, dtype=bool) & np.asarray(background_valid, dtype=bool)
    corrected[~both] = np.nan
    return corrected, both


__all__ = [
    "Corrections",
    "MAX_GAP_GUARD_PX",
    "apply_valid_range",
    "guard_invalid",
    "subtract_background",
    "sum_frames",
]
