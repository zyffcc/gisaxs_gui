"""WAXS background subtraction pure calculations."""

from __future__ import annotations

import numpy as np


def subtract_background(
    image: np.ndarray,
    background: np.ndarray,
    coefficient: float = 1.0,
) -> np.ndarray:
    """Return ``image - coefficient * background`` with a shared mask.

    The background frame must match the main image shape. Non-finite (NaN/Inf)
    values in either input are propagated to the result, so masked detector
    regions stay masked after subtraction. The result is float32 to match the
    WAXS image contract.
    """
    image_arr = np.asarray(image)
    background_arr = np.asarray(background)
    if image_arr.ndim != 2:
        raise ValueError(f"Expected a 2D WAXS image, got shape {image_arr.shape}")
    if background_arr.shape != image_arr.shape:
        raise ValueError(
            "Background image shape does not match the main image: "
            f"{background_arr.shape} != {image_arr.shape}"
        )

    result = (
        np.asarray(image_arr, dtype=np.float32)
        - float(coefficient) * np.asarray(background_arr, dtype=np.float32)
    )
    # Propagate non-finite mask regions (e.g. detector gaps) from either input.
    check = background_arr
    if image_arr.dtype.kind in "fc" and background_arr.dtype.kind not in "fc":
        check = image_arr
    elif image_arr.dtype.kind in "fc" and background_arr.dtype.kind in "fc":
        result = np.where(~(np.isfinite(image_arr) & np.isfinite(background_arr)), np.nan, result)
        return result
    if check.dtype.kind in "fc":
        result = np.where(~np.isfinite(check), np.nan, result)
    return result
