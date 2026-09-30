"""Which detector pixels carry a usable measurement."""

from __future__ import annotations

import numpy as np


def valid_pixels(data: np.ndarray, mask: np.ndarray | None = None, *, negatives_valid: bool = False) -> np.ndarray:
    """Finite pixels outside the loader's invalid mask; negative ones only with ``negatives_valid``.

    Pilatus/Eiger (integer counts) mark module gaps and bad pixels with negative values and the
    NeXus loader already turns rejected pixels into NaN, so both end up here. A floating-point
    frame (dark- or background-subtracted) has real negative values: ``negatives_valid``.
    """
    data = np.asarray(data)
    valid = np.isfinite(data) if negatives_valid else np.isfinite(data) & (data >= 0)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != data.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match the frame {data.shape}.")
        valid &= ~mask
    return valid


__all__ = ["valid_pixels"]
