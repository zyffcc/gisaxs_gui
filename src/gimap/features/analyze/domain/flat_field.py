"""Flat field (detector response): every pixel divided by its sensitivity relative to the average pixel.

The pixels of a detector do not respond alike (gain, threshold, sensor thickness, the edges of a module). A flat
field is a frame of a uniform exposure (a fluorescence foil, an amorphous scatterer, the detector maker's own
calibration); divided by its mean it is each pixel's response relative to the average pixel, and the corrected
value is ``I'_p = I_p / F_ff,p``. It changes only intensities, never q, and needs no geometry:

* **Normalisation** — ``F_ff = F_raw / m``, ``m`` the mean of ``F_raw`` over the pixels the flat field keeps, so
  ``F_ff`` is 1 on average over them. A pixel is kept where ``F_raw`` is finite, not masked by its reader and
  ``F_ff > FLOOR`` (0.05); the rest do not respond and are left out of every frame (masked) and counted.
  ``m`` and the kept pixels depend on each other: starting from every pixel above 0, ``m`` is recomputed over the
  pixels above ``FLOOR × m`` until they no longer change (a few rounds; each only removes pixels).
* **Composition** — with the intensity corrections, ``F_p = F_Ω F_P F_A F_ff`` (``intensity.py``): the flat field is
  divided right after the background (pyFAI: ``(I − dark) / flat``), the geometric factors at the end. The variance
  of a photon count divided by it is ``I_p / F_p²`` (``BinnedMean.add``, ``native_profile``).
* A flat field of another shape than the frame is refused (``FLAT_SHAPE``): never resampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

FLOOR = 0.05
"""Pixels whose normalised flat field is at or below this (dead, shadowed, a gap) are left out."""
MAX_ROUNDS = 50
FLAT_SHAPE = "The flat field {name} is {flat_shape}; the frame is {shape}."
NOT_2D = "The flat field {name} is not one 2D image (it has {dimensions} dimensions)."
NO_RESPONSE = "The flat field {name} has no pixel above 0."
"""Why a flat field cannot be used (templates: the file name is filled in; ``presentation/texts.py`` translates)."""


@dataclass(frozen=True)
class FlatField:
    """A normalised flat field: ``factor`` (float32, ``F_ff``; NaN where not kept), the pixels it keeps, and the
    mean ``m`` of the raw flat field over them (in the units of its file)."""

    factor: np.ndarray
    kept: np.ndarray
    mean: float
    floor: float = FLOOR

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.factor.shape[0]), int(self.factor.shape[1])

    @property
    def masked_pixels(self) -> int:
        """Detector pixels the flat field leaves out (at or below the floor, not finite, masked by its reader)."""
        return int(self.kept.size - np.count_nonzero(self.kept))


def normalise_flat_field(
    raw: np.ndarray, valid: Optional[np.ndarray] = None, *, floor: float = FLOOR, name: str = "",
) -> FlatField:
    """``raw`` divided by its mean over the pixels it keeps (module docstring); ``valid``: the pixels its reader
    did not mask. ``ValueError`` (``NOT_2D``, ``NO_RESPONSE``) when it cannot be a flat field."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(NOT_2D.format(name=name, dimensions=raw.ndim))
    with np.errstate(invalid="ignore"):
        usable = np.isfinite(raw) & (raw > 0)
    if valid is not None:
        usable &= np.asarray(valid, dtype=bool)
    if not usable.any():
        raise ValueError(NO_RESPONSE.format(name=name))
    kept = usable
    mean = float(raw[kept].mean())
    for _ in range(MAX_ROUNDS):  # each round removes pixels or stops: the largest pixel always stays
        fewer = usable & (raw > float(floor) * mean)
        if np.array_equal(fewer, kept):
            break
        kept, mean = fewer, float(raw[fewer].mean())
    factor = np.full(raw.shape, np.nan, dtype=np.float32)
    factor[kept] = (raw[kept] / mean).astype(np.float32)
    return FlatField(factor, kept, mean, float(floor))


def check_flat_shape(flat: FlatField, shape: tuple[int, int], name: str = "") -> None:
    """``ValueError`` (``FLAT_SHAPE``) when the flat field is not of the frame's ``shape``."""
    rows, columns = int(shape[0]), int(shape[1])
    if flat.shape != (rows, columns):
        raise ValueError(FLAT_SHAPE.format(
            name=name, flat_shape=f"{flat.shape[0]}×{flat.shape[1]}", shape=f"{rows}×{columns}"))


def apply_flat_field(
    data: np.ndarray, valid: np.ndarray, flat: FlatField, *, name: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """``data / F_ff`` (float32) and the valid pixels: those that were and that the flat field keeps."""
    check_flat_shape(flat, np.shape(data), name)
    with np.errstate(invalid="ignore", divide="ignore"):
        corrected = (np.asarray(data, dtype=np.float32) / flat.factor).astype(np.float32, copy=False)
    return corrected, np.asarray(valid, dtype=bool) & flat.kept


__all__ = [
    "FLAT_SHAPE", "FLOOR", "FlatField", "NOT_2D", "NO_RESPONSE", "apply_flat_field", "check_flat_shape",
    "normalise_flat_field",
]
