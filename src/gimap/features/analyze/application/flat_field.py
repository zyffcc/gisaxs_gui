"""The flat field of Analyze: read from its file once per version, what was used, and what the record says of it.

A flat field is read as a ``.npy`` (NumPy, no pickle) or as any frame the reader takes (CBF, TIFF, EDF,
NeXus/HDF5: its first frame, with the reader's mask), then normalised (``domain/flat_field.py``). Every frame of
a batch, a series or the folder watch is divided by the same flat field: the requests carry the same
``Corrections.flat_field_path``, and the file is read again only when it changes on disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from src.gimap.shared.file_digest import sha256_of

from ..domain import FlatField, check_flat_shape, normalise_flat_field
from ..domain.flat_field import FLAT_SHAPE, FLOOR, NO_RESPONSE, NOT_2D
from .reduction_inputs import image_stamp

APPLIED_AS = (
    "I / F_ff, right after the background: F_ff is the flat field divided by its mean over the pixels it keeps "
    "(F_ff > floor); pixels at or below the floor, not finite or masked by its reader are left out. With the GIWAXS "
    "intensity corrections F = F_solid_angle * F_polarisation * F_absorption * F_ff; the Poisson variance of a "
    "photon count becomes I / F^2"
)
NO_NUMBERS = "The flat field {name} holds no numbers."
FLAT_FIELD_TEMPLATES = (FLAT_SHAPE, NOT_2D, NO_RESPONSE, NO_NUMBERS)
"""Why a flat field cannot be used, as the page translates them (``presentation/texts.py``)."""


@dataclass(frozen=True)
class FlatFieldUse:
    """The flat field a frame was divided by: its file (path, SHA-256), the mean it was normalised by (in the
    units of the file), the floor and how many detector pixels it leaves out."""

    path: str
    sha256: Optional[str]
    mean: float
    masked_pixels: int
    floor: float
    shape: tuple[int, int]

    @property
    def name(self) -> str:
        return Path(self.path).name

    def to_record(self) -> dict[str, Any]:
        """The ``flat_field`` of a record's ``corrections`` block (``export_record.analysis_metadata``)."""
        return {
            "file": self.name, "path": self.path, "sha256": self.sha256, "normalisation_mean": float(self.mean),
            "floor": float(self.floor), "masked_pixels": int(self.masked_pixels), "shape": list(self.shape),
            "applied_as": APPLIED_AS,
        }


def read_flat_field(frames, path: str | Path) -> FlatField:
    """The normalised flat field of a file: ``.npy`` with ``numpy.load`` (no pickle), anything else through the
    frame reader ``frames`` (its first frame; the pixels its reader masks are left out). ``ValueError`` or
    ``OSError`` says why it cannot be used."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        data, valid = np.load(path, allow_pickle=False), None
    else:
        image = frames.load(path, 0)
        data = np.asarray(image.data, dtype=np.float64)
        valid = None if image.mask is None else ~np.asarray(image.mask, dtype=bool)
    if not np.issubdtype(np.asarray(data).dtype, np.number):
        raise ValueError(NO_NUMBERS.format(name=path.name))
    return normalise_flat_field(data, valid, name=path.name)


class FlatFieldMixin:
    """Needs ``frames`` (a ``FrameSource``); call ``_init_flat_field()`` in ``__init__``."""

    def _init_flat_field(self) -> None:
        self._flat_cache: Optional[tuple] = None

    def _flat_field(self, path: str, frame_shape: Optional[tuple[int, int]] = None) -> tuple[FlatField, FlatFieldUse]:
        """The flat field of ``path`` and what is recorded of it, read once per file version; ``ValueError``
        (``FLAT_SHAPE``) when it is not of ``frame_shape``."""
        key = image_stamp(path, self.frames)
        cache = self._flat_cache  # one tuple: analyses run on several threads
        if cache is None or cache[0] != key:
            flat = read_flat_field(self.frames, path)
            use = FlatFieldUse(str(path), sha256_of(path), flat.mean, flat.masked_pixels, flat.floor, flat.shape)
            cache = (key, flat, use)
            self._flat_cache = cache
        if frame_shape is not None:
            check_flat_shape(cache[1], frame_shape, Path(path).name)
        return cache[1], cache[2]

    def check_flat_field_file(self, path: str | Path, frame_shape: Optional[tuple[int, int]] = None) -> FlatFieldUse:
        """What the flat field of ``path`` is (``FlatFieldUse``); ``ValueError`` or ``OSError`` when it cannot be
        read or used, or is not of ``frame_shape`` — asked before it is kept, so a flat field of another detector
        is refused instead of failing every later analysis."""
        shape = None if frame_shape is None else (int(frame_shape[0]), int(frame_shape[1]))
        return self._flat_field(str(path), shape)[1]


__all__ = [
    "APPLIED_AS", "FLAT_SHAPE", "FLOOR", "NOT_2D", "NO_NUMBERS", "NO_RESPONSE", "FLAT_FIELD_TEMPLATES",
    "FlatFieldMixin", "FlatFieldUse", "read_flat_field",
]
