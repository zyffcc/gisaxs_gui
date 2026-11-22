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

from src.gimap.shared.file_digest import sha256_of

from ..domain import Corrections, FlatField, check_flat_shape, normalise_flat_field, subtract_background
from ..domain.flat_field import FLAT_SHAPE, FLOOR, NO_RESPONSE, NOT_2D
from .reference_frames import ReferenceFrame, dark_scale, detector_counts, file_version, read_reference

APPLIED_AS = (
    "I / F_ff after background subtraction. F_ff = (D_ff - r*K) / m, where m is the mean over retained "
    "pixels. Retain F_ff > floor. Conditional on m, Var(F_ff) = (Var(D_ff) + r^2*Var(K)) / m^2. "
    "Propagated errors include covariance when I and F_ff share K."
)
NO_NUMBERS = "The flat field {name} holds no numbers."
FLAT_FIELD_TEMPLATES = (FLAT_SHAPE, NOT_2D, NO_RESPONSE, NO_NUMBERS)
"""Why a flat field cannot be used, as the page translates them (``presentation/texts.py``)."""


@dataclass(frozen=True)
class FlatFieldUse:
    """The flat reference file, dark correction, normalising mean in counts and masked-pixel count."""

    path: str
    sha256: Optional[str]
    mean: float
    masked_pixels: int
    floor: float
    shape: tuple[int, int]
    dark_scale: float = 0.0
    dark_scale_source: str = "no dark frame"
    already_dark_subtracted: bool = False
    exposure_time_s: Optional[float] = None
    dark_exposure_time_s: Optional[float] = None

    @property
    def name(self) -> str:
        return Path(self.path).name

    def to_record(self) -> dict[str, Any]:
        """The ``flat_field`` of a record's ``corrections`` block (``export_record.analysis_metadata``)."""
        return {
            "file": self.name, "path": self.path, "sha256": self.sha256, "normalisation_mean": float(self.mean),
            "floor": float(self.floor), "masked_pixels": int(self.masked_pixels), "shape": list(self.shape),
            "applied_as": APPLIED_AS, "dark_scale": self.dark_scale, "dark_scale_source": self.dark_scale_source,
            "already_dark_subtracted": self.already_dark_subtracted,
            "exposure_time_s": self.exposure_time_s, "dark_exposure_time_s": self.dark_exposure_time_s,
        }


def read_flat_field(frames, path: str | Path) -> FlatField:
    """The normalised flat field of a file: ``.npy`` with ``numpy.load`` (no pickle), anything else through the
    frame reader ``frames`` (its first frame; the pixels its reader masks are left out). ``ValueError`` or
    ``OSError`` says why it cannot be used."""
    reference = read_reference(frames, file_version(path, 0, frames), "flat")
    return normalise_flat_field(reference.data, reference.valid, name=Path(path).name)


class FlatFieldMixin:
    """Needs ``frames`` (a ``FrameSource``); call ``_init_flat_field()`` in ``__init__``."""

    def _init_flat_field(self) -> None:
        self._flat_cache: Optional[tuple] = None
        self._references: dict[str, ReferenceFrame] = {}

    def _reference(self, kind: str, path: str, index: int = 0) -> ReferenceFrame:
        version = file_version(path, index, self.frames)
        cached = self._references.get(kind)
        if cached is None or cached.version != version:
            cached = read_reference(self.frames, version, kind)
            self._references[kind] = cached
        return cached

    def _flat_field(self, path: str, frame_shape=None, *, corrections=Corrections(), dark=None):
        reference = self._reference("flat", path)
        r, source = dark_scale(reference, dark, corrections.dark_scale, corrections.flat_already_dark_subtracted)
        key = (reference.version, dark.version if dark else None, r, source, corrections.flat_field_floor,
               corrections.counts_per_adu, corrections.offset_adu)
        cache = self._flat_cache  # one tuple: analyses run on several threads
        if cache is None or cache[0] != key:
            raw = detector_counts(reference.data, corrections, exposures=int(not corrections.flat_already_dark_subtracted))
            net, valid = raw, reference.valid
            if r:
                net, valid = subtract_background(raw, valid, detector_counts(dark.data, corrections),
                                                 dark.valid, r, what="dark")
            flat = normalise_flat_field(net, valid, name=Path(path).name, floor=corrections.flat_field_floor,
                                        flat_counts=raw)
            use = FlatFieldUse(str(path), sha256_of(path), flat.mean, flat.masked_pixels, flat.floor, flat.shape,
                               r, source, corrections.flat_already_dark_subtracted, reference.exposure_s,
                               dark.exposure_s if dark is not None else None)
            cache = (key, flat, use)
            self._flat_cache = cache
        if frame_shape is not None:
            check_flat_shape(cache[1], frame_shape, Path(path).name)
        return cache[1], cache[2]

    def check_flat_field_file(self, path: str | Path, frame_shape: Optional[tuple[int, int]] = None,
                              *, corrections=Corrections()) -> FlatFieldUse:
        """What the flat field of ``path`` is (``FlatFieldUse``); ``ValueError`` or ``OSError`` when it cannot be
        read or used, or is not of ``frame_shape`` — asked before it is kept, so a flat field of another detector
        is refused instead of failing every later analysis."""
        shape = None if frame_shape is None else (int(frame_shape[0]), int(frame_shape[1]))
        dark = self._reference("dark", corrections.dark_path, corrections.dark_frame) if corrections.dark_path else None
        return self._flat_field(str(path), shape, corrections=corrections, dark=dark)[1]


__all__ = [
    "APPLIED_AS", "FLAT_SHAPE", "FLOOR", "NOT_2D", "NO_NUMBERS", "NO_RESPONSE", "FLAT_FIELD_TEMPLATES",
    "FlatFieldMixin", "FlatFieldUse", "read_flat_field",
]
