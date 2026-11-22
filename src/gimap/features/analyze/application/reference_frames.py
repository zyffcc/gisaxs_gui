"""Detector references, their file versions and exposure times."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.gimap.shared.detector_io.output_values import floating

from ..domain import valid_pixels
from ..domain.normalisation import EXPOSURE_KEY, header_values
from ..domain.flat_field import NOT_2D
from .reduction_inputs import image_stamp


@dataclass(frozen=True)
class ReferenceFrame:
    data: np.ndarray
    valid: np.ndarray
    exposure_s: float | None
    version: tuple


def file_version(path, index, frames) -> tuple:
    return str(path), int(index), image_stamp(path, frames)


def read_reference(frames, version, kind) -> ReferenceFrame:
    filename, index = version[:2]
    path = Path(filename)
    if kind in ("flat", "sigma") and path.suffix.lower() == ".npy":
        raw, mask, meta = np.load(path, allow_pickle=False), None, {}
        if not np.issubdtype(raw.dtype, np.number):
            if kind == "flat":
                raise ValueError(f"The flat field {path.name} holds no numbers.")
            raise ValueError("The reference frame must contain numeric values.")
        meta["stored_dtype"] = raw.dtype.str
    else:
        image = frames.load(path, index)
        raw, mask, meta = image.data, image.mask, image.metadata or {}
    # A NumPy flat may be double precision. Preserve its mean before casting its response to float32.
    data = np.asarray(raw) if kind == "flat" else np.asarray(raw, dtype=np.float32)
    if data.ndim != 2:
        if kind == "flat":
            raise ValueError(NOT_2D.format(name=path.name, dimensions=data.ndim))
        raise ValueError("The reference frame must be a single 2D image.")
    valid = valid_pixels(data, mask, negatives_valid=floating(meta.get("stored_dtype", "")))
    return ReferenceFrame(data, valid, header_values(meta)[EXPOSURE_KEY], version)


def dark_scale(reference: ReferenceFrame, dark: ReferenceFrame | None, fallback: float,
               already_subtracted: bool, manual: float | None = None) -> tuple[float, str]:
    if already_subtracted:
        return 0.0, "already dark-subtracted"
    if dark is None:
        return 0.0, "no dark frame"
    if manual is not None:
        return manual, "entered"
    if reference.exposure_s is not None and dark.exposure_s is not None:
        return reference.exposure_s / dark.exposure_s, "exposure times"
    return fallback, "sample dark scale"


def detector_counts(raw: np.ndarray, corrections, *, exposures: int = 1) -> np.ndarray:
    """Raw ADU to counts. An already dark-subtracted reference carries zero offsets."""
    if corrections.offset_adu and exposures:
        raw = raw - np.float32(corrections.offset_adu * exposures)
    return raw * np.float32(corrections.counts_per_adu) if corrections.counts_per_adu != 1 else raw
