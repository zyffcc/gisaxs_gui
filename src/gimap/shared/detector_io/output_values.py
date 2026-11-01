"""The values a written detector frame holds, so that its pixels without data read back as such (and its values
as they were — except a floating-point frame written as CBF, see below).

GIMaP reads a pixel as holding no data when it is not finite, when the reader of the file masks it (a NeXus
``pixel_mask``, a part of a stitched canvas no module covers, the "no data" value of unsigned NeXus data, an
EDF ``Dummy``) or — in integer data — when it is negative: the gap and bad-pixel codes of counting detectors
(Pilatus −1, −2). ``invalid_pixels`` is that rule; Analyze's ``valid_pixels`` applies the same one.

The readers hand every frame over as 32-bit float. Written as it is read, a Pilatus frame would become a float
file: its gap codes would then be negative *data*, and the frame no longer counts. ``output_frame`` writes the
frame in the type it was stored in instead, so it reads back as the source did:

* **Integer frames** keep their stored type and every value; a stored negative code stays as it is. A pixel
  without data that has no stored code (the reader made it NaN: a NeXus ``pixel_mask`` pixel, a gap of a
  stitched canvas) is written as −1. An unsigned frame that needs such a code is written in the smallest signed
  type of at least its size that holds every value (uint16 → int16 or int32), except in HDF5: there it keeps
  its type and the largest value of the type marks the pixel (the NeXus convention). Values a type cannot hold
  are written as int64.
* **Floating-point frames** — and integer frames whose values are no longer whole numbers — are written as
  they are, with NaN on every pixel without data. CBF holds integers only: there the values are truncated to
  int32 (as fabio does) and a pixel without data is −1. Such a CBF keeps neither fractional parts nor negative
  data: a value of −1 or less reads back as a gap code (no data). The record counts both
  (``negative_data_read_as_gap_codes``, ``values_truncated``) and the pixels that read back without data
  (``pixels_read_back_without_data``).
* A **type the person chose** (Format Converter: 32-bit float, uint16): float as above; an integer type with
  −1 (signed) or its largest value (unsigned) on the pixels without data.
* An **HDF5** file also marks every pixel without data in the NeXus ``pixel_mask`` (bit 0, "gap": no data) at
  ``/entry/instrument/detector/pixel_mask`` (``writing.write_frame``).

``OutputFrame.record`` says what was written, for the metadata that goes with the frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

PIXEL_MASK_PATH = "/entry/instrument/detector/pixel_mask"
NO_DATA_BIT = 1
"""NeXus ``pixel_mask`` bit 0 ("gap"): the pixel holds no data. GIMaP and other NeXus readers reject it."""
NO_DATA_CODE = -1
INVALID_RULE = (
    "a pixel holds no data when it is not finite, when the reader of the source masked it, "
    "or, in integer data, when it is negative (detector gap codes)"
)
_SIGNED = (np.dtype(np.int16), np.dtype(np.int32), np.dtype(np.int64))  # (no int8: few readers take it)


def floating(stored_dtype: Any) -> bool:
    """A floating-point frame (dark- or background-subtracted): its negative values are data (as in Analyze).

    Decided by the type's kind, so a big-endian float (``>f4``, as h5py and fabio report it) counts as float;
    an unknown type (``""``) is read like integer counts."""
    name = str(stored_dtype or "").strip()
    if not name:
        return False
    try:
        return np.dtype(name).kind == "f"
    except (TypeError, ValueError):
        return name.startswith("float")


def invalid_pixels(data: Any, mask: Any = None, stored_dtype: Any = None) -> np.ndarray:
    """The pixels GIMaP reads as holding no data (see the module docstring).

    ``stored_dtype``: the type the frame was stored in (``metadata["stored_dtype"]``); ``None`` takes the type
    of ``data``. An unknown type (``""``) is read like integer counts, as Analyze reads it.
    """
    values = np.asarray(data)
    invalid = ~np.isfinite(values) if values.dtype.kind in "fc" else np.zeros(values.shape, dtype=bool)
    if not floating(values.dtype.name if stored_dtype is None else stored_dtype):
        with np.errstate(invalid="ignore"):
            invalid |= values < 0
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != values.shape:
            raise ValueError(f"Mask shape {mask.shape} does not match the frame {values.shape}.")
        invalid |= mask
    return invalid


@dataclass(frozen=True)
class OutputFrame:
    data: np.ndarray
    """The values to write."""
    invalid: np.ndarray
    """The pixels without data (for the HDF5 ``pixel_mask``)."""
    record: dict[str, Any] = field(default_factory=dict)
    """What was written: ``stored_dtype``, ``written_dtype``, ``pixels_without_data`` and how they are written."""


def _dtype(name: Any) -> Optional[np.dtype]:
    if not name:
        return None
    try:
        return np.dtype(str(name)).newbyteorder("=")  # a big-endian source is written in the native order
    except TypeError:
        return None


def _fits(dtype: np.dtype, low: float, high: float) -> bool:
    info = np.iinfo(dtype)
    return info.min <= low and high <= info.max


def _codes(values: np.ndarray, where: np.ndarray) -> str:
    found = np.unique(values[where])
    shown = ", ".join(str(int(value)) for value in found[:8])
    return shown + (", …" if found.size > 8 else "")


def _as_integers(values: np.ndarray, invalid: np.ndarray, stored: np.dtype, fmt: str):
    """``(array, how the pixels without data are written)`` in an integer type; ``None`` when the values are not
    whole numbers (a scaled or background-subtracted frame stays float)."""
    finite = np.isfinite(values) if values.dtype.kind in "fc" else np.ones(values.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        stored_code = invalid & finite & (values < 0) if stored.kind == "i" else np.zeros(values.shape, dtype=bool)
    kept = ~invalid | stored_code  # the values written as they are (``invalid`` holds every non-finite pixel)
    sample = values[kept]
    if values.dtype.kind in "fc" and sample.size and not np.array_equal(sample, np.trunc(sample)):
        return None
    low, high = (float(sample.min()), float(sample.max())) if sample.size else (0.0, 0.0)
    coded = invalid & ~kept  # pixels without data and without a stored code
    needs_code = bool(coded.any())
    code: Optional[int] = None
    if stored.kind == "i":
        target, code = stored, NO_DATA_CODE
    elif not needs_code and not (fmt == "hdf5" and high >= np.iinfo(stored).max):
        target = stored  # unsigned, nothing to mark (in HDF5 the largest value would read as "no data")
    elif fmt == "hdf5" and high < np.iinfo(stored).max:
        target, code = stored, int(np.iinfo(stored).max)
    else:
        code = NO_DATA_CODE
        target = next((kind for kind in _SIGNED if kind.itemsize >= stored.itemsize and _fits(kind, -1, high)),
                      np.dtype(np.int64))
    if not _fits(target, min(low, code if needs_code and code is not None else low), high):
        target, code = np.dtype(np.int64), NO_DATA_CODE
        if not _fits(target, min(low, -1.0), high):
            return None
    out = np.where(kept, values, 0).astype(target)
    if needs_code:
        out[coded] = code
    parts = []
    if stored_code.any():
        parts.append(f"their stored codes ({_codes(values, stored_code)})")
    if needs_code:
        what = f"{code} (the largest {target.name}: no data)" if code is not None and code > 0 else str(code)
        parts.append(f"{what}{' where there is no stored code' if parts else ''}")
    return out, "; ".join(parts)


def _as_floats(values: np.ndarray, invalid: np.ndarray, fmt: str):
    if fmt == "cbf":  # CBF holds integers: fabio would truncate the values to int32 too
        info = np.iinfo(np.int32)
        whole = np.clip(np.trunc(np.where(invalid | ~np.isfinite(values), NO_DATA_CODE, values)), info.min, info.max)
        return whole.astype(np.int32), f"{NO_DATA_CODE} (CBF holds integers)"
    out = np.array(values, dtype=values.dtype if values.dtype.kind == "f" else np.float32, copy=True)
    out[invalid] = np.nan
    return out, "NaN"


def _as_chosen(values: np.ndarray, invalid: np.ndarray, chosen: np.dtype, fmt: str):
    """A type the person chose: a float type as any float frame; an integer type (``values`` already in its
    range, e.g. scaled to uint16) with −1 (signed) or its largest value (unsigned: "no data", as NeXus writes it)
    on the pixels without data."""
    if chosen.kind == "f":
        return _as_floats(values.astype(chosen, copy=False), invalid, fmt)
    code = NO_DATA_CODE if chosen.kind == "i" else int(np.iinfo(chosen).max)
    out = np.where(invalid, code, values).astype(chosen)
    return out, str(code) if code < 0 else f"{code} (the largest {chosen.name}: no data)"


def output_frame(data: Any, fmt: str, *, invalid: Any = None, stored_dtype: Any = None,
                 chosen_dtype: Any = None) -> OutputFrame:
    """The values that write ``data`` in ``fmt`` (``tiff``, ``edf``, ``cbf``, ``npy``, ``hdf5``) so that GIMaP
    reads the same pixels as without data, and the same values and type (counts stay counts). The exception is
    a floating-point frame in CBF (integers only): fractional parts are cut and negative data read back as gap
    codes; the record counts them.

    ``invalid``: the pixels without data as read (``invalid_pixels`` of the frame when not given);
    ``stored_dtype``: the type the source stored the frame in (``None``: the type of ``data``);
    ``chosen_dtype``: a type the person chose instead of the stored one (``float32``; ``uint16`` for ``data``
    already converted to it).
    """
    values = np.asarray(data)
    fmt = str(fmt).lower()
    stored_name = values.dtype.name if stored_dtype is None else str(stored_dtype or "")
    if invalid is None:
        invalid = invalid_pixels(values, None, stored_name)
    invalid = np.asarray(invalid, dtype=bool)
    if invalid.shape != values.shape:
        raise ValueError(f"Mask shape {invalid.shape} does not match the frame {values.shape}.")
    if values.dtype.kind in "fc":
        invalid = invalid | ~np.isfinite(values)  # a value that is not a number is never data
    stored, chosen = _dtype(stored_name), _dtype(chosen_dtype)
    if chosen is not None:
        out, how = _as_chosen(values, invalid, chosen, fmt)
        as_float = chosen.kind == "f"
    else:
        written = _as_integers(values, invalid, stored, fmt) if stored is not None and stored.kind in "iu" else None
        out, how = written if written is not None else _as_floats(values, invalid, fmt)
        as_float = written is None
    count = int(np.count_nonzero(invalid))
    record: dict[str, Any] = {
        "stored_dtype": stored_name or "unknown",
        "written_dtype": out.dtype.name,
        "pixels_without_data": count,
        "pixels_without_data_written_as": how if count else "none",
        "rule": INVALID_RULE,
    }
    if fmt == "hdf5" and count:
        record["pixel_mask"] = f"{PIXEL_MASK_PATH} (NeXus bit 0, gap: no data)"
    if fmt == "cbf" and as_float:
        record.update(_cbf_losses(values, invalid, out, count))
    return OutputFrame(out, invalid, record)


def _cbf_losses(values: np.ndarray, invalid: np.ndarray, out: np.ndarray, count: int) -> dict[str, int]:
    """What a floating-point frame loses in CBF (int32): data ≤ −1 read back as gap codes (no data), and values
    whose fractional part was cut or that were clipped to the int32 range."""
    data = ~invalid
    gaps = data & (out < 0)
    with np.errstate(invalid="ignore"):
        cut = data & ~gaps & (out != values)
    losses = {"pixels_read_back_without_data": count + int(np.count_nonzero(gaps))}
    if gaps.any():
        losses["negative_data_read_as_gap_codes"] = int(np.count_nonzero(gaps))
    if cut.any():
        losses["values_truncated"] = int(np.count_nonzero(cut))
    return losses


__all__ = [
    "INVALID_RULE", "NO_DATA_BIT", "NO_DATA_CODE", "OutputFrame", "PIXEL_MASK_PATH", "floating", "invalid_pixels",
    "output_frame",
]
