"""Write one detector frame to a file other programs read (used by Format Converter and Batch Export).

The values are written as they are given (no rescaling); the format decides only the container:

* ``tiff`` — ``.tif`` through fabio (32-bit float stays float, integers stay integers);
* ``edf`` — ``.edf`` (ESRF data format) through fabio;
* ``cbf`` — ``.cbf`` through fabio; non-integer data become int32 as fabio writes them, NaN/inf as −1;
* ``npy`` — ``.npy`` (NumPy, no pickle);
* ``hdf5`` — ``.h5`` with the frame at ``/entry/data/data`` (NeXus: an NXdata group whose signal it is; gzip).

Which values to give: ``output_values.output_frame`` turns a frame as a reader returns it (32-bit float) back
into the type it was stored in, with its pixels without data marked so that they read back as such; its
``invalid`` goes to ``write_frame`` too, and an HDF5 file then marks those pixels in the NeXus
``pixel_mask`` (bit 0) of ``/entry/instrument/detector``.

``metadata`` (a JSON-able dict) goes into the header when the format has one.  The header of a CBF, EDF or
TIFF file holds ASCII only, and a record does not (a path in any script, "1 file · 4 frames"): there it is
written as ASCII JSON (``\\uXXXX`` escapes), which ``json.loads`` reads back exactly.  A CBF file also
carries the stem of its name in its text; one named outside ASCII is written under an ASCII name and renamed.
A write that fails leaves no empty file behind.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from .output_values import NO_DATA_BIT, NO_DATA_CODE

FRAME_FORMATS = {"tiff": ".tif", "edf": ".edf", "cbf": ".cbf", "npy": ".npy", "hdf5": ".h5"}


def frame_suffix(fmt: str) -> str:
    try:
        return FRAME_FORMATS[str(fmt).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported frame format: {fmt}") from exc


def nexus_detector(entry: Any) -> Any:
    """The NXdetector group ``instrument/detector`` of the HDF5 group ``entry`` (made when missing): where a
    NeXus reader looks for the ``pixel_mask`` of the data of the entry."""
    instrument = entry.require_group("instrument")
    instrument.attrs["NX_class"] = "NXinstrument"
    detector = instrument.require_group("detector")
    detector.attrs["NX_class"] = "NXdetector"
    return detector


def pixel_mask_values(invalid: np.ndarray) -> np.ndarray:
    """The NeXus ``pixel_mask`` bit field of ``invalid``: bit 0 ("gap": no data) on every pixel without data."""
    return np.where(np.asarray(invalid, dtype=bool), NO_DATA_BIT, 0).astype(np.uint32)


def _ascii_stand_in(path: Path) -> Path:
    """A name in the same folder that holds ASCII only (``path`` itself when it does): what is not ASCII in
    the stem becomes ``_``, and a few random characters keep it from being the name of another file."""
    if path.stem.isascii():
        return path
    stem = "".join(character if character.isascii() else "_" for character in path.stem)
    return path.with_name(f"{stem}.{uuid.uuid4().hex[:8]}.tmp{path.suffix}")


def _discard(path: Path, was_there: bool) -> None:
    """After a failed write: the file it created (or emptied) looks like a frame and cannot be read."""
    try:
        if path.is_file() and (not was_there or path.stat().st_size == 0):
            path.unlink()
    except OSError:
        pass


def write_frame(
    path: str | Path,
    data: np.ndarray,
    fmt: str,
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    invalid: Optional[np.ndarray] = None,
) -> Path:
    """Write ``data`` (2-D) to ``path`` in ``fmt`` (see the module docstring); returns the path.

    ``invalid``: the pixels without data (``output_frame(...).invalid``); an HDF5 file marks them in its
    ``pixel_mask``. The other formats have no mask: their values say it (see ``output_values``).
    """
    fmt = str(fmt).lower()
    frame_suffix(fmt)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    if invalid is not None:
        invalid = np.asarray(invalid, dtype=bool)
        if invalid.shape != data.shape:
            raise ValueError(f"Mask shape {invalid.shape} does not match the frame {data.shape}.")
    # HDF5 attributes hold any text; the headers fabio writes hold ASCII (see the module docstring).
    header = {"GIMaP_metadata": json.dumps(dict(metadata), ensure_ascii=fmt != "hdf5")} if metadata else {}
    was_there = path.exists()
    target = _ascii_stand_in(path) if fmt == "cbf" else path
    try:
        _write(target, data, fmt, header, bool(metadata), invalid)
        if target != path:
            os.replace(target, path)
    except Exception:
        _discard(path, was_there)
        if target != path:
            _discard(target, False)
        raise
    return path


def _write(path: Path, data: np.ndarray, fmt: str, header: dict, keep: bool, invalid: Optional[np.ndarray]) -> None:
    if fmt == "npy":
        np.save(str(path), data, allow_pickle=False)
    elif fmt == "hdf5":
        import h5py

        with h5py.File(str(path), "w") as handle:
            entry = handle.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            dataset = entry.create_dataset("data/data", data=data, compression="gzip", shuffle=True)
            # NeXus says which dataset is the image: a reader need not guess it from the size of the frame.
            entry["data"].attrs["NX_class"], entry["data"].attrs["signal"] = "NXdata", "data"
            if keep:
                dataset.attrs["metadata_json"] = header["GIMaP_metadata"]
            if invalid is not None and invalid.any():
                nexus_detector(entry).create_dataset(
                    "pixel_mask", data=pixel_mask_values(invalid), compression="gzip", shuffle=True
                )
    elif fmt == "tiff":
        from fabio.tifimage import TifImage

        TifImage(data=data, header=header).write(str(path))
    elif fmt == "edf":
        from fabio.edfimage import EdfImage

        EdfImage(data=data, header=header).write(str(path))
    else:  # cbf
        from fabio.cbfimage import CbfImage

        cbf_data = data
        if not np.issubdtype(cbf_data.dtype, np.integer):
            # CBF holds integers (fabio truncates to int32): a value that is not a number would become a count.
            cbf_data = np.where(np.isfinite(cbf_data), cbf_data, NO_DATA_CODE).astype(np.float32)
        CbfImage(data=cbf_data, header=header).write(str(path))


__all__ = ["FRAME_FORMATS", "frame_suffix", "nexus_detector", "pixel_mask_values", "write_frame"]
