"""Write one detector frame to a file other programs read (used by Format Converter and Batch Export).

The values are written as they are (no rescaling); the format decides only the container:

* ``tiff`` — ``.tif`` through fabio (32-bit float stays float, integers stay integers);
* ``edf`` — ``.edf`` (ESRF data format) through fabio;
* ``cbf`` — ``.cbf`` through fabio; non-integer data become float32 with NaN/inf as 0;
* ``npy`` — ``.npy`` (NumPy, no pickle);
* ``hdf5`` — ``.h5`` with the frame at ``/entry/data/data`` (NeXus: an NXdata group whose signal it is; gzip).

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

FRAME_FORMATS = {"tiff": ".tif", "edf": ".edf", "cbf": ".cbf", "npy": ".npy", "hdf5": ".h5"}


def frame_suffix(fmt: str) -> str:
    try:
        return FRAME_FORMATS[str(fmt).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported frame format: {fmt}") from exc


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


def write_frame(path: str | Path, data: np.ndarray, fmt: str, metadata: Optional[Mapping[str, Any]] = None) -> Path:
    """Write ``data`` (2-D) to ``path`` in ``fmt`` (see the module docstring); returns the path."""
    fmt = str(fmt).lower()
    frame_suffix(fmt)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    # HDF5 attributes hold any text; the headers fabio writes hold ASCII (see the module docstring).
    header = {"GIMaP_metadata": json.dumps(dict(metadata), ensure_ascii=fmt != "hdf5")} if metadata else {}
    was_there = path.exists()
    target = _ascii_stand_in(path) if fmt == "cbf" else path
    try:
        _write(target, data, fmt, header, bool(metadata))
        if target != path:
            os.replace(target, path)
    except Exception:
        _discard(path, was_there)
        if target != path:
            _discard(target, False)
        raise
    return path


def _write(path: Path, data: np.ndarray, fmt: str, header: dict, keep: bool) -> None:
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
            cbf_data = np.nan_to_num(cbf_data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        CbfImage(data=cbf_data, header=header).write(str(path))


__all__ = ["FRAME_FORMATS", "frame_suffix", "write_frame"]
