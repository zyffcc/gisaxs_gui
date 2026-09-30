"""Write one detector frame to a file other programs read (used by Format Converter and Batch Export).

The values are written as they are (no rescaling); the format decides only the container:

* ``tiff`` — ``.tif`` through fabio (32-bit float stays float, integers stay integers);
* ``edf`` — ``.edf`` (ESRF data format) through fabio;
* ``cbf`` — ``.cbf`` through fabio; non-integer data become float32 with NaN/inf as 0;
* ``npy`` — ``.npy`` (NumPy, no pickle);
* ``hdf5`` — ``.h5`` with the frame at ``/entry/data/data`` (NeXus-style, gzip).

``metadata`` (a JSON-able dict) goes into the header when the format has one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

FRAME_FORMATS = {"tiff": ".tif", "edf": ".edf", "cbf": ".cbf", "npy": ".npy", "hdf5": ".h5"}


def frame_suffix(fmt: str) -> str:
    try:
        return FRAME_FORMATS[str(fmt).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported frame format: {fmt}") from exc


def write_frame(path: str | Path, data: np.ndarray, fmt: str, metadata: Optional[Mapping[str, Any]] = None) -> Path:
    """Write ``data`` (2-D) to ``path`` in ``fmt`` (see the module docstring); returns the path."""
    fmt = str(fmt).lower()
    frame_suffix(fmt)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data)
    header = {"GIMaP_metadata": json.dumps(dict(metadata), ensure_ascii=False)} if metadata else {}
    if fmt == "npy":
        np.save(str(path), data, allow_pickle=False)
    elif fmt == "hdf5":
        import h5py

        with h5py.File(str(path), "w") as handle:
            entry = handle.create_group("entry")
            entry.attrs["NX_class"] = "NXentry"
            dataset = entry.create_dataset("data/data", data=data, compression="gzip", shuffle=True)
            if metadata:
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
    return path


__all__ = ["FRAME_FORMATS", "frame_suffix", "write_frame"]
