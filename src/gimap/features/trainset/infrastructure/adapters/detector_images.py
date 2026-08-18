"""Focused Trainset detector-data and generation behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import h5py
import numpy as np


def _largest_2d_hdf5_dataset(handle: h5py.File) -> np.ndarray:
    candidates: List[tuple[int, str]] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            candidates.append((int(np.prod(obj.shape[-2:])), name))

    handle.visititems(visitor)
    if not candidates:
        raise ValueError("No 2D dataset was found in the HDF5/Nexus file.")
    _, name = max(candidates)
    data = np.asarray(handle[name])
    while data.ndim > 2:
        data = data[0]
    return data


def load_scattering_image(path: str | Path) -> np.ndarray:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".nxs", ".cbf"}:
        # Reuse the same loader as GISAXS Image Input/calibration.  In
        # particular, P03 NXS modules require stitching, a canvas transpose and
        # one vertical flip; bypassing that path made the TrainSet ROI disagree
        # with the image users selected elsewhere in the GUI.
        from src.gimap.shared.detector_io import load_detector_image

        data = np.asarray(load_detector_image(source).data)
    elif suffix == ".npy":
        data = np.load(source)
    elif suffix == ".npz":
        archive = np.load(source)
        data = np.asarray(archive[archive.files[0]])
    elif suffix in {".h5", ".hdf5"}:
        with h5py.File(source, "r") as handle:
            data = _largest_2d_hdf5_dataset(handle)
    elif suffix == ".edf":
        import fabio

        data = np.asarray(fabio.open(str(source)).data)
    else:
        data = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if data is None:
            raise ValueError(f"Unsupported or unreadable scattering file: {source}")
        if data.ndim == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    data = np.squeeze(np.asarray(data, dtype=np.float32))
    if data.ndim != 2:
        raise ValueError(f"Expected a 2D detector image, got shape {data.shape}.")
    return data


def crop_roi(image: np.ndarray, roi: Dict[str, Any]) -> np.ndarray:
    x, y = int(roi["x"]), int(roi["y"])
    width, height = int(roi["width"]), int(roi["height"])
    return np.asarray(image[y : y + height, x : x + width])
