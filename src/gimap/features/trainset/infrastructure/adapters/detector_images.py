"""Focused Trainset detector-data and generation behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

SHARED_READER_SUFFIXES = frozenset({".cbf", ".nxs", ".nx", ".nx5", ".h5", ".hdf5", ".hdf"})
"""Read by the shared detector loader, like everywhere else in GIMaP (every NeXus / HDF5 suffix)."""


def load_scattering_image(path: str | Path) -> np.ndarray:
    """The first frame of a detector file as a float32 image (masked pixels are NaN for NeXus / HDF5)."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in SHARED_READER_SUFFIXES:
        # Reuse the same loader as GISAXS Image Input/calibration.  In
        # particular, P03 NXS modules require stitching, a canvas transpose and
        # one vertical flip; bypassing that path made the TrainSet ROI disagree
        # with the image users selected elsewhere in the GUI.  It also picks the
        # image dataset of an HDF5 file (never a pixel mask or a flat field) and
        # reads one frame of a stack, not the stack.
        from src.gimap.shared.detector_io import load_detector_image

        data = np.asarray(load_detector_image(source).data)
    elif suffix == ".npy":
        data = np.load(source)
    elif suffix == ".npz":
        archive = np.load(source)
        data = np.asarray(archive[archive.files[0]])
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
