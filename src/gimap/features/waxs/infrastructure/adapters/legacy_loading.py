"""Compatibility loading API formerly implemented by the legacy Qt page."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def detect_nxs_frame_count(file_path: str) -> int:
    from src.gimap.shared.detector_io import detect_nxs_frame_count as detect

    return int(detect(file_path))


def load_image_matrix(
    file_path: str,
    frame_idx: int = 0,
    dataset_path: str = "/entry/instrument/detector/data",
    dist_path: str = "/entry/instrument/detector/translation/distance",
    mask_path: str = "/entry/instrument/detector/pixel_mask",
) -> np.ndarray:
    """Load detector data while preserving the historical WAXS orientation."""

    del dist_path, mask_path
    from src.gimap.shared.detector_io import load_detector_image

    return load_detector_image(
        file_path, frame_idx=frame_idx, dataset_path=dataset_path
    ).data


def load_tiff_matrix(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        with Image.open(path) as image:
            arr = np.asarray(image)
    except Exception:
        import matplotlib.pyplot as plt

        arr = plt.imread(str(path))
    if arr.ndim == 3:
        arr = np.mean(arr[..., :3], axis=2)
    return np.asarray(arr, dtype=np.float32)
