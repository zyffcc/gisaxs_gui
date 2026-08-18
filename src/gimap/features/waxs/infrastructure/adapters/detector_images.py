"""Calibration detector loader backed WAXS image repository。"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class CalibrationWaxsImageRepository:
    def frame_count(self, path: Path) -> int:
        from src.gimap.shared.detector_io import detect_nxs_frame_count

        return int(detect_nxs_frame_count(str(path)))

    def load_frame(self, path: Path, frame_index: int) -> np.ndarray:
        from src.gimap.shared.detector_io import load_detector_image

        loaded = load_detector_image(path, frame_idx=frame_index)
        return np.asarray(loaded.data, dtype=np.float32)
