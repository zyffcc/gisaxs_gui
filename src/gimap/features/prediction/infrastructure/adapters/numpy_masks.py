"""NumPy mask adapter used by prediction module validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class NumpyPredictionMaskRepository:
    def load(self, path: Path) -> np.ndarray:
        source = Path(path)
        if not source.is_file() or source.suffix.casefold() != ".npy":
            raise ValueError("Mask file found but unsupported format (only .npy)")
        return np.load(source)
