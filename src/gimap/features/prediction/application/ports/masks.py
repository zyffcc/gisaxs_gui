"""Prediction preprocessing-mask storage port."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class PredictionMaskRepository(Protocol):
    def load(self, path: Path) -> np.ndarray: ...
