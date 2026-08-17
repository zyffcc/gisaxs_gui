"""Module-defined preprocessing port。"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ...domain import PredictionModule
from ..models import PreprocessedPredictionInput


class Preprocessor(Protocol):
    def preprocess(
        self, image: np.ndarray, module: PredictionModule
    ) -> PreprocessedPredictionInput: ...
