"""Module-defined preprocessing port。"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np

from ...domain import PredictionModule
from ..models import LoadedPredictionImage, PreprocessedPredictionInput


class Preprocessor(Protocol):
    def preprocess(
        self, image: np.ndarray, module: PredictionModule, context: Mapping[str, Any] | None = None
    ) -> PreprocessedPredictionInput:
        """``context``: what is known about the frames (``FrameContexts``), for a module that asks for it."""
        ...


class FrameContexts(Protocol):
    def describe(self, loaded: LoadedPredictionImage) -> Mapping[str, Any]:
        """Files, format, shape, detector, the frame's facts and, when known, its geometry and ``q_map()``."""
        ...
