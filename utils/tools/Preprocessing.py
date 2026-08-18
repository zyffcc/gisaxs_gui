"""Compatibility alias for prediction-owned image preprocessing."""

import sys

from src.gimap.features.prediction.infrastructure.adapters import image_preprocessing as _implementation

sys.modules[__name__] = _implementation
