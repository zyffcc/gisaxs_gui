"""Compatibility alias for the feature-owned implementation.

New production code imports the owner under ``src.gimap.features.fitting``.
"""

import sys

from src.gimap.features.fitting.infrastructure.adapters.fitting_pipeline import *  # noqa: F403
from src.gimap.features.fitting.infrastructure.adapters import fitting_pipeline as _implementation

sys.modules[__name__] = _implementation
