"""Compatibility alias for the feature-owned implementation.

New production code imports the owner under ``src.gimap.features.fitting``.
"""

import sys

from src.gimap.features.fitting.domain.q_space_geometry import *  # noqa: F403
from src.gimap.features.fitting.domain import q_space_geometry as _implementation

sys.modules[__name__] = _implementation
