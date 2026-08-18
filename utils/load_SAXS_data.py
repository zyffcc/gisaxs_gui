"""Compatibility alias for the feature-owned implementation.

New production code imports the owner under ``src.gimap.features.fitting``.
"""

import sys

from src.gimap.features.fitting.infrastructure.adapters.scattering_curve_files import *  # noqa: F403
from src.gimap.features.fitting.infrastructure.adapters import (
    scattering_curve_files as _implementation,
)

sys.modules[__name__] = _implementation
