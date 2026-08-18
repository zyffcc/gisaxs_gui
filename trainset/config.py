"""Legacy module alias for feature-owned trainset configuration."""

import sys

from src.gimap.features.trainset.infrastructure.adapters import configuration as _implementation

sys.modules[__name__] = _implementation
