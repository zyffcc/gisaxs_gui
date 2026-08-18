"""Legacy module alias for trainset simulation orchestration."""

import sys

from src.gimap.features.trainset.application import simulation as _implementation

sys.modules[__name__] = _implementation
