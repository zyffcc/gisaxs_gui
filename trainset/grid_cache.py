"""Legacy module alias preserving monkeypatch behavior for the grid cache."""

import sys

from src.gimap.features.trainset.infrastructure.adapters import grid_cache as _implementation

sys.modules[__name__] = _implementation
