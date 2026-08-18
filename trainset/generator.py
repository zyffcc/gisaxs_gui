"""Legacy module alias for the feature-owned dataset generator."""

import sys

from src.gimap.features.trainset.infrastructure.adapters import dataset_generator as _implementation

sys.modules[__name__] = _implementation
