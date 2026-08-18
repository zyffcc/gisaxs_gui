"""Legacy module alias for trainset execution adapters."""

import sys

from src.gimap.features.trainset.infrastructure.adapters import job_backends as _implementation

sys.modules[__name__] = _implementation
