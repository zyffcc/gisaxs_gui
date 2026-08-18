"""Legacy module alias for portable trainset job packaging."""

import sys

from src.gimap.features.trainset.infrastructure.adapters import portable_job_package as _implementation

sys.modules[__name__] = _implementation
