"""Compose focused particle state bindings."""

from .particle_model_state import ParticleModelStateMixin
from .fitting_parameter_persistence import FittingParameterPersistenceMixin


class ParticleStateMixin(ParticleModelStateMixin, FittingParameterPersistenceMixin):
    """Compatibility composition for focused particle state bindings."""

    pass
