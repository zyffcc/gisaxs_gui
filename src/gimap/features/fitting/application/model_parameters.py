"""Application-owned access to persisted fitting model parameters."""

from __future__ import annotations

import copy

from .ports import FittingModelParametersPort


class ManageFittingModelParameters:
    """Expose the legacy API while keeping its storage implementation outside Qt views."""

    def __init__(self, repository: FittingModelParametersPort):
        self._repository = repository

    @property
    def config_file(self) -> str:
        return str(self._repository.config_path)

    def load_parameters(self) -> bool:
        return self._repository.load()

    def save_parameters(self) -> bool:
        return self._repository.save()

    def get_parameter(self, section, key=None, default=None):
        values = self._repository.section(section, default)
        if key is None or not isinstance(values, dict):
            return values
        return values.get(key, default)

    def replace_section(self, section: str, values: dict) -> None:
        self._repository.replace_section(section, copy.deepcopy(values))

    def ensure_particle_entry(self, module, particle_id, shape="None"):
        return self._repository.ensure_particle(module, particle_id, shape)

    def remove_particle(self, module, particle_id) -> bool:
        return self._repository.remove_particle(module, particle_id)

    def get_all_particles(self, module):
        return self._repository.particles(module)

    def get_particle_parameter(self, module, particle_id, shape=None, param=None):
        return self._repository.particle_parameter(module, particle_id, shape, param)

    def set_particle_parameter(self, module, particle_id, shape, param, value):
        return self._repository.set_particle_parameter(
            module, particle_id, shape, param, value
        )

    def get_particle_shape(self, module, particle_id):
        return self._repository.particle_shape(module, particle_id)

    def set_particle_shape(self, module, particle_id, shape):
        return self._repository.set_particle_shape(module, particle_id, shape)

    def get_particle_enabled(self, module, particle_id):
        return self._repository.particle_enabled(module, particle_id)

    def set_particle_enabled(self, module, particle_id, enabled):
        return self._repository.set_particle_enabled(module, particle_id, enabled)

    def get_global_parameter(self, module, name, default=None):
        return self._repository.global_parameter(module, name, default)

    def set_global_parameter(self, module, name, value):
        return self._repository.set_global_parameter(module, name, value)

    def get_all_global_parameters(self, module):
        return self._repository.global_parameters(module)
