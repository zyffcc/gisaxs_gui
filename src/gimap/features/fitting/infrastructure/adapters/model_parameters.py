"""Adapter for the persisted fitting model-parameter document."""

from __future__ import annotations

import copy
from pathlib import Path

from config.model_parameters_manager import ModelParametersManager


class FittingModelParametersAdapter:
    def __init__(self, config_path: str | Path | None = None):
        self._manager = ModelParametersManager(
            str(config_path) if config_path is not None else None
        )

    @property
    def config_path(self) -> Path:
        return Path(self._manager.config_file)

    def load(self) -> bool:
        return self._manager.load_parameters()

    def save(self) -> bool:
        return self._manager.save_parameters()

    def section(self, module: str, default=None):
        return self._manager.get_parameter(module, None, default)

    def replace_section(self, module: str, values: dict) -> None:
        self._manager._parameters[module] = copy.deepcopy(values)

    def ensure_particle(self, module, particle_id, shape):
        return self._manager.ensure_particle_entry(module, particle_id, shape)

    def remove_particle(self, module, particle_id):
        return self._manager.remove_particle(module, particle_id)

    def particles(self, module):
        return self._manager.get_all_particles(module)

    def particle_parameter(self, module, particle_id, shape=None, param=None):
        return self._manager.get_particle_parameter(module, particle_id, shape, param)

    def set_particle_parameter(self, module, particle_id, shape, param, value):
        return self._manager.set_particle_parameter(
            module, particle_id, shape, param, value
        )

    def particle_shape(self, module, particle_id):
        return self._manager.get_particle_shape(module, particle_id)

    def set_particle_shape(self, module, particle_id, shape):
        return self._manager.set_particle_shape(module, particle_id, shape)

    def particle_enabled(self, module, particle_id):
        return self._manager.get_particle_enabled(module, particle_id)

    def set_particle_enabled(self, module, particle_id, enabled):
        return self._manager.set_particle_enabled(module, particle_id, enabled)

    def global_parameter(self, module, name, default=None):
        return self._manager.get_global_parameter(module, name, default)

    def set_global_parameter(self, module, name, value):
        return self._manager.set_global_parameter(module, name, value)

    def global_parameters(self, module):
        return self._manager.get_all_global_parameters(module)
