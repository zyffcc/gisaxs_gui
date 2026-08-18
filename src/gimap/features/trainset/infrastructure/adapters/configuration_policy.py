"""Existing Trainset configuration policy adapter."""

from __future__ import annotations

from .configuration import (
    default_project_config,
    merge_config,
    synchronize_parameter_specs,
    validate_project_config,
)


class TrainsetConfigurationAdapter:
    def default(self):
        return default_project_config()

    def merge(self, base, override):
        return merge_config(base, override)

    def synchronize(self, config):
        return synchronize_parameter_specs(config)

    def validate(self, config, **options):
        return validate_project_config(config, **options)
