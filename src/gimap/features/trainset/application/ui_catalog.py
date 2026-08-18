"""Stable Trainset configuration metadata exposed to presentation."""

from __future__ import annotations

from ..domain import (
    PHYSICAL_BACKGROUND_PARAMETERS,
    normalized_layers,
    trainable_parameter_names,
)
from ..domain.plugins import REGISTRY


class TrainsetUiCatalog:
    def background_parameters(self):
        return tuple(PHYSICAL_BACKGROUND_PARAMETERS)

    def plugins(self, kind: str):
        return tuple(REGISTRY.list(kind))

    def plugin(self, kind: str, key: str):
        return REGISTRY.get(kind, key)

    def normalized_layers(self, model_config):
        return normalized_layers(model_config)

    def trainable_names(self, config):
        return trainable_parameter_names(config)
