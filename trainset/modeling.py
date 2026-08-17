"""Legacy compatibility for trainset Keras modeling adapter。"""

from src.gimap.features.trainset.infrastructure.adapters.keras_modeling import (
    build_keras_model,
    build_optimizer,
    normalized_layers,
    resolve_keras_api,
    static_contract,
)

__all__ = [
    "build_keras_model",
    "build_optimizer",
    "normalized_layers",
    "resolve_keras_api",
    "static_contract",
]
