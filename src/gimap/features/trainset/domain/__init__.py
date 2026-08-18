"""Trainset domain public API。"""

from .geometry import q_vectors, roi_to_spherical_ranges
from .plugins import PluginRegistry, PluginSpec, REGISTRY
from .physical_background import PHYSICAL_BACKGROUND_PARAMETERS
from .model_contract import SUPPORTED_LAYER_TYPES, normalized_layers, static_contract
from .parameters import trainable_parameter_names

__all__ = [
    "PluginRegistry",
    "PHYSICAL_BACKGROUND_PARAMETERS",
    "PluginSpec",
    "REGISTRY",
    "q_vectors",
    "roi_to_spherical_ranges",
    "SUPPORTED_LAYER_TYPES",
    "normalized_layers",
    "static_contract",
    "trainable_parameter_names",
]
