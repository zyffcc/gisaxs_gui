"""Legacy compatibility for trainset domain plugin definitions。"""

from src.gimap.features.trainset.domain.plugins import (
    PluginRegistry,
    PluginSpec,
    REGISTRY,
)

__all__ = ["PluginRegistry", "PluginSpec", "REGISTRY"]
