from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable


@dataclass(frozen=True)
class PluginSpec:
    key: str
    label: str
    category: str
    description: str
    defaults: Dict[str, Any] = field(default_factory=dict)


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: Dict[str, Dict[str, PluginSpec]] = {}

    def register(self, spec: PluginSpec) -> None:
        self._plugins.setdefault(spec.category, {})[spec.key] = spec

    def get(self, category: str, key: str) -> PluginSpec:
        return self._plugins[category][key]

    def list(self, category: str) -> Iterable[PluginSpec]:
        return tuple(self._plugins.get(category, {}).values())


REGISTRY = PluginRegistry()

for item in (
    PluginSpec("sphere", "Sphere", "particle", "Homogeneous sphere form factor."),
    PluginSpec("spherical_segment", "Spherical segment", "particle", "Truncated sphere with independent radius and height."),
    PluginSpec("cylinder", "Cylinder", "particle", "Cylinder with radius, height and orientation."),
    PluginSpec("box", "Box", "particle", "Rectangular cuboid particle."),
    PluginSpec("none", "None", "interference", "No interference function."),
    PluginSpec("paracrystal", "Paracrystal", "interference", "Paracrystalline interference function.", {"domain_size_nm": 20.0}),
    PluginSpec("noise", "Noise", "preprocessing", "Random detector/noise model."),
    PluginSpec("mask", "Mask", "preprocessing", "Fixed or randomized detector mask."),
    PluginSpec("log", "Log transform", "preprocessing", "Numerically stable logarithmic intensity transform."),
    PluginSpec("normalize", "Normalization", "preprocessing", "Upper, lower, or range normalization."),
    PluginSpec("random_edge_crop", "Random edge crop", "preprocessing", "Randomly remove edge pixels for augmentation."),
):
    REGISTRY.register(item)
