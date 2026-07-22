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
    parameters: tuple[Dict[str, Any], ...] = ()


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
    PluginSpec(
        "sphere",
        "Sphere",
        "particle",
        "Homogeneous sphere form factor.",
        parameters=(
            {"key": "radius_nm", "label": "Radius R", "unit": "nm", "minimum": 1.0, "maximum": 15.0},
        ),
    ),
    PluginSpec(
        "spherical_segment",
        "Spherical segment",
        "particle",
        "Sphere truncated by a plane; height must not exceed 2R.",
        parameters=(
            {"key": "radius_nm", "label": "Sphere radius R", "unit": "nm", "minimum": 1.0, "maximum": 15.0},
            {"key": "height_nm", "label": "Truncation height h", "unit": "nm", "minimum": 1.0, "maximum": 20.0},
        ),
    ),
    PluginSpec(
        "cylinder",
        "Cylinder",
        "particle",
        "Right circular cylinder.",
        parameters=(
            {"key": "radius_nm", "label": "Radius R", "unit": "nm", "minimum": 1.0, "maximum": 15.0},
            {"key": "height_nm", "label": "Height H", "unit": "nm", "minimum": 1.0, "maximum": 30.0},
        ),
    ),
    PluginSpec(
        "box",
        "Box",
        "particle",
        "Rectangular cuboid particle.",
        parameters=(
            {"key": "length_x_nm", "label": "Length X", "unit": "nm", "minimum": 1.0, "maximum": 30.0},
            {"key": "length_y_nm", "label": "Length Y", "unit": "nm", "minimum": 1.0, "maximum": 30.0},
            {"key": "length_z_nm", "label": "Length Z", "unit": "nm", "minimum": 1.0, "maximum": 30.0},
        ),
    ),
    PluginSpec("none", "None", "interference", "No interference function."),
    PluginSpec(
        "paracrystal",
        "Paracrystal",
        "interference",
        "One-dimensional paracrystal with mean spacing D and relative disorder sigma/D.",
        parameters=(
            {"key": "D_nm", "label": "Mean spacing D", "unit": "nm", "minimum": 3.0, "maximum": 50.0},
            {"key": "sigma_D_ratio", "label": "Disorder sigma/D", "unit": "", "minimum": 0.05, "maximum": 0.20},
        ),
    ),
    PluginSpec("physical_background", "Physical background", "preprocessing", "Optional specular/Yoneda-like background augmentation."),
    PluginSpec("noise", "Noise", "preprocessing", "Random detector/noise model."),
    PluginSpec("mask", "Mask", "preprocessing", "Fixed or randomized detector mask."),
    PluginSpec("log", "Log transform", "preprocessing", "Numerically stable logarithmic intensity transform."),
    PluginSpec("normalize", "Normalization", "preprocessing", "Upper, lower, or range normalization."),
    PluginSpec("random_edge_crop", "Random edge crop", "preprocessing", "Randomly remove edge pixels for augmentation."),
):
    REGISTRY.register(item)
