"""Geometry-aware physical constraints for AI fitting and refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot, isfinite
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence


GEOMETRY_ALIASES = {
    "sphere": "sphere",
    "cylinder": "cylinder",
    "isotropic cylinder": "cylinder",
    "random cylinder": "cylinder",
    "vertical cylinder": "vertical_cylinder",
    "vertical_cylinder": "vertical_cylinder",
}


def normalize_geometry(name: str) -> str:
    key = str(name).strip().lower().replace("-", " ")
    return GEOMETRY_ALIASES.get(key, key.replace(" ", "_"))


@dataclass(frozen=True)
class ConstraintViolation:
    constraint_id: str
    component_index: int | None
    message: str
    formula: str


@dataclass(frozen=True)
class ConstraintDefinition:
    id: str
    label: str
    formula: str
    meaning: str
    geometries: frozenset[str]
    default_enabled: bool = True
    default_margin: float = 1.001
    minimum_margin: float = 1.0
    maximum_margin: float = 2.0

    def applies_to(self, geometries: Iterable[str]) -> bool:
        normalized = {normalize_geometry(item) for item in geometries}
        return not self.geometries or bool(normalized & self.geometries)


@dataclass
class ConstraintOption:
    enabled: bool
    margin: float

    def normalized(self, definition: ConstraintDefinition) -> "ConstraintOption":
        margin = min(max(float(self.margin), definition.minimum_margin), definition.maximum_margin)
        return ConstraintOption(bool(self.enabled), margin)


@dataclass
class ConstraintSet:
    registry: "ConstraintRegistry"
    options: Dict[str, ConstraintOption] = field(default_factory=dict)

    @classmethod
    def defaults(cls, registry: "ConstraintRegistry" | None = None) -> "ConstraintSet":
        registry = registry or constraint_registry
        return cls(
            registry=registry,
            options={
                definition.id: ConstraintOption(definition.default_enabled, definition.default_margin)
                for definition in registry.definitions()
            },
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any] | None,
        registry: "ConstraintRegistry" | None = None,
    ) -> "ConstraintSet":
        result = cls.defaults(registry)
        for key, value in (payload or {}).items():
            if key not in result.options or not isinstance(value, Mapping):
                continue
            definition = result.registry.get(key)
            option = ConstraintOption(
                bool(value.get("enabled", result.options[key].enabled)),
                float(value.get("margin", result.options[key].margin)),
            )
            result.options[key] = option.normalized(definition)
        return result

    def to_dict(self) -> Dict[str, Dict[str, float | bool]]:
        return {
            key: {"enabled": option.enabled, "margin": option.margin}
            for key, option in self.options.items()
        }

    def applicable(self, geometries: Iterable[str]) -> list[tuple[ConstraintDefinition, ConstraintOption]]:
        result = []
        for definition in self.registry.definitions():
            if definition.applies_to(geometries):
                result.append((definition, self.options[definition.id].normalized(definition)))
        return result

    def validate_components(self, components: Sequence[Mapping[str, Any]]) -> list[ConstraintViolation]:
        violations: list[ConstraintViolation] = []
        for index, component in enumerate(components):
            geometry = normalize_geometry(str(component.get("type", component.get("geometry", ""))))
            params = component.get("params", component)
            for definition, option in self.applicable([geometry]):
                if not option.enabled:
                    continue
                violations.extend(self.registry.evaluate(definition.id, index, geometry, params, option.margin))
        return violations

    def d_constraint_payload(self, spacing_rule: str = "max_diameter") -> Dict[str, Any]:
        option = self.options.get("hard_core_spacing")
        if option is None or not option.enabled:
            return {"presence": "optional", "spacing_rule": "free"}
        if spacing_rule not in {"max_diameter", "mean_diameter"}:
            raise ValueError("spacing_rule must be max_diameter or mean_diameter")
        return {
            "presence": "optional",
            "spacing_rule": spacing_rule,
            "margin": float(option.margin),
        }


class ConstraintRegistry:
    def __init__(self) -> None:
        self._definitions: Dict[str, ConstraintDefinition] = {}
        self._evaluators: Dict[str, Callable[..., list[ConstraintViolation]]] = {}

    def register(self, definition: ConstraintDefinition, evaluator: Callable[..., list[ConstraintViolation]]) -> None:
        if definition.id in self._definitions:
            raise ValueError(f"Constraint {definition.id!r} is already registered")
        self._definitions[definition.id] = definition
        self._evaluators[definition.id] = evaluator

    def definitions(self) -> tuple[ConstraintDefinition, ...]:
        return tuple(self._definitions.values())

    def get(self, constraint_id: str) -> ConstraintDefinition:
        try:
            return self._definitions[constraint_id]
        except KeyError as exc:
            raise KeyError(f"Unknown constraint {constraint_id!r}") from exc

    def evaluate(
        self,
        constraint_id: str,
        component_index: int,
        geometry: str,
        params: Mapping[str, Any],
        margin: float,
    ) -> list[ConstraintViolation]:
        return self._evaluators[constraint_id](component_index, geometry, params, margin)


def _number(params: Mapping[str, Any], *names: str) -> float | None:
    for name in names:
        if name in params and params[name] is not None:
            try:
                return float(params[name])
            except (TypeError, ValueError):
                return None
    return None


def exclusion_size(geometry: str, params: Mapping[str, Any]) -> float | None:
    """Return the trained hard-core size for one geometry.

    The random/isotropic cylinder uses its circumscribed-sphere diameter,
    ``sqrt((2R)^2 + h^2)``. Sphere and vertical cylinder use lateral ``2R``.
    """
    geometry = normalize_geometry(geometry)
    radius = _number(params, "R", "radius")
    if radius is None or radius <= 0:
        return None
    if geometry == "cylinder":
        height = _number(params, "h", "height")
        if height is None or height <= 0:
            return None
        return hypot(2.0 * radius, height)
    if geometry in {"sphere", "vertical_cylinder"}:
        return 2.0 * radius
    return None


def _positivity(index: int, geometry: str, params: Mapping[str, Any], margin: float) -> list[ConstraintViolation]:
    del geometry, margin
    violations = []
    for name, value in params.items():
        if name in {"type", "geometry"} or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not isfinite(numeric) or numeric < 0:
            violations.append(ConstraintViolation("positivity", index, f"{name} must be finite and >= 0 (got {value!r})", "p >= 0"))
    return violations


def _size_distribution(index: int, geometry: str, params: Mapping[str, Any], margin: float) -> list[ConstraintViolation]:
    del margin
    geometry = normalize_geometry(geometry)
    if geometry == "vertical_cylinder":
        sigma_r = _number(params, "sigma_R", "sigma_radius")
        violations = []
        if sigma_r is not None and not (0 < sigma_r <= 0.9):
            violations.append(
                ConstraintViolation(
                    "size_distribution",
                    index,
                    "vertical-cylinder sigma_R is fractional and must satisfy 0 < sigma_R <= 0.9",
                    "0 < sigma_R/R <= 0.9",
                )
            )
        checks = [("sigma_D", "D"), ("sigma_diameter", "diameter")]
    else:
        violations = []
        checks = [("sigma_R", "R"), ("sigma_radius", "radius"), ("sigma_D", "D"), ("sigma_diameter", "diameter")]
    if geometry == "cylinder":
        checks.extend([("sigma_h", "h"), ("sigma_height", "height")])
    for sigma_name, size_name in checks:
        sigma = _number(params, sigma_name)
        size = _number(params, size_name)
        if sigma is None or size is None:
            continue
        # D is an explicitly optional trained parameter.  Its absent encoding
        # is the pair D=0, sigma_D=0, which must not be rejected as a width
        # distribution violation.
        if size_name in {"D", "diameter"} and size == 0 and sigma == 0:
            continue
        if sigma <= 0 or sigma > 0.9 * size:
            violations.append(
                ConstraintViolation(
                    "size_distribution",
                    index,
                    f"{sigma_name} must satisfy 0 < {sigma_name} <= 0.9*{size_name}",
                    "0 < sigma_size <= 0.9*size",
                )
            )
    return violations


def _hard_core(index: int, geometry: str, params: Mapping[str, Any], margin: float) -> list[ConstraintViolation]:
    spacing = _number(params, "D", "diameter")
    if spacing is None or spacing == 0:
        return []  # The trained representation explicitly permits no D.
    threshold = exclusion_size(geometry, params)
    if threshold is None:
        return []
    required = float(margin) * threshold
    if spacing <= required:
        formula = "D > margin*sqrt((2R)^2+h^2)" if normalize_geometry(geometry) == "cylinder" else "D > margin*2R"
        return [ConstraintViolation("hard_core_spacing", index, f"D={spacing:g} must be > {required:g}", formula)]
    return []


constraint_registry = ConstraintRegistry()
constraint_registry.register(
    ConstraintDefinition(
        id="positivity",
        label="Parameter positivity",
        formula="p >= 0",
        meaning="Physical sizes, widths, spacings and intensities cannot be negative.",
        geometries=frozenset(),
        default_margin=1.0,
        minimum_margin=1.0,
        maximum_margin=1.0,
    ),
    _positivity,
)
constraint_registry.register(
    ConstraintDefinition(
        id="size_distribution",
        label="Size-distribution range",
        formula="0 < sigma_size <= 0.9*size",
        meaning="Distribution widths stay positive and below the corresponding mean size.",
        geometries=frozenset({"sphere", "cylinder", "vertical_cylinder"}),
        default_margin=1.0,
        minimum_margin=1.0,
        maximum_margin=1.0,
    ),
    _size_distribution,
)
constraint_registry.register(
    ConstraintDefinition(
        id="hard_core_spacing",
        label="Non-overlap spacing",
        formula="sphere/vertical: D > 2R; random cylinder: D > sqrt((2R)^2+h^2)",
        meaning="D=0 remains allowed. When D is present, centers must exceed the trained geometry-specific exclusion size.",
        geometries=frozenset({"sphere", "cylinder", "vertical_cylinder"}),
        default_margin=1.001,
        minimum_margin=1.000001,
        maximum_margin=1.2,
    ),
    _hard_core,
)
