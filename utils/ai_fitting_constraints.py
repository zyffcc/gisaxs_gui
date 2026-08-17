"""Legacy import compatibility for fitting physical constraints."""

from src.gimap.features.fitting.domain.physical_constraints import (
    ConstraintDefinition,
    ConstraintOption,
    ConstraintRegistry,
    ConstraintSet,
    ConstraintViolation,
    constraint_registry,
    exclusion_size,
    normalize_geometry,
)

__all__ = [
    "ConstraintDefinition",
    "ConstraintOption",
    "ConstraintRegistry",
    "ConstraintSet",
    "ConstraintViolation",
    "constraint_registry",
    "exclusion_size",
    "normalize_geometry",
]
