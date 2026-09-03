"""Canonical physical features used only for V5 cross-split leakage audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .amplitude_query_v5 import AMPLITUDE_QUERY_EMBEDDING_DIM
from .bounds_first_contract import BOUNDS_EMBEDDING_DIM
from .branch_catalog import decode_branch_pattern
from .branch_codec import ProfiledBranchCodec, ResolutionBounds, UNIT_CUBE_DIMENSIONS
from .canonical_branch_catalog import canonical_branch_pattern_id
from .canonical_component_slots import component_physical_dictionary_key
from .contract import (
    MAX_COMPONENTS,
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    full_component_bounds,
    gui_component_to_latent,
)
from .synthetic_recipe_v5 import V5CleanRecipe


V5_AMPLITUDE_LEAKAGE_DIM = MAX_COMPONENTS + 3
V5_CLEAN_PHYSICS_LEAKAGE_DIM = UNIT_CUBE_DIMENSIONS + V5_AMPLITUDE_LEAKAGE_DIM
V5_QUERY_BOUNDS_LEAKAGE_DIM = BOUNDS_EMBEDDING_DIM + AMPLITUDE_QUERY_EMBEDDING_DIM
V5_LEAKAGE_FEATURE_VERSION = (
    "global_physical_geometry26_arctan_log_total_scale_quotiented_effective_"
    "coefficient_composition7_and_geometry78_amplitude21_query_bounds_v3"
)


def coefficient_total_to_unit(value: float) -> float:
    """Injectively map every positive finite total coefficient into (0, 1)."""

    total = float(value)
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("total coefficient must be finite and positive")
    return float(0.5 + np.arctan(np.log10(total)) / np.pi)


def coefficient_total_from_unit(value: float) -> float:
    unit = float(value)
    if not np.isfinite(unit) or not 0.0 < unit < 1.0:
        raise ValueError("coefficient total unit value must lie inside (0, 1)")
    result = float(10.0 ** np.tan(np.pi * (unit - 0.5)))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("coefficient total unit value is outside finite reconstruction")
    return result


@dataclass(frozen=True)
class V5CleanLeakageFeatures:
    normalized_continuous_parameters: tuple[float, ...]
    normalized_query_bounds: tuple[float, ...]
    topology_id: int
    branch_pattern_id: int
    weakest_particle_fraction: float
    version: str = V5_LEAKAGE_FEATURE_VERSION

    def __post_init__(self) -> None:
        parameters = np.asarray(self.normalized_continuous_parameters, dtype=np.float64)
        bounds = np.asarray(self.normalized_query_bounds, dtype=np.float64)
        if (
            parameters.shape != (V5_CLEAN_PHYSICS_LEAKAGE_DIM,)
            or not np.all(np.isfinite(parameters))
            or np.any(parameters < 0.0)
            or np.any(parameters > 1.0)
        ):
            raise ValueError("clean-physics leakage vector has an invalid shape/value")
        if (
            bounds.shape != (V5_QUERY_BOUNDS_LEAKAGE_DIM,)
            or not np.all(np.isfinite(bounds))
            or np.any(bounds < 0.0)
            or np.any(bounds > 1.0)
        ):
            raise ValueError("query-bounds leakage vector has an invalid shape/value")
        if self.version != V5_LEAKAGE_FEATURE_VERSION:
            raise ValueError("unsupported V5 leakage feature version")
        object.__setattr__(
            self,
            "normalized_continuous_parameters",
            tuple(float(value) for value in parameters),
        )
        object.__setattr__(
            self,
            "normalized_query_bounds",
            tuple(float(value) for value in bounds),
        )


def v5_clean_leakage_features(recipe: V5CleanRecipe) -> V5CleanLeakageFeatures:
    """Encode global physical geometry and all generating linear coefficients.

    The geometry is encoded against the full versioned physical domains, not
    against each row's local user bounds.  The amplitude part is one-to-one:
    total coefficient plus the six-slot scale-quotiented effective-coefficient
    composition ``[BG, a1..a4, a_res] / total``.  This is not a constraint on
    the independent GUI ``Int_i`` values.
    """

    if not isinstance(recipe, V5CleanRecipe):
        raise TypeError("recipe must be a V5CleanRecipe")
    canonical_pattern = canonical_branch_pattern_id(
        recipe.query.topology_id,
        recipe.target.pattern_id,
    )
    d_flags, resolution_present = decode_branch_pattern(canonical_pattern)
    topology = recipe.query.topology
    bounds = tuple(
        full_component_bounds(
            shape,
            d_policy="required" if d_flags[slot] else "absent",
        )
        for slot, shape in enumerate(topology)
    )
    resolution_bounds = (
        ResolutionBounds(RESOLUTION_SIGMA_DOMAIN, RESOLUTION_NU_DOMAIN)
        if resolution_present
        else None
    )
    codec = ProfiledBranchCodec.build(
        topology,
        bounds,
        d_flags[: len(topology)],
        resolution_bounds=resolution_bounds,
    )
    latent = tuple(gui_component_to_latent(value) for value in recipe.target.truth_components)
    paired = list(zip(latent, recipe.amplitude.particle_amplitudes))
    start = 0
    while start < len(topology):
        stop = start + 1
        while stop < len(topology) and topology[stop] == topology[start]:
            stop += 1
        paired[start:stop] = sorted(
            paired[start:stop],
            key=lambda value: (
                value[0].log_D is not None,
                component_physical_dictionary_key(value[0]),
                value[1],
            ),
        )
        start = stop
    latent = tuple(value[0] for value in paired)
    particle_amplitudes = tuple(value[1] for value in paired)
    geometry = codec.encode(latent, recipe.target.truth_resolution).unit_cube

    coefficients = np.zeros(MAX_COMPONENTS + 2, dtype=np.float64)
    coefficients[0] = recipe.amplitude.background
    coefficients[1 : 1 + len(particle_amplitudes)] = particle_amplitudes
    coefficients[-1] = recipe.amplitude.resolution_amplitude
    total = float(np.sum(coefficients))
    fractions = coefficients / total
    continuous = (*geometry, coefficient_total_to_unit(total), *fractions)
    return V5CleanLeakageFeatures(
        normalized_continuous_parameters=continuous,
        normalized_query_bounds=(
            *recipe.query.bounds_embedding,
            *recipe.amplitude_query.model_embedding(1.0),
        ),
        topology_id=recipe.query.topology_id,
        branch_pattern_id=canonical_pattern,
        weakest_particle_fraction=min(recipe.amplitude.particle_weights),
    )


__all__ = [
    "V5_AMPLITUDE_LEAKAGE_DIM",
    "V5_CLEAN_PHYSICS_LEAKAGE_DIM",
    "V5_LEAKAGE_FEATURE_VERSION",
    "V5_QUERY_BOUNDS_LEAKAGE_DIM",
    "V5CleanLeakageFeatures",
    "coefficient_total_from_unit",
    "coefficient_total_to_unit",
    "v5_clean_leakage_features",
]
