"""Single exact-forward implementation for structurally valid V5 clean recipes."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from src.gimap.features.fitting.domain.scattering_model import make_mixed_model

from .amplitude_query_v5 import V5AmplitudeQuery
from .bounds_query_v5 import V5BoundsQuery, V5SolutionTarget
from .contract import CYLINDER
from .simulation import GridProvenance


V5_CLEAN_RECIPE_PROTOCOL_VERSION = (
    "posterior_v8_structural_query_target_independent_gui_k_int_grid_clean_recipe_v2"
)
V5_CLEAN_EXACT_FORWARD_PATH = "src.gimap.features.fitting.domain.scattering_model.make_mixed_model"


@runtime_checkable
class V5CleanAmplitude(Protocol):
    component_count: int
    resolution_present: bool
    background: float
    particle_amplitudes: tuple[float, ...]
    resolution_amplitude: float

    @property
    def k(self) -> float: ...

    @property
    def particle_weights(self) -> tuple[float, ...]: ...

    @property
    def int_res(self) -> float: ...

    @property
    def coefficient_vector(self) -> tuple[float, ...]: ...


@runtime_checkable
class V5CleanRecipeLike(Protocol):
    recipe_seed: int
    query: V5BoundsQuery
    amplitude_query: V5AmplitudeQuery
    target: V5SolutionTarget
    amplitude: V5CleanAmplitude
    grid: GridProvenance
    canonical_json: str
    sha256: str


def validate_v5_clean_recipe_like(recipe: object) -> V5CleanRecipeLike:
    """Validate the common physical contract without requiring one generator class."""

    if not isinstance(recipe, V5CleanRecipeLike):
        raise TypeError("recipe must implement the V5 clean-recipe protocol")
    if (
        not isinstance(recipe.canonical_json, str)
        or not isinstance(recipe.sha256, str)
        or sha256(recipe.canonical_json.encode("utf-8")).hexdigest() != recipe.sha256
    ):
        raise ValueError("clean recipe canonical JSON/SHA-256 does not reproduce")
    if not isinstance(recipe.query, V5BoundsQuery):
        raise TypeError("clean recipe query must be V5BoundsQuery")
    if not isinstance(recipe.amplitude_query, V5AmplitudeQuery):
        raise TypeError("clean recipe amplitude_query must be V5AmplitudeQuery")
    if not isinstance(recipe.target, V5SolutionTarget) or recipe.target.query != recipe.query:
        raise ValueError("clean recipe target must belong to its geometry query")
    if not isinstance(recipe.grid, GridProvenance):
        raise TypeError("clean recipe grid must be GridProvenance")
    if recipe.amplitude.component_count != len(recipe.target.truth_components):
        raise ValueError("clean recipe amplitude count does not match its target")
    constraint = recipe.amplitude_query.constraint_for_branch(
        resolution_present=recipe.amplitude.resolution_present
    )
    if not constraint.contains(
        recipe.amplitude.coefficient_vector,
        k=recipe.amplitude.k,
        atol=2.0e-9,
    ):
        raise ValueError("clean recipe amplitude escapes its query")
    return recipe


def authoritative_v5_gui_parameters(recipe: V5CleanRecipeLike) -> tuple[float, ...]:
    """Serialize any validated V5 clean recipe in frozen GUI model order."""

    recipe = validate_v5_clean_recipe_like(recipe)
    parameters: list[float] = []
    for component, weight in zip(recipe.target.truth_components, recipe.amplitude.particle_weights):
        d = 0.0 if component.D is None else component.D
        sigma_d = 0.0 if component.sigma_D is None else component.sigma_D
        if component.shape == CYLINDER:
            parameters.extend(
                (
                    weight,
                    component.R,
                    component.sigma_R,
                    component.h,
                    component.sigma_h,
                    d,
                    sigma_d,
                )
            )
        else:
            parameters.extend((weight, component.R, component.sigma_R, d, sigma_d))
    resolution = recipe.target.truth_resolution
    if resolution is None:
        sigma_res = nu_res = int_res = 0.0
    else:
        sigma_res = resolution.sigma_res
        nu_res = resolution.nu_res
        int_res = recipe.amplitude.int_res
    parameters.extend((recipe.amplitude.background, sigma_res, nu_res, int_res, recipe.amplitude.k))
    values = np.asarray(parameters, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise RuntimeError("V5 GUI parameter serialization produced NaN/Inf")
    return tuple(float(value) for value in values)


def evaluate_v5_clean_recipe_forward(
    recipe: V5CleanRecipeLike,
    q: Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate the one authoritative empirical 1D forward for a V5 recipe."""

    recipe = validate_v5_clean_recipe_like(recipe)
    q_values = recipe.grid.values() if q is None else np.asarray(q, dtype=np.float64)
    if (
        q_values.ndim != 1
        or not q_values.size
        or not np.all(np.isfinite(q_values))
        or np.any(q_values <= 0.0)
    ):
        raise ValueError("q must be a non-empty finite positive vector")
    model = make_mixed_model([value.shape for value in recipe.target.truth_components])
    intensity = np.asarray(
        model(q_values, *authoritative_v5_gui_parameters(recipe)), dtype=np.float64
    )
    if (
        intensity.shape != q_values.shape
        or not np.all(np.isfinite(intensity))
        or np.any(intensity <= 0.0)
    ):
        raise RuntimeError("authoritative V5 forward produced an invalid curve")
    intensity.setflags(write=False)
    return intensity


__all__ = [
    "V5_CLEAN_EXACT_FORWARD_PATH",
    "V5_CLEAN_RECIPE_PROTOCOL_VERSION",
    "V5CleanAmplitude",
    "V5CleanRecipeLike",
    "authoritative_v5_gui_parameters",
    "evaluate_v5_clean_recipe_forward",
    "validate_v5_clean_recipe_like",
]
