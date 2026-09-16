"""Compact clean-recipe contract for V5 bounds-conditioned studies.

The query, branch target, and coefficient composition use independent seeds.
Curves are evaluated only through the authoritative GUI mixed-model factory.
This module intentionally covers clean curves; observation/noise provenance is
attached by the separate acquisition layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Sequence

import numpy as np

from .amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_QUERY_SAMPLER_VERSION,
    amplitude_range_regime_for,
    sample_v5_amplitude_query,
)
from .amplitude_query_v5 import V5AmplitudeQuery
from .amplitude_sampling_v5 import (
    V5AmplitudeComposition,
    sample_v5_constrained_amplitude_composition,
)
from .bounds_query_v5 import (
    V5BoundsQuery,
    V5_LOCAL_TARGET_VERSION,
    V5SolutionTarget,
)
from .bounds_query_sampling_v5 import (
    V5_SOLUTION_TARGET_SAMPLER_VERSION,
    sample_v5_bounds_query,
    sample_v5_solution_target,
)
from .branch_catalog import decode_branch_pattern
from .clean_recipe_forward_v5 import (
    V5_CLEAN_EXACT_FORWARD_PATH,
    authoritative_v5_gui_parameters,
    evaluate_v5_clean_recipe_forward,
)
from .contract import TOPOLOGIES
from .canonical_component_slots import CANONICAL_COMPONENT_SLOTS_VERSION
from .simulation import GridProvenance


V5_CLEAN_RECIPE_SCHEMA = "gisaxs.posterior_v8.clean_recipe/v4"
V5_CLEAN_RECIPE_VERSION = (
    "posterior_v8_exact_closed_endpoint_complete_slot_gui_k_int_query_first_recipe_v6"
)
V5_EXACT_FORWARD_PATH = V5_CLEAN_EXACT_FORWARD_PATH
_UINT64_MASK = (1 << 64) - 1


def _seed(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= _UINT64_MASK:
        raise ValueError(f"{name} must fit in uint64")
    return result


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _derived_seed(recipe_seed: int, namespace: int) -> int:
    return _splitmix64(_seed(recipe_seed, "recipe_seed") ^ int(namespace))


def canonical_v5_clean_recipe_json(
    *,
    recipe_seed: int,
    query_seed: int,
    amplitude_query_seed: int,
    target_seed: int,
    amplitude_seed: int,
    query: V5BoundsQuery,
    amplitude_query: V5AmplitudeQuery,
    amplitude_range_regime: str,
    target: V5SolutionTarget,
    amplitude: V5AmplitudeComposition,
    grid: GridProvenance,
) -> str:
    """Serialize one validated clean recipe without regenerating its physics."""

    payload = {
        "schema_version": V5_CLEAN_RECIPE_SCHEMA,
        "generator_version": V5_CLEAN_RECIPE_VERSION,
        "exact_forward_path": V5_EXACT_FORWARD_PATH,
        "canonical_component_slots_version": CANONICAL_COMPONENT_SLOTS_VERSION,
        "local_target_version": V5_LOCAL_TARGET_VERSION,
        "solution_target_sampler_version": V5_SOLUTION_TARGET_SAMPLER_VERSION,
        "recipe_seed": recipe_seed,
        "query_seed": query_seed,
        "amplitude_query_seed": amplitude_query_seed,
        "target_seed": target_seed,
        "amplitude_seed": amplitude_seed,
        "query": json.loads(query.canonical_json),
        "query_sha256": query.sha256,
        "amplitude_query_sampler_version": V5_AMPLITUDE_QUERY_SAMPLER_VERSION,
        "amplitude_range_regime": amplitude_range_regime,
        "amplitude_query": json.loads(amplitude_query.canonical_json),
        "amplitude_query_sha256": amplitude_query.sha256,
        "branch_pattern_id": target.pattern_id,
        "local_target_unit": list(target.local_target_unit),
        "truth_components": [asdict(value) for value in target.truth_components],
        "truth_resolution": (
            None if target.truth_resolution is None else asdict(target.truth_resolution)
        ),
        "amplitude_composition": amplitude.audit_payload(),
        "branch_amplitude_constraint": amplitude_query.constraint_for_branch(
            resolution_present=amplitude.resolution_present
        ).to_audit_dict(),
        "grid": asdict(grid),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class V5CleanRecipe:
    """One replayable clean physical recipe for training or evaluation."""

    recipe_seed: int
    query_seed: int
    amplitude_query_seed: int
    target_seed: int
    amplitude_seed: int
    query: V5BoundsQuery
    amplitude_query: V5AmplitudeQuery
    amplitude_range_regime: str
    target: V5SolutionTarget
    amplitude: V5AmplitudeComposition
    grid: GridProvenance
    canonical_json: str
    sha256: str
    schema_version: str = V5_CLEAN_RECIPE_SCHEMA
    generator_version: str = V5_CLEAN_RECIPE_VERSION

    @classmethod
    def create(
        cls,
        *,
        recipe_seed: int,
        query_seed: int,
        amplitude_query_seed: int,
        target_seed: int,
        amplitude_seed: int,
        query: V5BoundsQuery,
        amplitude_query: V5AmplitudeQuery,
        amplitude_range_regime: str,
        target: V5SolutionTarget,
        amplitude: V5AmplitudeComposition,
        grid: GridProvenance,
    ) -> "V5CleanRecipe":
        seeds = tuple(
            _seed(value, name)
            for value, name in (
                (recipe_seed, "recipe_seed"),
                (query_seed, "query_seed"),
                (amplitude_query_seed, "amplitude_query_seed"),
                (target_seed, "target_seed"),
                (amplitude_seed, "amplitude_seed"),
            )
        )
        if not isinstance(query, V5BoundsQuery):
            raise TypeError("query must be a V5BoundsQuery")
        if not isinstance(amplitude_query, V5AmplitudeQuery):
            raise TypeError("amplitude_query must be a V5AmplitudeQuery")
        if amplitude_query.particle_count != len(query.topology):
            raise ValueError("geometry and amplitude queries have different component counts")
        if amplitude_query.resolution_presence_policy != query.resolution_presence_policy:
            raise ValueError("geometry and amplitude Resolution policies disagree")
        selected_range_regime = amplitude_range_regime_for(
            amplitude_range_regime,
            query_seed=seeds[2],
        )
        replayed_amplitude_query = sample_v5_amplitude_query(
            len(query.topology),
            resolution_presence_policy=query.resolution_presence_policy,
            query_seed=seeds[2],
            range_regime=selected_range_regime,
        )
        if amplitude_query != replayed_amplitude_query:
            raise ValueError("amplitude query does not reproduce from its query-first seed")
        if not isinstance(target, V5SolutionTarget) or target.query != query:
            raise ValueError("target must belong to the supplied V5 query")
        replayed_target = sample_v5_solution_target(
            query,
            target_seed=seeds[3],
            pattern_id=target.pattern_id,
            component_intensity_bounds=amplitude_query.component_intensities,
        )
        if target != replayed_target:
            raise ValueError("target does not reproduce under the complete geometry/Int contract")
        if not isinstance(amplitude, V5AmplitudeComposition):
            raise TypeError("amplitude must be a V5AmplitudeComposition")
        _, resolution_present = decode_branch_pattern(target.pattern_id)
        if amplitude.component_count != len(target.truth_components):
            raise ValueError("amplitude component count must match target topology")
        if amplitude.resolution_present != resolution_present:
            raise ValueError("amplitude Resolution presence must match target branch")
        amplitude_constraint = amplitude_query.constraint_for_branch(
            resolution_present=resolution_present
        )
        if not amplitude_constraint.contains(
            amplitude.coefficient_vector,
            k=amplitude.k,
            atol=2.0e-9,
        ):
            raise ValueError("amplitude composition is outside its pre-existing physical query")
        if not isinstance(grid, GridProvenance):
            raise TypeError("grid must be GridProvenance")
        canonical = canonical_v5_clean_recipe_json(
            recipe_seed=seeds[0],
            query_seed=seeds[1],
            amplitude_query_seed=seeds[2],
            target_seed=seeds[3],
            amplitude_seed=seeds[4],
            query=query,
            amplitude_query=amplitude_query,
            amplitude_range_regime=selected_range_regime,
            target=target,
            amplitude=amplitude,
            grid=grid,
        )
        return cls(
            *seeds,
            query,
            amplitude_query,
            selected_range_regime,
            target,
            amplitude,
            grid,
            canonical,
            sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def __post_init__(self) -> None:
        for value, name in (
            (self.recipe_seed, "recipe_seed"),
            (self.query_seed, "query_seed"),
            (self.amplitude_query_seed, "amplitude_query_seed"),
            (self.target_seed, "target_seed"),
            (self.amplitude_seed, "amplitude_seed"),
        ):
            _seed(value, name)
        if self.query.query_seed != self.query_seed:
            raise ValueError("query_seed disagrees with query provenance")
        expected_amplitude_query = sample_v5_amplitude_query(
            len(self.query.topology),
            resolution_presence_policy=self.query.resolution_presence_policy,
            query_seed=self.amplitude_query_seed,
            range_regime=self.amplitude_range_regime,
        )
        if self.amplitude_query != expected_amplitude_query:
            raise ValueError("amplitude-query provenance disagrees with recipe")
        if self.target.target_seed != self.target_seed or self.target.query != self.query:
            raise ValueError("target provenance disagrees with recipe")
        replayed_target = sample_v5_solution_target(
            self.query,
            target_seed=self.target_seed,
            pattern_id=self.target.pattern_id,
            component_intensity_bounds=self.amplitude_query.component_intensities,
        )
        if self.target != replayed_target:
            raise ValueError("target provenance escaped the complete geometry/Int contract")
        if self.amplitude.seed != self.amplitude_seed:
            raise ValueError("amplitude_seed disagrees with composition provenance")
        _, resolution_present = decode_branch_pattern(self.target.pattern_id)
        if (
            self.amplitude.component_count != len(self.target.truth_components)
            or self.amplitude.resolution_present != resolution_present
        ):
            raise ValueError("composition does not match target branch")
        if not self.amplitude_query.constraint_for_branch(
            resolution_present=resolution_present
        ).contains(self.amplitude.coefficient_vector, atol=2.0e-9):
            raise ValueError("composition escaped the query-first amplitude constraint")
        canonical = canonical_v5_clean_recipe_json(
            recipe_seed=self.recipe_seed,
            query_seed=self.query_seed,
            amplitude_query_seed=self.amplitude_query_seed,
            target_seed=self.target_seed,
            amplitude_seed=self.amplitude_seed,
            query=self.query,
            amplitude_query=self.amplitude_query,
            amplitude_range_regime=self.amplitude_range_regime,
            target=self.target,
            amplitude=self.amplitude,
            grid=self.grid,
        )
        expected = sha256(canonical.encode("utf-8")).hexdigest()
        if self.canonical_json != canonical or self.sha256 != expected:
            raise ValueError("V5 clean recipe does not reproduce its audit hash")
        if self.schema_version != V5_CLEAN_RECIPE_SCHEMA:
            raise ValueError("unsupported V5 clean recipe schema")
        if self.generator_version != V5_CLEAN_RECIPE_VERSION:
            raise ValueError("unsupported V5 clean recipe generator")


def sample_v5_clean_recipe(
    topology: Sequence[str],
    *,
    recipe_seed: int,
    amplitude_regime: str | None = None,
    amplitude_range_regime: str = "full",
    pattern_id: int | None = None,
    grid: GridProvenance | None = None,
) -> V5CleanRecipe:
    """Sample query -> branch/geometry -> composition using disjoint seeds."""

    shapes = tuple(topology)
    if shapes not in TOPOLOGIES:
        raise ValueError("topology must be one canonical Posterior V8 topology")
    seed = _seed(recipe_seed, "recipe_seed")
    query_seed, amplitude_query_seed, target_seed, amplitude_seed = (
        v5_clean_recipe_seed_lineage(seed)
    )
    query = sample_v5_bounds_query(shapes, query_seed=query_seed)
    selected_range_regime = amplitude_range_regime_for(
        amplitude_range_regime,
        query_seed=amplitude_query_seed,
    )
    amplitude_query = sample_v5_amplitude_query(
        len(shapes),
        resolution_presence_policy=query.resolution_presence_policy,
        query_seed=amplitude_query_seed,
        range_regime=selected_range_regime,
    )
    target = sample_v5_solution_target(
        query,
        target_seed=target_seed,
        pattern_id=pattern_id,
        component_intensity_bounds=amplitude_query.component_intensities,
    )
    _, resolution_present = decode_branch_pattern(target.pattern_id)
    amplitude_constraint = amplitude_query.constraint_for_branch(
        resolution_present=resolution_present
    )
    amplitude = sample_v5_constrained_amplitude_composition(
        amplitude_constraint,
        seed=amplitude_seed,
        regime="balanced_particles" if amplitude_regime is None else amplitude_regime,
    )
    selected_grid = GridProvenance(n_points=256) if grid is None else grid
    return V5CleanRecipe.create(
        recipe_seed=seed,
        query_seed=query_seed,
        amplitude_query_seed=amplitude_query_seed,
        target_seed=target_seed,
        amplitude_seed=amplitude_seed,
        query=query,
        amplitude_query=amplitude_query,
        amplitude_range_regime=selected_range_regime,
        target=target,
        amplitude=amplitude,
        grid=selected_grid,
    )


authoritative_gui_parameters = authoritative_v5_gui_parameters
evaluate_v5_clean_recipe = evaluate_v5_clean_recipe_forward


def v5_clean_recipe_seed_lineage(recipe_seed: int) -> tuple[int, int, int, int]:
    """Return the four frozen namespace-derived seeds for one pilot recipe."""

    seed = _seed(recipe_seed, "recipe_seed")
    return (
        _derived_seed(seed, 0x51554552),
        _derived_seed(seed, 0x41515545),
        _derived_seed(seed, 0x54415247),
        _derived_seed(seed, 0x414D504C),
    )


__all__ = [
    "V5_CLEAN_RECIPE_SCHEMA",
    "V5_CLEAN_RECIPE_VERSION",
    "V5_EXACT_FORWARD_PATH",
    "V5CleanRecipe",
    "authoritative_gui_parameters",
    "canonical_v5_clean_recipe_json",
    "evaluate_v5_clean_recipe",
    "sample_v5_clean_recipe",
    "v5_clean_recipe_seed_lineage",
]
