"""Strict cross-CPU decoder for persisted pilot clean recipes.

The legacy pilot generator deliberately records a platform-libm numeric policy.
Its seed is therefore provenance, not a portable reconstruction algorithm: the
same NumPy random stream can pass through ``log``/``exp`` with a few different
binary64 ULPs on different CPU families.  This decoder treats the canonical,
artifact-hashed payload as authoritative while revalidating every nested
contract, branch, physical range, target wire value, and exact-forward input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping

import numpy as np

from .amplitude_query_sampling_v5 import (
    V5_AMPLITUDE_QUERY_SAMPLER_VERSION,
    amplitude_range_regime_for,
)
from .amplitude_sampling_v5 import V5AmplitudeComposition
from .bounds_query_sampling_v5 import V5_SOLUTION_TARGET_SAMPLER_VERSION
from .bounds_query_v5 import (
    V5_LOCAL_TARGET_VERSION,
    V5BoundsQuery,
    V5SolutionTarget,
    bounds_query_from_json,
)
from .branch_catalog import decode_branch_pattern
from .canonical_component_slots import CANONICAL_COMPONENT_SLOTS_VERSION
from .clean_recipe_forward_v5 import V5_CLEAN_EXACT_FORWARD_PATH
from .contract import GuiComponentParameters, gui_component_to_latent
from .grouped_amplitude_join_v5 import amplitude_query_from_json
from .profiled_forward import ResolutionShape
from .simulation import GridProvenance
from .sobol_numeric_canonicalization_v5 import validate_v5_numeric_policy
from .synthetic_recipe_v5 import (
    V5_CLEAN_RECIPE_SCHEMA,
    V5_CLEAN_RECIPE_VERSION,
    V5CleanRecipe,
    canonical_v5_clean_recipe_json,
    v5_clean_recipe_seed_lineage,
)


_TOP_LEVEL_FIELDS = {
    "schema_version",
    "generator_version",
    "exact_forward_path",
    "canonical_component_slots_version",
    "local_target_version",
    "solution_target_sampler_version",
    "recipe_seed",
    "query_seed",
    "amplitude_query_seed",
    "target_seed",
    "amplitude_seed",
    "query",
    "query_sha256",
    "amplitude_query_sampler_version",
    "amplitude_range_regime",
    "amplitude_query",
    "amplitude_query_sha256",
    "branch_pattern_id",
    "local_target_unit",
    "truth_components",
    "truth_resolution",
    "amplitude_composition",
    "branch_amplitude_constraint",
    "grid",
}
_COMPONENT_FIELDS = {"shape", "R", "sigma_R", "h", "sigma_h", "D", "sigma_D"}
_AMPLITUDE_FIELDS = {
    "schema_version",
    "generator_version",
    "regime",
    "seed",
    "component_count",
    "resolution_present",
    "background",
    "k",
    "particle_amplitudes",
    "resolution_amplitude",
    "component_intensities",
    "int_res",
    "effective_particle_fractions",
    "selected_particle_slot",
    "coefficient_fraction_is_observability",
}
_UINT64_MAX = (1 << 64) - 1
_LATENT_COMPONENT_FIELDS = (
    "log_R",
    "sigma_R_fraction",
    "log_h",
    "sigma_h_fraction",
    "log_D",
    "sigma_D_fraction",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _strict_object(encoded: str) -> dict[str, object]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate persisted clean-recipe field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(encoded, object_pairs_hook=reject_duplicates)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("persisted clean recipe is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("persisted clean recipe must contain one JSON object")
    if _canonical_json(value) != encoded:
        raise ValueError("persisted clean recipe is not canonical JSON")
    return value


def _uint64(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= _UINT64_MAX:
        raise ValueError(f"{name} must fit in uint64")
    return result


def _same_float32(left: float, right: float) -> bool:
    return bool(np.float32(left) == np.float32(right))


def _fixed_axis_canonical_truth(
    *,
    codec,
    canonical,
    truth_components: tuple[GuiComponentParameters, ...],
    truth_resolution: ResolutionShape | None,
):
    """Use decoded endpoints only for coordinates with no continuous freedom.

    A persisted V5 recipe was generated and hashed on one CPU.  Its exact GUI
    truth can map back to a fixed coupled latent endpoint a few binary64 ULPs
    away on another CPU even though both values have the same declared float32
    target wire representation.  Varying coordinates remain the persisted
    truth and must still survive the full codec encode check below.
    """

    decoded_components, decoded_resolution = codec.decode(canonical)
    persisted_components = tuple(
        gui_component_to_latent(value) for value in truth_components
    )
    varying = codec.varying_mask
    canonical_components = []
    for slot, (persisted, decoded) in enumerate(
        zip(persisted_components, decoded_components, strict=True)
    ):
        updates: dict[str, float | None] = {}
        for axis, field in enumerate(_LATENT_COMPONENT_FIELDS):
            index = slot * len(_LATENT_COMPONENT_FIELDS) + axis
            persisted_value = getattr(persisted, field)
            decoded_value = getattr(decoded, field)
            if (persisted_value is None) != (decoded_value is None):
                raise ValueError("persisted truth latent presence disagrees with its branch")
            if not varying[index] and persisted_value is not None:
                if not _same_float32(persisted_value, decoded_value):
                    raise ValueError(
                        "persisted fixed-axis truth disagrees with its decoded endpoint"
                    )
                updates[field] = decoded_value
        canonical_components.append(replace(persisted, **updates))

    canonical_resolution = truth_resolution
    if truth_resolution is not None:
        assert decoded_resolution is not None
        resolution_updates = {}
        for index, field in (
            (len(varying) - 2, "sigma_res"),
            (len(varying) - 1, "nu_res"),
        ):
            persisted_value = getattr(truth_resolution, field)
            decoded_value = getattr(decoded_resolution, field)
            if not varying[index]:
                if not _same_float32(persisted_value, decoded_value):
                    raise ValueError(
                        "persisted fixed Resolution truth disagrees with its decoded endpoint"
                    )
                resolution_updates[field] = decoded_value
        canonical_resolution = replace(truth_resolution, **resolution_updates)

    return tuple(canonical_components), canonical_resolution


@dataclass(frozen=True)
class V5PersistedSolutionTarget(V5SolutionTarget):
    """A hash-bound target whose float64 physical truth came from another CPU."""

    def __post_init__(self) -> None:
        if self.version != V5_LOCAL_TARGET_VERSION:
            raise ValueError("unsupported persisted V5 local solution-target version")
        if not isinstance(self.query, V5BoundsQuery):
            raise TypeError("persisted target query must be a V5BoundsQuery")
        numeric_policy = validate_v5_numeric_policy(self.physical_numeric_policy_version)
        if numeric_policy != self.query.numeric_policy_version:
            raise ValueError("persisted target numeric policy does not match its query")
        pattern = _uint64(self.pattern_id, "pattern_id")
        target_seed = _uint64(self.target_seed, "target_seed")
        codec = self.query.codec_for(pattern)
        try:
            local = tuple(float(value) for value in self.local_target_unit)
        except (TypeError, ValueError) as exc:
            raise ValueError("persisted local target must be a numeric vector") from exc
        if not np.all(np.isfinite(local)) or any(value < 0.0 or value > 1.0 for value in local):
            raise ValueError("persisted local target escaped the closed unit cube")
        canonical = codec.canonical_coordinates(local)
        if canonical.unit_cube != local:
            raise ValueError("persisted local target does not use canonical inactive coordinates")

        components = tuple(self.truth_components)
        if not components or not all(isinstance(value, GuiComponentParameters) for value in components):
            raise TypeError("persisted truth must contain GUI component parameters")
        if tuple(value.shape for value in components) != self.query.topology:
            raise ValueError("persisted truth topology disagrees with its query")
        d_present, resolution_present = decode_branch_pattern(pattern)
        if tuple(value.D is not None for value in components) != d_present[: len(components)]:
            raise ValueError("persisted truth D presence disagrees with its branch")
        if (self.truth_resolution is not None) != resolution_present:
            raise ValueError("persisted truth Resolution presence disagrees with its branch")
        if self.truth_resolution is not None and not isinstance(
            self.truth_resolution, ResolutionShape
        ):
            raise TypeError("persisted Resolution truth must be a ResolutionShape")

        # Decode the exact stored wire target on this CPU to prove the branch
        # remains executable.  Separately encode the persisted physical truth
        # through the same query to prove it remains inside every coupled bound.
        decoded_components, decoded_resolution = codec.decode(canonical)
        if (
            tuple(value.shape for value in decoded_components) != self.query.topology
            or (decoded_resolution is not None) != resolution_present
        ):
            raise ValueError("persisted local target decodes to the wrong branch")
        canonical_components, canonical_resolution = _fixed_axis_canonical_truth(
            codec=codec,
            canonical=canonical,
            truth_components=components,
            truth_resolution=self.truth_resolution,
        )
        encoded_truth = codec.encode(canonical_components, canonical_resolution)
        # The training/evaluation target wire is float32.  Exact equality at
        # that declared wire precision preserves the target contract while the
        # persisted float64 truth remains authoritative for exact forward use.
        if not np.array_equal(
            np.asarray(encoded_truth.unit_cube, dtype=np.float32),
            np.asarray(canonical.unit_cube, dtype=np.float32),
        ):
            raise ValueError("persisted physical truth disagrees with its float32 local target")
        object.__setattr__(self, "pattern_id", pattern)
        object.__setattr__(self, "target_seed", target_seed)
        object.__setattr__(self, "local_target_unit", local)
        object.__setattr__(self, "truth_components", components)


@dataclass(frozen=True)
class V5PersistedCleanRecipe(V5CleanRecipe):
    """A strictly decoded canonical recipe without cross-CPU PRNG regeneration."""

    def __post_init__(self) -> None:
        seeds = tuple(
            _uint64(value, name)
            for value, name in (
                (self.recipe_seed, "recipe_seed"),
                (self.query_seed, "query_seed"),
                (self.amplitude_query_seed, "amplitude_query_seed"),
                (self.target_seed, "target_seed"),
                (self.amplitude_seed, "amplitude_seed"),
            )
        )
        if seeds[1:] != v5_clean_recipe_seed_lineage(seeds[0]):
            raise ValueError("persisted clean-recipe seed lineage is invalid")
        if self.query.query_seed != seeds[1]:
            raise ValueError("persisted query_seed disagrees with query provenance")
        if self.amplitude_range_regime != amplitude_range_regime_for(
            self.amplitude_range_regime,
            query_seed=seeds[2],
        ):
            raise ValueError("persisted amplitude range regime is invalid")
        if not isinstance(self.target, V5PersistedSolutionTarget):
            raise TypeError("persisted recipe requires a persisted solution target")
        if self.target.target_seed != seeds[3] or self.target.query != self.query:
            raise ValueError("persisted target provenance disagrees with recipe")
        if self.amplitude.seed != seeds[4]:
            raise ValueError("persisted amplitude_seed disagrees with composition provenance")
        if self.amplitude_query.particle_count != len(self.query.topology):
            raise ValueError("persisted geometry and amplitude query counts disagree")
        if (
            self.amplitude_query.resolution_presence_policy
            != self.query.resolution_presence_policy
        ):
            raise ValueError("persisted geometry and amplitude Resolution policies disagree")
        _, resolution_present = decode_branch_pattern(self.target.pattern_id)
        if (
            self.amplitude.component_count != len(self.target.truth_components)
            or self.amplitude.resolution_present != resolution_present
        ):
            raise ValueError("persisted amplitude composition disagrees with target branch")
        constraint = self.amplitude_query.constraint_for_branch(
            resolution_present=resolution_present
        )
        if not constraint.contains(
            self.amplitude.coefficient_vector,
            k=self.amplitude.k,
            atol=2.0e-9,
        ):
            raise ValueError("persisted amplitude composition escaped its physical query")
        if not isinstance(self.grid, GridProvenance):
            raise TypeError("persisted recipe grid must be GridProvenance")
        canonical = canonical_v5_clean_recipe_json(
            recipe_seed=seeds[0],
            query_seed=seeds[1],
            amplitude_query_seed=seeds[2],
            target_seed=seeds[3],
            amplitude_seed=seeds[4],
            query=self.query,
            amplitude_query=self.amplitude_query,
            amplitude_range_regime=self.amplitude_range_regime,
            target=self.target,
            amplitude=self.amplitude,
            grid=self.grid,
        )
        digest = sha256(canonical.encode("utf-8")).hexdigest()
        if self.canonical_json != canonical or self.sha256 != digest:
            raise ValueError("persisted clean recipe does not reproduce its audit hash")
        if self.schema_version != V5_CLEAN_RECIPE_SCHEMA:
            raise ValueError("unsupported persisted clean-recipe schema")
        if self.generator_version != V5_CLEAN_RECIPE_VERSION:
            raise ValueError("unsupported persisted clean-recipe generator")


def persisted_v5_clean_recipe_from_json(
    encoded: str,
    expected_sha256: str,
) -> V5PersistedCleanRecipe:
    """Decode one artifact-bound pilot recipe without replaying platform libm."""

    payload = _strict_object(encoded)
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("persisted clean-recipe fields are incomplete or unsupported")
    actual_sha = sha256(encoded.encode("utf-8")).hexdigest()
    if not isinstance(expected_sha256, str) or actual_sha != expected_sha256:
        raise ValueError("persisted clean-recipe JSON/SHA-256 does not reproduce")

    try:
        query_payload = payload["query"]
        if not isinstance(query_payload, Mapping):
            raise ValueError("persisted geometry query must be an object")
        query = bounds_query_from_json(
            _canonical_json(query_payload),
            str(payload["query_sha256"]),
        )
        amplitude_query_payload = payload["amplitude_query"]
        if not isinstance(amplitude_query_payload, Mapping):
            raise ValueError("persisted amplitude query must be an object")
        amplitude_query = amplitude_query_from_json(
            _canonical_json(amplitude_query_payload),
            str(payload["amplitude_query_sha256"]),
        )

        raw_amplitude = payload["amplitude_composition"]
        if not isinstance(raw_amplitude, Mapping) or set(raw_amplitude) != _AMPLITUDE_FIELDS:
            raise ValueError("persisted amplitude composition fields are unsupported")
        amplitude = V5AmplitudeComposition(
            component_count=raw_amplitude["component_count"],
            resolution_present=raw_amplitude["resolution_present"],
            regime=raw_amplitude["regime"],
            seed=raw_amplitude["seed"],
            background=raw_amplitude["background"],
            k=raw_amplitude["k"],
            particle_amplitudes=tuple(raw_amplitude["particle_amplitudes"]),
            resolution_amplitude=raw_amplitude["resolution_amplitude"],
            selected_particle_slot=raw_amplitude["selected_particle_slot"],
            schema_version=raw_amplitude["schema_version"],
            generator_version=raw_amplitude["generator_version"],
        )
        if amplitude.audit_payload() != raw_amplitude:
            raise ValueError("persisted amplitude derived fields do not reproduce")

        raw_components = payload["truth_components"]
        if not isinstance(raw_components, list) or any(
            not isinstance(value, Mapping) or set(value) != _COMPONENT_FIELDS
            for value in raw_components
        ):
            raise ValueError("persisted truth-component fields are unsupported")
        components = tuple(GuiComponentParameters(**value) for value in raw_components)
        raw_resolution = payload["truth_resolution"]
        if raw_resolution is None:
            resolution = None
        else:
            if not isinstance(raw_resolution, Mapping) or set(raw_resolution) != {
                "sigma_res",
                "nu_res",
            }:
                raise ValueError("persisted Resolution fields are unsupported")
            resolution = ResolutionShape(**raw_resolution)
        target = V5PersistedSolutionTarget(
            query=query,
            pattern_id=payload["branch_pattern_id"],
            target_seed=payload["target_seed"],
            local_target_unit=tuple(payload["local_target_unit"]),
            truth_components=components,
            truth_resolution=resolution,
            physical_numeric_policy_version=query.numeric_policy_version,
            version=payload["local_target_version"],
        )

        raw_grid = payload["grid"]
        if not isinstance(raw_grid, Mapping) or set(raw_grid) != {
            "kind",
            "q_min",
            "q_max",
            "n_points",
        }:
            raise ValueError("persisted grid fields are unsupported")
        grid = GridProvenance(**raw_grid)
        if asdict(grid) != raw_grid:
            raise ValueError("persisted grid fields do not reproduce")

        recipe = V5PersistedCleanRecipe(
            recipe_seed=payload["recipe_seed"],
            query_seed=payload["query_seed"],
            amplitude_query_seed=payload["amplitude_query_seed"],
            target_seed=payload["target_seed"],
            amplitude_seed=payload["amplitude_seed"],
            query=query,
            amplitude_query=amplitude_query,
            amplitude_range_regime=payload["amplitude_range_regime"],
            target=target,
            amplitude=amplitude,
            grid=grid,
            canonical_json=encoded,
            sha256=expected_sha256,
            schema_version=payload["schema_version"],
            generator_version=payload["generator_version"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("persisted clean recipe is invalid") from exc

    expected_constants = {
        "schema_version": V5_CLEAN_RECIPE_SCHEMA,
        "generator_version": V5_CLEAN_RECIPE_VERSION,
        "exact_forward_path": V5_CLEAN_EXACT_FORWARD_PATH,
        "canonical_component_slots_version": CANONICAL_COMPONENT_SLOTS_VERSION,
        "local_target_version": V5_LOCAL_TARGET_VERSION,
        "solution_target_sampler_version": V5_SOLUTION_TARGET_SAMPLER_VERSION,
        "amplitude_query_sampler_version": V5_AMPLITUDE_QUERY_SAMPLER_VERSION,
    }
    if any(payload[name] != value for name, value in expected_constants.items()):
        raise ValueError("persisted clean-recipe contract identity is unsupported")
    if payload["branch_amplitude_constraint"] != recipe.amplitude_query.constraint_for_branch(
        resolution_present=recipe.amplitude.resolution_present
    ).to_audit_dict():
        raise ValueError("persisted branch amplitude constraint does not reproduce")
    return recipe


__all__ = [
    "V5PersistedCleanRecipe",
    "V5PersistedSolutionTarget",
    "persisted_v5_clean_recipe_from_json",
]
