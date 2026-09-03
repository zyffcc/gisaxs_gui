"""Deterministic solution-only V4 bounds-first shard generation semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
from typing import Mapping

import numpy as np

from .bounds_first_contract import (
    BOUNDS_EMBEDDING_DIM,
    BoundsFirstLabel,
    local_varying_mask,
)
from .bounds_first_dataset import (
    PLACEMENT_CODE,
    RANGE_CODE,
    TASK_CODE,
    sample_solution_label,
    sample_user_bounds,
)
from .bounds_first_schedule import (
    BOUNDS_BRANCH_SCHEDULE_VERSION,
    SCHEDULE_SEMANTICS_SHA256,
    TOPOLOGY_SCHEDULES,
    scheduled_branch_and_bounds,
)
from .branch_catalog import decode_branch_pattern
from .branch_codec import UNIT_CUBE_DIMENSIONS
from .canonical_branch_catalog import canonical_branch_pattern_is_valid
from .contract import TOPOLOGIES
from .preprocessing import DEFAULT_CONTRACT, preprocess_curve
from .simulation import (
    MIN_EFFECTIVE_AMPLITUDE_FRACTION,
    OBSERVATION_GRID_KINDS,
    GridProvenance,
    NoiseProvenance,
    SimulationRecipe,
    sample_observation_view,
    simulate_recipe,
)


SHARD_SCHEMA_VERSION = "gisaxs.posterior_v8.bounds_first_solution_shard/v6"
SHARD_GENERATOR_VERSION = "posterior_v8_bounds_first_solution_shards_v4"
SEED_SCHEME_VERSION = "posterior_v8_bounds_first_global_recipe_u64_v1"
SPLIT_POLICY_VERSION = "posterior_v8_recipe_grouped_interpolation_split_v1"
SOLUTION_ONLY_PHASE = "bounds_first_solution_only_interpolation_pilot"
SPLIT_NAMES = ("train", "tuning_validation", "calibration", "test")
SPLIT_CODE = {name: index for index, name in enumerate(SPLIT_NAMES)}
SPLIT_FRACTIONS = {
    "train": 0.75,
    "tuning_validation": 0.10,
    "calibration": 0.05,
    "test": 0.10,
}
GRID_KIND_CODE = {name: index for index, name in enumerate(OBSERVATION_GRID_KINDS)}
PILOT_LIMITATIONS = (
    "solution_only_no_certified_no_solution_or_ood_rows",
    "recipe_grouped_interpolation_pilot",
    "no_parameter_guard_band",
    "not_a_strong_holdout_or_ood_split",
    "synthetic_observation_policy_not_yet_calibrated_to_real_cut_data",
)
_UINT64_MASK = (1 << 64) - 1

ARRAY_ORDER = (
    "x",
    "point_mask",
    "global_features",
    "topology_id",
    "component_count",
    "branch_pattern_id",
    "task_kind",
    "truth_available",
    "target_local_unit",
    "global_reference_unit",
    "active_dimension_mask",
    "local_varying_mask",
    "bounds_embedding",
    "bounds_sha256",
    "range_regime",
    "bound_placement",
    "recipe_seed",
    "global_recipe_index",
    "view_index",
    "bounds_seed",
    "local_target_seed",
    "amplitude_seed",
    "observation_seed",
    "recipe_group_id",
    "assigned_split",
    "grid_kind",
    "q_window_id",
    "noise_id",
    "mask_id",
    "crop_id",
    "raw_point_count",
    "q_window",
    "preprocess_q_range",
    "poisson_count_scale",
    "relative_sigma",
    "point_keep_probability",
)


def _integer(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class BoundsFirstShardConfig:
    master_seed: int = 20260903
    topology_schedule: str = "all34"
    views_per_recipe: int = 3
    max_raw_points: int = 512

    def __post_init__(self) -> None:
        seed = _integer(self.master_seed, "master_seed")
        if seed > _UINT64_MASK:
            raise ValueError("master_seed must fit in uint64")
        schedule = str(self.topology_schedule).strip().lower()
        if schedule not in TOPOLOGY_SCHEDULES:
            raise ValueError(f"topology_schedule must be one of {tuple(TOPOLOGY_SCHEDULES)}")
        views = _integer(self.views_per_recipe, "views_per_recipe", positive=True)
        points = _integer(self.max_raw_points, "max_raw_points", positive=True)
        if not 2 <= views <= 16:
            raise ValueError("views_per_recipe must be in [2, 16]")
        if not 128 <= points <= DEFAULT_CONTRACT.max_points:
            raise ValueError(
                f"max_raw_points must be in [128, {DEFAULT_CONTRACT.max_points}]"
            )
        object.__setattr__(self, "master_seed", seed)
        object.__setattr__(self, "topology_schedule", schedule)
        object.__setattr__(self, "views_per_recipe", views)
        object.__setattr__(self, "max_raw_points", points)


@dataclass(frozen=True)
class BoundsFirstShardSpec:
    shard_index: int
    start_recipe_index: int
    recipe_count: int

    def __post_init__(self) -> None:
        shard = _integer(self.shard_index, "shard_index")
        start = _integer(self.start_recipe_index, "start_recipe_index")
        count = _integer(self.recipe_count, "recipe_count", positive=True)
        if start + count > 1 << 63:
            raise ValueError("global recipe index range must fit in signed int64")
        object.__setattr__(self, "shard_index", shard)
        object.__setattr__(self, "start_recipe_index", start)
        object.__setattr__(self, "recipe_count", count)

    @property
    def stop_recipe_index(self) -> int:
        return self.start_recipe_index + self.recipe_count


@dataclass(frozen=True)
class BoundsFirstCleanRecipe:
    global_recipe_index: int
    recipe_seed: int
    bounds_seed: int
    local_target_seed: int
    amplitude_seed: int
    recipe_group_id: str
    assigned_split: str
    topology_occurrence: int
    bounds_combo_id: int
    label: BoundsFirstLabel
    simulation_recipe: SimulationRecipe
    branch_pattern_id: int


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def recipe_seed_for(master_seed: int, global_recipe_index: int) -> int:
    seed = _integer(master_seed, "master_seed")
    index = _integer(global_recipe_index, "global_recipe_index")
    if seed > _UINT64_MASK or index > _UINT64_MASK:
        raise ValueError("master seed and global recipe index must fit in uint64")
    return (_splitmix64(seed) + index) & _UINT64_MASK


def _derived_seed(recipe_seed: int, namespace: int) -> int:
    return _splitmix64(_integer(recipe_seed, "recipe_seed") ^ int(namespace))


def recipe_group_id_for(config: BoundsFirstShardConfig, global_recipe_index: int) -> str:
    if not isinstance(config, BoundsFirstShardConfig):
        raise TypeError("config must be BoundsFirstShardConfig")
    payload = {
        "split_policy_version": SPLIT_POLICY_VERSION,
        "seed_scheme": SEED_SCHEME_VERSION,
        "bounds_branch_schedule": BOUNDS_BRANCH_SCHEDULE_VERSION,
        "bounds_branch_schedule_sha256": SCHEDULE_SEMANTICS_SHA256,
        "master_seed": config.master_seed,
        "topology_schedule": config.topology_schedule,
        "global_recipe_index": _integer(global_recipe_index, "global_recipe_index"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(canonical.encode("utf-8")).hexdigest()


def split_for_recipe_group(recipe_group_id: str) -> str:
    if (
        not isinstance(recipe_group_id, str)
        or len(recipe_group_id) != 64
        or any(character not in "0123456789abcdef" for character in recipe_group_id)
    ):
        raise ValueError("recipe_group_id must be a lowercase SHA-256 digest")
    bucket = int(recipe_group_id[-8:], 16) % 10_000
    if bucket < 7_500:
        return "train"
    if bucket < 8_500:
        return "tuning_validation"
    return "calibration" if bucket < 9_000 else "test"


def _amplitudes(label: BoundsFirstLabel, amplitude_seed: int):
    rng = np.random.default_rng(np.random.SeedSequence([amplitude_seed, 0x414D504C]))
    count = len(label.truth_components)
    remaining = 1.0 - count * MIN_EFFECTIVE_AMPLITUDE_FRACTION
    fractions = MIN_EFFECTIVE_AMPLITUDE_FRACTION + remaining * rng.dirichlet(
        np.full(count, 2.0)
    )
    total = float(np.exp(rng.uniform(np.log(1.0e2), np.log(1.0e4))))
    amplitudes = tuple(float(total * value) for value in fractions)
    background = total * float(np.exp(rng.uniform(np.log(1.0e-6), np.log(1.0e-3))))
    resolution_amplitude = (
        0.0
        if label.truth_resolution is None
        else total * float(np.exp(rng.uniform(np.log(0.01), np.log(0.20))))
    )
    return amplitudes, float(background), float(resolution_amplitude)


def reconstruct_clean_recipe(
    config: BoundsFirstShardConfig, global_recipe_index: int
) -> BoundsFirstCleanRecipe:
    """Reconstruct bounds then local truth without consulting an observation."""

    if not isinstance(config, BoundsFirstShardConfig):
        raise TypeError("config must be BoundsFirstShardConfig")
    index = _integer(global_recipe_index, "global_recipe_index")
    recipe_seed = recipe_seed_for(config.master_seed, index)
    bounds_seed = _derived_seed(recipe_seed, 0x424F554E)
    target_seed = _derived_seed(recipe_seed, 0x54415247)
    amplitude_seed = _derived_seed(recipe_seed, 0x414D504C)
    (
        topology_id,
        occurrence,
        pattern,
        combo_id,
        regime,
        placement,
    ) = scheduled_branch_and_bounds(config.topology_schedule, index)
    topology = TOPOLOGIES[topology_id]
    padded_d, resolution_present = decode_branch_pattern(pattern)
    d_present = padded_d[: len(topology)]
    bounds = sample_user_bounds(
        topology,
        d_present,
        resolution_present,
        regime=regime,
        placement=placement,
        bounds_seed=bounds_seed,
    )
    label = sample_solution_label(bounds, local_target_seed=target_seed)
    amplitudes, background, resolution_amplitude = _amplitudes(label, amplitude_seed)
    simulation = SimulationRecipe(
        topology_id=topology_id,
        components=label.truth_components,
        effective_amplitudes=amplitudes,
        resolution=label.truth_resolution,
        resolution_effective_amplitude=resolution_amplitude,
        background=background,
        grid=GridProvenance(n_points=64),
        noise=NoiseProvenance(poisson_count_scale=None, relative_sigma=0.0),
        seed=recipe_seed,
        require_characteristic_coverage=False,
    )
    if not canonical_branch_pattern_is_valid(
        topology_id, pattern
    ):  # pragma: no cover
        raise RuntimeError("generated a non-canonical topology/branch pair")
    group_id = recipe_group_id_for(config, index)
    return BoundsFirstCleanRecipe(
        index,
        recipe_seed,
        bounds_seed,
        target_seed,
        amplitude_seed,
        group_id,
        split_for_recipe_group(group_id),
        occurrence,
        combo_id,
        label,
        simulation,
        pattern,
    )


def recipe_record(recipe: BoundsFirstCleanRecipe) -> dict[str, object]:
    label, simulation = recipe.label, recipe.simulation_recipe
    return {
        "global_recipe_index": recipe.global_recipe_index,
        "recipe_seed": recipe.recipe_seed,
        "bounds_seed": recipe.bounds_seed,
        "local_target_seed": recipe.local_target_seed,
        "amplitude_seed": recipe.amplitude_seed,
        "recipe_group_id": recipe.recipe_group_id,
        "assigned_split": recipe.assigned_split,
        "task_kind": "in_domain_solution",
        "truth_available": True,
        "topology_id": simulation.topology_id,
        "branch_pattern_id": recipe.branch_pattern_id,
        "topology_occurrence": recipe.topology_occurrence,
        "bounds_combo_id": recipe.bounds_combo_id,
        "schedule_semantics_sha256": SCHEDULE_SEMANTICS_SHA256,
        "bounds": json.loads(label.bounds.canonical_json),
        "bounds_sha256": label.bounds.sha256,
        "target_local_unit": list(label.local_target_unit),
        "global_reference_unit": list(label.global_reference_unit),
        "truth_components": [asdict(value) for value in label.truth_components],
        "truth_resolution": (
            None if label.truth_resolution is None else asdict(label.truth_resolution)
        ),
        "effective_amplitudes": list(simulation.effective_amplitudes),
        "background": simulation.background,
        "resolution_effective_amplitude": simulation.resolution_effective_amplitude,
    }


def _empty_arrays(recipe_count: int, views_per_recipe: int) -> dict[str, np.ndarray]:
    count = recipe_count * views_per_recipe
    points = DEFAULT_CONTRACT.max_points
    return {
        "x": np.empty((count, points, 3), dtype=np.float32),
        "point_mask": np.empty((count, points), dtype=np.bool_),
        "global_features": np.empty((count, 5), dtype=np.float32),
        "topology_id": np.empty(count, dtype=np.int32),
        "component_count": np.empty(count, dtype=np.uint8),
        "branch_pattern_id": np.empty(count, dtype=np.int32),
        "task_kind": np.empty(count, dtype=np.uint8),
        "truth_available": np.empty(count, dtype=np.bool_),
        "target_local_unit": np.empty((count, UNIT_CUBE_DIMENSIONS), dtype=np.float32),
        "global_reference_unit": np.empty((count, UNIT_CUBE_DIMENSIONS), dtype=np.float32),
        "active_dimension_mask": np.empty((count, UNIT_CUBE_DIMENSIONS), dtype=np.bool_),
        "local_varying_mask": np.empty((count, UNIT_CUBE_DIMENSIONS), dtype=np.bool_),
        "bounds_embedding": np.empty((count, BOUNDS_EMBEDDING_DIM), dtype=np.float32),
        "bounds_sha256": np.empty(count, dtype="S64"),
        "range_regime": np.empty(count, dtype=np.uint8),
        "bound_placement": np.empty(count, dtype=np.uint8),
        "recipe_seed": np.empty(count, dtype=np.uint64),
        "global_recipe_index": np.empty(count, dtype=np.int64),
        "view_index": np.empty(count, dtype=np.int16),
        "bounds_seed": np.empty(count, dtype=np.uint64),
        "local_target_seed": np.empty(count, dtype=np.uint64),
        "amplitude_seed": np.empty(count, dtype=np.uint64),
        "observation_seed": np.empty(count, dtype=np.uint64),
        "recipe_group_id": np.empty(count, dtype="S64"),
        "assigned_split": np.empty(count, dtype=np.uint8),
        "grid_kind": np.empty(count, dtype=np.uint8),
        "q_window_id": np.empty(count, dtype=np.uint8),
        "noise_id": np.empty(count, dtype=np.uint8),
        "mask_id": np.empty(count, dtype=np.uint8),
        "crop_id": np.empty(count, dtype=np.uint8),
        "raw_point_count": np.empty(count, dtype=np.int16),
        "q_window": np.empty((count, 2), dtype=np.float32),
        "preprocess_q_range": np.empty((count, 2), dtype=np.float32),
        "poisson_count_scale": np.empty(count, dtype=np.float32),
        "relative_sigma": np.empty(count, dtype=np.float32),
        "point_keep_probability": np.empty(count, dtype=np.float32),
    }


def generate_shard_arrays(
    config: BoundsFirstShardConfig, spec: BoundsFirstShardSpec
) -> tuple[dict[str, np.ndarray], tuple[dict[str, object], ...]]:
    """Generate one deterministic shard; every row is a genuine solution view."""

    if not isinstance(config, BoundsFirstShardConfig) or not isinstance(
        spec, BoundsFirstShardSpec
    ):
        raise TypeError("config and spec must be V4 bounds-first shard contracts")
    arrays = _empty_arrays(spec.recipe_count, config.views_per_recipe)
    records = []
    row = 0
    for index in range(spec.start_recipe_index, spec.stop_recipe_index):
        clean = reconstruct_clean_recipe(config, index)
        records.append(recipe_record(clean))
        label, simulation = clean.label, clean.simulation_recipe
        active = np.asarray(label.bounds.local_codec().active_mask, dtype=np.bool_)
        varying = np.asarray(local_varying_mask(label.bounds), dtype=np.bool_)
        for view_index in range(config.views_per_recipe):
            view = sample_observation_view(
                clean.recipe_seed, view_index, max_points=config.max_raw_points
            )
            simulated = simulate_recipe(view.simulation_recipe(simulation))
            curve = preprocess_curve(
                simulated.q,
                simulated.intensity,
                simulated.sigma,
                mask=view.selection_mask(simulated.q),
                q_range=view.preprocess_q_range,
            )
            values = {
                "x": curve.x,
                "point_mask": curve.point_mask,
                "global_features": curve.global_features,
                "topology_id": simulation.topology_id,
                "component_count": simulation.component_count,
                "branch_pattern_id": clean.branch_pattern_id,
                "task_kind": TASK_CODE["in_domain_solution"],
                "truth_available": True,
                "target_local_unit": label.local_target_unit,
                "global_reference_unit": label.global_reference_unit,
                "active_dimension_mask": active,
                "local_varying_mask": varying,
                "bounds_embedding": label.bounds.embedding,
                "bounds_sha256": label.bounds.sha256.encode("ascii"),
                "range_regime": RANGE_CODE[label.bounds.range_regime],
                "bound_placement": PLACEMENT_CODE[label.bounds.placement],
                "recipe_seed": clean.recipe_seed,
                "global_recipe_index": index,
                "view_index": view_index,
                "bounds_seed": clean.bounds_seed,
                "local_target_seed": clean.local_target_seed,
                "amplitude_seed": clean.amplitude_seed,
                "observation_seed": view.observation_seed,
                "recipe_group_id": clean.recipe_group_id.encode("ascii"),
                "assigned_split": SPLIT_CODE[clean.assigned_split],
                "grid_kind": GRID_KIND_CODE[view.grid.kind],
                "q_window_id": view.q_window_id,
                "noise_id": view.noise_id,
                "mask_id": view.mask_id,
                "crop_id": view.crop_id,
                "raw_point_count": view.grid.n_points,
                "q_window": (view.grid.q_min, view.grid.q_max),
                "preprocess_q_range": view.preprocess_q_range,
                "poisson_count_scale": view.noise.poisson_count_scale,
                "relative_sigma": view.noise.relative_sigma,
                "point_keep_probability": view.point_keep_probability,
            }
            for name, value in values.items():
                arrays[name][row] = value
            row += 1
    validate_shard_semantics(config, spec, arrays, tuple(records))
    return arrays, tuple(records)


def array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(arrays[name].shape), "dtype": arrays[name].dtype.str}
        for name in ARRAY_ORDER
    }


def validate_shard_semantics(config, spec, arrays, records) -> None:
    if tuple(arrays) != ARRAY_ORDER:
        raise ValueError("V4 shard arrays have the wrong schema/order")
    expected_schema = array_schema(
        _empty_arrays(spec.recipe_count, config.views_per_recipe)
    )
    if array_schema(arrays) != expected_schema:
        raise ValueError("V4 shard array shape or dtype is inconsistent")
    if len(records) != spec.recipe_count:
        raise ValueError("recipe provenance count is inconsistent")
    if not np.all(arrays["truth_available"]) or np.any(
        arrays["task_kind"] != TASK_CODE["in_domain_solution"]
    ):
        raise ValueError("V4 solution-only shards cannot contain negative/OOD rows")
    if not np.all(np.isfinite(arrays["x"])) or not np.all(
        np.isfinite(arrays["global_features"])
    ):
        raise ValueError("curve tensors contain non-finite values")
    if np.any(
        np.count_nonzero(arrays["point_mask"], axis=1)
        < DEFAULT_CONTRACT.min_valid_points
    ):
        raise ValueError("a curve view has too few valid points")
    varying = arrays["local_varying_mask"]
    targets = arrays["target_local_unit"]
    if np.any(targets[varying] <= 0.0) or np.any(targets[varying] >= 1.0):
        raise ValueError("V4 density targets must lie inside the open local unit cube")
    for clean_offset, index in enumerate(
        range(spec.start_recipe_index, spec.stop_recipe_index)
    ):
        clean = reconstruct_clean_recipe(config, index)
        if records[clean_offset] != recipe_record(clean):
            raise ValueError("recipe physical/bounds provenance is not reproducible")
        start = clean_offset * config.views_per_recipe
        rows = slice(start, start + config.views_per_recipe)
        label, simulation = clean.label, clean.simulation_recipe
        expected_constant = {
            "topology_id": simulation.topology_id,
            "component_count": simulation.component_count,
            "branch_pattern_id": clean.branch_pattern_id,
            "target_local_unit": np.asarray(label.local_target_unit, dtype=np.float32),
            "global_reference_unit": np.asarray(label.global_reference_unit, dtype=np.float32),
            "active_dimension_mask": np.asarray(
                label.bounds.local_codec().active_mask, dtype=np.bool_
            ),
            "local_varying_mask": np.asarray(
                local_varying_mask(label.bounds), dtype=np.bool_
            ),
            "bounds_embedding": np.asarray(label.bounds.embedding, dtype=np.float32),
            "bounds_sha256": label.bounds.sha256.encode("ascii"),
            "range_regime": RANGE_CODE[label.bounds.range_regime],
            "bound_placement": PLACEMENT_CODE[label.bounds.placement],
            "recipe_seed": clean.recipe_seed,
            "global_recipe_index": index,
            "bounds_seed": clean.bounds_seed,
            "local_target_seed": clean.local_target_seed,
            "amplitude_seed": clean.amplitude_seed,
            "recipe_group_id": clean.recipe_group_id.encode("ascii"),
            "assigned_split": SPLIT_CODE[clean.assigned_split],
        }
        for name, value in expected_constant.items():
            expected = np.broadcast_to(value, arrays[name][rows].shape)
            if not np.array_equal(arrays[name][rows], expected):
                raise ValueError(f"recipe views disagree on {name}")
        for view_index, row in enumerate(range(start, start + config.views_per_recipe)):
            view = sample_observation_view(
                clean.recipe_seed, view_index, max_points=config.max_raw_points
            )
            expected_view = {
                "view_index": view_index,
                "observation_seed": view.observation_seed,
                "grid_kind": GRID_KIND_CODE[view.grid.kind],
                "q_window_id": view.q_window_id,
                "noise_id": view.noise_id,
                "mask_id": view.mask_id,
                "crop_id": view.crop_id,
                "raw_point_count": view.grid.n_points,
                "q_window": np.asarray((view.grid.q_min, view.grid.q_max), np.float32),
                "preprocess_q_range": np.asarray(view.preprocess_q_range, np.float32),
                "poisson_count_scale": np.float32(view.noise.poisson_count_scale),
                "relative_sigma": np.float32(view.noise.relative_sigma),
                "point_keep_probability": np.float32(view.point_keep_probability),
            }
            for name, value in expected_view.items():
                if not np.array_equal(arrays[name][row], value):
                    raise ValueError(f"observation-view provenance disagrees on {name}")


def split_policy() -> dict[str, object]:
    return {
        "version": SPLIT_POLICY_VERSION,
        "assignment_unit": "clean_global_recipe_group",
        "hash_inputs": [
            "split_policy_version",
            "seed_scheme_version",
            "bounds_branch_schedule_sha256",
            "master_seed",
            "topology_schedule",
            "global_recipe_index",
        ],
        "hash": "sha256",
        "hash_bucket_modulus": 10_000,
        "train_bucket_stop": 7_500,
        "tuning_validation_bucket_stop": 8_500,
        "calibration_bucket_stop": 9_000,
        "fractions": SPLIT_FRACTIONS,
        "codes": SPLIT_CODE,
        "all_observation_views_inherit_recipe_split": True,
        "classification": "recipe_grouped_interpolation_pilot",
        "no_parameter_guard_band": True,
        "strong_holdout_or_ood_claim_allowed": False,
    }
