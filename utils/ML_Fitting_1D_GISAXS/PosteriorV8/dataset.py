"""Phase-2 multi-view candidate shards for Posterior V8.

Clean physics is keyed only by ``(master_seed, recipe_index)``.  A coarse
physical cell assigns that recipe, and every deterministic observation view,
to exactly one of the versioned 75/10/5/10
train/validation/calibration/test splits.  This is a leakage-control candidate
pipeline, not an identifiability claim; competing-model screening remains a
later phase.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from numbers import Integral
import os
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

import numpy as np

from .branch_catalog import (
    BRANCH_CATALOG_VERSION,
    branch_pattern_id,
    branch_pattern_is_valid,
    decode_branch_pattern,
)
from .branch_codec import (
    BRANCH_CODEC_VERSION,
    INACTIVE_UNIT_VALUE,
    ProfiledBranchCodec,
    ResolutionBounds,
    UNIT_CUBE_DIMENSIONS,
)
from .compatibility_calibration import (
    COMPATIBILITY_STRATUM_FIELDS,
    COMPATIBILITY_STRATUM_VERSION,
    STRATIFICATION_SEMANTICS,
)
from .contract import (
    CODEC_VERSION,
    CONTRACT_VERSION,
    FORWARD_MODEL_VERSION,
    MAX_COMPONENTS,
    NUM_TOPOLOGIES,
    RESOLUTION_NU_DOMAIN,
    RESOLUTION_SIGMA_DOMAIN,
    TOPOLOGIES,
    full_component_bounds,
    gui_component_to_latent,
)
from .preprocessing import (
    DEFAULT_CONTRACT,
    PREPROCESSING_VERSION,
    preprocess_curve,
)
from .simulation import (
    OBSERVATION_COUNT_SCALES,
    OBSERVATION_CROP_PROFILES,
    OBSERVATION_GRID_KINDS,
    OBSERVATION_KEEP_PROBABILITIES,
    OBSERVATION_NOISE_PROFILES,
    OBSERVATION_Q_WINDOWS,
    OBSERVATION_RELATIVE_SIGMAS,
    OBSERVATION_STRATUM_VERSION,
    OBSERVATION_VIEW_VERSION,
    SIMULATION_VERSION,
    NoiseProvenance,
    SimulationRecipe,
    sample_identifiable_recipe,
    sample_observation_view,
    simulate_recipe,
)


DATASET_SCHEMA_VERSION = "gisaxs.posterior_v8.multiview_candidate_npz/v3"
DATASET_GENERATOR_VERSION = "posterior_v8_phase2_observation_range_dataset_v3"
SEED_SCHEME_VERSION = "posterior_v8_clean_recipe_u64_bijection_v2"
PHYSICAL_CELL_VERSION = "posterior_v8_coarse_physical_cell_v1"
SPLIT_POLICY_VERSION = "posterior_v8_grouped_train_tune_calibration_test_v2"
RANGE_GENERATOR_VERSION = "posterior_v8_target_containing_user_range_v1"
PILOT_PHASE = "phase2_observation_range_candidate_pilot"
PILOT_LIMITATIONS = (
    "no_competing_model_identifiability_screen",
    "coarse_cell_grouping_reduces_but_does_not_prove_metric_separation",
    "synthetic_observation_policy_not_yet_calibrated_to_real_cut_data",
    "linear_amplitude_targets_are_not_part_of_this_dataset_contract",
)
SUPPORTED_SPLITS = ("train", "validation", "calibration", "test")
SPLIT_CODE = {name: index for index, name in enumerate(SUPPORTED_SPLITS)}
SPLIT_FRACTIONS = {
    "train": 0.75,
    "validation": 0.10,
    "calibration": 0.05,
    "test": 0.10,
}
TOPOLOGY_SCHEDULES = {
    "k1": tuple(range(3)),
    "all34": tuple(range(NUM_TOPOLOGIES)),
}
RANGE_REGIMES = ("full", "wide", "narrow")
RANGE_CODE = {name: index for index, name in enumerate(RANGE_REGIMES)}
GRID_KIND_CODE = {name: index for index, name in enumerate(OBSERVATION_GRID_KINDS)}
PHYSICAL_CELL_BINS = 4
_UINT64_MASK = (1 << 64) - 1

ARRAY_ORDER = (
    "x",
    "point_mask",
    "global_features",
    "topology_id",
    "component_count",
    "branch_pattern_id",
    "target_unit",
    "active_dimension_mask",
    "branch_low",
    "branch_high",
    "range_regime",
    "range_seed",
    "recipe_seed",
    "recipe_index",
    "view_index",
    "observation_seed",
    "physical_cell_id",
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


def _integer(value: int, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class PilotDatasetConfig:
    """Scientific generation settings shared by every shard in one dataset."""

    master_seed: int = 20260902
    topology_schedule: str = "all34"
    views_per_recipe: int = 2
    max_raw_points: int = 512

    def __post_init__(self) -> None:
        seed = _integer(self.master_seed, "master_seed")
        if seed > _UINT64_MASK:
            raise ValueError("master_seed must fit in 64 bits")
        schedule = str(self.topology_schedule).strip().lower()
        if schedule not in TOPOLOGY_SCHEDULES:
            raise ValueError(f"topology_schedule must be one of {tuple(TOPOLOGY_SCHEDULES)}")
        views = _integer(self.views_per_recipe, "views_per_recipe", positive=True)
        if not 2 <= views <= 16:
            raise ValueError("views_per_recipe must be in [2, 16]")
        points = _integer(self.max_raw_points, "max_raw_points", positive=True)
        if not 128 <= points <= DEFAULT_CONTRACT.max_points:
            raise ValueError(f"max_raw_points must be in [128, {DEFAULT_CONTRACT.max_points}]")
        object.__setattr__(self, "master_seed", seed)
        object.__setattr__(self, "topology_schedule", schedule)
        object.__setattr__(self, "views_per_recipe", views)
        object.__setattr__(self, "max_raw_points", points)


@dataclass(frozen=True)
class ShardSpec:
    shard_index: int
    start_index: int
    recipe_count: int

    def __post_init__(self) -> None:
        shard_index = _integer(self.shard_index, "shard_index")
        start = _integer(self.start_index, "start_index")
        count = _integer(self.recipe_count, "recipe_count", positive=True)
        if start + count > 1 << 63:
            raise ValueError("recipe index range must fit in signed int64")
        object.__setattr__(self, "shard_index", shard_index)
        object.__setattr__(self, "start_index", start)
        object.__setattr__(self, "recipe_count", count)

    @property
    def stop_index(self) -> int:
        return self.start_index + self.recipe_count


def _splitmix64(value: int) -> int:
    value = (int(value) + 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & ((1 << 64) - 1)
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & ((1 << 64) - 1)
    return value ^ (value >> 31)


def recipe_seed_for(master_seed: int, recipe_index: int) -> int:
    """Return a split-independent uint64 seed, bijective in recipe index."""

    config_seed = _integer(master_seed, "master_seed")
    if config_seed > _UINT64_MASK:
        raise ValueError("master_seed must fit in 64 bits")
    index = _integer(recipe_index, "recipe_index")
    if index > _UINT64_MASK:
        raise ValueError("recipe_index must fit in 64 bits")
    return (_splitmix64(config_seed) + index) & _UINT64_MASK


def topology_id_for_index(schedule: str, recipe_index: int) -> int:
    name = str(schedule).strip().lower()
    try:
        catalog = TOPOLOGY_SCHEDULES[name]
    except KeyError as exc:
        raise ValueError(f"unknown topology schedule {schedule!r}") from exc
    index = _integer(recipe_index, "recipe_index")
    return catalog[index % len(catalog)]


def _clean_noise() -> NoiseProvenance:
    return NoiseProvenance(poisson_count_scale=None, relative_sigma=0.0)


def reconstruct_recipe(
    *, recipe_seed: int, topology_id: int, config: PilotDatasetConfig
) -> SimulationRecipe:
    """Rebuild a row's compact recipe without storing repeated recipe JSON."""

    if not isinstance(config, PilotDatasetConfig):
        raise TypeError("config must be a PilotDatasetConfig")
    return sample_identifiable_recipe(
        _integer(recipe_seed, "recipe_seed"),
        topology_id=_integer(topology_id, "topology_id"),
        max_points=64,
        noise=_clean_noise(),
    )


def _full_codec(recipe: SimulationRecipe) -> ProfiledBranchCodec:
    d_present = tuple(component.D is not None for component in recipe.components)
    resolution_bounds = (
        ResolutionBounds(RESOLUTION_SIGMA_DOMAIN, RESOLUTION_NU_DOMAIN)
        if recipe.resolution is not None
        else None
    )
    return ProfiledBranchCodec.build(
        TOPOLOGIES[recipe.topology_id],
        tuple(
            full_component_bounds(shape, d_policy="optional")
            for shape in TOPOLOGIES[recipe.topology_id]
        ),
        d_present,
        resolution_bounds=resolution_bounds,
    )


def _branch_pattern(recipe: SimulationRecipe) -> int:
    d_flags = tuple(component.D is not None for component in recipe.components)
    padded = d_flags + (False,) * (MAX_COMPONENTS - len(d_flags))
    return branch_pattern_id(padded, recipe.resolution is not None)


def _unit_bins(values: np.ndarray, active: np.ndarray) -> tuple[int, ...]:
    clipped = np.minimum(np.asarray(values)[active], np.nextafter(1.0, 0.0))
    return tuple(int(value) for value in np.floor(clipped * PHYSICAL_CELL_BINS))


def _log_ratio_bin(value: float, low: float, high: float) -> int:
    coordinate = (np.log10(value) - low) / (high - low)
    return int(np.floor(np.clip(coordinate, 0.0, np.nextafter(1.0, 0.0)) * PHYSICAL_CELL_BINS))


def physical_cell_id(recipe: SimulationRecipe, target_unit: Sequence[float]) -> str:
    """Hash a reconstructible coarse physical cell, independent of observations."""

    if not isinstance(recipe, SimulationRecipe):
        raise TypeError("recipe must be a SimulationRecipe")
    codec = _full_codec(recipe)
    target = np.asarray(target_unit, dtype=np.float64)
    if target.shape != (UNIT_CUBE_DIMENSIONS,):
        raise ValueError("target_unit has the wrong shape")
    active = np.asarray(codec.active_mask, dtype=np.bool_)
    total = float(np.sum(recipe.effective_amplitudes))
    fractions = np.asarray(recipe.effective_amplitudes, dtype=np.float64) / total
    amplitude_bins = tuple(
        int(value)
        for value in np.floor(np.minimum(fractions, np.nextafter(1.0, 0.0)) * PHYSICAL_CELL_BINS)
    )
    resolution_ratio_bin = -1
    if recipe.resolution is not None:
        resolution_ratio_bin = _log_ratio_bin(
            recipe.resolution_effective_amplitude / total, -2.0, np.log10(0.20)
        )
    cell = {
        "version": PHYSICAL_CELL_VERSION,
        "topology_id": recipe.topology_id,
        "branch_pattern_id": _branch_pattern(recipe),
        "nonlinear_unit_bins": _unit_bins(target, active),
        "particle_fraction_bins": amplitude_bins,
        "background_ratio_bin": _log_ratio_bin(recipe.background / total, -6.0, -3.0),
        "resolution_ratio_bin": resolution_ratio_bin,
    }
    return sha256(_canonical_json(cell)).hexdigest()


def split_for_physical_cell(cell_id: str) -> str:
    """Assign a coarse physical cell to the stable 75/10/5/10 four-way split."""

    if (
        not isinstance(cell_id, str)
        or len(cell_id) != 64
        or any(character not in "0123456789abcdef" for character in cell_id)
    ):
        raise ValueError("cell_id must be a lowercase SHA-256 hex digest")
    bucket = int(cell_id[-8:], 16) % 10_000
    if bucket < 7_500:
        return "train"
    if bucket < 8_500:
        return "validation"
    return "calibration" if bucket < 9_000 else "test"


def _derived_seed(recipe_seed: int, view_index: int, namespace: int) -> int:
    state = np.random.SeedSequence(
        [_integer(recipe_seed, "recipe_seed"), _integer(view_index, "view_index"), namespace]
    ).generate_state(2, dtype=np.uint32)
    return int(state[0]) | (int(state[1]) << 32)


def user_range_for_target(
    target_unit: Sequence[float],
    active_mask: Sequence[bool],
    regime: str,
    range_seed: int,
    *,
    codec: ProfiledBranchCodec,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a target-containing normalized user range for one hard branch."""

    name = str(regime).strip().lower()
    if name not in RANGE_CODE:
        raise ValueError(f"range regime must be one of {RANGE_REGIMES}")
    target = np.asarray(target_unit, dtype=np.float64)
    active = np.asarray(active_mask)
    if target.shape != (UNIT_CUBE_DIMENSIONS,) or active.shape != target.shape:
        raise ValueError("target_unit and active_mask must both have shape (26,)")
    if active.dtype.kind != "b" or tuple(bool(value) for value in active) != codec.active_mask:
        raise ValueError("active_mask does not match the hard branch codec")
    if np.any(target[active] < 0.0) or np.any(target[active] > 1.0):
        raise ValueError("active target values must lie in [0, 1]")
    if np.any(target[~active] != INACTIVE_UNIT_VALUE):
        raise ValueError("inactive target values must equal 0.5")
    low = np.full(UNIT_CUBE_DIMENSIONS, INACTIVE_UNIT_VALUE, dtype=np.float64)
    high = low.copy()
    if name == "full":
        low[active], high[active] = 0.0, 1.0
    else:
        rng = np.random.default_rng(
            np.random.SeedSequence([_integer(range_seed, "range_seed"), 0x52414E47])
        )
        margin = (0.22, 0.55) if name == "wide" else (0.025, 0.15)
        indices = np.flatnonzero(active)
        low[indices] = np.maximum(0.0, target[indices] - rng.uniform(*margin, indices.size))
        high[indices] = np.minimum(1.0, target[indices] + rng.uniform(*margin, indices.size))
    if np.any(low[active] > target[active]) or np.any(high[active] < target[active]):
        raise RuntimeError("generated user range does not contain its truth target")
    validation_high = high.copy()
    validation_high[active & (validation_high == 1.0)] = np.nextafter(1.0, 0.0)
    try:
        codec.decode(low)
        codec.decode(validation_high)
    except (TypeError, ValueError) as exc:
        raise ValueError("generated range is invalid for its hard branch") from exc
    return low.astype(np.float32), high.astype(np.float32)


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
        "target_unit": np.empty((count, UNIT_CUBE_DIMENSIONS), dtype=np.float32),
        "active_dimension_mask": np.empty((count, UNIT_CUBE_DIMENSIONS), dtype=np.bool_),
        "branch_low": np.empty((count, UNIT_CUBE_DIMENSIONS), dtype=np.float32),
        "branch_high": np.empty((count, UNIT_CUBE_DIMENSIONS), dtype=np.float32),
        "range_regime": np.empty(count, dtype=np.uint8),
        "range_seed": np.empty(count, dtype=np.uint64),
        "recipe_seed": np.empty(count, dtype=np.uint64),
        "recipe_index": np.empty(count, dtype=np.int64),
        "view_index": np.empty(count, dtype=np.int16),
        "observation_seed": np.empty(count, dtype=np.uint64),
        "physical_cell_id": np.empty(count, dtype="S64"),
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


def generate_shard_arrays(config: PilotDatasetConfig, spec: ShardSpec) -> dict[str, np.ndarray]:
    """Generate one complete in-memory shard using only authoritative primitives."""

    if not isinstance(config, PilotDatasetConfig) or not isinstance(spec, ShardSpec):
        raise TypeError("config and spec must be PilotDatasetConfig and ShardSpec")
    arrays = _empty_arrays(spec.recipe_count, config.views_per_recipe)
    row = 0
    for recipe_index in range(spec.start_index, spec.stop_index):
        topology_id = topology_id_for_index(config.topology_schedule, recipe_index)
        recipe_seed = recipe_seed_for(config.master_seed, recipe_index)
        recipe = reconstruct_recipe(recipe_seed=recipe_seed, topology_id=topology_id, config=config)
        for shape in set(TOPOLOGIES[topology_id]):
            radii = [component.R for component in recipe.components if component.shape == shape]
            if radii != sorted(radii):
                raise ValueError("same-shape components violate canonical radius order")
        codec = _full_codec(recipe)
        coordinates = codec.encode(
            tuple(gui_component_to_latent(component) for component in recipe.components),
            recipe.resolution,
        )
        pattern_id = _branch_pattern(recipe)
        if not branch_pattern_is_valid(topology_id, pattern_id):  # pragma: no cover
            raise RuntimeError("generated an invalid topology/branch pair")
        active = np.asarray(coordinates.active_mask, dtype=np.bool_)
        target = np.asarray(coordinates.unit_cube, dtype=np.float64)
        cell_id = physical_cell_id(recipe, target)
        assigned_split = split_for_physical_cell(cell_id)
        for view_index in range(config.views_per_recipe):
            view = sample_observation_view(
                recipe_seed, view_index, max_points=config.max_raw_points
            )
            simulated = simulate_recipe(view.simulation_recipe(recipe))
            curve = preprocess_curve(
                simulated.q,
                simulated.intensity,
                simulated.sigma,
                mask=view.selection_mask(simulated.q),
                q_range=view.preprocess_q_range,
            )
            regime = RANGE_REGIMES[(recipe_index + view_index) % len(RANGE_REGIMES)]
            range_seed = _derived_seed(recipe_seed, view_index, 0x52414E47)
            branch_low, branch_high = user_range_for_target(
                target, active, regime, range_seed, codec=codec
            )
            arrays["x"][row] = curve.x
            arrays["point_mask"][row] = curve.point_mask
            arrays["global_features"][row] = curve.global_features
            arrays["topology_id"][row] = topology_id
            arrays["component_count"][row] = len(recipe.components)
            arrays["branch_pattern_id"][row] = pattern_id
            arrays["target_unit"][row] = target
            arrays["active_dimension_mask"][row] = active
            arrays["branch_low"][row] = branch_low
            arrays["branch_high"][row] = branch_high
            arrays["range_regime"][row] = RANGE_CODE[regime]
            arrays["range_seed"][row] = range_seed
            arrays["recipe_seed"][row] = recipe_seed
            arrays["recipe_index"][row] = recipe_index
            arrays["view_index"][row] = view_index
            arrays["observation_seed"][row] = view.observation_seed
            arrays["physical_cell_id"][row] = cell_id.encode("ascii")
            arrays["assigned_split"][row] = SPLIT_CODE[assigned_split]
            arrays["grid_kind"][row] = GRID_KIND_CODE[view.grid.kind]
            arrays["q_window_id"][row] = view.q_window_id
            arrays["noise_id"][row] = view.noise_id
            arrays["mask_id"][row] = view.mask_id
            arrays["crop_id"][row] = view.crop_id
            arrays["raw_point_count"][row] = view.grid.n_points
            arrays["q_window"][row] = (view.grid.q_min, view.grid.q_max)
            arrays["preprocess_q_range"][row] = view.preprocess_q_range
            arrays["poisson_count_scale"][row] = view.noise.poisson_count_scale
            arrays["relative_sigma"][row] = view.noise.relative_sigma
            arrays["point_keep_probability"][row] = view.point_keep_probability
            row += 1
    return arrays


def _array_schema(arrays: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    return {
        name: {"shape": list(arrays[name].shape), "dtype": arrays[name].dtype.str}
        for name in ARRAY_ORDER
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


@dataclass(frozen=True)
class NumpyShard:
    path: Path
    metadata: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]

    @property
    def sample_count(self) -> int:
        return int(self.arrays["topology_id"].shape[0])

    def split_mask(self, split: str | None = None) -> np.ndarray:
        if split is None:
            return np.ones(self.sample_count, dtype=np.bool_)
        name = str(split).strip().lower()
        if name not in SPLIT_CODE:
            raise ValueError(f"split must be one of {SUPPORTED_SPLITS}")
        return self.arrays["assigned_split"] == SPLIT_CODE[name]

    def calibration_stratum_values(self, row: int) -> dict[str, object]:
        return calibration_stratum_values(self.arrays, row)

    def training_data(
        self, *, split: str | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return model inputs and supervised labels without importing TensorFlow."""

        selected = self.split_mask(split)
        count = int(np.count_nonzero(selected))
        topology_id = self.arrays["topology_id"][selected]
        pattern_id = self.arrays["branch_pattern_id"][selected]
        topology = np.zeros((count, NUM_TOPOLOGIES), dtype=np.float32)
        topology[np.arange(count), topology_id] = 1.0
        d_present = np.zeros((count, MAX_COMPONENTS), dtype=np.float32)
        resolution_present = np.zeros((count, 1), dtype=np.float32)
        for row, value in enumerate(pattern_id):
            flags, resolution = decode_branch_pattern(int(value))
            d_present[row] = flags
            resolution_present[row, 0] = resolution
        inputs = {
            "x": self.arrays["x"][selected],
            "point_mask": self.arrays["point_mask"][selected],
            "global_features": self.arrays["global_features"][selected],
            "branch_topology": topology,
            "branch_d_present": d_present,
            "branch_resolution_present": resolution_present,
            "branch_low": self.arrays["branch_low"][selected],
            "branch_high": self.arrays["branch_high"][selected],
            "active_dimension_mask": self.arrays["active_dimension_mask"][selected].astype(
                np.float32
            ),
        }
        labels = {
            "topology_id": topology_id,
            "branch_pattern_id": pattern_id,
            "target_unit": self.arrays["target_unit"][selected],
            "active_dimension_mask": self.arrays["active_dimension_mask"][selected].astype(
                np.float32
            ),
        }
        return inputs, labels


def _expected_versions() -> dict[str, str]:
    return {
        "contract": CONTRACT_VERSION,
        "parameter_codec": CODEC_VERSION,
        "forward_model": FORWARD_MODEL_VERSION,
        "branch_catalog": BRANCH_CATALOG_VERSION,
        "branch_codec": BRANCH_CODEC_VERSION,
        "simulation": SIMULATION_VERSION,
        "observation_view": OBSERVATION_VIEW_VERSION,
        "observation_stratum": OBSERVATION_STRATUM_VERSION,
        "preprocessing": PREPROCESSING_VERSION,
        "seed_scheme": SEED_SCHEME_VERSION,
        "physical_cell": PHYSICAL_CELL_VERSION,
        "split_policy": SPLIT_POLICY_VERSION,
        "range_generator": RANGE_GENERATOR_VERSION,
    }


def _observation_policy(config: PilotDatasetConfig) -> dict[str, object]:
    return {
        "views_per_clean_recipe": config.views_per_recipe,
        "max_raw_points": config.max_raw_points,
        "grid_kinds": list(OBSERVATION_GRID_KINDS),
        "q_windows": [list(value) for value in OBSERVATION_Q_WINDOWS],
        "noise_profiles": [list(value) for value in OBSERVATION_NOISE_PROFILES],
        "poisson_count_scales": list(OBSERVATION_COUNT_SCALES),
        "relative_sigmas": list(OBSERVATION_RELATIVE_SIGMAS),
        "point_keep_probabilities": list(OBSERVATION_KEEP_PROBABILITIES),
        "crop_profiles": [list(value) for value in OBSERVATION_CROP_PROFILES],
        "policy_inputs": ["clean_recipe_seed", "view_index", "max_raw_points"],
        "unknown_physical_parameter_access": False,
        "missing_characteristic_scales_are_retained_as_ambiguity_or_ood": True,
        "calibration_stratum_fields": [
            "raw_point_count",
            "noise_id",
            "q_window_id",
        ],
        "compatibility_stratum_version": COMPATIBILITY_STRATUM_VERSION,
        "compatibility_stratum_fields": list(COMPATIBILITY_STRATUM_FIELDS),
        "compatibility_stratification_semantics": STRATIFICATION_SEMANTICS,
        "compatibility_stratum_mapping": {
            "point_count": "raw_point_count",
            "noise_id": f"{OBSERVATION_STRATUM_VERSION}:noise-<noise_id>",
            "q_window_id": f"{OBSERVATION_STRATUM_VERSION}:q-window-<q_window_id>",
        },
        "deterministic_crop_and_point_mask": True,
        "preprocessing_entrypoint": "PosteriorV8.preprocessing.preprocess_curve",
    }


def _split_policy() -> dict[str, object]:
    return {
        "assignment_unit": "coarse_physical_parameter_cell",
        "hash": "sha256",
        "hash_bucket_modulus": 10_000,
        "train_bucket_stop": 7_500,
        "validation_bucket_stop": 8_500,
        "calibration_bucket_stop": 9_000,
        "cell_bins_per_active_axis": PHYSICAL_CELL_BINS,
        "cell_features": [
            "topology_id",
            "branch_pattern_id",
            "active_full_range_branch_unit_coordinates",
            "particle_effective_amplitude_fractions",
            "log10_background_to_particle_amplitude_ratio",
            "log10_resolution_to_particle_amplitude_ratio_when_present",
        ],
        "fractions": SPLIT_FRACTIONS,
        "codes": SPLIT_CODE,
        "recipe_seed_is_split_independent": True,
    }


def _range_policy() -> dict[str, object]:
    return {
        "space": "global_branch_unit_cube",
        "regime_codes": RANGE_CODE,
        "full_active_bounds": [0.0, 1.0],
        "wide_margin_interval": [0.22, 0.55],
        "narrow_margin_interval": [0.025, 0.15],
        "inactive_value": INACTIVE_UNIT_VALUE,
        "truth_containment_required": True,
        "discrete_branch_changes": "reject",
    }


def calibration_stratum_values(arrays: Mapping[str, np.ndarray], row: int) -> dict[str, object]:
    """Map one row to observable acquisition fields, excluding truth/model K."""

    index = _integer(row, "row")
    if index >= arrays["topology_id"].shape[0]:
        raise IndexError("row is outside the shard")
    q_window_id = int(arrays["q_window_id"][index])
    noise_id = int(arrays["noise_id"][index])
    if not 0 <= q_window_id < len(OBSERVATION_Q_WINDOWS):
        raise ValueError("row has an invalid q_window_id")
    if not 0 <= noise_id < len(OBSERVATION_NOISE_PROFILES):
        raise ValueError("row has an invalid noise_id")
    return {
        "point_count": int(arrays["raw_point_count"][index]),
        "noise_id": f"{OBSERVATION_STRATUM_VERSION}:noise-{noise_id}",
        "q_window_id": f"{OBSERVATION_STRATUM_VERSION}:q-window-{q_window_id}",
    }


def _validate_loaded(
    path: Path, metadata: Mapping[str, object], arrays: Mapping[str, np.ndarray]
) -> None:
    if metadata.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            "unsupported dataset schema version; pre-four-way-split v1/v2 shards "
            "are intentionally rejected by the Phase-2 loader"
        )
    if metadata.get("dataset_generator_version") != DATASET_GENERATOR_VERSION:
        raise ValueError("unsupported dataset generator version")
    if metadata.get("phase") != PILOT_PHASE:
        raise ValueError("unsupported or overstated dataset phase")
    if tuple(metadata.get("pilot_limitations", ())) != PILOT_LIMITATIONS:
        raise ValueError("pilot limitation metadata is missing or inconsistent")
    if metadata.get("versions") != _expected_versions():
        raise ValueError("dataset scientific version contract does not match this loader")
    if metadata.get("preprocessing_contract") != asdict(DEFAULT_CONTRACT):
        raise ValueError("dataset preprocessing contract does not match this loader")
    sources = metadata.get("source_sha256")
    if (
        not isinstance(sources, dict)
        or not sources
        or not all(
            isinstance(name, str)
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for name, digest in sources.items()
        )
        or metadata.get("source_sha256_aggregate") != sha256(_canonical_json(sources)).hexdigest()
    ):
        raise ValueError("source SHA-256 provenance is missing or inconsistent")
    try:
        config = PilotDatasetConfig(**metadata["config"])
        shard_data = metadata["shard"]
        spec = ShardSpec(
            shard_data["shard_index"],
            shard_data["start_index"],
            shard_data["recipe_count"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid dataset config or shard metadata") from exc
    if metadata.get("observation_generation") != _observation_policy(config):
        raise ValueError("observation-generation metadata is inconsistent")
    if metadata.get("split_assignment") != _split_policy():
        raise ValueError("physical-cell split metadata is inconsistent")
    if metadata.get("range_generation") != _range_policy():
        raise ValueError("range-generation metadata is inconsistent")
    expected_rows = spec.recipe_count * config.views_per_recipe
    if (
        shard_data.get("stop_index_exclusive") != spec.stop_index
        or shard_data.get("row_count") != expected_rows
        or shard_data.get("file_name") != path.name
    ):
        raise ValueError("shard range or file name metadata is inconsistent")
    if tuple(arrays) != ARRAY_ORDER:
        raise ValueError("NPZ fields do not match the versioned dataset schema")
    if metadata.get("array_schema") != _array_schema(arrays):
        raise ValueError("NPZ array shape or dtype does not match metadata")
    expected_shapes = _array_schema(_empty_arrays(spec.recipe_count, config.views_per_recipe))
    if metadata["array_schema"] != expected_shapes:
        raise ValueError("NPZ array shape or dtype does not match the V8 schema")
    clean_indices = np.arange(spec.start_index, spec.stop_index, dtype=np.int64)
    indices = np.repeat(clean_indices, config.views_per_recipe)
    views = np.tile(np.arange(config.views_per_recipe, dtype=np.int16), spec.recipe_count)
    if not np.array_equal(arrays["recipe_index"], indices):
        raise ValueError("recipe_index does not match the declared contiguous range")
    if not np.array_equal(arrays["view_index"], views):
        raise ValueError("view_index does not enumerate every clean recipe view")
    expected_seeds = np.asarray(
        [recipe_seed_for(config.master_seed, int(index)) for index in indices],
        dtype=np.uint64,
    )
    if not np.array_equal(arrays["recipe_seed"], expected_seeds):
        raise ValueError("recipe seeds do not match clean recipe-index derivation")
    expected_topologies = np.asarray(
        [topology_id_for_index(config.topology_schedule, int(index)) for index in indices],
        dtype=np.int32,
    )
    if not np.array_equal(arrays["topology_id"], expected_topologies):
        raise ValueError("topology ids do not match the balanced schedule")
    topology_values, topology_counts = np.unique(arrays["topology_id"], return_counts=True)
    topology_histogram = {
        str(int(key)): int(value) for key, value in zip(topology_values, topology_counts)
    }
    pattern_values, pattern_counts = np.unique(arrays["branch_pattern_id"], return_counts=True)
    pattern_histogram = {
        str(int(key)): int(value) for key, value in zip(pattern_values, pattern_counts)
    }
    if metadata.get("topology_histogram") != topology_histogram:
        raise ValueError("topology histogram does not match shard values")
    if metadata.get("branch_pattern_histogram") != pattern_histogram:
        raise ValueError("branch-pattern histogram does not match shard values")
    for field, metadata_name, upper in (
        ("assigned_split", "split_histogram", len(SPLIT_CODE)),
        ("range_regime", "range_regime_histogram", len(RANGE_CODE)),
        ("grid_kind", "grid_kind_histogram", len(GRID_KIND_CODE)),
        ("q_window_id", "q_window_histogram", len(OBSERVATION_Q_WINDOWS)),
        ("noise_id", "noise_histogram", len(OBSERVATION_NOISE_PROFILES)),
        ("mask_id", "mask_histogram", len(OBSERVATION_KEEP_PROBABILITIES)),
        ("crop_id", "crop_histogram", len(OBSERVATION_CROP_PROFILES)),
    ):
        if np.any(arrays[field] >= upper):
            raise ValueError(f"{field} contains an unknown code")
        keys, counts = np.unique(arrays[field], return_counts=True)
        histogram = {str(int(key)): int(value) for key, value in zip(keys, counts)}
        if metadata.get(metadata_name) != histogram:
            raise ValueError(f"{metadata_name} does not match shard values")
    if metadata.get("physical_cell_count") != int(np.unique(arrays["physical_cell_id"]).size):
        raise ValueError("physical-cell count does not match shard values")
    if metadata.get("recipe_seed_range") != {
        "minimum": int(np.min(expected_seeds)),
        "maximum": int(np.max(expected_seeds)),
    }:
        raise ValueError("recipe seed range does not match shard values")
    if (
        not np.all(np.isfinite(arrays["x"]))
        or not np.all(np.isfinite(arrays["global_features"]))
        or np.any(
            np.count_nonzero(arrays["point_mask"], axis=1) < DEFAULT_CONTRACT.min_valid_points
        )
    ):
        raise ValueError("preprocessed curve tensors are non-finite or under-populated")
    for clean_row, recipe_index in enumerate(clean_indices):
        first = clean_row * config.views_per_recipe
        rows = slice(first, first + config.views_per_recipe)
        topology_id = int(arrays["topology_id"][first])
        pattern_id = int(arrays["branch_pattern_id"][first])
        if not branch_pattern_is_valid(topology_id, pattern_id):
            raise ValueError("shard contains an invalid topology/branch pair")
        recipe = reconstruct_recipe(
            recipe_seed=int(expected_seeds[first]), topology_id=topology_id, config=config
        )
        codec = _full_codec(recipe)
        expected_coordinates = codec.encode(
            tuple(gui_component_to_latent(component) for component in recipe.components),
            recipe.resolution,
        )
        active = np.asarray(codec.active_mask, dtype=np.bool_)
        target = np.asarray(expected_coordinates.unit_cube, dtype=np.float32)
        cell_id = physical_cell_id(recipe, expected_coordinates.unit_cube)
        split = split_for_physical_cell(cell_id)
        if not np.all(arrays["topology_id"][rows] == topology_id):
            raise ValueError("views of one recipe disagree on topology")
        if not np.all(arrays["component_count"][rows] == len(recipe.components)):
            raise ValueError("component_count does not match clean topology")
        if not np.all(arrays["branch_pattern_id"][rows] == pattern_id):
            raise ValueError("views of one recipe disagree on branch pattern")
        if not np.all(arrays["active_dimension_mask"][rows] == active):
            raise ValueError("active dimension mask does not match the hard branch")
        if not np.all(arrays["target_unit"][rows] == target):
            raise ValueError("views of one recipe disagree on encoded truth")
        if not np.all(arrays["physical_cell_id"][rows] == cell_id.encode("ascii")):
            raise ValueError("physical cell provenance does not match clean truth")
        if not np.all(arrays["assigned_split"][rows] == SPLIT_CODE[split]):
            raise ValueError("all views of a physical cell must share its assigned split")
        for view_index, row in enumerate(range(first, first + config.views_per_recipe)):
            view = sample_observation_view(
                int(expected_seeds[first]), view_index, max_points=config.max_raw_points
            )
            if (
                int(arrays["observation_seed"][row]) != view.observation_seed
                or int(arrays["grid_kind"][row]) != GRID_KIND_CODE[view.grid.kind]
                or int(arrays["q_window_id"][row]) != view.q_window_id
                or int(arrays["noise_id"][row]) != view.noise_id
                or int(arrays["mask_id"][row]) != view.mask_id
                or int(arrays["crop_id"][row]) != view.crop_id
                or int(arrays["raw_point_count"][row]) != view.grid.n_points
                or not np.array_equal(
                    arrays["q_window"][row],
                    np.asarray((view.grid.q_min, view.grid.q_max), dtype=np.float32),
                )
                or not np.array_equal(
                    arrays["preprocess_q_range"][row],
                    np.asarray(view.preprocess_q_range, dtype=np.float32),
                )
                or arrays["poisson_count_scale"][row] != np.float32(view.noise.poisson_count_scale)
                or arrays["relative_sigma"][row] != np.float32(view.noise.relative_sigma)
                or arrays["point_keep_probability"][row] != np.float32(view.point_keep_probability)
            ):
                raise ValueError("observation-view provenance is inconsistent")
            regime = RANGE_REGIMES[int(arrays["range_regime"][row])]
            range_seed = _derived_seed(int(expected_seeds[first]), view_index, 0x52414E47)
            low, high = user_range_for_target(
                expected_coordinates.unit_cube,
                active,
                regime,
                range_seed,
                codec=codec,
            )
            if (
                int(arrays["range_seed"][row]) != range_seed
                or not np.array_equal(arrays["branch_low"][row], low)
                or not np.array_equal(arrays["branch_high"][row], high)
            ):
                raise ValueError("user-range provenance is inconsistent")


def load_shard(path: str | os.PathLike[str]) -> NumpyShard:
    """Load one shard and fail closed on checksum, version, shape, or semantics."""

    npz_path = Path(path)
    metadata_path = npz_path.with_suffix(".json")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read shard metadata {metadata_path}") from exc
    if _file_sha256(npz_path) != metadata.get("npz_sha256"):
        raise ValueError("NPZ checksum does not match metadata")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        raise ValueError(f"could not read NPZ shard {npz_path}") from exc
    _validate_loaded(npz_path, metadata, arrays)
    for array in arrays.values():
        array.setflags(write=False)
    return NumpyShard(
        npz_path,
        MappingProxyType(metadata),
        MappingProxyType(arrays),
    )


def iter_numpy_batches(
    shard_paths: Sequence[str | os.PathLike[str]],
    batch_size: int,
    *,
    split: str | None = None,
    drop_last: bool = False,
) -> Iterator[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """Stream validated shards as model-input/label NumPy batches."""

    size = _integer(batch_size, "batch_size", positive=True)
    if split is not None and str(split).strip().lower() not in SPLIT_CODE:
        raise ValueError(f"split must be one of {SUPPORTED_SPLITS}")
    seen: set[tuple[int, int]] = set()
    dataset_fingerprint: tuple[object, ...] | None = None
    for path in shard_paths:
        shard = load_shard(path)
        fingerprint = (
            json.dumps(shard.metadata["config"], sort_keys=True),
            shard.metadata["source_sha256_aggregate"],
        )
        if dataset_fingerprint is None:
            dataset_fingerprint = fingerprint
        elif fingerprint != dataset_fingerprint:
            raise ValueError("shards do not belong to one config/source dataset")
        for recipe_index, view_index in zip(
            shard.arrays["recipe_index"], shard.arrays["view_index"]
        ):
            key = (int(recipe_index), int(view_index))
            if key in seen:
                raise ValueError("duplicate recipe_index/view_index found across shards")
            seen.add(key)
        inputs, labels = shard.training_data(split=split)
        selected_count = labels["topology_id"].shape[0]
        for start in range(0, selected_count, size):
            stop = min(start + size, selected_count)
            if drop_last and stop - start < size:
                continue
            yield (
                {name: value[start:stop] for name, value in inputs.items()},
                {name: value[start:stop] for name, value in labels.items()},
            )


__all__ = [
    "ARRAY_ORDER",
    "DATASET_GENERATOR_VERSION",
    "DATASET_SCHEMA_VERSION",
    "GRID_KIND_CODE",
    "NumpyShard",
    "PHYSICAL_CELL_VERSION",
    "PILOT_LIMITATIONS",
    "PILOT_PHASE",
    "PilotDatasetConfig",
    "RANGE_CODE",
    "RANGE_GENERATOR_VERSION",
    "RANGE_REGIMES",
    "SEED_SCHEME_VERSION",
    "SPLIT_POLICY_VERSION",
    "SUPPORTED_SPLITS",
    "SPLIT_CODE",
    "SPLIT_FRACTIONS",
    "ShardSpec",
    "TOPOLOGY_SCHEDULES",
    "calibration_stratum_values",
    "generate_shard_arrays",
    "iter_numpy_batches",
    "load_shard",
    "physical_cell_id",
    "recipe_seed_for",
    "reconstruct_recipe",
    "split_for_physical_cell",
    "topology_id_for_index",
    "user_range_for_target",
]
