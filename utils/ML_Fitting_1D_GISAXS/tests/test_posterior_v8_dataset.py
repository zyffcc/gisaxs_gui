from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import build_dataset as build_module
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import decode_branch_pattern
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_dataset import build_shard
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    CompatibilityStratum,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import TOPOLOGIES
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.dataset import (
    ARRAY_ORDER,
    DATASET_GENERATOR_VERSION,
    DATASET_SCHEMA_VERSION,
    GRID_KIND_CODE,
    PILOT_LIMITATIONS,
    PILOT_PHASE,
    RANGE_CODE,
    RANGE_REGIMES,
    SPLIT_CODE,
    SPLIT_FRACTIONS,
    SPLIT_POLICY_VERSION,
    SUPPORTED_SPLITS,
    PilotDatasetConfig,
    ShardSpec,
    calibration_stratum_values,
    generate_shard_arrays,
    iter_numpy_batches,
    load_shard,
    physical_cell_id,
    recipe_seed_for,
    reconstruct_recipe,
    split_for_physical_cell,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.preprocessing import preprocess_curve
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.simulation import (
    sample_observation_view,
    simulate_recipe,
)


def _config(**changes):
    values = {
        "master_seed": 1203,
        "topology_schedule": "k1",
        "views_per_recipe": 2,
        "max_raw_points": 128,
    }
    values.update(changes)
    return PilotDatasetConfig(**values)


def test_clean_recipe_seed_is_split_independent_and_bijective():
    seeds = [recipe_seed_for(91, index) for index in range(300)]
    assert len(set(seeds)) == len(seeds)
    assert seeds == [recipe_seed_for(91, index) for index in range(300)]


def test_four_way_split_contract_and_hash_boundaries_are_exact():
    assert DATASET_SCHEMA_VERSION.endswith("/v3")
    assert DATASET_GENERATOR_VERSION.endswith("_v3")
    assert SUPPORTED_SPLITS == ("train", "validation", "calibration", "test")
    assert SPLIT_CODE == {
        "train": 0,
        "validation": 1,
        "calibration": 2,
        "test": 3,
    }
    assert SPLIT_FRACTIONS == {
        "train": 0.75,
        "validation": 0.10,
        "calibration": 0.05,
        "test": 0.10,
    }
    assert sum(SPLIT_FRACTIONS.values()) == pytest.approx(1.0)

    expected_at_boundary = {
        0: "train",
        7_499: "train",
        7_500: "validation",
        8_499: "validation",
        8_500: "calibration",
        8_999: "calibration",
        9_000: "test",
        9_999: "test",
    }
    for bucket, expected in expected_at_boundary.items():
        cell_id = f"{bucket:064x}"
        assert split_for_physical_cell(cell_id) == expected


def test_all_views_of_a_coarse_physical_cell_have_one_split():
    arrays = generate_shard_arrays(_config(topology_schedule="all34"), ShardSpec(0, 0, 120))
    cell_splits = {}
    for cell_bytes, split_code in zip(arrays["physical_cell_id"], arrays["assigned_split"]):
        cell = cell_bytes.decode("ascii")
        cell_splits.setdefault(cell, set()).add(int(split_code))
        assert SPLIT_CODE[split_for_physical_cell(cell)] == int(split_code)
    assert all(len(values) == 1 for values in cell_splits.values())
    assert set(arrays["assigned_split"].tolist()) == set(SPLIT_CODE.values())

    for recipe_index in range(120):
        rows = arrays["recipe_index"] == recipe_index
        assert np.count_nonzero(rows) == 2
        assert np.unique(arrays["physical_cell_id"][rows]).size == 1
        assert np.unique(arrays["assigned_split"][rows]).size == 1


def test_all_34_topologies_are_balanced_per_clean_recipe_and_reconstructible():
    config = _config(topology_schedule="all34")
    arrays = generate_shard_arrays(config, ShardSpec(0, 0, 34))
    np.testing.assert_array_equal(arrays["topology_id"][::2], np.arange(34))
    np.testing.assert_array_equal(arrays["topology_id"][0::2], arrays["topology_id"][1::2])

    row = 25
    recipe = reconstruct_recipe(
        recipe_seed=int(arrays["recipe_seed"][row]),
        topology_id=int(arrays["topology_id"][row]),
        config=config,
    )
    cell = physical_cell_id(recipe, arrays["target_unit"][row])
    assert arrays["physical_cell_id"][row].decode("ascii") == cell
    view = sample_observation_view(
        recipe.seed, int(arrays["view_index"][row]), max_points=config.max_raw_points
    )
    simulated = simulate_recipe(view.simulation_recipe(recipe))
    expected = preprocess_curve(
        simulated.q,
        simulated.intensity,
        simulated.sigma,
        mask=view.selection_mask(simulated.q),
        q_range=view.preprocess_q_range,
    )
    np.testing.assert_array_equal(arrays["x"][row], expected.x)
    np.testing.assert_array_equal(arrays["point_mask"][row], expected.point_mask)


def test_observation_views_vary_grid_noise_crop_and_mask_but_share_truth():
    config = _config(views_per_recipe=3, max_raw_points=512)
    arrays = generate_shard_arrays(config, ShardSpec(0, 7, 12))
    assert set(arrays["grid_kind"].tolist()) == set(GRID_KIND_CODE.values())
    assert np.unique(arrays["raw_point_count"]).size >= 3
    assert np.unique(arrays["poisson_count_scale"]).size >= 4
    assert np.unique(arrays["relative_sigma"]).size >= 3
    assert np.unique(arrays["point_keep_probability"]).size >= 3
    assert np.unique(arrays["q_window_id"]).size >= 3
    assert np.unique(arrays["noise_id"]).size >= 4
    assert np.unique(arrays["mask_id"]).size >= 3
    assert np.unique(arrays["crop_id"]).size >= 3
    assert np.any(arrays["preprocess_q_range"] != arrays["q_window"])

    for recipe_index in range(7, 19):
        rows = np.flatnonzero(arrays["recipe_index"] == recipe_index)
        assert len(rows) == 3
        np.testing.assert_array_equal(
            arrays["target_unit"][rows],
            np.repeat(arrays["target_unit"][rows[:1]], 3, axis=0),
        )
        assert np.unique(arrays["observation_seed"][rows]).size == 3
        assert len({arrays["x"][row].tobytes() for row in rows}) == 3

    strata = [
        tuple(calibration_stratum_values(arrays, row).items())
        for row in range(arrays["topology_id"].size)
    ]
    assert len(set(strata)) < len(strata)
    example = dict(strata[0])
    assert set(example) == {"point_count", "noise_id", "q_window_id"}
    assert example["noise_id"].startswith("posterior_v8_observation_stratum_v1:")
    assert example["q_window_id"].startswith("posterior_v8_observation_stratum_v1:")
    assert CompatibilityStratum(**example).point_count == example["point_count"]


def test_full_wide_narrow_ranges_contain_truth_and_preserve_branch():
    arrays = generate_shard_arrays(_config(views_per_recipe=3), ShardSpec(0, 0, 9))
    assert set(arrays["range_regime"].tolist()) == set(RANGE_CODE.values())
    for row in range(arrays["topology_id"].size):
        active = arrays["active_dimension_mask"][row]
        target = arrays["target_unit"][row]
        low, high = arrays["branch_low"][row], arrays["branch_high"][row]
        assert np.all(low[active] <= target[active])
        assert np.all(target[active] <= high[active])
        np.testing.assert_array_equal(low[~active], 0.5)
        np.testing.assert_array_equal(high[~active], 0.5)
        regime = RANGE_REGIMES[int(arrays["range_regime"][row])]
        if regime == "full":
            np.testing.assert_array_equal(low[active], 0.0)
            np.testing.assert_array_equal(high[active], 1.0)
        flags, resolution = decode_branch_pattern(int(arrays["branch_pattern_id"][row]))
        component_count = len(TOPOLOGIES[int(arrays["topology_id"][row])])
        assert not any(flags[component_count:])
        assert isinstance(resolution, bool)


def test_build_is_byte_deterministic_and_records_phase2_provenance(tmp_path):
    config, spec = _config(), ShardSpec(7, 3, 4)
    first = build_shard(tmp_path / "one", config, spec)
    second = build_shard(tmp_path / "two", config, spec)
    assert first.npz_path.read_bytes() == second.npz_path.read_bytes()
    assert first.metadata_path.read_bytes() == second.metadata_path.read_bytes()
    metadata = json.loads(first.metadata_path.read_text())
    assert metadata["dataset_schema_version"] == DATASET_SCHEMA_VERSION
    assert metadata["dataset_generator_version"] == DATASET_GENERATOR_VERSION
    assert metadata["phase"] == PILOT_PHASE
    assert tuple(metadata["pilot_limitations"]) == PILOT_LIMITATIONS
    assert metadata["split_assignment"]["assignment_unit"] == "coarse_physical_parameter_cell"
    assert metadata["versions"]["split_policy"] == SPLIT_POLICY_VERSION
    assert metadata["split_assignment"]["fractions"] == SPLIT_FRACTIONS
    assert metadata["split_assignment"]["codes"] == SPLIT_CODE
    assert metadata["split_assignment"]["train_bucket_stop"] == 7_500
    assert metadata["split_assignment"]["validation_bucket_stop"] == 8_500
    assert metadata["split_assignment"]["calibration_bucket_stop"] == 9_000
    assert metadata["observation_generation"]["views_per_clean_recipe"] == 2
    assert metadata["observation_generation"]["unknown_physical_parameter_access"] is False
    assert (
        metadata["observation_generation"][
            "missing_characteristic_scales_are_retained_as_ambiguity_or_ood"
        ]
        is True
    )
    assert metadata["observation_generation"]["calibration_stratum_fields"] == [
        "raw_point_count",
        "noise_id",
        "q_window_id",
    ]
    assert metadata["observation_generation"]["compatibility_stratum_fields"] == [
        "point_count",
        "noise_id",
        "q_window_id",
    ]
    assert (
        "candidate" in metadata["observation_generation"]["compatibility_stratification_semantics"]
    )
    assert metadata["range_generation"]["discrete_branch_changes"] == "reject"
    assert metadata["shard"]["recipe_count"] == 4
    assert metadata["shard"]["row_count"] == 8
    assert set(metadata["array_schema"]) == set(ARRAY_ORDER)
    assert len(metadata["source_sha256"]) >= 9
    with pytest.raises(FileExistsError, match="overwrite"):
        build_shard(tmp_path / "one", config, spec)


def test_gitless_source_snapshot_is_supported(tmp_path):
    root = tmp_path / "snapshot"
    (root / "src").mkdir(parents=True)
    module_dir = root / "utils" / "ML_Fitting_1D_GISAXS" / "PosteriorV8"
    module_dir.mkdir(parents=True)
    module_file = module_dir / "build_dataset.py"
    module_file.touch()
    assert not (root / ".git").exists()
    assert build_module._repository_root(module_file) == root


def test_loader_and_batch_iterator_filter_cell_assigned_splits(tmp_path):
    artifact = build_shard(tmp_path, _config(topology_schedule="all34"), ShardSpec(0, 0, 40))
    shard = load_shard(artifact.npz_path)
    assert shard.sample_count == 80
    total = 0
    for split, code in SPLIT_CODE.items():
        expected = int(np.count_nonzero(shard.arrays["assigned_split"] == code))
        inputs, labels = shard.training_data(split=split)
        assert inputs["x"].shape[0] == labels["topology_id"].shape[0] == expected
        total += expected
    assert total == shard.sample_count
    train_count = sum(
        labels["topology_id"].shape[0]
        for _, labels in iter_numpy_batches([artifact.npz_path], 7, split="train")
    )
    assert train_count == np.count_nonzero(shard.arrays["assigned_split"] == SPLIT_CODE["train"])
    calibration_count = sum(
        labels["topology_id"].shape[0]
        for _, labels in iter_numpy_batches([artifact.npz_path], 7, split="calibration")
    )
    assert calibration_count == np.count_nonzero(
        shard.arrays["assigned_split"] == SPLIT_CODE["calibration"]
    )


def test_loader_rejects_pre_four_way_split_schema_and_wrong_dtype(tmp_path):
    artifact = build_shard(tmp_path, _config(), ShardSpec(0, 0, 2))
    metadata = json.loads(artifact.metadata_path.read_text())
    metadata["dataset_schema_version"] = "gisaxs.posterior_v8.identifiable_npz/v1"
    artifact.metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="pre-four-way-split v1/v2.*rejected"):
        load_shard(artifact.npz_path)

    metadata["dataset_schema_version"] = DATASET_SCHEMA_VERSION
    with np.load(artifact.npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["topology_id"] = arrays["topology_id"].astype(np.int64)
    np.savez_compressed(artifact.npz_path, **arrays)
    metadata["npz_sha256"] = sha256(artifact.npz_path.read_bytes()).hexdigest()
    artifact.metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="shape or dtype"):
        load_shard(artifact.npz_path)


def test_batch_iterator_rejects_overlapping_recipe_views(tmp_path):
    config = _config()
    first = build_shard(tmp_path, config, ShardSpec(0, 0, 2))
    second = build_shard(tmp_path, config, ShardSpec(1, 1, 2))
    with pytest.raises(ValueError, match="duplicate recipe_index/view_index"):
        list(iter_numpy_batches([first.npz_path, second.npz_path], 2))


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: PilotDatasetConfig(topology_schedule="unknown"), "topology_schedule"),
        (lambda: PilotDatasetConfig(views_per_recipe=1), "views_per_recipe"),
        (lambda: PilotDatasetConfig(max_raw_points=63), "max_raw_points"),
        (lambda: ShardSpec(0, 0, 0), "recipe_count"),
    ],
)
def test_invalid_phase2_contracts_fail_closed(factory, message):
    with pytest.raises((TypeError, ValueError), match=message):
        factory()
