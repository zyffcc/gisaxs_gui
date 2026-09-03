from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_contract import (
    BOUND_PLACEMENTS,
    BOUNDS_EMBEDDING_DIM,
    RANGE_REGIMES,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_dataset import (
    PLACEMENT_CODE,
    RANGE_CODE,
    TASK_CODE,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_schedule import (
    BOUNDS_BRANCH_SCHEDULE_VERSION,
    SCHEDULE_SEMANTICS_SHA256,
    TOPOLOGY_SCHEDULES,
    full_factorial_prefix_recipe_count,
    scheduled_branch_and_bounds,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_first_shards import (
    ARRAY_ORDER,
    PILOT_LIMITATIONS,
    SHARD_SCHEMA_VERSION,
    SOLUTION_ONLY_PHASE,
    SPLIT_CODE,
    SPLIT_NAMES,
    BoundsFirstShardConfig,
    BoundsFirstShardSpec,
    generate_shard_arrays,
    recipe_group_id_for,
    recipe_seed_for,
    split_for_recipe_group,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_bounds_first_shards import (
    DEFAULT_MAXWELL_OUTPUT_ROOT,
    audit_shards,
    build_parser,
    build_shard,
    canonical_json_bytes,
    load_shard,
    write_merge_audit,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.canonical_branch_catalog import (
    CANONICAL_BRANCH_CATALOG_VERSION,
    CANONICAL_VALID_BRANCH_PATTERN_MASK,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.canonical_component_slots import (
    CANONICAL_COMPONENT_SLOTS_VERSION,
)


SLURM_SCRIPT = (
    Path(__file__).parents[1]
    / "PosteriorV8"
    / "slurm"
    / "bounds_first_v4_dataset_cpu.sbatch"
)


def _config(**changes):
    values = {
        "master_seed": 8031,
        "topology_schedule": "k1",
        "views_per_recipe": 2,
        "max_raw_points": 128,
    }
    values.update(changes)
    return BoundsFirstShardConfig(**values)


def test_solution_shard_covers_bounds_schedule_and_groups_every_recipe_view():
    config = _config()
    arrays, records = generate_shard_arrays(
        config, BoundsFirstShardSpec(4, 0, 18)
    )

    assert tuple(arrays) == ARRAY_ORDER
    assert len(records) == 18
    assert set(arrays["range_regime"]) == set(RANGE_CODE.values())
    assert set(arrays["bound_placement"]) == set(PLACEMENT_CODE.values())
    assert arrays["bounds_embedding"].shape == (36, BOUNDS_EMBEDDING_DIM)
    assert np.all(arrays["truth_available"])
    assert np.all(arrays["task_kind"] == TASK_CODE["in_domain_solution"])
    assert "branch_low" not in arrays and "branch_high" not in arrays
    assert {record["bounds"]["range_regime"] for record in records} == set(
        RANGE_REGIMES
    )
    assert {record["bounds"]["placement"] for record in records} == set(
        BOUND_PLACEMENTS
    )

    for recipe_index in range(18):
        rows = np.flatnonzero(arrays["global_recipe_index"] == recipe_index)
        assert rows.size == config.views_per_recipe
        for name in (
            "target_local_unit",
            "global_reference_unit",
            "bounds_embedding",
            "bounds_sha256",
            "recipe_group_id",
            "assigned_split",
        ):
            assert np.unique(arrays[name][rows], axis=0).shape[0] == 1
        assert np.unique(arrays["observation_seed"][rows]).size == rows.size
        group_id = arrays["recipe_group_id"][rows[0]].decode("ascii")
        assert group_id == recipe_group_id_for(config, recipe_index)
        assert int(arrays["assigned_split"][rows[0]]) == SPLIT_CODE[
            split_for_recipe_group(group_id)
        ]


def test_global_recipe_identity_does_not_depend_on_shard_partitioning():
    config = _config()
    first, first_records = generate_shard_arrays(
        config, BoundsFirstShardSpec(0, 7, 1)
    )
    moved, moved_records = generate_shard_arrays(
        config, BoundsFirstShardSpec(99, 7, 1)
    )

    assert first_records == moved_records
    for name in ARRAY_ORDER:
        np.testing.assert_array_equal(first[name], moved[name])
    seeds = [recipe_seed_for(config.master_seed, index) for index in range(500)]
    assert len(set(seeds)) == len(seeds)
    assert {
        split_for_recipe_group(recipe_group_id_for(config, index))
        for index in range(500)
    } == set(SPLIT_NAMES)
    for bucket, expected in {
        0: "train",
        7_499: "train",
        7_500: "tuning_validation",
        8_499: "tuning_validation",
        8_500: "calibration",
        8_999: "calibration",
        9_000: "test",
        9_999: "test",
    }.items():
        assert split_for_recipe_group(f"{bucket:064x}") == expected


@pytest.mark.parametrize("schedule", tuple(TOPOLOGY_SCHEDULES))
def test_each_topology_has_full_factorial_branch_and_bounds_schedule(schedule):
    topology_ids = TOPOLOGY_SCHEDULES[schedule]
    for topology_slot, topology_id in enumerate(topology_ids):
        valid_patterns = {
            pattern
            for pattern, valid in enumerate(
                CANONICAL_VALID_BRANCH_PATTERN_MASK[topology_id]
            )
            if valid
        }
        observed_pairs = []
        for occurrence in range(18 * len(valid_patterns)):
            index = topology_slot + occurrence * len(topology_ids)
            result = scheduled_branch_and_bounds(schedule, index)
            observed_topology, observed_occurrence, pattern, combo = result[:4]
            assert observed_topology == topology_id
            assert observed_occurrence == occurrence
            observed_pairs.append((pattern, combo))
        assert len(set(observed_pairs)) == 18 * len(valid_patterns)
        assert set(observed_pairs) == {
            (pattern, combo) for pattern in valid_patterns for combo in range(18)
        }


def test_k1_k2_schedule_is_the_first_nine_catalog_topologies():
    assert TOPOLOGY_SCHEDULES["k1_k2"] == tuple(range(9))
    assert full_factorial_prefix_recipe_count("k1") == 216
    assert full_factorial_prefix_recipe_count("k1_k2") == 1_296
    assert full_factorial_prefix_recipe_count("all34") == 14_688
    parser = build_parser()
    args = parser.parse_args(
        [
            "build",
            "--output-dir",
            "/tmp/v4-stage",
            "--shard-index",
            "0",
            "--start-recipe-index",
            "0",
            "--recipe-count",
            "1",
            "--topology-schedule",
            "k1_k2",
        ]
    )
    assert args.topology_schedule == "k1_k2"


def test_k4_all_d_and_resolution_branch_generates_from_local_bounds():
    topology_slot = 33
    index = next(
        topology_slot + occurrence * 34
        for occurrence in range(18 * 32)
        if scheduled_branch_and_bounds(
            "all34", topology_slot + occurrence * 34
        )[2]
        == 31
    )
    arrays, records = generate_shard_arrays(
        _config(topology_schedule="all34"), BoundsFirstShardSpec(8, index, 1)
    )

    assert np.all(arrays["component_count"] == 4)
    assert np.all(arrays["branch_pattern_id"] == 31)
    assert records[0]["truth_resolution"] is not None
    assert all(component["D"] is not None for component in records[0]["truth_components"])


def test_build_load_is_deterministic_exclusive_and_fully_provenanced(tmp_path):
    config, spec = _config(), BoundsFirstShardSpec(3, 11, 2)
    first = build_shard(tmp_path / "first", config, spec)
    replay = build_shard(tmp_path / "replay", config, spec)

    assert first.npz_path.read_bytes() == replay.npz_path.read_bytes()
    assert first.metadata_path.read_bytes() == replay.metadata_path.read_bytes()
    before = (first.npz_path.read_bytes(), first.metadata_path.read_bytes())
    with pytest.raises(FileExistsError, match="overwrite"):
        build_shard(tmp_path / "first", config, spec)
    assert before == (first.npz_path.read_bytes(), first.metadata_path.read_bytes())

    shard = load_shard(first.npz_path)
    metadata = shard.metadata
    assert metadata["dataset_schema_version"] == SHARD_SCHEMA_VERSION
    assert metadata["phase"] == SOLUTION_ONLY_PHASE
    assert metadata["bounds_generation"]["truth_conditioned_bounds_generation"] is False
    assert metadata["bounds_generation"]["task_population"] == "in_domain_solution_only"
    assert metadata["versions"]["bounds_branch_schedule"] == (
        BOUNDS_BRANCH_SCHEDULE_VERSION
    )
    assert metadata["versions"]["canonical_branch_catalog"] == (
        CANONICAL_BRANCH_CATALOG_VERSION
    )
    assert metadata["versions"]["canonical_component_slots"] == (
        CANONICAL_COMPONENT_SLOTS_VERSION
    )
    assert (
        "utils/ML_Fitting_1D_GISAXS/PosteriorV8/canonical_component_slots.py"
        in metadata["source_sha256"]
    )
    assert metadata["bounds_generation"]["branch_bounds_schedule_sha256"] == (
        SCHEDULE_SEMANTICS_SHA256
    )
    assert metadata["split_assignment"]["no_parameter_guard_band"] is True
    assert tuple(metadata["pilot_limitations"]) == PILOT_LIMITATIONS
    assert len(metadata["source_sha256"]) >= 10
    assert len(metadata["source_sha256_aggregate"]) == 64
    assert sha256(first.npz_path.read_bytes()).hexdigest() == metadata["npz_sha256"]
    assert shard.recipe_count == spec.recipe_count
    assert shard.row_count == spec.recipe_count * config.views_per_recipe
    assert all(record["truth_available"] for record in metadata["recipe_records"])


def test_merge_audit_accepts_disjoint_shards_and_rejects_overlap_or_mismatch(
    tmp_path,
):
    config = _config()
    first = build_shard(
        tmp_path / "one", config, BoundsFirstShardSpec(0, 0, 2)
    )
    second = build_shard(
        tmp_path / "two", config, BoundsFirstShardSpec(1, 2, 2)
    )
    overlap = build_shard(
        tmp_path / "overlap", config, BoundsFirstShardSpec(2, 1, 1)
    )
    mismatch = build_shard(
        tmp_path / "mismatch",
        _config(master_seed=config.master_seed + 1),
        BoundsFirstShardSpec(3, 4, 1),
    )

    audit = audit_shards((second.npz_path, first.npz_path))
    assert audit["merge_eligible"] is True
    assert audit["recipe_coverage_contiguous"] is True
    assert audit["recipe_index_gaps"] == []
    assert audit["recipe_count"] == 4
    assert audit["row_count"] == 8
    assert sum(audit["recipe_split_counts"].values()) == 4
    audit_path = tmp_path / "merge-audit.json"
    stored = write_merge_audit(audit_path, (first.npz_path, second.npz_path))
    assert json.loads(audit_path.read_text()) == stored
    with pytest.raises(FileExistsError, match="overwrite"):
        write_merge_audit(audit_path, (first.npz_path, second.npz_path))
    with pytest.raises(ValueError, match="overlapping global_recipe_index"):
        audit_shards((first.npz_path, overlap.npz_path))
    with pytest.raises(ValueError, match="config/source/schema mismatch"):
        audit_shards((first.npz_path, mismatch.npz_path))


def test_loader_rejects_tampered_solution_or_source_contract(tmp_path):
    artifact = build_shard(
        tmp_path, _config(), BoundsFirstShardSpec(0, 0, 1)
    )
    metadata = json.loads(artifact.metadata_path.read_text())
    metadata["pilot_limitations"].remove("no_parameter_guard_band")
    artifact.metadata_path.write_bytes(canonical_json_bytes(metadata))
    with pytest.raises(ValueError, match="limitation provenance"):
        load_shard(artifact.npz_path)


def test_loader_rejects_tampered_component_slot_contract(tmp_path):
    artifact = build_shard(
        tmp_path, _config(), BoundsFirstShardSpec(0, 0, 1)
    )
    metadata = json.loads(artifact.metadata_path.read_text())
    metadata["versions"]["canonical_component_slots"] = "legacy"
    artifact.metadata_path.write_bytes(canonical_json_bytes(metadata))

    with pytest.raises(ValueError, match="scientific version contract"):
        load_shard(artifact.npz_path)


def test_cli_defaults_and_maxwell_slurm_array_contract():
    args = build_parser().parse_args(
        [
            "build",
            "--shard-index",
            "0",
            "--start-recipe-index",
            "0",
            "--recipe-count",
            "1",
        ]
    )
    assert args.output_dir == DEFAULT_MAXWELL_OUTPUT_ROOT
    assert str(args.output_dir).startswith("/data/dust/user/zhaiyufe/")
    assert "GISAXS_POSTERIOR_V8_20260902" in str(args.output_dir)

    source = SLURM_SCRIPT.read_text(encoding="utf-8")
    for required in (
        "#SBATCH --array=",
        "SLURM_ARRAY_TASK_ID",
        "dataset_start + shard_index * recipe_count",
        "PosteriorV8.build_bounds_first_shards build",
        "--start-recipe-index",
        "--views-per-recipe",
        "/data/dust/user/zhaiyufe/",
    ):
        assert required in source
    assert "POSTERIOR_V8_V4_OUTPUT_DIR must be under" in source
    assert "GISAXS_POSTERIOR_V8_20260902" in source
    assert "PosteriorV8.train" not in source


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: BoundsFirstShardConfig(views_per_recipe=1), "views_per_recipe"),
        (lambda: BoundsFirstShardConfig(max_raw_points=64), "max_raw_points"),
        (lambda: BoundsFirstShardConfig(topology_schedule="unknown"), "schedule"),
        (lambda: BoundsFirstShardSpec(0, 0, 0), "recipe_count"),
    ],
)
def test_invalid_shard_contracts_fail_closed(factory, message):
    with pytest.raises((TypeError, ValueError), match=message):
        factory()
