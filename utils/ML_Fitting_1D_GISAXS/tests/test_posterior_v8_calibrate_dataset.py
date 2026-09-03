from __future__ import annotations

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.build_dataset import build_shard
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.calibrate_dataset import (
    build_dataset_calibration,
    calibration_samples_from_shards,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    ACQUISITION_POLICY_REQUIRED_COMPONENTS,
    acquisition_policy_payload,
    load_compatibility_calibration,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.dataset import (
    SPLIT_CODE,
    PilotDatasetConfig,
    ShardSpec,
    load_shard,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    v5_acquisition_policy_id,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.simulation import sample_observation_view
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.uncertainty_provenance_v5 import (
    V5UncertaintyProvenance,
)


def test_reserved_split_build_is_grouped_reproducible_and_immutable(tmp_path):
    dataset = build_shard(
        tmp_path / "data",
        PilotDatasetConfig(
            master_seed=9102,
            topology_schedule="k1",
            views_per_recipe=2,
            max_raw_points=128,
        ),
        ShardSpec(shard_index=0, start_index=0, recipe_count=120),
    )
    shard = load_shard(dataset.npz_path)
    calibration_rows = shard.arrays["assigned_split"] == SPLIT_CODE["calibration"]
    expected_groups = np.unique(shard.arrays["recipe_index"][calibration_rows]).size
    assert expected_groups > 0

    first_samples, first_split_hash = calibration_samples_from_shards((shard,))
    second_samples, second_split_hash = calibration_samples_from_shards((shard,))
    assert first_samples == second_samples
    assert first_split_hash == second_split_hash
    assert len({value.independent_group_id for value in first_samples}) == expected_groups
    assert all(value.score >= 0.0 for value in first_samples)
    assert all(
        0 < value.effective_valid_point_count <= value.stratum.design_point_count
        for value in first_samples
    )
    assert all(value.measurement_sigma_available for value in first_samples)
    decoded_policies = [
        acquisition_policy_payload(value.acquisition_policy_id) for value in first_samples
    ]
    assert all(
        set(value) == {"version", *ACQUISITION_POLICY_REQUIRED_COMPONENTS}
        for value in decoded_policies
    )
    assert all(
        value["grid"]["design_point_count"] == sample.stratum.design_point_count
        and value["sigma"]["noise_id"] == sample.stratum.noise_id
        and value["grid"]["q_window_id"] == sample.stratum.q_window_id
        for sample, value in zip(first_samples, decoded_policies)
    )
    assert all(
        set(value.stratum.__dict__) == {"point_count", "noise_id", "q_window_id"}
        for value in first_samples
    )
    for sample in first_samples:
        _, recipe_index_text, view_index_text = sample.sample_id.split(":")
        recipe_index = int(recipe_index_text)
        view_index = int(view_index_text)
        row = int(
            np.flatnonzero(
                (shard.arrays["recipe_index"] == recipe_index)
                & (shard.arrays["view_index"] == view_index)
            )[0]
        )
        expected_view = sample_observation_view(
            int(shard.arrays["recipe_seed"][row]),
            view_index,
            max_points=128,
        )
        assert sample.acquisition_policy_id == v5_acquisition_policy_id(
            expected_view,
            V5UncertaintyProvenance("simulated_sigma"),
        )

    output = tmp_path / "calibration.json"
    artifact = build_dataset_calibration(
        (dataset.npz_path,),
        output,
        target_coverage=0.5,
        minimum_samples_per_stratum=1,
    )
    assert artifact.input_summary.independent_group_count == expected_groups
    assert artifact.input_summary.recipe_stratum_count == len(first_samples)
    assert artifact.input_summary.effective_valid_point_count_min == min(
        value.effective_valid_point_count for value in first_samples
    )
    assert artifact.input_summary.effective_valid_point_count_max == max(
        value.effective_valid_point_count for value in first_samples
    )
    assert artifact.calibration_split_sha256 == first_split_hash
    assert load_compatibility_calibration(output) == artifact
    with pytest.raises(FileExistsError, match="overwrite"):
        build_dataset_calibration(
            (dataset.npz_path,),
            output,
            target_coverage=0.5,
            minimum_samples_per_stratum=1,
        )


def test_calibration_rejects_a_collection_without_reserved_rows(tmp_path):
    dataset = build_shard(
        tmp_path / "data",
        PilotDatasetConfig(
            master_seed=1,
            topology_schedule="k1",
            views_per_recipe=2,
            max_raw_points=128,
        ),
        ShardSpec(shard_index=0, start_index=0, recipe_count=1),
    )
    shard = load_shard(dataset.npz_path)
    if np.any(shard.arrays["assigned_split"] == SPLIT_CODE["calibration"]):
        pytest.skip("deterministic one-row fixture landed in calibration")
    with pytest.raises(ValueError, match="no reserved calibration rows"):
        calibration_samples_from_shards((shard,))
