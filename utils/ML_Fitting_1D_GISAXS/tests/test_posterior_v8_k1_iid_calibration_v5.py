from __future__ import annotations

import copy

import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_iid_calibration_plan_v5 import (
    V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM,
    V5_K1_IID_CALIBRATION_TOTAL_SAMPLES,
    build_v5_k1_iid_calibration_plan,
    materialize_v5_k1_iid_calibration_recipe,
    validate_v5_k1_iid_calibration_plan,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.k1_iid_calibration_shard_v5 import (
    build_v5_k1_iid_calibration_shard,
    validate_v5_k1_iid_calibration_shard,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    sample_v5_uncertainty_provenance,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.simulation import (
    OBSERVATION_DESIGN_STRATUM_UNIVERSE,
    sample_observation_design_stratum,
)


def test_formal_iid_calibration_plan_is_exactly_balanced_at_target_scale():
    plan = build_v5_k1_iid_calibration_plan()

    assert validate_v5_k1_iid_calibration_plan(plan) == plan
    assert plan["randomized_sobol_or_qmc_points_used"] is False
    assert plan["samples_per_stratum"] == V5_K1_IID_CALIBRATION_SAMPLES_PER_STRATUM == 2667
    assert plan["total_samples"] == V5_K1_IID_CALIBRATION_TOTAL_SAMPLES == 160020
    assert len(plan["strata"]) == 60
    assert {value["sample_count"] for value in plan["strata"]} == {2667}
    assert len(
        {
            tuple(sorted(value["compatibility_stratum"].items()))
            for value in plan["strata"]
        }
    ) == 60
    assert plan["minimum_samples_per_stratum"] == 200
    assert all(value is False for value in plan["claim_limits"].values())


def test_iid_recipe_identity_and_sigma_view_replay_without_curve_selection():
    first = materialize_v5_k1_iid_calibration_recipe(
        stratum_ordinal=17,
        sample_ordinal=11,
    )
    second = materialize_v5_k1_iid_calibration_recipe(
        stratum_ordinal=17,
        sample_ordinal=11,
    )

    assert first == second
    assert sample_observation_design_stratum(
        first.recipe.recipe_seed, first.view_index
    ) == OBSERVATION_DESIGN_STRATUM_UNIVERSE[17]
    assert sample_v5_uncertainty_provenance(
        first.recipe.recipe_seed, first.view_index
    ).measurement_sigma_available
    assert first.recipe.target.pattern_id == first.branch.pattern_id
    assert first.recipe.query.topology == (first.branch.shape,)


def test_iid_score_shard_replays_complete_sample_and_rejects_tampering():
    plan = build_v5_k1_iid_calibration_plan()
    shard = build_v5_k1_iid_calibration_shard(
        calibration_plan=plan,
        stratum_ordinal=59,
        sample_ordinal_start=3,
        sample_count=2,
    )

    assert validate_v5_k1_iid_calibration_shard(shard, calibration_plan=plan) == shard
    assert shard["manifest"]["sample_count"] == 2
    assert [value["sample_ordinal"] for value in shard["samples"]] == [3, 4]
    assert all(value["measurement_sigma_available"] for value in shard["samples"])
    assert all(value["score"] >= 0.0 for value in shard["samples"])
    assert shard["scientific_acceptance_evidence"] is False

    tampered = copy.deepcopy(shard)
    tampered["samples"][0]["score"] += 1.0
    with pytest.raises(ValueError, match="self-hash"):
        validate_v5_k1_iid_calibration_shard(tampered, calibration_plan=plan)


def test_iid_score_shard_keeps_measurement_sigma_positive_for_low_scale_stratum():
    plan = build_v5_k1_iid_calibration_plan()

    shard = build_v5_k1_iid_calibration_shard(
        calibration_plan=plan,
        stratum_ordinal=0,
        sample_count=16,
    )

    assert validate_v5_k1_iid_calibration_shard(shard, calibration_plan=plan) == shard
    assert len(shard["samples"]) == 16
    assert all(value["measurement_sigma_available"] for value in shard["samples"])
    assert all(value["score"] >= 0.0 for value in shard["samples"])


def test_calibration_plan_and_windows_fail_closed():
    plan = build_v5_k1_iid_calibration_plan()
    tampered = copy.deepcopy(plan)
    tampered["randomized_sobol_or_qmc_points_used"] = True
    with pytest.raises(ValueError, match="self-hash"):
        validate_v5_k1_iid_calibration_plan(tampered)

    with pytest.raises(ValueError, match="window"):
        build_v5_k1_iid_calibration_shard(
            calibration_plan=plan,
            stratum_ordinal=0,
            sample_ordinal_start=2666,
            sample_count=2,
        )
