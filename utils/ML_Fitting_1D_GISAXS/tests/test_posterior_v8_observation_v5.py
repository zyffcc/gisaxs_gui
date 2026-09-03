from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    ACQUISITION_POLICY_ID_VERSION,
    acquisition_policy_payload,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.observation_v5 import (
    V5_ENCODER_PROXY_RELATIVE_SIGMA,
    V5_OBSERVATION_DATA_VIEW_SCHEMA,
    V5_TRAINING_UNCERTAINTY_KINDS,
    V5_UNCERTAINTY_VIEW_POLICY_VERSION,
    build_v5_observation_data_view,
    build_v5_observation_data_views,
    sample_v5_uncertainty_provenance,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.simulation import NOISE_APPLICATION_VERSION
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.synthetic_recipe_v5 import (
    evaluate_v5_clean_recipe,
    sample_v5_clean_recipe,
)


def _assert_same_arrays(first, second):
    for name in (
        "q",
        "clean_intensity",
        "intensity",
        "encoder_sigma",
        "selection_mask",
        "uncertainty_provenance",
    ):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))
    if first.measurement_sigma is None:
        assert second.measurement_sigma is None
    else:
        np.testing.assert_array_equal(first.measurement_sigma, second.measurement_sigma)
    if first.acceptance_sigma_log is None:
        assert second.acceptance_sigma_log is None
    else:
        np.testing.assert_array_equal(
            first.acceptance_sigma_log,
            second.acceptance_sigma_log,
        )


def test_v5_view_is_replayable_and_policy_never_reads_unknown_physics():
    sphere = sample_v5_clean_recipe(("sphere",), recipe_seed=77)
    cylinder = sample_v5_clean_recipe(("cylinder",), recipe_seed=77)

    first = build_v5_observation_data_view(sphere, 0, split_id="train")
    replay = build_v5_observation_data_view(sphere, 0, split_id="train")
    other_physics = build_v5_observation_data_view(cylinder, 0, split_id="train")

    _assert_same_arrays(first, replay)
    assert first.observation == replay.observation == other_physics.observation
    assert first.uncertainty == replay.uncertainty == other_physics.uncertainty
    assert first.acquisition_policy_id == other_physics.acquisition_policy_id
    assert not np.array_equal(first.clean_intensity, other_physics.clean_intensity)
    np.testing.assert_array_equal(
        first.clean_intensity,
        evaluate_v5_clean_recipe(sphere, first.q),
    )


def test_v5_views_cover_both_sigma_policies_and_inherit_one_recipe_split():
    recipe = sample_v5_clean_recipe(
        ("sphere", "cylinder", "vertical_cylinder"),
        recipe_seed=77,
    )
    views = build_v5_observation_data_views(
        recipe,
        (0, 1),
        split_id="tuning_validation",
    )

    assert {value.uncertainty.kind for value in views} == set(V5_TRAINING_UNCERTAINTY_KINDS)
    assert {value.split_id for value in views} == {"tuning_validation"}
    assert {value.clean_recipe_sha256 for value in views} == {recipe.sha256}
    assert len({value.observation.observation_seed for value in views}) == 2
    assert len({value.acquisition_policy_id for value in views}) == 2

    simulated = next(value for value in views if value.uncertainty.kind == "simulated_sigma")
    assert simulated.measurement_sigma is not None
    assert simulated.acceptance_sigma_log is not None
    np.testing.assert_array_equal(simulated.encoder_sigma, simulated.measurement_sigma)
    _, valid_intensity, valid_sigma = simulated.preprocessed.valid_arrays()
    np.testing.assert_array_equal(
        simulated.acceptance_sigma_log,
        valid_sigma / valid_intensity,
    )
    np.testing.assert_array_equal(
        simulated.uncertainty_provenance,
        np.asarray((0.0, 1.0, 0.0), dtype=np.float32),
    )

    proxy = next(
        value for value in views if value.uncertainty.kind == "encoder_proxy_missing_sigma"
    )
    assert proxy.measurement_sigma is None
    assert proxy.acceptance_sigma_log is None
    np.testing.assert_allclose(
        proxy.encoder_sigma,
        V5_ENCODER_PROXY_RELATIVE_SIGMA * proxy.intensity,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        proxy.uncertainty_provenance,
        np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
    )
    assert proxy.encoder_curve_inputs()["uncertainty_provenance"].shape == (3,)
    assert proxy.encoder_curve_inputs(add_batch_axis=True)["uncertainty_provenance"].shape == (1, 3)


def test_v5_view_records_design_effective_and_complete_acquisition_identity():
    recipe = sample_v5_clean_recipe(("vertical_cylinder",), recipe_seed=109)
    view = build_v5_observation_data_view(recipe, 3, split_id="calibration")

    within_crop = (view.q >= view.observation.preprocess_q_range[0]) & (
        view.q <= view.observation.preprocess_q_range[1]
    )
    expected_effective = int(np.count_nonzero(view.selection_mask & within_crop))
    assert view.design_point_count == view.observation.grid.n_points == view.q.size
    assert view.effective_valid_point_count == expected_effective
    assert view.effective_valid_point_count == view.preprocessed.stats["valid_before_downsampling"]
    assert view.preprocessed.valid_count == expected_effective

    policy = acquisition_policy_payload(view.acquisition_policy_id)
    assert policy["version"] == ACQUISITION_POLICY_ID_VERSION
    assert policy["grid"] == {
        "kind": view.observation.grid.kind,
        "design_point_count": view.design_point_count,
        "q_min": view.observation.grid.q_min,
        "q_max": view.observation.grid.q_max,
        "q_window_id": policy["grid"]["q_window_id"],
    }
    assert policy["mask"]["mask_id"].endswith(f"mask-{view.observation.mask_id}")
    assert policy["crop"]["crop_id"].endswith(f"crop-{view.observation.crop_id}")
    assert policy["view"]["view_index"] == 3
    assert policy["sigma"]["noise_id"].endswith(f"noise-{view.observation.noise_id}")
    sigma_source = policy["sigma"]["sigma_log_source"]
    assert NOISE_APPLICATION_VERSION in sigma_source
    assert V5_UNCERTAINTY_VIEW_POLICY_VERSION in sigma_source
    assert view.uncertainty.version in sigma_source
    assert view.uncertainty.kind in sigma_source

    audit = view.audit_payload()
    assert audit["schema_version"] == V5_OBSERVATION_DATA_VIEW_SCHEMA
    assert audit["uncertainty_view_policy_version"] == (V5_UNCERTAINTY_VIEW_POLICY_VERSION)
    assert audit["design_point_count"] == view.design_point_count
    assert audit["effective_valid_point_count"] == expected_effective
    assert audit["acquisition_policy_id"] == view.acquisition_policy_id


def test_each_view_can_be_built_independently_without_sequence_state():
    recipe = sample_v5_clean_recipe(("sphere", "sphere"), recipe_seed=991)
    plural = build_v5_observation_data_views(recipe, (5, 2), split_id="test")
    alone = build_v5_observation_data_view(recipe, 2, split_id="test")

    _assert_same_arrays(plural[1], alone)
    assert plural[1].observation == alone.observation
    assert plural[1].uncertainty == alone.uncertainty
    assert sample_v5_uncertainty_provenance(recipe.recipe_seed, 2) == alone.uncertainty


def test_v5_observation_contract_fails_closed_on_provenance_mixing():
    recipe = sample_v5_clean_recipe(("sphere",), recipe_seed=77)
    views = build_v5_observation_data_views(recipe, (0, 1), split_id="train")
    proxy = next(value for value in views if value.measurement_sigma is None)

    with pytest.raises(ValueError, match="unique"):
        build_v5_observation_data_views(recipe, (0, 0), split_id="train")
    with pytest.raises(ValueError, match="cannot be empty"):
        build_v5_observation_data_views(recipe, (), split_id="train")
    with pytest.raises(ValueError, match="non-empty"):
        build_v5_observation_data_view(recipe, 0, split_id=" ")
    with pytest.raises(ValueError, match="acquisition_policy_id"):
        replace(proxy, acquisition_policy_id="legacy-incomplete-policy")
    with pytest.raises(ValueError, match="cannot expose"):
        replace(proxy, measurement_sigma=np.ones(proxy.design_point_count))
    with pytest.raises(ValueError, match="schema"):
        replace(proxy, schema_version="gisaxs.posterior_v8.observation_data_view/v0")
