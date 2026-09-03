from __future__ import annotations

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model import (
    BRANCH_DIM,
    build_proposal_model,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective import (
    ACTIVE_DIMENSION_MASKS,
    TrainingObjectiveConfig,
    active_dimension_mask_for,
    compute_training_objective,
)


def _outputs(batch, modes=3, seed=5):
    rng = np.random.default_rng(seed)
    return {
        "topology_logits": tf.constant(rng.normal(size=(batch, 34)), tf.float32),
        "branch_pattern_logits": tf.constant(
            rng.normal(size=(batch, 34, 32)), tf.float32
        ),
        "mixture_logits": tf.constant(rng.normal(size=(batch, modes)), tf.float32),
        "mixture_loc": tf.constant(
            rng.normal(scale=0.3, size=(batch, modes, BRANCH_DIM)), tf.float32
        ),
        "mixture_logscale": tf.constant(
            rng.uniform(-2.0, -0.5, size=(batch, modes, BRANCH_DIM)), tf.float32
        ),
    }


def _labels(topology_ids, pattern_ids, *, seed=9):
    topology_ids = np.asarray(topology_ids, dtype=np.int32)
    pattern_ids = np.asarray(pattern_ids, dtype=np.int32)
    masks = np.asarray(
        [
            active_dimension_mask_for(int(topology), int(pattern))
            for topology, pattern in zip(topology_ids, pattern_ids)
        ],
        dtype=np.float32,
    )
    target = np.random.default_rng(seed).uniform(
        0.1, 0.9, size=(topology_ids.size, BRANCH_DIM)
    ).astype(np.float32)
    return {
        "topology_id": topology_ids,
        "branch_pattern_id": pattern_ids,
        "target_unit": target,
        "active_dimension_mask": masks,
    }


def test_active_dimension_mask_layout_is_exact_for_hard_branches():
    assert np.asarray(ACTIVE_DIMENSION_MASKS).shape == (34, 32, BRANCH_DIM)
    sphere_no_d = active_dimension_mask_for(0, 0)
    assert tuple(index for index, active in enumerate(sphere_no_d) if active) == (0, 1)
    cylinder_d_resolution = active_dimension_mask_for(1, 17)
    assert tuple(
        index for index, active in enumerate(cylinder_d_resolution) if active
    ) == (0, 1, 2, 3, 4, 5, 24, 25)
    with pytest.raises(ValueError, match="invalid"):
        active_dimension_mask_for(0, 2)


def test_known_logits_produce_expected_metrics_and_weighted_total():
    labels = _labels([0, 1, 2], [0, 1, 16])
    outputs = _outputs(3)
    topology = np.full((3, 34), -5.0, dtype=np.float32)
    topology[0, 0] = 5.0
    topology[1, 2] = 5.0
    topology[1, 1] = 4.0
    topology[2, 2] = 5.0
    patterns = np.full((3, 34, 32), -5.0, dtype=np.float32)
    patterns[0, 0, 0] = 5.0
    patterns[1, 1, 0] = 5.0
    patterns[1, 1, 1] = 4.0
    patterns[2, 2, 16] = 5.0
    outputs["topology_logits"] = tf.constant(topology)
    outputs["branch_pattern_logits"] = tf.constant(patterns)
    config = TrainingObjectiveConfig(
        topology_weight=2.0,
        branch_pattern_weight=3.0,
        continuous_weight=0.5,
        topology_recall_k=2,
    )

    result = compute_training_objective(outputs, labels, config)

    assert result["topology_accuracy"].numpy() == pytest.approx(2.0 / 3.0)
    assert result["branch_pattern_accuracy"].numpy() == pytest.approx(2.0 / 3.0)
    assert result["topology_recall_at_k"].numpy() == pytest.approx(1.0)
    expected_total = (
        2.0 * result["topology_loss"]
        + 3.0 * result["branch_pattern_loss"]
        + 0.5 * result["continuous_nll"]
    )
    np.testing.assert_allclose(result["loss"], expected_total, rtol=1e-7, atol=1e-7)

    compiled = tf.function(lambda out, lab: compute_training_objective(out, lab, config))
    compiled_result = compiled(outputs, labels)
    np.testing.assert_allclose(compiled_result["loss"], result["loss"], rtol=1e-6)


def test_inactive_target_values_do_not_change_any_loss():
    outputs = _outputs(1)
    labels = _labels([4], [19])
    changed = {name: np.array(value, copy=True) for name, value in labels.items()}
    inactive = changed["active_dimension_mask"] == 0.0
    changed["target_unit"][inactive] = np.linspace(
        -100.0, 100.0, np.count_nonzero(inactive), dtype=np.float32
    )

    first = compute_training_objective(outputs, labels)
    second = compute_training_objective(outputs, changed)

    for name in ("loss", "topology_loss", "branch_pattern_loss", "continuous_nll"):
        np.testing.assert_allclose(first[name], second[name], rtol=0.0, atol=0.0)


def test_invalid_pattern_nan_and_mismatched_mask_fail_closed():
    outputs = _outputs(1)
    labels = _labels([0], [0])

    invalid_pattern = {name: np.array(value, copy=True) for name, value in labels.items()}
    invalid_pattern["branch_pattern_id"][0] = 2
    with pytest.raises(tf.errors.InvalidArgumentError, match="invalid for topology"):
        compute_training_objective(outputs, invalid_pattern)

    wrong_mask = {name: np.array(value, copy=True) for name, value in labels.items()}
    wrong_mask["active_dimension_mask"][0, 2] = 1.0
    with pytest.raises(tf.errors.InvalidArgumentError, match="does not match branch"):
        compute_training_objective(outputs, wrong_mask)

    nan_target = {name: np.array(value, copy=True) for name, value in labels.items()}
    nan_target["target_unit"][0, -1] = np.nan
    with pytest.raises(tf.errors.InvalidArgumentError, match="NaN/Inf"):
        compute_training_objective(outputs, nan_target)

    nan_outputs = dict(outputs)
    bad_topology = outputs["topology_logits"].numpy()
    bad_topology[0, 3] = np.nan
    nan_outputs["topology_logits"] = bad_topology
    with pytest.raises(tf.errors.InvalidArgumentError, match="NaN/Inf"):
        compute_training_objective(nan_outputs, labels)


def test_mixed_precision_model_has_finite_objective_and_gradients():
    previous_policy = tf.keras.mixed_precision.global_policy()
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        tf.keras.utils.set_random_seed(41)
        model = build_proposal_model(
            max_points=32, width=16, encoder_blocks=1, mixture_components=3
        )
        labels = _labels([4, 4], [19, 19])
        mask = labels["active_dimension_mask"]
        rng = np.random.default_rng(42)
        inputs = {
            "x": rng.normal(size=(2, 32, 3)).astype(np.float32),
            "point_mask": np.ones((2, 32), dtype=bool),
            "global_features": rng.normal(size=(2, 5)).astype(np.float32),
            "branch_topology": np.eye(34, dtype=np.float32)[[4, 4]],
            "branch_d_present": np.asarray([[1, 1, 0, 0]] * 2, dtype=np.float32),
            "branch_resolution_present": np.ones((2, 1), dtype=np.float32),
            "branch_low": np.zeros((2, BRANCH_DIM), dtype=np.float32),
            "branch_high": np.ones((2, BRANCH_DIM), dtype=np.float32),
            "active_dimension_mask": mask,
        }

        with tf.GradientTape() as tape:
            outputs = model(inputs, training=True)
            result = compute_training_objective(outputs, labels)
        gradients = tape.gradient(result["loss"], model.trainable_variables)

        assert result["loss"].dtype == tf.float32
        assert np.isfinite(result["loss"].numpy())
        assert all(gradient is not None for gradient in gradients)
        assert all(np.all(np.isfinite(gradient.numpy())) for gradient in gradients)
    finally:
        tf.keras.mixed_precision.set_global_policy(previous_policy)


def test_objective_config_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError, match="at least one"):
        TrainingObjectiveConfig(0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="recall"):
        TrainingObjectiveConfig(topology_recall_k=35)
    with pytest.raises(ValueError, match="epsilon"):
        TrainingObjectiveConfig(logistic_epsilon=0.5)
