from __future__ import annotations

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import (
    BRANCH_PATTERN_COUNT,
    VALID_BRANCH_PATTERN_MASK,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model import (
    BRANCH_DIM,
    BRANCH_PARAMETER_LAYOUT,
    DEFAULT_MAX_POINTS,
    DEFAULT_MIXTURE_COMPONENTS,
    build_proposal_model,
    mask_invalid_branch_pattern_logits,
    masked_logistic_normal_nll,
    proposal_output_shapes,
    sample_logistic_normal_mixture,
)


def _inputs(*, batch=2, max_points=48, valid_points=31, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(batch, max_points, 3)).astype(np.float32)
    point_mask = np.zeros((batch, max_points), dtype=bool)
    point_mask[:, :valid_points] = True
    topology = np.zeros((batch, 34), dtype=np.float32)
    topology[:, 4] = 1.0
    d_present = np.zeros((batch, 4), dtype=np.float32)
    d_present[:, :2] = 1.0
    resolution = np.ones((batch, 1), dtype=np.float32)
    active = np.zeros((batch, BRANCH_DIM), dtype=np.float32)
    active[:, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 24, 25]] = 1.0
    low = rng.uniform(0.0, 0.35, size=(batch, BRANCH_DIM)).astype(np.float32)
    high = rng.uniform(0.65, 1.0, size=(batch, BRANCH_DIM)).astype(np.float32)
    return {
        "x": x,
        "point_mask": point_mask,
        "global_features": rng.normal(size=(batch, 5)).astype(np.float32),
        "branch_topology": topology,
        "branch_d_present": d_present,
        "branch_resolution_present": resolution,
        "branch_low": low,
        "branch_high": high,
        "active_dimension_mask": active,
    }


def _small_model(*, max_points=48, mixture_components=4):
    return build_proposal_model(
        max_points=max_points,
        width=24,
        encoder_blocks=2,
        mixture_components=mixture_components,
    )


def test_layout_and_default_output_shapes_are_stable():
    assert DEFAULT_MAX_POINTS == 1000
    assert BRANCH_DIM == 26
    assert len(BRANCH_PARAMETER_LAYOUT) == len(set(BRANCH_PARAMETER_LAYOUT)) == 26
    assert BRANCH_PARAMETER_LAYOUT[:6] == (
        "component_1.log_R",
        "component_1.sigma_R_fraction",
        "component_1.log_h",
        "component_1.sigma_h_fraction",
        "component_1.log_D",
        "component_1.sigma_D_fraction",
    )
    assert BRANCH_PARAMETER_LAYOUT[-2:] == (
        "resolution.log_sigma_res",
        "resolution.nu_res",
    )
    expected = proposal_output_shapes(3)
    assert expected["topology_logits"] == (3, 34)
    assert expected["branch_pattern_logits"] == (3, 34, 32)
    assert expected["mixture_logits"] == (3, DEFAULT_MIXTURE_COMPONENTS)
    assert expected["mixture_loc"] == (3, DEFAULT_MIXTURE_COMPONENTS, 26)

    model = build_proposal_model(max_points=32, width=16, encoder_blocks=1)
    outputs = model(_inputs(batch=3, max_points=32, valid_points=20), training=False)
    assert {name: tuple(value.shape) for name, value in outputs.items()} == expected
    assert np.all(np.isfinite(outputs["mixture_logscale"].numpy()))
    assert np.max(outputs["mixture_logscale"].numpy()) <= 1.0
    assert np.min(outputs["mixture_logscale"].numpy()) >= -5.0


def test_padding_values_cannot_change_any_prediction():
    tf.keras.utils.set_random_seed(11)
    model = _small_model()
    original = _inputs(batch=2, max_points=48, valid_points=27)
    changed = {name: value.copy() for name, value in original.items()}
    changed["x"][:, 27:, :] = np.random.default_rng(99).normal(
        0.0, 1.0e6, size=changed["x"][:, 27:, :].shape
    )
    changed["x"][0, -1, 0] = np.nan
    changed["x"][1, -2, 1] = np.inf

    first = model(original, training=False)
    second = model(changed, training=False)

    for name in first:
        np.testing.assert_allclose(first[name], second[name], rtol=0.0, atol=0.0)


def test_discrete_heads_are_independent_of_all_hard_branch_inputs():
    tf.keras.utils.set_random_seed(13)
    model = _small_model()
    first_inputs = _inputs(batch=2, max_points=48, valid_points=30)
    second_inputs = {name: value.copy() for name, value in first_inputs.items()}
    second_inputs["branch_topology"][:] = 0.0
    second_inputs["branch_topology"][:, 27] = 1.0
    second_inputs["branch_d_present"][:] = 1.0 - second_inputs["branch_d_present"]
    second_inputs["branch_resolution_present"][:] = 0.0
    second_inputs["active_dimension_mask"][:] = 1.0
    second_inputs["branch_low"][:] = 0.05
    second_inputs["branch_high"][:] = 0.95

    first = model(first_inputs, training=False)
    second = model(second_inputs, training=False)

    for name in (
        "topology_logits",
        "branch_pattern_logits",
    ):
        np.testing.assert_array_equal(first[name], second[name])
    assert not np.array_equal(first["mixture_loc"], second["mixture_loc"])


def test_masked_logistic_normal_has_finite_gradients_and_ignores_inactive_targets():
    tf.keras.utils.set_random_seed(17)
    model = _small_model(mixture_components=3)
    inputs = _inputs(batch=2, max_points=48, valid_points=29)
    target = tf.random.uniform((2, BRANCH_DIM), minval=0.05, maxval=0.95, seed=3)

    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        nll = masked_logistic_normal_nll(
            target,
            inputs["active_dimension_mask"],
            outputs["mixture_logits"],
            outputs["mixture_loc"],
            outputs["mixture_logscale"],
        )
        loss = tf.reduce_mean(nll)
        loss += 1.0e-3 * sum(
            tf.reduce_mean(tf.square(outputs[name]))
            for name in (
                "topology_logits",
                "branch_pattern_logits",
            )
        )
    gradients = tape.gradient(loss, model.trainable_variables)

    assert np.isfinite(loss.numpy())
    assert all(gradient is not None for gradient in gradients)
    assert all(np.all(np.isfinite(gradient.numpy())) for gradient in gradients)

    altered = tf.where(
        tf.cast(inputs["active_dimension_mask"], tf.bool), target, 1.0 - target
    )
    altered_nll = masked_logistic_normal_nll(
        altered,
        inputs["active_dimension_mask"],
        outputs["mixture_logits"],
        outputs["mixture_loc"],
        outputs["mixture_logscale"],
    )
    np.testing.assert_allclose(nll, altered_nll, rtol=0.0, atol=0.0)


def test_sampling_is_deterministic_and_neutralizes_inactive_dimensions():
    model = _small_model(mixture_components=3)
    inputs = _inputs(batch=2, max_points=48, valid_points=25)
    outputs = model(inputs, training=False)
    first, first_component = sample_logistic_normal_mixture(
        outputs["mixture_logits"],
        outputs["mixture_loc"],
        outputs["mixture_logscale"],
        inputs["active_dimension_mask"],
        sample_count=8,
        seed=(123, 456),
    )
    second, second_component = sample_logistic_normal_mixture(
        outputs["mixture_logits"],
        outputs["mixture_loc"],
        outputs["mixture_logscale"],
        inputs["active_dimension_mask"],
        sample_count=8,
        seed=(123, 456),
    )

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_component, second_component)
    assert tuple(first.shape) == (2, 8, BRANCH_DIM)
    assert np.all(first.numpy() > 0.0) and np.all(first.numpy() < 1.0)
    active = inputs["active_dimension_mask"][:, None, :].astype(bool)
    inactive_values = np.broadcast_to(first.numpy(), active.shape[:-2] + first.shape[1:])[~active.repeat(8, axis=1)]
    np.testing.assert_array_equal(inactive_values, np.full_like(inactive_values, 0.5))


def test_invalid_catalog_patterns_are_hard_masked():
    assert BRANCH_PATTERN_COUNT == 32
    assert np.asarray(VALID_BRANCH_PATTERN_MASK).shape == (34, 32)
    assert np.count_nonzero(VALID_BRANCH_PATTERN_MASK) == 700

    raw = tf.zeros((2, 34, 32), dtype=tf.float32)
    masked = mask_invalid_branch_pattern_logits(raw).numpy()
    valid = np.asarray(VALID_BRANCH_PATTERN_MASK, dtype=bool)[None, :, :]
    np.testing.assert_array_equal(masked[np.broadcast_to(valid, masked.shape)], 0.0)
    np.testing.assert_array_equal(
        masked[~np.broadcast_to(valid, masked.shape)], -1.0e9
    )


def test_safe_keras_round_trip_preserves_predictions(tmp_path):
    tf.keras.utils.set_random_seed(23)
    model = _small_model(max_points=32, mixture_components=3)
    inputs = _inputs(batch=2, max_points=32, valid_points=21)
    expected = model(inputs, training=False)
    path = tmp_path / "posterior_v8.keras"

    model.save(path)
    loaded = tf.keras.models.load_model(path, safe_mode=True)
    actual = loaded(inputs, training=False)

    assert loaded.name == "posterior_v8_proposal"
    for name in expected:
        np.testing.assert_allclose(expected[name], actual[name], rtol=0.0, atol=0.0)


def test_branch_bounds_fail_closed_outside_normalized_unit_domain():
    model = _small_model()
    inputs = _inputs(batch=1, max_points=48, valid_points=20)
    inputs["branch_low"][0, 0] = -0.01
    with pytest.raises(tf.errors.InvalidArgumentError, match="branch_low must be in"):
        model(inputs, training=False)

    soft_branch = _inputs(batch=1, max_points=48, valid_points=20)
    soft_branch["branch_topology"][0, 4] = 0.5
    soft_branch["branch_topology"][0, 5] = 0.5
    with pytest.raises(tf.errors.InvalidArgumentError, match="must be binary"):
        model(soft_branch, training=False)
