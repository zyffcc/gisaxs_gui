"""Truth labels must reconstruct the GUI's curve without optimization."""
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Training import differentiable_physics as physics
from TrainSetBuild import schema
from TrainSetBuild.physics_adapter import evaluate_clean


@pytest.fixture(autouse=True)
def restore_profile():
    previous = physics._DATASET_PROFILE
    yield
    physics.configure_dataset_physics({"dataset_profile": previous})


@pytest.mark.parametrize("types", [(1,), (2,), (3,), (1, 2, 3, 1)])
@pytest.mark.parametrize("structure", [False, True])
@pytest.mark.parametrize("resolution", [False, True])
def test_truth_forward_matches_authoritative(types, structure, resolution):
    physics.configure_dataset_physics({"dataset_profile": "legacy_v3"})
    q = np.geomspace(.01, 1.0, 80).astype("float32")
    params = np.zeros((1, 4, 6), "float32")
    type_ids = np.zeros((1, 4), "int32")
    active = np.zeros((1, 4), "float32")
    components = []
    for slot, type_id in enumerate(types):
        physical = np.array([12 + slot, .12 if type_id == 3 else 1.3, 25, 2.5,
                             30 if structure else 0, 2 if structure else 0])
        params[0, slot] = schema.normalize_params(physical, type_id)
        type_ids[0, slot] = type_id
        active[0, slot] = 1
        components.append(dict(type_id=type_id, params_phys=physical, weight=1 / len(types)))
    globals_phys = np.array([.002, .04, 3.5, .15 if resolution else 0, 17])
    globals_norm = schema.normalize_global(globals_phys)[None, :]
    expected = evaluate_clean(q, components, dict(zip(schema.GLOBAL_PARAM_NAMES, globals_phys)))
    args = (tf.constant(q[None, :]), tf.constant(type_ids), tf.constant(active),
            tf.constant(params), tf.zeros((1, 4)), tf.constant(globals_norm),
            tf.fill((1, 4), 80.0 if structure else -80.0), tf.constant([float(resolution)]))
    actual = physics.reconstruct_intensity(*args).numpy()[0]
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=1e-7)
    # The prediction-driven path must use the same semantics and finite gradients.
    typed_params = tf.Variable(np.repeat(params[:, :, None, :], 4, axis=2))
    with tf.GradientTape() as tape:
        soft = physics.reconstruct_intensity_soft(
            args[0], tf.one_hot(type_ids, 4) * 160.0 - 80.0,
            tf.constant(active * 160 - 80), typed_params, args[4], args[5], args[6], args[7])
        loss = tf.reduce_sum(tf.math.log(soft))
    np.testing.assert_allclose(soft.numpy()[0], expected, rtol=5e-5, atol=1e-7)
    assert np.isfinite(tape.gradient(loss, typed_params).numpy()).all()


@pytest.mark.parametrize("profile", [None, "unknown", "universal_v4"])
def test_missing_or_unsupported_profile_rejected(profile):
    with pytest.raises(ValueError, match="dataset_profile"):
        physics.configure_dataset_physics({"dataset_profile": profile})


def test_v5_normalization_unchanged():
    physics.configure_dataset_physics({"dataset_profile": "universal_v5"})
    actual = physics.denormalize_component_params(tf.fill((6,), .5)).numpy()
    expected = [schema.denormalize_value(.5, schema.V5_PARAM_NORM_RANGES[name])
                for name in schema.PARAM_NAMES]
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
