"""Check global parameters keep their candidate axis during reconstruction."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Training import differentiable_physics as physics


@pytest.mark.parametrize("shared", [True, False])
def test_reconstruction_global_candidate_axis(shared):
    globals_ = tf.ones((1, 5)) if shared else tf.reshape(tf.range(10, dtype=tf.float32), (1, 2, 5))

    def reconstruct(q, types, exist, params, weights, globals_flat, *args, **kwargs):
        tf.debugging.assert_equal(tf.shape(globals_flat), [2, 5])
        expected = tf.ones((2, 5)) if shared else tf.reshape(globals_, (2, 5))
        tf.debugging.assert_equal(globals_flat, expected)
        return tf.ones_like(q)

    with patch.object(physics, "reconstruct_intensity_soft", reconstruct):
        @tf.function
        def evaluate():
            return physics.multi_hypothesis_reconstruction_errors(
                tf.ones((1, 8)), tf.ones((1, 8)), tf.ones((1, 8), tf.bool),
                tf.zeros((1, 2, 4, 4)), tf.zeros((1, 2, 4)),
                tf.zeros((1, 2, 4, 4, 6)), tf.zeros((1, 2, 4)),
                globals_, tf.zeros((1, 2, 4)), tf.zeros((1,)), q_stride=1,
            )
        assert evaluate().shape == (1, 2)
