import numpy as np
import tensorflow as tf

from TrainSetBuild import schema
from Training.losses import _ordinal_tier_binary_crossentropy, compute_set_losses


def test_saturated_wrong_tier_prediction_keeps_a_finite_gradient():
    raw = tf.Variable(tf.fill((2, 16, 3), 30.0))
    target = tf.zeros((2, 16, 3), tf.float32)

    with tf.GradientTape() as tape:
        loss = _ordinal_tier_binary_crossentropy(target, raw)
    gradient = tape.gradient(loss, raw)

    assert np.isfinite(float(loss.numpy()))
    assert bool(tf.reduce_all(tf.math.is_finite(gradient)))
    assert float(tf.reduce_mean(gradient[..., 2]).numpy()) > 0.001


def test_set_loss_is_finite_and_differentiable_with_dustbin_modes():
    batch, hypotheses, modes = 2, 16, 16
    solution_mask = np.zeros((batch, modes), np.float32)
    solution_mask[0, :3] = 1.0
    solution_mask[1, 0] = 1.0
    slot_type = np.zeros((batch, modes, schema.MAX_SLOTS), np.int32)
    slot_type[:, :, 0] = schema.TYPE_SPHERE
    slot_exist = (slot_type > 0).astype(np.float32)
    params = np.zeros((batch, modes, schema.MAX_SLOTS, schema.P_MAX), np.float32)
    params[0, :3, 0, 0] = [0.2, 0.5, 0.8]
    param_mask = np.zeros_like(params)
    param_mask[:, :, 0, [0, 4]] = 1.0
    slot_weight = np.zeros((batch, modes, schema.MAX_SLOTS), np.float32)
    slot_weight[:, :, 0] = 1.0
    tier = np.full((batch, modes), -1.0, np.float32)
    tier[0, :3] = [0.0, 0.01, 0.05]
    tier[1, 0] = 0.0
    labels = {
        "solution_mask": tf.constant(solution_mask),
        "solution_tier": tf.constant(tier),
        "solution_label_weight": tf.constant(solution_mask),
        "solution_slot_type": tf.constant(slot_type),
        "solution_slot_exist": tf.constant(slot_exist),
        "solution_slot_params_norm": tf.constant(params),
        "solution_slot_param_mask": tf.constant(param_mask),
        "solution_slot_weight": tf.constant(slot_weight),
        "global_params_norm": tf.zeros((batch, schema.G_MAX)),
        "global_param_mask": tf.ones((batch, schema.G_MAX)),
        "q": tf.tile(tf.linspace(0.01, 2.0, 64)[tf.newaxis, :], [batch, 1]),
        "I_clean": tf.ones((batch, 64), tf.float32),
        "point_mask": tf.ones((batch, 64), tf.bool),
        "resolution_present": tf.ones((batch,), tf.float32),
    }
    variables = {
        "exist_logit": tf.Variable(tf.zeros((batch, hypotheses, schema.MAX_SLOTS))),
        "type_logits": tf.Variable(tf.zeros((batch, hypotheses, schema.MAX_SLOTS, schema.NUM_TYPES))),
        "param_mu_norm": tf.Variable(tf.fill((batch, hypotheses, schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX), 0.5)),
        "param_logstd_raw": tf.Variable(tf.zeros((batch, hypotheses, schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX))),
        "weight_logit": tf.Variable(tf.zeros((batch, hypotheses, schema.MAX_SLOTS))),
        "d_present_logit": tf.Variable(tf.zeros((batch, hypotheses, schema.MAX_SLOTS))),
        "candidate_active_logit": tf.Variable(tf.zeros((batch, hypotheses))),
        "tier_probability": tf.Variable(tf.fill((batch, hypotheses, 3), 0.5)),
        "global_mu_norm_shared": tf.Variable(tf.zeros((batch, schema.G_MAX))),
        "global_logstd_raw_shared": tf.Variable(tf.zeros((batch, schema.G_MAX))),
        "resolution_present_logit": tf.Variable(tf.zeros((batch,))),
    }
    with tf.GradientTape() as tape:
        losses = compute_set_losses(labels, variables)
    gradients = tape.gradient(losses["total_loss"], list(variables.values()))
    assert all(np.isfinite(float(value.numpy())) for value in losses.values())
    assert all(gradient is not None for gradient in gradients)
    assert all(bool(tf.reduce_all(tf.math.is_finite(gradient))) for gradient in gradients)
    assert float(losses["ground_truth_modes"].numpy()) == 2.0
