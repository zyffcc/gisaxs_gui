"""Losses with exact permutation matching for four unordered slots."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import tensorflow as tf

from TrainSetBuild import schema
from Training.differentiable_physics import reconstruction_log_huber_loss

PERMUTATIONS = tf.constant(list(itertools.permutations(range(schema.MAX_SLOTS))), dtype=tf.int32)


@dataclass
class LossWeights:
    exist: float = 1.0
    type: float = 1.0
    param: float = 2.0
    weight: float = 0.5
    global_: float = 1.0
    quality: float = 0.0
    d_presence: float = 0.5
    spacing: float = 2.0
    reconstruction: object = 0.0
    reconstruction_q_stride: int = 16
    reconstruction_samples_per_batch: int = 2
    count: float = 1.0


def _denormalize_log_tensor(x, name):
    spec = schema.PARAM_NORM_RANGES[name]
    x = tf.clip_by_value(tf.cast(x, tf.float32), 0.0, 1.0)
    return tf.exp(tf.math.log(float(spec.low)) + x * (tf.math.log(float(spec.high)) - tf.math.log(float(spec.low))))


def _safe_mean(x, mask=None, axis=None):
    x = tf.cast(x, tf.float32)
    if mask is None:
        return tf.reduce_mean(x, axis=axis)
    mask = tf.cast(mask, tf.float32)
    return tf.reduce_sum(x * mask, axis=axis) / tf.maximum(tf.reduce_sum(mask, axis=axis), 1.0)


def _gather_best(values_by_perm, best_idx):
    # values_by_perm: [P, B], best_idx: [B]
    b = tf.range(tf.shape(best_idx)[0], dtype=tf.int32)
    return tf.gather_nd(values_by_perm, tf.stack([best_idx, b], axis=1))


def _component_count_distribution(exist_logits):
    """Poisson-binomial P(K=0..MAX_SLOTS) from independent slot probabilities."""
    probabilities = tf.sigmoid(tf.cast(exist_logits, tf.float32))
    distribution = tf.ones((tf.shape(probabilities)[0], 1), dtype=tf.float32)
    for slot in range(schema.MAX_SLOTS):
        p = probabilities[:, slot : slot + 1]
        distribution = tf.pad(distribution * (1.0 - p), [[0, 0], [0, 1]]) + tf.pad(
            distribution * p, [[0, 0], [1, 0]]
        )
    return distribution


def compute_losses(labels, preds, weights: LossWeights | None = None):
    weights = weights or LossWeights()
    target_type = tf.cast(labels["slot_type"], tf.int32)
    target_exist = tf.cast(labels["slot_exist"], tf.float32)
    target_params = tf.cast(labels["slot_params_norm"], tf.float32)
    target_mask = tf.cast(labels["slot_param_mask"], tf.float32)
    target_weight = tf.cast(labels["slot_weight"], tf.float32)
    target_global = tf.cast(labels["global_params_norm"], tf.float32)

    pred_exist_logits = tf.cast(preds["exist_logit"], tf.float32)
    pred_type_logits = tf.cast(preds["type_logits"], tf.float32)
    pred_params = tf.cast(preds["param_mu_norm"], tf.float32)
    pred_param_logstd = tf.clip_by_value(tf.cast(preds["param_logstd_raw"], tf.float32), -5.0, 1.0)
    pred_weight = tf.nn.softmax(tf.cast(preds["weight_logit"], tf.float32), axis=-1)
    pred_global = tf.cast(preds["global_mu_norm"], tf.float32)
    pred_global_logstd = tf.clip_by_value(tf.cast(preds["global_logstd_raw"], tf.float32), -5.0, 1.0)
    pred_d_present_logits = tf.cast(preds["d_present_logit"], tf.float32)

    exist_costs = []
    type_costs = []
    param_costs = []
    weight_costs = []
    total_costs = []
    d_presence_costs = []

    for p in tf.unstack(PERMUTATIONS):
        tt = tf.gather(target_type, p, axis=1)
        te = tf.gather(target_exist, p, axis=1)
        tp = tf.gather(target_params, p, axis=1)
        tm = tf.gather(target_mask, p, axis=1)
        tw = tf.gather(target_weight, p, axis=1)
        d_present = tm[:, :, 4] * te

        exist_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=te, logits=pred_exist_logits)
        exist_loss_b = tf.reduce_mean(exist_ce, axis=1)

        type_ce = tf.keras.losses.sparse_categorical_crossentropy(tt, pred_type_logits, from_logits=True)
        type_weight = 1.0 + te
        type_loss_b = tf.reduce_sum(type_ce * type_weight, axis=1) / tf.maximum(tf.reduce_sum(type_weight, axis=1), 1.0)

        onehot = tf.one_hot(tt, schema.NUM_TYPES, dtype=tf.float32)
        pp = tf.reduce_sum(pred_params * onehot[:, :, :, tf.newaxis], axis=2)
        ps = tf.reduce_sum(pred_param_logstd * onehot[:, :, :, tf.newaxis], axis=2)
        param_nll = 0.5 * (tf.square(tp - pp) / tf.exp(2.0 * ps) + 2.0 * ps)
        param_mask = tm * te[:, :, tf.newaxis]
        param_loss_b = tf.reduce_sum(param_nll * param_mask, axis=[1, 2]) / tf.maximum(tf.reduce_sum(param_mask, axis=[1, 2]), 1.0)

        weight_sq = tf.square(pred_weight - tw)
        weight_loss_b = tf.reduce_mean(weight_sq, axis=1)
        d_presence_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_present, logits=pred_d_present_logits)
        d_presence_loss_b = tf.reduce_sum(d_presence_ce * te, axis=1) / tf.maximum(tf.reduce_sum(te, axis=1), 1.0)

        total_b = (
            weights.exist * exist_loss_b
            + weights.type * type_loss_b
            + weights.param * param_loss_b
            + weights.weight * weight_loss_b
            + weights.d_presence * d_presence_loss_b
        )
        exist_costs.append(exist_loss_b)
        type_costs.append(type_loss_b)
        param_costs.append(param_loss_b)
        weight_costs.append(weight_loss_b)
        total_costs.append(total_b)
        d_presence_costs.append(d_presence_loss_b)

    exist_costs = tf.stack(exist_costs, axis=0)
    type_costs = tf.stack(type_costs, axis=0)
    param_costs = tf.stack(param_costs, axis=0)
    weight_costs = tf.stack(weight_costs, axis=0)
    total_costs = tf.stack(total_costs, axis=0)
    d_presence_costs = tf.stack(d_presence_costs, axis=0)
    best_idx = tf.argmin(total_costs, axis=0, output_type=tf.int32)

    slot_loss = tf.reduce_mean(_gather_best(total_costs, best_idx))
    exist_loss = tf.reduce_mean(_gather_best(exist_costs, best_idx))
    type_loss = tf.reduce_mean(_gather_best(type_costs, best_idx))
    param_loss = tf.reduce_mean(_gather_best(param_costs, best_idx))
    weight_loss = tf.reduce_mean(_gather_best(weight_costs, best_idx))
    d_presence_loss = tf.reduce_mean(_gather_best(d_presence_costs, best_idx))
    global_nll = 0.5 * (tf.square(target_global - pred_global) / tf.exp(2.0 * pred_global_logstd) + 2.0 * pred_global_logstd)
    global_loss = tf.reduce_mean(global_nll)
    quality_loss = tf.reduce_mean(tf.square(tf.cast(preds["quality"], tf.float32))) * 0.0
    total_loss = slot_loss + weights.global_ * global_loss + weights.quality * quality_loss

    # Metrics use best matching only. Gather target slots for each batch with the selected permutation.
    best_perm = tf.gather(PERMUTATIONS, best_idx)
    batch_indices = tf.tile(tf.range(tf.shape(best_perm)[0])[:, tf.newaxis], [1, schema.MAX_SLOTS])
    gather_idx = tf.stack([batch_indices, best_perm], axis=-1)
    matched_type = tf.gather_nd(target_type, gather_idx)
    matched_exist = tf.gather_nd(target_exist, gather_idx)
    matched_mask = tf.gather_nd(target_mask, gather_idx)
    matched_onehot = tf.one_hot(matched_type, schema.NUM_TYPES, dtype=tf.float32)
    matched_pred_params = tf.reduce_sum(pred_params * matched_onehot[:, :, :, tf.newaxis], axis=2)
    pred_r = _denormalize_log_tensor(matched_pred_params[:, :, 0], "R")
    pred_h = _denormalize_log_tensor(matched_pred_params[:, :, 2], "h")
    pred_d = _denormalize_log_tensor(matched_pred_params[:, :, 4], "D")
    is_random_cylinder = tf.cast(tf.equal(matched_type, schema.TYPE_CYLINDER), tf.float32)
    exclusion_size = (1.0 - is_random_cylinder) * (2.0 * pred_r) + is_random_cylinder * tf.sqrt(
        tf.square(2.0 * pred_r) + tf.square(pred_h)
    )
    active = tf.cast(matched_exist > 0.5, tf.float32)
    max_size = tf.reduce_max(tf.where(active > 0.0, exclusion_size, tf.zeros_like(exclusion_size)), axis=1)
    mean_size = tf.reduce_sum(exclusion_size * active, axis=1) / tf.maximum(tf.reduce_sum(active, axis=1), 1.0)
    d_rule = tf.cast(labels["d_spacing_rule"], tf.float32)
    threshold = d_rule[:, schema.D_RULE_MAX] * max_size + d_rule[:, schema.D_RULE_MEAN] * mean_size
    d_present_mask = matched_mask[:, :, 4] * active
    spacing_violation = tf.nn.relu(threshold[:, tf.newaxis] * 1.001 - pred_d) / float(schema.PARAM_RANGES["D"].high)
    spacing_loss = _safe_mean(tf.square(spacing_violation), d_present_mask)
    reconstruction_weight = tf.cast(weights.reconstruction, tf.float32)

    def compute_reconstruction():
        return reconstruction_log_huber_loss(
            labels["q"],
            labels["I_clean"],
            labels["point_mask"],
            matched_type,
            matched_exist,
            matched_pred_params,
            preds["weight_logit"],
            pred_global,
            pred_d_present_logits,
            q_stride=weights.reconstruction_q_stride,
            max_samples_per_batch=weights.reconstruction_samples_per_batch,
        )

    if all(key in labels for key in ("q", "I_clean", "point_mask")):
        reconstruction_loss = tf.cond(
            reconstruction_weight > 0.0,
            compute_reconstruction,
            lambda: tf.constant(0.0, dtype=tf.float32),
        )
    else:
        reconstruction_loss = tf.constant(0.0, dtype=tf.float32)
    total_loss = total_loss + weights.spacing * spacing_loss + reconstruction_weight * reconstruction_loss
    count_distribution = _component_count_distribution(pred_exist_logits)
    target_count = tf.cast(tf.reduce_sum(target_exist, axis=1), tf.int32)
    target_count_probability = tf.gather(count_distribution, target_count, axis=1, batch_dims=1)
    count_loss = tf.reduce_mean(-tf.math.log(tf.maximum(target_count_probability, 1e-8)))
    predicted_count = tf.argmax(count_distribution, axis=1, output_type=tf.int32)
    count_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_count, target_count), tf.float32))
    total_loss = total_loss + weights.count * count_loss
    pred_type = tf.argmax(pred_type_logits, axis=-1, output_type=tf.int32)
    type_acc = tf.reduce_mean(tf.cast(tf.equal(pred_type, matched_type), tf.float32))
    nonempty_mask = tf.cast(matched_exist > 0.5, tf.float32)
    nonempty_type_acc = _safe_mean(tf.cast(tf.equal(pred_type, matched_type), tf.float32), nonempty_mask)

    return {
        "total_loss": total_loss,
        "exist_loss": exist_loss,
        "type_loss": type_loss,
        "param_loss": param_loss,
        "weight_loss": weight_loss,
        "global_loss": global_loss,
        "quality_loss": quality_loss,
        "d_presence_loss": d_presence_loss,
        "spacing_loss": spacing_loss,
        "reconstruction_loss": reconstruction_loss,
        "count_loss": count_loss,
        "component_count_accuracy": count_accuracy,
        "slot_type_accuracy": type_acc,
        "nonempty_type_accuracy": nonempty_type_acc,
    }
