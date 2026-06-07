"""Losses with exact permutation matching for four unordered slots."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import tensorflow as tf

from TrainSetBuild import schema

PERMUTATIONS = tf.constant(list(itertools.permutations(range(schema.MAX_SLOTS))), dtype=tf.int32)


@dataclass
class LossWeights:
    exist: float = 1.0
    type: float = 1.0
    param: float = 2.0
    weight: float = 0.5
    global_: float = 1.0
    quality: float = 0.0


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

    exist_costs = []
    type_costs = []
    param_costs = []
    weight_costs = []
    total_costs = []

    for p in tf.unstack(PERMUTATIONS):
        tt = tf.gather(target_type, p, axis=1)
        te = tf.gather(target_exist, p, axis=1)
        tp = tf.gather(target_params, p, axis=1)
        tm = tf.gather(target_mask, p, axis=1)
        tw = tf.gather(target_weight, p, axis=1)

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

        total_b = (
            weights.exist * exist_loss_b
            + weights.type * type_loss_b
            + weights.param * param_loss_b
            + weights.weight * weight_loss_b
        )
        exist_costs.append(exist_loss_b)
        type_costs.append(type_loss_b)
        param_costs.append(param_loss_b)
        weight_costs.append(weight_loss_b)
        total_costs.append(total_b)

    exist_costs = tf.stack(exist_costs, axis=0)
    type_costs = tf.stack(type_costs, axis=0)
    param_costs = tf.stack(param_costs, axis=0)
    weight_costs = tf.stack(weight_costs, axis=0)
    total_costs = tf.stack(total_costs, axis=0)
    best_idx = tf.argmin(total_costs, axis=0, output_type=tf.int32)

    slot_loss = tf.reduce_mean(_gather_best(total_costs, best_idx))
    exist_loss = tf.reduce_mean(_gather_best(exist_costs, best_idx))
    type_loss = tf.reduce_mean(_gather_best(type_costs, best_idx))
    param_loss = tf.reduce_mean(_gather_best(param_costs, best_idx))
    weight_loss = tf.reduce_mean(_gather_best(weight_costs, best_idx))
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
        "slot_type_accuracy": type_acc,
        "nonempty_type_accuracy": nonempty_type_acc,
    }
