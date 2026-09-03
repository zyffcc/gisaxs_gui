"""Permutation-invariant best-of-M losses for direct multi-solution proposals."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import tensorflow as tf

from TrainSetBuild import schema
from Training.differentiable_physics import multi_hypothesis_reconstruction_errors

PERMUTATIONS = tf.constant(list(itertools.permutations(range(schema.MAX_SLOTS))), dtype=tf.int32)


@dataclass
class LossWeights:
    exist: float = 1.0
    type: float = 1.0
    param: float = 2.0
    weight: float = 0.5
    global_: float = 1.0
    quality: float = 0.20
    d_presence: float = 0.5
    spacing: float = 2.0
    reconstruction: object = 0.0
    reconstruction_q_stride: int = 32
    reconstruction_samples_per_batch: int = 1
    reconstruction_sampling_mode: object = 0
    reconstruction_multiscale_min_points: object = 64
    reconstruction_multiscale_max_points: object = 128
    count: float = 1.0
    set_count: float = 0.5
    diversity: float = 0.10
    usage_balance: float = 0.01
    diversity_margin: float = 0.12
    winner_activation: float = 0.10
    activation_quality: float = 0.25
    candidate_complexity: float = 0.02
    activation_temperature: float = 0.03
    inactive_physics_floor: float = 0.05


def _ensure_hypothesis_axis(preds):
    """Keep H=1 backward compatibility for diagnostic scripts."""
    out = dict(preds)
    ranks = {
        "exist_logit": 2,
        "type_logits": 3,
        "param_mu_norm": 4,
        "param_logstd_raw": 4,
        "weight_logit": 2,
        "global_mu_norm": 2,
        "global_logstd_raw": 2,
        "d_present_logit": 2,
    }
    for key, old_rank in ranks.items():
        if out[key].shape.rank == old_rank:
            out[key] = out[key][:, tf.newaxis, ...]
    if "hypothesis_logit" not in out:
        out["hypothesis_logit"] = tf.zeros(tf.shape(out["exist_logit"])[:2], tf.float32)
    if "candidate_active_logit" not in out:
        # Old models had no independent gate and therefore treated every
        # hypothesis as active.
        out["candidate_active_logit"] = tf.ones(tf.shape(out["exist_logit"])[:2], tf.float32) * 30.0
    return out


def _gather_best_permutation(values, best_perm_idx):
    # values [P,B,H], best_perm_idx [B,H]
    return tf.gather(tf.transpose(values, [1, 2, 0]), best_perm_idx, axis=2, batch_dims=2)


def _gather_hypothesis(values, best_h):
    # values [B,H,...], best_h [B]
    return tf.gather(values, best_h, axis=1, batch_dims=1)


def _component_count_distribution(exist_logits):
    probabilities = tf.sigmoid(tf.cast(exist_logits, tf.float32))
    distribution = tf.ones(tf.concat([tf.shape(probabilities)[:-1], [1]], axis=0), tf.float32)
    for slot in range(schema.MAX_SLOTS):
        p = probabilities[..., slot : slot + 1]
        rank = distribution.shape.rank
        paddings_left = [[0, 0]] * (rank - 1) + [[0, 1]]
        paddings_right = [[0, 0]] * (rank - 1) + [[1, 0]]
        distribution = tf.pad(distribution * (1.0 - p), paddings_left) + tf.pad(distribution * p, paddings_right)
    return distribution


def _denormalize_log_tensor(x, name):
    spec = schema.PARAM_NORM_RANGES[name]
    x = tf.clip_by_value(tf.cast(x, tf.float32), 0.0, 1.0)
    return tf.exp(tf.math.log(float(spec.low)) + x * (tf.math.log(float(spec.high)) - tf.math.log(float(spec.low))))


def _mean_selected(matrix, best_h):
    return tf.reduce_mean(_gather_hypothesis(matrix, best_h))


def _gather_permutation_cost(values, indices):
    """values [P,B,H,L], indices [B,H,L] -> [B,H,L]."""
    return tf.gather(tf.transpose(values, [1, 2, 3, 0]), indices, axis=3, batch_dims=3)


def _sinkhorn(log_alpha, iterations=20):
    log_assignment = tf.cast(log_alpha, tf.float32)
    for _ in range(iterations):
        log_assignment -= tf.reduce_logsumexp(log_assignment, axis=2, keepdims=True)
        log_assignment -= tf.reduce_logsumexp(log_assignment, axis=1, keepdims=True)
    return tf.exp(log_assignment)


def _ordinal_tier_binary_crossentropy(target, raw):
    """Stable BCE for nested p(<.01) <= p(<.03) <= p(<.05) outputs.

    Computing BCE from clipped probabilities gives exactly zero gradient for
    a confidently wrong warm-started head.  These log-probabilities keep that
    gradient finite without changing the ordinal probability definition used
    by inference.
    """
    target = tf.cast(target, tf.float32)
    raw = tf.cast(raw, tf.float32)
    log_sigmoid = tf.math.log_sigmoid(raw)
    log_one_minus_sigmoid = tf.math.log_sigmoid(-raw)

    log_p05 = log_sigmoid[..., 2]
    log_p03 = log_p05 + log_sigmoid[..., 1]
    log_p01 = log_p03 + log_sigmoid[..., 0]

    log_not_p05 = log_one_minus_sigmoid[..., 2]
    log_not_p03 = tf.reduce_logsumexp(
        tf.stack(
            [
                log_not_p05,
                log_p05 + log_one_minus_sigmoid[..., 1],
            ],
            axis=-1,
        ),
        axis=-1,
    )
    log_not_p01 = tf.reduce_logsumexp(
        tf.stack(
            [
                log_not_p03,
                log_p03 + log_one_minus_sigmoid[..., 0],
            ],
            axis=-1,
        ),
        axis=-1,
    )

    log_probability = tf.stack([log_p01, log_p03, log_p05], axis=-1)
    log_not_probability = tf.stack(
        [log_not_p01, log_not_p03, log_not_p05], axis=-1
    )
    return -tf.reduce_mean(
        target * log_probability + (1.0 - target) * log_not_probability
    )


def compute_set_losses(labels, preds, weights: LossWeights | None = None):
    """Permutation-invariant slot costs plus dustbin set matching over modes."""
    weights = weights or LossWeights()
    target_type = tf.cast(labels["solution_slot_type"], tf.int32)             # [B,L,S]
    target_exist = tf.cast(labels["solution_slot_exist"], tf.float32)
    target_params = tf.cast(labels["solution_slot_params_norm"], tf.float32)
    target_param_mask = tf.cast(labels["solution_slot_param_mask"], tf.float32)
    target_weight = tf.cast(labels["solution_slot_weight"], tf.float32)
    solution_mask = tf.cast(labels["solution_mask"], tf.float32)
    solution_tier = tf.cast(labels["solution_tier"], tf.float32)
    label_weight = tf.cast(labels["solution_label_weight"], tf.float32)

    pred_exist = tf.cast(preds["exist_logit"], tf.float32)                    # [B,H,S]
    pred_type = tf.cast(preds["type_logits"], tf.float32)
    pred_params = tf.cast(preds["param_mu_norm"], tf.float32)
    pred_std = tf.clip_by_value(tf.cast(preds["param_logstd_raw"], tf.float32), -5.0, 1.0)
    pred_weight = tf.nn.softmax(tf.cast(preds["weight_logit"], tf.float32), axis=-1)
    pred_d = tf.cast(preds["d_present_logit"], tf.float32)
    active_logit = tf.cast(preds["candidate_active_logit"], tf.float32)
    tier_probability = tf.clip_by_value(tf.cast(preds["tier_probability"], tf.float32), 1e-6, 1.0 - 1e-6)

    exist_costs, type_costs, param_costs, weight_costs, d_costs, total_costs = [], [], [], [], [], []
    for permutation in tf.unstack(PERMUTATIONS):
        tt = tf.gather(target_type, permutation, axis=2)[:, tf.newaxis, :, :]
        te = tf.gather(target_exist, permutation, axis=2)[:, tf.newaxis, :, :]
        tp = tf.gather(target_params, permutation, axis=2)[:, tf.newaxis, :, :, :]
        tm = tf.gather(target_param_mask, permutation, axis=2)[:, tf.newaxis, :, :, :]
        tw = tf.gather(target_weight, permutation, axis=2)[:, tf.newaxis, :, :]

        common_slot_shape = [
            tf.shape(pred_exist)[0], tf.shape(pred_exist)[1], tf.shape(target_type)[1], schema.MAX_SLOTS
        ]
        te_full = tf.broadcast_to(te, common_slot_shape)
        pred_exist_full = tf.broadcast_to(pred_exist[:, :, tf.newaxis, :], common_slot_shape)
        exist = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=te_full, logits=pred_exist_full), axis=3
        )
        type_labels = tf.broadcast_to(
            tt,
            [tf.shape(pred_type)[0], tf.shape(pred_type)[1], tf.shape(target_type)[1], schema.MAX_SLOTS],
        )
        type_logits_full = tf.broadcast_to(
            pred_type[:, :, tf.newaxis, :, :],
            [tf.shape(pred_type)[0], tf.shape(pred_type)[1], tf.shape(target_type)[1], schema.MAX_SLOTS, schema.NUM_TYPES],
        )
        type_ce = tf.keras.losses.sparse_categorical_crossentropy(type_labels, type_logits_full, from_logits=True)
        type_factor = 1.0 + te_full
        type_cost = tf.reduce_sum(type_ce * type_factor, axis=3) / tf.maximum(tf.reduce_sum(type_factor, axis=3), 1.0)
        onehot = tf.one_hot(tt, schema.NUM_TYPES, dtype=tf.float32)
        selected_params = tf.reduce_sum(pred_params[:, :, tf.newaxis, :, :, :] * onehot[..., tf.newaxis], axis=4)
        selected_std = tf.reduce_sum(pred_std[:, :, tf.newaxis, :, :, :] * onehot[..., tf.newaxis], axis=4)
        param_nll = 0.5 * (tf.square(tp - selected_params) / tf.exp(2.0 * selected_std) + 2.0 * selected_std)
        param_factor = tm * te[..., tf.newaxis]
        param_cost = tf.reduce_sum(param_nll * param_factor, axis=[3, 4]) / tf.maximum(
            tf.reduce_sum(param_factor, axis=[3, 4]), 1.0
        )
        weight_cost = tf.reduce_mean(tf.square(pred_weight[:, :, tf.newaxis, :] - tw), axis=3)
        d_target = tf.broadcast_to(tm[..., 4] * te, common_slot_shape)
        pred_d_full = tf.broadcast_to(pred_d[:, :, tf.newaxis, :], common_slot_shape)
        d_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=d_target, logits=pred_d_full)
        d_cost = tf.reduce_sum(d_ce * te_full, axis=3) / tf.maximum(tf.reduce_sum(te_full, axis=3), 1.0)
        total = (
            weights.exist * exist + weights.type * type_cost + weights.param * param_cost
            + weights.weight * weight_cost + weights.d_presence * d_cost
        )
        exist_costs.append(exist); type_costs.append(type_cost); param_costs.append(param_cost)
        weight_costs.append(weight_cost); d_costs.append(d_cost); total_costs.append(total)

    total_stack = tf.stack(total_costs)
    best_perm_idx = tf.argmin(total_stack, axis=0, output_type=tf.int32)
    structural_cost = _gather_permutation_cost(total_stack, best_perm_idx)
    exist_cost = _gather_permutation_cost(tf.stack(exist_costs), best_perm_idx)
    type_cost = _gather_permutation_cost(tf.stack(type_costs), best_perm_idx)
    param_cost = _gather_permutation_cost(tf.stack(param_costs), best_perm_idx)
    weight_cost = _gather_permutation_cost(tf.stack(weight_costs), best_perm_idx)
    d_cost = _gather_permutation_cost(tf.stack(d_costs), best_perm_idx)

    active_positive = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(active_logit), logits=active_logit)
    active_negative = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(active_logit), logits=active_logit)
    component_count_distribution = _component_count_distribution(pred_exist)
    target_count = tf.cast(tf.reduce_sum(target_exist, axis=2), tf.int32)
    target_count_onehot = tf.one_hot(target_count, schema.MAX_SLOTS + 1, dtype=tf.float32)
    count_probability = tf.reduce_sum(
        component_count_distribution[:, :, tf.newaxis, :] * target_count_onehot[:, tf.newaxis, :, :],
        axis=3,
    )
    count_ce = -tf.math.log(tf.maximum(count_probability, 1e-7))
    valid_cost = (
        structural_cost * label_weight[:, tf.newaxis, :]
        + float(weights.set_count) * count_ce
        + 0.25 * active_positive[:, :, tf.newaxis]
    )
    assignment_cost = tf.where(
        solution_mask[:, tf.newaxis, :] > 0.5,
        valid_cost,
        active_negative[:, :, tf.newaxis],
    )
    assignment = tf.stop_gradient(_sinkhorn(-assignment_cost / 0.10, iterations=24))
    valid_assignment = assignment * solution_mask[:, tf.newaxis, :]
    invalid_assignment = assignment * (1.0 - solution_mask[:, tf.newaxis, :])
    valid_den = tf.maximum(tf.reduce_sum(valid_assignment * label_weight[:, tf.newaxis, :]), 1.0)
    coverage_loss = tf.reduce_sum(valid_assignment * label_weight[:, tf.newaxis, :] * structural_cost) / valid_den
    dustbin_loss = (
        tf.reduce_sum(valid_assignment * active_positive[:, :, tf.newaxis])
        + tf.reduce_sum(invalid_assignment * active_negative[:, :, tf.newaxis])
    ) / tf.cast(tf.shape(active_logit)[0] * tf.shape(active_logit)[1], tf.float32)

    pred_global = tf.cast(
        preds["global_mu_norm_shared"]
        if "global_mu_norm_shared" in preds else preds["global_mu_norm"][:, 0, :],
        tf.float32,
    )
    pred_global_std = tf.clip_by_value(
        tf.cast(
            preds["global_logstd_raw_shared"]
            if "global_logstd_raw_shared" in preds else preds["global_logstd_raw"][:, 0, :],
            tf.float32,
        ),
        -5.0,
        1.0,
    )
    target_global = tf.cast(labels["global_params_norm"], tf.float32)
    global_mask = tf.cast(labels.get("global_param_mask", tf.ones_like(target_global)), tf.float32)
    global_nll = 0.5 * (tf.square(target_global - pred_global) / tf.exp(2.0 * pred_global_std) + 2.0 * pred_global_std)
    global_loss = tf.reduce_sum(global_nll * global_mask) / tf.maximum(tf.reduce_sum(global_mask), 1.0)

    resolution_target = tf.cast(labels.get("resolution_present", global_mask[:, 3] > 0.5), tf.float32)
    resolution_logit = tf.cast(preds["resolution_present_logit"], tf.float32)
    resolution_example_weight = tf.where(resolution_target > 0.5, 1.0, 4.0)
    resolution_presence_loss = tf.reduce_sum(
        resolution_example_weight * tf.nn.sigmoid_cross_entropy_with_logits(
            labels=resolution_target, logits=resolution_logit
        )
    ) / tf.reduce_sum(resolution_example_weight)

    physics_errors = multi_hypothesis_reconstruction_errors(
        labels["q"], labels["I_clean"], labels["point_mask"],
        pred_type, pred_exist, pred_params, tf.cast(preds["weight_logit"], tf.float32),
        pred_global, pred_d, resolution_logit,
        q_stride=weights.reconstruction_q_stride,
        max_samples_per_batch=weights.reconstruction_samples_per_batch,
        sampling_mode=weights.reconstruction_sampling_mode,
        multiscale_min_points=weights.reconstruction_multiscale_min_points,
        multiscale_max_points=weights.reconstruction_multiscale_max_points,
    )
    physics_batch = tf.shape(physics_errors)[0]
    assigned_candidate_mass = tf.reduce_sum(valid_assignment[:physics_batch], axis=2)
    reconstruction_loss = tf.reduce_sum(
        assigned_candidate_mass * tf.math.log1p(physics_errors)
    ) / tf.maximum(tf.reduce_sum(assigned_candidate_mass), 1.0)

    # The quality head is calibrated against what inference actually checks:
    # this candidate's current exact/differentiable curve error, not the
    # neighbour-search tier attached to a historical sidecar solution.
    physics_tier_target = tf.stack(
        [
            tf.cast(physics_errors < 0.01, tf.float32),
            tf.cast(physics_errors < 0.03, tf.float32),
            tf.cast(physics_errors < 0.05, tf.float32),
        ], axis=-1,
    )
    physics_tier_target = tf.stop_gradient(physics_tier_target)
    selected_tier_probability = tier_probability[:physics_batch]
    if "tier_raw" in preds:
        tier_bce = _ordinal_tier_binary_crossentropy(
            physics_tier_target,
            tf.cast(preds["tier_raw"][:physics_batch], tf.float32),
        )
    else:
        # Backward-compatible path for diagnostic callers that only provide
        # probabilities.  The training view always exposes tier_raw.
        tier_bce = -tf.reduce_mean(
            physics_tier_target * tf.math.log(selected_tier_probability)
            + (1.0 - physics_tier_target)
            * tf.math.log(1.0 - selected_tier_probability)
        )
    # False-positive pressure on the <0.05 head is intentionally asymmetric:
    # the deployment gate targets >=80% precision, and can sacrifice recall.
    quality_false_positive_penalty = tf.reduce_mean(
        selected_tier_probability[..., 2] * (1.0 - physics_tier_target[..., 2])
    )
    tier_quality_loss = tier_bce + quality_false_positive_penalty
    count_loss = tf.reduce_sum(valid_assignment * count_ce) / tf.maximum(tf.reduce_sum(valid_assignment), 1.0)
    total_loss = (
        coverage_loss + weights.global_ * global_loss + 0.50 * dustbin_loss
        + 0.50 * resolution_presence_loss + weights.quality * tier_quality_loss
        + float(weights.set_count) * count_loss
        + tf.cast(weights.reconstruction, tf.float32) * reconstruction_loss
    )

    valid_mass = tf.maximum(tf.reduce_sum(valid_assignment), 1.0)
    pred_count = tf.argmax(component_count_distribution, axis=2, output_type=tf.int32)
    count_correct = tf.cast(tf.equal(pred_count[:, :, tf.newaxis], target_count[:, tf.newaxis, :]), tf.float32)
    count_accuracy = tf.reduce_sum(valid_assignment * count_correct) / valid_mass
    active_probability = tf.sigmoid(active_logit)
    ground_truth_modes = tf.reduce_mean(tf.reduce_sum(solution_mask, axis=1))
    expected_active = tf.reduce_mean(tf.reduce_sum(active_probability, axis=1))

    def assigned_mean(component):
        return tf.reduce_sum(valid_assignment * component) / valid_mass

    zero = tf.constant(0.0, tf.float32)
    return {
        "total_loss": total_loss,
        "set_coverage_loss": coverage_loss,
        "dustbin_loss": dustbin_loss,
        "tier_quality_loss": tier_quality_loss,
        "exist_loss": assigned_mean(exist_cost), "type_loss": assigned_mean(type_cost),
        "param_loss": assigned_mean(param_cost), "weight_loss": assigned_mean(weight_cost),
        "global_loss": global_loss, "quality_loss": tier_quality_loss,
        "d_presence_loss": assigned_mean(d_cost), "spacing_loss": zero,
        "reconstruction_loss": reconstruction_loss,
        "physics_coverage_loss": tf.reduce_mean(tf.reduce_min(physics_errors, axis=1)),
        "resolution_presence_loss": resolution_presence_loss,
        "resolution_presence_accuracy": tf.reduce_mean(tf.cast(
            tf.equal(resolution_logit > 0.0, resolution_target > 0.5), tf.float32
        )),
        "quality_false_positive_penalty": quality_false_positive_penalty,
        "count_loss": count_loss, "diversity_loss": zero, "raw_parameter_diversity_loss": zero,
        "physics_valid_diversity_loss": zero, "hypothesis_usage_balance_loss": zero,
        "hypothesis_confidence_loss": zero, "winner_activation_loss": dustbin_loss,
        "activation_quality_loss": tier_quality_loss, "candidate_complexity_loss": zero,
        "expected_active_candidates": expected_active, "pseudo_active_candidates": ground_truth_modes,
        "ground_truth_modes": ground_truth_modes,
        "active_hypotheses": tf.reduce_mean(
            tf.reduce_sum(tf.cast(active_probability > 0.5, tf.float32), axis=1)
        ),
        "hypothesis_usage_entropy": zero,
        "component_count_accuracy": count_accuracy,
        "slot_type_accuracy": tf.exp(-assigned_mean(type_cost)),
        "nonempty_type_accuracy": tf.exp(-assigned_mean(type_cost)),
    }


def compute_losses(labels, preds, weights: LossWeights | None = None):
    weights = weights or LossWeights()
    if "solution_mask" in labels:
        return compute_set_losses(labels, preds, weights)
    preds = _ensure_hypothesis_axis(preds)
    target_type = tf.cast(labels["slot_type"], tf.int32)
    target_exist = tf.cast(labels["slot_exist"], tf.float32)
    target_params = tf.cast(labels["slot_params_norm"], tf.float32)
    target_mask = tf.cast(labels["slot_param_mask"], tf.float32)
    target_weight = tf.cast(labels["slot_weight"], tf.float32)
    target_global = tf.cast(labels["global_params_norm"], tf.float32)
    global_mask = tf.cast(labels.get("global_param_mask", tf.ones_like(target_global)), tf.float32)

    pred_exist = tf.cast(preds["exist_logit"], tf.float32)
    pred_type = tf.cast(preds["type_logits"], tf.float32)
    pred_params = tf.cast(preds["param_mu_norm"], tf.float32)
    pred_std = tf.clip_by_value(tf.cast(preds["param_logstd_raw"], tf.float32), -5.0, 1.0)
    pred_weight = tf.nn.softmax(tf.cast(preds["weight_logit"], tf.float32), axis=-1)
    pred_global = tf.cast(preds["global_mu_norm"], tf.float32)
    pred_global_std = tf.clip_by_value(tf.cast(preds["global_logstd_raw"], tf.float32), -5.0, 1.0)
    pred_d = tf.cast(preds["d_present_logit"], tf.float32)
    hypotheses = tf.shape(pred_exist)[1]

    exist_costs, type_costs, param_costs = [], [], []
    weight_costs, d_presence_costs, total_costs = [], [], []
    for permutation in tf.unstack(PERMUTATIONS):
        tt = tf.gather(target_type, permutation, axis=1)[:, tf.newaxis, :]
        te = tf.gather(target_exist, permutation, axis=1)[:, tf.newaxis, :]
        tp = tf.gather(target_params, permutation, axis=1)[:, tf.newaxis, :, :]
        tm = tf.gather(target_mask, permutation, axis=1)[:, tf.newaxis, :, :]
        tw = tf.gather(target_weight, permutation, axis=1)[:, tf.newaxis, :]
        d_present = tm[..., 4] * te

        exist_bh = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.broadcast_to(te, tf.shape(pred_exist)), logits=pred_exist), axis=2)
        tt_full = tf.broadcast_to(tt, tf.shape(pred_type)[:-1])
        type_ce = tf.keras.losses.sparse_categorical_crossentropy(tt_full, pred_type, from_logits=True)
        type_factor = 1.0 + te
        type_bh = tf.reduce_sum(type_ce * type_factor, axis=2) / tf.maximum(tf.reduce_sum(type_factor, axis=2), 1.0)

        onehot = tf.one_hot(tt_full, schema.NUM_TYPES, dtype=tf.float32)
        selected_params = tf.reduce_sum(pred_params * onehot[..., tf.newaxis], axis=3)
        selected_std = tf.reduce_sum(pred_std * onehot[..., tf.newaxis], axis=3)
        param_nll = 0.5 * (tf.square(tp - selected_params) / tf.exp(2.0 * selected_std) + 2.0 * selected_std)
        param_factor = tm * te[..., tf.newaxis]
        param_bh = tf.reduce_sum(param_nll * param_factor, axis=[2, 3]) / tf.maximum(tf.reduce_sum(param_factor, axis=[2, 3]), 1.0)
        weight_bh = tf.reduce_mean(tf.square(pred_weight - tw), axis=2)
        d_ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.broadcast_to(d_present, tf.shape(pred_d)), logits=pred_d)
        d_bh = tf.reduce_sum(d_ce * te, axis=2) / tf.maximum(tf.reduce_sum(te, axis=2), 1.0)
        total_bh = weights.exist * exist_bh + weights.type * type_bh + weights.param * param_bh + weights.weight * weight_bh + weights.d_presence * d_bh
        exist_costs.append(exist_bh)
        type_costs.append(type_bh)
        param_costs.append(param_bh)
        weight_costs.append(weight_bh)
        d_presence_costs.append(d_bh)
        total_costs.append(total_bh)

    exist_costs = tf.stack(exist_costs)
    type_costs = tf.stack(type_costs)
    param_costs = tf.stack(param_costs)
    weight_costs = tf.stack(weight_costs)
    d_presence_costs = tf.stack(d_presence_costs)
    total_costs = tf.stack(total_costs)
    best_perm_idx = tf.argmin(total_costs, axis=0, output_type=tf.int32)  # [B,H]

    slot_bh = _gather_best_permutation(total_costs, best_perm_idx)
    exist_bh = _gather_best_permutation(exist_costs, best_perm_idx)
    type_bh = _gather_best_permutation(type_costs, best_perm_idx)
    param_bh = _gather_best_permutation(param_costs, best_perm_idx)
    weight_bh = _gather_best_permutation(weight_costs, best_perm_idx)
    d_presence_bh = _gather_best_permutation(d_presence_costs, best_perm_idx)

    best_permutations = tf.gather(PERMUTATIONS, best_perm_idx)  # [B,H,S]
    target_type_h = tf.tile(target_type[:, tf.newaxis, :], [1, hypotheses, 1])
    target_exist_h = tf.tile(target_exist[:, tf.newaxis, :], [1, hypotheses, 1])
    target_mask_h = tf.tile(target_mask[:, tf.newaxis, :, :], [1, hypotheses, 1, 1])
    matched_type = tf.gather(target_type_h, best_permutations, axis=2, batch_dims=2)
    matched_exist = tf.gather(target_exist_h, best_permutations, axis=2, batch_dims=2)
    matched_mask = tf.gather(target_mask_h, best_permutations, axis=2, batch_dims=2)
    matched_onehot = tf.one_hot(matched_type, schema.NUM_TYPES, dtype=tf.float32)
    matched_pred_params = tf.reduce_sum(pred_params * matched_onehot[..., tf.newaxis], axis=3)

    global_nll = 0.5 * (tf.square(target_global[:, tf.newaxis, :] - pred_global) / tf.exp(2.0 * pred_global_std) + 2.0 * pred_global_std)
    global_bh = tf.reduce_sum(global_nll * global_mask[:, tf.newaxis, :], axis=2) / tf.maximum(
        tf.reduce_sum(global_mask, axis=1, keepdims=True), 1.0
    )
    count_distribution = _component_count_distribution(pred_exist)
    target_count = tf.cast(tf.reduce_sum(target_exist, axis=1), tf.int32)
    target_count_h = tf.tile(target_count[:, tf.newaxis], [1, hypotheses])
    count_probability = tf.gather(count_distribution, target_count_h, axis=2, batch_dims=2)
    count_bh = -tf.math.log(tf.maximum(count_probability, 1e-8))

    pred_r = _denormalize_log_tensor(matched_pred_params[..., 0], "R")
    pred_spacing = _denormalize_log_tensor(matched_pred_params[..., 4], "D")
    pred_spacing_width = _denormalize_log_tensor(matched_pred_params[..., 5], "sigma_D")
    exclusion = 2.0 * pred_r
    active = tf.cast(matched_exist > 0.5, tf.float32)
    max_size = tf.reduce_max(tf.where(active > 0.0, exclusion, tf.zeros_like(exclusion)), axis=2)
    mean_size = tf.reduce_sum(exclusion * active, axis=2) / tf.maximum(tf.reduce_sum(active, axis=2), 1.0)
    d_rule = tf.cast(labels["d_spacing_rule"], tf.float32)[:, tf.newaxis, :]
    global_threshold = d_rule[..., schema.D_RULE_MAX] * max_size + d_rule[..., schema.D_RULE_MEAN] * mean_size
    threshold = global_threshold[..., tf.newaxis] + d_rule[..., schema.D_RULE_COMPONENT, tf.newaxis] * exclusion
    d_mask = matched_mask[..., 4] * active
    spacing_violation = tf.nn.relu(threshold * 1.001 - pred_spacing) / float(schema.PARAM_RANGES["D"].high)
    width_violation = tf.nn.relu(0.05 * pred_spacing - pred_spacing_width) / float(schema.PARAM_NORM_RANGES["sigma_D"].high)
    spacing_bh = tf.reduce_sum((tf.square(spacing_violation) + tf.square(width_violation)) * d_mask, axis=2) / tf.maximum(tf.reduce_sum(d_mask, axis=2), 1.0)

    supervised_bh = slot_bh + weights.global_ * global_bh + weights.spacing * spacing_bh + weights.count * count_bh
    best_h = tf.argmin(supervised_bh, axis=1, output_type=tf.int32)
    winner_loss = tf.reduce_mean(_gather_hypothesis(supervised_bh, best_h))

    confidence_loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(best_h, tf.cast(preds["hypothesis_logit"], tf.float32), from_logits=True))
    candidate_active_logit = tf.cast(preds["candidate_active_logit"], tf.float32)
    candidate_active_probability = tf.sigmoid(candidate_active_logit)
    winner_mask = tf.one_hot(best_h, tf.shape(supervised_bh)[1], dtype=tf.float32)
    winner_active_logit = _gather_hypothesis(candidate_active_logit, best_h)
    winner_activation_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(winner_active_logit), logits=winner_active_logit)
    )
    hard_usage = tf.reduce_mean(tf.one_hot(best_h, tf.shape(supervised_bh)[1], dtype=tf.float32), axis=0)
    # Unlike hard argmin counts, this assignment is differentiable and can
    # actively pull unused hypotheses toward regions where they can win.
    soft_assignment = tf.nn.softmax(-supervised_bh / 0.5, axis=1)
    usage = tf.reduce_mean(soft_assignment, axis=0)
    uniform_usage = 1.0 / tf.cast(tf.shape(supervised_bh)[1], tf.float32)
    usage_balance_loss = tf.cast(tf.shape(supervised_bh)[1], tf.float32) * tf.reduce_sum(tf.square(usage - uniform_usage))
    usage_entropy = -tf.reduce_sum(usage * tf.math.log(tf.maximum(usage, 1e-8)))

    type_probability = tf.nn.softmax(pred_type, axis=-1)
    expected_params = tf.reduce_sum(pred_params * type_probability[..., tf.newaxis], axis=3)
    candidate_vector = tf.concat(
        [tf.reshape(expected_params, [tf.shape(expected_params)[0], tf.shape(expected_params)[1], -1]), pred_weight, pred_global],
        axis=2,
    )
    delta = candidate_vector[:, :, tf.newaxis, :] - candidate_vector[:, tf.newaxis, :, :]
    distance = tf.sqrt(tf.reduce_mean(tf.square(delta), axis=-1) + 1e-8)
    h_float = tf.cast(tf.shape(candidate_vector)[1], tf.float32)
    off_diagonal = 1.0 - tf.eye(tf.shape(candidate_vector)[1], batch_shape=tf.shape(candidate_vector)[:1])
    # Only proposals that are independently active (plus the supervised
    # winner) participate in diversity.  Inactive capacity is free to remain
    # unused when a curve has fewer physical solutions than H_max.
    diversity_activity = tf.maximum(candidate_active_probability, winner_mask)
    diversity_pair_weight = diversity_activity[:, :, tf.newaxis] * diversity_activity[:, tf.newaxis, :] * off_diagonal
    diversity_penalty = tf.square(tf.nn.relu(float(weights.diversity_margin) - distance))
    diversity_loss = tf.reduce_sum(diversity_penalty * diversity_pair_weight) / tf.maximum(
        tf.reduce_sum(diversity_pair_weight), h_float
    )

    reconstruction_weight = tf.cast(weights.reconstruction, tf.float32)
    if all(key in labels for key in ("q", "I_clean", "point_mask")):
        def enabled_physics_losses():
            errors = multi_hypothesis_reconstruction_errors(
                labels["q"], labels["I_clean"], labels["point_mask"], pred_type, pred_exist, pred_params,
                preds["weight_logit"], pred_global, pred_d,
                preds.get(
                    "resolution_present_logit",
                    tf.ones(tf.shape(pred_global)[0], tf.float32) * 20.0,
                ),
                q_stride=weights.reconstruction_q_stride,
                max_samples_per_batch=weights.reconstruction_samples_per_batch,
                sampling_mode=weights.reconstruction_sampling_mode,
                multiscale_min_points=weights.reconstruction_multiscale_min_points,
                multiscale_max_points=weights.reconstruction_multiscale_max_points,
            )
            selected_n = tf.shape(errors)[0]
            active_probability = candidate_active_probability[:selected_n]
            selected_winner = winner_mask[:selected_n]
            # The labelled winner is never allowed to disappear. Other heads
            # retain a small exploration gradient, but cannot dominate the
            # loss merely because over-complete capacity exists.
            effective_activity = tf.maximum(active_probability, selected_winner)
            physics_weight = float(weights.inactive_physics_floor) + (
                1.0 - float(weights.inactive_physics_floor)
            ) * effective_activity
            valid_reconstruction = tf.reduce_sum(physics_weight * errors) / tf.maximum(
                tf.reduce_sum(physics_weight), 1.0
            )
            coverage_loss = tf.reduce_mean(tf.reduce_min(errors, axis=1))
            gated_reconstruction = coverage_loss + 0.5 * valid_reconstruction

            # There is no labelled number of inverse solutions.  Calibrate the
            # independent gate from stopped-gradient forward error instead:
            # an exact/near-exact alternative receives a target near one, a bad
            # proposal near zero, while the known winner is always positive.
            quality_target = tf.exp(
                -tf.stop_gradient(errors) / max(float(weights.activation_temperature), 1e-6)
            )
            quality_target = tf.maximum(quality_target, selected_winner)
            activation_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=quality_target,
                    logits=candidate_active_logit[:selected_n],
                )
            )
            nonwinner = 1.0 - selected_winner
            complexity_loss = tf.reduce_sum(active_probability * nonwinner) / tf.maximum(
                tf.reduce_sum(nonwinner), 1.0
            )
            selected_vectors = candidate_vector[:selected_n]
            selected_delta = selected_vectors[:, :, tf.newaxis, :] - selected_vectors[:, tf.newaxis, :, :]
            selected_distance = tf.sqrt(tf.reduce_mean(tf.square(selected_delta), axis=-1) + 1e-8)
            valid_pair_weight = (
                quality_target[:, :, tf.newaxis]
                * quality_target[:, tf.newaxis, :]
                * off_diagonal[:selected_n]
            )
            valid_diversity_penalty = tf.square(
                tf.nn.relu(float(weights.diversity_margin) - selected_distance)
            )
            physics_diversity = tf.reduce_sum(valid_diversity_penalty * valid_pair_weight) / tf.maximum(
                tf.reduce_sum(valid_pair_weight), h_float
            )
            expected_active = tf.reduce_mean(tf.reduce_sum(active_probability, axis=1))
            pseudo_active = tf.reduce_mean(tf.reduce_sum(tf.cast(quality_target >= 0.5, tf.float32), axis=1))
            return gated_reconstruction, coverage_loss, activation_loss, complexity_loss, physics_diversity, expected_active, pseudo_active

        def disabled_physics_losses():
            zero = tf.constant(0.0, tf.float32)
            expected = tf.reduce_mean(tf.reduce_sum(candidate_active_probability, axis=1))
            return zero, zero, zero, zero, zero, expected, zero

        reconstruction_loss, physics_coverage_loss, activation_quality_loss, candidate_complexity_loss, physics_diversity_loss, expected_active_candidates, pseudo_active_candidates = tf.cond(
            reconstruction_weight > 0.0,
            enabled_physics_losses,
            disabled_physics_losses,
        )
    else:
        reconstruction_loss = tf.constant(0.0, tf.float32)
        physics_coverage_loss = tf.constant(0.0, tf.float32)
        activation_quality_loss = tf.constant(0.0, tf.float32)
        candidate_complexity_loss = tf.constant(0.0, tf.float32)
        physics_diversity_loss = tf.constant(0.0, tf.float32)
        expected_active_candidates = tf.reduce_mean(tf.reduce_sum(candidate_active_probability, axis=1))
        pseudo_active_candidates = tf.constant(0.0, tf.float32)

    physics_enabled = tf.cast(reconstruction_weight > 0.0, tf.float32)
    effective_diversity_loss = (1.0 - physics_enabled) * diversity_loss + physics_enabled * physics_diversity_loss
    total_loss = (
        winner_loss
        + weights.quality * confidence_loss
        + weights.winner_activation * winner_activation_loss
        + weights.diversity * effective_diversity_loss
        + weights.usage_balance * usage_balance_loss
        + reconstruction_weight * reconstruction_loss
        + physics_enabled * weights.activation_quality * activation_quality_loss
        + physics_enabled * weights.candidate_complexity * candidate_complexity_loss
    )
    selected_matched_type = _gather_hypothesis(matched_type, best_h)
    selected_matched_exist = _gather_hypothesis(matched_exist, best_h)
    selected_pred_type = tf.argmax(_gather_hypothesis(pred_type, best_h), axis=-1, output_type=tf.int32)
    nonempty = tf.cast(selected_matched_exist > 0.5, tf.float32)
    type_correct = tf.cast(tf.equal(selected_pred_type, selected_matched_type), tf.float32)
    predicted_count = tf.argmax(_gather_hypothesis(count_distribution, best_h), axis=-1, output_type=tf.int32)

    return {
        "total_loss": total_loss,
        "exist_loss": _mean_selected(exist_bh, best_h),
        "type_loss": _mean_selected(type_bh, best_h),
        "param_loss": _mean_selected(param_bh, best_h),
        "weight_loss": _mean_selected(weight_bh, best_h),
        "global_loss": _mean_selected(global_bh, best_h),
        "quality_loss": confidence_loss,
        "d_presence_loss": _mean_selected(d_presence_bh, best_h),
        "spacing_loss": _mean_selected(spacing_bh, best_h),
        "reconstruction_loss": reconstruction_loss,
        "physics_coverage_loss": physics_coverage_loss,
        "count_loss": _mean_selected(count_bh, best_h),
        "diversity_loss": effective_diversity_loss,
        "raw_parameter_diversity_loss": diversity_loss,
        "physics_valid_diversity_loss": physics_diversity_loss,
        "hypothesis_usage_balance_loss": usage_balance_loss,
        "hypothesis_confidence_loss": confidence_loss,
        "winner_activation_loss": winner_activation_loss,
        "activation_quality_loss": activation_quality_loss,
        "candidate_complexity_loss": candidate_complexity_loss,
        "expected_active_candidates": expected_active_candidates,
        "pseudo_active_candidates": pseudo_active_candidates,
        "active_hypotheses": tf.reduce_sum(tf.cast(hard_usage > 0.0, tf.float32)),
        "hypothesis_usage_entropy": usage_entropy,
        "component_count_accuracy": tf.reduce_mean(tf.cast(tf.equal(predicted_count, target_count), tf.float32)),
        "slot_type_accuracy": tf.reduce_mean(type_correct),
        "nonempty_type_accuracy": tf.reduce_sum(type_correct * nonempty) / tf.maximum(tf.reduce_sum(nonempty), 1.0),
    }
