"""Differentiable V5 forward model, numerically aligned with the NumPy exact model."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from TrainSetBuild import schema


_V5_GLOBAL_NORM_VERSION = schema.V5_GLOBAL_NORM_VERSION


def configure_v5_global_norm_version(version):
    """Select the dataset-level V5 nuisance normalization before tracing."""
    global _V5_GLOBAL_NORM_VERSION
    schema.v5_global_norm_ranges(version)  # validate eagerly
    _V5_GLOBAL_NORM_VERSION = str(version)


def _denorm(x, spec):
    x = tf.clip_by_value(tf.cast(x, tf.float32), 0.0, 1.0)
    if spec.transform == "log":
        return tf.exp(tf.math.log(float(spec.low)) + x * (tf.math.log(float(spec.high)) - tf.math.log(float(spec.low))))
    return float(spec.low) + x * float(spec.high - spec.low)


def denormalize_component_params(params_norm):
    return tf.stack(
        [_denorm(params_norm[..., i], schema.V5_PARAM_NORM_RANGES[name]) for i, name in enumerate(schema.PARAM_NAMES)],
        axis=-1,
    )


def denormalize_global_params(global_norm):
    ranges = schema.v5_global_norm_ranges(_V5_GLOBAL_NORM_VERSION)
    return tf.stack(
        [_denorm(global_norm[..., i], ranges[name]) for i, name in enumerate(schema.V5_GLOBAL_TARGET_NAMES)],
        axis=-1,
    )


def _sphere_amplitude(x):
    x2 = tf.square(x)
    series = 1.0 - x2 / 10.0 + tf.square(x2) / 280.0
    safe_x = tf.where(tf.abs(x) < 1e-4, tf.ones_like(x), x)
    regular = 3.0 * (tf.sin(safe_x) - safe_x * tf.cos(safe_x)) / tf.pow(safe_x, 3)
    return tf.where(tf.abs(x) < 1e-4, series, regular)


def _radial_cylinder_amplitude(x):
    x2 = tf.square(x)
    series = 1.0 - x2 / 8.0 + tf.square(x2) / 192.0
    safe_x = tf.where(tf.abs(x) < 1e-4, tf.ones_like(x), x)
    regular = 2.0 * tf.math.special.bessel_j1(safe_x) / safe_x
    return tf.where(tf.abs(x) < 1e-4, series, regular)


def _sinc(x):
    x2 = tf.square(x)
    series = 1.0 - x2 / 6.0 + tf.square(x2) / 120.0
    safe_x = tf.where(tf.abs(x) < 1e-4, tf.ones_like(x), x)
    return tf.where(tf.abs(x) < 1e-4, series, tf.sin(safe_x) / safe_x)


def _gaussian_nodes(mu, sigma, n, nsig):
    fraction = tf.linspace(0.0, 1.0, int(n))
    low = tf.maximum(mu - float(nsig) * sigma, 0.0)
    high = mu + float(nsig) * sigma
    nodes = low[..., tf.newaxis] + (high - low)[..., tf.newaxis] * fraction
    safe_sigma = tf.maximum(sigma[..., tf.newaxis], 1e-12)
    weights = tf.exp(-0.5 * tf.square((nodes - mu[..., tf.newaxis]) / safe_sigma))
    weights /= tf.maximum(tf.reduce_sum(weights, axis=-1, keepdims=True), 1e-30)
    nodes = tf.maximum(nodes, 1e-8)
    return nodes, weights


def sphere_form_factor(q, r, sigma_r):
    radii, weights = _gaussian_nodes(r, sigma_r, n=25, nsig=4.0)
    x = q[:, :, tf.newaxis, :] * radii[:, :, :, tf.newaxis]
    return tf.reduce_sum(tf.square(_sphere_amplitude(x)) * weights[:, :, :, tf.newaxis], axis=2)


def vertical_cylinder_form_factor(q, r, sigma_r_fraction):
    radii, weights = _gaussian_nodes(r, r * sigma_r_fraction, n=26, nsig=3.0)
    x = q[:, :, tf.newaxis, :] * radii[:, :, :, tf.newaxis]
    form = tf.square(_radial_cylinder_amplitude(x))
    return tf.reduce_sum(form * weights[:, :, :, tf.newaxis], axis=2)


def random_cylinder_form_factor(q, r, sigma_r, h, sigma_h):
    radii, wr = _gaussian_nodes(r, sigma_r, n=13, nsig=4.0)
    heights, wh = _gaussian_nodes(h, sigma_h, n=13, nsig=4.0)
    alpha = tf.linspace(0.0, np.pi / 2.0, 24)
    wa = tf.sin(alpha)
    wa /= tf.reduce_sum(wa)
    # R and h are independent, so <Fr(R)^2 Fh(h)^2> factorizes into
    # <Fr(R)^2> * <Fh(h)^2> for every orientation.  This is mathematically
    # identical to the old R x h outer product but avoids its huge 6-D tensor.
    q5 = q[:, :, tf.newaxis, tf.newaxis, :]
    sa = tf.sin(alpha)[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    ca = tf.cos(alpha)[tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    radial = _radial_cylinder_amplitude(
        q5 * radii[:, :, :, tf.newaxis, tf.newaxis] * sa
    )
    axial = _sinc(
        q5 * heights[:, :, :, tf.newaxis, tf.newaxis] * ca / 2.0
    )
    radial_mean = tf.reduce_sum(
        tf.square(radial) * wr[:, :, :, tf.newaxis, tf.newaxis], axis=2
    )
    axial_mean = tf.reduce_sum(
        tf.square(axial) * wh[:, :, :, tf.newaxis, tf.newaxis], axis=2
    )
    return tf.reduce_sum(
        radial_mean * axial_mean * wa[tf.newaxis, tf.newaxis, :, tf.newaxis], axis=2
    )


def _multiscale_point_indices(target, point_mask, min_points, max_points):
    """Per-curve random/peak/high-q indices for the final physics curriculum."""
    n_total = tf.shape(target)[1]
    min_points = tf.maximum(tf.cast(min_points, tf.int32), 1)
    max_points = tf.maximum(tf.cast(max_points, tf.int32), min_points)
    max_points = tf.minimum(max_points, n_total)
    budget = tf.random.uniform([], minval=min_points, maxval=max_points + 1, dtype=tf.int32)
    log_target = tf.math.log(tf.maximum(tf.cast(target, tf.float32), 1e-30))
    first = log_target[:, 1:] - log_target[:, :-1]
    curvature = tf.pad(tf.abs(first[:, 1:] - first[:, :-1]), [[0, 0], [1, 1]])
    curvature /= tf.maximum(tf.reduce_max(curvature, axis=1, keepdims=True), 1e-6)
    valid_count = tf.maximum(tf.reduce_sum(tf.cast(point_mask, tf.int32), axis=1), 1)
    positions = tf.range(n_total, dtype=tf.float32)[tf.newaxis, :]
    relative_q = positions / tf.cast(valid_count[:, tf.newaxis], tf.float32)
    high_q = tf.clip_by_value((relative_q - 0.55) / 0.45, 0.0, 1.0)
    random_score = tf.random.uniform(tf.shape(log_target), dtype=tf.float32)
    score = 0.35 * random_score + 0.35 * curvature + 0.30 * high_q
    score = tf.where(point_mask, score, tf.ones_like(score) * -1e9)
    ordered = tf.argsort(score, axis=1, direction="DESCENDING")
    selected = tf.gather(ordered, tf.range(budget), axis=1)
    return tf.sort(selected, axis=1)


def structure_factor(q, d, sigma_d, d_probability):
    log_phi = -np.pi * tf.square(q) * tf.square(sigma_d[:, :, tf.newaxis])
    phi = tf.exp(log_phi)
    # Algebraically identical to the NumPy expression, but avoids catastrophic
    # float32 cancellation when q is small and phi/cos(qD) are both near one.
    one_minus_phi = -tf.math.expm1(log_phi)
    numerator = -tf.math.expm1(2.0 * log_phi)
    denominator = tf.square(one_minus_phi) + 4.0 * phi * tf.square(
        tf.sin(0.5 * q * d[:, :, tf.newaxis])
    )
    structured = numerator / tf.maximum(denominator, 1e-15)
    return 1.0 + d_probability[:, :, tf.newaxis] * (structured - 1.0)


def _masked_median(values, mask):
    huge = tf.ones_like(values) * tf.float32.max
    ordered = tf.sort(tf.where(mask, values, huge), axis=1)
    count = tf.maximum(tf.reduce_sum(tf.cast(mask, tf.int32), axis=1), 1)
    lo = tf.maximum((count - 1) // 2, 0)
    hi = count // 2
    return 0.5 * (
        tf.gather(ordered, lo, axis=1, batch_dims=1)
        + tf.gather(ordered, hi, axis=1, batch_dims=1)
    )


def _v5_add_resolution_background(q, particle, global_norm, resolution_present, point_mask=None):
    rho_bg, sigma_res, nu_res, rho_res, _unused = tf.unstack(denormalize_global_params(global_norm), axis=-1)
    if point_mask is None:
        point_mask = q > 0.0
    else:
        point_mask = tf.cast(point_mask, tf.bool)
    particle_median = _masked_median(particle, point_mask)
    background = rho_bg * particle_median
    g = 1.0 / (1.0 + tf.pow(tf.maximum(q, 0.0) / tf.maximum(sigma_res[:, tf.newaxis], 1e-12), nu_res[:, tf.newaxis]))
    # V5 generation defines Q_low as the first five valid samples.  Native
    # records always store valid q contiguously from index zero.
    particle_low_max = tf.reduce_max(particle[:, :5], axis=1)
    g_low_max = tf.reduce_max(g[:, :5], axis=1)
    int_res = rho_res * particle_low_max / tf.maximum(g_low_max, 1e-30)
    presence = tf.clip_by_value(tf.cast(resolution_present, tf.float32), 0.0, 1.0)
    resolution = presence[:, tf.newaxis] * int_res[:, tf.newaxis] * g
    return tf.maximum(particle + resolution + background[:, tf.newaxis], 1e-30)


def reconstruct_intensity(q, matched_type, matched_exist, params_norm, weight_logits, global_norm, d_present_logits, resolution_present, point_mask=None):
    params = denormalize_component_params(params_norm)
    r, sigma_r, h, sigma_h, d, sigma_d = tf.unstack(params, axis=-1)
    q_slots = q[:, tf.newaxis, :]
    p_sphere = sphere_form_factor(q_slots, r, r * sigma_r)
    p_cylinder = random_cylinder_form_factor(q_slots, r, r * sigma_r, h, h * sigma_h)
    p_vertical = vertical_cylinder_form_factor(q_slots, r, sigma_r)
    type_onehot = tf.one_hot(tf.cast(matched_type, tf.int32), schema.NUM_TYPES, dtype=tf.float32)
    form = (
        type_onehot[:, :, schema.TYPE_SPHERE, tf.newaxis] * p_sphere
        + type_onehot[:, :, schema.TYPE_CYLINDER, tf.newaxis] * p_cylinder
        + type_onehot[:, :, schema.TYPE_VERTICAL_CYLINDER, tf.newaxis] * p_vertical
    )
    d_probability = tf.sigmoid(tf.cast(d_present_logits, tf.float32))
    form *= structure_factor(q_slots, d, d * sigma_d, d_probability)
    active = tf.cast(matched_exist > 0.5, tf.float32)
    masked_logits = tf.cast(weight_logits, tf.float32) + (1.0 - active) * -1e4
    weights = tf.nn.softmax(masked_logits, axis=-1) * active
    mixture = tf.reduce_sum(weights[:, :, tf.newaxis] * form, axis=1)

    return _v5_add_resolution_background(q, mixture, global_norm, resolution_present, point_mask)


def reconstruct_intensity_soft(q, type_logits, exist_logits, params_norm_by_type, weight_logits, global_norm, d_present_logits, resolution_present, point_mask=None):
    """Differentiable forward model driven entirely by one predicted candidate."""
    type_probability = tf.nn.softmax(tf.cast(type_logits, tf.float32), axis=-1)
    exist_probability = tf.sigmoid(tf.cast(exist_logits, tf.float32))
    q_slots = q[:, tf.newaxis, :]

    def typed_form(type_id):
        params = denormalize_component_params(params_norm_by_type[:, :, type_id, :])
        r, sigma_r, h, sigma_h, d, sigma_d = tf.unstack(params, axis=-1)
        if type_id == schema.TYPE_SPHERE:
            form = sphere_form_factor(q_slots, r, r * sigma_r)
        elif type_id == schema.TYPE_CYLINDER:
            form = random_cylinder_form_factor(q_slots, r, r * sigma_r, h, h * sigma_h)
        else:
            form = vertical_cylinder_form_factor(q_slots, r, sigma_r)
        d_probability = tf.sigmoid(tf.cast(d_present_logits, tf.float32))
        return form * structure_factor(q_slots, d, d * sigma_d, d_probability)

    form = tf.zeros_like(q_slots, dtype=tf.float32)
    for type_id in (schema.TYPE_SPHERE, schema.TYPE_CYLINDER, schema.TYPE_VERTICAL_CYLINDER):
        form += type_probability[:, :, type_id, tf.newaxis] * typed_form(type_id)

    masked_weight_logits = tf.cast(weight_logits, tf.float32) + tf.math.log(tf.maximum(exist_probability, 1e-6))
    component_weights = tf.nn.softmax(masked_weight_logits, axis=-1)
    mixture = tf.reduce_sum(component_weights[:, :, tf.newaxis] * form, axis=1)
    return _v5_add_resolution_background(q, mixture, global_norm, resolution_present, point_mask)


def multi_hypothesis_reconstruction_errors(
    q,
    target_intensity,
    point_mask,
    type_logits,
    exist_logits,
    params_norm_by_type,
    weight_logits,
    global_norm,
    d_present_logits,
    resolution_present_logit,
    q_stride=32,
    huber_delta=0.30,
    max_samples_per_batch=1,
    sampling_mode=0,
    multiscale_min_points=64,
    multiscale_max_points=128,
):
    """Return a physical reconstruction error for each selected proposal.

    The result has shape ``[selected_batch, hypotheses]``.  Keeping this
    dimension lets the caller disable proposals that do not describe a curve,
    instead of forcing every over-complete proposal slot to fit it.
    """
    selected_n = tf.minimum(
        tf.shape(q)[0], tf.maximum(tf.cast(max_samples_per_batch, tf.int32), 1)
    )
    selected = tf.range(selected_n)
    q = tf.cast(tf.gather(q, selected), tf.float32)
    target = tf.cast(tf.gather(target_intensity, selected), tf.float32)
    point_mask_selected = tf.gather(point_mask, selected)

    def stride_sample():
        stride = tf.maximum(tf.cast(q_stride, tf.int32), 1)
        indices = tf.range(0, tf.shape(q)[1], delta=stride)
        return (
            tf.gather(q, indices, axis=1),
            tf.gather(target, indices, axis=1),
            tf.cast(tf.gather(point_mask_selected, indices, axis=1), tf.float32),
        )

    def multiscale_sample():
        indices = _multiscale_point_indices(
            target,
            point_mask_selected,
            multiscale_min_points,
            multiscale_max_points,
        )
        return (
            tf.gather(q, indices, axis=1, batch_dims=1),
            tf.gather(target, indices, axis=1, batch_dims=1),
            tf.cast(tf.gather(point_mask_selected, indices, axis=1, batch_dims=1), tf.float32),
        )

    q, target, mask = tf.cond(
        tf.cast(sampling_mode, tf.int32) > 0, multiscale_sample, stride_sample
    )
    type_logits = tf.gather(type_logits, selected)
    exist_logits = tf.gather(exist_logits, selected)
    params_norm_by_type = tf.gather(params_norm_by_type, selected)
    weight_logits = tf.gather(weight_logits, selected)
    global_norm = tf.gather(global_norm, selected)
    d_present_logits = tf.gather(d_present_logits, selected)
    resolution_present_logit = tf.gather(resolution_present_logit, selected)
    hypotheses = tf.shape(type_logits)[1]
    points = tf.shape(q)[1]

    q_flat = tf.reshape(tf.tile(q[:, tf.newaxis, :], [1, hypotheses, 1]), [-1, points])
    target_flat = tf.reshape(tf.tile(target[:, tf.newaxis, :], [1, hypotheses, 1]), [-1, points])
    mask_flat = tf.reshape(tf.tile(mask[:, tf.newaxis, :], [1, hypotheses, 1]), [-1, points])
    prediction = reconstruct_intensity_soft(
        q_flat,
        tf.reshape(type_logits, [-1, schema.MAX_SLOTS, schema.NUM_TYPES]),
        tf.reshape(exist_logits, [-1, schema.MAX_SLOTS]),
        tf.reshape(params_norm_by_type, [-1, schema.MAX_SLOTS, schema.NUM_TYPES, schema.P_MAX]),
        tf.reshape(weight_logits, [-1, schema.MAX_SLOTS]),
        tf.reshape(
            tf.tile(global_norm[:, tf.newaxis, :], [1, hypotheses, 1]),
            [-1, schema.G_MAX],
        ),
        tf.reshape(d_present_logits, [-1, schema.MAX_SLOTS]),
        tf.reshape(tf.tile(tf.sigmoid(resolution_present_logit)[:, tf.newaxis], [1, hypotheses]), [-1]),
        point_mask=tf.cast(mask_flat > 0.5, tf.bool),
    )
    residual = tf.math.log(prediction) - tf.math.log(tf.maximum(target_flat, 1e-30))
    per_candidate = tf.sqrt(
        tf.reduce_sum(tf.square(residual) * mask_flat, axis=1)
        / tf.maximum(tf.reduce_sum(mask_flat, axis=1), 1.0)
        + 1e-12
    )
    return tf.reshape(per_candidate, [selected_n, hypotheses])


def multi_hypothesis_reconstruction_loss(
    q,
    target_intensity,
    point_mask,
    type_logits,
    exist_logits,
    params_norm_by_type,
    weight_logits,
    global_norm,
    d_present_logits,
    resolution_present_logit,
    q_stride=32,
    huber_delta=0.30,
    max_samples_per_batch=1,
):
    """Backward-compatible mean loss across all direct hypotheses."""
    errors = multi_hypothesis_reconstruction_errors(
        q,
        target_intensity,
        point_mask,
        type_logits,
        exist_logits,
        params_norm_by_type,
        weight_logits,
        global_norm,
        d_present_logits,
        resolution_present_logit,
        q_stride=q_stride,
        huber_delta=huber_delta,
        max_samples_per_batch=max_samples_per_batch,
    )
    return tf.reduce_mean(errors)


def reconstruction_log_huber_loss(
    q,
    target_intensity,
    point_mask,
    matched_type,
    matched_exist,
    params_norm,
    weight_logits,
    global_norm,
    d_present_logits,
    resolution_present_logit,
    q_stride=16,
    huber_delta=0.30,
    max_samples_per_batch=2,
):
    batch_n = tf.shape(q)[0]
    selected_n = tf.minimum(batch_n, max(int(max_samples_per_batch), 1))
    selected = tf.range(selected_n)
    q = tf.gather(q, selected)
    target_intensity = tf.gather(target_intensity, selected)
    point_mask = tf.gather(point_mask, selected)
    matched_type = tf.gather(matched_type, selected)
    matched_exist = tf.gather(matched_exist, selected)
    params_norm = tf.gather(params_norm, selected)
    weight_logits = tf.gather(weight_logits, selected)
    global_norm = tf.gather(global_norm, selected)
    d_present_logits = tf.gather(d_present_logits, selected)
    stride = max(int(q_stride), 1)
    q = tf.cast(q[:, ::stride], tf.float32)
    target = tf.cast(target_intensity[:, ::stride], tf.float32)
    mask = tf.cast(point_mask[:, ::stride], tf.float32)
    prediction = reconstruct_intensity(
        q, matched_type, matched_exist, params_norm, weight_logits, global_norm, d_present_logits,
        tf.sigmoid(tf.gather(resolution_present_logit, selected)), point_mask=point_mask,
    )
    residual = tf.math.log(prediction) - tf.math.log(tf.maximum(target, 1e-30))
    abs_residual = tf.abs(residual)
    delta = float(huber_delta)
    huber = tf.where(abs_residual <= delta, 0.5 * tf.square(residual), delta * (abs_residual - 0.5 * delta))
    return tf.reduce_sum(huber * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
