"""Low-cost differentiable approximation of the NumPy GISAXS forward model."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from TrainSetBuild import schema


def _denorm(x, spec):
    x = tf.clip_by_value(tf.cast(x, tf.float32), 0.0, 1.0)
    if spec.transform == "log":
        return tf.exp(tf.math.log(float(spec.low)) + x * (tf.math.log(float(spec.high)) - tf.math.log(float(spec.low))))
    return float(spec.low) + x * float(spec.high - spec.low)


def denormalize_component_params(params_norm):
    return tf.stack(
        [_denorm(params_norm[..., i], schema.PARAM_NORM_RANGES[name]) for i, name in enumerate(schema.PARAM_NAMES)],
        axis=-1,
    )


def denormalize_global_params(global_norm):
    return tf.stack(
        [_denorm(global_norm[..., i], schema.GLOBAL_NORM_RANGES[name]) for i, name in enumerate(schema.GLOBAL_PARAM_NAMES)],
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
    safe_q = tf.maximum(q[:, :, tf.newaxis, :], 1e-12)
    form = tf.square(radii[:, :, :, tf.newaxis] * tf.math.special.bessel_j1(x) / safe_q)
    return tf.reduce_sum(form * weights[:, :, :, tf.newaxis], axis=2) * 1e-6


def random_cylinder_form_factor(q, r, sigma_r, h, sigma_h):
    radii, wr = _gaussian_nodes(r, sigma_r, n=13, nsig=4.0)
    heights, wh = _gaussian_nodes(h, sigma_h, n=13, nsig=4.0)
    alpha = tf.linspace(0.0, np.pi / 2.0, 24)
    wa = tf.sin(alpha)
    wa /= tf.reduce_sum(wa)
    q6 = q[:, :, tf.newaxis, tf.newaxis, tf.newaxis, :]
    r6 = radii[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    h6 = heights[:, :, tf.newaxis, :, tf.newaxis, tf.newaxis]
    sa = tf.sin(alpha)[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    ca = tf.cos(alpha)[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    amplitude = _radial_cylinder_amplitude(q6 * r6 * sa) * _sinc(q6 * h6 * ca / 2.0)
    weights = (
        wr[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
        * wh[:, :, tf.newaxis, :, tf.newaxis, tf.newaxis]
        * wa[tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis, :, tf.newaxis]
    )
    return tf.reduce_sum(tf.square(amplitude) * weights, axis=[2, 3, 4])


def structure_factor(q, d, sigma_d, d_probability):
    phi = tf.exp(-np.pi * tf.square(q) * tf.square(sigma_d[:, :, tf.newaxis]))
    numerator = tf.maximum(1.0 - tf.square(phi), 1e-8)
    denominator = 1.0 + tf.square(phi) - 2.0 * phi * tf.cos(q * d[:, :, tf.newaxis])
    structured = numerator / tf.maximum(denominator, 1e-8)
    return 1.0 + d_probability[:, :, tf.newaxis] * (structured - 1.0)


def reconstruct_intensity(q, matched_type, matched_exist, params_norm, weight_logits, global_norm, d_present_logits):
    params = denormalize_component_params(params_norm)
    r, sigma_r, h, sigma_h, d, sigma_d = tf.unstack(params, axis=-1)
    q_slots = q[:, tf.newaxis, :]
    p_sphere = sphere_form_factor(q_slots, r, sigma_r)
    p_cylinder = random_cylinder_form_factor(q_slots, r, sigma_r, h, sigma_h)
    p_vertical = vertical_cylinder_form_factor(q_slots, r, sigma_r)
    type_onehot = tf.one_hot(tf.cast(matched_type, tf.int32), schema.NUM_TYPES, dtype=tf.float32)
    form = (
        type_onehot[:, :, schema.TYPE_SPHERE, tf.newaxis] * p_sphere
        + type_onehot[:, :, schema.TYPE_CYLINDER, tf.newaxis] * p_cylinder
        + type_onehot[:, :, schema.TYPE_VERTICAL_CYLINDER, tf.newaxis] * p_vertical
    )
    d_probability = tf.sigmoid(tf.cast(d_present_logits, tf.float32))
    form *= structure_factor(q_slots, d, sigma_d, d_probability)
    active = tf.cast(matched_exist > 0.5, tf.float32)
    masked_logits = tf.cast(weight_logits, tf.float32) + (1.0 - active) * -1e4
    weights = tf.nn.softmax(masked_logits, axis=-1) * active
    mixture = tf.reduce_sum(weights[:, :, tf.newaxis] * form, axis=1)

    global_phys = denormalize_global_params(global_norm)
    bg, sigma_res, nu_res, int_res, scale = tf.unstack(global_phys, axis=-1)
    resolution = int_res[:, tf.newaxis] / (
        1.0 + tf.pow(tf.abs(q) / tf.maximum(sigma_res[:, tf.newaxis], 1e-12), nu_res[:, tf.newaxis])
    )
    return tf.maximum(bg[:, tf.newaxis] + scale[:, tf.newaxis] * (mixture + resolution), 1e-30)


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
        q, matched_type, matched_exist, params_norm, weight_logits, global_norm, d_present_logits
    )
    residual = tf.math.log(prediction) - tf.math.log(tf.maximum(target, 1e-30))
    abs_residual = tf.abs(residual)
    delta = float(huber_delta)
    huber = tf.where(abs_residual <= delta, 0.5 * tf.square(residual), delta * (abs_residual - 0.5 * delta))
    return tf.reduce_sum(huber * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
