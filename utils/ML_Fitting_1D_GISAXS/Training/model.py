"""TensorFlow 2.15 Set Transformer-style slot model."""

from __future__ import annotations

import tensorflow as tf

from TrainSetBuild import schema

FORCE_EXIST_LOGIT = 30.0
FORCE_EMPTY_LOGIT = -30.0
TYPE_MASK_LOGIT = -1e4


class SlotQueryBase(tf.keras.layers.Layer):
    def __init__(self, max_slots=schema.MAX_SLOTS, dim=128, **kwargs):
        super().__init__(**kwargs)
        self.max_slots = max_slots
        self.dim = dim

    def build(self, input_shape):
        self.query = self.add_weight(
            name="slot_query_base",
            shape=(self.max_slots, self.dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, batch_like):
        batch = tf.shape(batch_like)[0]
        return tf.tile(self.query[tf.newaxis, :, :], [batch, 1, 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_slots": self.max_slots, "dim": self.dim})
        return cfg


def gelu_dense(x, units, name):
    x = tf.keras.layers.Dense(units, name=f"{name}_dense")(x)
    return tf.keras.layers.Activation(tf.nn.gelu, name=f"{name}_gelu")(x)


def transformer_block(x, mask, dim=128, heads=4, key_dim=32, ffn_dim=256, name="enc"):
    attn_mask = tf.keras.layers.Lambda(lambda m: tf.cast(m[:, tf.newaxis, :], tf.bool), name=f"{name}_attn_mask")(mask)
    a = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim, name=f"{name}_mha")(x, x, attention_mask=attn_mask)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x + a)
    f = gelu_dense(x, ffn_dim, f"{name}_ffn1")
    f = tf.keras.layers.Dense(dim, name=f"{name}_ffn2")(f)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x + f)


def decoder_block(q, z, point_mask, dim=128, heads=4, key_dim=32, ffn_dim=256, name="dec"):
    s = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim, name=f"{name}_self_mha")(q, q)
    q = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln_self")(q + s)
    cross_mask = tf.keras.layers.Lambda(lambda m: tf.cast(m[:, tf.newaxis, :], tf.bool), name=f"{name}_cross_mask")(point_mask)
    c = tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=key_dim, name=f"{name}_cross_mha")(q, z, attention_mask=cross_mask)
    q = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln_cross")(q + c)
    f = gelu_dense(q, ffn_dim, f"{name}_ffn1")
    f = tf.keras.layers.Dense(dim, name=f"{name}_ffn2")(f)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln_ffn")(q + f)


def build_model(
    max_points=schema.MAX_POINTS,
    max_slots=schema.MAX_SLOTS,
    num_types=schema.NUM_TYPES,
    p_max=schema.P_MAX,
    g_max=schema.G_MAX,
    dim=128,
    encoder_blocks=4,
    decoder_blocks=2,
):
    inputs = {
        "x": tf.keras.Input(shape=(max_points, 3), dtype=tf.float32, name="x"),
        "point_mask": tf.keras.Input(shape=(max_points,), dtype=tf.bool, name="point_mask"),
        "global_features": tf.keras.Input(shape=(5,), dtype=tf.float32, name="global_features"),
        "type_allowed": tf.keras.Input(shape=(max_slots, num_types), dtype=tf.float32, name="type_allowed"),
        "param_low_norm": tf.keras.Input(shape=(max_slots, num_types, p_max), dtype=tf.float32, name="param_low_norm"),
        "param_high_norm": tf.keras.Input(shape=(max_slots, num_types, p_max), dtype=tf.float32, name="param_high_norm"),
        "param_range_mask": tf.keras.Input(shape=(max_slots, num_types, p_max), dtype=tf.float32, name="param_range_mask"),
        "force_exist": tf.keras.Input(shape=(max_slots,), dtype=tf.float32, name="force_exist"),
        "global_low_norm": tf.keras.Input(shape=(g_max,), dtype=tf.float32, name="global_low_norm"),
        "global_high_norm": tf.keras.Input(shape=(g_max,), dtype=tf.float32, name="global_high_norm"),
        "global_range_mask": tf.keras.Input(shape=(g_max,), dtype=tf.float32, name="global_range_mask"),
    }

    z = tf.keras.layers.Dense(dim, name="point_dense1")(inputs["x"])
    z = tf.keras.layers.Activation(tf.nn.gelu, name="point_gelu1")(z)
    z = tf.keras.layers.Dense(dim, name="point_dense2")(z)
    for i in range(encoder_blocks):
        z = transformer_block(z, inputs["point_mask"], dim=dim, name=f"encoder_{i}")

    mask_f = tf.keras.layers.Lambda(lambda m: tf.cast(m, tf.float32)[:, :, tf.newaxis], name="mask_float")(inputs["point_mask"])
    z_masked = tf.keras.layers.Multiply(name="z_masked")([z, mask_f])
    h_sum = tf.keras.layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="h_sum")(z_masked)
    h_den = tf.keras.layers.Lambda(lambda m: tf.maximum(tf.reduce_sum(tf.cast(m, tf.float32), axis=1, keepdims=True), 1.0), name="h_den")(inputs["point_mask"])
    h = tf.keras.layers.Lambda(lambda xs: xs[0] / xs[1], name="masked_mean")([h_sum, h_den])
    h = tf.keras.layers.Concatenate(name="h_with_global_features")([h, inputs["global_features"]])
    h_proj = tf.keras.layers.Dense(dim, activation=tf.nn.gelu, name="global_h_projection")(h)

    flat_cons = tf.keras.layers.Concatenate(name="constraint_concat")(
        [
            tf.keras.layers.Flatten()(inputs["type_allowed"]),
            tf.keras.layers.Flatten()(inputs["param_low_norm"]),
            tf.keras.layers.Flatten()(inputs["param_high_norm"]),
            tf.keras.layers.Flatten()(inputs["param_range_mask"]),
            inputs["force_exist"],
            inputs["global_low_norm"],
            inputs["global_high_norm"],
            inputs["global_range_mask"],
        ]
    )
    cons = tf.keras.layers.Dense(max_slots * dim, activation=tf.nn.gelu, name="constraint_dense1")(flat_cons)
    cons = tf.keras.layers.Dense(max_slots * dim, activation=tf.nn.gelu, name="constraint_dense2")(cons)
    cons = tf.keras.layers.Reshape((max_slots, dim), name="constraint_embedding")(cons)

    base_q = SlotQueryBase(max_slots=max_slots, dim=dim, name="slot_query_base")(inputs["x"])
    h_slot = tf.keras.layers.Dense(max_slots * dim, name="slot_h_dense")(h_proj)
    h_slot = tf.keras.layers.Reshape((max_slots, dim), name="slot_h_reshape")(h_slot)
    q = tf.keras.layers.Add(name="conditioned_slot_queries")([base_q, cons, h_slot])

    for i in range(decoder_blocks):
        q = decoder_block(q, z, inputs["point_mask"], dim=dim, name=f"decoder_{i}")

    exist_logit = tf.keras.layers.Dense(1, name="exist_logit_dense")(q)
    exist_logit_raw = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="exist_logit_raw")(exist_logit)
    exist_logit = tf.keras.layers.Lambda(
        lambda xs: tf.where(
            xs[1] > 0.5,
            tf.ones_like(xs[0]) * FORCE_EXIST_LOGIT,
            tf.where(xs[1] > -0.5, tf.ones_like(xs[0]) * FORCE_EMPTY_LOGIT, xs[0]),
        ),
        name="exist_logit",
    )([exist_logit_raw, inputs["force_exist"]])

    type_logits_raw = tf.keras.layers.Dense(num_types, name="type_logits_raw")(q)
    type_logits = tf.keras.layers.Lambda(lambda xs: xs[0] + (1.0 - xs[1]) * TYPE_MASK_LOGIT, name="type_logits")(
        [type_logits_raw, inputs["type_allowed"]]
    )

    param_raw = tf.keras.layers.Dense(num_types * p_max, name="param_mu_raw_dense")(q)
    param_raw = tf.keras.layers.Reshape((max_slots, num_types, p_max), name="param_mu_raw")(param_raw)
    param_low_eff = tf.keras.layers.Lambda(
        lambda xs: tf.where(xs[2] > 0.0, xs[0], tf.zeros_like(xs[0])),
        name="param_low_eff",
    )([inputs["param_low_norm"], inputs["param_high_norm"], inputs["param_range_mask"]])
    param_high_eff = tf.keras.layers.Lambda(
        lambda xs: tf.where(xs[2] > 0.0, xs[1], tf.ones_like(xs[1])),
        name="param_high_eff",
    )([inputs["param_low_norm"], inputs["param_high_norm"], inputs["param_range_mask"]])
    param_mu_norm = tf.keras.layers.Lambda(
        lambda xs: xs[1] + (xs[2] - xs[1]) * tf.sigmoid(xs[0]),
        name="param_mu_norm",
    )([param_raw, param_low_eff, param_high_eff])
    param_logstd_raw = tf.keras.layers.Dense(num_types * p_max, name="param_logstd_raw_dense")(q)
    param_logstd_raw = tf.keras.layers.Reshape((max_slots, num_types, p_max), name="param_logstd_raw_reshape")(param_logstd_raw)
    param_logstd_raw = tf.keras.layers.Lambda(lambda t: tf.clip_by_value(t, -5.0, 1.0), name="param_logstd_raw")(param_logstd_raw)

    weight_logit = tf.keras.layers.Dense(1, name="weight_logit_dense")(q)
    weight_logit = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="weight_logit")(weight_logit)

    g_raw = tf.keras.layers.Dense(g_max, name="global_mu_raw")(h_proj)
    global_low_eff = tf.keras.layers.Lambda(
        lambda xs: tf.where(xs[2] > 0.0, xs[0], tf.zeros_like(xs[0])),
        name="global_low_eff",
    )([inputs["global_low_norm"], inputs["global_high_norm"], inputs["global_range_mask"]])
    global_high_eff = tf.keras.layers.Lambda(
        lambda xs: tf.where(xs[2] > 0.0, xs[1], tf.ones_like(xs[1])),
        name="global_high_eff",
    )([inputs["global_low_norm"], inputs["global_high_norm"], inputs["global_range_mask"]])
    global_mu_norm = tf.keras.layers.Lambda(
        lambda xs: xs[1] + (xs[2] - xs[1]) * tf.sigmoid(xs[0]),
        name="global_mu_norm",
    )([g_raw, global_low_eff, global_high_eff])
    global_logstd_raw = tf.keras.layers.Dense(g_max, name="global_logstd_raw_dense")(h_proj)
    global_logstd_raw = tf.keras.layers.Lambda(lambda t: tf.clip_by_value(t, -5.0, 1.0), name="global_logstd_raw")(global_logstd_raw)
    quality = tf.keras.layers.Dense(1, name="quality")(h_proj)

    outputs = {
        "exist_logit": exist_logit,
        "type_logits": type_logits,
        "param_mu_raw": param_raw,
        "param_mu_norm": param_mu_norm,
        "param_logstd_raw": param_logstd_raw,
        "weight_logit": weight_logit,
        "global_mu_raw": g_raw,
        "global_mu_norm": global_mu_norm,
        "global_logstd_raw": global_logstd_raw,
        "quality": quality,
    }
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ML1DGISAXSSlotModel")
