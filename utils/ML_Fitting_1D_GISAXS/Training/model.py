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


class BranchModeQuery(tf.keras.layers.Layer):
    """Independent learnable mode and slot queries for branch-conditioned decoding."""

    def __init__(self, num_hypotheses=16, max_slots=schema.MAX_SLOTS, dim=128, **kwargs):
        super().__init__(**kwargs)
        self.num_hypotheses = int(num_hypotheses)
        self.max_slots = int(max_slots)
        self.dim = int(dim)

    def build(self, input_shape):
        self.mode_query = self.add_weight(
            name="mode_query",
            shape=(self.num_hypotheses, self.dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        self.slot_query = self.add_weight(
            name="slot_query",
            shape=(self.max_slots, self.dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, batch_like):
        batch = tf.shape(batch_like)[0]
        query = self.mode_query[:, tf.newaxis, :] + self.slot_query[tf.newaxis, :, :]
        return tf.tile(query[tf.newaxis, :, :, :], [batch, 1, 1, 1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "num_hypotheses": self.num_hypotheses,
                "max_slots": self.max_slots,
                "dim": self.dim,
            }
        )
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


def branch_mode_decoder_block(
    q,
    z,
    point_mask,
    num_hypotheses,
    max_slots,
    dim=128,
    heads=4,
    key_dim=32,
    ffn_dim=256,
    name="branch_dec",
):
    """Decode each continuous mode independently while sharing curve memory."""
    q_flat = tf.keras.layers.Lambda(
        lambda t: tf.reshape(t, [-1, max_slots, dim]), name=f"{name}_flatten_modes"
    )(q)
    z_flat = tf.keras.layers.Lambda(
        lambda t: tf.reshape(
            tf.tile(t[:, tf.newaxis, :, :], [1, num_hypotheses, 1, 1]),
            [-1, tf.shape(t)[1], dim],
        ),
        name=f"{name}_tile_memory",
    )(z)
    mask_flat = tf.keras.layers.Lambda(
        lambda t: tf.reshape(
            tf.tile(t[:, tf.newaxis, :], [1, num_hypotheses, 1]),
            [-1, tf.shape(t)[1]],
        ),
        name=f"{name}_tile_mask",
    )(point_mask)
    decoded = decoder_block(
        q_flat,
        z_flat,
        mask_flat,
        dim=dim,
        heads=heads,
        key_dim=key_dim,
        ffn_dim=ffn_dim,
        name=name,
    )
    return tf.keras.layers.Lambda(
        lambda t: tf.reshape(t, [-1, num_hypotheses, max_slots, dim]),
        name=f"{name}_restore_modes",
    )(decoded)


def build_model(
    max_points=schema.MAX_POINTS,
    max_slots=schema.MAX_SLOTS,
    num_types=schema.NUM_TYPES,
    p_max=schema.P_MAX,
    g_max=schema.G_MAX,
    dim=128,
    encoder_blocks=4,
    decoder_blocks=2,
    num_hypotheses=16,
    include_resolution_presence_head=True,
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
        "d_allowed": tf.keras.Input(shape=(max_slots, 2), dtype=tf.float32, name="d_allowed"),
        "d_spacing_rule": tf.keras.Input(shape=(schema.NUM_D_RULES,), dtype=tf.float32, name="d_spacing_rule"),
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
            tf.keras.layers.Flatten()(inputs["d_allowed"]),
            inputs["d_spacing_rule"],
        ]
    )
    cons = tf.keras.layers.Dense(max_slots * dim, activation=tf.nn.gelu, name="constraint_dense1")(flat_cons)
    cons = tf.keras.layers.Dense(max_slots * dim, activation=tf.nn.gelu, name="constraint_dense2")(cons)
    cons = tf.keras.layers.Reshape((max_slots, dim), name="constraint_embedding")(cons)

    base_q = BranchModeQuery(
        num_hypotheses=num_hypotheses,
        max_slots=max_slots,
        dim=dim,
        name="branch_mode_query",
    )(inputs["x"])
    h_slot = tf.keras.layers.Dense(max_slots * dim, name="slot_h_dense")(h_proj)
    h_slot = tf.keras.layers.Reshape((max_slots, dim), name="slot_h_reshape")(h_slot)
    branch_token = tf.keras.layers.Dense(dim, activation=tf.nn.gelu, name="branch_token")(flat_cons)
    q = tf.keras.layers.Lambda(
        lambda xs: xs[0] + xs[1][:, tf.newaxis, :, :] + xs[2][:, tf.newaxis, :, :]
        + xs[3][:, tf.newaxis, tf.newaxis, :],
        name="conditioned_branch_mode_queries",
    )([base_q, cons, h_slot, branch_token])

    for i in range(decoder_blocks):
        q = branch_mode_decoder_block(
            q,
            z,
            inputs["point_mask"],
            num_hypotheses=num_hypotheses,
            max_slots=max_slots,
            dim=dim,
            name=f"decoder_{i}",
        )

    exist_logit_raw = tf.keras.layers.Dense(1, name="exist_logit_dense")(q)
    exist_logit_raw = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="exist_logit_raw")(exist_logit_raw)
    exist_logit = tf.keras.layers.Lambda(
        lambda xs: tf.where(
            xs[1][:, tf.newaxis, :] > 0.5,
            tf.ones_like(xs[0]) * FORCE_EXIST_LOGIT,
            tf.where(xs[1][:, tf.newaxis, :] > -0.5, tf.ones_like(xs[0]) * FORCE_EMPTY_LOGIT, xs[0]),
        ),
        name="exist_logit",
    )([exist_logit_raw, inputs["force_exist"]])

    type_logits_raw = tf.keras.layers.Dense(num_types, name="type_logits_raw_dense")(q)
    type_logits = tf.keras.layers.Lambda(lambda xs: xs[0] + (1.0 - xs[1][:, tf.newaxis, :, :]) * TYPE_MASK_LOGIT, name="type_logits")(
        [type_logits_raw, inputs["type_allowed"]]
    )

    param_raw = tf.keras.layers.Dense(num_types * p_max, name="param_mu_raw_dense")(q)
    param_raw = tf.keras.layers.Reshape((num_hypotheses, max_slots, num_types, p_max), name="param_mu_raw")(param_raw)
    param_low_eff = tf.keras.layers.Lambda(
        lambda xs: tf.where(xs[2] > 0.0, xs[0], tf.zeros_like(xs[0])),
        name="param_low_eff",
    )([inputs["param_low_norm"], inputs["param_high_norm"], inputs["param_range_mask"]])
    param_high_eff = tf.keras.layers.Lambda(
        lambda xs: tf.where(xs[2] > 0.0, xs[1], tf.ones_like(xs[1])),
        name="param_high_eff",
    )([inputs["param_low_norm"], inputs["param_high_norm"], inputs["param_range_mask"]])
    param_mu_norm = tf.keras.layers.Lambda(
        lambda xs: xs[1][:, tf.newaxis, :, :, :] + (xs[2] - xs[1])[:, tf.newaxis, :, :, :] * tf.sigmoid(xs[0]),
        name="param_mu_norm",
    )([param_raw, param_low_eff, param_high_eff])
    param_logstd_raw = tf.keras.layers.Dense(num_types * p_max, name="param_logstd_raw_dense")(q)
    param_logstd_raw = tf.keras.layers.Reshape((num_hypotheses, max_slots, num_types, p_max), name="param_logstd_reshape")(param_logstd_raw)
    param_logstd_raw = tf.keras.layers.Lambda(lambda t: tf.clip_by_value(t, -5.0, 1.0), name="param_logstd_raw")(param_logstd_raw)

    d_present_logit_raw = tf.keras.layers.Dense(1, name="d_present_logit_dense")(q)
    d_present_logit_raw = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="d_present_logit_raw")(d_present_logit_raw)
    d_present_logit = tf.keras.layers.Lambda(
        lambda xs: tf.where(
            xs[1][:, tf.newaxis, :, 0] < 0.5,
            tf.ones_like(xs[0]) * FORCE_EXIST_LOGIT,
            tf.where(xs[1][:, tf.newaxis, :, 1] < 0.5, tf.ones_like(xs[0]) * FORCE_EMPTY_LOGIT, xs[0]),
        ),
        name="d_present_logit",
    )([d_present_logit_raw, inputs["d_allowed"]])

    weight_logit = tf.keras.layers.Dense(1, name="weight_logit_dense")(q)
    weight_logit = tf.keras.layers.Lambda(lambda t: tf.squeeze(t, axis=-1), name="weight_logit")(weight_logit)

    mode_state = tf.keras.layers.Lambda(lambda t: tf.reduce_mean(t, axis=2), name="mode_state_pool")(q)
    mode_context = tf.keras.layers.Lambda(
        lambda xs: tf.concat(
            [xs[0], tf.tile(xs[1][:, tf.newaxis, :], [1, num_hypotheses, 1])], axis=-1
        ),
        name="mode_global_context",
    )([mode_state, h_proj])
    mode_context = tf.keras.layers.Dense(dim, activation=tf.nn.gelu, name="mode_global_projection")(mode_context)
    # BG/resolution are properties of the observed curve, not of the
    # particle-shape alternatives in the multi-solution sidecar. Decode them
    # once from shared curve/constraint context and tile only for legacy
    # inference/physics callers that still expect a hypothesis axis.
    shared_global_context = tf.keras.layers.Concatenate(name="shared_global_context")([h_proj, branch_token])
    shared_global_context = tf.keras.layers.Dense(dim, activation=tf.nn.gelu, name="shared_global_projection")(
        shared_global_context
    )
    g_raw_shared = tf.keras.layers.Dense(g_max, name="global_mu_raw_dense")(shared_global_context)
    global_low_eff = tf.keras.layers.Lambda(
        lambda xs: tf.where(xs[2] > 0.0, xs[0], tf.zeros_like(xs[0])),
        name="global_low_eff",
    )([inputs["global_low_norm"], inputs["global_high_norm"], inputs["global_range_mask"]])
    global_high_eff = tf.keras.layers.Lambda(
        lambda xs: tf.where(xs[2] > 0.0, xs[1], tf.ones_like(xs[1])),
        name="global_high_eff",
    )([inputs["global_low_norm"], inputs["global_high_norm"], inputs["global_range_mask"]])
    global_mu_norm_shared = tf.keras.layers.Lambda(
        lambda xs: xs[1] + (xs[2] - xs[1]) * tf.sigmoid(xs[0]),
        name="global_mu_norm_shared",
    )([g_raw_shared, global_low_eff, global_high_eff])
    global_logstd_raw_shared = tf.keras.layers.Dense(g_max, name="global_logstd_raw_dense")(shared_global_context)
    global_logstd_raw_shared = tf.keras.layers.Lambda(
        lambda t: tf.clip_by_value(t, -5.0, 1.0), name="global_logstd_raw_shared"
    )(global_logstd_raw_shared)
    if include_resolution_presence_head:
        resolution_present_logit = tf.keras.layers.Dense(
            1,
            bias_initializer=tf.keras.initializers.Constant(2.0),
            name="resolution_present_logit_dense",
        )(shared_global_context)
        resolution_present_logit = tf.keras.layers.Lambda(
            lambda t: tf.squeeze(t, axis=-1), name="resolution_present_logit"
        )(resolution_present_logit)
    global_mu_norm = tf.keras.layers.Lambda(
        lambda t: tf.tile(t[:, tf.newaxis, :], [1, num_hypotheses, 1]), name="global_mu_norm"
    )(global_mu_norm_shared)
    global_logstd_raw = tf.keras.layers.Lambda(
        lambda t: tf.tile(t[:, tf.newaxis, :], [1, num_hypotheses, 1]), name="global_logstd_raw"
    )(global_logstd_raw_shared)
    g_raw = tf.keras.layers.Lambda(
        lambda t: tf.tile(t[:, tf.newaxis, :], [1, num_hypotheses, 1]), name="global_mu_raw"
    )(g_raw_shared)
    candidate_active_logit = tf.keras.layers.Dense(
        1,
        bias_initializer=tf.keras.initializers.Constant(-0.5),
        name="candidate_active_logit_dense",
    )(mode_context)
    candidate_active_logit = tf.keras.layers.Lambda(
        lambda t: tf.squeeze(t, axis=-1), name="candidate_active_logit"
    )(candidate_active_logit)
    tier_raw = tf.keras.layers.Dense(3, name="tier_quality_raw_dense")(mode_context)
    tier_probability = tf.keras.layers.Lambda(
        lambda t: tf.stack(
            [
                tf.sigmoid(t[..., 2]) * tf.sigmoid(t[..., 1]) * tf.sigmoid(t[..., 0]),
                tf.sigmoid(t[..., 2]) * tf.sigmoid(t[..., 1]),
                tf.sigmoid(t[..., 2]),
            ],
            axis=-1,
        ),
        name="tier_probability",
    )(tier_raw)
    hypothesis_logit = tf.keras.layers.Lambda(
        lambda p: tf.math.log(tf.clip_by_value(p[..., 2], 1e-6, 1.0 - 1e-6))
        - tf.math.log1p(-tf.clip_by_value(p[..., 2], 1e-6, 1.0 - 1e-6)),
        name="hypothesis_logit",
    )(tier_probability)
    quality = hypothesis_logit

    outputs = {
        "exist_logit": exist_logit,
        "type_logits": type_logits,
        "param_mu_raw": param_raw,
        "param_mu_norm": param_mu_norm,
        "param_logstd_raw": param_logstd_raw,
        "d_present_logit": d_present_logit,
        "weight_logit": weight_logit,
        "global_mu_raw": g_raw,
        "global_mu_norm": global_mu_norm,
        "global_logstd_raw": global_logstd_raw,
        "global_mu_raw_shared": g_raw_shared,
        "global_mu_norm_shared": global_mu_norm_shared,
        "global_logstd_raw_shared": global_logstd_raw_shared,
        "quality": quality,
        "hypothesis_logit": hypothesis_logit,
        "candidate_active_logit": candidate_active_logit,
        "tier_probability": tier_probability,
    }
    if include_resolution_presence_head:
        outputs["resolution_present_logit"] = resolution_present_logit
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="ML1DGISAXSSlotModel")


def build_training_model(**kwargs):
    """Return a training view and its artifact-compatible inference view.

    The training view exposes the ordinal head's raw logits for stable BCE.
    Both models share every layer and variable.  Saving the inference view
    preserves the established `.keras` graph and old artifact load order.
    """
    inference_model = build_model(**kwargs)
    outputs = dict(inference_model.output)
    outputs["tier_raw"] = inference_model.get_layer("tier_quality_raw_dense").output
    training_model = tf.keras.Model(
        inputs=inference_model.inputs,
        outputs=outputs,
        name="ML1DGISAXSSlotTrainingModel",
    )
    return training_model, inference_model
