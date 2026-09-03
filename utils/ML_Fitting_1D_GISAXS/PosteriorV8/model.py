"""TensorFlow 2.15 proposal network for the Posterior V8 hybrid solver.

The network predicts discrete branches from the curve alone.  Its continuous
head is separately conditioned on one *hard* topology/D/resolution branch and
returns a mixture of diagonal logistic-normal proposals in the branch codec's
unit cube.  Exact forward physics and refinement deliberately live elsewhere.
"""

from __future__ import annotations

from numbers import Integral
from typing import Mapping, Sequence

import tensorflow as tf

from .branch_catalog import (
    BRANCH_PATTERN_COUNT,
    TOPOLOGY_SLOT_MASK,
    VALID_BRANCH_PATTERN_MASK,
)
from .contract import MAX_COMPONENTS, NUM_TOPOLOGIES


MODEL_VERSION = "posterior_v8_masked_conv_logistic_normal_v1"
DEFAULT_MAX_POINTS = 1000
POINT_FEATURE_DIM = 3
GLOBAL_FEATURE_DIM = 5
DEFAULT_MIXTURE_COMPONENTS = 12

# Four canonical component slots, followed by the optional resolution shape.
# All values sampled from these coordinates are in [0, 1] and are decoded by
# the same branch codec that enforces user bounds and coupled physical rules.
COMPONENT_PARAMETER_FIELDS = (
    "log_R",
    "sigma_R_fraction",
    "log_h",
    "sigma_h_fraction",
    "log_D",
    "sigma_D_fraction",
)
COMPONENT_PARAMETER_STRIDE = len(COMPONENT_PARAMETER_FIELDS)
RESOLUTION_PARAMETER_FIELDS = ("log_sigma_res", "nu_res")
BRANCH_PARAMETER_LAYOUT = tuple(
    f"component_{slot + 1}.{field}"
    for slot in range(MAX_COMPONENTS)
    for field in COMPONENT_PARAMETER_FIELDS
) + tuple(f"resolution.{field}" for field in RESOLUTION_PARAMETER_FIELDS)
BRANCH_DIM = len(BRANCH_PARAMETER_LAYOUT)
RESOLUTION_OFFSET = MAX_COMPONENTS * COMPONENT_PARAMETER_STRIDE

if NUM_TOPOLOGIES != 34 or MAX_COMPONENTS != 4 or BRANCH_DIM != 26:
    raise RuntimeError("Posterior V8 model layout no longer matches its versioned contract")


@tf.keras.utils.register_keras_serializable(package="GIMaPPosteriorV8")
class ApplyPointMask(tf.keras.layers.Layer):
    """Zero padded point features before every local mixing operation."""

    def call(self, inputs):
        features, point_mask = inputs
        mask = tf.cast(point_mask, tf.bool)[..., tf.newaxis]
        return tf.where(mask, features, tf.zeros_like(features))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@tf.keras.utils.register_keras_serializable(package="GIMaPPosteriorV8")
class MaskedStatisticsPooling(tf.keras.layers.Layer):
    """Masked mean, standard deviation and max without padded-point leakage."""

    def call(self, inputs):
        features, point_mask = inputs
        values = tf.cast(features, tf.float32)
        mask = tf.cast(point_mask, tf.float32)[..., tf.newaxis]
        count = tf.maximum(tf.reduce_sum(mask, axis=1), 1.0)
        mean = tf.reduce_sum(values * mask, axis=1) / count
        variance = tf.reduce_sum(tf.square(values - mean[:, tf.newaxis, :]) * mask, axis=1)
        variance = variance / count
        negative = tf.fill(tf.shape(values), tf.constant(-1.0e30, tf.float32))
        maximum = tf.reduce_max(tf.where(mask > 0.0, values, negative), axis=1)
        any_valid = tf.reduce_any(point_mask, axis=1, keepdims=True)
        maximum = tf.where(any_valid, maximum, tf.zeros_like(maximum))
        return tf.concat([mean, tf.sqrt(tf.maximum(variance, 0.0) + 1.0e-8), maximum], axis=-1)

    def compute_output_shape(self, input_shape):
        feature_shape = tf.TensorShape(input_shape[0])
        return tf.TensorShape((feature_shape[0], feature_shape[-1] * 3))


@tf.keras.utils.register_keras_serializable(package="GIMaPPosteriorV8")
class MaskedBranchBounds(tf.keras.layers.Layer):
    """Expose normalized user envelopes while neutralizing inactive dimensions."""

    def call(self, inputs):
        low, high, active_mask = inputs
        low = tf.cast(low, tf.float32)
        high = tf.cast(high, tf.float32)
        mask = tf.cast(active_mask, tf.float32)
        assertions = (
            tf.debugging.assert_all_finite(low, "branch_low must be finite"),
            tf.debugging.assert_all_finite(high, "branch_high must be finite"),
            tf.debugging.assert_greater_equal(low, 0.0, message="branch_low must be in [0, 1]"),
            tf.debugging.assert_less_equal(high, 1.0, message="branch_high must be in [0, 1]"),
            tf.debugging.assert_less_equal(low, high, message="branch_low must not exceed high"),
            tf.debugging.assert_greater_equal(mask, 0.0, message="active mask must be in [0, 1]"),
            tf.debugging.assert_less_equal(mask, 1.0, message="active mask must be in [0, 1]"),
            tf.debugging.assert_equal(
                mask, tf.round(mask), message="active mask must be binary"
            ),
        )
        with tf.control_dependencies(assertions):
            mask = tf.identity(mask)
        return tf.concat([low * mask, high * mask, mask], axis=-1)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape[0])
        return tf.TensorShape((shape[0], shape[-1] * 3))


@tf.keras.utils.register_keras_serializable(package="GIMaPPosteriorV8")
class HardBranchCondition(tf.keras.layers.Layer):
    """Validate and concatenate one hard topology/presence branch."""

    def call(self, inputs):
        topology, d_present, resolution_present = [
            tf.cast(value, tf.float32) for value in inputs
        ]
        binary = (topology, d_present, resolution_present)
        assertions = [
            tf.debugging.assert_all_finite(value, "hard branch values must be finite")
            for value in binary
        ]
        assertions.extend(
            tf.debugging.assert_equal(
                value, tf.round(value), message="hard branch values must be binary"
            )
            for value in binary
        )
        assertions.append(
            tf.debugging.assert_equal(
                tf.reduce_sum(topology, axis=-1),
                1.0,
                message="branch_topology must be one-hot",
            )
        )
        topology_id = tf.argmax(topology, axis=-1, output_type=tf.int32)
        allowed_slots = tf.gather(tf.constant(TOPOLOGY_SLOT_MASK), topology_id)
        assertions.append(
            tf.debugging.assert_less_equal(
                d_present,
                tf.cast(allowed_slots, tf.float32),
                message="D cannot be present in an inactive topology slot",
            )
        )
        with tf.control_dependencies(assertions):
            return tf.concat(
                [tf.identity(topology), d_present, resolution_present], axis=-1
            )

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0][0], NUM_TOPOLOGIES + MAX_COMPONENTS + 1))


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _masked_conv_encoder(x, point_mask, *, width: int, blocks: int):
    z = ApplyPointMask(name="mask_input")([x, point_mask])
    z = tf.keras.layers.Conv1D(width, 5, padding="same", use_bias=False, name="stem_conv")(z)
    z = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="stem_norm")(z)
    z = tf.keras.layers.Activation("gelu", name="stem_gelu")(z)
    z = ApplyPointMask(name="mask_stem")([z, point_mask])
    for index in range(blocks):
        residual = z
        z = ApplyPointMask(name=f"block_{index}_mask_in")([z, point_mask])
        z = tf.keras.layers.SeparableConv1D(
            width,
            5,
            padding="same",
            dilation_rate=2 ** (index % 6),
            use_bias=False,
            name=f"block_{index}_separable_conv",
        )(z)
        z = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name=f"block_{index}_norm_1"
        )(z)
        z = tf.keras.layers.Activation("gelu", name=f"block_{index}_gelu")(z)
        z = tf.keras.layers.Conv1D(width, 1, use_bias=False, name=f"block_{index}_mix")(z)
        z = tf.keras.layers.Add(name=f"block_{index}_residual")([residual, z])
        z = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name=f"block_{index}_norm_2"
        )(z)
        z = ApplyPointMask(name=f"block_{index}_mask_out")([z, point_mask])
    return MaskedStatisticsPooling(name="masked_statistics")([z, point_mask])


def build_proposal_model(
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    width: int = 128,
    encoder_blocks: int = 6,
    mixture_components: int = DEFAULT_MIXTURE_COMPONENTS,
) -> tf.keras.Model:
    """Build the branch-conditioned V8 proposal network.

    ``branch_low/high`` are normalized physical envelopes relative to the
    global latent domain.  Samples from the returned logistic-normal mixture
    remain in the separate 26-D branch unit cube; decoding is not performed by
    this network.
    """

    max_points = _positive_integer(max_points, "max_points")
    width = _positive_integer(width, "width")
    encoder_blocks = _positive_integer(encoder_blocks, "encoder_blocks")
    mixture_components = _positive_integer(mixture_components, "mixture_components")

    inputs = {
        "x": tf.keras.Input((max_points, POINT_FEATURE_DIM), name="x", dtype=tf.float32),
        "point_mask": tf.keras.Input((max_points,), name="point_mask", dtype=tf.bool),
        "global_features": tf.keras.Input(
            (GLOBAL_FEATURE_DIM,), name="global_features", dtype=tf.float32
        ),
        "branch_topology": tf.keras.Input(
            (NUM_TOPOLOGIES,), name="branch_topology", dtype=tf.float32
        ),
        "branch_d_present": tf.keras.Input(
            (MAX_COMPONENTS,), name="branch_d_present", dtype=tf.float32
        ),
        "branch_resolution_present": tf.keras.Input(
            (1,), name="branch_resolution_present", dtype=tf.float32
        ),
        "branch_low": tf.keras.Input((BRANCH_DIM,), name="branch_low", dtype=tf.float32),
        "branch_high": tf.keras.Input((BRANCH_DIM,), name="branch_high", dtype=tf.float32),
        "active_dimension_mask": tf.keras.Input(
            (BRANCH_DIM,), name="active_dimension_mask", dtype=tf.float32
        ),
    }

    pooled = _masked_conv_encoder(
        inputs["x"], inputs["point_mask"], width=width, blocks=encoder_blocks
    )
    curve_state = tf.keras.layers.Concatenate(name="curve_with_global")(
        [pooled, inputs["global_features"]]
    )
    curve_state = tf.keras.layers.Dense(width * 2, activation="gelu", name="curve_dense_1")(
        curve_state
    )
    curve_state = tf.keras.layers.Dense(width, activation="gelu", name="curve_dense_2")(
        curve_state
    )

    # Both discrete heads intentionally have no graph path from branch inputs.
    topology_logits = tf.keras.layers.Dense(
        NUM_TOPOLOGIES, name="topology_logits", dtype=tf.float32
    )(curve_state)
    branch_pattern_logits = tf.keras.layers.Dense(
        NUM_TOPOLOGIES * BRANCH_PATTERN_COUNT,
        name="branch_pattern_logits_flat",
        dtype=tf.float32,
    )(curve_state)
    branch_pattern_logits = tf.keras.layers.Reshape(
        (NUM_TOPOLOGIES, BRANCH_PATTERN_COUNT),
        name="branch_pattern_logits",
        dtype=tf.float32,
    )(branch_pattern_logits)

    bounds_context = MaskedBranchBounds(name="masked_branch_bounds")(
        [inputs["branch_low"], inputs["branch_high"], inputs["active_dimension_mask"]]
    )
    branch_condition = HardBranchCondition(name="hard_branch_values")(
        [
            inputs["branch_topology"],
            inputs["branch_d_present"],
            inputs["branch_resolution_present"],
        ]
    )
    conditional = tf.keras.layers.Concatenate(name="hard_branch_condition")(
        [
            curve_state,
            branch_condition,
            bounds_context,
        ]
    )
    conditional = tf.keras.layers.Dense(
        width * 2, activation="gelu", name="conditional_dense_1"
    )(conditional)
    conditional = tf.keras.layers.Dense(width * 2, activation="gelu", name="conditional_dense_2")(
        conditional
    )

    mixture_logits = tf.keras.layers.Dense(
        mixture_components, name="mixture_logits", dtype=tf.float32
    )(conditional)
    mixture_loc = tf.keras.layers.Dense(
        mixture_components * BRANCH_DIM, name="mixture_loc_flat", dtype=tf.float32
    )(conditional)
    mixture_loc = tf.keras.layers.Reshape(
        (mixture_components, BRANCH_DIM), name="mixture_loc", dtype=tf.float32
    )(mixture_loc)
    logscale = tf.keras.layers.Dense(
        mixture_components * BRANCH_DIM, name="mixture_logscale_flat", dtype=tf.float32
    )(conditional)
    logscale = tf.keras.layers.Reshape(
        (mixture_components, BRANCH_DIM), name="mixture_logscale_raw", dtype=tf.float32
    )(logscale)
    logscale = tf.keras.layers.Activation(
        "tanh", name="mixture_logscale_tanh", dtype=tf.float32
    )(logscale)
    mixture_logscale = tf.keras.layers.Rescaling(
        scale=3.0, offset=-2.0, name="mixture_logscale", dtype=tf.float32
    )(logscale)

    return tf.keras.Model(
        inputs=inputs,
        outputs={
            "topology_logits": topology_logits,
            "branch_pattern_logits": branch_pattern_logits,
            "mixture_logits": mixture_logits,
            "mixture_loc": mixture_loc,
            "mixture_logscale": mixture_logscale,
        },
        name="posterior_v8_proposal",
    )


def masked_logistic_normal_nll(
    target_unit,
    active_dimension_mask,
    mixture_logits,
    mixture_loc,
    mixture_logscale,
    *,
    epsilon: float = 1.0e-5,
):
    """Per-example mixture NLL using only explicitly active branch dimensions."""

    target = tf.cast(target_unit, tf.float32)
    active = tf.cast(active_dimension_mask, tf.float32) > 0.5
    target = tf.where(active, target, tf.fill(tf.shape(target), 0.5))
    target = tf.clip_by_value(target, epsilon, 1.0 - epsilon)
    latent = tf.math.log(target) - tf.math.log1p(-target)
    loc = tf.cast(mixture_loc, tf.float32)
    logscale = tf.cast(mixture_logscale, tf.float32)
    standardized = (latent[:, tf.newaxis, :] - loc) * tf.exp(-logscale)
    normal_log_prob = -0.5 * tf.square(standardized) - logscale
    normal_log_prob -= 0.5 * tf.math.log(tf.constant(2.0 * 3.141592653589793))
    jacobian = -tf.math.log(target) - tf.math.log1p(-target)
    dimension_log_prob = normal_log_prob + jacobian[:, tf.newaxis, :]
    active_float = tf.cast(active, tf.float32)[:, tf.newaxis, :]
    component_log_prob = tf.reduce_sum(dimension_log_prob * active_float, axis=-1)
    log_weights = tf.nn.log_softmax(tf.cast(mixture_logits, tf.float32), axis=-1)
    return -tf.reduce_logsumexp(log_weights + component_log_prob, axis=-1)


def mask_invalid_branch_pattern_logits(logits, *, invalid_logit: float = -1.0e9):
    """Apply the 34-topology hard mask used by both training and beam search."""

    values = tf.convert_to_tensor(logits)
    tf.debugging.assert_equal(
        tf.shape(values)[-2:],
        [NUM_TOPOLOGIES, BRANCH_PATTERN_COUNT],
        message="branch pattern logits must end in [34, 32]",
    )
    valid = tf.constant(VALID_BRANCH_PATTERN_MASK, dtype=tf.bool)
    return tf.where(valid, values, tf.cast(invalid_logit, values.dtype))


def sample_logistic_normal_mixture(
    mixture_logits,
    mixture_loc,
    mixture_logscale,
    active_dimension_mask,
    *,
    sample_count: int = 1,
    seed: Sequence[int] = (0, 0),
):
    """Draw unit-cube samples; inactive coordinates are set to neutral 0.5."""

    sample_count = _positive_integer(sample_count, "sample_count")
    logits = tf.cast(mixture_logits, tf.float32)
    loc = tf.cast(mixture_loc, tf.float32)
    logscale = tf.cast(mixture_logscale, tf.float32)
    seed_tensor = tf.convert_to_tensor(seed, dtype=tf.int32)
    tf.debugging.assert_equal(tf.shape(seed_tensor), [2], message="seed must contain two integers")
    component = tf.random.stateless_categorical(logits, sample_count, seed_tensor)
    chosen_loc = tf.gather(loc, component, axis=1, batch_dims=1)
    chosen_logscale = tf.gather(logscale, component, axis=1, batch_dims=1)
    noise_seed = tf.random.experimental.stateless_fold_in(seed_tensor, 1)
    noise = tf.random.stateless_normal(tf.shape(chosen_loc), noise_seed, dtype=tf.float32)
    samples = tf.math.sigmoid(chosen_loc + tf.exp(chosen_logscale) * noise)
    active = tf.cast(active_dimension_mask, tf.float32)[:, tf.newaxis, :] > 0.5
    samples = tf.where(active, samples, tf.fill(tf.shape(samples), 0.5))
    return samples, component


def proposal_output_shapes(
    batch_size: int | None = None,
    *,
    mixture_components: int = DEFAULT_MIXTURE_COMPONENTS,
) -> Mapping[str, tuple[int | None, ...]]:
    """Stable tensor contract for trainers and inference adapters."""

    return {
        "topology_logits": (batch_size, NUM_TOPOLOGIES),
        "branch_pattern_logits": (
            batch_size,
            NUM_TOPOLOGIES,
            BRANCH_PATTERN_COUNT,
        ),
        "mixture_logits": (batch_size, mixture_components),
        "mixture_loc": (batch_size, mixture_components, BRANCH_DIM),
        "mixture_logscale": (batch_size, mixture_components, BRANCH_DIM),
    }


__all__ = [
    "BRANCH_DIM",
    "BRANCH_PATTERN_COUNT",
    "BRANCH_PARAMETER_LAYOUT",
    "COMPONENT_PARAMETER_FIELDS",
    "COMPONENT_PARAMETER_STRIDE",
    "DEFAULT_MAX_POINTS",
    "DEFAULT_MIXTURE_COMPONENTS",
    "GLOBAL_FEATURE_DIM",
    "MODEL_VERSION",
    "POINT_FEATURE_DIM",
    "RESOLUTION_OFFSET",
    "RESOLUTION_PARAMETER_FIELDS",
    "build_proposal_model",
    "masked_logistic_normal_nll",
    "mask_invalid_branch_pattern_logits",
    "proposal_output_shapes",
    "sample_logistic_normal_mixture",
]
