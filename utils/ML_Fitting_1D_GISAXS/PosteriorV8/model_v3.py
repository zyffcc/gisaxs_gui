"""Bounds-conditioned local-coordinate proposal network (V3).

Unlike V1/V2, this graph consumes the authoritative 78-D encoding of actual
GUI physical intervals.  It never labels that tensor as a 26-D global-codec
box.  Discrete heads remain functions of the observed curve alone; only the
continuous proposal head sees the requested hard branch and user bounds.
"""

from __future__ import annotations

from numbers import Integral

import tensorflow as tf

from .bounds_first_contract import BOUNDS_EMBEDDING_DIM
from .bounds_model_contract import (
    BOUNDS_PROPOSAL_MODEL_NAME,
    BOUNDS_PROPOSAL_MODEL_VERSION,
    CURVE_GLOBAL_FEATURE_DIM,
    CURVE_POINT_FEATURE_DIM,
    MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS,
    MODEL_BRANCH_CATALOG_VERSION,
    MODEL_COMPONENT_SLOTS_VERSION,
    MODEL_OUTPUT_COORDINATE_SEMANTICS,
)
from .branch_catalog import BRANCH_PATTERN_COUNT
from .canonical_branch_catalog import CANONICAL_VALID_BRANCH_PATTERN_MASK
from .contract import MAX_COMPONENTS, NUM_TOPOLOGIES
from .model import (
    BRANCH_DIM,
    DEFAULT_MAX_POINTS,
    DEFAULT_MIXTURE_COMPONENTS,
    GLOBAL_FEATURE_DIM,
    POINT_FEATURE_DIM,
    HardBranchCondition,
    _masked_conv_encoder,
)
from .training_objective import ACTIVE_DIMENSION_MASKS

if (
    POINT_FEATURE_DIM != CURVE_POINT_FEATURE_DIM
    or GLOBAL_FEATURE_DIM != CURVE_GLOBAL_FEATURE_DIM
):  # pragma: no cover
    raise RuntimeError("V3 curve input contract disagrees with the shared encoder")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@tf.keras.utils.register_keras_serializable(package="GIMaPPosteriorV8")
class PhysicalBoundsContext(tf.keras.layers.Layer):
    """Validate and expose actual GUI bounds plus local codec masks."""

    def call(self, inputs):
        embedding, active_mask, varying_mask = [
            tf.cast(value, tf.float32) for value in inputs
        ]
        low = embedding[..., 0::3]
        high = embedding[..., 1::3]
        presence = embedding[..., 2::3]
        active = tf.cast(active_mask, tf.float32)
        varying = tf.cast(varying_mask, tf.float32)
        absent = presence < 0.5
        assertions = (
            tf.debugging.assert_all_finite(
                embedding, "bounds_embedding contains NaN/Inf"
            ),
            tf.debugging.assert_greater_equal(
                embedding, 0.0, message="bounds_embedding must be in [0, 1]"
            ),
            tf.debugging.assert_less_equal(
                embedding, 1.0, message="bounds_embedding must be in [0, 1]"
            ),
            tf.debugging.assert_equal(
                presence,
                tf.round(presence),
                message="bounds presence values must be binary",
            ),
            tf.debugging.assert_less_equal(
                low, high, message="physical lower bounds must not exceed upper bounds"
            ),
            tf.debugging.assert_equal(
                tf.boolean_mask(low, absent),
                tf.fill(tf.shape(tf.boolean_mask(low, absent)), 0.5),
                message="absent physical lower bounds must equal 0.5",
            ),
            tf.debugging.assert_equal(
                tf.boolean_mask(high, absent),
                tf.fill(tf.shape(tf.boolean_mask(high, absent)), 0.5),
                message="absent physical upper bounds must equal 0.5",
            ),
            tf.debugging.assert_all_finite(active, "active mask contains NaN/Inf"),
            tf.debugging.assert_all_finite(varying, "varying mask contains NaN/Inf"),
            tf.debugging.assert_equal(
                active, tf.round(active), message="active mask must be binary"
            ),
            tf.debugging.assert_greater_equal(
                active, 0.0, message="active mask must be binary"
            ),
            tf.debugging.assert_less_equal(
                active, 1.0, message="active mask must be binary"
            ),
            tf.debugging.assert_equal(
                varying, tf.round(varying), message="varying mask must be binary"
            ),
            tf.debugging.assert_greater_equal(
                varying, 0.0, message="varying mask must be binary"
            ),
            tf.debugging.assert_less_equal(
                varying, 1.0, message="varying mask must be binary"
            ),
            tf.debugging.assert_equal(
                presence,
                active,
                message="physical bounds presence must equal local-codec active mask",
            ),
            tf.debugging.assert_less_equal(
                varying,
                active,
                message="varying dimensions must be semantically active",
            ),
        )
        with tf.control_dependencies(assertions):
            return tf.concat(
                [tf.identity(embedding), tf.identity(active), tf.identity(varying)],
                axis=-1,
            )

    def compute_output_shape(self, input_shape):
        batch = tf.TensorShape(input_shape[0])[0]
        return tf.TensorShape((batch, BOUNDS_EMBEDDING_DIM + 2 * BRANCH_DIM))


@tf.keras.utils.register_keras_serializable(package="GIMaPPosteriorV8")
class StructuralActiveMask(tf.keras.layers.Layer):
    """Require the local active mask to match the selected hard branch."""

    def call(self, inputs):
        topology, d_present, resolution_present, active_mask = [
            tf.cast(value, tf.float32) for value in inputs
        ]
        binary_values = (topology, d_present, resolution_present)
        assertions = [
            tf.debugging.assert_all_finite(value, "hard branch contains NaN/Inf")
            for value in binary_values
        ]
        assertions.extend(
            tf.debugging.assert_equal(
                value, tf.round(value), message="hard branch values must be binary"
            )
            for value in binary_values
        )
        assertions.extend(
            tf.debugging.assert_greater_equal(
                value, 0.0, message="hard branch values must be binary"
            )
            for value in binary_values
        )
        assertions.extend(
            tf.debugging.assert_less_equal(
                value, 1.0, message="hard branch values must be binary"
            )
            for value in binary_values
        )
        assertions.append(
            tf.debugging.assert_equal(
                tf.reduce_sum(topology, axis=-1),
                1.0,
                message="hard topology must be one-hot",
            )
        )
        with tf.control_dependencies(assertions):
            topology, d_present, resolution_present = (
                tf.identity(value) for value in binary_values
            )
        topology_id = tf.argmax(topology, axis=-1, output_type=tf.int32)
        d_bits = tf.cast(tf.round(d_present), tf.int32)
        pattern_id = tf.reduce_sum(
            d_bits * tf.constant((1, 2, 4, 8), tf.int32), axis=-1
        )
        pattern_id += 16 * tf.cast(
            tf.round(resolution_present[:, 0]), tf.int32
        )
        canonical = tf.gather_nd(
            tf.constant(CANONICAL_VALID_BRANCH_PATTERN_MASK),
            tf.stack([topology_id, pattern_id], axis=-1),
        )
        expected = tf.cast(
            tf.gather_nd(
                tf.constant(ACTIVE_DIMENSION_MASKS),
                tf.stack([topology_id, pattern_id], axis=-1),
            ),
            tf.float32,
        )
        active = tf.cast(active_mask, tf.float32)
        assertions = (
            tf.debugging.assert_equal(
                canonical,
                tf.ones_like(canonical),
                message="hard branch is not the canonical physical representative",
            ),
            tf.debugging.assert_equal(
                active,
                expected,
                message="active mask does not match the selected hard branch",
            ),
        )
        with tf.control_dependencies(assertions):
            return tf.identity(active)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[-1])


def build_bounds_proposal_model(
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    width: int = 128,
    encoder_blocks: int = 6,
    mixture_components: int = DEFAULT_MIXTURE_COMPONENTS,
) -> tf.keras.Model:
    """Build the V3 graph with genuine physical-bound conditioning."""

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
        "bounds_embedding": tf.keras.Input(
            (BOUNDS_EMBEDDING_DIM,), name="bounds_embedding", dtype=tf.float32
        ),
        "active_dimension_mask": tf.keras.Input(
            (BRANCH_DIM,), name="active_dimension_mask", dtype=tf.float32
        ),
        "varying_dimension_mask": tf.keras.Input(
            (BRANCH_DIM,), name="varying_dimension_mask", dtype=tf.float32
        ),
    }
    pooled = _masked_conv_encoder(
        inputs["x"], inputs["point_mask"], width=width, blocks=encoder_blocks
    )
    curve_state = tf.keras.layers.Concatenate(name="curve_with_global")(
        [pooled, inputs["global_features"]]
    )
    curve_state = tf.keras.layers.Dense(
        width * 2, activation="gelu", name="curve_dense_1"
    )(curve_state)
    curve_state = tf.keras.layers.Dense(
        width, activation="gelu", name="curve_dense_2"
    )(curve_state)

    # These two tensors have no graph path from branch or bounds inputs.
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

    hard_branch = HardBranchCondition(name="hard_branch_values")(
        [
            inputs["branch_topology"],
            inputs["branch_d_present"],
            inputs["branch_resolution_present"],
        ]
    )
    structural_active = StructuralActiveMask(name="structural_active_mask")(
        [
            inputs["branch_topology"],
            inputs["branch_d_present"],
            inputs["branch_resolution_present"],
            inputs["active_dimension_mask"],
        ]
    )
    physical_bounds = PhysicalBoundsContext(name="physical_gui_bounds_context")(
        [
            inputs["bounds_embedding"],
            structural_active,
            inputs["varying_dimension_mask"],
        ]
    )
    conditional = tf.keras.layers.Concatenate(name="bounds_first_condition")(
        [curve_state, hard_branch, physical_bounds]
    )
    conditional = tf.keras.layers.Dense(
        width * 2, activation="gelu", name="conditional_dense_1"
    )(conditional)
    conditional = tf.keras.layers.Dense(
        width * 2, activation="gelu", name="conditional_dense_2"
    )(conditional)
    mixture_logits = tf.keras.layers.Dense(
        mixture_components, name="mixture_logits", dtype=tf.float32
    )(conditional)
    mixture_loc = tf.keras.layers.Dense(
        mixture_components * BRANCH_DIM, name="mixture_loc_flat", dtype=tf.float32
    )(conditional)
    mixture_loc = tf.keras.layers.Reshape(
        (mixture_components, BRANCH_DIM), name="mixture_loc", dtype=tf.float32
    )(mixture_loc)
    raw_logscale = tf.keras.layers.Dense(
        mixture_components * BRANCH_DIM,
        name="mixture_logscale_flat",
        dtype=tf.float32,
    )(conditional)
    raw_logscale = tf.keras.layers.Reshape(
        (mixture_components, BRANCH_DIM),
        name="mixture_logscale_raw",
        dtype=tf.float32,
    )(raw_logscale)
    raw_logscale = tf.keras.layers.Activation(
        "tanh", name="mixture_logscale_tanh", dtype=tf.float32
    )(raw_logscale)
    mixture_logscale = tf.keras.layers.Rescaling(
        scale=3.0, offset=-2.0, name="mixture_logscale", dtype=tf.float32
    )(raw_logscale)
    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "topology_logits": topology_logits,
            "branch_pattern_logits": branch_pattern_logits,
            "mixture_logits": mixture_logits,
            "mixture_loc": mixture_loc,
            "mixture_logscale": mixture_logscale,
        },
        name=BOUNDS_PROPOSAL_MODEL_NAME,
    )
    model.posterior_v8_model_version = BOUNDS_PROPOSAL_MODEL_VERSION
    model.posterior_v8_branch_catalog_version = MODEL_BRANCH_CATALOG_VERSION
    model.posterior_v8_component_slots_version = MODEL_COMPONENT_SLOTS_VERSION
    model.posterior_v8_input_bounds_coordinate_semantics = (
        MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS
    )
    model.posterior_v8_output_coordinate_semantics = MODEL_OUTPUT_COORDINATE_SEMANTICS
    return model


__all__ = [
    "PhysicalBoundsContext",
    "StructuralActiveMask",
    "build_bounds_proposal_model",
]
