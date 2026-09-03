"""V5 per-hard-branch compatibility proposal and local-coordinate MDN.

The caller enumerates one user-query contextual branch catalog and invokes
this graph once per candidate hard branch (or as a batch of such candidates).
There are intentionally no curve-only topology or branch-classification
heads. The scalar score is an uncalibrated frozen-protocol search-yield logit,
not a posterior probability, branch-solvability claim, or exact decision.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
import tensorflow as tf

from .amplitude_query_v5 import (
    AMPLITUDE_AXIS_KEYS,
    AMPLITUDE_QUERY_EMBEDDING_DIM,
    AMPLITUDE_QUERY_PADDING_VALUE,
)
from .bounds_first_contract import BOUNDS_EMBEDDING_DIM
from .branch_catalog import BRANCH_PATTERN_COUNT, VALID_BRANCH_PATTERN_MASK
from .contract import MAX_COMPONENTS, NUM_TOPOLOGIES, TOPOLOGIES
from .model import (
    BRANCH_DIM,
    DEFAULT_MAX_POINTS,
    DEFAULT_MIXTURE_COMPONENTS,
    GLOBAL_FEATURE_DIM,
    POINT_FEATURE_DIM,
    _masked_conv_encoder,
)
from .model_v5_contract import (
    CURVE_GLOBAL_FEATURE_DIM,
    CURVE_POINT_FEATURE_DIM,
    MODEL_V5_INPUT_KEYS,
    MODEL_V5_NAME,
    MODEL_V5_OUTPUT_KEYS,
    MODEL_V5_SCHEMA,
    MODEL_V5_VERSION,
    UNCERTAINTY_PROVENANCE_STATES,
    model_v5_contract_payload,
)
from .training_objective import ACTIVE_DIMENSION_MASKS


if (
    POINT_FEATURE_DIM != CURVE_POINT_FEATURE_DIM or GLOBAL_FEATURE_DIM != CURVE_GLOBAL_FEATURE_DIM
):  # pragma: no cover
    raise RuntimeError("V5 curve input contract disagrees with the shared encoder")


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _binary_assertions(value, name: str):
    return (
        tf.debugging.assert_all_finite(value, f"{name} contains NaN/Inf"),
        tf.debugging.assert_equal(value, tf.round(value), message=f"{name} must be binary"),
        tf.debugging.assert_greater_equal(value, 0.0, message=f"{name} must be binary"),
        tf.debugging.assert_less_equal(value, 1.0, message=f"{name} must be binary"),
    )


@tf.keras.utils.register_keras_serializable(package="GIMaPPosteriorV8V5")
class BranchConditionedPhysicalContext(tf.keras.layers.Layer):
    """Validate one wire branch and its three distinct physical-axis masks."""

    def call(self, inputs):
        (
            topology_id_input,
            pattern_input,
            embedding_input,
            available_input,
            active_input,
            varying_input,
        ) = inputs
        topology_column = tf.cast(topology_id_input, tf.int32)
        pattern_column = tf.cast(pattern_input, tf.int32)
        embedding = tf.cast(embedding_input, tf.float32)
        available = tf.cast(available_input, tf.float32)
        active = tf.cast(active_input, tf.float32)
        varying = tf.cast(varying_input, tf.float32)
        topology_id = topology_column[:, 0]
        pattern = pattern_column[:, 0]

        initial_assertions = [
            tf.debugging.assert_greater_equal(
                topology_id, 0, message="branch_topology_id must be in [0, 33]"
            ),
            tf.debugging.assert_less(
                topology_id,
                NUM_TOPOLOGIES,
                message="branch_topology_id must be in [0, 33]",
            ),
        ]
        initial_assertions.extend(
            (
                tf.debugging.assert_greater_equal(
                    pattern, 0, message="branch_pattern_id must be in [0, 31]"
                ),
                tf.debugging.assert_less(
                    pattern,
                    BRANCH_PATTERN_COUNT,
                    message="branch_pattern_id must be in [0, 31]",
                ),
                tf.debugging.assert_all_finite(embedding, "bounds_embedding contains NaN/Inf"),
                tf.debugging.assert_greater_equal(
                    embedding, 0.0, message="bounds_embedding must be in [0, 1]"
                ),
                tf.debugging.assert_less_equal(
                    embedding, 1.0, message="bounds_embedding must be in [0, 1]"
                ),
            )
        )
        for value, name in (
            (available, "available_dimension_mask"),
            (active, "active_dimension_mask"),
            (varying, "varying_dimension_mask"),
        ):
            initial_assertions.extend(_binary_assertions(value, name))

        with tf.control_dependencies(initial_assertions):
            topology_id = tf.identity(topology_id)
            pattern = tf.identity(pattern)
            embedding = tf.identity(embedding)
            available = tf.identity(available)
            active = tf.identity(active)
            varying = tf.identity(varying)

        branch_indices = tf.stack([topology_id, pattern], axis=-1)
        valid_wire_branch = tf.gather_nd(tf.constant(VALID_BRANCH_PATTERN_MASK), branch_indices)
        all_structural_masks = tf.constant(ACTIVE_DIMENSION_MASKS)
        expected_active = tf.cast(tf.gather_nd(all_structural_masks, branch_indices), tf.float32)
        maximum_available = tf.cast(
            tf.reduce_any(tf.gather(all_structural_masks, topology_id), axis=1),
            tf.float32,
        )

        low = embedding[..., 0::3]
        high = embedding[..., 1::3]
        presence = embedding[..., 2::3]
        absent = presence < 0.5
        semantic_assertions = (
            tf.debugging.assert_equal(
                valid_wire_branch,
                tf.ones_like(valid_wire_branch),
                message="branch_pattern_id is invalid for branch_topology",
            ),
            tf.debugging.assert_less_equal(
                low,
                high,
                message="physical lower bounds must not exceed upper bounds",
            ),
            tf.debugging.assert_equal(
                presence,
                tf.round(presence),
                message="bounds embedding presence values must be binary",
            ),
            tf.debugging.assert_equal(
                tf.boolean_mask(low, absent),
                tf.fill(tf.shape(tf.boolean_mask(low, absent)), 0.5),
                message="unavailable physical lower bounds must equal 0.5",
            ),
            tf.debugging.assert_equal(
                tf.boolean_mask(high, absent),
                tf.fill(tf.shape(tf.boolean_mask(high, absent)), 0.5),
                message="unavailable physical upper bounds must equal 0.5",
            ),
            tf.debugging.assert_equal(
                presence,
                available,
                message=("bounds embedding presence must equal available_dimension_mask"),
            ),
            tf.debugging.assert_less_equal(
                available,
                maximum_available,
                message="available axes conflict with branch_topology",
            ),
            tf.debugging.assert_equal(
                active,
                expected_active,
                message="active_dimension_mask does not match the selected wire branch",
            ),
            tf.debugging.assert_less_equal(
                active,
                available,
                message="active dimensions must be available in the user query",
            ),
            tf.debugging.assert_less_equal(
                varying,
                active,
                message="varying dimensions must be active in the selected branch",
            ),
        )
        with tf.control_dependencies(semantic_assertions):
            topology_one_hot = tf.one_hot(
                tf.identity(topology_id), NUM_TOPOLOGIES, dtype=tf.float32
            )
            pattern_one_hot = tf.one_hot(
                tf.identity(pattern), BRANCH_PATTERN_COUNT, dtype=tf.float32
            )
            return tf.concat(
                [
                    topology_one_hot,
                    pattern_one_hot,
                    tf.identity(embedding),
                    tf.identity(available),
                    tf.identity(active),
                    tf.identity(varying),
                ],
                axis=-1,
            )

    def compute_output_shape(self, input_shape):
        batch = tf.TensorShape(input_shape[0])[0]
        dimensions = NUM_TOPOLOGIES + BRANCH_PATTERN_COUNT + BOUNDS_EMBEDDING_DIM + 3 * BRANCH_DIM
        return tf.TensorShape((batch, dimensions))


@tf.keras.utils.register_keras_serializable(package="GIMaPPosteriorV8V5")
class AmplitudeBoundsContext(tf.keras.layers.Layer):
    """Validate the full GUI amplitude-range condition for one user query."""

    def call(self, inputs):
        topology_id_input, geometry_embedding_input, amplitude_embedding_input = inputs
        topology_id = tf.cast(topology_id_input[:, 0], tf.int32)
        geometry_embedding = tf.cast(geometry_embedding_input, tf.float32)
        embedding = tf.cast(amplitude_embedding_input, tf.float32)
        low = embedding[..., 0::3]
        high = embedding[..., 1::3]
        presence = embedding[..., 2::3]
        absent = presence < 0.5

        component_presence = np.zeros((NUM_TOPOLOGIES, len(AMPLITUDE_AXIS_KEYS)), dtype=np.float32)
        component_presence[:, :2] = 1.0
        for topology_id_value, topology in enumerate(TOPOLOGIES):
            component_presence[topology_id_value, 2 : 2 + len(topology)] = 1.0
        expected_presence = tf.gather(tf.constant(component_presence), topology_id)
        geometry_presence = geometry_embedding[..., 2::3]
        resolution_available = tf.reduce_max(geometry_presence[..., -2:], axis=-1)
        expected_presence = tf.concat(
            [expected_presence[..., :-1], resolution_available[..., tf.newaxis]],
            axis=-1,
        )
        intensity_high = high[..., 2 : 2 + MAX_COMPONENTS]
        intensity_present = presence[..., 2 : 2 + MAX_COMPONENTS]

        assertions = (
            tf.debugging.assert_all_finite(
                embedding, "amplitude_bounds_embedding contains NaN/Inf"
            ),
            tf.debugging.assert_greater_equal(
                embedding, 0.0, message="amplitude_bounds_embedding must be in [0, 1]"
            ),
            tf.debugging.assert_less_equal(
                embedding, 1.0, message="amplitude_bounds_embedding must be in [0, 1]"
            ),
            tf.debugging.assert_less_equal(
                low,
                high,
                message="amplitude physical lower bounds must not exceed upper bounds",
            ),
            tf.debugging.assert_equal(
                presence,
                tf.round(presence),
                message="amplitude bounds presence values must be binary",
            ),
            tf.debugging.assert_equal(
                tf.boolean_mask(low, absent),
                tf.fill(
                    tf.shape(tf.boolean_mask(low, absent)),
                    tf.cast(AMPLITUDE_QUERY_PADDING_VALUE, tf.float32),
                ),
                message="unavailable amplitude lower bounds must use canonical padding",
            ),
            tf.debugging.assert_equal(
                tf.boolean_mask(high, absent),
                tf.fill(
                    tf.shape(tf.boolean_mask(high, absent)),
                    tf.cast(AMPLITUDE_QUERY_PADDING_VALUE, tf.float32),
                ),
                message="unavailable amplitude upper bounds must use canonical padding",
            ),
            tf.debugging.assert_equal(
                presence,
                expected_presence,
                message="amplitude bounds presence conflicts with topology or Resolution query",
            ),
            tf.debugging.assert_greater(
                tf.reduce_max(intensity_high * intensity_present, axis=-1),
                0.0,
                message="at least one component Int range must permit positive amplitude",
            ),
        )
        with tf.control_dependencies(assertions):
            return tf.identity(embedding)

    def compute_output_shape(self, input_shape):
        batch = tf.TensorShape(input_shape[0])[0]
        return tf.TensorShape((batch, AMPLITUDE_QUERY_EMBEDDING_DIM))


@tf.keras.utils.register_keras_serializable(package="GIMaPPosteriorV8V5")
class UncertaintyProvenanceContext(tf.keras.layers.Layer):
    """Validate the measured/simulated/missing-sigma provenance one-hot."""

    def call(self, inputs):
        values = tf.cast(inputs, tf.float32)
        assertions = (*_binary_assertions(values, "uncertainty_provenance"),)
        assertions += (
            tf.debugging.assert_equal(
                tf.reduce_sum(values, axis=-1),
                1.0,
                message="uncertainty_provenance must be one-hot",
            ),
        )
        with tf.control_dependencies(assertions):
            return tf.identity(values)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape)


@tf.keras.utils.register_keras_serializable(package="GIMaPPosteriorV8V5")
class V5ContractStamp(tf.keras.layers.Layer):
    """Persist the exact V5 schema/version inside the serialized Keras graph."""

    def __init__(
        self,
        *,
        schema_version: str = MODEL_V5_SCHEMA,
        model_version: str = MODEL_V5_VERSION,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if schema_version != MODEL_V5_SCHEMA:
            raise ValueError("unsupported V5 model schema")
        if model_version != MODEL_V5_VERSION:
            raise ValueError("unsupported V5 model version")
        self.schema_version = schema_version
        self.model_version = model_version

    def call(self, inputs):
        return tf.identity(inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "schema_version": self.schema_version,
                "model_version": self.model_version,
            }
        )
        return config


def build_branch_conditioned_proposal_model(
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    width: int = 128,
    encoder_blocks: int = 6,
    mixture_components: int = DEFAULT_MIXTURE_COMPONENTS,
) -> tf.keras.Model:
    """Build the V5 per-candidate compatibility-logit and local-MDN graph."""

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
        "uncertainty_provenance": tf.keras.Input(
            (len(UNCERTAINTY_PROVENANCE_STATES),),
            name="uncertainty_provenance",
            dtype=tf.float32,
        ),
        "branch_topology_id": tf.keras.Input((1,), name="branch_topology_id", dtype=tf.int32),
        "branch_pattern_id": tf.keras.Input((1,), name="branch_pattern_id", dtype=tf.int32),
        "geometry_bounds_embedding": tf.keras.Input(
            (BOUNDS_EMBEDDING_DIM,), name="geometry_bounds_embedding", dtype=tf.float32
        ),
        "amplitude_bounds_embedding": tf.keras.Input(
            (AMPLITUDE_QUERY_EMBEDDING_DIM,),
            name="amplitude_bounds_embedding",
            dtype=tf.float32,
        ),
        "available_dimension_mask": tf.keras.Input(
            (BRANCH_DIM,), name="available_dimension_mask", dtype=tf.float32
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
    uncertainty_provenance = UncertaintyProvenanceContext(name="uncertainty_provenance_context")(
        inputs["uncertainty_provenance"]
    )
    curve_state = tf.keras.layers.Concatenate(name="curve_with_global_and_uncertainty")(
        [pooled, inputs["global_features"], uncertainty_provenance]
    )
    curve_state = tf.keras.layers.Dense(width * 2, activation="gelu", name="curve_dense_1")(
        curve_state
    )
    curve_state = tf.keras.layers.Dense(width, activation="gelu", name="curve_dense_2")(curve_state)

    branch_context = BranchConditionedPhysicalContext(name="branch_conditioned_physical_context")(
        [
            inputs["branch_topology_id"],
            inputs["branch_pattern_id"],
            inputs["geometry_bounds_embedding"],
            inputs["available_dimension_mask"],
            inputs["active_dimension_mask"],
            inputs["varying_dimension_mask"],
        ]
    )
    branch_context = V5ContractStamp(name="v5_contract_stamp")(branch_context)
    amplitude_context = AmplitudeBoundsContext(name="amplitude_bounds_context")(
        [
            inputs["branch_topology_id"],
            inputs["geometry_bounds_embedding"],
            inputs["amplitude_bounds_embedding"],
        ]
    )
    conditional = tf.keras.layers.Concatenate(name="curve_bounds_branch_condition")(
        [curve_state, branch_context, amplitude_context]
    )
    conditional = tf.keras.layers.Dense(width * 2, activation="gelu", name="conditional_dense_1")(
        conditional
    )
    conditional = tf.keras.layers.Dense(width * 2, activation="gelu", name="conditional_dense_2")(
        conditional
    )

    proposal_search_yield_logit = tf.keras.layers.Dense(
        1, name="proposal_search_yield_logit", dtype=tf.float32
    )(conditional)
    mixture_logits = tf.keras.layers.Dense(
        mixture_components, name="mixture_logits", dtype=tf.float32
    )(conditional)
    mixture_loc = tf.keras.layers.Dense(
        mixture_components * BRANCH_DIM,
        name="mixture_loc_flat",
        dtype=tf.float32,
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
        scale=3.0,
        offset=-2.0,
        name="mixture_logscale",
        dtype=tf.float32,
    )(raw_logscale)
    model = tf.keras.Model(
        inputs=inputs,
        outputs={
            "proposal_search_yield_logit": proposal_search_yield_logit,
            "mixture_logits": mixture_logits,
            "mixture_loc": mixture_loc,
            "mixture_logscale": mixture_logscale,
        },
        name=MODEL_V5_NAME,
    )
    model.posterior_v8_model_schema = MODEL_V5_SCHEMA
    model.posterior_v8_model_version = MODEL_V5_VERSION
    model.posterior_v8_model_contract = model_v5_contract_payload()
    return model


def validate_model_v5_graph_contract(model: tf.keras.Model) -> tf.keras.Model:
    """Reject legacy or structurally incompatible Keras graphs fail closed."""

    if not isinstance(model, tf.keras.Model):
        raise TypeError("model must be a Keras Model")
    if model.name != MODEL_V5_NAME:
        raise ValueError("model has an incompatible V5 name")
    input_names = {value.name.split(":", 1)[0].rsplit("/", 1)[-1] for value in model.inputs}
    if input_names != set(MODEL_V5_INPUT_KEYS):
        raise ValueError("model has incompatible V5 inputs")
    if set(model.output_names) != set(MODEL_V5_OUTPUT_KEYS):
        raise ValueError("model has incompatible V5 outputs")
    stamps = [layer for layer in model.layers if isinstance(layer, V5ContractStamp)]
    if len(stamps) != 1:
        raise ValueError("model is missing the unique V5 contract stamp")
    stamp = stamps[0]
    if stamp.schema_version != MODEL_V5_SCHEMA or stamp.model_version != MODEL_V5_VERSION:
        raise ValueError("model has an incompatible V5 contract stamp")
    return model


__all__ = [
    "AmplitudeBoundsContext",
    "BranchConditionedPhysicalContext",
    "UncertaintyProvenanceContext",
    "V5ContractStamp",
    "build_branch_conditioned_proposal_model",
    "validate_model_v5_graph_contract",
]
