"""Local-target v2 proposal model contract.

The neural architecture remains intentionally identical to the frozen global-
target v1 baseline.  A distinct model name/version plus the required training
manifest prevents identical tensor shapes from hiding incompatible coordinate
semantics.
"""

from __future__ import annotations

import tensorflow as tf

from .local_target import (
    GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS,
    LOCAL_TARGET_COORDINATE_SEMANTICS,
)
from .model import build_proposal_model


LOCAL_PROPOSAL_MODEL_VERSION = "posterior_v8_masked_conv_logistic_normal_local_target_v2"
LOCAL_PROPOSAL_MODEL_NAME = "posterior_v8_local_target_proposal_v2"
MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS = GLOBAL_BRANCH_BOX_COORDINATE_SEMANTICS
MODEL_OUTPUT_COORDINATE_SEMANTICS = LOCAL_TARGET_COORDINATE_SEMANTICS


def build_local_proposal_model(**kwargs) -> tf.keras.Model:
    """Build the v1 graph under the explicit local-target v2 model identity."""

    baseline = build_proposal_model(**kwargs)
    model = tf.keras.Model(
        inputs=baseline.inputs,
        outputs=baseline.output,
        name=LOCAL_PROPOSAL_MODEL_NAME,
    )
    # These attributes are convenient for an in-process builder.  The v2
    # training manifest remains the persistent source of truth after saving.
    model.posterior_v8_model_version = LOCAL_PROPOSAL_MODEL_VERSION
    model.posterior_v8_input_bounds_coordinate_semantics = (
        MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS
    )
    model.posterior_v8_output_coordinate_semantics = MODEL_OUTPUT_COORDINATE_SEMANTICS
    return model


__all__ = [
    "LOCAL_PROPOSAL_MODEL_NAME",
    "LOCAL_PROPOSAL_MODEL_VERSION",
    "MODEL_INPUT_BOUNDS_COORDINATE_SEMANTICS",
    "MODEL_OUTPUT_COORDINATE_SEMANTICS",
    "build_local_proposal_model",
]
