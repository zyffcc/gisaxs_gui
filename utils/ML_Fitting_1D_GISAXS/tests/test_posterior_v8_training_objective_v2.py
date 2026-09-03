from __future__ import annotations

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import TOPOLOGIES
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model import build_proposal_model
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v2 import (
    LOCAL_PROPOSAL_MODEL_NAME,
    LOCAL_PROPOSAL_MODEL_VERSION,
    MODEL_OUTPUT_COORDINATE_SEMANTICS,
    build_local_proposal_model,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_training import (
    load_trained_proposal_model,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective import (
    active_dimension_mask_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective_v2 import (
    LocalTrainingObjectiveConfig,
    compute_local_training_objective,
    normalized_masked_logistic_normal_nll,
)


def test_continuous_nll_is_comparable_across_different_varying_dimension_counts():
    target = tf.fill((2, 26), tf.constant(0.5, tf.float32))
    topology_k1 = next(index for index, value in enumerate(TOPOLOGIES) if len(value) == 1)
    topology_k4 = next(index for index, value in enumerate(TOPOLOGIES) if len(value) == 4)
    varying = np.asarray(
        [
            active_dimension_mask_for(topology_k1, 0),
            active_dimension_mask_for(topology_k4, 0),
        ],
        dtype=np.float32,
    )
    assert varying[0].sum() < varying[1].sum()
    values = normalized_masked_logistic_normal_nll(
        target,
        varying,
        tf.zeros((2, 1), tf.float32),
        tf.zeros((2, 1, 26), tf.float32),
        tf.zeros((2, 1, 26), tf.float32),
    ).numpy()

    assert values[0] == pytest.approx(values[1], rel=1e-6, abs=1e-6)


def _objective_inputs():
    topology_logits = tf.Variable(tf.zeros((1, 34), tf.float32))
    pattern_logits = tf.Variable(tf.zeros((1, 34, 32), tf.float32))
    outputs = {
        "topology_logits": topology_logits,
        "branch_pattern_logits": pattern_logits,
        "mixture_logits": tf.zeros((1, 1), tf.float32),
        "mixture_loc": tf.zeros((1, 1, 26), tf.float32),
        "mixture_logscale": tf.zeros((1, 1, 26), tf.float32),
    }
    active = np.asarray(active_dimension_mask_for(0, 0), dtype=np.float32)[None, :]
    varying = np.zeros_like(active)
    varying[0, np.flatnonzero(active)[0]] = 1.0
    labels = {
        "topology_id": tf.constant([0], tf.int32),
        "branch_pattern_id": tf.constant([0], tf.int32),
        "target_local": tf.fill((1, 26), tf.constant(0.5, tf.float32)),
        "active_dimension_mask": active,
        "varying_dimension_mask": varying,
    }
    return outputs, labels, topology_logits, pattern_logits


def test_continuous_semantics_do_not_change_discrete_head_gradients():
    outputs, labels, topology_logits, pattern_logits = _objective_inputs()
    with tf.GradientTape() as discrete_tape:
        discrete_loss = compute_local_training_objective(
            outputs,
            labels,
            LocalTrainingObjectiveConfig(continuous_weight=0.0),
        )["loss"]
    discrete_gradients = discrete_tape.gradient(
        discrete_loss, (topology_logits, pattern_logits)
    )
    with tf.GradientTape() as joint_tape:
        joint_loss = compute_local_training_objective(
            outputs,
            labels,
            LocalTrainingObjectiveConfig(continuous_weight=10_000.0),
        )["loss"]
    joint_gradients = joint_tape.gradient(joint_loss, (topology_logits, pattern_logits))

    for discrete, joint in zip(discrete_gradients, joint_gradients):
        assert discrete is not None and joint is not None
        assert np.linalg.norm(discrete.numpy()) > 0.0
        np.testing.assert_allclose(joint.numpy(), discrete.numpy(), rtol=0.0, atol=0.0)


def test_local_model_identity_blocks_cross_loading_while_v1_still_loads(tmp_path):
    local = build_local_proposal_model(
        max_points=16, width=4, encoder_blocks=1, mixture_components=2
    )
    assert local.name == LOCAL_PROPOSAL_MODEL_NAME
    assert local.posterior_v8_model_version == LOCAL_PROPOSAL_MODEL_VERSION
    assert local.posterior_v8_output_coordinate_semantics == (
        MODEL_OUTPUT_COORDINATE_SEMANTICS
    )
    local_path = tmp_path / "local.keras"
    local.save(local_path)
    with pytest.raises(ValueError, match="incompatible"):
        load_trained_proposal_model(local_path)

    global_model = build_proposal_model(
        max_points=16, width=4, encoder_blocks=1, mixture_components=2
    )
    global_path = tmp_path / "global.keras"
    global_model.save(global_path)
    loaded = load_trained_proposal_model(global_path)
    assert loaded.name == "posterior_v8_proposal"
