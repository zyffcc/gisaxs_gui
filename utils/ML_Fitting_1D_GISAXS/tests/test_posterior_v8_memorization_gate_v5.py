from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_supervision_v5 import (
    ExactCompatibleProvenance,
    FrozenSearchProvenance,
    KNOWN_TRUTH_ORACLE_PROTOCOL_ID,
    KNOWN_TRUTH_ORACLE_TERMINATION_REASON,
    V5CandidateSupervision,
    stack_candidate_supervision_v5,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.memorization_gate_v5 import (
    V5_MEMORIZATION_GATE_SCHEMA,
    V5MemorizationGateConfig,
    run_v5_memorization_gate,
)


@dataclass(frozen=True)
class _ObjectiveConfig:
    name: str = "toy-local-only"

    def audit_payload(self):
        return {"name": self.name}


class _Dataset:
    def __init__(self):
        active = np.zeros(26, dtype=bool)
        active[:2] = True
        target = np.full(26, 0.5, dtype=np.float32)
        target[:2] = (0.2, 0.8)
        search = FrozenSearchProvenance(
            search_artifact_id="known-truth-one-call",
            search_artifact_sha256="b" * 64,
            protocol_id=KNOWN_TRUTH_ORACLE_PROTOCOL_ID,
            protocol_sha256="c" * 64,
            evaluator_version="exact-forward-v1",
            metric_name="exact_forward_logrmse",
            threshold_name="curve_equivalence",
            threshold_value=0.02,
            threshold_source_id="test",
            exact_forward_call_budget=1,
            exact_forward_calls_used=1,
            termination_reason=KNOWN_TRUTH_ORACLE_TERMINATION_REASON,
            completed=True,
            compatible_representative_count=1,
        )
        exact = ExactCompatibleProvenance(
            artifact_id="known-truth",
            artifact_sha256="a" * 64,
            metric_value=0.0,
            bounds_passed=True,
            physics_passed=True,
        )
        candidate = V5CandidateSupervision(
            clean_recipe_id="recipe",
            candidate_id="candidate",
            outcome="compatible_found",
            active_dimension_mask=active,
            varying_dimension_mask=active,
            search_provenance=search,
            exact_compatible=exact,
            target_local=target,
            generating_candidate_match=True,
        )
        self.labels = stack_candidate_supervision_v5((candidate,), clean_recipe_indices=(0,))

    def joined_numpy(self, *, include_unverified=False):
        assert include_unverified is False
        # Deliberately not a historical V5 input name: the runner discovers it.
        return {"future_context": np.asarray([[1.0, -1.0]], np.float32)}, self.labels


def _model():
    context = tf.keras.Input((2,), name="future_context")
    raw = tf.keras.layers.Dense(
        26,
        kernel_initializer="zeros",
        bias_initializer="zeros",
        name="local_location",
    )(context)
    location = tf.keras.layers.Reshape((1, 26), name="mixture_loc")(raw)
    mixture_logits = tf.keras.layers.Lambda(lambda value: value[:, :1] * 0.0)(raw)
    mixture_logscale = tf.keras.layers.Lambda(lambda value: tf.zeros_like(value))(location)
    score = tf.keras.layers.Lambda(lambda value: tf.reduce_mean(value, axis=-1, keepdims=True))(raw)
    return tf.keras.Model(
        context,
        {
            "proposal_search_yield_logit": score,
            "mixture_logits": mixture_logits,
            "mixture_loc": location,
            "mixture_logscale": mixture_logscale,
        },
    )


def _objective(outputs, labels, config):
    assert isinstance(config, _ObjectiveConfig)
    median = tf.math.sigmoid(outputs["mixture_loc"][:, 0, :])
    target = tf.cast(labels["target_local"], tf.float32)
    varying = tf.cast(labels["varying_dimension_mask"], tf.float32)
    loss = tf.math.divide_no_nan(
        tf.reduce_sum(tf.square(median - target) * varying),
        tf.reduce_sum(varying),
    )
    return {"loss": loss}


def test_memorization_runner_discovers_inputs_and_reports_hashed_gate():
    config = V5MemorizationGateConfig(
        steps=120,
        learning_rate=0.05,
        max_final_target_median_rms=0.02,
        minimum_loss_reduction=0.01,
    )
    _, result = run_v5_memorization_gate(
        _Dataset(),
        _model,
        objective=_objective,
        objective_config=_ObjectiveConfig(),
        config=config,
    )

    assert result.model_input_keys == ("future_context",)
    assert result.target_count == 1
    assert result.varying_coordinate_count == 2
    assert result.initial_target_median_rms > result.final_target_median_rms
    assert result.final_target_median_rms < 0.02
    assert result.initial_loss > result.final_loss
    assert result.passed
    assert len(result.result_sha256) == 64
    assert result.audit_payload()["schema_version"] == V5_MEMORIZATION_GATE_SCHEMA
    assert result.audit_payload()["scientific_role"].endswith("not_model_acceptance")


def test_memorization_gate_config_fails_closed():
    with pytest.raises(ValueError, match="positive"):
        V5MemorizationGateConfig(steps=0)
    with pytest.raises(ValueError, match="non-negative"):
        V5MemorizationGateConfig(minimum_loss_reduction=-1.0)
