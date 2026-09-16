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
    def __init__(self, varying_masks: tuple[np.ndarray, ...] | None = None):
        active = np.zeros(26, dtype=bool)
        active[:2] = True
        masks = (active,) if varying_masks is None else varying_masks
        candidates = []
        for index, varying in enumerate(masks):
            target = np.full(26, 0.5, dtype=np.float32)
            target[varying] = np.asarray((0.2, 0.8), dtype=np.float32)[: np.count_nonzero(varying)]
            search = FrozenSearchProvenance(
                search_artifact_id=f"known-truth-one-call-{index}",
                search_artifact_sha256=f"{index + 1:064x}",
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
                artifact_id=f"known-truth-{index}",
                artifact_sha256=f"{index + 11:064x}",
                metric_value=0.0,
                bounds_passed=True,
                physics_passed=True,
            )
            candidates.append(
                V5CandidateSupervision(
                    clean_recipe_id=f"recipe-{index}",
                    candidate_id=f"candidate-{index}",
                    outcome="compatible_found",
                    active_dimension_mask=active,
                    varying_dimension_mask=varying,
                    search_provenance=search,
                    exact_compatible=exact,
                    target_local=target,
                    generating_candidate_match=True,
                )
            )
        self.labels = stack_candidate_supervision_v5(
            tuple(candidates), clean_recipe_indices=tuple(range(len(candidates)))
        )

    def joined_numpy(self, *, include_unverified=False):
        assert include_unverified is False
        # Deliberately not a historical V5 input name: the runner discovers it.
        count = int(self.labels["target_local"].shape[0])
        context = np.tile(np.asarray([[1.0, -1.0]], np.float32), (count, 1))
        return {"future_context": context}, self.labels


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
    assert result.learnable_target_count == 1
    assert result.fully_fixed_target_count == 0
    assert result.varying_coordinate_count == 2
    assert result.initial_target_median_rms > result.final_target_median_rms
    assert result.final_target_median_rms < 0.02
    assert result.initial_loss > result.final_loss
    assert result.passed
    assert len(result.result_sha256) == 64
    assert result.audit_payload()["schema_version"] == V5_MEMORIZATION_GATE_SCHEMA
    assert result.audit_payload()["scientific_role"].endswith("not_model_acceptance")


def test_memorization_metrics_exclude_valid_fully_fixed_targets():
    varying = np.zeros(26, dtype=bool)
    varying[:2] = True
    fixed = np.zeros(26, dtype=bool)
    _, result = run_v5_memorization_gate(
        _Dataset((varying, fixed)),
        _model,
        objective=_objective,
        objective_config=_ObjectiveConfig(),
        config=V5MemorizationGateConfig(
            steps=120,
            learning_rate=0.05,
            max_final_target_median_rms=0.02,
            minimum_loss_reduction=0.01,
        ),
    )

    assert result.target_count == 2
    assert result.learnable_target_count == 1
    assert result.fully_fixed_target_count == 1
    assert result.varying_coordinate_count == 2
    assert result.passed


def test_memorization_gate_rejects_a_dataset_with_no_learnable_target():
    fixed = np.zeros(26, dtype=bool)
    with pytest.raises(ValueError, match="learnable local target"):
        run_v5_memorization_gate(
            _Dataset((fixed,)),
            _model,
            objective=_objective,
            objective_config=_ObjectiveConfig(),
            config=V5MemorizationGateConfig(steps=1),
        )


def test_memorization_gate_bounds_every_forward_batch():
    varying = np.zeros(26, dtype=bool)
    varying[:2] = True

    def bounded_objective(outputs, labels, config):
        tf.debugging.assert_less_equal(tf.shape(labels["target_local"])[0], 2)
        return _objective(outputs, labels, config)

    _, result = run_v5_memorization_gate(
        _Dataset((varying,) * 5),
        _model,
        objective=bounded_objective,
        objective_config=_ObjectiveConfig(),
        config=V5MemorizationGateConfig(
            steps=5,
            batch_size=2,
            learning_rate=0.05,
            max_final_target_median_rms=1.0,
            minimum_loss_reduction=0.0,
        ),
    )

    assert result.target_count == 5
    assert result.learnable_target_count == 5
    assert result.config.batch_size == 2


def test_memorization_gate_config_fails_closed():
    with pytest.raises(ValueError, match="positive"):
        V5MemorizationGateConfig(steps=0)
    with pytest.raises(ValueError, match="non-negative"):
        V5MemorizationGateConfig(minimum_loss_reduction=-1.0)
    with pytest.raises(ValueError, match="batch_size"):
        V5MemorizationGateConfig(batch_size=0)
    with pytest.raises(ValueError, match="must not exceed"):
        V5MemorizationGateConfig(
            learning_rate=1.0e-5,
            final_learning_rate=2.0e-5,
        )
    with pytest.raises(ValueError, match="learning_rate_schedule"):
        V5MemorizationGateConfig(learning_rate_schedule="plateau")
    with pytest.raises(ValueError, match="constant schedule"):
        V5MemorizationGateConfig(learning_rate_schedule="constant")
    with pytest.raises(ValueError, match="at least one local objective weight"):
        V5MemorizationGateConfig(
            local_mdn_weight=0.0,
            local_coverage_weight=0.0,
            operational_top_l_alignment_weight=0.0,
        )


def test_memorization_gate_default_objective_is_center_aligned_and_audited():
    config = V5MemorizationGateConfig()
    assert config.local_mdn_weight == 1.0
    assert config.local_coverage_weight == 100.0
    assert config.operational_top_l_alignment_weight == 1.0
    assert config.steps == 18000
    assert config.learning_rate == 3.0e-3
    assert config.final_learning_rate == 3.0e-5
    assert config.learning_rate_schedule == "cosine_decay"
