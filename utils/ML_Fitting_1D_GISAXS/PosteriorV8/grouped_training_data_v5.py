"""Fail-closed recipe-macro data adapter for V5.1 grouped training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Sequence

import numpy as np
import tensorflow as tf

from .candidate_supervision_v5 import (
    CANDIDATE_SUPERVISION_TENSOR_KEYS,
    SEARCH_OUTCOME_CODE,
)
from .grouped_amplitude_join_v5 import amplitude_query_from_json
from .grouped_dataset_v5 import (
    V5GroupedDataset,
    candidate_context_array,
    candidate_input,
    candidate_label,
    clean_array,
    observation_array,
    observation_input,
)
from .model_v5_contract import MODEL_V5_INPUT_KEYS
from .proposal_execution_policy_v5 import (
    V5_PROPOSAL_EXECUTION_POLICY,
    V5_PROPOSAL_EXECUTION_POLICY_SHA256,
    validate_v5_proposal_execution_policy_sha256,
)
from .training_objective_v5 import V5CandidateObjectiveConfig


V5_GROUPED_TRAINING_CONFIG_SCHEMA = "gisaxs.posterior_v8.grouped_training_config/v1"
V5_GROUPED_TRAINING_CONFIG_VERSION = (
    "proposal_policy_bound_mass_coverage_operational_top4_v1"
)
V5_GROUPED_TRAINING_SEMANTICS = (
    "uniform_clean_recipe_batches_parent_positive_warmup_sidecar_verified_full_"
    "complete_observation_branch_groups_multirepresentative_coverage_and_pairwise_"
    "ranking_equal_recipe_count_per_replica_with_expansion_safety_receipt_"
    "proposal_policy_bound_operational_top4_v5"
)


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_float(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


@dataclass(frozen=True, kw_only=True)
class V5GroupedTrainingConfig:
    """Immutable scientific and runtime configuration for one exclusive run."""

    warmup_epochs: int = 10
    full_epochs: int = 0
    recipes_per_replica: int = 4
    validation_recipes_per_batch: int = 16
    steps_per_epoch: int | None = None
    full_steps_per_epoch: int | None = None
    learning_rate: float = 1.0e-4
    seed: int = 20260903
    width: int = 128
    encoder_blocks: int = 6
    mixture_components: int = 12
    gradient_clip_norm: float = 10.0
    mixed_precision: bool = False
    deterministic_ops: bool = True
    train_split: str = "train"
    validation_split: str = "tuning_validation"
    search_yield_weight: float = 1.0
    pairwise_ranking_weight: float = 1.0
    local_mdn_weight: float = 1.0
    local_coverage_weight: float = 1.0
    operational_top_l_alignment_weight: float = 1.0
    logistic_epsilon: float = 1.0e-5
    local_coverage_temperature: float = 0.05
    operational_hit_rms_threshold: float = 0.05
    operational_duplicate_rms_threshold: float = 0.02
    proposal_execution_policy_sha256: str = V5_PROPOSAL_EXECUTION_POLICY_SHA256
    allow_same_split_for_smoke: bool = False
    allow_unsafe_sidecar_expansion_for_engineering: bool = False

    def __post_init__(self) -> None:
        for name in ("warmup_epochs", "full_epochs"):
            object.__setattr__(self, name, _nonnegative_integer(getattr(self, name), name))
        if self.warmup_epochs + self.full_epochs < 1:
            raise ValueError("at least one training epoch is required")
        for name in (
            "recipes_per_replica",
            "validation_recipes_per_batch",
            "width",
            "encoder_blocks",
            "mixture_components",
        ):
            object.__setattr__(self, name, _positive_integer(getattr(self, name), name))
        for name in ("steps_per_epoch", "full_steps_per_epoch"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive_integer(value, name))
        if self.mixture_components != V5_PROPOSAL_EXECUTION_POLICY.mixture_component_count:
            raise ValueError(
                "mixture_components must match the frozen proposal execution policy"
            )
        for name in ("learning_rate", "gradient_clip_norm"):
            object.__setattr__(self, name, _positive_float(getattr(self, name), name))
        if isinstance(self.seed, (bool, np.bool_)) or int(self.seed) != self.seed:
            raise TypeError("seed must be an integer")
        if not 0 <= int(self.seed) < 2**31:
            raise ValueError("seed must fit in signed int32")
        object.__setattr__(self, "seed", int(self.seed))
        for name in (
            "mixed_precision",
            "deterministic_ops",
            "allow_same_split_for_smoke",
            "allow_unsafe_sidecar_expansion_for_engineering",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        object.__setattr__(self, "train_split", _text(self.train_split, "train_split"))
        object.__setattr__(
            self,
            "validation_split",
            _text(self.validation_split, "validation_split"),
        )
        if self.train_split == self.validation_split and not self.allow_same_split_for_smoke:
            raise ValueError(
                "train and validation splits must differ outside an explicit smoke run"
            )
        objective = V5CandidateObjectiveConfig(
            search_yield_weight=self.search_yield_weight,
            pairwise_ranking_weight=self.pairwise_ranking_weight,
            local_mdn_weight=self.local_mdn_weight,
            local_coverage_weight=self.local_coverage_weight,
            operational_top_l_alignment_weight=(
                self.operational_top_l_alignment_weight
            ),
            logistic_epsilon=self.logistic_epsilon,
            local_coverage_temperature=self.local_coverage_temperature,
            operational_hit_rms_threshold=self.operational_hit_rms_threshold,
            operational_duplicate_rms_threshold=(
                self.operational_duplicate_rms_threshold
            ),
            proposal_execution_policy_sha256=self.proposal_execution_policy_sha256,
        )
        object.__setattr__(self, "search_yield_weight", objective.search_yield_weight)
        object.__setattr__(
            self, "pairwise_ranking_weight", objective.pairwise_ranking_weight
        )
        object.__setattr__(self, "local_mdn_weight", objective.local_mdn_weight)
        object.__setattr__(
            self, "local_coverage_weight", objective.local_coverage_weight
        )
        object.__setattr__(
            self,
            "operational_top_l_alignment_weight",
            objective.operational_top_l_alignment_weight,
        )
        object.__setattr__(self, "logistic_epsilon", objective.logistic_epsilon)
        object.__setattr__(
            self, "local_coverage_temperature", objective.local_coverage_temperature
        )
        object.__setattr__(
            self,
            "operational_hit_rms_threshold",
            objective.operational_hit_rms_threshold,
        )
        object.__setattr__(
            self,
            "operational_duplicate_rms_threshold",
            objective.operational_duplicate_rms_threshold,
        )
        object.__setattr__(
            self,
            "proposal_execution_policy_sha256",
            validate_v5_proposal_execution_policy_sha256(
                objective.proposal_execution_policy_sha256
            ),
        )
        if (
            self.warmup_epochs > 0
            and objective.local_mdn_weight == 0.0
            and objective.local_coverage_weight == 0.0
            and objective.operational_top_l_alignment_weight == 0.0
        ):
            raise ValueError(
                "warmup requires at least one local objective weight to be positive"
            )

    def audit_payload(self) -> dict[str, object]:
        ranking_requested = bool(
            self.full_epochs > 0 and self.pairwise_ranking_weight > 0.0
        )
        return {
            **asdict(self),
            "schema": V5_GROUPED_TRAINING_CONFIG_SCHEMA,
            "version": V5_GROUPED_TRAINING_CONFIG_VERSION,
            "proposal_execution_policy": V5_PROPOSAL_EXECUTION_POLICY.audit_payload(),
            "ranking_qualification": {
                "warmup_only_eligible": False,
                "configured_full_ranking_phase": ranking_requested,
                "eligible_from_configuration_alone": False,
                "status": (
                    "pending_completed_full_search_supervision"
                    if ranking_requested
                    else "ineligible_without_full_ranking_phase"
                ),
            },
        }


class RecipeMacroAdapter:
    """Materialize complete clean-recipe groups from compact relational tables."""

    def __init__(self, dataset: V5GroupedDataset):
        self.dataset = dataset
        arrays = dataset.arrays
        self._clean_split = arrays[clean_array("split_id")]
        self._observation_recipe = arrays[observation_array("recipe_index")]
        self._candidate_recipe = arrays[candidate_context_array("recipe_index")]
        self._outcome = arrays[candidate_label("search_outcome_code")]
        queries = tuple(
            amplitude_query_from_json(str(encoded), str(digest))
            for encoded, digest in zip(
                arrays[clean_array("amplitude_query_canonical_json")],
                arrays[clean_array("amplitude_query_sha256")],
            )
        )
        references = arrays[observation_array("intensity_reference")]
        self._amplitude_by_observation = np.asarray(
            [
                queries[int(recipe)].model_embedding(float(references[index]))
                for index, recipe in enumerate(self._observation_recipe)
            ],
            dtype=np.float32,
        )

    def recipes(self, split: str) -> np.ndarray:
        result = np.flatnonzero(self._clean_split == split).astype(np.int32)
        if result.size == 0:
            raise ValueError(f"dataset contains no clean recipes for split {split!r}")
        return result

    def outcome_counts(self, recipes: np.ndarray) -> dict[str, int]:
        values = self._outcome[np.isin(self._candidate_recipe, recipes)]
        return {
            name: int(np.count_nonzero(values == code))
            for name, code in SEARCH_OUTCOME_CODE.items()
        }

    def protocol_sha256(self, recipes: np.ndarray) -> str:
        selected = np.isin(self._candidate_recipe, recipes)
        verified = selected & (self._outcome != SEARCH_OUTCOME_CODE["unverified"])
        values = np.unique(self.dataset.arrays[candidate_label("search_protocol_sha256")][verified])
        if values.size != 1:
            raise ValueError("each training split must use exactly one frozen search protocol")
        return str(values[0])

    def numpy_batch(
        self,
        recipes: Sequence[int],
        *,
        phase: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        observed_parts: list[np.ndarray] = []
        candidate_parts: list[np.ndarray] = []
        for recipe in recipes:
            observations = np.flatnonzero(self._observation_recipe == int(recipe))
            candidates = np.flatnonzero(self._candidate_recipe == int(recipe))
            if phase == "warmup":
                candidates = candidates[
                    self._outcome[candidates] == SEARCH_OUTCOME_CODE["compatible_found"]
                ]
            elif phase != "full":
                raise ValueError("phase must be warmup or full")
            if observations.size == 0 or candidates.size == 0:
                raise ValueError("recipe batch has no observations or eligible candidates")
            observed_parts.append(np.repeat(observations, candidates.size))
            candidate_parts.append(np.tile(candidates, observations.size))
        observed = np.concatenate(observed_parts).astype(np.int32, copy=False)
        candidates = np.concatenate(candidate_parts).astype(np.int32, copy=False)
        arrays = self.dataset.arrays
        inputs: dict[str, np.ndarray] = {}
        for name in MODEL_V5_INPUT_KEYS:
            if name == "amplitude_bounds_embedding":
                inputs[name] = self._amplitude_by_observation[observed]
            elif observation_input(name) in arrays:
                inputs[name] = arrays[observation_input(name)][observed]
            else:
                inputs[name] = arrays[candidate_input(name)][candidates]
        labels = {
            name: arrays[candidate_label(name)][candidates]
            for name in CANDIDATE_SUPERVISION_TENSOR_KEYS
        }
        return inputs, labels

    def tensor_batch(self, recipes: Sequence[int], *, phase: str):
        inputs, labels = self.numpy_batch(recipes, phase=phase)
        return (
            {name: tf.convert_to_tensor(value) for name, value in inputs.items()},
            {name: tf.convert_to_tensor(value) for name, value in labels.items()},
        )


__all__ = [
    "RecipeMacroAdapter",
    "V5_GROUPED_TRAINING_CONFIG_SCHEMA",
    "V5_GROUPED_TRAINING_CONFIG_VERSION",
    "V5_GROUPED_TRAINING_SEMANTICS",
    "V5GroupedTrainingConfig",
]
