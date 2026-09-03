"""Three-state, recipe-balanced training objective for V5 proposals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite
from typing import Mapping

import tensorflow as tf

from .candidate_supervision_v5 import (
    CANDIDATE_SUPERVISION_TENSOR_KEYS,
    CANDIDATE_SUPERVISION_V5_SCHEMA,
    CANDIDATE_SUPERVISION_V5_VERSION,
    KNOWN_TRUTH_ORACLE_PROTOCOL_ID,
    KNOWN_TRUTH_ORACLE_TERMINATION_REASON,
    NEGATIVE_TERMINATION_REASONS,
    POSITIVE_TERMINATION_REASON,
    RECIPE_WEIGHTING_SEMANTICS,
    SEARCH_OUTCOME_CODE,
)
from .model import BRANCH_DIM
from .model_v5_contract import MODEL_V5_OUTPUT_KEYS
from .proposal_execution_policy_v5 import (
    V5_PROPOSAL_EXECUTION_POLICY,
    V5_PROPOSAL_EXECUTION_POLICY_SHA256,
    validate_v5_proposal_execution_policy_sha256,
)
from .training_objective_v2 import normalized_masked_logistic_normal_nll


CANDIDATE_TRAINING_OBJECTIVE_V5_SCHEMA = "gisaxs.posterior_v8.candidate_training_objective/v7"
CANDIDATE_TRAINING_OBJECTIVE_V5_VERSION = (
    "posterior_v8_recipe_pairwise_yield_local_mdn_mass_coverage_soft_rank_top4_v5"
)
MISSING_CLASS_SEMANTICS = (
    "head_with_zero_eligible_examples_contributes_exact_zero_and_reports_zero_counts_v1"
)
PAIRWISE_RANKING_SEMANTICS = (
    "all_completed_positive_negative_pairs_within_each_clean_recipe_weighted_"
    "class_normalized_then_recipe_macro_logistic_v1"
)
LOCAL_COVERAGE_SEMANTICS = (
    "negative_temperature_log_mixture_mass_of_masked_local_median_rms_"
    "then_recipe_macro_hard_best_of_m_is_diagnostic_only_v2"
)
LOCAL_HARD_COVERAGE_BASELINE_SEMANTICS = (
    "hard_best_of_all_mixture_medians_masked_local_rms_diagnostic_only_v1"
)
OPERATIONAL_TOP_L_METRICS_SEMANTICS = (
    "mixtures_ranked_by_logit_stable_tie_index_then_first_policy_top_l_medians_"
    "report_best_rms_target_recall_miss_censored_hit_rank_mass_entropy_effective_"
    "utilization_and_earlier_rank_duplicate_fraction_v1"
)
OPERATIONAL_TOP_L_ALIGNMENT_SEMANTICS = (
    "target_distance_softmax_weighted_pairwise_sigmoid_soft_rank_with_smooth_"
    "penalty_above_policy_top_l_plus_half_enters_training_loss_v1"
)
LOCAL_COVERAGE_RMS_STABILITY_EPSILON = 1.0e-12


def _weight(value: float, name: str) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


@dataclass(frozen=True)
class V5CandidateObjectiveConfig:
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

    def __post_init__(self) -> None:
        search_yield = _weight(self.search_yield_weight, "search_yield_weight")
        pairwise_ranking = _weight(
            self.pairwise_ranking_weight, "pairwise_ranking_weight"
        )
        local_mdn = _weight(self.local_mdn_weight, "local_mdn_weight")
        local_coverage = _weight(
            self.local_coverage_weight, "local_coverage_weight"
        )
        operational_top_l_alignment = _weight(
            self.operational_top_l_alignment_weight,
            "operational_top_l_alignment_weight",
        )
        if not any(
            value > 0.0
            for value in (
                search_yield,
                pairwise_ranking,
                local_mdn,
                local_coverage,
                operational_top_l_alignment,
            )
        ):
            raise ValueError("at least one objective weight must be positive")
        epsilon = float(self.logistic_epsilon)
        if not isfinite(epsilon) or not 0.0 < epsilon < 0.5:
            raise ValueError("logistic_epsilon must be finite and in (0, 0.5)")
        for name in (
            "local_coverage_temperature",
            "operational_hit_rms_threshold",
            "operational_duplicate_rms_threshold",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
            object.__setattr__(self, name, value)
        policy_sha256 = validate_v5_proposal_execution_policy_sha256(
            self.proposal_execution_policy_sha256
        )
        object.__setattr__(self, "search_yield_weight", search_yield)
        object.__setattr__(self, "pairwise_ranking_weight", pairwise_ranking)
        object.__setattr__(self, "local_mdn_weight", local_mdn)
        object.__setattr__(self, "local_coverage_weight", local_coverage)
        object.__setattr__(
            self,
            "operational_top_l_alignment_weight",
            operational_top_l_alignment,
        )
        object.__setattr__(self, "logistic_epsilon", epsilon)
        object.__setattr__(self, "proposal_execution_policy_sha256", policy_sha256)

    def audit_payload(self) -> dict[str, object]:
        return {
            "schema_version": CANDIDATE_TRAINING_OBJECTIVE_V5_SCHEMA,
            "version": CANDIDATE_TRAINING_OBJECTIVE_V5_VERSION,
            "candidate_supervision_schema": CANDIDATE_SUPERVISION_V5_SCHEMA,
            "candidate_supervision_version": CANDIDATE_SUPERVISION_V5_VERSION,
            "recipe_weighting_semantics": RECIPE_WEIGHTING_SEMANTICS,
            "missing_class_semantics": MISSING_CLASS_SEMANTICS,
            "pairwise_ranking_semantics": PAIRWISE_RANKING_SEMANTICS,
            "local_coverage_semantics": LOCAL_COVERAGE_SEMANTICS,
            "local_hard_coverage_baseline_semantics": (
                LOCAL_HARD_COVERAGE_BASELINE_SEMANTICS
            ),
            "operational_top_l_metrics_semantics": (
                OPERATIONAL_TOP_L_METRICS_SEMANTICS
            ),
            "operational_top_l_alignment_semantics": (
                OPERATIONAL_TOP_L_ALIGNMENT_SEMANTICS
            ),
            "proposal_execution_policy": V5_PROPOSAL_EXECUTION_POLICY.audit_payload(),
            "proposal_execution_policy_sha256": self.proposal_execution_policy_sha256,
            "config": asdict(self),
        }


def _required_tensors(mapping, keys, kind: str) -> dict[str, tf.Tensor]:
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{kind} must be a mapping")
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"{kind} is missing required values: {missing}")
    return {key: tf.convert_to_tensor(mapping[key]) for key in keys}


def _integer_vector(value: tf.Tensor, name: str) -> tf.Tensor:
    if value.dtype == tf.bool or not value.dtype.is_integer:
        raise TypeError(f"{name} must use an integer tensor dtype")
    rank = tf.debugging.assert_rank(value, 1, message=f"{name} must be rank one")
    with tf.control_dependencies((rank,)):
        return tf.cast(tf.identity(value), tf.int32)


def _boolean_vector(value: tf.Tensor, name: str) -> tf.Tensor:
    if value.dtype != tf.bool:
        raise TypeError(f"{name} must use a boolean tensor dtype")
    rank = tf.debugging.assert_rank(value, 1, message=f"{name} must be rank one")
    with tf.control_dependencies((rank,)):
        return tf.identity(value)


def _string_vector(value: tf.Tensor, name: str) -> tf.Tensor:
    if value.dtype != tf.string:
        raise TypeError(f"{name} must use a string tensor dtype")
    rank = tf.debugging.assert_rank(value, 1, message=f"{name} must be rank one")
    with tf.control_dependencies((rank,)):
        return tf.identity(value)


def _validated_outputs(outputs: Mapping[str, object]):
    tensors = _required_tensors(outputs, MODEL_V5_OUTPUT_KEYS, "outputs")
    score = tf.cast(tensors["proposal_search_yield_logit"], tf.float32)
    mixture_logits = tf.cast(tensors["mixture_logits"], tf.float32)
    mixture_loc = tf.cast(tensors["mixture_loc"], tf.float32)
    mixture_logscale = tf.cast(tensors["mixture_logscale"], tf.float32)
    batch = tf.shape(score)[0]
    modes = tf.shape(mixture_logits)[1]
    assertions = (
        tf.debugging.assert_rank(score, 2),
        tf.debugging.assert_rank(mixture_logits, 2),
        tf.debugging.assert_rank(mixture_loc, 3),
        tf.debugging.assert_rank(mixture_logscale, 3),
        tf.debugging.assert_positive(batch, message="candidate batch must be non-empty"),
        tf.debugging.assert_equal(tf.shape(score), [batch, 1]),
        tf.debugging.assert_equal(tf.shape(mixture_logits)[0], batch),
        tf.debugging.assert_positive(modes, message="at least one mixture mode is required"),
        tf.debugging.assert_equal(tf.shape(mixture_loc), [batch, modes, BRANCH_DIM]),
        tf.debugging.assert_equal(tf.shape(mixture_logscale), [batch, modes, BRANCH_DIM]),
        tf.debugging.assert_all_finite(score, "proposal_search_yield_logit contains NaN/Inf"),
        tf.debugging.assert_all_finite(mixture_logits, "mixture_logits contains NaN/Inf"),
        tf.debugging.assert_all_finite(mixture_loc, "mixture_loc contains NaN/Inf"),
        tf.debugging.assert_all_finite(mixture_logscale, "mixture_logscale contains NaN/Inf"),
    )
    with tf.control_dependencies(assertions):
        return tuple(
            tf.identity(value)
            for value in (score[:, 0], mixture_logits, mixture_loc, mixture_logscale)
        )


def _validated_labels(labels: Mapping[str, object], batch: tf.Tensor):
    values = _required_tensors(labels, CANDIDATE_SUPERVISION_TENSOR_KEYS, "labels")
    outcome = _integer_vector(values["search_outcome_code"], "search_outcome_code")
    recipe_index = _integer_vector(values["clean_recipe_index"], "clean_recipe_index")
    call_budget = _integer_vector(
        values["search_exact_forward_call_budget"],
        "search_exact_forward_call_budget",
    )
    calls_used = _integer_vector(
        values["search_exact_forward_calls_used"],
        "search_exact_forward_calls_used",
    )
    compatible_count = _integer_vector(
        values["search_compatible_representative_count"],
        "search_compatible_representative_count",
    )
    has_target = _boolean_vector(values["has_local_target"], "has_local_target")
    bounds_passed = _boolean_vector(values["exact_bounds_passed"], "exact_bounds_passed")
    physics_passed = _boolean_vector(values["exact_physics_passed"], "exact_physics_passed")
    completed = _boolean_vector(values["search_completed"], "search_completed")
    exact_string_names = ("exact_artifact_id", "exact_artifact_sha256")
    search_string_names = (
        "search_artifact_id",
        "search_artifact_sha256",
        "search_protocol_id",
        "search_protocol_sha256",
        "search_evaluator_version",
        "search_metric_name",
        "search_threshold_name",
        "search_threshold_source_id",
        "search_termination_reason",
    )
    string_names = exact_string_names + search_string_names
    strings = {name: _string_vector(values[name], name) for name in string_names}
    metric = tf.cast(values["exact_metric_value"], tf.float32)
    threshold = tf.cast(values["search_threshold_value"], tf.float32)
    target = tf.cast(values["target_local"], tf.float32)
    active = tf.cast(values["active_dimension_mask"], tf.float32)
    varying = tf.cast(values["varying_dimension_mask"], tf.float32)
    recipe_weight = tf.cast(values["clean_recipe_weight"], tf.float32)
    candidate_weight = tf.cast(values["candidate_weight"], tf.float32)
    vectors = (
        outcome,
        recipe_index,
        call_budget,
        calls_used,
        compatible_count,
        has_target,
        bounds_passed,
        physics_passed,
        completed,
        metric,
        threshold,
        recipe_weight,
        candidate_weight,
        *strings.values(),
    )
    assertions = [
        *(tf.debugging.assert_equal(tf.shape(value), [batch]) for value in vectors),
        tf.debugging.assert_equal(tf.shape(target), [batch, BRANCH_DIM]),
        tf.debugging.assert_equal(tf.shape(active), [batch, BRANCH_DIM]),
        tf.debugging.assert_equal(tf.shape(varying), [batch, BRANCH_DIM]),
        tf.debugging.assert_greater_equal(outcome, 0),
        tf.debugging.assert_less_equal(outcome, 2),
        tf.debugging.assert_greater_equal(recipe_index, 0),
        tf.debugging.assert_greater_equal(call_budget, 0),
        tf.debugging.assert_greater_equal(calls_used, 0),
        tf.debugging.assert_greater_equal(compatible_count, 0),
        tf.debugging.assert_all_finite(metric, "exact metric contains NaN/Inf"),
        tf.debugging.assert_all_finite(threshold, "search threshold contains NaN/Inf"),
        tf.debugging.assert_greater_equal(metric, 0.0),
        tf.debugging.assert_greater_equal(threshold, 0.0),
        tf.debugging.assert_all_finite(target, "target_local contains NaN/Inf"),
        tf.debugging.assert_greater_equal(target, 0.0),
        tf.debugging.assert_less_equal(target, 1.0),
        tf.debugging.assert_all_finite(active, "active_dimension_mask contains NaN/Inf"),
        tf.debugging.assert_all_finite(varying, "varying_dimension_mask contains NaN/Inf"),
        tf.debugging.assert_equal(active, tf.round(active), message="active mask must be binary"),
        tf.debugging.assert_equal(
            varying, tf.round(varying), message="varying mask must be binary"
        ),
        tf.debugging.assert_greater_equal(active, 0.0),
        tf.debugging.assert_less_equal(active, 1.0),
        tf.debugging.assert_greater_equal(varying, 0.0),
        tf.debugging.assert_less_equal(varying, 1.0),
        tf.debugging.assert_all_finite(recipe_weight, "clean_recipe_weight contains NaN/Inf"),
        tf.debugging.assert_all_finite(candidate_weight, "candidate_weight contains NaN/Inf"),
        tf.debugging.assert_positive(recipe_weight),
        tf.debugging.assert_positive(candidate_weight),
    ]
    with tf.control_dependencies(assertions):
        outcome = tf.identity(outcome)
        recipe_index = tf.identity(recipe_index)
        call_budget = tf.identity(call_budget)
        calls_used = tf.identity(calls_used)
        compatible_count = tf.identity(compatible_count)
        has_target = tf.identity(has_target)
        bounds_passed = tf.identity(bounds_passed)
        physics_passed = tf.identity(physics_passed)
        completed = tf.identity(completed)
        metric = tf.identity(metric)
        threshold = tf.identity(threshold)
        target = tf.identity(target)
        active = tf.identity(active)
        varying = tf.identity(varying)
        recipe_weight = tf.identity(recipe_weight)
        candidate_weight = tf.identity(candidate_weight)
        strings = {name: tf.identity(value) for name, value in strings.items()}

    labeled = outcome != SEARCH_OUTCOME_CODE["unverified"]
    compatible = outcome == SEARCH_OUTCOME_CODE["compatible_found"]
    negative = outcome == SEARCH_OUTCOME_CODE["no_compatible_found_within_frozen_search_budget"]
    unverified = tf.logical_not(labeled)
    known_truth_oracle = tf.equal(
        strings["search_protocol_id"], KNOWN_TRUTH_ORACLE_PROTOCOL_ID
    )
    verified = tf.logical_and(labeled, tf.logical_not(known_truth_oracle))
    exact_nonempty = tf.stack(
        [tf.strings.length(strings[name]) > 0 for name in exact_string_names], axis=-1
    )
    search_nonempty = tf.stack(
        [tf.strings.length(strings[name]) > 0 for name in search_string_names], axis=-1
    )
    exact_all_nonempty = tf.reduce_all(exact_nonempty, axis=-1)
    exact_all_empty = tf.reduce_all(tf.logical_not(exact_nonempty), axis=-1)
    search_all_nonempty = tf.reduce_all(search_nonempty, axis=-1)
    search_all_empty = tf.reduce_all(tf.logical_not(search_nonempty), axis=-1)
    exact_digest = tf.strings.regex_full_match(strings["exact_artifact_sha256"], "[0-9a-f]{64}")
    search_digests = tf.logical_and(
        tf.strings.regex_full_match(strings["search_artifact_sha256"], "[0-9a-f]{64}"),
        tf.strings.regex_full_match(strings["search_protocol_sha256"], "[0-9a-f]{64}"),
    )
    derived_compatible = tf.logical_and(
        metric <= threshold,
        tf.logical_and(bounds_passed, physics_passed),
    )
    negative_reason = tf.reduce_any(
        tf.equal(
            strings["search_termination_reason"][:, None],
            tf.constant(NEGATIVE_TERMINATION_REASONS)[None, :],
        ),
        axis=-1,
    )
    positive_reason = tf.equal(
        strings["search_termination_reason"], POSITIVE_TERMINATION_REASON
    )
    canonical = tf.logical_or(varying < 0.5, tf.logical_not(has_target)[:, None])
    canonical_targets = tf.boolean_mask(target, canonical)
    verified_protocols = tf.boolean_mask(strings["search_protocol_sha256"], verified)
    known_truth_contract = tf.logical_and(
        compatible,
        tf.logical_and(
            tf.equal(call_budget, 1),
            tf.logical_and(
                tf.equal(calls_used, 1),
                tf.logical_and(
                    tf.equal(compatible_count, 1),
                    tf.equal(
                        strings["search_termination_reason"],
                        KNOWN_TRUTH_ORACLE_TERMINATION_REASON,
                    ),
                ),
            ),
        ),
    )

    unique_recipe, recipe_group = tf.unique(recipe_index)
    recipe_count = tf.size(unique_recipe)
    min_recipe_weight = tf.math.unsorted_segment_min(recipe_weight, recipe_group, recipe_count)
    max_recipe_weight = tf.math.unsorted_segment_max(recipe_weight, recipe_group, recipe_count)
    semantic_assertions = (
        tf.debugging.assert_less_equal(
            varying, active, message="varying dimensions must be active"
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(search_all_nonempty, labeled),
            tf.ones_like(tf.boolean_mask(search_all_nonempty, labeled)),
            message="verified outcomes require complete frozen-search provenance",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(search_digests, labeled),
            tf.ones_like(tf.boolean_mask(search_digests, labeled)),
            message="verified outcomes require lowercase search/protocol SHA256 digests",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(search_all_empty, unverified),
            tf.ones_like(tf.boolean_mask(search_all_empty, unverified)),
            message="unverified outcomes must not carry completed-search provenance",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(exact_all_nonempty, compatible),
            tf.ones_like(tf.boolean_mask(exact_all_nonempty, compatible)),
            message="compatible-found labels require exact artifact provenance",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(exact_digest, compatible),
            tf.ones_like(tf.boolean_mask(exact_digest, compatible)),
            message="compatible-found labels require a lowercase exact artifact SHA256",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(exact_all_empty, tf.logical_not(compatible)),
            tf.ones_like(tf.boolean_mask(exact_all_empty, tf.logical_not(compatible))),
            message="non-positive outcomes must not carry exact-compatible artifacts",
        ),
        tf.debugging.assert_equal(tf.boolean_mask(metric, unverified), 0.0),
        tf.debugging.assert_equal(tf.boolean_mask(threshold, unverified), 0.0),
        tf.debugging.assert_equal(
            tf.boolean_mask(bounds_passed, unverified),
            tf.zeros_like(tf.boolean_mask(bounds_passed, unverified)),
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(physics_passed, unverified),
            tf.zeros_like(tf.boolean_mask(physics_passed, unverified)),
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(derived_compatible, compatible),
            tf.ones_like(tf.boolean_mask(derived_compatible, compatible)),
            message="positive outcome disagrees with exact compatibility evidence",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(metric, tf.logical_not(compatible)),
            tf.zeros_like(tf.boolean_mask(metric, tf.logical_not(compatible))),
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(bounds_passed, tf.logical_not(compatible)),
            tf.zeros_like(tf.boolean_mask(bounds_passed, tf.logical_not(compatible))),
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(physics_passed, tf.logical_not(compatible)),
            tf.zeros_like(tf.boolean_mask(physics_passed, tf.logical_not(compatible))),
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(completed, labeled),
            tf.ones_like(tf.boolean_mask(completed, labeled)),
            message="verified outcomes require a completed frozen search",
        ),
        tf.debugging.assert_positive(tf.boolean_mask(call_budget, labeled)),
        tf.debugging.assert_equal(
            tf.boolean_mask(calls_used, verified),
            tf.boolean_mask(call_budget, verified),
            message="verified outcomes must consume the equal frozen branch budget",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(call_budget, unverified),
            tf.zeros_like(tf.boolean_mask(call_budget, unverified)),
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(calls_used, unverified),
            tf.zeros_like(tf.boolean_mask(calls_used, unverified)),
        ),
        tf.debugging.assert_positive(
            tf.boolean_mask(compatible_count, compatible),
            message="compatible-found search must report at least one representative",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(compatible_count, tf.logical_not(compatible)),
            tf.zeros_like(tf.boolean_mask(compatible_count, tf.logical_not(compatible))),
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(negative_reason, negative),
            tf.ones_like(tf.boolean_mask(negative_reason, negative)),
            message="negative outcome requires frozen-protocol exhaustion provenance",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(
                positive_reason,
                tf.logical_and(compatible, tf.logical_not(known_truth_oracle)),
            ),
            tf.ones_like(
                tf.boolean_mask(
                    positive_reason,
                    tf.logical_and(compatible, tf.logical_not(known_truth_oracle)),
                )
            ),
            message="positive outcome requires the canonical full-budget termination",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(known_truth_contract, known_truth_oracle),
            tf.ones_like(tf.boolean_mask(known_truth_contract, known_truth_oracle)),
            message="known-truth oracle evidence violates its one-call warmup contract",
        ),
        tf.debugging.assert_less_equal(
            tf.size(tf.unique(verified_protocols)[0]),
            1,
            message="one training batch must use one frozen search protocol",
        ),
        tf.debugging.assert_equal(
            tf.boolean_mask(has_target, tf.logical_not(compatible)),
            tf.zeros_like(tf.boolean_mask(has_target, tf.logical_not(compatible))),
            message="only exact-compatible labels may carry local targets",
        ),
        tf.debugging.assert_equal(
            canonical_targets,
            tf.fill(tf.shape(canonical_targets), tf.constant(0.5, tf.float32)),
            message="fixed, inactive, and missing target coordinates must equal 0.5",
        ),
        tf.debugging.assert_equal(
            min_recipe_weight,
            max_recipe_weight,
            message="clean_recipe_weight must be constant within a clean recipe",
        ),
    )
    with tf.control_dependencies(semantic_assertions):
        return {
            "outcome": tf.identity(outcome),
            "recipe_index": tf.identity(recipe_index),
            "has_target": tf.identity(has_target),
            "target": tf.identity(target),
            "active": tf.identity(active),
            "varying": tf.identity(varying),
            "recipe_weight": tf.identity(recipe_weight),
            "candidate_weight": tf.identity(candidate_weight),
            "verified": tf.identity(verified),
            "compatible": tf.identity(compatible),
            "negative": tf.identity(negative),
            "unverified": tf.identity(unverified),
        }


def _recipe_macro_mean(
    values: tf.Tensor,
    indices: tf.Tensor,
    *,
    recipe_index: tf.Tensor,
    recipe_weight: tf.Tensor,
    candidate_weight: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    selected_recipe = tf.gather(recipe_index, indices)
    selected_recipe_weight = tf.gather(recipe_weight, indices)
    selected_candidate_weight = tf.gather(candidate_weight, indices)
    unique_recipe, group = tf.unique(selected_recipe)
    recipe_count = tf.size(unique_recipe)
    candidate_sum = tf.math.unsorted_segment_sum(selected_candidate_weight, group, recipe_count)
    normalized_candidate = tf.math.divide_no_nan(
        selected_candidate_weight, tf.gather(candidate_sum, group)
    )
    effective = normalized_candidate * selected_recipe_weight
    denominator = tf.reduce_sum(effective)
    mean = tf.math.divide_no_nan(tf.reduce_sum(values * effective), denominator)
    return mean, recipe_count, denominator


def _recipe_pairwise_ranking(
    score: tf.Tensor,
    label: Mapping[str, tf.Tensor],
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Compare every completed positive/negative pair inside one clean recipe."""

    positive = tf.logical_and(label["verified"], label["compatible"])
    negative = tf.logical_and(label["verified"], label["negative"])
    same_recipe = tf.equal(
        label["recipe_index"][:, None], label["recipe_index"][None, :]
    )
    pair_mask = tf.logical_and(
        same_recipe,
        tf.logical_and(positive[:, None], negative[None, :]),
    )
    pair_indices = tf.cast(tf.where(pair_mask), tf.int32)
    positive_indices = pair_indices[:, 0]
    negative_indices = pair_indices[:, 1]
    positive_score = tf.gather(score, positive_indices)
    negative_score = tf.gather(score, negative_indices)
    per_pair = tf.nn.softplus(negative_score - positive_score)

    pair_recipe = tf.gather(label["recipe_index"], positive_indices)
    pair_recipe_weight = tf.gather(label["recipe_weight"], positive_indices)
    raw_pair_weight = tf.gather(
        label["candidate_weight"], positive_indices
    ) * tf.gather(label["candidate_weight"], negative_indices)
    unique_recipe, recipe_group = tf.unique(pair_recipe)
    recipe_count = tf.size(unique_recipe)
    within_recipe_sum = tf.math.unsorted_segment_sum(
        raw_pair_weight, recipe_group, recipe_count
    )
    normalized_pair_weight = tf.math.divide_no_nan(
        raw_pair_weight, tf.gather(within_recipe_sum, recipe_group)
    )
    effective = normalized_pair_weight * pair_recipe_weight
    denominator = tf.reduce_sum(effective)
    loss = tf.math.divide_no_nan(tf.reduce_sum(per_pair * effective), denominator)
    accuracy = tf.math.divide_no_nan(
        tf.reduce_sum(
            tf.cast(positive_score > negative_score, tf.float32) * effective
        ),
        denominator,
    )
    return loss, accuracy, tf.shape(pair_indices)[0], recipe_count, denominator


def _mixture_median_masked_rms(
    target: tf.Tensor,
    varying: tf.Tensor,
    mixture_loc: tf.Tensor,
) -> tf.Tensor:
    """Return stable masked RMS for every target-row/mixture pair."""

    median = tf.math.sigmoid(mixture_loc)
    mask = varying[:, None, :]
    squared = tf.square(median - target[:, None, :]) * mask
    varying_count = tf.reduce_sum(mask, axis=-1)
    mean_squared = tf.math.divide_no_nan(
        tf.reduce_sum(squared, axis=-1), varying_count
    )
    epsilon = tf.constant(LOCAL_COVERAGE_RMS_STABILITY_EPSILON, tf.float32)
    return tf.sqrt(mean_squared + epsilon) - tf.sqrt(epsilon)


def _mixture_median_best_of_m_masked_rms(
    target: tf.Tensor,
    varying: tf.Tensor,
    mixture_loc: tf.Tensor,
) -> tf.Tensor:
    """Return the legacy hard-best diagnostic for each expanded target row."""

    return tf.reduce_min(
        _mixture_median_masked_rms(target, varying, mixture_loc), axis=-1
    )


def _mixture_mass_aware_coverage(
    component_rms: tf.Tensor,
    mixture_logits: tf.Tensor,
    *,
    temperature: float,
) -> tf.Tensor:
    """Soft coverage that charges a target when its nearby mode has little mass."""

    scale = tf.constant(float(temperature), tf.float32)
    log_mass = tf.nn.log_softmax(mixture_logits, axis=-1)
    return -scale * tf.reduce_logsumexp(
        log_mass - component_rms / scale,
        axis=-1,
    )


def _mixture_operational_top_l_alignment(
    component_rms: tf.Tensor,
    mixture_logits: tf.Tensor,
    *,
    distance_temperature: float,
) -> tf.Tensor:
    """Penalize a target's differentiable covering-mixture rank above policy L."""

    distance_scale = tf.constant(float(distance_temperature), tf.float32)
    target_affinity = tf.nn.softmax(-component_rms / distance_scale, axis=-1)
    logit_scale = tf.constant(
        V5_PROPOSAL_EXECUTION_POLICY.soft_rank_logit_temperature, tf.float32
    )
    # Row i estimates how many other mixture logits outrank mixture i. The
    # diagonal is removed explicitly so an exact tie has rank 1 + (M - 1) / 2.
    pairwise_higher = tf.math.sigmoid(
        (mixture_logits[:, None, :] - mixture_logits[:, :, None]) / logit_scale
    )
    mixture_count = tf.shape(mixture_logits)[1]
    off_diagonal = 1.0 - tf.eye(mixture_count, dtype=tf.float32)[None, :, :]
    soft_rank = 1.0 + tf.reduce_sum(pairwise_higher * off_diagonal, axis=-1)
    target_soft_rank = tf.reduce_sum(target_affinity * soft_rank, axis=-1)

    top_l = tf.cast(V5_PROPOSAL_EXECUTION_POLICY.per_branch_top_l, tf.float32)
    cutoff = top_l + 0.5
    cutoff_scale = tf.constant(
        V5_PROPOSAL_EXECUTION_POLICY.soft_rank_cutoff_temperature, tf.float32
    )
    smooth_excess = cutoff_scale * tf.nn.softplus(
        (target_soft_rank - cutoff) / cutoff_scale
    )
    available_excess = tf.maximum(
        tf.cast(mixture_count, tf.float32) - top_l,
        1.0,
    )
    return smooth_excess / available_excess


def _operational_top_l_metrics(
    component_rms: tf.Tensor,
    varying: tf.Tensor,
    mixture_logits: tf.Tensor,
    mixture_loc: tf.Tensor,
    *,
    hit_rms_threshold: float,
    duplicate_rms_threshold: float,
) -> dict[str, tf.Tensor]:
    """Return per-target diagnostics for the mixtures inference will retain."""

    order = tf.argsort(
        mixture_logits, axis=-1, direction="DESCENDING", stable=True
    )[:, : V5_PROPOSAL_EXECUTION_POLICY.per_branch_top_l]
    top_rms = tf.gather(component_rms, order, axis=1, batch_dims=1)
    top_mass = tf.gather(
        tf.nn.softmax(mixture_logits, axis=-1), order, axis=1, batch_dims=1
    )
    best_rms = tf.reduce_min(top_rms, axis=-1)
    hits = top_rms <= tf.constant(hit_rms_threshold, tf.float32)
    any_hit = tf.reduce_any(hits, axis=-1)
    first_hit_rank = (
        tf.argmax(tf.cast(hits, tf.int32), axis=-1, output_type=tf.int32) + 1
    )
    selected_count = tf.shape(top_rms)[1]
    miss_censored_rank = tf.where(
        any_hit,
        first_hit_rank,
        tf.fill(tf.shape(first_hit_rank), selected_count + 1),
    )

    log_mass = tf.nn.log_softmax(mixture_logits, axis=-1)
    mass = tf.exp(log_mass)
    entropy = -tf.reduce_sum(mass * log_mass, axis=-1)
    effective_components = tf.exp(entropy)
    component_count = tf.cast(tf.shape(mixture_logits)[1], tf.float32)

    top_median = tf.gather(
        tf.math.sigmoid(mixture_loc), order, axis=1, batch_dims=1
    )
    difference = top_median[:, :, None, :] - top_median[:, None, :, :]
    mask = varying[:, None, None, :]
    pair_mean_squared = tf.math.divide_no_nan(
        tf.reduce_sum(tf.square(difference) * mask, axis=-1),
        tf.reduce_sum(mask, axis=-1),
    )
    epsilon = tf.constant(LOCAL_COVERAGE_RMS_STABILITY_EPSILON, tf.float32)
    pair_rms = tf.sqrt(pair_mean_squared + epsilon) - tf.sqrt(epsilon)
    earlier = tf.linalg.band_part(
        tf.ones((selected_count, selected_count), dtype=tf.bool), -1, 0
    )
    earlier = tf.logical_and(
        earlier, tf.logical_not(tf.eye(selected_count, dtype=tf.bool))
    )
    is_duplicate = tf.reduce_any(
        tf.logical_and(
            pair_rms <= tf.constant(duplicate_rms_threshold, tf.float32),
            earlier[None, :, :],
        ),
        axis=-1,
    )
    duplicate_fraction = tf.math.divide_no_nan(
        tf.reduce_sum(tf.cast(is_duplicate, tf.float32), axis=-1),
        tf.cast(selected_count, tf.float32),
    )
    return {
        "best_rms": best_rms,
        "target_recall": tf.cast(any_hit, tf.float32),
        "hit_rank": tf.cast(miss_censored_rank, tf.float32),
        "top_l_mass": tf.reduce_sum(top_mass, axis=-1),
        "entropy": entropy,
        "effective_components": effective_components,
        "utilization": tf.math.divide_no_nan(effective_components, component_count),
        "duplicate_fraction": duplicate_fraction,
    }


def compute_v5_candidate_training_objective(
    outputs: Mapping[str, object],
    labels: Mapping[str, object],
    config: V5CandidateObjectiveConfig = V5CandidateObjectiveConfig(),
) -> dict[str, tf.Tensor]:
    """Train frozen-search yield only from completed outcomes and MDN from positives."""

    if not isinstance(config, V5CandidateObjectiveConfig):
        raise TypeError("config must be a V5CandidateObjectiveConfig")
    score, mixture_logits, mixture_loc, mixture_logscale = _validated_outputs(outputs)
    label = _validated_labels(labels, tf.shape(score)[0])

    verified_indices = tf.cast(tf.where(label["verified"])[:, 0], tf.int32)
    verified_scores = tf.gather(score, verified_indices)
    verified_targets = tf.cast(tf.gather(label["compatible"], verified_indices), tf.float32)
    search_yield_per_candidate = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=verified_targets, logits=verified_scores
    )
    search_yield_bce, search_yield_recipe_count, search_yield_weight_sum = _recipe_macro_mean(
        search_yield_per_candidate,
        verified_indices,
        recipe_index=label["recipe_index"],
        recipe_weight=label["recipe_weight"],
        candidate_weight=label["candidate_weight"],
    )
    search_yield_accuracy, _, _ = _recipe_macro_mean(
        tf.cast(tf.equal(verified_scores >= 0.0, verified_targets > 0.5), tf.float32),
        verified_indices,
        recipe_index=label["recipe_index"],
        recipe_weight=label["recipe_weight"],
        candidate_weight=label["candidate_weight"],
    )
    positive_yield_rate, _, _ = _recipe_macro_mean(
        verified_targets,
        verified_indices,
        recipe_index=label["recipe_index"],
        recipe_weight=label["recipe_weight"],
        candidate_weight=label["candidate_weight"],
    )
    (
        pairwise_ranking_loss,
        pairwise_ranking_accuracy,
        pairwise_ranking_pair_count,
        pairwise_ranking_recipe_count,
        pairwise_ranking_weight_sum,
    ) = _recipe_pairwise_ranking(score, label)

    varying_count = tf.reduce_sum(label["varying"], axis=-1)
    mdn_eligible = tf.logical_and(
        tf.logical_and(label["compatible"], label["has_target"]),
        varying_count > 0.0,
    )
    mdn_indices = tf.cast(tf.where(mdn_eligible)[:, 0], tf.int32)
    continuous_per_candidate = normalized_masked_logistic_normal_nll(
        tf.gather(label["target"], mdn_indices),
        tf.gather(label["varying"], mdn_indices),
        tf.gather(mixture_logits, mdn_indices),
        tf.gather(mixture_loc, mdn_indices),
        tf.gather(mixture_logscale, mdn_indices),
        epsilon=config.logistic_epsilon,
    )
    local_mdn_nll, local_mdn_recipe_count, local_mdn_weight_sum = _recipe_macro_mean(
        continuous_per_candidate,
        mdn_indices,
        recipe_index=label["recipe_index"],
        recipe_weight=label["recipe_weight"],
        candidate_weight=label["candidate_weight"],
    )
    eligible_target = tf.gather(label["target"], mdn_indices)
    eligible_varying = tf.gather(label["varying"], mdn_indices)
    eligible_logits = tf.gather(mixture_logits, mdn_indices)
    eligible_loc = tf.gather(mixture_loc, mdn_indices)
    component_rms = _mixture_median_masked_rms(
        eligible_target,
        eligible_varying,
        eligible_loc,
    )
    coverage_per_candidate = _mixture_mass_aware_coverage(
        component_rms,
        eligible_logits,
        temperature=config.local_coverage_temperature,
    )
    alignment_per_candidate = _mixture_operational_top_l_alignment(
        component_rms,
        eligible_logits,
        distance_temperature=config.local_coverage_temperature,
    )
    (
        local_mixture_mass_coverage,
        local_coverage_recipe_count,
        local_coverage_weight_sum,
    ) = _recipe_macro_mean(
        coverage_per_candidate,
        mdn_indices,
        recipe_index=label["recipe_index"],
        recipe_weight=label["recipe_weight"],
        candidate_weight=label["candidate_weight"],
    )
    local_operational_top_l_alignment, _, _ = _recipe_macro_mean(
        alignment_per_candidate,
        mdn_indices,
        recipe_index=label["recipe_index"],
        recipe_weight=label["recipe_weight"],
        candidate_weight=label["candidate_weight"],
    )
    hard_best_per_candidate = tf.reduce_min(component_rms, axis=-1)
    local_median_best_of_m_rms, _, _ = _recipe_macro_mean(
        hard_best_per_candidate,
        mdn_indices,
        recipe_index=label["recipe_index"],
        recipe_weight=label["recipe_weight"],
        candidate_weight=label["candidate_weight"],
    )
    operational = _operational_top_l_metrics(
        component_rms,
        eligible_varying,
        eligible_logits,
        eligible_loc,
        hit_rms_threshold=config.operational_hit_rms_threshold,
        duplicate_rms_threshold=config.operational_duplicate_rms_threshold,
    )
    operational_macro = {
        name: _recipe_macro_mean(
            value,
            mdn_indices,
            recipe_index=label["recipe_index"],
            recipe_weight=label["recipe_weight"],
            candidate_weight=label["candidate_weight"],
        )[0]
        for name, value in operational.items()
    }
    mean_varying_count, _, _ = _recipe_macro_mean(
        tf.gather(varying_count, mdn_indices),
        mdn_indices,
        recipe_index=label["recipe_index"],
        recipe_weight=label["recipe_weight"],
        candidate_weight=label["candidate_weight"],
    )

    loss = (
        config.search_yield_weight * search_yield_bce
        + config.pairwise_ranking_weight * pairwise_ranking_loss
        + config.local_mdn_weight * local_mdn_nll
        + config.local_coverage_weight * local_mixture_mass_coverage
        + config.operational_top_l_alignment_weight
        * local_operational_top_l_alignment
    )
    finite = tf.debugging.assert_all_finite(loss, "V5 candidate objective is non-finite")
    with tf.control_dependencies((finite,)):
        loss = tf.identity(loss)

    outcome = label["outcome"]
    positive_code = SEARCH_OUTCOME_CODE["compatible_found"]
    negative_code = SEARCH_OUTCOME_CODE["no_compatible_found_within_frozen_search_budget"]
    return {
        "loss": loss,
        "search_yield_bce": search_yield_bce,
        "pairwise_ranking_loss": pairwise_ranking_loss,
        "pairwise_ranking_accuracy": pairwise_ranking_accuracy,
        "local_mdn_nll": local_mdn_nll,
        "local_mixture_mass_coverage": local_mixture_mass_coverage,
        "local_operational_top_l_alignment": local_operational_top_l_alignment,
        "local_median_best_of_m_rms": local_median_best_of_m_rms,
        "local_median_best_top_l_rms": operational_macro["best_rms"],
        "local_target_recall_top_l": operational_macro["target_recall"],
        "local_hit_mixture_rank_top_l": operational_macro["hit_rank"],
        "local_top_l_mixture_mass": operational_macro["top_l_mass"],
        "local_mixture_entropy": operational_macro["entropy"],
        "local_effective_mixture_components": operational_macro[
            "effective_components"
        ],
        "local_effective_mixture_utilization": operational_macro["utilization"],
        "local_duplicate_fraction_top_l": operational_macro["duplicate_fraction"],
        "proposal_search_yield_accuracy": search_yield_accuracy,
        "verified_positive_yield_rate": positive_yield_rate,
        "mean_varying_dimension_count": mean_varying_count,
        "candidate_count": tf.size(outcome),
        "verified_count": tf.math.count_nonzero(label["verified"], dtype=tf.int32),
        "compatible_found_count": tf.math.count_nonzero(outcome == positive_code, dtype=tf.int32),
        "no_compatible_found_within_budget_count": tf.math.count_nonzero(
            outcome == negative_code, dtype=tf.int32
        ),
        "unverified_count": tf.math.count_nonzero(label["unverified"], dtype=tf.int32),
        "local_target_count": tf.math.count_nonzero(label["has_target"], dtype=tf.int32),
        "local_mdn_contributing_count": tf.size(mdn_indices),
        "local_coverage_contributing_count": tf.size(mdn_indices),
        "pairwise_ranking_pair_count": pairwise_ranking_pair_count,
        "pairwise_ranking_recipe_count": pairwise_ranking_recipe_count,
        "search_yield_recipe_count": search_yield_recipe_count,
        "local_mdn_recipe_count": local_mdn_recipe_count,
        "local_coverage_recipe_count": local_coverage_recipe_count,
        "search_yield_effective_weight_sum": search_yield_weight_sum,
        "pairwise_ranking_effective_weight_sum": pairwise_ranking_weight_sum,
        "local_mdn_effective_weight_sum": local_mdn_weight_sum,
        "local_coverage_effective_weight_sum": local_coverage_weight_sum,
    }


__all__ = [
    "CANDIDATE_TRAINING_OBJECTIVE_V5_SCHEMA",
    "CANDIDATE_TRAINING_OBJECTIVE_V5_VERSION",
    "LOCAL_HARD_COVERAGE_BASELINE_SEMANTICS",
    "LOCAL_COVERAGE_RMS_STABILITY_EPSILON",
    "LOCAL_COVERAGE_SEMANTICS",
    "MISSING_CLASS_SEMANTICS",
    "OPERATIONAL_TOP_L_METRICS_SEMANTICS",
    "OPERATIONAL_TOP_L_ALIGNMENT_SEMANTICS",
    "PAIRWISE_RANKING_SEMANTICS",
    "V5CandidateObjectiveConfig",
    "compute_v5_candidate_training_objective",
]
