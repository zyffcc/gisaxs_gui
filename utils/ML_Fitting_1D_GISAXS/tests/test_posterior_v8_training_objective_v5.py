from __future__ import annotations

import json

import numpy as np
import pytest


tf = pytest.importorskip("tensorflow")

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_supervision_v5 import (
    ExactCompatibleProvenance,
    FrozenSearchProvenance,
    KNOWN_TRUTH_ORACLE_PROTOCOL_ID,
    KNOWN_TRUTH_ORACLE_TERMINATION_REASON,
    POSITIVE_TERMINATION_REASON,
    V5CandidateSupervision,
    stack_candidate_supervision_v5,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.training_objective_v5 import (
    CANDIDATE_TRAINING_OBJECTIVE_V5_SCHEMA,
    CANDIDATE_TRAINING_OBJECTIVE_V5_VERSION,
    V5CandidateObjectiveConfig,
    compute_v5_candidate_training_objective,
)


def _search(*, found: int):
    return FrozenSearchProvenance(
        search_artifact_id=f"search/{found}",
        search_artifact_sha256="b" * 64,
        protocol_id="frozen-v5",
        protocol_sha256="c" * 64,
        evaluator_version="forward-v9",
        metric_name="exact_forward_logrmse",
        threshold_name="curve_equivalence_logrmse",
        threshold_value=0.02,
        threshold_source_id="paper-protocol-v2",
        exact_forward_call_budget=64,
        exact_forward_calls_used=64,
        termination_reason=(
            POSITIVE_TERMINATION_REASON
            if found
            else "exact_forward_budget_exhausted_without_compatible"
        ),
        completed=True,
        compatible_representative_count=found,
    )


def _exact():
    return ExactCompatibleProvenance(
        artifact_id="exact/representative",
        artifact_sha256="a" * 64,
        metric_value=0.01,
        bounds_passed=True,
        physics_passed=True,
    )


def _candidate(
    outcome: str,
    recipe: str,
    *,
    target: bool = False,
    varying=(0, 1),
    clean_recipe_weight: float = 1.0,
    candidate_weight: float = 1.0,
):
    active = np.zeros(26, dtype=np.float32)
    active[[0, 1, 4, 5]] = 1.0
    density = np.zeros(26, dtype=np.float32)
    density[list(varying)] = 1.0
    local = None
    if target:
        local = np.full(26, 0.5, dtype=np.float32)
        for index, value in zip(varying, (0.25, 0.75)):
            local[index] = value
    positive = outcome == "compatible_found"
    verified = outcome != "unverified"
    return V5CandidateSupervision(
        clean_recipe_id=recipe,
        candidate_id=f"{recipe}/{outcome}/{candidate_weight}",
        outcome=outcome,
        active_dimension_mask=active,
        varying_dimension_mask=density,
        search_provenance=_search(found=1 if positive else 0) if verified else None,
        exact_compatible=_exact() if positive else None,
        target_local=local,
        clean_recipe_weight=clean_recipe_weight,
        candidate_weight=candidate_weight,
    )


def _labels(candidates, recipe_indices):
    return stack_candidate_supervision_v5(candidates, clean_recipe_indices=recipe_indices)


def _outputs(scores, *, modes=2):
    batch = len(scores)
    return {
        "proposal_search_yield_logit": np.asarray(scores, np.float32).reshape(batch, 1),
        "mixture_logits": np.zeros((batch, modes), dtype=np.float32),
        "mixture_loc": np.zeros((batch, modes, 26), dtype=np.float32),
        "mixture_logscale": np.zeros((batch, modes, 26), dtype=np.float32),
    }


def _scalar(result, name):
    return result[name].numpy().item()


def _dense(gradient):
    return tf.convert_to_tensor(gradient).numpy()


def _isolated_config(**updates):
    values = {
        "search_yield_weight": 0.0,
        "pairwise_ranking_weight": 0.0,
        "local_mdn_weight": 0.0,
        "local_coverage_weight": 0.0,
        "operational_top_l_alignment_weight": 0.0,
    }
    values.update(updates)
    return V5CandidateObjectiveConfig(**values)


def test_unverified_has_zero_bce_and_mdn_contribution():
    labels = _labels(
        (
            _candidate("compatible_found", "r0", target=True),
            _candidate("unverified", "r0"),
        ),
        (0, 0),
    )
    first_outputs = _outputs((0.3, -2.0))
    first = compute_v5_candidate_training_objective(first_outputs, labels)
    changed_outputs = {key: value.copy() for key, value in first_outputs.items()}
    changed_outputs["proposal_search_yield_logit"][1, 0] = 100.0
    changed_outputs["mixture_logits"][1] = (100.0, -100.0)
    changed_outputs["mixture_loc"][1] = 100.0
    changed_outputs["mixture_logscale"][1] = 100.0
    changed = compute_v5_candidate_training_objective(changed_outputs, labels)
    assert _scalar(first, "loss") == pytest.approx(_scalar(changed, "loss"))
    assert _scalar(first, "search_yield_bce") == pytest.approx(_scalar(changed, "search_yield_bce"))
    assert _scalar(first, "local_mdn_nll") == pytest.approx(_scalar(changed, "local_mdn_nll"))
    assert _scalar(first, "verified_count") == 1
    assert _scalar(first, "unverified_count") == 1


def test_known_truth_warmup_trains_local_density_but_never_search_yield():
    labels = _labels((_candidate("compatible_found", "warmup", target=True),), (0,))
    labels["search_protocol_id"] = np.asarray([KNOWN_TRUTH_ORACLE_PROTOCOL_ID])
    labels["search_exact_forward_call_budget"][0] = 1
    labels["search_exact_forward_calls_used"][0] = 1
    labels["search_termination_reason"] = np.asarray(
        [KNOWN_TRUTH_ORACLE_TERMINATION_REASON]
    )
    first = compute_v5_candidate_training_objective(_outputs((-100.0,)), labels)
    changed = compute_v5_candidate_training_objective(_outputs((100.0,)), labels)
    assert _scalar(first, "search_yield_bce") == 0.0
    assert _scalar(changed, "search_yield_bce") == 0.0
    assert _scalar(first, "verified_count") == 0
    assert _scalar(first, "local_mdn_contributing_count") == 1
    assert _scalar(first, "local_mdn_nll") == pytest.approx(
        _scalar(changed, "local_mdn_nll")
    )


def test_negative_has_no_mdn_gradient_and_fixed_axes_are_excluded():
    labels = _labels(
        (
            _candidate("compatible_found", "r0", target=True, varying=(0,)),
            _candidate("no_compatible_found_within_frozen_search_budget", "r1"),
        ),
        (0, 1),
    )
    variables = (
        tf.Variable([[0.2], [-0.3]], dtype=tf.float32),
        tf.Variable(tf.zeros((2, 2), tf.float32)),
        tf.Variable(tf.zeros((2, 2, 26), tf.float32)),
        tf.Variable(tf.zeros((2, 2, 26), tf.float32)),
    )
    outputs = dict(
        zip(
            ("proposal_search_yield_logit", "mixture_logits", "mixture_loc", "mixture_logscale"),
            variables,
        )
    )
    with tf.GradientTape() as tape:
        result = compute_v5_candidate_training_objective(outputs, labels)
    gradients = tape.gradient(result["loss"], variables)
    dense = tuple(_dense(gradient) for gradient in gradients)
    assert all(np.all(np.isfinite(gradient)) for gradient in dense)
    np.testing.assert_array_equal(dense[2][1], 0.0)
    np.testing.assert_array_equal(dense[3][1], 0.0)
    np.testing.assert_array_equal(dense[2][0, :, 1:], 0.0)
    np.testing.assert_array_equal(dense[3][0, :, 1:], 0.0)
    assert _scalar(result, "local_mdn_contributing_count") == 1
    assert _scalar(result, "mean_varying_dimension_count") == pytest.approx(1.0)


def test_recipe_macro_weighting_ignores_candidate_multiplicity():
    negative = "no_compatible_found_within_frozen_search_budget"
    short = compute_v5_candidate_training_objective(
        _outputs((0.0, 2.0)),
        _labels((_candidate(negative, "r0"), _candidate(negative, "r1")), (0, 1)),
    )
    repeated = compute_v5_candidate_training_objective(
        _outputs((0.0, 2.0, 2.0, 2.0)),
        _labels(
            (
                _candidate(negative, "r0"),
                _candidate(negative, "r1"),
                _candidate(negative, "r1"),
                _candidate(negative, "r1"),
            ),
            (0, 1, 1, 1),
        ),
    )
    expected = (np.logaddexp(0.0, 0.0) + np.logaddexp(0.0, 2.0)) / 2.0
    assert _scalar(short, "search_yield_bce") == pytest.approx(expected)
    assert _scalar(repeated, "search_yield_bce") == pytest.approx(expected)
    assert _scalar(repeated, "search_yield_recipe_count") == 2
    assert _scalar(repeated, "compatible_found_count") == 0
    assert _scalar(repeated, "local_mdn_nll") == 0.0


@pytest.mark.parametrize(
    "outcome",
    (
        "compatible_found",
        "no_compatible_found_within_frozen_search_budget",
    ),
)
def test_pairwise_ranking_missing_class_is_exact_zero(outcome):
    labels = _labels((_candidate(outcome, "r0"),), (0,))
    result = compute_v5_candidate_training_objective(
        _outputs((3.0,)),
        labels,
        _isolated_config(pairwise_ranking_weight=1.0),
    )

    assert _scalar(result, "loss") == 0.0
    assert _scalar(result, "pairwise_ranking_loss") == 0.0
    assert _scalar(result, "pairwise_ranking_pair_count") == 0
    assert _scalar(result, "pairwise_ranking_recipe_count") == 0


def test_pairwise_ranking_never_pairs_across_clean_recipes():
    labels = _labels(
        (
            _candidate("compatible_found", "r0"),
            _candidate("no_compatible_found_within_frozen_search_budget", "r1"),
        ),
        (0, 1),
    )
    result = compute_v5_candidate_training_objective(
        _outputs((-3.0, 3.0)),
        labels,
        _isolated_config(pairwise_ranking_weight=1.0),
    )

    assert _scalar(result, "loss") == 0.0
    assert _scalar(result, "pairwise_ranking_pair_count") == 0


def test_pairwise_ranking_prefers_positive_above_completed_negative_and_has_gradient():
    labels = _labels(
        (
            _candidate("compatible_found", "r0"),
            _candidate("no_compatible_found_within_frozen_search_budget", "r0"),
        ),
        (0, 0),
    )
    config = _isolated_config(pairwise_ranking_weight=1.0)
    good = compute_v5_candidate_training_objective(
        _outputs((2.0, -2.0)), labels, config
    )
    bad = compute_v5_candidate_training_objective(
        _outputs((-2.0, 2.0)), labels, config
    )

    assert _scalar(good, "pairwise_ranking_loss") < _scalar(
        bad, "pairwise_ranking_loss"
    )
    assert _scalar(good, "pairwise_ranking_accuracy") == 1.0
    assert _scalar(bad, "pairwise_ranking_accuracy") == 0.0
    assert _scalar(good, "search_yield_bce") < _scalar(bad, "search_yield_bce")
    assert _scalar(good, "pairwise_ranking_pair_count") == 1

    reversed_outputs = {
        name: value[::-1].copy() for name, value in _outputs((2.0, -2.0)).items()
    }
    reversed_labels = {name: value[::-1].copy() for name, value in labels.items()}
    reversed_result = compute_v5_candidate_training_objective(
        reversed_outputs, reversed_labels, config
    )
    assert _scalar(reversed_result, "pairwise_ranking_loss") == pytest.approx(
        _scalar(good, "pairwise_ranking_loss")
    )

    scores = tf.Variable([[-2.0], [2.0]], dtype=tf.float32)
    outputs = _outputs((0.0, 0.0))
    outputs["proposal_search_yield_logit"] = scores
    with tf.GradientTape() as tape:
        loss = compute_v5_candidate_training_objective(outputs, labels, config)["loss"]
    gradient = tape.gradient(loss, scores).numpy().reshape(-1)
    assert np.all(np.isfinite(gradient))
    assert gradient[0] < 0.0 < gradient[1]


def test_all_expanded_targets_need_distinct_mixture_medians_for_low_coverage_loss():
    labels = _labels(
        (
            _candidate("compatible_found", "r0", target=True, varying=(0,)),
            _candidate("compatible_found", "r0", target=True, varying=(0,)),
        ),
        (0, 0),
    )
    labels["target_local"][:, 0] = (0.2, 0.8)
    config = _isolated_config(
        local_coverage_weight=1.0,
        operational_top_l_alignment_weight=1.0,
    )
    collapsed = _outputs((0.0, 0.0), modes=2)
    collapsed["mixture_loc"][:, :, 0] = np.log(0.2 / 0.8)
    covered = _outputs((0.0, 0.0), modes=2)
    covered["mixture_loc"][:, 0, 0] = np.log(0.2 / 0.8)
    covered["mixture_loc"][:, 1, 0] = np.log(0.8 / 0.2)

    collapsed_result = compute_v5_candidate_training_objective(
        collapsed, labels, config
    )
    covered_result = compute_v5_candidate_training_objective(covered, labels, config)
    assert _scalar(collapsed_result, "local_median_best_of_m_rms") > 0.25
    assert _scalar(covered_result, "local_median_best_of_m_rms") < 1.0e-5
    assert _scalar(covered_result, "local_mixture_mass_coverage") < _scalar(
        collapsed_result, "local_mixture_mass_coverage"
    )
    assert _scalar(covered_result, "local_coverage_contributing_count") == 2

    locations = tf.Variable(tf.zeros((2, 2, 26), dtype=tf.float32))
    differentiable = _outputs((0.0, 0.0), modes=2)
    differentiable["mixture_loc"] = locations
    with tf.GradientTape() as tape:
        loss = compute_v5_candidate_training_objective(
            differentiable, labels, config
        )["loss"]
    gradient = _dense(tape.gradient(loss, locations))
    assert np.all(np.isfinite(gradient))
    assert np.any(gradient[:, :, 0] != 0.0)
    np.testing.assert_array_equal(gradient[:, :, 1:], 0.0)


def test_low_mass_hidden_correct_mode_is_penalized_by_operational_coverage():
    labels = _labels(
        (_candidate("compatible_found", "r0", target=True, varying=(0,)),),
        (0,),
    )
    labels["target_local"][0, 0] = 0.8
    correct_logit = np.log(0.8 / 0.2)
    wrong_logit = np.log(0.2 / 0.8)
    hidden = _outputs((0.0,), modes=12)
    hidden["mixture_loc"][:, :, 0] = wrong_logit
    hidden["mixture_loc"][:, 11, 0] = correct_logit
    hidden["mixture_logits"][:, 11] = -12.0
    promoted = {name: np.array(value, copy=True) for name, value in hidden.items()}
    promoted["mixture_logits"][:, 11] = 12.0
    config = _isolated_config(
        local_coverage_weight=1.0,
        operational_top_l_alignment_weight=1.0,
    )
    hidden_result = compute_v5_candidate_training_objective(hidden, labels, config)
    promoted_result = compute_v5_candidate_training_objective(promoted, labels, config)
    assert _scalar(hidden_result, "loss") == pytest.approx(
        _scalar(hidden_result, "local_mixture_mass_coverage")
        + _scalar(hidden_result, "local_operational_top_l_alignment")
    )
    assert _scalar(hidden_result, "local_median_best_of_m_rms") < 1.0e-5
    assert _scalar(promoted_result, "local_median_best_of_m_rms") < 1.0e-5
    assert _scalar(hidden_result, "local_mixture_mass_coverage") > 0.4
    assert _scalar(promoted_result, "local_mixture_mass_coverage") < 1.0e-3
    assert _scalar(hidden_result, "local_operational_top_l_alignment") > 0.8
    assert _scalar(promoted_result, "local_operational_top_l_alignment") < 1.0e-5
    assert _scalar(hidden_result, "local_target_recall_top_l") == 0.0
    assert _scalar(promoted_result, "local_target_recall_top_l") == 1.0
    assert _scalar(hidden_result, "local_median_best_top_l_rms") > 0.5
    assert _scalar(promoted_result, "local_median_best_top_l_rms") < 1.0e-5
    assert _scalar(hidden_result, "local_hit_mixture_rank_top_l") == 5.0
    assert _scalar(promoted_result, "local_hit_mixture_rank_top_l") == 1.0
    assert _scalar(hidden_result, "local_duplicate_fraction_top_l") > 0.5
    entropy = _scalar(hidden_result, "local_mixture_entropy")
    effective = _scalar(hidden_result, "local_effective_mixture_components")
    assert effective == pytest.approx(np.exp(entropy), rel=1.0e-5)
    assert _scalar(hidden_result, "local_effective_mixture_utilization") == pytest.approx(
        effective / 12.0, rel=1.0e-5
    )
    assert 0.0 < _scalar(hidden_result, "local_top_l_mixture_mass") < 1.0

    locations = tf.Variable(hidden["mixture_loc"])
    logits = tf.Variable(hidden["mixture_logits"])
    differentiable = dict(hidden)
    differentiable["mixture_loc"] = locations
    differentiable["mixture_logits"] = logits
    alignment_config = _isolated_config(
        operational_top_l_alignment_weight=1.0
    )
    with tf.GradientTape() as tape:
        alignment_result = compute_v5_candidate_training_objective(
            differentiable, labels, alignment_config
        )
        loss = alignment_result["loss"]
    assert _scalar(alignment_result, "loss") == pytest.approx(
        _scalar(alignment_result, "local_operational_top_l_alignment")
    )
    location_gradient, logit_gradient = tape.gradient(loss, (locations, logits))
    location_gradient = _dense(location_gradient)
    logit_gradient = _dense(logit_gradient)
    assert np.all(np.isfinite(location_gradient))
    assert np.all(np.isfinite(logit_gradient))
    np.testing.assert_array_equal(location_gradient[:, :, 1:], 0.0)
    assert logit_gradient[0, 11] < 0.0


def test_mass_coverage_remains_finite_for_extreme_finite_mixture_logits():
    labels = _labels(
        (_candidate("compatible_found", "r0", target=True, varying=(0,)),),
        (0,),
    )
    outputs = _outputs((0.0,), modes=12)
    outputs["mixture_logits"][0] = np.linspace(-1.0e6, 1.0e6, 12)
    logits = tf.Variable(outputs["mixture_logits"])
    differentiable = dict(outputs)
    differentiable["mixture_logits"] = logits
    config = _isolated_config(local_coverage_weight=1.0)
    with tf.GradientTape() as tape:
        result = compute_v5_candidate_training_objective(
            differentiable, labels, config
        )
    gradient = _dense(tape.gradient(result["loss"], logits))
    assert np.isfinite(_scalar(result, "loss"))
    assert np.all(np.isfinite(gradient))


def test_duplicate_fraction_marks_later_ranks_against_earlier_ranks():
    labels = _labels(
        (_candidate("compatible_found", "r0", target=True, varying=(0,)),),
        (0,),
    )
    outputs = _outputs((0.0,), modes=12)
    outputs["mixture_logits"][0] = np.arange(12, 0, -1, dtype=np.float32)
    medians = np.asarray((0.5, 0.485, 0.515, 0.9), dtype=np.float32)
    outputs["mixture_loc"][0, :4, 0] = np.log(medians / (1.0 - medians))
    result = compute_v5_candidate_training_objective(
        outputs,
        labels,
        _isolated_config(local_coverage_weight=1.0),
    )
    assert _scalar(result, "local_duplicate_fraction_top_l") == pytest.approx(0.5)


def test_zero_varying_target_is_excluded_from_all_local_terms():
    labels = _labels(
        (_candidate("compatible_found", "r0", target=True, varying=()),),
        (0,),
    )
    result = compute_v5_candidate_training_objective(
        _outputs((0.0,), modes=12),
        labels,
        _isolated_config(search_yield_weight=1.0),
    )
    for name in (
        "local_mdn_nll",
        "local_mixture_mass_coverage",
        "local_operational_top_l_alignment",
        "local_median_best_top_l_rms",
        "local_duplicate_fraction_top_l",
    ):
        assert _scalar(result, name) == 0.0
    assert _scalar(result, "local_mdn_contributing_count") == 0


def test_expanded_representative_weights_preserve_local_objective_mass():
    config = _isolated_config(local_mdn_weight=1.0, local_coverage_weight=1.0)
    single_labels = _labels(
        (_candidate("compatible_found", "r0", target=True),), (0,)
    )
    repeated_labels = _labels(
        (
            _candidate(
                "compatible_found", "r0", target=True, candidate_weight=0.5
            ),
            _candidate(
                "compatible_found", "r0", target=True, candidate_weight=0.5
            ),
        ),
        (0, 0),
    )
    single = compute_v5_candidate_training_objective(
        _outputs((0.0,)), single_labels, config
    )
    repeated = compute_v5_candidate_training_objective(
        _outputs((0.0, 0.0)), repeated_labels, config
    )

    assert _scalar(repeated, "local_mdn_nll") == pytest.approx(
        _scalar(single, "local_mdn_nll")
    )
    assert _scalar(repeated, "local_median_best_of_m_rms") == pytest.approx(
        _scalar(single, "local_median_best_of_m_rms")
    )
    assert _scalar(repeated, "local_mixture_mass_coverage") == pytest.approx(
        _scalar(single, "local_mixture_mass_coverage")
    )
    assert _scalar(repeated, "local_mdn_effective_weight_sum") == pytest.approx(1.0)
    assert _scalar(repeated, "local_coverage_effective_weight_sum") == pytest.approx(
        1.0
    )


def test_expanded_representative_weights_do_not_multiply_pairwise_or_bce_mass():
    negative = "no_compatible_found_within_frozen_search_budget"
    config = _isolated_config(
        search_yield_weight=1.0, pairwise_ranking_weight=1.0
    )
    single = compute_v5_candidate_training_objective(
        _outputs((1.0, -1.0)),
        _labels(
            (
                _candidate("compatible_found", "r0", candidate_weight=1.0),
                _candidate(negative, "r0", candidate_weight=1.0),
            ),
            (0, 0),
        ),
        config,
    )
    expanded = compute_v5_candidate_training_objective(
        _outputs((1.0, 1.0, -1.0)),
        _labels(
            (
                _candidate("compatible_found", "r0", candidate_weight=0.5),
                _candidate("compatible_found", "r0", candidate_weight=0.5),
                _candidate(negative, "r0", candidate_weight=1.0),
            ),
            (0, 0, 0),
        ),
        config,
    )

    assert _scalar(expanded, "search_yield_bce") == pytest.approx(
        _scalar(single, "search_yield_bce")
    )
    assert _scalar(expanded, "pairwise_ranking_loss") == pytest.approx(
        _scalar(single, "pairwise_ranking_loss")
    )
    assert _scalar(expanded, "pairwise_ranking_effective_weight_sum") == pytest.approx(
        1.0
    )


def test_missing_verified_class_is_stable_and_audited():
    labels = _labels((_candidate("unverified", "r0"), _candidate("unverified", "r1")), (0, 1))
    result = compute_v5_candidate_training_objective(_outputs((-100.0, 100.0)), labels)
    for name in (
        "loss",
        "search_yield_bce",
        "local_mdn_nll",
        "proposal_search_yield_accuracy",
        "verified_positive_yield_rate",
    ):
        assert _scalar(result, name) == 0.0
    assert _scalar(result, "candidate_count") == 2
    assert _scalar(result, "verified_count") == 0
    assert _scalar(result, "unverified_count") == 2
    assert _scalar(result, "search_yield_recipe_count") == 0


def test_tensor_contract_rejects_forged_evidence_target_and_mixed_protocol():
    labels = _labels(
        (
            _candidate("compatible_found", "r0", target=True),
            _candidate("no_compatible_found_within_frozen_search_budget", "r0"),
        ),
        (0, 0),
    )
    forged = {key: value.copy() for key, value in labels.items()}
    forged["search_completed"][1] = False
    with pytest.raises(tf.errors.InvalidArgumentError, match="completed"):
        compute_v5_candidate_training_objective(_outputs((0.0, 0.0)), forged)
    wrong_target = {key: value.copy() for key, value in labels.items()}
    wrong_target["target_local"][0, 2] = 0.7
    with pytest.raises(tf.errors.InvalidArgumentError, match="must equal 0.5"):
        compute_v5_candidate_training_objective(_outputs((0.0, 0.0)), wrong_target)
    mixed = {key: value.copy() for key, value in labels.items()}
    mixed["search_protocol_sha256"][1] = "d" * 64
    with pytest.raises(tf.errors.InvalidArgumentError, match="one frozen"):
        compute_v5_candidate_training_objective(_outputs((0.0, 0.0)), mixed)


def test_objective_is_plain_tensor_order_independent_and_versioned():
    labels = _labels((_candidate("compatible_found", "r", target=True),), (7,))
    outputs = _outputs((0.2,))
    expected = compute_v5_candidate_training_objective(outputs, labels)
    actual = compute_v5_candidate_training_objective(
        dict(reversed(list(outputs.items()))), dict(reversed(list(labels.items())))
    )
    assert _scalar(expected, "loss") == pytest.approx(_scalar(actual, "loss"))
    payload = V5CandidateObjectiveConfig(
        search_yield_weight=2.0, local_mdn_weight=0.5
    ).audit_payload()
    assert payload["schema_version"] == CANDIDATE_TRAINING_OBJECTIVE_V5_SCHEMA
    assert payload["version"] == CANDIDATE_TRAINING_OBJECTIVE_V5_VERSION
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


def test_objective_rejects_bad_config_incomplete_budget_and_wrong_positive_reason():
    with pytest.raises(ValueError, match="at least one"):
        V5CandidateObjectiveConfig(
            search_yield_weight=0.0,
            pairwise_ranking_weight=0.0,
            local_mdn_weight=0.0,
            local_coverage_weight=0.0,
            operational_top_l_alignment_weight=0.0,
        )
    with pytest.raises(ValueError, match="proposal_execution_policy_sha256"):
        V5CandidateObjectiveConfig(proposal_execution_policy_sha256="0" * 64)
    labels = _labels((_candidate("unverified", "r"),), (0,))
    labels.pop("search_metric_name")
    with pytest.raises(ValueError, match="missing"):
        compute_v5_candidate_training_objective(_outputs((0.0,)), labels)

    negative_labels = _labels(
        (_candidate("no_compatible_found_within_frozen_search_budget", "r"),), (0,)
    )
    negative_labels["search_exact_forward_calls_used"][0] = 12
    negative_labels["search_termination_reason"][0] = (
        "frozen_protocol_completed_without_compatible"
    )
    with pytest.raises(tf.errors.InvalidArgumentError, match="equal frozen branch budget"):
        compute_v5_candidate_training_objective(_outputs((0.0,)), negative_labels)

    positive_labels = _labels((_candidate("compatible_found", "r", target=True),), (0,))
    positive_labels["search_termination_reason"][0] = "compatible_target_reached"
    with pytest.raises(tf.errors.InvalidArgumentError, match="canonical full-budget termination"):
        compute_v5_candidate_training_objective(_outputs((0.0,)), positive_labels)
