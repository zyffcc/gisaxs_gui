from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PosteriorV8.bounds_first_contract import (
    BOUNDS_EMBEDDING_VERSION,
    LOCAL_TARGET_SEMANTICS,
    local_varying_mask,
)
from PosteriorV8.bounds_first_shards import BoundsFirstShardConfig, reconstruct_clean_recipe
from PosteriorV8.bounds_local_inference import (
    BoundsLocalBranchCondition,
    CallableBoundsLocalProposalModel,
    LocalProposalSample,
    build_bounds_local_model_inputs,
    codec_effective_varying_mask,
    rank_canonical_joint_branches,
    sample_local_logistic_normal_mixture,
)
from PosteriorV8.bounds_local_production import (
    physical_seed_from_local,
    run_bounds_local_raw_inference,
)
from PosteriorV8.bounds_local_verified import run_verified_bounds_local_inference
from PosteriorV8.bounds_model_contract import (
    BOUNDS_PROPOSAL_MODEL_VERSION,
    MODEL_BRANCH_CATALOG_VERSION,
    MODEL_COMPONENT_SLOTS_VERSION,
    MODEL_INPUT_KEYS,
    MODEL_OUTPUT_KEYS,
)
from PosteriorV8.branch_catalog import branch_pattern_id
from PosteriorV8.canonical_branch_catalog import canonical_branch_pattern_is_valid
from PosteriorV8.contract import (
    SPHERE,
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    full_component_bounds,
    gui_component_to_latent,
    latent_component_to_gui,
    topology_id_for,
)
from PosteriorV8.evaluation import EvaluationThresholds, ObservedCurve
from PosteriorV8.inference_proposals import (
    ContinuousProposalOutput,
    DiscreteProposalOutput,
    JointBranchScore,
)
from PosteriorV8.one_click_inference import InferenceBudget, RefinementOutcome
from PosteriorV8.production_bridge import (
    ProductionBranchFactory,
    ProductionExactRefiner,
    UserSearchSpace,
)
from PosteriorV8.profiled_forward import component_unit_basis
from PosteriorV8.reference_bank import CompetingBranch
from PosteriorV8.rescue_inference import RescuePolicy


def _bounds(*, all_fixed: bool = False):
    return GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(8.0, 8.0) if all_fixed else ClosedInterval(8.0, 8.0),
        sigma_R=(ClosedInterval(0.96, 0.96) if all_fixed else ClosedInterval(0.80, 1.20)),
    )


def _score(branch):
    return JointBranchScore(
        branch=branch,
        topology_rank=1,
        pattern_rank_within_topology=1,
        joint_rank=1,
        topology_log_score=-0.1,
        conditional_pattern_log_score=-0.2,
        joint_log_score=-0.3,
    )


def _setup(*, all_fixed=False):
    factory = ProductionBranchFactory(
        UserSearchSpace.for_components((_bounds(all_fixed=all_fixed),))
    )
    branch = CompetingBranch(topology_id=topology_id_for((SPHERE,)), pattern_id=0)
    context = factory.context_for(branch)
    assert context is not None
    condition = BoundsLocalBranchCondition.build(_score(branch), context)
    components, resolution = context.user_bounds_codec.decode(np.full(26, 0.5))
    q = np.geomspace(1e-3, 1.0, 48)
    intensity = 0.1 + 1.7 * component_unit_basis(
        q,
        latent_component_to_gui(components[0]),
    )
    curve = ObservedCurve(
        curve_id="bounds-local-k1",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
    )
    curve_inputs = {
        "x": np.ones((32, 3), dtype=np.float32),
        "point_mask": np.ones(32, dtype=bool),
        "global_features": np.zeros(5, dtype=np.float32),
    }
    return factory, branch, condition, components, resolution, curve, curve_inputs


def _mapping(topology_id, *, logscale=-100.0):
    topology = np.full((1, 34), -20.0)
    topology[0, topology_id] = 20.0
    patterns = np.full((1, 34, 32), -20.0)
    patterns[0, topology_id, 0] = 20.0
    return {
        "topology_logits": topology,
        "branch_pattern_logits": patterns,
        "mixture_logits": np.zeros((1, 1)),
        "mixture_loc": np.zeros((1, 1, 26)),
        "mixture_logscale": np.full((1, 1, 26), logscale),
    }


class _LocalModel:
    posterior_v8_model_version = BOUNDS_PROPOSAL_MODEL_VERSION
    posterior_v8_input_bounds_coordinate_semantics = BOUNDS_EMBEDDING_VERSION
    posterior_v8_output_coordinate_semantics = LOCAL_TARGET_SEMANTICS
    posterior_v8_branch_catalog_version = MODEL_BRANCH_CATALOG_VERSION
    posterior_v8_component_slots_version = MODEL_COMPONENT_SLOTS_VERSION

    def __init__(self, topology_id):
        self.topology_id = topology_id
        self.conditions = []

    def predict_discrete(self, curve_inputs):
        mapping = _mapping(self.topology_id)
        return DiscreteProposalOutput(
            topology_logits=mapping["topology_logits"][0],
            branch_pattern_logits=mapping["branch_pattern_logits"][0],
        )

    def predict_continuous(self, curve_inputs, condition):
        self.conditions.append(condition)
        return ContinuousProposalOutput(
            mixture_logits=np.zeros(1),
            mixture_loc=np.zeros((1, 26)),
            mixture_logscale=np.full((1, 26), -100.0),
        )


def _budget(*, samples=1, total=8):
    return InferenceBudget(
        topology_beam_size=1,
        branch_beam_size=1,
        mixture_components_per_branch=1,
        samples_per_mixture=samples,
        per_candidate_forward_evaluation_limit=1,
        forward_evaluation_limit=total,
    )


def _thresholds(*, parameter_distance=0.0):
    return EvaluationThresholds(
        raw_exact_log_rmse_max=0.01,
        standardized_exact_log_rmse_max=1.0,
        parameter_mode_distance_max=parameter_distance,
        raw_curve_equivalence_log_rmse_max=0.01,
        reference_mode_distance_max=0.03,
    )


def _source(result, source):
    return next(item for item in result.audit.sources if item.source == source)


def test_condition_and_batch_inputs_preserve_actual_fixed_asymmetric_gui_bounds():
    _, _, condition, _, _, _, inputs = _setup()
    assert condition.active_dimension_mask[:2] == (True, True)
    assert condition.local_varying_mask[:2] == (False, True)
    assert not any(condition.active_dimension_mask[2:])
    tensors = build_bounds_local_model_inputs(inputs, condition)

    assert tuple(tensors) == MODEL_INPUT_KEYS
    assert tensors["bounds_embedding"].shape == (1, 78)
    assert tensors["active_dimension_mask"].shape == (1, 26)
    assert tensors["varying_dimension_mask"].shape == (1, 26)
    np.testing.assert_array_equal(
        tensors["varying_dimension_mask"][0, :2],
        [0.0, 1.0],
    )
    # R is fixed at a non-central physical value; this is not a global 26-D box.
    assert tensors["bounds_embedding"][0, 0] == tensors["bounds_embedding"][0, 1]
    assert tensors["bounds_embedding"][0, 0] != 0.5
    assert tensors["bounds_embedding"][0, 2] == 1.0
    altered = list(condition.bounds_embedding)
    altered[0] += 1.0e-4
    with pytest.raises(ValueError, match="does not match"):
        replace(condition, bounds_embedding=tuple(altered))


@pytest.mark.parametrize(
    "recipe_index",
    [0, 1, 17, 18, 33, 34, 67, 68, 101, 233, 511, 700, 701, 1223],
)
def test_runtime_varying_mask_matches_bounds_first_training_contract(recipe_index):
    recipe = reconstruct_clean_recipe(
        BoundsFirstShardConfig(views_per_recipe=2),
        recipe_index,
    )
    assert codec_effective_varying_mask(recipe.label.bounds.local_codec()) == (
        local_varying_mask(recipe.label.bounds)
    )


def test_callable_adapter_enforces_frozen_input_output_and_artifact_semantics():
    _, branch, condition, _, _, _, inputs = _setup()
    captured = []

    def predictor(values):
        captured.append(values)
        return _mapping(branch.topology_id)

    model = CallableBoundsLocalProposalModel(
        predictor=predictor,
        posterior_v8_model_version=BOUNDS_PROPOSAL_MODEL_VERSION,
        posterior_v8_input_bounds_coordinate_semantics=BOUNDS_EMBEDDING_VERSION,
        posterior_v8_output_coordinate_semantics=LOCAL_TARGET_SEMANTICS,
        posterior_v8_branch_catalog_version=MODEL_BRANCH_CATALOG_VERSION,
        posterior_v8_component_slots_version=MODEL_COMPONENT_SLOTS_VERSION,
        probe_condition=condition,
    )
    model.predict_discrete(inputs)
    model.predict_continuous(inputs, condition)
    assert all(tuple(item) == MODEL_INPUT_KEYS for item in captured)
    assert set(_mapping(branch.topology_id)) == set(MODEL_OUTPUT_KEYS)

    with pytest.raises(ValueError, match="model_version"):
        CallableBoundsLocalProposalModel(
            predictor=predictor,
            posterior_v8_model_version="posterior_v8_global_v1",
            posterior_v8_input_bounds_coordinate_semantics=BOUNDS_EMBEDDING_VERSION,
            posterior_v8_output_coordinate_semantics=LOCAL_TARGET_SEMANTICS,
            posterior_v8_branch_catalog_version=MODEL_BRANCH_CATALOG_VERSION,
            posterior_v8_component_slots_version=MODEL_COMPONENT_SLOTS_VERSION,
            probe_condition=condition,
        )
    with pytest.raises(ValueError, match="component_slots_version"):
        replace(model, posterior_v8_component_slots_version="legacy")

    def incomplete(_):
        values = _mapping(branch.topology_id)
        values.pop("mixture_loc")
        return values

    incomplete_model = replace(model, predictor=incomplete)
    with pytest.raises(ValueError, match="output keys"):
        incomplete_model.predict_continuous(inputs, condition)


def test_v3_ranking_excludes_noncanonical_repeated_shape_d_permutations():
    topology_id = topology_id_for((SPHERE, SPHERE))
    noncanonical = branch_pattern_id((True, False, False, False), False)
    canonical = branch_pattern_id((False, True, False, False), False)
    topology = np.full(34, -20.0)
    topology[topology_id] = 20.0
    patterns = np.full((34, 32), -20.0)
    patterns[topology_id, canonical] = 10.0
    patterns[topology_id, noncanonical] = 1_000.0
    output = DiscreteProposalOutput(
        topology_logits=topology,
        branch_pattern_logits=patterns,
    )

    ranked = rank_canonical_joint_branches(output, topology_limit=1)
    assert ranked[0].branch.pattern_id == canonical
    assert all(
        canonical_branch_pattern_is_valid(
            item.branch.topology_id,
            item.branch.pattern_id,
        )
        for item in ranked
    )
    changed = patterns.copy()
    changed[topology_id, noncanonical] = -1_000.0
    replay = rank_canonical_joint_branches(
        DiscreteProposalOutput(
            topology_logits=topology,
            branch_pattern_logits=changed,
        ),
        topology_limit=1,
    )
    np.testing.assert_allclose(
        [item.joint_log_score for item in ranked],
        [item.joint_log_score for item in replay],
        rtol=0.0,
        atol=0.0,
    )


def test_bounds_local_condition_rejects_noncanonical_hard_branch():
    bounds = tuple(full_component_bounds(SPHERE, d_policy="optional") for _ in range(2))
    factory = ProductionBranchFactory(UserSearchSpace.for_components(bounds))
    branch = CompetingBranch(
        topology_id=topology_id_for((SPHERE, SPHERE)),
        pattern_id=branch_pattern_id((True, False, False, False), False),
    )
    context = factory.context_for(branch)
    assert context is not None
    with pytest.raises(ValueError, match="canonical hard branch"):
        BoundsLocalBranchCondition.build(_score(branch), context)


def test_local_logistic_normal_sample_decodes_directly_and_round_trips_without_global_alias():
    _, _, condition, _, _, _, _ = _setup()
    output = ContinuousProposalOutput(
        mixture_logits=np.zeros(1),
        mixture_loc=np.zeros((1, 26)),
        mixture_logscale=np.full((1, 26), -2.0),
    )
    proposal = sample_local_logistic_normal_mixture(
        output,
        condition,
        mixture_limit=1,
        samples_per_mixture=1,
        seed=17,
    )[0]

    assert not hasattr(proposal, "global_unit")
    assert proposal.local_unit[0] == 0.5
    assert proposal.local_unit[1] != 0.5
    physical = physical_seed_from_local(proposal)
    np.testing.assert_allclose(
        physical.user_local_coordinates.unit_cube,
        proposal.local_unit,
        rtol=0.0,
        atol=5e-12,
    )


def test_legacy_local_seed_without_amplitude_context_preserves_labelled_slots():
    common_bounds = GuiComponentBounds(
        SPHERE,
        R=ClosedInterval(5.0, 25.0),
        sigma_R=ClosedInterval(0.5, 3.0),
    )
    factory = ProductionBranchFactory(
        UserSearchSpace.for_components((common_bounds, common_bounds))
    )
    branch = CompetingBranch(
        topology_id=topology_id_for((SPHERE, SPHERE)),
        pattern_id=0,
    )
    context = factory.context_for(branch)
    assert context is not None
    condition = BoundsLocalBranchCondition.build(_score(branch), context)
    large = gui_component_to_latent(GuiComponentParameters(SPHERE, R=20.0, sigma_R=2.0))
    small = gui_component_to_latent(GuiComponentParameters(SPHERE, R=10.0, sigma_R=1.0))
    labelled = context.user_bounds_codec.encode((large, small), None)
    proposal = LocalProposalSample(
        condition=condition,
        mixture_index=0,
        mixture_rank=1,
        sample_index=1,
        mixture_log_weight=0.0,
        local_unit=labelled.unit_cube,
    )

    physical = physical_seed_from_local(proposal)
    assert tuple(np.exp(item.log_R) for item in physical.components) == pytest.approx((20.0, 10.0))
    assert physical.user_local_coordinates == context.user_bounds_codec.encode(
        physical.components,
        None,
    )


def test_local_candidate_rejects_out_of_bounds_and_noncanonical_fixed_coordinates():
    _, _, condition, _, _, _, _ = _setup()
    valid = LocalProposalSample(
        condition=condition,
        mixture_index=0,
        mixture_rank=1,
        sample_index=1,
        mixture_log_weight=0.0,
        local_unit=tuple(np.full(26, 0.5)),
    )
    outside = list(valid.local_unit)
    outside[1] = 1.01
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        replace(valid, local_unit=tuple(outside))
    fixed_changed = list(valid.local_unit)
    fixed_changed[0] = 0.4
    with pytest.raises(ValueError, match="fixed and inactive"):
        replace(valid, local_unit=tuple(fixed_changed))


def test_fixed_local_neural_and_sobol_seeds_deduplicate_under_one_budget_audit():
    factory, branch, _, _, _, curve, inputs = _setup(all_fixed=True)
    result = run_bounds_local_raw_inference(
        inputs,
        curve,
        model=_LocalModel(branch.topology_id),
        branch_factory=factory,
        inference_budget=_budget(total=8),
        rescue_policy=RescuePolicy(
            target_candidate_count=2,
            fallback_attempt_limit=3,
            sobol_seeds_per_branch=3,
            physical_dedup_tolerance=0.0,
        ),
        seed=23,
    )

    assert result.status == "raw_partial"
    assert len(result.candidates) == 1
    assert result.neural_generation.component_slots_version == MODEL_COMPONENT_SLOTS_VERSION
    assert _source(result, "neural").candidates_returned == 1
    assert _source(result, "sobol").duplicates == 3
    assert result.audit.forward_evaluations_used == 1


def test_incompatible_local_candidates_continue_to_fallback_and_never_claim_no_solution():
    factory, branch, _, _, _, curve, inputs = _setup()

    class IncompatibleRefiner(ProductionExactRefiner):
        def refine_physical(self, *args, **kwargs):
            outcome = super().refine_physical(*args, **kwargs)
            assert outcome.status == "success"
            return replace(
                outcome,
                value=replace(
                    outcome.value,
                    exact_intensity=self.curve.intensity * np.e,
                ),
            )

    result = run_verified_bounds_local_inference(
        inputs,
        curve,
        model=_LocalModel(branch.topology_id),
        branch_factory=factory,
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        inference_budget=_budget(total=8),
        rescue_policy=RescuePolicy(
            target_candidate_count=1,
            fallback_attempt_limit=3,
            sobol_seeds_per_branch=3,
        ),
        refiner=IncompatibleRefiner(curve=curve),
        seed=29,
    )

    assert result.status == "no_compatible_mode_found_within_budget"
    assert result.verified_parameter_mode_count == 0
    assert result.evaluation_report.accepted_count == 0
    assert len(result.raw_generation.candidates) == 4
    assert result.audit.fallback_attempts_used == 3
    assert "no_solution" not in result.status


def test_verified_search_counts_compatible_modes_before_observability_labels():
    factory, branch, _, _, _, curve, inputs = _setup()

    class CompatibleRefiner(ProductionExactRefiner):
        def refine_physical(self, *args, **kwargs):
            outcome = super().refine_physical(*args, **kwargs)
            assert outcome.status == "success"
            return replace(
                outcome,
                value=replace(outcome.value, exact_intensity=self.curve.intensity),
            )

    result = run_verified_bounds_local_inference(
        inputs,
        curve,
        model=_LocalModel(branch.topology_id),
        branch_factory=factory,
        thresholds=_thresholds(parameter_distance=0.0),
        target_parameter_mode_count=2,
        inference_budget=_budget(total=8),
        rescue_policy=RescuePolicy(
            target_candidate_count=1,
            fallback_attempt_limit=4,
            sobol_seeds_per_branch=4,
        ),
        refiner=CompatibleRefiner(curve=curve),
        seed=31,
    )

    assert result.status == "compatible_target_reached"
    assert result.verified_parameter_mode_count == 2
    assert len(result.raw_generation.candidates) == 2
    assert result.audit.fallback_attempts_used == 1
    assert any(item.status == "provisional_or_unknown" for item in result.observability_assessments)


def test_wrong_artifact_identity_fails_closed_before_any_model_prediction():
    factory, branch, _, _, _, curve, inputs = _setup()
    model = _LocalModel(branch.topology_id)
    model.posterior_v8_output_coordinate_semantics = "global_unit_cube_v1"
    calls = []
    model.predict_discrete = lambda values: calls.append(values)

    with pytest.raises(ValueError, match="output_coordinate_semantics"):
        run_bounds_local_raw_inference(
            inputs,
            curve,
            model=model,
            branch_factory=factory,
            inference_budget=_budget(),
            rescue_policy=RescuePolicy(
                target_candidate_count=1,
                fallback_attempt_limit=1,
                sobol_seeds_per_branch=1,
            ),
        )
    assert calls == []


def test_zero_refined_candidates_reports_budgeted_search_failure_not_no_solution():
    factory, branch, _, _, _, curve, inputs = _setup()

    class AlwaysFails(ProductionExactRefiner):
        def refine_physical(self, *args, **kwargs):
            return RefinementOutcome(
                status="failed",
                forward_evaluations=1,
                message="intentional failure",
            )

    result = run_verified_bounds_local_inference(
        inputs,
        curve,
        model=_LocalModel(branch.topology_id),
        branch_factory=factory,
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        inference_budget=_budget(total=4),
        rescue_policy=RescuePolicy(
            target_candidate_count=1,
            fallback_attempt_limit=3,
            sobol_seeds_per_branch=3,
        ),
        refiner=AlwaysFails(curve=curve),
        seed=37,
    )

    assert result.status == "no_candidate_found_within_budget"
    assert result.raw_generation.candidates == ()
    assert result.evaluation_report is None
    assert result.audit.forward_evaluations_used == 4
    assert "no_solution" not in result.status
