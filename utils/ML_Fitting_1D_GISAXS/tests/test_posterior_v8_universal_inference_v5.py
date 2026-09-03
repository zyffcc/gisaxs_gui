from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_query_v5 import V5AmplitudeQuery
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.bounds_query_v5 import (
    AxisRangeDesign,
    V5BoundsQuery,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.candidate_refinement_v5 import (
    v5_external_local_refinement_seed,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    SPHERE,
    VERTICAL_CYLINDER,
    ClosedInterval,
    GuiComponentBounds,
    latent_component_to_gui,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    EvaluationThresholds,
    ObservedCurve,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.preprocessing import preprocess_curve
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import build_design_matrix
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.proposal_execution_policy_v5 import (
    V5_PROPOSAL_EXECUTION_POLICY,
    V5_PROPOSAL_EXECUTION_POLICY_SHA256,
    V5ProposalExecutionPolicy,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.uncertainty_provenance_v5 import (
    V5UncertaintyProvenance,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_inference_contract_v5 import (
    V5_UNIVERSAL_INFERENCE_BUDGET_SCHEMA,
    V5_UNIVERSAL_INFERENCE_BUDGET_VERSION,
    V5UniversalInferenceBudget,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_inference_v5 import (
    run_v5_universal_one_click_inference,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.universal_query_v5 import (
    V5TopologyQuery,
    build_v5_universal_candidate_context,
)


def _geometry_query(shape: str, query_seed: int) -> V5BoundsQuery:
    bounds = (
        GuiComponentBounds(
            shape,
            R=ClosedInterval(12.0, 12.0),
            sigma_R=(
                ClosedInterval(0.2, 0.2)
                if shape == VERTICAL_CYLINDER
                else ClosedInterval(1.2, 1.2)
            ),
        ),
    )
    return V5BoundsQuery.create(
        query_seed=query_seed,
        generation_attempt=0,
        component_bounds=bounds,
        resolution_presence_policy="absent",
        resolution_bounds=None,
        axis_designs=(
            AxisRangeDesign("component[0].R", "fixed", "interior"),
            AxisRangeDesign("component[0].sigma_R", "fixed", "interior"),
        ),
    )


def _topology_query(shape: str, query_seed: int) -> V5TopologyQuery:
    geometry = _geometry_query(shape, query_seed)
    amplitude = V5AmplitudeQuery.create(
        background=ClosedInterval(0.0, 10.0),
        k=ClosedInterval(1.0e-3, 10.0),
        component_intensities=(ClosedInterval(1.0, 1.0),),
        resolution_presence_policy="absent",
        int_res=None,
    )
    return V5TopologyQuery(geometry, amplitude)


def _case(*, missing_sigma=False):
    sphere = _topology_query(SPHERE, 11)
    vertical = _topology_query(VERTICAL_CYLINDER, 12)
    codec = vertical.geometry.codec_for(0)
    components, _ = codec.decode(np.full(26, 0.5))
    q = np.geomspace(0.008, 1.2, 72)
    intensity = build_design_matrix(
        q, tuple(latent_component_to_gui(value) for value in components)
    ) @ np.asarray((0.1, 2.0))
    relative_sigma = 0.015 if missing_sigma else 0.02
    sigma = relative_sigma * intensity
    preprocessed = preprocess_curve(q, intensity, sigma)
    uncertainty = (
        V5UncertaintyProvenance("encoder_proxy_missing_sigma", relative_sigma)
        if missing_sigma
        else V5UncertaintyProvenance("simulated_sigma")
    )
    context = build_v5_universal_candidate_context(
        preprocessed,
        uncertainty,
        (sphere, vertical),
    )
    observed = ObservedCurve(
        curve_id="universal-v5",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
        sigma_log=(None if missing_sigma else np.full(q.shape, relative_sigma)),
    )
    return context, observed


class _Model:
    def __init__(self, scores):
        self.scores = dict(scores)
        self.calls = []

    def __call__(self, inputs, *, training):
        self.calls.append((inputs, training))
        topology_ids = np.asarray(inputs["branch_topology_id"]).reshape(-1)
        count = topology_ids.size
        modes = V5_PROPOSAL_EXECUTION_POLICY.mixture_component_count
        locations = np.zeros((count, modes, 26), dtype=np.float32)
        return {
            "proposal_search_yield_logit": np.asarray(
                [[self.scores[int(value)]] for value in topology_ids], dtype=np.float32
            ),
            "mixture_logits": np.zeros((count, modes), dtype=np.float32),
            "mixture_loc": locations,
            "mixture_logscale": np.full_like(locations, -8.0),
        }


class _TwoMixtureModel(_Model):
    def __call__(self, inputs, *, training):
        self.calls.append((inputs, training))
        topology_ids = np.asarray(inputs["branch_topology_id"]).reshape(-1)
        count = topology_ids.size
        modes = V5_PROPOSAL_EXECUTION_POLICY.mixture_component_count
        locations = np.zeros((count, modes, 26), dtype=np.float32)
        locations[:, 1, :] = 1.0
        logits = np.full((count, modes), -10.0, dtype=np.float32)
        logits[:, :2] = (2.0, 0.0)
        return {
            "proposal_search_yield_logit": np.asarray(
                [[self.scores[int(value)]] for value in topology_ids], dtype=np.float32
            ),
            "mixture_logits": logits,
            "mixture_loc": locations,
            "mixture_logscale": np.full_like(locations, -8.0),
        }


def _model():
    return _Model(
        {
            topology_id_for((SPHERE,)): -3.0,
            topology_id_for((VERTICAL_CYLINDER,)): 4.0,
        }
    )


def _budget(**changes):
    return replace(
        V5UniversalInferenceBudget(
            mixture_components_per_branch=(
                V5_PROPOSAL_EXECUTION_POLICY.per_branch_top_l
            ),
            stochastic_draws_per_mixture=0,
            include_mixture_medians=True,
            neural_primary_attempt_limit=2,
            fallback_attempt_limit=0,
            sobol_seeds_per_branch=0,
            per_candidate_forward_evaluation_limit=4,
            forward_evaluation_limit=8,
        ),
        **changes,
    )


def _thresholds(*, loose=False):
    value = 1.0e6 if loose else 1.0e-5
    return EvaluationThresholds(
        raw_exact_log_rmse_max=value,
        standardized_exact_log_rmse_max=value,
        parameter_mode_distance_max=0.05,
        raw_curve_equivalence_log_rmse_max=value,
        reference_mode_distance_max=0.10,
    )


def test_one_batched_model_call_globally_ranks_branches_and_stops_on_exact_mode():
    context, observed = _case()
    model = _model()
    result = run_v5_universal_one_click_inference(
        context,
        observed,
        model=model,
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        budget=_budget(),
        seed=31,
    )

    vertical_id = topology_id_for((VERTICAL_CYLINDER,))
    assert len(model.calls) == 1 and model.calls[0][1] is False
    assert model.calls[0][0]["x"].shape[0] == context.branch_count
    assert result.globally_ranked_branch_keys[0] == f"topology-{vertical_id:02d}:wire-00"
    assert result.status == result.termination_reason == "compatible_target_reached"
    assert [value.topology_id for value in result.candidates] == [vertical_id]
    assert result.evaluation_call_count == 1
    assert result.evaluation_report is not None
    assert result.evaluation_report.accepted_count == 1
    assert result.compatible_representatives == result.candidates
    audit = result.to_audit_dict()
    assert (
        audit["proposal_execution_policy_sha256"]
        == V5_PROPOSAL_EXECUTION_POLICY_SHA256
    )


def test_parameter_modes_stay_topology_local_while_curve_groups_may_cross_topology():
    context, observed = _case()
    result = run_v5_universal_one_click_inference(
        context,
        observed,
        model=_model(),
        thresholds=_thresholds(loose=True),
        target_parameter_mode_count=2,
        budget=_budget(),
        seed=32,
    )

    report = result.evaluation_report
    assert report is not None
    assert len(report.parameter_modes) == 2
    assert {value.topology_id for value in report.parameter_modes} == {
        topology_id_for((SPHERE,)),
        topology_id_for((VERTICAL_CYLINDER,)),
    }
    assert len(report.curve_groups) == 1
    assert report.curve_equivalence_redundancy_ratio == pytest.approx(0.5)
    assert result.evaluation_call_count == 2


def test_neural_primary_schedule_visits_ranked_branches_before_second_mixture():
    context, observed = _case()
    base = _model()
    model = _TwoMixtureModel(base.scores)
    result = run_v5_universal_one_click_inference(
        context,
        observed,
        model=model,
        thresholds=_thresholds(loose=True),
        target_parameter_mode_count=2,
        budget=_budget(
            neural_primary_attempt_limit=2,
        ),
        seed=3201,
    )

    assert result.status == "compatible_target_reached"
    first_two = result.candidate_provenance[:2]
    assert len(first_two) == 2
    assert first_two[0].global_branch_key != first_two[1].global_branch_key
    assert [value.attempt_rank for value in first_two] == [1, 2]


def test_retrieval_precedes_sobol_rescue_and_uses_the_same_exact_path():
    context, observed = _case()
    vertical_id = topology_id_for((VERTICAL_CYLINDER,))
    batch = next(value for value in context.batches if value.query.topology_id == vertical_id)
    retrieval = v5_external_local_refinement_seed(
        batch,
        source="retrieval",
        source_id="retrieval-nearest-0001",
        pattern_id=0,
        local_unit=np.full(26, 0.5),
    )
    result = run_v5_universal_one_click_inference(
        context,
        observed,
        model=_model(),
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        budget=_budget(
            neural_primary_attempt_limit=0,
            fallback_attempt_limit=2,
            sobol_seeds_per_branch=1,
        ),
        retrieval_seeds=(retrieval,),
        seed=33,
    )

    assert result.candidate_provenance[0].source == "retrieval"
    assert result.attempts[0].stage == "retrieval_rescue"
    sobol = next(value for value in result.source_accounting if value.source == "sobol")
    assert sobol.potential_seed_count == context.branch_count
    assert sobol.materialized_seed_count == 0


def test_automatic_sobol_rescue_follows_ranked_branches():
    context, observed = _case()
    result = run_v5_universal_one_click_inference(
        context,
        observed,
        model=_model(),
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        budget=_budget(
            neural_primary_attempt_limit=0,
            fallback_attempt_limit=1,
            sobol_seeds_per_branch=1,
        ),
        seed=34,
    )

    assert result.candidate_provenance[0].source == "sobol"
    assert result.candidate_provenance[0].global_branch_key == (
        result.globally_ranked_branch_keys[0]
    )
    assert result.source_accounting[2].materialized_seed_count == 1


def test_exact_forward_budget_is_shared_across_topologies_and_never_overruns():
    context, observed = _case()
    result = run_v5_universal_one_click_inference(
        context,
        observed,
        model=_model(),
        thresholds=_thresholds(loose=True),
        target_parameter_mode_count=2,
        budget=_budget(forward_evaluation_limit=1),
        seed=35,
    )

    assert result.status == "compatible_partial"
    assert result.termination_reason == "exact_forward_budget_exhausted"
    assert result.forward_evaluations_used == 1
    assert result.forward_evaluations_remaining == 0
    assert len(result.attempts) == len(result.candidates) == 1
    assert not result.all_scheduled_seeds_processed


def test_zero_budget_is_honest_no_candidate_within_budget_not_no_solution():
    context, observed = _case()
    model = _model()
    result = run_v5_universal_one_click_inference(
        context,
        observed,
        model=model,
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        budget=_budget(
            neural_primary_attempt_limit=0,
            fallback_attempt_limit=1,
            sobol_seeds_per_branch=1,
            forward_evaluation_limit=0,
        ),
        seed=36,
    )

    assert len(model.calls) == 1
    assert result.status == "no_candidate_found_within_budget"
    assert result.termination_reason == "exact_forward_budget_exhausted"
    assert result.evaluation_report is None
    assert not result.attempts and not result.candidates
    assert result.source_accounting[2].materialized_seed_count == 0
    assert "not_no_solution" in result.scientific_claim


def test_mismatched_exact_curve_fails_before_the_model_is_called():
    context, observed = _case()
    changed = ObservedCurve(
        curve_id=observed.curve_id,
        source_kind=observed.source_kind,
        q=observed.q,
        intensity=2.0 * observed.intensity,
        sigma_log=observed.sigma_log,
    )
    model = _model()
    with pytest.raises(ValueError, match="not built from this ObservedCurve"):
        run_v5_universal_one_click_inference(
            context,
            changed,
            model=model,
            thresholds=_thresholds(),
            target_parameter_mode_count=1,
            budget=_budget(),
        )
    assert not model.calls


@pytest.mark.parametrize("missing_sigma", (False, True))
def test_acceptance_uncertainty_cannot_switch_between_evidence_and_encoder_proxy(
    missing_sigma,
):
    context, observed = _case(missing_sigma=missing_sigma)
    changed = ObservedCurve(
        curve_id=observed.curve_id,
        source_kind=observed.source_kind,
        q=observed.q,
        intensity=observed.intensity,
        sigma_log=(
            np.full(observed.q.shape, 0.015) if missing_sigma else None
        ),
    )
    model = _model()
    expected = "must not become" if missing_sigma else "requires acceptance"
    with pytest.raises(ValueError, match=expected):
        run_v5_universal_one_click_inference(
            context,
            changed,
            model=model,
            thresholds=_thresholds(),
            target_parameter_mode_count=1,
            budget=_budget(),
        )
    assert not model.calls


def test_stale_retrieval_query_fails_before_model_or_exact_budget():
    context, observed = _case()
    vertical_id = topology_id_for((VERTICAL_CYLINDER,))
    batch = next(value for value in context.batches if value.query.topology_id == vertical_id)
    current = v5_external_local_refinement_seed(
        batch,
        source="retrieval",
        source_id="stale-retrieval",
        pattern_id=0,
        local_unit=np.full(26, 0.5),
    )
    stale = replace(current, query_sha256="0" * 64)
    model = _model()
    with pytest.raises(ValueError, match="stale retrieval seed"):
        run_v5_universal_one_click_inference(
            context,
            observed,
            model=model,
            thresholds=_thresholds(),
            target_parameter_mode_count=1,
            budget=_budget(),
            retrieval_seeds=(stale,),
        )
    assert not model.calls


def test_real_v5_keras_graph_accepts_the_single_universal_batch():
    pytest.importorskip("tensorflow")
    from utils.ML_Fitting_1D_GISAXS.PosteriorV8.model_v5 import (
        build_branch_conditioned_proposal_model,
    )

    context, observed = _case()
    model = build_branch_conditioned_proposal_model(
        max_points=1000,
        width=8,
        encoder_blocks=1,
        mixture_components=V5_PROPOSAL_EXECUTION_POLICY.mixture_component_count,
    )
    result = run_v5_universal_one_click_inference(
        context,
        observed,
        model=model,
        thresholds=_thresholds(),
        target_parameter_mode_count=1,
        budget=_budget(forward_evaluation_limit=0),
        seed=37,
    )

    assert result.model_call_count == 1
    assert len(result.globally_ranked_branch_keys) == context.branch_count
    assert result.status == "no_candidate_found_within_budget"


def test_proposal_policy_mismatch_and_wrong_model_mixture_count_fail_closed():
    with pytest.raises(TypeError, match="mixture_component_count"):
        V5ProposalExecutionPolicy(mixture_component_count=True)
    with pytest.raises(ValueError, match="policy is frozen"):
        V5ProposalExecutionPolicy(per_branch_top_l=3)
    with pytest.raises(ValueError, match="proposal_execution_policy_sha256"):
        V5UniversalInferenceBudget(proposal_execution_policy_sha256="0" * 64)
    with pytest.raises(ValueError, match="mixture_components_per_branch"):
        V5UniversalInferenceBudget(mixture_components_per_branch=3)
    with pytest.raises(ValueError, match="include_mixture_medians"):
        V5UniversalInferenceBudget(
            include_mixture_medians=False,
            stochastic_draws_per_mixture=1,
        )
    context, observed = _case()

    class WrongMixtureCount(_Model):
        def __call__(self, inputs, *, training):
            output = super().__call__(inputs, training=training)
            output["mixture_logits"] = output["mixture_logits"][:, :11]
            output["mixture_loc"] = output["mixture_loc"][:, :11, :]
            output["mixture_logscale"] = output["mixture_logscale"][:, :11, :]
            return output

    with pytest.raises(ValueError, match="model mixture count"):
        run_v5_universal_one_click_inference(
            context,
            observed,
            model=WrongMixtureCount(_model().scores),
            thresholds=_thresholds(),
            target_parameter_mode_count=1,
            budget=_budget(),
        )
    budget_audit = V5UniversalInferenceBudget().audit_payload()
    assert (
        budget_audit["proposal_execution_policy_sha256"]
        == V5_PROPOSAL_EXECUTION_POLICY_SHA256
    )
    assert budget_audit["schema"] == V5_UNIVERSAL_INFERENCE_BUDGET_SCHEMA
    assert budget_audit["version"] == V5_UNIVERSAL_INFERENCE_BUDGET_VERSION
