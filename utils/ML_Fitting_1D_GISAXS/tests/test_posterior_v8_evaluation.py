from __future__ import annotations

import json

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    LatentComponentParameters,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    EVALUATION_AUDIT_SCHEMA,
    PARAMETER_NORMALIZATION_VERSION,
    PROPOSAL_SCORE_SEMANTICS,
    CandidateInput,
    EvaluationThresholds,
    LinearSolutionSnapshot,
    ObservedCurve,
    ReferenceMode,
    evaluate_candidates,
    natural_log_rmse,
    normalized_latent_parameter_distance,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import ResolutionShape


def _component(shape="sphere", radius=10.0, width=0.1):
    kwargs = {}
    if shape == "cylinder":
        kwargs = {"log_h": np.log(30.0), "sigma_h_fraction": 0.12}
    return LatentComponentParameters(
        shape=shape,
        log_R=np.log(radius),
        sigma_R_fraction=width,
        **kwargs,
    )


def test_linear_solution_snapshot_preserves_an_explicit_noncanonical_gui_k():
    snapshot = LinearSolutionSnapshot(
        background=0.1,
        particle_amplitudes=(20.0, 5.0),
        resolution_amplitude=2.0,
        k=10.0,
    )
    assert snapshot.k == 10.0
    assert snapshot.component_weights == (2.0, 0.5)
    assert snapshot.int_res == 0.2
    assert snapshot.effective_component_fractions == pytest.approx((0.8, 0.2))
    assert snapshot.effective_resolution_ratio == pytest.approx(0.08)

    legacy = LinearSolutionSnapshot(
        background=0.1,
        particle_amplitudes=(20.0, 5.0),
        resolution_amplitude=2.0,
    )
    assert legacy.k == 25.0


def _candidate(
    candidate_id,
    rank,
    log_offset,
    *,
    components=None,
    bounds=True,
    physics=True,
    score=None,
    resolution=None,
    resolution_amplitude=None,
    particle_amplitudes=None,
    background=0.1,
):
    components = tuple(components or (_component(),))
    amplitudes = tuple(
        particle_amplitudes or tuple(float(index + 1) for index in range(len(components)))
    )
    linear_solution = LinearSolutionSnapshot(
        background=background,
        particle_amplitudes=amplitudes,
        resolution_amplitude=(
            (0.2 if resolution_amplitude is None else resolution_amplitude)
            if resolution is not None
            else 0.0
        ),
    )
    return CandidateInput(
        candidate_id=candidate_id,
        proposal_rank=rank,
        topology_id=topology_id_for(item.shape for item in components),
        components=components,
        resolution=resolution,
        linear_solution=linear_solution,
        exact_intensity=np.exp(np.arange(4, dtype=float) + log_offset),
        bounds_pass=bounds,
        physics_pass=physics,
        proposal_score_raw=score,
    )


def _curve(sigma_log=None, source_kind="synthetic"):
    return ObservedCurve(
        curve_id="curve-001",
        source_kind=source_kind,
        q=np.geomspace(0.01, 1.0, 4),
        intensity=np.exp(np.arange(4, dtype=float)),
        sigma_log=sigma_log,
    )


def _thresholds(**changes):
    values = {
        "raw_exact_log_rmse_max": 0.20,
        "standardized_exact_log_rmse_max": 1.50,
        "parameter_mode_distance_max": 0.02,
        "raw_curve_equivalence_log_rmse_max": 0.03,
        "reference_mode_distance_max": 0.03,
    }
    values.update(changes)
    return EvaluationThresholds(**values)


def test_natural_log_rmse_and_optional_sigma_log_weighting():
    observed = np.array([1.0, np.e, np.e**2])
    predicted = observed * np.exp(0.1)
    assert natural_log_rmse(observed, predicted) == pytest.approx(0.1)
    assert natural_log_rmse(observed, predicted, sigma_log=np.full(3, 0.02)) == pytest.approx(5.0)
    with pytest.raises(ValueError, match="strictly positive"):
        natural_log_rmse(observed, [1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="same shape"):
        natural_log_rmse(observed, predicted[:-1])


def test_exact_bounds_physics_gates_and_best_of_n_use_explicit_rank_not_score():
    candidates = (
        _candidate("rank-1", 1, 0.25, score=1000.0),
        _candidate("rank-2", 2, 0.10, bounds=False, score=-1000.0),
        _candidate("rank-3", 3, 0.05, score=0.0),
    )
    report = evaluate_candidates(
        _curve(), candidates, thresholds=_thresholds(), best_of_n=(1, 2, 3)
    )

    assert [item.accepted for item in report.candidates] == [False, False, True]
    assert [item.exact_valid for item in report.candidates] == [False, True, True]
    assert report.accepted_count == 1
    assert report.valid_precision == pytest.approx(1 / 3)
    assert report.bounds_pass_rate == pytest.approx(2 / 3)
    assert report.physics_pass_rate == 1.0
    assert report.best_of_n[0].best_candidate_id == "rank-1"
    assert report.best_of_n[-1].best_candidate_id == "rank-3"
    assert report.best_of_n[1].accepted_found is False
    assert report.best_of_n[-1].best_accepted_candidate_id == "rank-3"
    assert report.proposal_score_semantics == PROPOSAL_SCORE_SEMANTICS


def test_parameter_modes_then_curve_equivalence_keep_lowest_error_representative():
    near_a = _component(radius=10.1)
    far_b = _component(radius=30.0)
    cylinder = _component("cylinder", radius=12.0)
    candidates = (
        _candidate("a", 1, 0.02, components=(_component(radius=10.0),)),
        _candidate("a-duplicate", 2, 0.03, components=(near_a,)),
        _candidate("b", 3, 0.01, components=(far_b,)),
        _candidate("c", 4, 0.12, components=(cylinder,)),
    )
    report = evaluate_candidates(_curve(), candidates, thresholds=_thresholds(), best_of_n=(1, 4))

    assert len(report.parameter_modes) == 3
    assert report.parameter_modes[0].member_candidate_ids == ("a", "a-duplicate")
    assert len(report.curve_groups) == 2
    first_group = report.curve_groups[0]
    assert set(first_group.member_candidate_ids) == {"a", "a-duplicate", "b"}
    assert first_group.representative_candidate_id == "b"
    assert report.representative_candidate_ids == ("b", "c")
    assert report.duplicate_ratio == pytest.approx(0.25)
    assert report.curve_equivalence_redundancy_ratio == pytest.approx(1 / 3)


def test_repeated_shape_parameter_distance_is_permutation_invariant():
    components = (_component(radius=8.0), _component(radius=40.0))
    resolution = ResolutionShape(sigma_res=0.01, nu_res=5.0)
    left = _candidate(
        "left",
        1,
        0.01,
        components=components,
        resolution=resolution,
        resolution_amplitude=0.3,
        particle_amplitudes=(1.0, 2.0),
    )
    right = _candidate(
        "right",
        2,
        0.01,
        components=components[::-1],
        particle_amplitudes=(4.0, 2.0),
        background=0.2,
        resolution=resolution,
        resolution_amplitude=0.6,
    )
    assert normalized_latent_parameter_distance(left, right) == pytest.approx(0.0)
    report = evaluate_candidates(_curve(), (left, right), thresholds=_thresholds(), best_of_n=(2,))
    assert len(report.parameter_modes) == 1


def test_profiled_scale_invariant_composition_distinguishes_parameter_modes():
    particle_dominated = _candidate(
        "particle", 1, 0.01, particle_amplitudes=(10.0,), background=0.01
    )
    background_dominated = _candidate(
        "background", 2, 0.01, particle_amplitudes=(0.1,), background=10.0
    )
    assert normalized_latent_parameter_distance(particle_dominated, background_dominated) > 0.02
    report = evaluate_candidates(
        _curve(),
        (particle_dominated, background_dominated),
        thresholds=_thresholds(),
        best_of_n=(2,),
    )
    assert len(report.parameter_modes) == 2


def test_resolution_composition_is_part_of_full_parameter_distance():
    resolution = ResolutionShape(sigma_res=0.01, nu_res=5.0)
    weak_resolution = _candidate(
        "weak-resolution",
        1,
        0.01,
        resolution=resolution,
        resolution_amplitude=0.01,
    )
    strong_resolution = _candidate(
        "strong-resolution",
        2,
        0.01,
        resolution=resolution,
        resolution_amplitude=10.0,
    )

    assert normalized_latent_parameter_distance(weak_resolution, strong_resolution) > 0.02


def test_reference_mode_requires_profiled_composition_and_fails_closed():
    candidate = _candidate("candidate", 1, 0.01)
    with pytest.raises(TypeError, match="LinearSolutionSnapshot"):
        ReferenceMode(
            reference_id="missing-composition",
            topology_id=candidate.topology_id,
            components=candidate.components,
            resolution=None,
            linear_solution=None,
        )

    malformed = _candidate("malformed", 2, 0.01)
    object.__setattr__(malformed, "linear_solution", None)
    with pytest.raises(TypeError, match="scale-quotiented parameter distance requires"):
        normalized_latent_parameter_distance(malformed, candidate)


def test_reference_matching_uses_full_profiled_composition_not_only_geometry():
    candidate = _candidate("candidate", 1, 0.01)
    reference = ReferenceMode(
        reference_id="different-composition",
        topology_id=candidate.topology_id,
        components=candidate.components,
        resolution=None,
        linear_solution=LinearSolutionSnapshot(
            background=10.0,
            particle_amplitudes=(0.1,),
            resolution_amplitude=0.0,
        ),
    )

    assert normalized_latent_parameter_distance(candidate, reference) > 0.02
    report = evaluate_candidates(
        _curve(),
        (candidate,),
        thresholds=_thresholds(reference_mode_distance_max=0.02),
        best_of_n=(1,),
        reference_modes=(reference,),
    )
    assert report.mode_recall == 0.0
    assert report.reference_matches[0].nearest_candidate_id == "candidate"
    assert not report.reference_matches[0].recalled


def test_complete_linkage_prevents_parameter_mode_chaining():
    candidates = tuple(
        _candidate(
            candidate_id,
            rank,
            0.01,
            components=(_component(radius=radius),),
        )
        for candidate_id, rank, radius in (
            ("left", 1, 10.0),
            ("middle", 2, 11.0),
            ("right", 3, 12.1),
        )
    )
    assert normalized_latent_parameter_distance(candidates[0], candidates[1]) < 0.02
    assert normalized_latent_parameter_distance(candidates[1], candidates[2]) < 0.02
    assert normalized_latent_parameter_distance(candidates[0], candidates[2]) > 0.02

    report = evaluate_candidates(_curve(), candidates, thresholds=_thresholds(), best_of_n=(3,))
    assert len(report.parameter_modes) == 2
    assert report.clustering_linkage == "deterministic_agglomerative_complete_linkage/v2"


def test_reference_mode_recall_uses_topology_and_normalized_parameter_distance():
    sphere = _candidate("sphere", 1, 0.01, components=(_component(radius=10.0),))
    cylinder = _candidate("cylinder", 2, 0.02, components=(_component("cylinder", radius=12.0),))
    references = (
        ReferenceMode(
            reference_id="sphere-truth",
            topology_id=sphere.topology_id,
            components=(_component(radius=10.2),),
            resolution=None,
            linear_solution=sphere.linear_solution,
        ),
        ReferenceMode(
            reference_id="cylinder-truth",
            topology_id=cylinder.topology_id,
            components=(_component("cylinder", radius=12.1),),
            resolution=None,
            linear_solution=cylinder.linear_solution,
        ),
        ReferenceMode(
            reference_id="missing",
            topology_id=topology_id_for(["vertical_cylinder"]),
            components=(_component("vertical_cylinder", radius=18.0),),
            resolution=None,
            linear_solution=LinearSolutionSnapshot(
                background=0.1,
                particle_amplitudes=(1.0,),
                resolution_amplitude=0.0,
            ),
        ),
    )
    report = evaluate_candidates(
        _curve(),
        (sphere, cylinder),
        thresholds=_thresholds(),
        best_of_n=(2,),
        reference_modes=references,
    )
    assert [item.recalled for item in report.reference_matches] == [True, True, False]
    assert report.mode_recall == pytest.approx(2 / 3)
    assert report.reference_matches[-1].nearest_candidate_id is None
    assert report.reference_match_count == 2
    assert report.unmatched_compatible_mode_count == 0
    assert report.symmetric_chamfer_parameter_distance is not None
    assert report.hausdorff_parameter_distance == pytest.approx(1.0)


def test_reference_recall_uses_one_to_one_parameter_mode_matching():
    prediction = _candidate("single-mode", 1, 0.01, components=(_component(radius=10.0),))
    references = tuple(
        ReferenceMode(
            reference_id=reference_id,
            topology_id=prediction.topology_id,
            components=(_component(radius=radius),),
            resolution=None,
            linear_solution=prediction.linear_solution,
        )
        for reference_id, radius in (("mode-a", 10.0), ("mode-b", 10.1))
    )

    report = evaluate_candidates(
        _curve(),
        (prediction,),
        thresholds=_thresholds(reference_mode_distance_max=0.03),
        best_of_n=(1,),
        reference_modes=references,
    )

    # Both references have the same nearest prediction, but that prediction
    # can recall at most one distinct reference mode.
    assert [item.nearest_candidate_id for item in report.reference_matches] == [
        "single-mode",
        "single-mode",
    ]
    assert sum(item.recalled for item in report.reference_matches) == 1
    assert sum(item.matched_candidate_id is not None for item in report.reference_matches) == 1
    assert report.mode_recall == pytest.approx(0.5)
    assert report.reference_match_count == 1
    assert report.prediction_reference_match_fraction == pytest.approx(1.0)
    assert report.unmatched_compatible_mode_count == 0


def test_reference_set_distances_penalize_missing_topology_and_are_json_safe():
    sphere = _candidate("sphere", 1, 0.01, components=(_component(radius=10.0),))
    references = (
        ReferenceMode(
            reference_id="sphere-truth",
            topology_id=sphere.topology_id,
            components=(_component(radius=10.0),),
            resolution=None,
            linear_solution=sphere.linear_solution,
        ),
        ReferenceMode(
            reference_id="missing-cylinder",
            topology_id=topology_id_for(["cylinder"]),
            components=(_component("cylinder", radius=12.0),),
            resolution=None,
            linear_solution=LinearSolutionSnapshot(
                background=0.1,
                particle_amplitudes=(1.0,),
                resolution_amplitude=0.0,
            ),
        ),
    )

    report = evaluate_candidates(
        _curve(),
        (sphere,),
        thresholds=_thresholds(),
        best_of_n=(1,),
        reference_modes=references,
    )
    payload = json.loads(report.to_json())
    assert payload["audit_schema"] == EVALUATION_AUDIT_SCHEMA
    assert payload["parameter_normalization_version"] == PARAMETER_NORMALIZATION_VERSION

    assert report.symmetric_chamfer_parameter_distance == pytest.approx(0.25)
    assert report.hausdorff_parameter_distance == pytest.approx(1.0)
    assert payload["reference_set_unmatched_distance"] == 1.0
    assert payload["reference_matching_version"].startswith("maximum_cardinality")
    assert payload["reference_set_distance_version"].startswith("symmetric_chamfer")


def test_reference_matching_uses_closest_member_of_each_predicted_mode():
    representative = _candidate(
        "best-curve-representative",
        1,
        0.001,
        components=(_component(radius=11.5),),
    )
    closer_member = _candidate(
        "closest-parameter-member",
        2,
        0.01,
        components=(_component(radius=10.0),),
    )
    reference = ReferenceMode(
        reference_id="truth",
        topology_id=representative.topology_id,
        components=(_component(radius=10.0),),
        resolution=None,
        linear_solution=closer_member.linear_solution,
    )
    report = evaluate_candidates(
        _curve(),
        (representative, closer_member),
        thresholds=_thresholds(
            parameter_mode_distance_max=0.03,
            reference_mode_distance_max=0.005,
        ),
        best_of_n=(2,),
        reference_modes=(reference,),
    )

    assert len(report.parameter_modes) == 1
    assert report.parameter_modes[0].representative_candidate_id == ("best-curve-representative")
    assert report.reference_matches[0].recalled
    assert report.reference_matches[0].matched_candidate_id == "closest-parameter-member"
    assert report.reference_matches[0].matched_parameter_distance == pytest.approx(0.0)


def test_sigma_log_selects_an_independent_standardized_primary_gate():
    report = evaluate_candidates(
        _curve(sigma_log=np.full(4, 0.1)),
        (_candidate("only", 1, 0.05),),
        thresholds=_thresholds(
            raw_exact_log_rmse_max=0.01,
            standardized_exact_log_rmse_max=0.6,
        ),
        best_of_n=(1,),
    )
    assessment = report.candidates[0]
    assert assessment.raw_exact_log_rmse == pytest.approx(0.05)
    assert assessment.standardized_exact_log_rmse == pytest.approx(0.5)
    assert assessment.primary_gate_error == pytest.approx(0.5)
    assert assessment.exact_valid
    assert report.primary_gate_metric_name == "sigma_log_standardized_natural_log_rmse"
    assert report.primary_gate_threshold == 0.6
    assert report.raw_metric_name == "raw_natural_log_rmse"

    raw_curve_report = evaluate_candidates(
        _curve(sigma_log=np.full(4, 0.001)),
        (
            _candidate("first", 1, 0.01, components=(_component(radius=10.0),)),
            _candidate("second", 2, 0.02, components=(_component(radius=30.0),)),
        ),
        thresholds=_thresholds(standardized_exact_log_rmse_max=30.0),
        best_of_n=(2,),
    )
    assert len(raw_curve_report.parameter_modes) == 2
    assert len(raw_curve_report.curve_groups) == 1
    assert raw_curve_report.curve_equivalence_metric_name == "raw_natural_log_rmse"
    assert raw_curve_report.curve_equivalence_threshold == 0.03


def test_deterministic_noise_floor_raw_point12_standardized_one_is_accepted():
    truth_candidate = _candidate("generating-truth", 1, 0.12)
    thresholds = _thresholds(
        raw_exact_log_rmse_max=0.05,
        standardized_exact_log_rmse_max=1.5,
    )
    noisy_report = evaluate_candidates(
        _curve(sigma_log=np.full(4, 0.12)),
        (truth_candidate,),
        thresholds=thresholds,
        best_of_n=(1,),
    )
    noisy = noisy_report.candidates[0]
    assert noisy.raw_exact_log_rmse == pytest.approx(0.12)
    assert noisy.standardized_exact_log_rmse == pytest.approx(1.0)
    assert noisy.primary_gate_error == pytest.approx(1.0)
    assert noisy.accepted
    assert noisy_report.primary_gate_metric_name == "sigma_log_standardized_natural_log_rmse"
    assert noisy_report.primary_gate_threshold == 1.5

    noiseless_report = evaluate_candidates(
        _curve(), (truth_candidate,), thresholds=thresholds, best_of_n=(1,)
    )
    noiseless = noiseless_report.candidates[0]
    assert noiseless.raw_exact_log_rmse == pytest.approx(0.12)
    assert noiseless.standardized_exact_log_rmse is None
    assert not noiseless.accepted
    assert noiseless_report.primary_gate_metric_name == "raw_natural_log_rmse"
    assert noiseless_report.primary_gate_threshold == 0.05


def test_resolution_presence_and_shape_are_part_of_parameter_mode_distance():
    absent = _candidate("absent", 1, 0.01)
    resolution = ResolutionShape(sigma_res=0.01, nu_res=5.0)
    present = _candidate("present", 2, 0.01, resolution=resolution)
    shifted = _candidate("shifted", 3, 0.01, resolution=ResolutionShape(sigma_res=0.02, nu_res=7.0))
    assert np.isinf(normalized_latent_parameter_distance(absent, present))
    assert normalized_latent_parameter_distance(present, shifted) > 0.0

    report = evaluate_candidates(
        _curve(), (absent, present, shifted), thresholds=_thresholds(), best_of_n=(3,)
    )
    assert len(report.parameter_modes) == 3
    assert len(report.curve_groups) == 1


def test_real_cut_data_audit_is_json_safe_and_records_every_threshold():
    report = evaluate_candidates(
        _curve(source_kind="real_cut_data"),
        (_candidate("only", 1, 0.01, score=-17.0),),
        thresholds=_thresholds(),
        best_of_n=(1,),
    )
    payload = json.loads(report.to_json())
    assert payload["source_kind"] == "real_cut_data"
    assert payload["contract_version"].startswith("posterior_v8_contract_")
    assert payload["codec_version"].startswith("posterior_v8_")
    assert payload["forward_model_version"].startswith("gimap_fitting_")
    assert payload["thresholds"] == {
        "parameter_mode_distance_max": 0.02,
        "raw_curve_equivalence_log_rmse_max": 0.03,
        "raw_exact_log_rmse_max": 0.2,
        "reference_mode_distance_max": 0.03,
        "standardized_exact_log_rmse_max": 1.5,
    }
    assert payload["candidates"][0]["proposal_score_raw"] == -17.0
    assert payload["candidates"][0]["linear_solution"] == {
        "background": 0.1,
        "component_weights": [1.0],
        "int_res": 0.0,
        "k": 1.0,
        "particle_amplitudes": [1.0],
        "resolution_amplitude": 0.0,
    }
    assert payload["candidates"][0]["resolution"] is None
    assert "scale-invariant profiled" in payload["parameter_distance_scope"]
    assert "mandatory" in payload["parameter_distance_scope"]
    assert "when available" not in payload["parameter_distance_scope"]
    assert payload["proposal_score_semantics"] == "opaque_raw_score_not_probability"
    assert payload["mode_recall"] is None


def test_inputs_are_defensively_immutable_and_invalid_batches_fail_closed():
    q = np.geomspace(0.01, 1.0, 4)
    intensity = np.exp(np.arange(4, dtype=float))
    curve = ObservedCurve(
        curve_id="immutable",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
    )
    candidate = _candidate("one", 1, 0.01)
    q[0] = 99.0
    intensity[0] = 99.0
    assert curve.q[0] != 99.0 and curve.intensity[0] != 99.0
    with pytest.raises(ValueError):
        curve.q[0] = 0.1
    with pytest.raises(ValueError):
        candidate.exact_intensity[0] = 1.0

    with pytest.raises(ValueError, match="contiguous"):
        evaluate_candidates(
            curve,
            (candidate, _candidate("two", 3, 0.02)),
            thresholds=_thresholds(),
            best_of_n=(1,),
        )
    with pytest.raises(ValueError, match="within"):
        evaluate_candidates(curve, (candidate,), thresholds=_thresholds(), best_of_n=(2,))
    with pytest.raises(ValueError, match="does not match"):
        bad_length = CandidateInput(
            candidate_id="short",
            proposal_rank=1,
            topology_id=candidate.topology_id,
            components=candidate.components,
            resolution=candidate.resolution,
            linear_solution=candidate.linear_solution,
            exact_intensity=np.ones(3),
            bounds_pass=True,
            physics_pass=True,
        )
        evaluate_candidates(curve, (bad_length,), thresholds=_thresholds(), best_of_n=(1,))
    with pytest.raises(ValueError, match="non-negative"):
        EvaluationThresholds(
            raw_exact_log_rmse_max=-1.0,
            standardized_exact_log_rmse_max=1.5,
            parameter_mode_distance_max=0.1,
            raw_curve_equivalence_log_rmse_max=0.1,
            reference_mode_distance_max=0.1,
        )
    with pytest.raises(ValueError, match="finite"):
        CandidateInput(
            candidate_id="nan-score",
            proposal_rank=1,
            topology_id=candidate.topology_id,
            components=candidate.components,
            resolution=candidate.resolution,
            linear_solution=candidate.linear_solution,
            exact_intensity=np.ones(4),
            bounds_pass=True,
            physics_pass=True,
            proposal_score_raw=np.nan,
        )
    with pytest.raises(ValueError, match="absent resolution"):
        CandidateInput(
            candidate_id="inconsistent-linear-solution",
            proposal_rank=1,
            topology_id=candidate.topology_id,
            components=candidate.components,
            resolution=None,
            linear_solution=LinearSolutionSnapshot(
                background=0.1,
                particle_amplitudes=(1.0,),
                resolution_amplitude=0.2,
            ),
            exact_intensity=np.ones(4),
            bounds_pass=True,
            physics_pass=True,
        )
