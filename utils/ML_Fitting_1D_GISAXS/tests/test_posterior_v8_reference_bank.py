from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import competing_model_search, reference_bank
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_catalog import branch_pattern_id
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.compatibility_calibration import (
    CompatibilityCalibrationSample,
    CompatibilityStratum,
    fit_compatibility_calibration,
    make_acquisition_policy_id,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.competing_model_search import (
    CompetingSearchConfig,
    run_competing_model_search,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    GuiComponentParameters,
    gui_component_to_latent,
    latent_component_to_gui,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    CandidateInput,
    LinearSolutionSnapshot,
    ObservedCurve,
    ReferenceMode,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import (
    build_design_matrix,
    evaluate_profiled_forward,
    profile_linear_amplitudes,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.reference_bank import (
    EVALUATED_NETWORK_PROPOSAL_SOURCE,
    PRIMARY_REFERENCE_SEARCH_SOURCE,
    CalibrationObservationProvenance,
    CompatibilityPolicy,
    CompetingBranch,
    assess_candidate,
    branch_for_reference,
    complete_linkage_groups,
    enumerate_competing_branches,
    enumerate_legacy_competing_branches,
    generating_diagnostic,
    primary_reference_groups,
    strict_minimal_secondary_groups,
    write_reference_bank_atomic,
)


def _curve(*, noisy: bool = False) -> ObservedCurve:
    q = np.linspace(0.01, 0.5, 64)
    intensity = 1.0 + np.exp(-4.0 * q)
    return ObservedCurve(
        curve_id="curve-1",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
        sigma_log=np.full_like(q, 0.1) if noisy else None,
    )


def _latent(shape: str, radius: float):
    if shape == "cylinder":
        value = GuiComponentParameters(
            shape=shape,
            R=radius,
            sigma_R=0.1 * radius,
            h=40.0,
            sigma_h=4.0,
        )
    else:
        value = GuiComponentParameters(
            shape=shape,
            R=radius,
            sigma_R=0.1 if shape == "vertical_cylinder" else 0.1 * radius,
        )
    return gui_component_to_latent(value)


def _candidate(
    curve: ObservedCurve,
    *,
    candidate_id: str,
    rank: int,
    topology_id: int,
    components,
    amplitudes=None,
) -> CandidateInput:
    amplitudes = amplitudes or (1.0,) * len(components)
    return CandidateInput(
        candidate_id=candidate_id,
        proposal_rank=rank,
        topology_id=topology_id,
        components=tuple(components),
        resolution=None,
        linear_solution=LinearSolutionSnapshot(
            background=0.1,
            particle_amplitudes=tuple(amplitudes),
            resolution_amplitude=0.0,
        ),
        exact_intensity=curve.intensity,
        bounds_pass=True,
        physics_pass=True,
    )


def test_enumerates_canonical_418_catalog_and_keeps_legacy_baseline_explicit():
    all_branches = enumerate_competing_branches()
    pilot = enumerate_competing_branches(maximum_components=2)
    legacy = enumerate_legacy_competing_branches()

    assert len(all_branches) == 418
    assert len(pilot) == 54
    assert len(legacy) == 700
    assert len(set(all_branches)) == 418
    assert sum(len(branch.topology) == 1 for branch in pilot) == 12
    assert sum(len(branch.topology) == 2 for branch in pilot) == 42
    assert all(len(branch.d_present) == len(branch.topology) for branch in all_branches)


def test_complete_linkage_does_not_chain_distant_endpoint_modes():
    values = (0.0, 0.02, 0.04)
    groups = complete_linkage_groups(values, lambda left, right: abs(left - right), 0.03)

    assert sorted(len(group) for group in groups) == [1, 2]
    assert all(
        max(abs(left - right) for left in group for right in group) <= 0.03 for group in groups
    )


def test_reference_search_rejects_noncanonical_repeated_shape_branch():
    noncanonical = CompetingBranch(
        topology_id=3,
        pattern_id=branch_pattern_id((True, False, False, False), False),
    )

    with pytest.raises(ValueError, match="canonical 418-branch"):
        run_competing_model_search(
            _curve(),
            config=CompetingSearchConfig(seed=1, starts_per_branch=1, max_nfev=1),
            policy=CompatibilityPolicy(raw_log_rmse_max=0.1),
            branches=(noncanonical,),
        )


def test_compatibility_is_separate_from_visibility_and_effective_topology():
    q = np.linspace(0.01, 0.5, 64)
    components = (_latent("sphere", 20.0), _latent("sphere", 40.0))
    gui = tuple(latent_component_to_gui(value) for value in components)
    exact = build_design_matrix(q, gui) @ np.asarray((0.1, 1.0, 1.0e-5))
    curve = ObservedCurve(
        curve_id="weak-k2",
        source_kind="synthetic",
        q=q,
        intensity=exact,
    )
    candidate = CandidateInput(
        candidate_id="weak-k2",
        proposal_rank=1,
        topology_id=3,
        components=components,
        resolution=None,
        linear_solution=LinearSolutionSnapshot(
            background=0.1,
            particle_amplitudes=(1.0, 1.0e-5),
            resolution_amplitude=0.0,
        ),
        exact_intensity=exact,
        bounds_pass=True,
        physics_pass=True,
    )
    result = assess_candidate(
        curve,
        candidate,
        CompetingBranch(topology_id=3, pattern_id=0),
        attempt_rank=7,
        search_source=PRIMARY_REFERENCE_SEARCH_SOURCE,
        policy=CompatibilityPolicy(raw_log_rmse_max=0.05),
    )

    assert result.curve_compatible
    assert result.observability.particles[1].decision == "unneeded"
    assert result.effective_component_indices == ()
    assert result.effective_topology == ()
    assert not result.declared_components_observable
    assert result.primary_reference_eligible
    assert not result.strict_minimal_secondary_eligible
    assert not result.reference_eligible


def test_noisy_policy_supports_fixed_and_exact_stratified_calibration():
    q = np.linspace(0.01, 0.5, 64)
    component = _latent("sphere", 30.0)
    exact = build_design_matrix(q, (latent_component_to_gui(component),)) @ np.asarray((0.1, 1.0))
    curve = ObservedCurve(
        curve_id="noisy",
        source_kind="synthetic",
        q=q,
        intensity=exact,
        sigma_log=np.full_like(q, 0.1),
    )
    candidate = CandidateInput(
        candidate_id="noisy",
        proposal_rank=1,
        topology_id=0,
        components=(component,),
        resolution=None,
        linear_solution=LinearSolutionSnapshot(
            background=0.1,
            particle_amplitudes=(1.0,),
            resolution_amplitude=0.0,
        ),
        exact_intensity=exact,
        bounds_pass=True,
        physics_pass=True,
    )
    branch = CompetingBranch(topology_id=0, pattern_id=0)

    fixed = assess_candidate(
        curve,
        candidate,
        branch,
        attempt_rank=1,
        search_source=PRIMARY_REFERENCE_SEARCH_SOURCE,
        policy=CompatibilityPolicy(raw_log_rmse_max=0.05, standardized_log_rmse_max=1.5),
    )
    assert fixed.threshold_source == "fixed_standardized"
    assert fixed.primary_reference_eligible
    assert fixed.strict_minimal_secondary_eligible
    assert fixed.reference_eligible
    network_secondary = assess_candidate(
        curve,
        candidate,
        branch,
        attempt_rank=1,
        search_source=EVALUATED_NETWORK_PROPOSAL_SOURCE,
        policy=CompatibilityPolicy(raw_log_rmse_max=0.05, standardized_log_rmse_max=1.5),
    )
    assert not network_secondary.primary_reference_eligible
    assert network_secondary.strict_minimal_secondary_eligible
    assert "network" in network_secondary.primary_reference_qualification_reason

    stratum = CompatibilityStratum(
        point_count=64,
        noise_id="noise-v1",
        q_window_id="q-window-full-v1",
    )
    acquisition_policy_id = make_acquisition_policy_id(
        grid={
            "kind": "linear",
            "design_point_count": 64,
            "q_min": 0.01,
            "q_max": 0.5,
            "q_window_id": stratum.q_window_id,
        },
        mask={"mask_id": "none-v1", "point_keep_probability": 1.0},
        crop={"crop_id": "none-v1", "q_min": 0.01, "q_max": 0.5},
        view={
            "observation_view_version": "reference-test-view-v1",
            "view_index": 0,
            "observation_seed_derivation": "fixed-reference-test-v1",
        },
        sigma={
            "noise_id": stratum.noise_id,
            "poisson_count_scale": None,
            "relative_sigma": 0.1,
            "sigma_floor_fraction": 1.0e-6,
            "sigma_log_source": "provided-sigma-log",
        },
    )
    calibration = fit_compatibility_calibration(
        tuple(
            CompatibilityCalibrationSample(
                sample_id=f"sample-{index}",
                independent_group_id=f"recipe-{index}",
                stratum=stratum,
                score=score,
                effective_valid_point_count=64,
                acquisition_policy_id=acquisition_policy_id,
                measurement_sigma_available=True,
            )
            for index, score in enumerate((0.8, 0.9, 1.0, 1.1, 1.2))
        ),
        dataset_manifest_sha256="1" * 64,
        calibration_split_sha256="2" * 64,
        target_coverage=0.6,
        minimum_samples_per_stratum=5,
    )
    calibrated_policy = CompatibilityPolicy(
        raw_log_rmse_max=0.05,
        calibration=calibration,
        calibration_observation=CalibrationObservationProvenance(
            point_count=64,
            noise_id="noise-v1",
            q_window_id="q-window-full-v1",
        ),
    )
    calibrated = assess_candidate(
        curve,
        candidate,
        branch,
        attempt_rank=1,
        search_source=PRIMARY_REFERENCE_SEARCH_SOURCE,
        policy=calibrated_policy,
    )
    assert calibrated.threshold_source == "calibrated_stratum"
    assert not calibrated.used_calibration_fallback
    threshold, source, used_fallback = calibrated_policy.threshold_for(curve)
    assert threshold == pytest.approx(calibrated.compatibility_threshold)
    assert source == "calibrated_stratum"
    assert not used_fallback
    with pytest.raises(TypeError):
        calibrated_policy.threshold_for(curve, 2)
    policy_audit = competing_model_search._policy_payload(calibrated_policy)
    assert policy_audit["calibration_schema"] == calibration.schema
    assert policy_audit["calibration_stratum_fields"] == [
        "point_count",
        "noise_id",
        "q_window_id",
    ]
    assert "candidate_independent" in policy_audit["calibration_stratification_semantics"]


def test_d_present_is_fail_closed_until_independent_toggle_search_exists():
    q = np.geomspace(0.006, 1.5, 180)
    gui = GuiComponentParameters(
        shape="sphere",
        R=16.0,
        sigma_R=1.6,
        D=48.0,
        sigma_D=5.0,
    )
    latent = gui_component_to_latent(gui)
    exact = build_design_matrix(q, (gui,)) @ np.asarray((0.01, 1.0))
    curve = ObservedCurve(
        curve_id="d-observability",
        source_kind="synthetic",
        q=q,
        intensity=exact,
    )
    candidate = CandidateInput(
        candidate_id="d-present",
        proposal_rank=1,
        topology_id=0,
        components=(latent,),
        resolution=None,
        linear_solution=LinearSolutionSnapshot(
            background=0.01,
            particle_amplitudes=(1.0,),
            resolution_amplitude=0.0,
        ),
        exact_intensity=exact,
        bounds_pass=True,
        physics_pass=True,
    )
    reference = ReferenceMode(
        reference_id="d-present",
        topology_id=0,
        components=(latent,),
        resolution=None,
        linear_solution=candidate.linear_solution,
    )

    result = assess_candidate(
        curve,
        candidate,
        branch_for_reference(reference),
        attempt_rank=1,
        search_source=PRIMARY_REFERENCE_SEARCH_SOURCE,
        policy=CompatibilityPolicy(raw_log_rmse_max=0.01),
    )

    assert result.curve_compatible
    assert result.observability.particles[0].decision == "needed"
    assert result.observability.d_terms[0].decision == "unknown"
    assert not result.declared_components_observable
    assert result.primary_reference_eligible
    assert not result.strict_minimal_secondary_eligible
    assert not result.reference_eligible
    compatible_modes, compatible_curve_groups = primary_reference_groups(
        (result,), CompatibilityPolicy(raw_log_rmse_max=0.01)
    )
    assert len(compatible_modes) == 1
    assert len(compatible_curve_groups) == 1
    assert compatible_modes[0]["observability_status_counts"] == {
        "confirmed_effective": 0,
        "confirmed_redundant": 0,
        "provisional_or_unknown": 1,
    }
    assert compatible_modes[0]["reference_set_role"] == "primary_reference_denominator"
    assert "paper_reference_eligible" not in compatible_modes[0]


def test_observability_failure_is_post_search_unknown_not_primary_exclusion(monkeypatch):
    curve = _curve()
    candidate = _candidate(
        curve,
        candidate_id="post-search-observability-failure",
        rank=1,
        topology_id=0,
        components=(_latent("sphere", 20.0),),
    )

    def fail_observability(*args, **kwargs):
        raise RuntimeError("diagnostic unavailable")

    monkeypatch.setattr(reference_bank, "assess_candidate_observability", fail_observability)
    result = assess_candidate(
        curve,
        candidate,
        CompetingBranch(topology_id=0, pattern_id=0),
        attempt_rank=1,
        search_source=PRIMARY_REFERENCE_SEARCH_SOURCE,
        policy=CompatibilityPolicy(raw_log_rmse_max=0.01),
    )

    assert result.primary_reference_eligible
    assert not result.strict_minimal_secondary_eligible
    assert result.observability.status == "provisional_or_unknown"
    assert result.observability.exact_forward_calls == 0
    assert "diagnostic unavailable" in result.observability.diagnostic_error


def test_primary_qualification_is_fail_honest_for_empty_unsaturated_and_plateau():
    empty = competing_model_search._primary_reference_qualification(
        0,
        (
            {"delta_primary_reference_mode_count": 0},
            {"delta_primary_reference_mode_count": 0},
        ),
        ({"key": "branch", "preparation_status": "ready"},),
    )
    unsaturated = competing_model_search._primary_reference_qualification(
        1,
        (
            {"delta_primary_reference_mode_count": 1},
            {"delta_primary_reference_mode_count": 1},
        ),
        ({"key": "branch", "preparation_status": "ready"},),
    )
    plateau = competing_model_search._primary_reference_qualification(
        1,
        (
            {"delta_primary_reference_mode_count": 1},
            {"delta_primary_reference_mode_count": 0},
        ),
        ({"key": "branch", "preparation_status": "ready"},),
    )

    assert empty["qualification_status"] == "EMPTY_WITHIN_FINITE_SEARCH"
    assert unsaturated["qualification_status"] == "UNSATURATED_WITHIN_FINITE_SEARCH"
    assert plateau["qualification_status"] == "SCHEDULE_PLATEAU_OBSERVED"
    for value in (empty, unsaturated, plateau):
        assert value["completeness_certificate_status"] == "NOT_ESTABLISHED"
        assert value["paper_freeze_status"] == "NOT_QUALIFIED_NO_INDEPENDENT_CERTIFICATE"
        assert value["does_not_establish_all_solutions_or_no_solution"]


def test_generating_recovery_and_competing_model_ambiguity_are_distinct():
    q = np.linspace(0.01, 0.5, 64)
    sphere_gui = latent_component_to_gui(_latent("sphere", 30.0))
    cylinder_gui = latent_component_to_gui(_latent("cylinder", 30.0))
    intensity = build_design_matrix(q, (sphere_gui,)) @ np.asarray((0.1, 1.0))
    curve = ObservedCurve(
        curve_id="ambiguous",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
    )
    policy = CompatibilityPolicy(
        raw_log_rmse_max=0.05,
        curve_equivalence_log_rmse_max=0.02,
    )
    sphere_profile = profile_linear_amplitudes(q, intensity, (sphere_gui,), sigma=intensity)
    cylinder_profile = profile_linear_amplitudes(q, intensity, (cylinder_gui,), sigma=intensity)
    sphere = CandidateInput(
        candidate_id="sphere",
        proposal_rank=1,
        topology_id=0,
        components=(_latent("sphere", 30.0),),
        resolution=None,
        linear_solution=LinearSolutionSnapshot.from_profiled_forward(sphere_profile),
        exact_intensity=evaluate_profiled_forward(q, sphere_profile),
        bounds_pass=True,
        physics_pass=True,
    )
    cylinder = CandidateInput(
        candidate_id="cylinder",
        proposal_rank=2,
        topology_id=1,
        components=(_latent("cylinder", 30.0),),
        resolution=None,
        linear_solution=LinearSolutionSnapshot.from_profiled_forward(cylinder_profile),
        exact_intensity=evaluate_profiled_forward(q, cylinder_profile),
        bounds_pass=True,
        physics_pass=True,
    )
    discoveries = (
        assess_candidate(
            curve,
            sphere,
            CompetingBranch(topology_id=0, pattern_id=0),
            attempt_rank=1,
            search_source=PRIMARY_REFERENCE_SEARCH_SOURCE,
            policy=policy,
        ),
        assess_candidate(
            curve,
            cylinder,
            CompetingBranch(topology_id=1, pattern_id=0),
            attempt_rank=2,
            search_source=PRIMARY_REFERENCE_SEARCH_SOURCE,
            policy=policy,
        ),
    )
    modes, curve_groups = primary_reference_groups(discoveries, policy)
    strict_modes, _ = strict_minimal_secondary_groups(discoveries, policy)
    reference = ReferenceMode(
        reference_id="generating",
        topology_id=0,
        components=sphere.components,
        resolution=None,
        linear_solution=sphere.linear_solution,
    )
    diagnostic = generating_diagnostic(
        discoveries, modes, strict_modes, reference, distance_max=0.05
    )

    assert len(modes) == 2
    assert len(curve_groups) == 1
    assert diagnostic["generating_mode_recovered"]
    assert diagnostic["competing_model_ambiguity_discovered"]
    assert diagnostic["alternative_primary_reference_branch_keys"] == ["topology_01:branch_00"]


def test_search_round_robin_budget_saturation_and_atomic_artifact(monkeypatch, tmp_path):
    curve = _curve()

    def fake_refinement(q, intensity, **kwargs):
        components = tuple(kwargs["seed_components"])
        resolution = kwargs["resolution_seed"]
        profile = profile_linear_amplitudes(
            q,
            intensity,
            tuple(latent_component_to_gui(value) for value in components),
            resolution=resolution,
            sigma=intensity,
        )
        return SimpleNamespace(
            final_profile=profile,
            final_latent_components=components,
            final_resolution=resolution,
            exact_forward_intensity=evaluate_profiled_forward(q, profile),
            bounds_satisfied=True,
            success=True,
            nfev=3,
            residual_calls=4,
            initial_log_rmse=0.2,
            final_log_rmse=0.0,
        )

    monkeypatch.setattr(competing_model_search, "refine_profiled_branch", fake_refinement)
    branches = (
        CompetingBranch(topology_id=0, pattern_id=0),
        CompetingBranch(topology_id=1, pattern_id=0),
    )
    payload = run_competing_model_search(
        curve,
        config=CompetingSearchConfig(
            seed=7,
            starts_per_branch=2,
            max_nfev=5,
            saturation_rounds=(1, 2),
        ),
        policy=CompatibilityPolicy(raw_log_rmse_max=1.0),
        branches=branches,
    )

    assert payload["state"] == "COMPUTATION_COMPLETE"
    assert payload["primary_search_budget"] == {
        **payload["primary_search_budget"],
        "scheduled_attempts": 4,
        "attempted_refinements": 4,
        "returned_candidates": 4,
        "optimizer_converged": 4,
        "refinement_nfev": 12,
        "residual_calls": 16,
    }
    assert payload["primary_reference_inventory"]["branch_keys"] == [
        "topology_00:branch_00",
        "topology_01:branch_00",
    ]
    assert [
        value["scheduled_attempt_budget"] for value in payload["primary_mode_discovery_saturation"]
    ] == [2, 4]
    assert payload["primary_search_budget"]["observability_calls_included"] is False
    assert payload["post_search_observability_budget"]["total_exact_forward_calls"] == 4
    assert (
        payload["post_search_observability_budget"]["affects_primary_search_budget_or_saturation"]
        is False
    )
    assert "minimum_particle_weight" not in payload["compatibility_policy"]
    assert "minimum_resolution_ratio" not in payload["compatibility_policy"]
    assert payload["candidates"][0]["observability"]["policy_version"]
    assert payload["branch_enumeration_version"].startswith("posterior_v8_canonical_418")
    assert payload["search_space"]["full_canonical_catalog_branch_count"] == 418
    assert isinstance(payload["primary_reference_modes"], list)
    assert isinstance(payload["primary_reference_curve_equivalence_groups"], list)
    assert isinstance(payload["strict_minimal_secondary_modes"], list)
    assert payload["primary_reference_source_policy"]["denominator_is_network_free"]
    assert not payload["primary_reference_source_policy"][
        "evaluated_network_proposals_allowed_in_primary_denominator"
    ]
    assert (
        payload["primary_reference_qualification"]["completeness_certificate_status"]
        == "NOT_ESTABLISHED"
    )
    assert payload["primary_reference_qualification"][
        "does_not_establish_all_solutions_or_no_solution"
    ]
    for legacy in (
        "reference_modes",
        "compatible_parameter_modes",
        "search_budget",
        "mode_discovery_saturation",
    ):
        assert legacy not in payload
    assert len(payload["scientific_payload_sha256"]) == 64

    path = tmp_path / "reference-bank.json"
    write_reference_bank_atomic(path, payload)
    assert path.exists()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_reference_bank_atomic(path, payload)

    old_schema = copy.deepcopy(payload)
    old_schema["schema"] = "gisaxs.posterior_v8.discovered_mode_reference_bank/v1"
    with pytest.raises(ValueError, match="incompatible reference-bank schema"):
        write_reference_bank_atomic(tmp_path / "old-schema.json", old_schema)

    legacy_policy = copy.deepcopy(payload)
    legacy_policy["compatibility_policy"]["minimum_particle_weight"] = 0.01
    with pytest.raises(ValueError, match="legacy coefficient observability"):
        write_reference_bank_atomic(tmp_path / "legacy-policy.json", legacy_policy)

    mixed_semantics = copy.deepcopy(payload)
    mixed_semantics["candidates"][0]["reference_eligible"] = True
    with pytest.raises(ValueError, match="legacy reference_eligible"):
        write_reference_bank_atomic(tmp_path / "mixed-semantics.json", mixed_semantics)

    tampered = copy.deepcopy(payload)
    tampered["candidates"][0]["raw_log_rmse"] = 123.0
    with pytest.raises(ValueError, match="digest does not match"):
        write_reference_bank_atomic(tmp_path / "tampered.json", tampered)
