from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.component_observability import (
    DELETION_EVALUATED,
    DELETION_FORWARD_BUDGET_EXHAUSTED,
    DELETION_K0_EVALUATED,
    ObservabilityPolicy,
    ReducedModelSearchEvidence,
    assess_candidate_observability,
    diagnose_component_observability,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    full_component_bounds,
    gui_component_to_latent,
    topology_id_for,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.reduced_model_search import (
    BoundedReducedModelSearcher,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    RAW_LOG_RMSE_METRIC,
    CandidateInput,
    LinearSolutionSnapshot,
    ObservedCurve,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import (
    NonlinearComponent,
    ResolutionShape,
    build_design_matrix,
    component_unit_basis,
    evaluate_profiled_forward,
    profile_linear_amplitudes,
)


def _curve(q, components, coefficients, resolution=None):
    return build_design_matrix(q, components, resolution) @ np.asarray(
        coefficients, dtype=np.float64
    )


def test_equal_coefficients_are_not_mistaken_for_equal_observability():
    q = np.geomspace(0.004, 2.0, 400)
    components = (
        NonlinearComponent("sphere", R=10.0, sigma_R=1.0),
        NonlinearComponent("vertical cylinder", R=10.0, sigma_R=0.1),
    )
    intensity = _curve(q, components, [0.01, 1.0, 1.0])
    profile = profile_linear_amplitudes(q, intensity, components, sigma=intensity)

    report = diagnose_component_observability(q, intensity, profile)

    assert profile.particle_amplitudes == pytest.approx((1.0, 1.0), rel=1e-10)
    assert profile.component_weights == pytest.approx((0.5, 0.5), rel=1e-10)
    assert [item.shape for item in report.components] == [
        "sphere",
        "vertical_cylinder",
    ]
    assert [item.remaining_component_indices for item in report.components] == [
        (1,),
        (0,),
    ]
    assert all(item.deletion_status == DELETION_EVALUATED for item in report.components)

    sphere, vertical = report.components
    assert (
        sphere.contribution.relative_to_observed_rms
        > 100.0 * vertical.contribution.relative_to_observed_rms
    )
    assert sphere.raw_log_score_increment > 100.0 * vertical.raw_log_score_increment
    assert sphere.reduced_profile is not None
    assert sphere.reduced_profile.components == (components[1],)
    assert sphere.reduced_profile.particle_amplitudes[0] > 100.0
    assert vertical.reduced_profile is not None
    assert vertical.reduced_profile.components == (components[0],)
    assert report.full_scores.raw_log_rmse < 1e-12
    assert report.full_scores.standardized_log_rmse is None


def test_noisy_diagnostics_include_resolution_background_and_are_scale_invariant():
    q = np.geomspace(0.004, 2.0, 400)
    components = (
        NonlinearComponent("sphere", R=12.0, sigma_R=1.2, D=38.0, sigma_D=4.0),
        NonlinearComponent(
            "vertical cylinder",
            R=24.0,
            sigma_R=0.18,
            D=72.0,
            sigma_D=7.0,
        ),
    )
    resolution = ResolutionShape(sigma_res=0.015, nu_res=6.0)
    clean = _curve(q, components, [0.02, 0.9, 3.2, 0.18], resolution)
    phase = np.linspace(0.0, 9.0 * np.pi, q.size)
    sigma_log = 0.025 + 0.015 * np.square(np.sin(np.linspace(0.0, 4.0 * np.pi, q.size)))
    observed = clean * np.exp(0.25 * sigma_log * np.sin(phase))

    profile = profile_linear_amplitudes(
        q,
        observed,
        components,
        resolution=resolution,
        sigma=observed * sigma_log,
    )
    report = diagnose_component_observability(q, observed, profile, sigma_log=sigma_log)

    np.testing.assert_allclose(
        report.full_exact_intensity,
        evaluate_profiled_forward(q, profile),
        rtol=0.0,
        atol=0.0,
    )
    assert report.full_scores.raw_log_rmse > 0.0
    assert report.full_scores.standardized_log_rmse > 0.0
    assert report.background_contribution.label == "background"
    assert report.background_contribution.measurement_standardized_rms > 0.0
    assert report.resolution_contribution is not None
    assert report.resolution_contribution.label == "resolution"
    assert report.resolution_contribution.measurement_standardized_rms > 0.0
    assert all(item.deletion_status == DELETION_EVALUATED for item in report.components)
    assert all(item.reduced_profile.resolution == resolution for item in report.components)
    assert all(
        item.reduced_profile.resolution_amplitude != pytest.approx(profile.resolution_amplitude)
        for item in report.components
    )
    for item in report.components:
        assert item.raw_log_score_increment == pytest.approx(
            item.reduced_scores.raw_log_rmse - report.full_scores.raw_log_rmse
        )
        assert item.standardized_log_score_increment == pytest.approx(
            item.reduced_scores.standardized_log_rmse - report.full_scores.standardized_log_rmse
        )

    first_curve = profile.particle_amplitudes[0] * component_unit_basis(q, components[0])
    expected_standardized_rms = np.sqrt(np.mean(np.square(first_curve / (observed * sigma_log))))
    assert report.components[0].contribution.measurement_standardized_rms == pytest.approx(
        expected_standardized_rms, rel=1e-13
    )

    scale = 1000.0
    scaled_profile = profile_linear_amplitudes(
        q,
        scale * observed,
        components,
        resolution=resolution,
        sigma=scale * observed * sigma_log,
    )
    scaled = diagnose_component_observability(
        q, scale * observed, scaled_profile, sigma_log=sigma_log
    )

    assert scaled.full_scores.raw_log_rmse == pytest.approx(
        report.full_scores.raw_log_rmse, abs=2e-13
    )
    assert scaled.full_scores.standardized_log_rmse == pytest.approx(
        report.full_scores.standardized_log_rmse, abs=2e-12
    )
    np.testing.assert_allclose(
        scaled.full_exact_intensity / scale,
        report.full_exact_intensity,
        rtol=2e-13,
        atol=2e-13,
    )
    for original, rescaled in zip(report.components, scaled.components):
        assert rescaled.contribution.amplitude == pytest.approx(
            scale * original.contribution.amplitude, rel=2e-13
        )
        assert rescaled.contribution.relative_to_observed_rms == pytest.approx(
            original.contribution.relative_to_observed_rms, rel=2e-13
        )
        assert rescaled.contribution.relative_to_model_rms == pytest.approx(
            original.contribution.relative_to_model_rms, rel=2e-13
        )
        assert rescaled.contribution.measurement_standardized_rms == pytest.approx(
            original.contribution.measurement_standardized_rms, rel=2e-13
        )
        assert rescaled.raw_log_score_increment == pytest.approx(
            original.raw_log_score_increment, abs=2e-13
        )
        assert rescaled.standardized_log_score_increment == pytest.approx(
            original.standardized_log_score_increment, abs=2e-12
        )


def test_clean_single_particle_deletion_uses_k0_nested_null():
    q = np.geomspace(0.006, 1.8, 300)
    component = NonlinearComponent("sphere", R=13.0, sigma_R=1.4, D=41.0, sigma_D=4.5)
    intensity = _curve(q, (component,), [0.015, 1.7])
    profile = profile_linear_amplitudes(q, intensity, (component,), sigma=intensity)

    report = diagnose_component_observability(q, intensity, profile)

    assert report.resolution_contribution is None
    assert report.background_contribution.measurement_standardized_rms is None
    assert report.components[0].contribution.measurement_standardized_rms is None
    deletion = report.components[0]
    assert deletion.deletion_status == DELETION_K0_EVALUATED
    assert deletion.remaining_component_indices == ()
    assert deletion.reduced_profile is not None
    assert deletion.reduced_scores is not None
    assert deletion.raw_log_score_increment > 0.0
    assert deletion.standardized_log_score_increment is None
    assert report.exact_forward_calls == 2
    assert report.full_exact_intensity.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        report.full_exact_intensity[0] = 0.0


def test_diagnostic_rejects_inconsistent_profile_and_invalid_uncertainty():
    q = np.geomspace(0.01, 1.0, 80)
    component = NonlinearComponent("sphere", R=9.0, sigma_R=0.8)
    intensity = _curve(q, (component,), [0.01, 1.0])
    profile = profile_linear_amplitudes(q, intensity, (component,))

    with pytest.raises(ValueError, match="component weights are inconsistent"):
        diagnose_component_observability(
            q,
            intensity,
            replace(profile, component_weights=(0.6, 0.4)),
        )
    with pytest.raises(ValueError, match="sigma_log must contain"):
        diagnose_component_observability(
            q,
            intensity,
            profile,
            sigma_log=np.zeros_like(q),
        )


def _candidate_from_profile(candidate_id, q, intensity, profile):
    return CandidateInput(
        candidate_id=candidate_id,
        proposal_rank=1,
        topology_id=topology_id_for(tuple(item.shape for item in profile.components)),
        components=tuple(gui_component_to_latent(item) for item in profile.components),
        resolution=profile.resolution,
        linear_solution=LinearSolutionSnapshot.from_profiled_forward(profile),
        exact_intensity=evaluate_profiled_forward(q, profile),
        bounds_pass=True,
        physics_pass=True,
    )


def _assess(q, intensity, profile, *, threshold, limit=6):
    curve = ObservedCurve(
        curve_id="observability-test",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
    )
    return assess_candidate_observability(
        curve,
        _candidate_from_profile("candidate", q, intensity, profile),
        primary_metric_name=RAW_LOG_RMSE_METRIC,
        primary_compatibility_threshold=threshold,
        policy=ObservabilityPolicy(exact_forward_call_limit=limit),
    )


def test_k1_near_flat_background_null_confirms_particle_unneeded():
    q = np.geomspace(0.006, 1.8, 300)
    component = NonlinearComponent("sphere", R=13.0, sigma_R=1.4)
    intensity = _curve(q, (component,), [1.0, 1.0e-8])
    profile = profile_linear_amplitudes(q, intensity, (component,), sigma=intensity)

    result = _assess(q, intensity, profile, threshold=1.0e-4)

    assert result.particles[0].decision == "unneeded"
    assert result.status == "confirmed_redundant"
    assert result.exact_forward_calls == 1


def test_k1_real_feature_is_needed_against_exhaustive_k0_null():
    q = np.geomspace(0.006, 1.8, 300)
    component = NonlinearComponent("sphere", R=13.0, sigma_R=1.4)
    intensity = _curve(q, (component,), [0.01, 1.7])
    profile = profile_linear_amplitudes(q, intensity, (component,), sigma=intensity)

    result = _assess(q, intensity, profile, threshold=1.0e-3)

    assert result.particles[0].decision == "needed"
    assert result.particles[0].evidence_scope == "exhaustive_k0_nested_null"
    assert result.status == "confirmed_effective"


def test_large_coefficient_weak_k2_basis_is_deleted_by_curve_gate():
    q = np.geomspace(0.004, 2.0, 400)
    components = (
        NonlinearComponent("sphere", R=10.0, sigma_R=1.0),
        NonlinearComponent("vertical cylinder", R=10.0, sigma_R=0.1),
    )
    intensity = _curve(q, components, [0.01, 1.0, 100.0])
    profile = profile_linear_amplitudes(q, intensity, components, sigma=intensity)

    result = _assess(q, intensity, profile, threshold=0.05)

    assert profile.component_weights[1] > 0.9
    assert result.particles[1].decision == "unneeded"
    assert result.status == "confirmed_redundant"


def test_resolution_delete_and_budget_ledger_are_explicit():
    q = np.geomspace(0.004, 2.0, 400)
    component = NonlinearComponent("sphere", R=12.0, sigma_R=1.2)
    resolution = ResolutionShape(sigma_res=0.015, nu_res=6.0)
    intensity = _curve(q, (component,), [0.02, 1.0, 1.0e-8], resolution)
    profile = profile_linear_amplitudes(
        q, intensity, (component,), resolution=resolution, sigma=intensity
    )

    result = _assess(q, intensity, profile, threshold=1.0e-4, limit=1)

    assert result.exact_forward_calls == 1
    assert result.exact_forward_call_limit == 1
    assert result.resolution.decision == "unknown"
    assert result.resolution.deletion_status == DELETION_FORWARD_BUDGET_EXHAUSTED
    assert result.budget_exhausted

    complete = _assess(q, intensity, profile, threshold=1.0e-4, limit=2)
    assert complete.exact_forward_calls == 2
    assert complete.resolution.decision == "unneeded"
    assert complete.status == "confirmed_redundant"


def test_k1_with_resolution_fixed_shape_null_is_not_claimed_exhaustive():
    q = np.geomspace(0.004, 2.0, 400)
    component = NonlinearComponent("sphere", R=12.0, sigma_R=1.2)
    resolution = ResolutionShape(sigma_res=0.015, nu_res=6.0)
    intensity = _curve(q, (component,), [0.02, 1.0, 0.25], resolution)
    profile = profile_linear_amplitudes(
        q, intensity, (component,), resolution=resolution, sigma=intensity
    )

    result = _assess(q, intensity, profile, threshold=1.0e-8, limit=2)

    assert result.particles[0].decision == "provisional_needed"
    assert result.particles[0].evidence_scope == "fixed_geometry_reduced_model_only"
    assert result.status == "provisional_or_unknown"


def test_reduced_search_budget_is_reserved_fairly_across_unresolved_features():
    q = np.geomspace(0.004, 2.0, 300)
    components = (
        NonlinearComponent("sphere", R=11.0, sigma_R=1.1, D=35.0, sigma_D=3.5),
        NonlinearComponent(
            "cylinder",
            R=22.0,
            sigma_R=2.2,
            h=55.0,
            sigma_h=5.5,
            D=70.0,
            sigma_D=7.0,
        ),
    )
    resolution = ResolutionShape(sigma_res=0.014, nu_res=5.0)
    intensity = _curve(q, components, [0.01, 1.0, 1.3, 0.2], resolution)
    profile = profile_linear_amplitudes(
        q, intensity, components, resolution=resolution, sigma=intensity
    )
    curve = ObservedCurve(
        curve_id="fair-observability-budget",
        source_kind="synthetic",
        q=q,
        intensity=intensity,
    )
    candidate = _candidate_from_profile("fair", q, intensity, profile)

    class AuditSearcher:
        def __init__(self):
            self.allocations = {}

        def search(self, curve, candidate, **kwargs):
            label = kwargs["label"]
            allowance = kwargs["max_forward_evaluations"]
            self.allocations[label] = allowance
            return ReducedModelSearchEvidence(
                label=label,
                status="search_unavailable",
                best_primary_score=None,
                primary_compatibility_threshold=kwargs["primary_compatibility_threshold"],
                forward_calls=0,
                attempted_starts=0,
                required_starts=kwargs["required_starts"],
                search_version="test-audit-search/v1",
                attempts=(),
                detail="audit-only fixture",
            )

    searcher = AuditSearcher()
    result = assess_candidate_observability(
        curve,
        candidate,
        primary_metric_name=RAW_LOG_RMSE_METRIC,
        primary_compatibility_threshold=1.0e-8,
        policy=ObservabilityPolicy(exact_forward_call_limit=64),
        reduced_model_searcher=searcher,
    )

    assert set(searcher.allocations) == {
        "particle_0",
        "particle_1",
        "resolution",
        "d_0",
        "d_1",
    }
    assert max(searcher.allocations.values()) - min(searcher.allocations.values()) <= 1
    assert sum(searcher.allocations.values()) == 64 - result.exact_forward_calls
    assessed = (*result.particles, result.resolution, *result.d_terms)
    assert {
        item.label: item.reduced_search_forward_call_limit for item in assessed
    } == searcher.allocations


def test_bounded_reduced_search_finds_compatible_particle_deletion_with_counted_calls():
    q = np.geomspace(0.006, 1.5, 80)
    components = (
        NonlinearComponent("sphere", R=13.0, sigma_R=1.3),
        NonlinearComponent("cylinder", R=25.0, sigma_R=2.5, h=50.0, sigma_h=5.0),
    )
    intensity = _curve(q, components, [0.01, 1.2, 1.0e-8])
    profile = profile_linear_amplitudes(q, intensity, components, sigma=intensity)
    candidate = _candidate_from_profile("reduced-search", q, intensity, profile)
    curve = ObservedCurve(
        curve_id="reduced-search",
        source_kind="synthetic",
        q=q,
        intensity=candidate.exact_intensity,
    )
    searcher = BoundedReducedModelSearcher(
        component_bounds=tuple(
            full_component_bounds(item.shape, d_policy="optional") for item in components
        ),
        resolution_bounds=None,
    )

    evidence = searcher.search(
        curve,
        candidate,
        label="particle_1",
        primary_metric_name=RAW_LOG_RMSE_METRIC,
        primary_compatibility_threshold=1.0e-5,
        max_forward_evaluations=8,
        required_starts=1,
        per_start_forward_evaluation_limit=8,
    )

    assert evidence.status == "compatible_reduced_model_found"
    assert 0 < evidence.forward_calls <= 8
    assert evidence.forward_calls == sum(item.forward_calls for item in evidence.attempts)
    assert evidence.best_primary_score <= evidence.primary_compatibility_threshold
