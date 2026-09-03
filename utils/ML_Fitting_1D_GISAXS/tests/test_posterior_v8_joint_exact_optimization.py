from __future__ import annotations

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import (
    joint_exact_optimization as joint_module,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_polish import (
    AmplitudeBounds,
    polish_profiled_amplitudes,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import ResolutionBounds
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    gui_bounds_to_latent,
    gui_component_to_latent,
    latent_component_to_gui,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.evaluation import (
    STANDARDIZED_LOG_RMSE_METRIC,
    natural_log_rmse,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.gui_amplitude_constraints import (
    GUI_AMPLITUDE_CONSTRAINT_VERSION,
    GuiAmplitudeConstraint,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.joint_exact_optimization import (
    JOINT_EXACT_OPTIMIZATION_VERSION,
    optimize_joint_exact_branch,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import (
    ResolutionShape,
    build_design_matrix,
    evaluate_profiled_forward,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_refinement import (
    refine_profiled_branch,
)


def _curve(q, components, coefficients, resolution=None):
    return build_design_matrix(q, components, resolution) @ np.asarray(
        coefficients, dtype=np.float64
    )


def _single_sphere_problem():
    q = np.geomspace(0.006, 1.5, 200)
    truth = GuiComponentParameters("sphere", R=12.0, sigma_R=1.2, D=35.0, sigma_D=3.5)
    truth_resolution = ResolutionShape(sigma_res=0.018, nu_res=6.0)
    intensity = _curve(q, (truth,), (0.003, 2.4, 0.15), truth_resolution)
    bounds = GuiComponentBounds(
        "sphere",
        R=ClosedInterval(9.0, 16.0),
        sigma_R=ClosedInterval(0.6, 2.0),
        D=ClosedInterval(28.0, 42.0),
        sigma_D=ClosedInterval(2.0, 5.0),
    )
    seed = gui_component_to_latent(
        GuiComponentParameters("sphere", R=13.5, sigma_R=1.55, D=38.0, sigma_D=4.2)
    )
    resolution_bounds = ResolutionBounds(
        sigma_res=ClosedInterval(0.012, 0.025),
        nu_res=ClosedInterval(4.5, 8.0),
    )
    resolution_seed = ResolutionShape(0.021, 5.2)
    return q, truth, intensity, bounds, seed, resolution_bounds, resolution_seed


def test_k1_clean_joint_oracle_matches_same_seed_sequential_baseline():
    q, truth, intensity, bounds, seed, resolution_bounds, resolution_seed = _single_sphere_problem()

    joint = optimize_joint_exact_branch(
        q,
        intensity,
        (bounds,),
        (seed,),
        resolution_bounds=resolution_bounds,
        resolution_seed=resolution_seed,
        max_exact_evaluations=200,
    )
    profiled = refine_profiled_branch(
        q,
        intensity,
        (bounds,),
        (seed,),
        resolution_bounds=resolution_bounds,
        resolution_seed=resolution_seed,
        max_nfev=180,
    )
    sequential = polish_profiled_amplitudes(q, intensity, profiled.final_profile, max_nfev=150)

    assert joint.bounds_satisfied
    assert joint.success
    assert joint.exact_forward_calls <= joint.max_exact_evaluations
    assert joint.initial_metric == pytest.approx(profiled.initial_log_rmse, rel=1e-13)
    assert joint.final_metric <= sequential.final_metric + 1e-12
    assert joint.final_metric < 1e-10
    assert joint.final_coefficients == pytest.approx((0.003, 2.4, 0.15), rel=1e-8)
    final = latent_component_to_gui(joint.final_latent_components[0])
    assert final.R == pytest.approx(truth.R, rel=1e-7)
    assert final.sigma_R == pytest.approx(truth.sigma_R, rel=1e-7)
    assert final.D == pytest.approx(truth.D, rel=1e-7)
    assert final.sigma_D == pytest.approx(truth.sigma_D, rel=1e-7)
    assert resolution_bounds.contains(joint.final_resolution)
    np.testing.assert_allclose(
        joint.exact_intensity,
        evaluate_profiled_forward(q, joint.final_profile),
        rtol=2e-12,
        atol=2e-13,
    )
    assert joint.exact_intensity.flags.writeable is False


def test_k2_noisy_standardized_oracle_is_fairly_comparable_to_sequential_solver():
    q = np.geomspace(0.006, 1.8, 160)
    truth = (
        GuiComponentParameters("sphere", R=11.0, sigma_R=1.1),
        GuiComponentParameters("vertical cylinder", R=25.0, sigma_R=0.17),
    )
    clean = _curve(q, truth, (0.015, 0.8, 500.0))
    sigma_log = 0.02 + 0.01 * np.square(np.sin(np.linspace(0.0, 3.0 * np.pi, q.size)))
    observed = clean * np.exp(0.4 * sigma_log * np.sin(np.linspace(0.0, 8.0 * np.pi, q.size)))
    bounds = (
        GuiComponentBounds("sphere", ClosedInterval(8.0, 15.0), ClosedInterval(0.5, 1.8)),
        GuiComponentBounds(
            "vertical cylinder",
            ClosedInterval(19.0, 32.0),
            ClosedInterval(0.10, 0.30),
        ),
    )
    seeds = tuple(
        gui_component_to_latent(item)
        for item in (
            GuiComponentParameters("sphere", R=13.0, sigma_R=1.5),
            GuiComponentParameters("vertical cylinder", R=29.0, sigma_R=0.24),
        )
    )

    joint = optimize_joint_exact_branch(
        q,
        observed,
        bounds,
        seeds,
        sigma_log=sigma_log,
        metric_name=STANDARDIZED_LOG_RMSE_METRIC,
        max_exact_evaluations=160,
    )
    profiled = refine_profiled_branch(q, observed, bounds, seeds, sigma_log=sigma_log, max_nfev=120)
    sequential = polish_profiled_amplitudes(
        q,
        observed,
        profiled.final_profile,
        sigma_log=sigma_log,
        metric_name=STANDARDIZED_LOG_RMSE_METRIC,
        max_nfev=100,
    )

    assert joint.initial_metric == pytest.approx(profiled.initial_weighted_log_rmse, rel=1e-13)
    assert joint.final_metric <= sequential.final_metric + 1e-8
    assert joint.final_metric < joint.initial_metric
    assert joint.final_standardized_log_rmse == pytest.approx(joint.final_metric)
    assert joint.final_raw_log_rmse == pytest.approx(
        natural_log_rmse(joint.exact_intensity, observed)
    )
    assert joint.final_resolution is None
    assert joint.final_profile.resolution_present is False
    assert joint.amplitude_scales[2] > 5.0 * joint.amplitude_scales[1]
    assert all(
        gui_bounds_to_latent(item_bounds).contains(item)
        for item_bounds, item in zip(bounds, joint.final_latent_components)
    )
    assert joint.exact_forward_calls < joint.max_exact_evaluations


def test_fixed_zero_particle_amplitude_is_an_exact_supported_boundary():
    q = np.geomspace(0.008, 1.5, 160)
    components = (
        GuiComponentParameters("sphere", R=10.0, sigma_R=1.0),
        GuiComponentParameters("vertical cylinder", R=24.0, sigma_R=0.18),
    )
    intensity = _curve(q, components, (0.01, 1.2, 0.0))
    bounds = (
        GuiComponentBounds("sphere", ClosedInterval(8.0, 13.0), ClosedInterval(0.6, 1.5)),
        GuiComponentBounds(
            "vertical cylinder",
            ClosedInterval(20.0, 30.0),
            ClosedInterval(0.10, 0.25),
        ),
    )
    seeds = tuple(
        gui_component_to_latent(item)
        for item in (
            GuiComponentParameters("sphere", R=11.5, sigma_R=1.3),
            GuiComponentParameters("vertical cylinder", R=27.0, sigma_R=0.22),
        )
    )
    amplitude_bounds = AmplitudeBounds(
        lower=(0.0, 0.0, 0.0),
        upper=(0.1, 3.0, 0.0),
    )

    result = optimize_joint_exact_branch(
        q,
        intensity,
        bounds,
        seeds,
        amplitude_bounds=amplitude_bounds,
        max_exact_evaluations=220,
    )

    assert result.requested_amplitude_bounds == amplitude_bounds
    assert result.final_coefficients[2] == 0.0
    assert result.final_profile.particle_amplitudes[1] == 0.0
    assert result.effective_amplitude_bounds.contains(result.final_coefficients)
    assert result.final_metric < 1e-9


def test_k4_branch_order_and_fixed_user_ranges_work_under_a_smoke_budget():
    q = np.geomspace(0.01, 1.2, 64)
    components = (
        GuiComponentParameters("sphere", R=5.0, sigma_R=0.5),
        GuiComponentParameters("sphere", R=9.0, sigma_R=0.9),
        GuiComponentParameters("vertical cylinder", R=16.0, sigma_R=0.14),
        GuiComponentParameters("vertical cylinder", R=27.0, sigma_R=0.20),
    )
    coefficients = (0.02, 0.3, 0.5, 50.0, 100.0)
    intensity = _curve(q, components, coefficients)
    bounds = tuple(
        GuiComponentBounds(
            component.shape,
            ClosedInterval(component.R - 0.01, component.R + 0.01),
            ClosedInterval(
                component.sigma_R - 0.001,
                component.sigma_R + 0.001,
            ),
        )
        for component in components
    )
    fixed_amplitudes = AmplitudeBounds(coefficients, coefficients)

    result = optimize_joint_exact_branch(
        q,
        intensity,
        bounds,
        tuple(gui_component_to_latent(item) for item in components),
        amplitude_bounds=fixed_amplitudes,
        max_exact_evaluations=16,
    )

    assert tuple(item.shape for item in result.final_profile.components) == (
        "sphere",
        "sphere",
        "vertical_cylinder",
        "vertical_cylinder",
    )
    assert result.final_coefficients == pytest.approx(coefficients)
    assert result.final_metric < 1e-12
    assert result.exact_forward_calls <= 16
    assert result.bounds_satisfied


def test_exact_forward_budget_is_hard_and_best_seen_survives_exhaustion(monkeypatch):
    q = np.geomspace(0.01, 1.0, 80)
    truth = GuiComponentParameters("sphere", R=10.0, sigma_R=1.0)
    intensity = _curve(q, (truth,), (0.01, 1.0))
    bounds = GuiComponentBounds("sphere", ClosedInterval(8.0, 14.0), ClosedInterval(0.5, 2.0))
    seed = gui_component_to_latent(GuiComponentParameters("sphere", R=12.0, sigma_R=1.5))
    real_exact = joint_module.evaluate_profiled_forward
    calls = []

    def counted_exact(q_values, profile):
        calls.append(profile)
        return real_exact(q_values, profile)

    monkeypatch.setattr(joint_module, "evaluate_profiled_forward", counted_exact)
    result = optimize_joint_exact_branch(q, intensity, (bounds,), (seed,), max_exact_evaluations=7)

    assert result.budget_exhausted
    assert result.success is False
    assert result.exact_forward_calls == len(calls) == 7
    assert result.objective_requests > result.exact_forward_calls
    assert result.cache_hits >= 1
    assert 1 <= result.best_exact_call <= result.exact_forward_calls
    assert result.final_metric <= result.initial_metric
    assert result.returned_source in {"initial", "optimizer_intermediate"}
    assert all(
        bounds.R.contains(item.components[0].R)
        and bounds.sigma_R.contains(item.components[0].sigma_R)
        for item in calls
    )


def test_joint_oracle_rejects_invalid_metric_budget_and_particle_bounds():
    q = np.geomspace(0.01, 1.0, 80)
    component = GuiComponentParameters("sphere", R=10.0, sigma_R=1.0)
    intensity = _curve(q, (component,), (0.01, 1.0))
    bounds = GuiComponentBounds("sphere", ClosedInterval(8.0, 14.0), ClosedInterval(0.5, 2.0))
    seed = gui_component_to_latent(component)

    with pytest.raises(ValueError, match="requires sigma_log"):
        optimize_joint_exact_branch(
            q,
            intensity,
            (bounds,),
            (seed,),
            metric_name=STANDARDIZED_LOG_RMSE_METRIC,
        )
    with pytest.raises(ValueError, match="positive integer"):
        optimize_joint_exact_branch(q, intensity, (bounds,), (seed,), max_exact_evaluations=0)
    with pytest.raises(ValueError, match="particle amplitude upper bound"):
        optimize_joint_exact_branch(
            q,
            intensity,
            (bounds,),
            (seed,),
            amplitude_bounds=AmplitudeBounds((0.0, 0.0), (1.0, 0.0)),
        )


def _coupled_k2_problem():
    q = np.geomspace(0.006, 1.8, 120)
    truth = (
        GuiComponentParameters("sphere", R=11.0, sigma_R=1.1),
        GuiComponentParameters("vertical cylinder", R=25.0, sigma_R=0.17),
    )
    intensity = _curve(q, truth, (0.04, 0.9, 2.1))
    bounds = (
        GuiComponentBounds("sphere", ClosedInterval(10.8, 11.2), ClosedInterval(1.08, 1.12)),
        GuiComponentBounds(
            "vertical cylinder",
            ClosedInterval(24.8, 25.2),
            ClosedInterval(0.165, 0.175),
        ),
    )
    constraint = GuiAmplitudeConstraint(
        background=ClosedInterval(0.01, 0.10),
        component_intensities=(
            ClosedInterval(0.20, 0.40),
            ClosedInterval(0.60, 0.80),
        ),
        k=ClosedInterval(2.0, 4.0),
        resolution_present=False,
    )
    return (
        q,
        intensity,
        bounds,
        tuple(gui_component_to_latent(item) for item in truth),
        constraint,
    )


def test_coupled_gui_ranges_hold_for_every_exact_objective_state(monkeypatch):
    q, intensity, bounds, seeds, constraint = _coupled_k2_problem()
    real_exact = joint_module.evaluate_profiled_forward
    calls = []

    def audited_exact(q_values, profile):
        calls.append(profile)
        return real_exact(q_values, profile)

    monkeypatch.setattr(joint_module, "evaluate_profiled_forward", audited_exact)
    result = optimize_joint_exact_branch(
        q,
        intensity,
        bounds,
        seeds,
        gui_amplitude_constraint=constraint,
        max_exact_evaluations=120,
    )

    assert result.version == JOINT_EXACT_OPTIMIZATION_VERSION
    assert result.version.endswith("_v3")
    assert result.success
    assert result.final_metric < 1e-10
    assert result.gui_amplitude_constraint is constraint
    assert result.coefficient_polytope is not None
    assert result.coefficient_polytope.version == GUI_AMPLITUDE_CONSTRAINT_VERSION
    assert result.requested_amplitude_bounds is None
    assert result.initial_constraint_audit is not None
    assert result.initial_constraint_audit.all_constraints_satisfied
    assert result.final_constraint_audit is not None
    assert result.final_constraint_audit.all_constraints_satisfied
    assert result.initial_profile.amplitude_constraint_audit is not None
    assert result.final_profile.amplitude_constraint_audit is not None
    assert constraint.contains(result.final_coefficients, k=result.final_profile.k)
    assert result.final_coefficients == pytest.approx((0.04, 0.9, 2.1), rel=1e-9)
    assert len(calls) == result.exact_forward_calls
    assert calls
    assert all(
        profile.amplitude_constraint_audit is not None
        and profile.amplitude_constraint_audit.all_constraints_satisfied
        and constraint.contains(
            (
                profile.background,
                *profile.particle_amplitudes,
                *((profile.resolution_amplitude,) if profile.resolution is not None else ()),
            ),
            k=profile.k,
        )
        for profile in calls
    )


def test_coupled_resolution_ranges_are_jointly_optimized_with_geometry():
    q, truth, intensity, bounds, _, resolution_bounds, _ = _single_sphere_problem()
    seed = gui_component_to_latent(
        GuiComponentParameters("sphere", R=12.02, sigma_R=1.198, D=35.02, sigma_D=3.498)
    )
    tight_bounds = GuiComponentBounds(
        "sphere",
        R=ClosedInterval(11.9, 12.1),
        sigma_R=ClosedInterval(1.19, 1.21),
        D=ClosedInterval(34.9, 35.1),
        sigma_D=ClosedInterval(3.49, 3.51),
    )
    tight_resolution_bounds = ResolutionBounds(
        sigma_res=ClosedInterval(0.0179, 0.0181),
        nu_res=ClosedInterval(5.99, 6.01),
    )
    constraint = GuiAmplitudeConstraint(
        background=ClosedInterval(0.001, 0.010),
        component_intensities=(ClosedInterval(1.0, 1.0),),
        k=ClosedInterval(1.0, 4.0),
        int_res=ClosedInterval(0.02, 0.10),
        resolution_present=True,
    )

    result = optimize_joint_exact_branch(
        q,
        intensity,
        (tight_bounds,),
        (seed,),
        resolution_bounds=tight_resolution_bounds,
        resolution_seed=ResolutionShape(0.01802, 5.998),
        gui_amplitude_constraint=constraint,
        max_exact_evaluations=160,
    )

    assert result.success
    assert result.final_metric < 1e-10
    assert result.final_metric < result.initial_metric
    assert result.final_coefficients == pytest.approx((0.003, 2.4, 0.15), rel=1e-8)
    assert result.final_profile.int_res == pytest.approx(0.0625, rel=1e-8)
    assert result.final_constraint_audit is not None
    assert result.final_constraint_audit.all_constraints_satisfied
    final = latent_component_to_gui(result.final_latent_components[0])
    assert final.R == pytest.approx(truth.R, rel=1e-7)
    assert tight_resolution_bounds.contains(result.final_resolution)
    # The looser bounds returned by the helper are intentionally unused; this
    # assertion guards the exact branch contract used by this test.
    assert resolution_bounds.contains(result.final_resolution)


def test_coupled_constraint_budget_is_hard_and_resolution_absence_is_audited(
    monkeypatch,
):
    q, intensity, bounds, seeds, constraint = _coupled_k2_problem()
    real_exact = joint_module.evaluate_profiled_forward
    calls = []

    def counted_exact(q_values, profile):
        calls.append(profile)
        return real_exact(q_values, profile)

    monkeypatch.setattr(joint_module, "evaluate_profiled_forward", counted_exact)
    result = optimize_joint_exact_branch(
        q,
        intensity,
        bounds,
        seeds,
        gui_amplitude_constraint=constraint,
        max_exact_evaluations=7,
    )

    assert result.budget_exhausted
    assert result.success is False
    assert result.exact_forward_calls == len(calls) == 7
    assert result.objective_requests > result.exact_forward_calls
    assert result.cache_hits >= 1
    assert result.final_resolution is None
    assert result.final_profile.resolution_present is False
    assert result.coefficient_polytope is not None
    assert result.coefficient_polytope.resolution_present is False
    assert result.final_constraint_audit is not None
    assert result.final_constraint_audit.all_constraints_satisfied
    assert all(profile.resolution is None for profile in calls)


def test_coupled_constraint_rejects_infeasible_seed_branch_and_axis_box_mix(
    monkeypatch,
):
    q, intensity, bounds, seeds, constraint = _coupled_k2_problem()
    calls = []

    def forbidden_exact(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("invalid constraint must fail before exact forward")

    monkeypatch.setattr(joint_module, "evaluate_profiled_forward", forbidden_exact)
    # This is inside the independent axis projection but has forbidden
    # canonical weights (0.8, 0.2).
    with pytest.raises(ValueError, match="outside the requested GUI amplitude polytope"):
        optimize_joint_exact_branch(
            q,
            intensity,
            bounds,
            seeds,
            gui_amplitude_constraint=constraint,
            amplitude_seed=(0.04, 1.6, 0.4),
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        optimize_joint_exact_branch(
            q,
            intensity,
            bounds,
            seeds,
            amplitude_bounds=AmplitudeBounds((0.0,) * 3, (10.0,) * 3),
            gui_amplitude_constraint=constraint,
        )
    resolution_constraint = GuiAmplitudeConstraint(
        background=ClosedInterval(0.0, 1.0),
        component_intensities=(ClosedInterval(0.0, 1.0),) * 2,
        k=ClosedInterval(1.0, 2.0),
        int_res=ClosedInterval(0.0, 1.0),
        resolution_present=True,
    )
    with pytest.raises(ValueError, match="Resolution presence"):
        optimize_joint_exact_branch(
            q,
            intensity,
            bounds,
            seeds,
            gui_amplitude_constraint=resolution_constraint,
        )
    assert calls == []


def test_explicit_none_gui_constraint_preserves_default_numerics():
    q, _, intensity, bounds, seed, resolution_bounds, resolution_seed = _single_sphere_problem()
    kwargs = {
        "resolution_bounds": resolution_bounds,
        "resolution_seed": resolution_seed,
        "max_exact_evaluations": 24,
    }

    legacy = optimize_joint_exact_branch(q, intensity, (bounds,), (seed,), **kwargs)
    explicit = optimize_joint_exact_branch(
        q,
        intensity,
        (bounds,),
        (seed,),
        gui_amplitude_constraint=None,
        **kwargs,
    )

    assert explicit.initial_coefficients == legacy.initial_coefficients
    assert explicit.final_coefficients == legacy.final_coefficients
    assert explicit.initial_active_coordinates == legacy.initial_active_coordinates
    assert explicit.final_active_coordinates == legacy.final_active_coordinates
    assert explicit.initial_metric == legacy.initial_metric
    assert explicit.final_metric == legacy.final_metric
    np.testing.assert_array_equal(explicit.exact_intensity, legacy.exact_intensity)
    assert explicit.gui_amplitude_constraint is None
    assert explicit.coefficient_polytope is None
    assert explicit.initial_constraint_audit is None
    assert explicit.final_constraint_audit is None
