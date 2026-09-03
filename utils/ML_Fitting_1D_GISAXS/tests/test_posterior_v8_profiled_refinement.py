from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.gimap.features.fitting.domain.physical_constraints import exclusion_size
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.branch_codec import ProfiledBranchCodec
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import (
    ClosedInterval,
    GuiComponentBounds,
    GuiComponentParameters,
    gui_bounds_to_latent,
    gui_component_to_latent,
    latent_component_to_gui,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import (
    ResolutionShape,
    build_design_matrix,
    evaluate_profiled_forward,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import profiled_refinement as refinement_module
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_refinement import (
    HARD_CORE_SPACING_MARGIN,
    ResolutionBounds,
    refine_profiled_branch,
)


def _exact_curve(q, components, coefficients, resolution=None):
    return build_design_matrix(q, components, resolution) @ np.asarray(
        coefficients, dtype=np.float64
    )


def test_refines_single_sphere_with_d_and_resolution_from_an_imperfect_seed():
    q = np.geomspace(0.006, 1.5, 260)
    truth = GuiComponentParameters("sphere", R=12.0, sigma_R=1.2, D=35.0, sigma_D=3.5)
    truth_resolution = ResolutionShape(sigma_res=0.018, nu_res=6.0)
    intensity = _exact_curve(q, (truth,), (0.003, 2.4, 0.15), truth_resolution)
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

    result = refine_profiled_branch(
        q,
        intensity,
        (bounds,),
        (seed,),
        resolution_bounds=resolution_bounds,
        resolution_seed=ResolutionShape(0.021, 5.2),
        sigma_log=np.full(q.shape, 0.025),
        max_nfev=180,
    )

    assert result.success
    assert result.bounds_satisfied
    assert result.nfev <= 180
    assert result.residual_calls >= result.nfev
    assert result.final_log_rmse < result.initial_log_rmse * 1e-2
    assert result.final_log_rmse < 2e-5
    assert result.final_resolution is not None
    assert resolution_bounds.contains(result.final_resolution)
    fitted = latent_component_to_gui(result.final_latent_components[0])
    assert fitted.R == pytest.approx(truth.R, rel=2e-5)
    assert fitted.sigma_R == pytest.approx(truth.sigma_R, rel=2e-5)
    assert fitted.D == pytest.approx(truth.D, rel=2e-5)
    assert fitted.sigma_D == pytest.approx(truth.sigma_D, rel=2e-5)
    assert result.final_resolution.sigma_res == pytest.approx(truth_resolution.sigma_res, rel=2e-5)
    assert result.final_resolution.nu_res == pytest.approx(truth_resolution.nu_res, rel=2e-5)
    np.testing.assert_allclose(
        result.exact_forward_intensity,
        evaluate_profiled_forward(q, result.final_profile),
        rtol=1e-12,
        atol=1e-12,
    )


def test_refines_vertical_cylinder_with_absent_d_and_resolution():
    q = np.geomspace(0.008, 1.8, 240)
    truth = GuiComponentParameters("vertical cylinder", R=24.0, sigma_R=0.18)
    intensity = _exact_curve(q, (truth,), (0.012, 3.0))
    bounds = GuiComponentBounds(
        "vertical cylinder",
        R=ClosedInterval(18.0, 32.0),
        sigma_R=ClosedInterval(0.10, 0.30),
    )
    seed = gui_component_to_latent(
        GuiComponentParameters("vertical cylinder", R=28.0, sigma_R=0.24)
    )

    result = refine_profiled_branch(q, intensity, (bounds,), (seed,), max_nfev=120)

    final_gui = latent_component_to_gui(result.final_latent_components[0])
    assert result.success
    assert result.final_resolution is None
    assert result.final_profile.resolution_present is False
    assert final_gui.D is None and final_gui.sigma_D is None
    assert final_gui.R == pytest.approx(truth.R, rel=2e-5)
    assert final_gui.sigma_R == pytest.approx(truth.sigma_R, rel=2e-5)
    assert result.final_log_rmse < result.initial_log_rmse * 1e-3
    assert result.final_log_rmse < 1e-6
    np.testing.assert_allclose(
        result.exact_forward_intensity,
        intensity,
        rtol=2e-6,
        atol=1e-10,
    )


def test_fixed_active_axis_is_derived_and_never_enters_scipy_vector(monkeypatch):
    q = np.geomspace(0.008, 1.4, 100)
    truth = GuiComponentParameters("sphere", R=12.0, sigma_R=1.2, D=32.0, sigma_D=4.0)
    resolution = ResolutionShape(0.02, 5.0)
    intensity = _exact_curve(q, (truth,), (0.01, 2.0, 0.1), resolution)
    bounds = GuiComponentBounds(
        "sphere",
        R=ClosedInterval(12.0, 12.0),
        sigma_R=ClosedInterval(0.8, 1.6),
        D=ClosedInterval(32.0, 32.0),
        sigma_D=ClosedInterval(4.0, 4.0),
    )
    seed = gui_component_to_latent(
        GuiComponentParameters("sphere", R=12.0, sigma_R=1.0, D=32.0, sigma_D=4.0)
    )
    resolution_bounds = ResolutionBounds(ClosedInterval(0.02, 0.02), ClosedInterval(5.0, 5.0))
    codec = ProfiledBranchCodec.build(
        ("sphere",),
        (bounds,),
        (True,),
        resolution_bounds=resolution_bounds,
    )
    captured: dict[str, object] = {}

    def inspect_least_squares(fun, x0, *, bounds, max_nfev, **kwargs):
        captured["x0"] = np.asarray(x0).copy()
        captured["bounds"] = tuple(np.asarray(value).copy() for value in bounds)
        captured["max_nfev"] = max_nfev
        fun(x0)
        return SimpleNamespace(
            x=np.asarray(x0).copy(),
            success=True,
            status=1,
            message="test optimizer",
            nfev=1,
            njev=1,
        )

    monkeypatch.setattr(refinement_module, "least_squares", inspect_least_squares)
    result = refine_profiled_branch(
        q,
        intensity,
        (bounds,),
        (seed,),
        resolution_bounds=resolution_bounds,
        resolution_seed=resolution,
        max_nfev=17,
    )

    assert codec.active_indices == (0, 1, 4, 5, 24, 25)
    assert codec.varying_indices == (1,)
    assert tuple(codec.varying_mask[index] for index in codec.active_indices) == (
        False,
        True,
        False,
        False,
        False,
        False,
    )
    assert np.asarray(captured["x0"]).shape == (1,)
    assert all(value.shape == (1,) for value in captured["bounds"])
    assert captured["max_nfev"] == 17
    final = latent_component_to_gui(result.final_latent_components[0])
    assert final.R == 12.0
    assert final.D == 32.0
    assert final.sigma_D == 4.0
    assert result.final_resolution == resolution
    assert result.nfev == result.residual_calls == 1


def test_all_fixed_active_axes_skip_scipy_and_use_one_exact_profile(monkeypatch):
    q = np.geomspace(0.008, 1.4, 80)
    truth = GuiComponentParameters("sphere", R=12.0, sigma_R=1.2)
    intensity = _exact_curve(q, (truth,), (0.01, 2.0))
    bounds = GuiComponentBounds(
        "sphere",
        R=ClosedInterval(12.0, 12.0),
        sigma_R=ClosedInterval(1.2, 1.2),
    )
    seed = gui_component_to_latent(truth)

    def scipy_must_not_run(*args, **kwargs):
        raise AssertionError("all-fixed geometry must not enter SciPy")

    monkeypatch.setattr(refinement_module, "least_squares", scipy_must_not_run)
    phases: list[str] = []
    result = refine_profiled_branch(
        q,
        intensity,
        (bounds,),
        (seed,),
        exact_forward_call_hook=phases.append,
    )

    assert result.success
    assert result.nfev == result.residual_calls == 0
    assert result.exact_forward_calls == 1
    assert phases == ["initial_profile_verification"]
    assert result.initial_profile is result.final_profile


def test_coupled_absolute_width_bounds_are_obeyed_at_every_evaluation(monkeypatch):
    q = np.geomspace(0.01, 1.4, 220)
    truth = GuiComponentParameters("sphere", R=20.0, sigma_R=3.0, D=44.0, sigma_D=5.0)
    intensity = _exact_curve(q, (truth,), (0.01, 1.8))
    bounds = GuiComponentBounds(
        "sphere",
        R=ClosedInterval(14.0, 27.0),
        sigma_R=ClosedInterval(2.6, 3.3),
        D=ClosedInterval(36.0, 55.0),
        sigma_D=ClosedInterval(4.6, 5.5),
    )
    seed = gui_component_to_latent(
        GuiComponentParameters("sphere", R=24.0, sigma_R=3.2, D=50.0, sigma_D=5.4)
    )
    real_profile = refinement_module.profile_linear_amplitudes
    evaluated = []

    def checked_profile(q_values, observed, components, **kwargs):
        component = components[0]
        assert bounds.R.contains(component.R)
        assert bounds.sigma_R.contains(component.sigma_R)
        assert bounds.D is not None and bounds.D.contains(component.D)
        assert bounds.sigma_D is not None and bounds.sigma_D.contains(component.sigma_D)
        evaluated.append(component)
        return real_profile(q_values, observed, components, **kwargs)

    monkeypatch.setattr(refinement_module, "profile_linear_amplitudes", checked_profile)

    result = refine_profiled_branch(q, intensity, (bounds,), (seed,), max_nfev=150)

    latent = result.final_latent_components[0]
    gui = latent_component_to_gui(latent)
    assert gui_bounds_to_latent(bounds).contains(latent)
    assert bounds.R.contains(gui.R)
    assert bounds.sigma_R.contains(gui.sigma_R)
    assert bounds.D is not None and bounds.D.contains(gui.D)
    assert bounds.sigma_D is not None and bounds.sigma_D.contains(gui.sigma_D)
    assert len(evaluated) > result.nfev
    assert result.bounds_satisfied


@pytest.mark.parametrize(
    ("truth", "bounds", "seed"),
    [
        (
            GuiComponentParameters("sphere", 10.0, 1.0, D=24.0, sigma_D=2.4),
            GuiComponentBounds(
                "sphere",
                ClosedInterval(8.0, 12.0),
                ClosedInterval(0.5, 1.5),
                D=ClosedInterval(18.0, 27.0),
                sigma_D=ClosedInterval(1.0, 3.0),
            ),
            GuiComponentParameters("sphere", 11.0, 1.2, D=24.0, sigma_D=2.2),
        ),
        (
            GuiComponentParameters(
                "cylinder",
                5.0,
                0.5,
                h=12.0,
                sigma_h=1.2,
                D=20.0,
                sigma_D=2.0,
            ),
            GuiComponentBounds(
                "cylinder",
                ClosedInterval(4.0, 7.0),
                ClosedInterval(0.3, 1.0),
                h=ClosedInterval(8.0, 16.0),
                sigma_h=ClosedInterval(0.6, 2.0),
                D=ClosedInterval(16.0, 25.0),
                sigma_D=ClosedInterval(1.0, 3.0),
            ),
            GuiComponentParameters(
                "cylinder",
                6.0,
                0.7,
                h=14.0,
                sigma_h=1.5,
                D=22.0,
                sigma_D=2.2,
            ),
        ),
        (
            GuiComponentParameters("vertical cylinder", 10.0, 0.16, D=24.0, sigma_D=2.4),
            GuiComponentBounds(
                "vertical cylinder",
                ClosedInterval(8.0, 12.0),
                ClosedInterval(0.10, 0.25),
                D=ClosedInterval(18.0, 27.0),
                sigma_D=ClosedInterval(1.0, 3.0),
            ),
            GuiComponentParameters("vertical cylinder", 11.0, 0.20, D=24.0, sigma_D=2.2),
        ),
    ],
)
def test_every_profile_evaluation_obeys_authoritative_hard_core_rule(
    monkeypatch, truth, bounds, seed
):
    q = np.geomspace(0.01, 1.5, 140)
    intensity = _exact_curve(q, (truth,), (0.01, 2.0))
    real_profile = refinement_module.profile_linear_amplitudes
    evaluated = []

    def checked_profile(q_values, observed, components, **kwargs):
        component = components[0]
        params = {"R": component.R}
        if component.h is not None:
            params["h"] = component.h
        required = HARD_CORE_SPACING_MARGIN * exclusion_size(component.shape, params)
        assert component.D is not None and component.D > required
        evaluated.append(component)
        return real_profile(q_values, observed, components, **kwargs)

    monkeypatch.setattr(refinement_module, "profile_linear_amplitudes", checked_profile)
    result = refine_profiled_branch(
        q,
        intensity,
        (bounds,),
        (gui_component_to_latent(seed),),
        max_nfev=100,
    )

    assert result.bounds_satisfied
    assert evaluated
    assert result.final_log_rmse < result.initial_log_rmse


def test_narrow_feasible_d_range_remains_hard_core_safe(monkeypatch):
    q = np.geomspace(0.01, 1.2, 160)
    truth = GuiComponentParameters("sphere", 10.08, 0.70, D=20.19, sigma_D=1.05)
    intensity = _exact_curve(q, (truth,), (0.01, 1.7))
    bounds = GuiComponentBounds(
        "sphere",
        ClosedInterval(9.9, 10.1),
        ClosedInterval(0.5, 1.0),
        D=ClosedInterval(20.02, 20.21),
        sigma_D=ClosedInterval(1.01, 1.10),
    )
    seed = gui_component_to_latent(
        GuiComponentParameters("sphere", 10.0, 0.7, D=20.10, sigma_D=1.04)
    )
    real_profile = refinement_module.profile_linear_amplitudes
    checked_count = 0

    def checked_profile(q_values, observed, components, **kwargs):
        nonlocal checked_count
        component = components[0]
        required = HARD_CORE_SPACING_MARGIN * exclusion_size(component.shape, {"R": component.R})
        assert component.D > required
        assert bounds.D.contains(component.D)
        assert bounds.sigma_D.contains(component.sigma_D)
        checked_count += 1
        return real_profile(q_values, observed, components, **kwargs)

    monkeypatch.setattr(refinement_module, "profile_linear_amplitudes", checked_profile)
    result = refine_profiled_branch(q, intensity, (bounds,), (seed,), max_nfev=100)

    assert checked_count > result.nfev
    assert result.bounds_satisfied
    assert result.final_log_rmse < result.initial_log_rmse


def test_infeasible_hard_core_branch_and_violating_seed_fail_closed():
    q = np.geomspace(0.01, 1.0, 80)
    component = GuiComponentParameters("sphere", 10.0, 0.7, D=20.01, sigma_D=1.02)
    intensity = _exact_curve(q, (component,), (0.01, 1.0))
    impossible = GuiComponentBounds(
        "sphere",
        ClosedInterval(10.0, 10.1),
        ClosedInterval(0.5, 1.0),
        D=ClosedInterval(20.0, 20.019),
        sigma_D=ClosedInterval(1.0, 1.1),
    )
    with pytest.raises(ValueError, match="no geometry satisfying hard-core spacing"):
        refine_profiled_branch(q, intensity, (impossible,), (gui_component_to_latent(component),))

    feasible = GuiComponentBounds(
        "sphere",
        ClosedInterval(9.0, 11.0),
        ClosedInterval(0.5, 1.0),
        D=ClosedInterval(19.0, 25.0),
        sigma_D=ClosedInterval(1.0, 2.0),
    )
    violating_seed = gui_component_to_latent(
        GuiComponentParameters("sphere", 10.0, 0.7, D=20.0, sigma_D=1.1)
    )
    with pytest.raises(ValueError, match="seed violates hard-core spacing"):
        refine_profiled_branch(q, intensity, (feasible,), (violating_seed,))


def test_refinement_rejects_invalid_branch_curve_and_controls():
    q = np.geomspace(0.01, 1.0, 80)
    component = GuiComponentParameters("sphere", R=10.0, sigma_R=1.0)
    intensity = _exact_curve(q, (component,), (0.01, 1.0))
    sphere_bounds = GuiComponentBounds(
        "sphere", ClosedInterval(8.0, 14.0), ClosedInterval(0.5, 2.0)
    )
    sphere_seed = gui_component_to_latent(component)
    vertical_seed = gui_component_to_latent(
        GuiComponentParameters("vertical cylinder", R=10.0, sigma_R=0.1)
    )

    with pytest.raises(ValueError, match="same one-dimensional shape"):
        refine_profiled_branch(q, intensity[:-1], (sphere_bounds,), (sphere_seed,))
    with pytest.raises(ValueError, match="sigma_log.*strictly positive"):
        refine_profiled_branch(
            q,
            intensity,
            (sphere_bounds,),
            (sphere_seed,),
            sigma_log=np.zeros_like(q),
        )
    with pytest.raises(ValueError, match="does not match hard topology"):
        refine_profiled_branch(q, intensity, (sphere_bounds,), (vertical_seed,))
    with pytest.raises(ValueError, match="both be supplied or both be absent"):
        refine_profiled_branch(
            q,
            intensity,
            (sphere_bounds,),
            (sphere_seed,),
            resolution_bounds=ResolutionBounds(
                ClosedInterval(0.01, 0.02), ClosedInterval(4.0, 8.0)
            ),
        )
    with pytest.raises(ValueError, match="positive integer"):
        refine_profiled_branch(q, intensity, (sphere_bounds,), (sphere_seed,), max_nfev=0)
