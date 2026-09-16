from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from utils.ML_Fitting_1D_GISAXS.PosteriorV8.amplitude_polish import (
    AmplitudeBounds,
    polish_profiled_amplitudes,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8 import amplitude_polish as amplitude_polish_module
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.contract import ClosedInterval
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.gui_amplitude_constraints import (
    CANONICAL_AMPLITUDE_GAUGE,
    COEFFICIENT_POLYTOPE_SCHEMA,
    GUI_AMPLITUDE_CONSTRAINT_SCHEMA,
    GUI_AMPLITUDE_CONSTRAINT_VERSION,
    GuiAmplitudeConstraint,
)
from utils.ML_Fitting_1D_GISAXS.PosteriorV8.profiled_forward import (
    NonlinearComponent,
    POLYTOPE_FEASIBLE_NNLS_STATUS,
    ResolutionShape,
    build_design_matrix,
    evaluate_profiled_forward,
    profile_linear_amplitudes,
    to_gui_mixed_model_parameters,
)


def _constraint(*, resolution_present=True):
    return GuiAmplitudeConstraint(
        background=ClosedInterval(0.01, 0.10),
        component_intensities=(
            ClosedInterval(0.20, 0.40),
            ClosedInterval(0.60, 0.80),
        ),
        k=ClosedInterval(2.0, 4.0),
        int_res=ClosedInterval(0.10, 0.30) if resolution_present else None,
        resolution_present=resolution_present,
    )


def _components():
    return (
        NonlinearComponent("sphere", R=9.0, sigma_R=0.8),
        NonlinearComponent(
            "vertical cylinder",
            R=24.0,
            sigma_R=0.18,
            D=68.0,
            sigma_D=6.0,
        ),
    )


def test_gui_ranges_convert_to_auditable_coupled_coefficient_polytope():
    constraint = _constraint()
    polytope = constraint.coefficient_polytope()
    coefficients = (0.05, 0.9, 2.1, 0.6)

    assert polytope.schema == COEFFICIENT_POLYTOPE_SCHEMA
    assert polytope.version == GUI_AMPLITUDE_CONSTRAINT_VERSION
    assert polytope.canonical_gauge == CANONICAL_AMPLITUDE_GAUGE
    assert polytope.coefficient_order == ("BG", "a_1", "a_2", "a_res")
    assert polytope.contains(coefficients)
    assert constraint.contains(coefficients)
    audit = constraint.assess(coefficients, k=3.0)
    assert audit.schema == GUI_AMPLITUDE_CONSTRAINT_SCHEMA
    assert audit.all_constraints_satisfied
    assert {item.name: item.value for item in audit.range_checks} == pytest.approx(
        {"BG": 0.05, "k": 3.0, "Int_1": 0.3, "Int_2": 0.7, "int_Res": 0.2}
    )
    payload = constraint.to_audit_dict()
    assert payload["coefficient_polytope"]["linear_inequalities"]
    assert payload["canonical_gauge"] == CANONICAL_AMPLITUDE_GAUGE
    assert not constraint.contains((0.05, np.nan, 2.1, 0.6))
    assert not constraint.contains(coefficients[:-1])
    with pytest.raises(ValueError, match="wrong shape or contain non-finite"):
        constraint.assess(coefficients[:-1])


def test_coupled_intensity_constraints_are_not_approximated_by_axis_bounds():
    polytope = _constraint().coefficient_polytope()
    # Every coefficient is inside its marginal axis range, but a_1 requires
    # k=4 while a_2 requires k=2, so no shared k witness exists.
    coefficients = np.asarray((0.05, 1.6, 1.2, 0.2))
    assert np.all(coefficients >= np.asarray(polytope.axis_lower))
    assert np.all(coefficients <= np.asarray(polytope.axis_upper))
    assert not polytope.contains(coefficients)


def test_resolution_absent_has_no_resolution_axis_and_canonical_zero_int_res():
    constraint = _constraint(resolution_present=False)
    polytope = constraint.coefficient_polytope()
    coefficients = (0.05, 0.9, 2.1)

    assert polytope.coefficient_order == ("BG", "a_1", "a_2")
    assert polytope.contains(coefficients)
    audit = constraint.assess(coefficients)
    assert "int_Res" not in {item.name for item in audit.range_checks}
    assert audit.all_constraints_satisfied
    assert not polytope.contains((*coefficients, 0.0))


def test_independent_gui_int_ranges_do_not_require_a_simplex():
    common = {
        "background": ClosedInterval(0.0, 1.0),
        "k": ClosedInterval(1.0, 2.0),
        "int_res": None,
        "resolution_present": False,
    }
    below_one = GuiAmplitudeConstraint(
        component_intensities=(ClosedInterval(0.0, 0.4),) * 2,
        **common,
    )
    above_one = GuiAmplitudeConstraint(
        component_intensities=(ClosedInterval(0.6, 1.0),) * 2,
        **common,
    )
    assert below_one.contains((0.5, 0.2, 0.3), k=1.0)
    assert above_one.contains((0.5, 0.7, 0.8), k=1.0)
    with pytest.raises(ValueError, match="positive"):
        GuiAmplitudeConstraint(
            component_intensities=(ClosedInterval(0.0, 0.0),) * 2,
            **common,
        )
    with pytest.raises(ValueError, match="strictly positive"):
        GuiAmplitudeConstraint(
            component_intensities=(ClosedInterval(1.0, 1.0),),
            **{**common, "k": ClosedInterval(0.0, 0.0)},
        )
    zero_inclusive = GuiAmplitudeConstraint(
        component_intensities=(ClosedInterval(1.0, 1.0),),
        **{**common, "k": ClosedInterval(0.0, 2.0)},
    )
    assert zero_inclusive.contains((0.5, 1.0))
    assert not zero_inclusive.contains((0.5, 0.0))
    with pytest.raises(ValueError, match="must be omitted"):
        GuiAmplitudeConstraint(
            component_intensities=(ClosedInterval(1.0, 1.0),),
            **{**common, "int_res": ClosedInterval(0.1, 1.0)},
        )


def test_profile_linear_amplitudes_solves_inside_full_gui_polytope():
    q = np.geomspace(0.006, 1.8, 300)
    components = _components()
    resolution = ResolutionShape(sigma_res=0.015, nu_res=5.0)
    expected = np.asarray((0.04, 0.9, 2.1, 0.6))
    intensity = build_design_matrix(q, components, resolution) @ expected
    constraint = _constraint()

    result = profile_linear_amplitudes(
        q,
        intensity,
        components,
        resolution=resolution,
        amplitude_constraint=constraint,
    )

    assert result.amplitude_constraint_audit is not None
    assert result.amplitude_constraint_audit.all_constraints_satisfied
    assert constraint.contains(
        (result.background, *result.particle_amplitudes, result.resolution_amplitude),
        k=result.k,
    )
    assert result.background == pytest.approx(expected[0], rel=2e-6, abs=1e-9)
    assert result.particle_amplitudes == pytest.approx(expected[1:3], rel=5e-6)
    assert result.resolution_amplitude == pytest.approx(expected[3], rel=2e-6)


def test_profile_linear_amplitudes_handles_legal_near_zero_intensity_scale():
    q = np.geomspace(0.001, 5.0, 256)
    components = (
        NonlinearComponent(
            "sphere",
            R=20.023190759873142,
            sigma_R=9.158663931766,
        ),
    )
    expected = np.asarray((0.0, 5.940319000883555e-302))
    intensity = build_design_matrix(q, components) @ expected
    constraint = GuiAmplitudeConstraint(
        background=ClosedInterval(0.0, 1.0e8),
        component_intensities=(ClosedInterval(0.0, 1.0e8),),
        k=ClosedInterval(0.01, 1.0e8),
        resolution_present=False,
    )

    result = profile_linear_amplitudes(
        q,
        intensity,
        components,
        sigma=intensity,
        amplitude_constraint=constraint,
    )

    assert result.amplitude_constraint_audit is not None
    assert result.amplitude_constraint_audit.all_constraints_satisfied
    assert result.particle_amplitudes[0] > 0.0
    np.testing.assert_allclose(result.fitted_intensity, intensity, rtol=5e-7, atol=0.0)


def test_amplitude_polish_start_snaps_only_upper_endpoint_roundoff():
    bounds = AmplitudeBounds((0.0, 0.4), (0.1, 1.6))
    rounded = np.asarray((np.nextafter(0.1, np.inf), np.nextafter(1.6, np.inf)))

    snapped = amplitude_polish_module._feasible_initial(rounded, bounds)  # noqa: SLF001

    np.testing.assert_array_equal(snapped, np.asarray(bounds.upper))
    with pytest.raises(ValueError, match="outside the requested bounds"):
        amplitude_polish_module._feasible_initial(  # noqa: SLF001
            np.asarray((0.1 + 1.0e-8, 1.6)),
            bounds,
        )


def test_feasible_nnls_optimum_avoids_remote_polytope_midpoint_fallback():
    q = np.geomspace(0.001, 5.0, 256)
    components = (NonlinearComponent("sphere", R=1.0, sigma_R=0.02),)
    expected = np.asarray((4.5228926311615585e-4, 1.242412783898598e-305))
    intensity = build_design_matrix(q, components) @ expected
    constraint = GuiAmplitudeConstraint(
        background=ClosedInterval(0.0, 1.0e8),
        component_intensities=(
            ClosedInterval(np.nextafter(0.0, 1.0), 1.0e8),
        ),
        k=ClosedInterval(0.01, 1.0e8),
        resolution_present=False,
    )

    result = profile_linear_amplitudes(
        q,
        intensity,
        components,
        sigma=intensity,
        amplitude_constraint=constraint,
    )

    assert result.solver_status == POLYTOPE_FEASIBLE_NNLS_STATUS
    assert "global non-negative least-squares optimum" in result.solver_message
    assert result.amplitude_constraint_audit is not None
    assert result.amplitude_constraint_audit.all_constraints_satisfied
    np.testing.assert_allclose(result.fitted_intensity, intensity, rtol=5e-7, atol=0.0)


def test_profile_constraint_branch_mismatch_fails_closed_and_default_is_compatible():
    q = np.geomspace(0.01, 1.0, 100)
    component = NonlinearComponent("sphere", R=12.0, sigma_R=1.0)
    intensity = build_design_matrix(q, (component,)) @ np.asarray((0.02, 1.5))
    default = profile_linear_amplitudes(q, intensity, (component,))
    assert default.amplitude_constraint_audit is None

    with pytest.raises(ValueError, match="particle count"):
        profile_linear_amplitudes(
            q,
            intensity,
            (component,),
            amplitude_constraint=_constraint(resolution_present=False),
        )


def test_exact_log_amplitude_polish_stays_inside_the_full_gui_polytope():
    q = np.geomspace(0.006, 1.8, 300)
    components = _components()
    resolution = ResolutionShape(sigma_res=0.015, nu_res=5.0)
    expected = np.asarray((0.04, 0.9, 2.1, 0.6))
    intensity = build_design_matrix(q, components, resolution) @ expected
    constraint = _constraint()
    exact = profile_linear_amplitudes(
        q,
        intensity,
        components,
        resolution=resolution,
        amplitude_constraint=constraint,
    )
    initial = replace(
        exact,
        background=0.08,
        particle_amplitudes=(1.15, 2.05),
        resolution_amplitude=0.35,
    )

    result = polish_profiled_amplitudes(
        q,
        intensity,
        initial,
        amplitude_constraint=constraint,
        max_nfev=200,
    )

    assert result.final_metric <= result.initial_metric
    assert result.coefficient_polytope is not None
    assert result.gui_amplitude_constraint is constraint
    assert result.initial_constraint_audit is not None
    assert result.initial_constraint_audit.all_constraints_satisfied
    assert result.final_constraint_audit is not None
    assert result.final_constraint_audit.all_constraints_satisfied
    assert constraint.contains(result.final_coefficients)
    assert result.final_profile.amplitude_constraint_audit == result.final_constraint_audit
    assert result.final_coefficients == pytest.approx(tuple(expected), rel=3e-5, abs=1e-8)


def test_profile_and_polish_respect_independent_gui_int_ranges():
    q = np.geomspace(0.006, 1.8, 300)
    components = _components()
    resolution = ResolutionShape(sigma_res=0.015, nu_res=5.0)
    outside = np.asarray((0.04, 1.8, 0.2, 0.4))
    intensity = build_design_matrix(q, components, resolution) @ outside
    constraint = _constraint()

    unconstrained = profile_linear_amplitudes(
        q,
        intensity,
        components,
        resolution=resolution,
    )
    constrained = profile_linear_amplitudes(
        q,
        intensity,
        components,
        resolution=resolution,
        amplitude_constraint=constraint,
    )
    polished = polish_profiled_amplitudes(
        q,
        intensity,
        constrained,
        amplitude_constraint=constraint,
        max_nfev=200,
    )

    assert unconstrained.component_weights[0] > 0.8
    assert constraint.component_intensities[0].contains(
        constrained.component_weights[0]
    )
    assert polished.final_profile.component_weights[0] <= 0.4 + 1e-9
    assert polished.final_constraint_audit is not None
    assert polished.final_constraint_audit.all_constraints_satisfied
    assert constraint.contains(polished.final_coefficients)


def test_amplitude_polish_rejects_polytope_infeasible_start_and_axis_box_mix():
    q = np.geomspace(0.006, 1.8, 300)
    components = _components()
    resolution = ResolutionShape(sigma_res=0.015, nu_res=5.0)
    constraint = _constraint()
    expected = np.asarray((0.04, 0.9, 2.1, 0.6))
    intensity = build_design_matrix(q, components, resolution) @ expected
    exact = profile_linear_amplitudes(
        q,
        intensity,
        components,
        resolution=resolution,
        amplitude_constraint=constraint,
    )
    # Each coefficient remains inside the marginal axis projection, while no
    # one shared k can satisfy both independent Int ranges.
    wrong_weights = replace(
        exact,
        particle_amplitudes=(1.6, 1.2),
        resolution_amplitude=0.2,
    )

    with pytest.raises(ValueError, match="outside the requested GUI amplitude polytope"):
        polish_profiled_amplitudes(
            q,
            intensity,
            wrong_weights,
            amplitude_constraint=constraint,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        polish_profiled_amplitudes(
            q,
            intensity,
            exact,
            bounds=AmplitudeBounds((0.0,) * 4, (10.0,) * 4),
            amplitude_constraint=constraint,
        )


def test_k1_k10_int2_round_trips_the_exact_gui_parameterization():
    q = np.geomspace(0.01, 1.0, 160)
    component = NonlinearComponent("sphere", R=12.0, sigma_R=1.0)
    constraint = GuiAmplitudeConstraint(
        background=ClosedInterval(0.05, 0.05),
        component_intensities=(ClosedInterval(2.0, 2.0),),
        k=ClosedInterval(10.0, 10.0),
        resolution_present=False,
    )
    coefficients = (0.05, 20.0)
    intensity = build_design_matrix(q, (component,)) @ np.asarray(coefficients)

    assert constraint.contains(coefficients)
    assert constraint.contains(coefficients, k=10.0)
    assert not constraint.contains(coefficients, k=20.0)
    invalid_actual = constraint.assess(coefficients, k=20.0)
    assert invalid_actual.coefficient_polytope_satisfied
    assert not invalid_actual.auxiliary_k_is_witness
    assert not invalid_actual.all_constraints_satisfied

    result = profile_linear_amplitudes(
        q,
        intensity,
        (component,),
        amplitude_constraint=constraint,
    )
    spec, parameters = to_gui_mixed_model_parameters(result)
    assert spec == ["sphere"]
    assert result.k == pytest.approx(10.0)
    assert result.component_weights == pytest.approx((2.0,))
    assert parameters[0] == pytest.approx(2.0)
    assert parameters[-1] == pytest.approx(10.0)
    assert evaluate_profiled_forward(q, result) == pytest.approx(intensity, rel=2e-10)


def test_fixed_multicomponent_and_resolution_boundaries_keep_one_shared_k():
    constraint = GuiAmplitudeConstraint(
        background=ClosedInterval(0.0, 0.0),
        component_intensities=(
            ClosedInterval(0.25, 0.25),
            ClosedInterval(2.0, 2.0),
        ),
        k=ClosedInterval(4.0, 4.0),
        int_res=ClosedInterval(0.5, 0.5),
        resolution_present=True,
    )
    coefficients = (0.0, 1.0, 8.0, 2.0)
    audit = constraint.assess(coefficients, k=4.0)
    assert audit.all_constraints_satisfied
    assert audit.auxiliary_k_feasible_interval == pytest.approx((4.0, 4.0))
    assert not constraint.contains((0.0, 1.0, 8.0, 2.0), k=3.999)
