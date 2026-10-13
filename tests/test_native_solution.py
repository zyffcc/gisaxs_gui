"""A quick-fit solution (``native_v5``) put into Fitting's manual model: the same curve, stated when not."""

from __future__ import annotations

import numpy as np
import pytest

from src.gimap.features.fitting.domain.native_solution import native_solution_mapping
from src.gimap.features.fitting.infrastructure.adapters.experimental_fit import forward

Q_NM = np.linspace(0.05, 2.5, 240)
GLOBALS = {"background": 1.3, "resolution_amplitude": 800.0, "sigma_Res": 0.02, "nu_Res": 4.0}
SPHERE = {"R": 9.4, "sigma_R": 0.3, "D": 53.4, "sigma_D": 0.38}
CYLINDER = {"R": 6.1, "sigma_R": 0.45, "h": 20.7, "sigma_h": 0.18, "D": 398.0, "sigma_D": 0.19}
VERTICAL = {"R": 7.8, "sigma_R": 0.32, "D": 31.2, "sigma_D": 0.38}
TYPE_IDS = {"sphere": 1, "random_cylinder": 2, "vertical_cylinder": 3}


def _row(kind: str, params: dict, amplitude: float = 1200.0, global_params=None) -> dict:
    components = [{"type": kind, "type_id": TYPE_IDS[kind], "weight": 1.0, "amplitude": amplitude, "params": params}]
    curve = forward(Q_NM, components, GLOBALS)
    return {"workflow": "native_v5", "combination": kind, "components": components,
            "global_params": dict(GLOBALS if global_params is None else global_params),
            "native_q": (-Q_NM).tolist(), "native_fit": curve.tolist()}


@pytest.mark.parametrize("kind, params", [("sphere", SPHERE), ("random_cylinder", CYLINDER)])
def test_the_manual_model_draws_the_same_curve(kind, params) -> None:
    result = native_solution_mapping(_row(kind, params))
    assert result.max_deviation < 1e-6  # the same formulas once the spreads are in nm
    component = result.mapping.components[0]
    assert component.shape == {"sphere": "Sphere", "random_cylinder": "Cylinder"}[kind]
    values = component.parameters
    assert values["intensity"] == pytest.approx(1200.0, rel=1e-6)
    assert values["radius"] == params["R"] and values["sigma_radius"] == pytest.approx(params["R"] * params["sigma_R"])
    assert values["sigma_diameter"] == pytest.approx(params["D"] * params["sigma_D"])  # σD in nm in Fitting
    if kind == "random_cylinder":
        assert values["sigma_height"] == pytest.approx(params["h"] * params["sigma_h"])
    globals_ = result.mapping.global_parameters
    assert globals_["background"] == pytest.approx(1.3) and globals_["int_res"] == pytest.approx(800.0)
    assert (globals_["sigma_res"], globals_["nu_res"], globals_["k_value"]) == (0.02, 4.0, 1.0)


def test_a_vertical_cylinder_is_loaded_as_a_start_and_the_difference_is_measured() -> None:
    result = native_solution_mapping(_row("vertical_cylinder", VERTICAL))
    values = result.mapping.components[0].parameters
    assert result.mapping.components[0].shape == "Vertical Cylinder"
    assert values["sigma_radius"] == pytest.approx(0.32)  # Fitting's Vertical Cylinder takes σR/R
    assert values["intensity"] > 0 and result.max_deviation > 0.05  # radii weighted by R⁴ there: not the same curve


def test_background_terms_in_other_units_are_found_from_the_curve() -> None:
    normalised = {**GLOBALS, "background": 1.3e-3, "resolution_amplitude": 0.8}
    result = native_solution_mapping(_row("sphere", SPHERE, global_params=normalised))
    assert result.max_deviation < 1e-6
    assert result.mapping.global_parameters["background"] == pytest.approx(1.3, rel=1e-4)


def test_a_solution_without_components_or_with_an_unknown_family_is_refused() -> None:
    with pytest.raises(ValueError):
        native_solution_mapping({"components": []})
    with pytest.raises(ValueError, match="no manual model"):
        native_solution_mapping({"components": [{"type": "core_shell", "params": SPHERE}]})


def test_show_in_fitting_puts_the_solution_into_components_and_global(tmp_path) -> None:
    from tests.test_fitting_presentation import _fitting_window

    app, window = _fitting_window()
    binding = window.runtime.fitting
    curve = tmp_path / "galaxi_fit_input.dat"
    row = _row("sphere", SPHERE)
    q_file = -Q_NM / 10.0  # the curve Analyze writes: q in Å⁻¹
    np.savetxt(curve, np.column_stack([q_file, row["native_fit"], np.sqrt(row["native_fit"])]))
    binding.import_1d_file(curve, q_view="fold")
    window.components.fitting_workspace.show_fit_curve()
    app.processEvents()

    assert binding.show_workflow_candidate(row) is True
    widget = binding._iter_particle_widget_ids()[0]
    assert binding.get_particle_shape(widget) == "Sphere"
    assert binding._get_particle_parameter(widget, "R", 0.0) == pytest.approx(9.4)
    assert binding._get_particle_parameter(widget, "sigma_R", 0.0) == pytest.approx(9.4 * 0.3, rel=1e-6)
    assert binding._get_particle_parameter(widget, "Int", 0.0) == pytest.approx(1200.0, rel=1e-4)
    assert window.fitBGValue.value() == pytest.approx(1.3, abs=1e-3)
    assert window.fitIntResValue.value() == pytest.approx(800.0, rel=1e-4)
    assert binding.has_fitting_data and binding.fitting.get("meta", {}).get("source") != "native_v5"  # Fitting's model
    assert "draws the same curve" in window.fittingInlineFeedback.text()
    drawn = binding.fitting  # q as shown (Å⁻¹); the model works in nm⁻¹
    assert drawn["meta"]["q_source_unit"] == "angstrom"
    expected = forward(10.0 * np.abs(np.asarray(drawn["q"])), row["components"], GLOBALS)
    np.testing.assert_allclose(drawn["I"], expected, rtol=1e-4)  # the solution's curve, drawn by Fitting's model
    window.close()


def test_a_1d_predict_row_without_distance_or_resolution_peak_converts() -> None:
    """1D Predict gives D = None (no interference) and normalised background terms (sigma_Res None)."""
    components = [{"type": "sphere", "type_id": 1, "weight": 0.7, "amplitude": 900.0,
                   "params": {"R": 8.0, "sigma_R": 0.3, "h": None, "sigma_h": None, "D": None, "sigma_D": None}}]
    flat = {**GLOBALS, "resolution_amplitude": 0.0}
    no_structure = [{**components[0], "params": {**components[0]["params"], "D": 1e9, "sigma_D": 0.01}}]
    curve = forward(Q_NM, no_structure, flat)  # S(q) → 1 for a huge, ordered distance
    row = {"combination": "sphere", "components": components,
           "global_params": {"rho_BG": 1e-4, "sigma_Res": None, "nu_Res": None, "rho_Res": None},
           "native_q": Q_NM.tolist(), "native_fit": curve.tolist()}
    result = native_solution_mapping(row)
    assert result.mapping.components[0].parameters["diameter"] == 0.0  # no S(q) in the manual model
    assert result.mapping.global_parameters["int_res"] == 0.0
    assert result.mapping.global_parameters["background"] == pytest.approx(1.3, rel=1e-3)  # found from the curve
    assert result.max_deviation < 1e-3
