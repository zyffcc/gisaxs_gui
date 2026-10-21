"""The fit method ``fit1d`` of 1D Predict: its options, its panel, and a fit through the job process.

The fits need the external gisaxs1d kit (``GISAXS1D_KIT_HOME`` or ``utils/gisaxs1d_gui_kit``) and are skipped
without it. They run in a job process, as in the program, so the kit's code is never imported here.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.gimap.app.jobs import JobRequest
from src.gimap.features.fitting.application.workflow_v5 import (
    default_options,
    fit1d_kit_home,
    validate_options,
)
from src.gimap.features.fitting.domain.native_solution import native_solution_mapping
from src.gimap.integrations.jobs import LocalProcessJobRunner

HANDLER = "src.gimap.features.fitting.infrastructure.adapters.workflow_v5:run_workflow_job"
needs_kit = pytest.mark.skipif(
    not (fit1d_kit_home() / "gisaxs1d_kit" / "core.py").is_file(),
    reason="the gisaxs1d kit (GISAXS1D_KIT_HOME or utils/gisaxs1d_gui_kit) is not installed",
)
QZ = 0.5
"""An arbitrary out-of-plane q for the synthetic fixture (nm⁻¹)."""


def _example(step: int = 2):
    """Synthetic sphere and cylinder populations; no unpublished sample is embedded in tests."""
    from scipy.special import j1

    q = np.geomspace(0.035, 1.8, 240)
    x = 3.0 * np.hypot(q, QZ)
    sphere = (3.0 * (np.sin(x) - x * np.cos(x)) / x**3)**2
    radial = q * 18.0
    cylinder = (2.0 * j1(radial) / radial)**2
    spread = np.exp(-0.5 * (q * 2.0)**2)
    structure = (1.0 - spread**2) / (1.0 - 2.0 * spread * np.cos(q * 11.0) + spread**2)
    mean = 8000.0 * sphere * structure + 1500.0 * cylinder + 30.0 + 300.0 / (1.0 + (q / 0.08)**2)
    intensity = np.random.default_rng(314).poisson(mean).astype(float)
    sigma = np.sqrt(np.maximum(intensity, 1.0))
    return q[::step], intensity[::step], sigma[::step]


def _run(payload):
    return LocalProcessJobRunner().run(JobRequest(handler=HANDLER, payload=payload, timeout_seconds=600))


def test_fit1d_options_are_validated():
    assert default_options()["fit1d_policy"] == "fast" and default_options()["fit1d_qz"] is None
    options = validate_options(dict(method="fit1d", components=(3, 1), fit1d_qz=QZ, fit1d_policy="auto"))
    assert options["components"] == [3, 1] and options["fit1d_qz"] == QZ and options["fit1d_mode"] == "gisaxs"
    assert validate_options(dict(method="fit1d", components=[1], nu_res=2.5))["nu_res"] == 2.5
    with pytest.raises(ValueError, match="needs the complete composition"):
        validate_options(dict(method="fit1d"))  # one curve does not identify the composition
    with pytest.raises(ValueError, match="no random cylinder"):
        validate_options(dict(method="fit1d", components=[1, 2]))
    with pytest.raises(ValueError, match="cannot fix sigma_res"):
        validate_options(dict(method="fit1d", components=[1], sigma_res=0.01))
    for bad in (dict(fit1d_policy="preview"), dict(fit1d_mode="giwaxs"), dict(fit1d_qz=-1.0),
                dict(fit1d_qz=float("nan")), dict(fit1d_model=""), dict(fit1d_sigma_floor=0.9)):
        with pytest.raises(ValueError, match="fit1d_"):
            validate_options(dict(method="fit1d", components=[1], **bad))
    # The other methods keep their rules; the fit1d values travel with them unchanged.
    assert validate_options(dict(method="experimental", components=[2], fit1d_policy="robust"))["fit1d_policy"] == "robust"
    with pytest.raises(ValueError):
        validate_options(dict(method="model", nu_res=2))


def test_the_kit_arguments_follow_the_options():
    from src.gimap.features.fitting.infrastructure.adapters.fit1d_method import kit_arguments

    item = dict(q=np.linspace(0.1, 2, 20), observed=np.ones(20), sigma=np.full(20, 0.1), side="positive", sign=1)
    arguments = kit_arguments(item, validate_options(dict(method="fit1d", components=[1, 3, 3], fit1d_qz=QZ,
                                                           fit1d_model="au_np_si", nu_res=3.0)))
    assert arguments["composition"] == "sphere,vc,vc" and arguments["qz"] == QZ and arguments["q_unit"] == "nm^-1"
    assert arguments["model"] == "au_np_si" and arguments["policy"] == "fast" and arguments["lowq_nu"] == 3.0
    saxs = kit_arguments(item, validate_options(dict(method="fit1d", components=[1], fit1d_qz=QZ, fit1d_mode="saxs",
                                                     nu_res=3.0)))
    assert saxs["mode"] == "saxs" and saxs["qz"] == 0.0 and "lowq_nu" not in saxs  # SAXS has no qz and no low-q term


def test_the_settings_offer_fit1d_and_keep_its_values():
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    dialog = WorkflowV5Dialog()
    assert dialog.method.findData("fit1d") >= 0
    assert not dialog.fit1d_policy.isEnabled() and dialog.fix_sigma.isEnabled()
    assert dialog.options()["fit1d_qz"] is None and dialog.options()["fit1d_policy"] == "fast"
    dialog.method.setCurrentIndex(dialog.method.findData("fit1d"))
    assert dialog.fit1d_policy.isEnabled() and dialog.fit1d_qz.isEnabled() and not dialog.fix_sigma.isEnabled()
    assert not dialog.numerical.isEnabled() and not dialog.amplitude_calibration.isEnabled()
    with pytest.raises(ValueError, match="needs the complete composition"):
        dialog.options()
    dialog.save_settings()  # said in the status line, not raised
    assert "needs the complete composition" in dialog.status.text()
    dialog.component_boxes[0].setCurrentIndex(1)
    dialog.component_boxes[1].setCurrentIndex(3)
    dialog.fix_sigma.setChecked(True)  # left over from another method: fit1d does not take it
    dialog.fit1d_use_qz.setChecked(True)
    dialog.fit1d_qz.setValue(QZ)
    dialog.fit1d_policy.setCurrentIndex(dialog.fit1d_policy.findData("robust"))
    dialog.fit1d_sigma_floor.setValue(3.0)
    options = dialog.options()
    assert options["method"] == "fit1d" and options["components"] == [1, 3] and options["sigma_res"] is None
    assert options["fit1d_qz"] == QZ and options["fit1d_policy"] == "robust"
    assert math.isclose(options["fit1d_sigma_floor"], 0.03)
    dialog.close()
    saved = WorkflowV5Dialog(options=dict(options, fit1d_model="my_network", fit1d_mode="saxs"))
    assert saved.options() == validate_options(dict(options, fit1d_model="my_network", fit1d_mode="saxs"))
    saved.close()


def test_main_fit_and_in_situ_keep_the_saved_fit1d_method():
    from src.gimap.features.fitting.presentation.bindings.ai_job_execution import workflow_options_for_mode

    saved = {**default_options(), "method": "fit1d", "components": [1, 3]}
    assert workflow_options_for_mode(saved, "full")["method"] == "fit1d"
    assert workflow_options_for_mode(saved, "fit1d")["method"] == "fit1d"
    assert workflow_options_for_mode(saved, "fast")["method"] == "model"


def test_a_missing_kit_says_where_to_put_it(tmp_path, monkeypatch):
    monkeypatch.setenv("GISAXS1D_KIT_HOME", str(tmp_path / "no_kit_here"))
    q = np.linspace(0.05, 3, 40)
    result = _run(dict(output_dir=str(tmp_path / "output"), q=q.tolist(), intensity=(1 / q).tolist(),
                       sigma=(0.05 / q).tolist(),
                       options=dict(method="fit1d", components=[1], side="positive")))
    assert result.status == "failed"
    message = result.error.message
    assert "fit1d kit was not found" in message and "no_kit_here" in message
    assert "gisaxs1d_gui_kit" in message and "GISAXS1D_KIT_HOME" in message


@needs_kit
def test_current_curve_gives_a_native_row_with_fit1d_s_own_curve(tmp_path):
    q, intensity, sigma = _example()
    result = _run(dict(output_dir=str(tmp_path / "output"), q=q.tolist(), intensity=intensity.tolist(),
                       sigma=sigma.tolist(),
                       options=dict(method="fit1d", components=[1, 3], side="positive", fit1d_qz=QZ)))
    assert result.succeeded, result.error
    value = result.value
    assert value["summary"]["profile"].startswith("fit1d") and value["records"][0]["status"] == "complete"
    (row,) = value["candidates"]
    assert row["workflow"] == "native_v5" and row["best_source"] == "fit1d" and row["rank"] == 1
    assert row["side"] == "positive" and row["combination"] == "sphere + vertical_cylinder"
    assert row["file"] == "Current curve" and row["conditions_source"] == "user_composition"
    sphere, cylinder = row["components"]
    assert (sphere["type"], sphere["type_id"]) == ("sphere", 1)
    assert (cylinder["type"], cylinder["type_id"]) == ("vertical_cylinder", 3)
    for component in row["components"]:
        assert set(component["params"]) == {"R", "sigma_R", "D", "sigma_D"}
        assert all(isinstance(v, float) and math.isfinite(v) for v in component["params"].values())
        assert component["amplitude"] >= 0 and 0 <= component["weight"] <= 1
        own = component["fit1d"]  # sigma_D is the kit's legacy_sigma_D; its own w / D stays with the row
        assert math.isclose(component["params"]["sigma_D"], own["w_rel"] / math.sqrt(2 * math.pi), rel_tol=1e-12)
        assert math.isclose(component["params"]["R"], own["R_nm"], rel_tol=1e-12)
    assert math.isclose(sum(c["weight"] for c in row["components"]), 1.0, rel_tol=1e-9)
    assert sphere["params"]["R"] > 0  # a finite physical result; parameter accuracy is tested by the research suite
    assert set(row["global_params"]) == {"background", "resolution_amplitude", "sigma_Res", "nu_Res"}
    assert row["global_params"]["nu_Res"] == 2.0 and row["global_params"]["sigma_Res"] > 0
    # The curves are fit1d's own, at the measured points.
    assert len(row["native_q"]) == len(row["native_fit"]) == len(row["observed"]) == len(row["sigma"]) == len(q)
    np.testing.assert_allclose(row["native_q"], q, rtol=1e-12)
    np.testing.assert_allclose(row["observed"], intensity, rtol=1e-12)
    assert row["display_q"] == row["native_q"] and row["display_fit"] == row["native_fit"]
    assert np.all(np.isfinite(row["native_fit"])) and row["best_log_rmse"] < 1.0
    assert math.isclose(row["best_chi2_weighted"], row["fit1d"]["chi2_per_point"], rel_tol=1e-6)
    assert math.isclose(row["signed_weighted_rms"] ** 2, row["best_chi2_weighted"], rel_tol=1e-9)
    assert row["fit1d"]["qz_nm_inv"] == QZ and row["fit1d"]["qz_given"] and row["fit1d"]["composition"] == "sphere,vc"
    assert "number-weighted" in row["unit_contract"]["sigma_R"] and "legacy_sigma_D" in row["unit_contract"]["sigma_D"]
    assert "different curve" in row["validation_scope"]
    assert not any("qz = 0" in warning for warning in row["warnings"])
    for name in row["fit1d"]["flags"]:  # what the kit flags is said in words
        assert name in ("at_bound", "weakly_constrained", "quality", "ood", "ambiguous", "sigma_assumed", "fallback")
    statuses = {r["key"]: r["status"] for r in row["fit1d"]["table"]}
    for key, status in statuses.items():
        if status in ("ambiguous", "weak", "at_bound") and not key.startswith(("low_q", "background")):
            assert any(warning.startswith(key + " = ") for warning in row["warnings"]), key
    # The manual model takes the sizes; how far its curve is from fit1d's is measured, not assumed.
    converted = native_solution_mapping(row)
    assert [c.shape for c in converted.mapping.components] == ["Sphere", "Vertical Cylinder"]
    assert converted.mapping.components[0].parameters["radius"] == sphere["params"]["R"]
    assert math.isfinite(converted.max_deviation) and converted.max_deviation >= 0
    side = tmp_path / "output" / "0001_Current curve" / "positive"
    saved = json.loads((side / "solutions.json").read_text(encoding="utf-8"))
    assert saved["method"] == "fit1d" and saved["solutions"][0]["components"] == row["components"]
    display = np.load(side / "display.npz")
    np.testing.assert_allclose(display["rendered_curves"][0], row["native_fit"])
    assert any(name.startswith("component_") for name in display.files)
    assert row["display_components"]
    for name, values in row["display_components"].items():
        np.testing.assert_allclose(display["component_" + name], values)
    assert np.loadtxt(side / "fitting_curves.csv", delimiter=",", skiprows=1).shape == (len(q), 2)


@needs_kit
def test_files_are_fitted_per_half_and_a_bad_file_does_not_stop_the_batch(tmp_path):
    q, intensity, sigma = _example(step=4)
    both = tmp_path / "both_halves.dat"  # the same half on both sides of q = 0
    np.savetxt(both, np.column_stack([np.r_[-q[::-1], q], np.r_[intensity[::-1], intensity],
                                      np.r_[sigma[::-1], sigma]]))
    bare = tmp_path / "no_sigma.dat"
    np.savetxt(bare, np.column_stack([q, intensity]))
    invalid = tmp_path / "invalid.dat"
    invalid.write_text("bad data")
    result = _run(dict(output_dir=str(tmp_path / "output"), files=[str(invalid), str(both), str(bare)],
                       options=dict(method="fit1d", components=[1, 3], fit1d_sigma_floor=0.03)))
    assert result.succeeded, result.error
    assert [r["status"] for r in result.value["records"]] == ["failed", "complete", "complete"]
    rows = result.value["candidates"]
    assert [(r["file"], r["side"]) for r in rows] == [
        ("both_halves.dat", "positive"), ("both_halves.dat", "negative"), ("no_sigma.dat", "positive")]
    positive, negative, bare_row = rows
    assert all(v < 0 for v in negative["native_q"]) and all(v > 0 for v in positive["native_q"])
    np.testing.assert_allclose(negative["native_fit"], positive["native_fit"], rtol=1e-9)  # the same curve, mirrored
    for row in rows:  # no qz given: said in the row, not guessed
        assert any("qz = 0" in warning for warning in row["warnings"])
        assert row["fit1d"]["qz_nm_inv"] == 0 and not row["fit1d"]["qz_given"] and row["fit1d"]["sigma_floor"] == 0.03
    assert any("sigma was estimated" in warning for warning in bare_row["warnings"])
    assert not any("sigma was estimated" in warning for warning in positive["warnings"])
    assert json.loads((tmp_path / "output/top20_candidates.json").read_text(encoding="utf-8"))[0]["best_source"] == "fit1d"


def test_in_situ_routes_a_recipe_with_fit1d_to_fit1d():
    from types import SimpleNamespace

    from src.gimap.features.fitting.presentation.bindings.insitu_curve_processing import InsituCurveProcessingMixin

    binding = InsituCurveProcessingMixin()
    binding._insitu_workflow_settings = lambda: dict(use_previous=False, full_auto_fit=True, auto_refine=False)
    binding._log_insitu_workflow = lambda *args: None
    binding.ui = SimpleNamespace()
    binding.fitting_view_model = SimpleNamespace(insitu=SimpleNamespace(recipe=SimpleNamespace(
        model={"workflow_v5": dict(method="fit1d", components=[1, 3])})))
    calls = []

    def start(mode):
        calls.append(mode)
        binding._ai_job_thread = object()

    binding._start_ai_prediction = start
    binding._run_insitu_workflow_fit({})
    assert calls == ["fit1d"]


@needs_kit
def test_the_fit_of_the_open_curve_and_of_in_situ_frames_reaches_fit1d(tmp_path):
    """Main Fit and in-situ frames go through ``GenerateCandidates(AiPipelinePredictor)`` with the saved options."""
    from src.gimap.features.fitting.application import CandidateGenerationRequest, GenerateCandidates
    from src.gimap.features.fitting.application.workflow_v5 import bundled_workflow
    from src.gimap.features.fitting.infrastructure.adapters.ai_pipeline import AiPipelinePredictor
    from src.gimap.features.fitting.presentation.bindings.ai_job_execution import workflow_options_for_mode

    q, intensity, sigma = _example(step=4)
    saved = validate_options(dict(method="fit1d", components=[1, 3], fit1d_qz=QZ, side="both"))
    request = CandidateGenerationRequest(
        model_path=bundled_workflow(), output_dir=tmp_path / "v5_run",
        q=np.r_[-q[::-1], q], intensity=np.r_[intensity[::-1], intensity], sigma=np.r_[sigma[::-1], sigma],
        profile={"name": "Full"}, constraints={"workflow_v5": workflow_options_for_mode(saved, "fit1d")},
    )
    result = GenerateCandidates(AiPipelinePredictor(), LocalProcessJobRunner()).execute(request)
    assert result.profile_name.startswith("fit1d") and result.exit_code == 0
    assert [(row["side"], row["best_source"], row["workflow"]) for row in result.candidates] == [
        ("positive", "fit1d", "native_v5"), ("negative", "fit1d", "native_v5")]
    for row in result.candidates:  # what the in-situ record keeps of a native_v5 row
        assert row["signed_weighted_rms"] > 0 and row["unit_contract"]["q"] == "nm^-1"
        assert row["components"][0]["params"]["R"] > 0


@needs_kit
def test_a_fit1d_row_goes_into_components_as_a_start_and_says_its_curve_differs(tmp_path):
    from tests.test_fitting_presentation import _fitting_window

    q, intensity, sigma = _example(step=4)
    result = _run(dict(output_dir=str(tmp_path / "output"), q=q.tolist(), intensity=intensity.tolist(),
                       sigma=sigma.tolist(),
                       options=dict(method="fit1d", components=[1, 3], side="positive", fit1d_qz=QZ)))
    (row,) = result.value["candidates"]
    app, window = _fitting_window()
    binding = window.runtime.fitting
    curve = tmp_path / "cut_fit_input.dat"
    np.savetxt(curve, np.column_stack([q / 10.0, intensity, sigma]))  # the curve Analyze writes: q in Å⁻¹
    binding.import_1d_file(curve, q_view="fold")
    window.components.fitting_workspace.show_fit_curve()
    app.processEvents()
    assert binding.show_workflow_candidate(row) is True
    shapes = [binding.get_particle_shape(widget) for widget in binding._iter_particle_widget_ids()]
    assert shapes[:2] == ["Sphere", "Vertical Cylinder"] and set(shapes[2:]) <= {"None"}  # unused slots
    first = binding._iter_particle_widget_ids()[0]
    assert binding._get_particle_parameter(first, "R", 0.0) == pytest.approx(row["components"][0]["params"]["R"])
    feedback = window.fittingInlineFeedback.text()
    assert "as a start" in feedback and "fit1d's curve" in feedback and "own physics" in feedback
    window.close()


@needs_kit
def test_four_populations_and_both_networks_are_available(tmp_path):
    q, intensity, sigma = _example(step=4)
    for network in ("general", "au_np_si"):
        result = _run(dict(output_dir=str(tmp_path / network), q=q.tolist(), intensity=intensity.tolist(),
                           sigma=sigma.tolist(), options=dict(method="fit1d", components=[1, 1, 3, 3],
                                                              side="positive", fit1d_qz=QZ, fit1d_model=network)))
        assert result.succeeded, result.error
        (row,) = result.value["candidates"]
        assert [c["type_id"] for c in row["components"]] == [1, 1, 3, 3]
        assert row["fit1d"]["model"] == network
        assert sum(name.startswith("pop") for name in row["display_components"]) == 4
        np.testing.assert_allclose(np.sum(list(row["display_components"].values()), axis=0), row["display_fit"])
        assert len(row["display_q"]) == len(q)


@needs_kit
def test_effective_sigma_and_component_preview_are_preserved(tmp_path):
    from src.gimap.features.fitting.presentation.workflow_v5_dialog import WorkflowV5Dialog

    q, intensity, sigma = _example(step=4)
    result = _run(dict(output_dir=str(tmp_path / "floor"), q=q.tolist(), intensity=intensity.tolist(),
                       sigma=sigma.tolist(), options=dict(method="fit1d", components=[1, 3], side="positive",
                                                         fit1d_qz=QZ, fit1d_sigma_floor=0.03)))
    assert result.succeeded, result.error
    (row,) = result.value["candidates"]
    effective = np.hypot(sigma, 0.03 * intensity)
    np.testing.assert_allclose(row["sigma"], effective)
    np.testing.assert_allclose(row["measurement_sigma"], sigma)
    residual = (np.asarray(row["native_fit"]) - intensity) / effective
    assert row["best_chi2_weighted"] == pytest.approx(float(np.mean(residual**2)))
    dialog = WorkflowV5Dialog()
    dialog.set_results([row], tmp_path)
    dialog._preview(0)
    labels = [line.get_label() for line in dialog.figure.axes[0].lines]
    assert set(row["display_components"]).issubset(labels)
    dialog.close()


# -- constraints, parameters at an edge and candidates (single-curve page) ------------------------------------

SINGLE = "src.gimap.features.fitting.infrastructure.adapters.single_fit1d:run_single_fit1d"
SETTINGS = dict(fit1d_policy="fast", fit1d_mode="gisaxs", fit1d_model="general", fit1d_qz=QZ, fit1d_sigma_floor=0.0)


def _data(q, intensity, sigma):
    from types import SimpleNamespace

    return SimpleNamespace(q=np.asarray(q, float), intensity=np.asarray(intensity, float),
                           sigma=None if sigma is None else np.asarray(sigma, float))


def _single(data, model, **settings):
    from src.gimap.features.fitting.application.single_fit1d import fit1d_request

    request = fit1d_request(data, model, {**SETTINGS, **settings})
    result = LocalProcessJobRunner().run(JobRequest(handler=SINGLE, payload=request.payload, timeout_seconds=600))
    assert result.succeeded, result.error
    return result.value


def _sphere_and_cylinder(changes=None):
    """Model: a vertical cylinder first, then a sphere (fit1d numbers the sphere pop1)."""
    from src.gimap.features.fitting.application.single_fit import FitModel, new_component

    model = FitModel((new_component("vertical_cylinder"), new_component("sphere")))
    for path, values in (changes or {}).items():
        model = model.with_parameter(path, **values)
    return model


def test_the_model_s_fixed_values_and_changed_ranges_become_fit1d_constraints():
    from src.gimap.features.fitting.domain.fit1d_bounds import as_kit, model_constraints

    model = _sphere_and_cylinder({(1, "R"): dict(value=3.0, free=False), (0, "D"): dict(lower=50.0, upper=400.0),
                                    (0, "sigma_D"): dict(value=0.1, free=False)})
    model = model.with_parameter(("globals", "background"), value=0.0, free=False)
    constraints = model_constraints(model)
    kit = as_kit(constraints)
    # pop1 is the sphere (Model's second particle), pop2 the cylinder; unchanged default ranges are not passed.
    assert kit == {"pop1": {"R_nm": 3.0}, "pop2": {"D_nm": [50.0, 400.0], "w_rel": pytest.approx(0.1 * math.sqrt(2 * math.pi))},
                   "background": 0.0}
    assert {item.name: item.component for item in constraints} == {
        "pop1.R_nm": 1, "pop2.D_nm": 0, "pop2.w_rel": 0, "background": None}
    assert all(item.source == "model" for item in constraints)
    # SAXS has no structure factor: no D, no σD/D.
    assert as_kit(model_constraints(model, structure_factor=False)) == {"pop1": {"R_nm": 3.0}, "background": 0.0}
    with pytest.raises(ValueError, match="sigma_R_rel fixed at 0.01"):
        model_constraints(_sphere_and_cylinder({(1, "sigma_R"): dict(value=0.01, free=False)}))
    # A range partly outside what fit1d accepts is clipped to it.
    clipped = model_constraints(_sphere_and_cylinder({(1, "sigma_R"): dict(lower=0.0, upper=0.2)}))
    assert as_kit(clipped) == {"pop1": {"sigma_R_rel": [0.02, 0.2]}}


def test_a_range_beyond_what_the_curve_resolves_is_said_before_the_fit():
    from src.gimap.features.fitting.domain.fit1d_bounds import BEYOND, measurable, model_constraints, range_notes

    assert measurable(0.05, 2.0)["R_nm"] == pytest.approx((0.25, 100.0))
    assert measurable(0.05, 2.0)["D_nm"] == pytest.approx((math.pi, 40 * math.pi))
    narrow = model_constraints(_sphere_and_cylinder({(1, "R"): dict(lower=2.0, upper=4.0)}))
    assert range_notes(narrow, 0.05, 2.0) == []  # narrower: nothing to say, an edge is resolved by the fit
    wide = model_constraints(_sphere_and_cylinder({(1, "R"): dict(lower=0.2, upper=450.0)}))
    notes = range_notes(wide, 0.05, 2.0)
    assert [note.template for note in notes] == [BEYOND] and "reaches 450 nm" in notes[0].text


def test_parameters_at_an_edge_are_told_apart_and_answered():
    from src.gimap.features.fitting.domain.fit1d_bounds import Constraint
    from src.gimap.features.fitting.domain.fit1d_edges import (
        AT_OTHER, REMAINING, classify_edges, remaining_note, resolution)

    user = Constraint("pop2", "D_nm", "range", 50.0, 400.0, 0)
    fixed = Constraint("pop1", "D_nm", "fixed", 30.0, 30.0, 1)
    populations = [dict(name="pop1", R_nm=99.0, sigma_R_rel=0.03, D_nm=30.0),
                   dict(name="pop2", R_nm=18.0, D_nm=400.0, w_rel=0.5)]
    edges = classify_edges(["pop1.R_nm@upper", "pop1.sigma_R_rel@lower", "pop2.D_nm@upper", "pop2.w_rel@upper",
                            "pop1.D_nm@lower", "low_q.nu@upper"], [user, fixed], populations)
    # A value the user fixed is never an edge to answer.
    assert [(edge.name, edge.kind) for edge in edges] == [
        ("pop1.R_nm", "default"), ("pop1.sigma_R_rel", "physical"), ("pop2.D_nm", "user"), ("pop2.w_rel", "default"),
        ("low_q.nu", "other")]
    radius = resolution(edges[0], [user], populations, 0.05, 2.0)
    assert (radius.kind, radius.source, radius.low, radius.high) == ("range", "widened", 0.25, 2 * 100.0)
    assert resolution(edges[1], [user], populations, 0.05, 2.0) is None  # physical: nothing to widen (``settle``)
    distance = resolution(edges[2], [user], populations, 0.05, 2.0)  # the user's range is widened as well
    assert (distance.kind, distance.source, distance.low, distance.high, distance.component) == (
        "range", "widened", 50.0, 800.0, 0)
    order = resolution(edges[3], [user], populations, 0.05, 2.0)
    assert (order.low, order.high) == (0.02, 1.0)
    assert resolution(edges[3], [order], populations, 0.05, 2.0) is None  # already at the kit's limit
    assert resolution(edges[4], [user], populations, 0.05, 2.0) is None
    assert remaining_note(edges[0]).template == REMAINING and remaining_note(edges[4]).template == AT_OTHER


def _kit_result(chi2, at_bound, n=100, **values):
    population = dict(name="pop1", R_nm=values.get("R_nm", 4.5), sigma_R_rel=values.get("sigma_R_rel", 0.2))
    table = [dict(key="pop1.sigma_R_rel", status="at_bound", text="0.6", error=0.1)]
    return dict(summary=dict(chi2_per_point=chi2 / n, chi2=chi2, n_points=n, dof=n - 10), populations=[population],
                table=table), dict(results=[dict(flags=dict(at_bound=at_bound))])


def test_the_fit_widens_what_stops_at_an_edge_and_keeps_only_what_does_not_make_it_worse():
    from src.gimap.features.fitting.domain.fit1d_bounds import Constraint
    from src.gimap.features.fitting.domain.fit1d_edges import KEPT, WIDENED_USER, resolve

    user = Constraint("pop1", "R_nm", "range", 1.0, 4.5, 0)
    asked = []

    def fit(constraints):
        asked.append({item.name: item for item in constraints})
        return _kit_result(110.0, ["pop1.sigma_R_rel@lower"], R_nm=6.0, sigma_R_rel=0.03)

    first = _kit_result(900.0, ["pop1.R_nm@upper", "pop1.sigma_R_rel@lower"], R_nm=4.5, sigma_R_rel=0.03)
    done = resolve(fit, [user], *first, 0.05, 2.0)
    # R widened; σR/R at its physical limit is not widened (``settle`` gives it as a bound).
    assert asked[0]["pop1.R_nm"].high == 9.0 and "pop1.sigma_R_rel" not in asked[0] and len(asked) == 1
    assert [note.template for note in done.actions] == [WIDENED_USER] and done.refits == 1
    assert "your range (4.5 nm): widened to 9 nm" in done.actions[0].text and "9 → 1.1" in done.actions[0].text
    assert [edge.name for edge in done.edges] == ["pop1.sigma_R_rel"]
    worse = resolve(lambda constraints: _kit_result(2000.0, []), [user], *first, 0.05, 2.0)
    assert worse.result is first[0] and [note.template for note in worse.actions] == [KEPT]
    assert worse.constraints == [user]


def test_a_value_that_stays_at_an_edge_is_given_as_the_least_extreme_bound_the_curve_allows():
    from src.gimap.features.fitting.domain.fit1d_edges import (
        AT_LIMIT, BOUNDED, Resolved, classify_edges, mark_bounds, settle)

    first = _kit_result(100.0, ["pop1.sigma_R_rel@upper"], sigma_R_rel=0.6)
    edges = classify_edges(["pop1.sigma_R_rel@upper"], [], first[0]["populations"])

    def flat_above(limit):  # every σR/R above ``limit`` fits as well; below, χ² rises fast
        def fit(constraints):
            value = next(c.low for c in constraints if c.name == "pop1.sigma_R_rel")
            return _kit_result(100.0 + max(0.0, limit - value) * 400, [], sigma_R_rel=value)
        return fit

    done = settle(flat_above(0.5), Resolved(*first, [], [], edges, 0))
    bound = next(c for c in done.constraints if c.name == "pop1.sigma_R_rel")
    assert (bound.kind, bound.source, bound.low) == ("fixed", "bound_upper", pytest.approx(0.5))
    assert [note.template for note in done.actions] == [BOUNDED] and "every value ≥ 0.5 fits" in done.actions[0].text
    assert not done.edges and done.refits == 3  # 0.55, 0.50 within the tolerance; 0.45 not
    mark_bounds(done.result, done.constraints)
    assert done.result["table"][0]["status"] == "one_sided" and done.result["table"][0]["text"] == "≥ 0.5"
    # The curve wants a value beyond the model's own limit: said, not hidden.
    beyond = settle(flat_above(0.9), Resolved(*first, [], [], edges, 0))
    assert [note.template for note in beyond.actions] == [AT_LIMIT] and beyond.result is first[0]


def test_a_value_of_fittings_model_that_stays_at_an_edge_is_fixed_at_its_bound():
    from types import SimpleNamespace

    from src.gimap.features.fitting.application.single_fit import FitModel, new_component
    from src.gimap.features.fitting.domain.fit_edges import BOUNDED, settle

    model = FitModel((new_component("sphere", radius=6.0),)).with_parameter((0, "sigma_R"), value=0.01)

    def result(model, chi2):
        return SimpleNamespace(model=model, chi2_reduced=chi2 / 90, points=100, free=10, stopped=False,
                               at_bounds=((0, "sigma_R"),))

    def fit(changed):  # every σR/R below 0.1 fits as well
        value = changed.get((0, "sigma_R")).value
        return result(changed, 90.0 + max(0.0, value - 0.1) * 400)

    done, notes = settle(fit, result(model, 90.0))
    spread = done.model.get((0, "sigma_R"))
    assert spread.value == pytest.approx(0.06)  # 0.06 fits as well; 0.11 does not
    assert not spread.free and [note.template for note in notes] == [BOUNDED] and "≤" in notes[0].text


def test_a_frame_follows_the_previous_one_inside_a_window_unless_a_free_fit_is_clearly_better():
    from src.gimap.features.fitting.domain.fit1d_bounds import Constraint
    from src.gimap.features.fitting.domain.fit1d_follow import FOLLOWED, JUMPED, follow_note, same_basin, tolerance, window

    previous = [dict(name="pop1", R_nm=3.0, sigma_R_rel=0.1, D_nm=30.0, w_rel=0.2),
                dict(name="pop2", R_nm=10.0, inactive=True)]
    fixed = Constraint("pop1", "D_nm", "fixed", 28.0, 28.0, 1)
    narrow = Constraint("pop1", "R_nm", "range", 2.5, 2.8, 1)
    near = {item.name: item for item in window(previous, [fixed, narrow])}
    assert near["pop1.D_nm"] is fixed  # the user's fixed value stays
    assert (near["pop1.R_nm"].low, near["pop1.R_nm"].high, near["pop1.R_nm"].source) == (2.5, 2.8, "follow")
    assert near["pop1.sigma_R_rel"].low == pytest.approx(0.02) and near["pop1.sigma_R_rel"].high == pytest.approx(0.25)
    assert near["pop1.w_rel"].low == pytest.approx(0.05) and not any(name.startswith("pop2") for name in near)
    assert tolerance(100.0, 200) == 25.0 and tolerance(4000.0, 2000) == pytest.approx(40.0 * 2.0)
    assert same_basin(100.0, 120.0, 200) and not same_basin(100.0, 130.0, 200)
    assert follow_note(100.0, 120.0, 200).template == FOLLOWED and follow_note(100.0, 130.0, 200).template == JUMPED


def test_a_value_of_the_model_at_an_edge_is_widened_then_a_spread_fixed():
    from src.gimap.features.fitting.application.single_fit import FitData, FitModel, evaluate, fit_resolving, new_component

    q = np.linspace(0.08, 2.0, 160)
    truth = FitModel((new_component("sphere", radius=6.0),)).with_parameter((0, "Int"), value=5e4)
    clean = evaluate(truth, q) + 2.0
    data = FitData(q, clean + np.random.default_rng(7).normal(0, 1, q.size) * np.sqrt(clean), np.sqrt(clean))
    start = FitModel((new_component("sphere", radius=4.0),)).with_parameter((0, "R"), lower=2.0, upper=4.5)
    result, actions, remaining = fit_resolving(start, data)
    assert result.model.get((0, "R")).value == pytest.approx(6.0, rel=0.02) and result.chi2_reduced < 1.2
    assert result.model.get((0, "R")).upper == 9.0 and result.model.get((0, "sigma_R")).free  # not fixed: not at a limit
    assert any(note.values["path"] == [0, "R"] and "widened to 9" in note.text for note in actions)
    assert not remaining
    plain, none, _ = fit_resolving(start, data, resolve=False)
    assert plain.model.get((0, "R")).value == pytest.approx(4.5, rel=1e-3) and none == []


@needs_kit
def test_fixed_values_reach_fit1d_and_are_recorded():
    q, intensity, sigma = _example()
    model = _sphere_and_cylinder({(1, "R"): dict(value=3.1, free=False)})
    value = _single(_data(q, intensity, sigma), model)
    native = value["native"]
    sphere = next(p for p in native["populations"] if p["name"] == "pop1")
    assert sphere["shape"] == "sphere" and sphere["R_nm"] == 3.1 and sphere["status"]["R_nm"] == "fixed"
    used = native["fit1d_constraints"]["used"]
    assert (used[0]["name"], used[0]["kind"], used[0]["source"]) == ("pop1.R_nm", "fixed", "model")
    assert native["fit1d_constraints"]["given"] == used[:1]
    assert all(item["source"] in ("widened", "bound_upper", "bound_lower") for item in used[1:])  # done by the fit
    assert native["fit1d_constraints"]["population_order"] == [1, 0]
    assert native["fit1d_bounds"]["measurable"]["R_nm"] == pytest.approx([0.5 / q.max(), 5 / q.min()])
    # The row for Model, in Model's order: the cylinder first.
    from src.gimap.features.fitting.application.single_fit import model_from_solution
    from src.gimap.features.fitting.application.single_fit1d import in_model_order

    mapped, _ = model_from_solution(in_model_order(value["mapping_row"], model), model)
    assert [c.family for c in mapped.components] == ["vertical_cylinder", "sphere"]
    assert mapped.get((1, "R")).value == pytest.approx(3.1) and not mapped.get((1, "R")).free


def _cylinders(radius=3.0, distance=150.0, q_min=0.05, seed=1):
    """Vertical cylinders (R, 8 % spread) in a paracrystal of distance D: D/R = 50 lies beyond fit1d's default 40 R."""
    from scipy.special import j1

    q = np.geomspace(q_min, 2.0, 200)
    radii = np.linspace(0.8, 1.2, 21) * radius
    weights = np.exp(-0.5 * ((radii - radius) / (0.08 * radius)) ** 2) * radii**4
    form = sum(w * (2 * j1(q * r) / (q * r)) ** 2 for w, r in zip(weights, radii)) / weights.sum()
    spread = np.exp(-0.5 * (q * 0.15 * distance) ** 2)
    structure = (1 - spread**2) / (1 - 2 * spread * np.cos(q * distance) + spread**2)
    intensity = np.random.default_rng(seed).poisson(2e4 * form * structure + 20).astype(float)
    return q, intensity, np.sqrt(np.maximum(intensity, 1.0))


@needs_kit
def test_a_default_edge_is_widened_and_fitted_again():
    from src.gimap.features.fitting.application.single_fit import FitModel, new_component

    value = _single(_data(*_cylinders()), FitModel((new_component("vertical_cylinder"),)), fit1d_qz=None)
    native = value["native"]
    bounds = native["fit1d_bounds"]
    assert 1 <= bounds["refits"] <= 2
    assert any(note["values"]["name"] == "pop1 D" and note["values"]["side"] == "upper" for note in bounds["actions"])
    assert native["populations"][0]["D_nm"] == pytest.approx(150.0, rel=0.05)
    assert not [edge for edge in bounds["edges"] if edge["key"] == "D_nm"]
    widened = [item for item in native["fit1d_constraints"]["used"] if item["source"] == "widened"]
    assert widened and widened[0]["name"] == "pop1.D_nm" and widened[0]["high"] > 150.0
    assert any("default range" in note["text"] for note in bounds["actions"])


@needs_kit
def test_auto_composition_ranks_candidates_with_their_own_parameters():
    from src.gimap.features.fitting.application.single_fit import FitModel

    q, intensity, sigma = _example()
    value = _single(_data(q, intensity, sigma), FitModel(), fit1d_composition="auto:sphere,vc:3")
    candidates = value["candidates"]
    assert 2 <= len(candidates) <= 6
    assert [c["rank"] for c in candidates] == list(range(1, len(candidates) + 1))
    assert [c["delta_bic"] for c in candidates] == sorted(c["delta_bic"] for c in candidates)
    assert candidates[0]["composition"] == value["native"]["summary"]["composition"]
    alike = [c for c in candidates if c["indistinguishable"]]
    for candidate in alike:
        assert candidate["indistinguishable_from"] and candidate["composition"] not in candidate["indistinguishable_from"]
    for candidate in candidates:
        assert candidate["table"] and all("status" in row for row in candidate["table"])
        assert candidate["mapping_row"]["components"]
        shapes = {"sphere": "sphere", "vc": "vertical_cylinder"}
        assert [c["type"] for c in candidate["mapping_row"]["components"]] == [
            shapes[s] for s in candidate["composition"].split(",")]
    light = value["native"]["candidates"]
    assert [c["composition"] for c in light] == [c["composition"] for c in candidates]
    assert "table" not in light[0] and value["native"]["fit1d_constraints"]["composition"] == "auto:sphere,vc:3"
