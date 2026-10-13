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
