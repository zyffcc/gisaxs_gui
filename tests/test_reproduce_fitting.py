"""Round trips through the Fitting page's CSV and JSON export and the headless reproducer."""

import json
from dataclasses import replace

import numpy as np
import pytest

from src.gimap.app.reproduce import reproduce_record
from tests.test_fit_page import _curve_file, _page, _sphere, _wait
from tests.test_fit_page_predict import NETWORK, _sphere_file
from tests.repeatability import assert_replayed


def _curve_csv(path):
    with path.open(encoding="utf-8") as stream:
        return np.loadtxt((line for line in stream if not line.startswith("#")), delimiter=",", skiprows=1)


def test_manual_fit_restarts_from_recorded_model_and_budget(tmp_path):
    page = _page()
    try:
        source = _curve_file(tmp_path, signed=True)
        assert page.open_curve(source)
        page.set_model(_sphere(5.1))
        page.budget_spin.setValue(96)
        page.run_fit("local")
        _wait(page)
        assert page.session.result is not None
        path = tmp_path / "manual.csv"
        page.export_data_dialog(path)
        record_path = path.with_suffix(".json")
        recipe = json.loads(record_path.read_text(encoding="utf-8"))["reproduction"]
        assert recipe["start_model"]["components"][0]["params"]["R"]["value"] == 5.1
        assert recipe["max_evaluations"] == 96 and recipe["source"]["sha256"]
        report = reproduce_record(record_path, tmp_path / "replayed")
        assert report["status"] == "identical", report
        assert {"fit_curve", "parameters", "covariance"} <= report["tables"].keys()
        curve_file = tmp_path / "replayed" / report["tables"]["fit_curve"]["file"]
        assert _curve_csv(curve_file).tobytes() == _curve_csv(path).tobytes()

        # A changed starting constraint must produce a different fitted radius and curve.
        changed = json.loads(record_path.read_text(encoding="utf-8"))
        changed["reproduction"]["start_model"]["components"][0]["params"]["R"]["free"] = False
        edited = tmp_path / "edited.json"
        edited.write_text(json.dumps(changed), encoding="utf-8")
        different = reproduce_record(edited, tmp_path / "rerun")
        assert different["tables"]["parameters"]["columns"]["0_R"]["status"] == "different"
        source.write_text("different bytes", encoding="ascii")
        assert page.project_state()["results"]["curve_sha256"] == recipe["source"]["sha256"]
        with pytest.raises(ValueError, match="SHA-256 mismatch.*" + source.name):
            reproduce_record(record_path, tmp_path / "changed")
        source.unlink()
        with pytest.raises(ValueError, match="Missing referenced file.*" + source.name):
            reproduce_record(record_path, tmp_path / "missing")
    finally:
        page.dispose()


def test_predict_repeats_the_posterior_and_selected_curve(tmp_path, monkeypatch):
    from src.gimap.features.fitting.presentation.single import predict

    monkeypatch.setenv("GIMAP_ML1D_NETWORK", str(NETWORK))
    page = _page()
    try:
        source = _sphere_file(tmp_path)
        assert page.open_curve(source)
        page.set_model(_sphere(4.7))

        original = predict.predict_request
        def small_request(*args):
            request = original(*args, seed=253)
            return replace(request, payload=dict(request.payload, samples=120, local_samples=64, refine_modes=1, explored=1))

        monkeypatch.setattr(predict, "predict_request", small_request)
        page.run_fit("predict")
        _wait(page)
        assert page.session.solutions, page.status_label.text()
        path = tmp_path / "posterior.csv"
        page.export_data_dialog(path)
        record_path = path.with_suffix(".json")
        recipe = json.loads(record_path.read_text(encoding="utf-8"))["reproduction"]
        plan = recipe["prediction"]
        assert recipe["operation"] == "selected_solution"
        assert plan["options"]["seed"] == 253 and plan["options"]["samples"] == 120
        assert plan["weights"]["sha256"] and plan["network_record"]["sha256"]
        report = reproduce_record(record_path, tmp_path / "replayed")
        assert report["status"] == "identical", report
        curve_file = tmp_path / "replayed" / report["tables"]["fit_curve"]["file"]
        assert _curve_csv(curve_file).tobytes() == _curve_csv(path).tobytes()
        assert report["tables"]["solution_1_parameters"]["columns"]["R_1_low68"]["status"] == "identical"
        assert np.isfinite(page.session.solutions[0].chi2)
    finally:
        page.dispose()


def test_refine_replays_accepted_restarts_from_the_original_model(tmp_path):
    from src.gimap.features.fitting.application.fit_state import result_from_dict, result_to_dict
    from src.gimap.features.fitting.application.single_fit import fit_record, fit_resolving
    from tests.test_refine_restarts import _data, _model

    data, start = _data(), _model(7.)
    result, _, _ = fit_resolving(start, data, max_evaluations=12)
    assert result.restarts > 0
    restored = result_from_dict(result_to_dict(result))
    record = fit_record(restored.model, data, None, result=restored, side="mean", q_range=None)
    recipe = record["reproduction"]
    assert recipe["start_model"]["components"][0]["params"]["R"]["value"] == 7.
    assert recipe["restart_budget"] == 12 and recipe["restarts"] == result.restarts
    path = tmp_path / "refine.json"
    path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    report = reproduce_record(path, tmp_path / "replayed")
    assert report["status"] == "identical", report


def test_joint_fit_repeats_shared_parameter_and_independent_scales(tmp_path):
    from src.gimap.features.fitting.application.joint_fit import CurveSpec, joint_record, run_joint_fit
    from src.gimap.features.fitting.application.single_fit import evaluate

    start = _sphere(5.2)
    q = np.linspace(.05, 1.8, 48)
    intensity = evaluate(_sphere(5.6), q)
    specs = [CurveSpec("first", q, intensity), CurveSpec("second", q, 2 * intensity)]
    result = run_joint_fit(specs, start, None, {(0, "R")})
    path = tmp_path / "joint.json"
    path.write_text(json.dumps(joint_record(result, specs, None), allow_nan=False), encoding="utf-8")
    report = reproduce_record(path, tmp_path / "replayed")
    assert report["status"] == "identical", report
    assert {"curve_1_fit_curve", "curve_2_fit_curve", "covariance", "joint_fit"} <= report["tables"].keys()


def test_joint_fit_replays_file_widths_and_vertical_geometry(tmp_path):
    from src.gimap.features.fitting.application.joint_fit import (
        CurveSpec, curve_model, joint_record, run_joint_fit, substrate_of,
    )
    from src.gimap.features.fitting.application.single_fit import evaluate

    q = np.linspace(.5, 1.5, 32)
    dq = np.linspace(.002, .035, q.size)
    base = replace(_sphere(5.2), resolution_mode="file", weighting="poisson")
    for path, parameter in base.parameters():
        if parameter.free and path != (0, "Int"):
            base = base.with_parameter(path, free=False)
    substrate = substrate_of("Si")
    horizontal = CurveSpec("first", q, np.ones_like(q), dq=dq)
    vertical = CurveSpec("second", q, np.ones_like(q), kind="vertical", alpha_i=.2, wavelength=.1, q_parallel=.3)
    # A 2 % ripple keeps the residuals, χ² and errors above the rounding level of the model (``assert_replayed``).
    ripple = 1 + 0.02 * np.sin(40 * q)
    specs = [replace(spec, intensity=ripple * evaluate(curve_model(spec, base, substrate, True), q, dq=spec.dq))
             for spec in (horizontal, vertical)]
    result = run_joint_fit(specs, base, substrate, set())
    record = joint_record(result, specs, substrate)
    first, second = record["reproduction"]["curves"]
    assert first["points"]["dq"] == dq.tolist() and first["points"]["weight_mode"] == "poisson"
    assert second["start_model"]["vertical_cut"]["q_parallel"] == .3
    path = tmp_path / "joint.json"
    path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
    report = reproduce_record(path, tmp_path / "replayed")
    assert_replayed(report)
    assert report["tables"]["curve_1_fit_curve"]["columns"]["dQ_nm^-1"]["status"] == "identical"


def test_1d_predict_summary_replays_its_verified_side_record(tmp_path, monkeypatch):
    from src.gimap.features.fitting.application.workflow_v5 import default_options
    from src.gimap.features.fitting.infrastructure.adapters import posterior_job, workflow_v5

    monkeypatch.setenv("GIMAP_ML1D_NETWORK", str(NETWORK))
    defaults = posterior_job.predict_options
    monkeypatch.setattr(posterior_job, "predict_options", lambda values: {
        **defaults(values), "samples": 80, "local_samples": 64, "refine_modes": 1, "explored": 1})
    folder = tmp_path / "batch"
    options = dict(default_options(), method="posterior", components=[1], side="positive", q_unit="A^-1")
    workflow_v5.run_workflow_job(dict(files=[str(_sphere_file(tmp_path))], output_dir=str(folder), options=options),
                                 lambda *args: None, lambda: False)
    record = folder / "fitting_run_summary.json"
    report = reproduce_record(record, tmp_path / "replayed")
    assert report["status"] == "identical"
    assert "display_curves" in report["results"][0]["tables"]
