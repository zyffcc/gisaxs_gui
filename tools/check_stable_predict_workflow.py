"""Real-process and offscreen-window release checks for the stable 1D workflow.

Run from the repository root with the supported GUI Python environment. The
saved detector geometry becomes an in-memory Analyze instrument profile (the
settings file is never edited); the UI replay goes Analyze → Send to Fitting →
1D Predict, like a user.
This is an integration replay of the current experimental CBF, not evidence of
generalization to unseen samples or uniquely correct particle parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_FONT_DPI", "96")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
HANDLER = "src.gimap.features.fitting.infrastructure.adapters.workflow_v5:run_workflow_job"
CBF = ROOT / "TestSAXSdata/jg_gisaxs_4nm_old_3ml_insitu_ds03_00001_00033.cbf"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def runtime_provenance():
    paths = (
        "src/gimap/features/fitting/infrastructure/adapters/stable_blue.py",
        "src/gimap/features/fitting/domain/blue_rc_forward.py",
        "modules/Fitting_1D_Model/Workflow_v5/stable_blue_rc_v1/MANIFEST.json",
    )
    manifest = json.loads((ROOT / paths[-1]).read_text(encoding="utf-8"))
    return dict(
        model_id=manifest["model_id"],
        forward_version=manifest["forward_version"],
        inference_backend=manifest["inference_backend"],
        source_sha256={
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths
        },
    )


def canonical_payload(output):
    source = ROOT / "validation/cause_audit_20260921/inputs.json"
    inputs = json.loads(source.read_text(encoding="utf-8"))
    arrays = {key: [] for key in ("q", "intensity", "sigma", "count")}
    for side, sign in (("positive", 1), ("negative", -1)):
        item = inputs[side]
        arrays["q"].extend((sign * np.asarray(item["q"])).tolist())
        for target, key in (("intensity", "y"), ("sigma", "sigma"), ("count", "count")):
            arrays[target].extend(item[key])
    return dict(
        output_dir=str(output),
        q=arrays["q"],
        intensity=arrays["intensity"],
        sigma=arrays["sigma"],
        sigma_estimated=True,
        observation_metadata=dict(
            source="native_cbf_columns",
            valid_pixel_counts=arrays["count"],
            intensity_unit="counts_per_pixel",
            threshold_enabled=False,
            mirror_replaced_pixels=0,
            stack_count=1,
            fixture=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
        options=dict(method="stable", components=[2], render_points=500, side="both"),
    )


def check_rows(rows, *, payload=None):
    """Verify physical reconstruction, scores and the observed/display distinction."""
    from src.gimap.features.fitting.infrastructure.adapters.stable_blue import forward_row

    assert rows, "No candidates returned"
    scores = []
    for row in rows:
        q, observed, sigma, predicted = (
            np.asarray(row[key], float) for key in ("native_q", "observed", "sigma", "native_fit")
        )
        assert q.shape == observed.shape == sigma.shape == predicted.shape
        assert len(q) >= 8 and np.isfinite(predicted).all() and (predicted > 0).all()
        np.testing.assert_allclose(forward_row(q, row), predicted, rtol=2e-7, atol=1e-8)
        np.testing.assert_allclose(
            forward_row(row["display_q"], row), row["display_fit"], rtol=2e-7, atol=1e-8
        )
        assert len(row["display_q"]) == 500
        keep = observed > 0
        score = float(np.sqrt(np.mean(np.log(predicted[keep] / observed[keep]) ** 2)))
        assert abs(score - row["best_log_rmse"]) < 1e-8
        assert row["probability_status"] == "Not calibrated"
        assert row["best_source"] in (
            "stable_neural",
            "stable_amplitude_calibrated",
            "stable_numerical_fallback",
        ), row["best_source"]
        if payload is not None:
            sign = 1 if row["side"] == "positive" else -1
            input_q = np.asarray(payload["q"])
            indices = np.flatnonzero(input_q * sign > 0)
            indices = indices[np.argsort(abs(input_q[indices]))]
            np.testing.assert_array_equal(q, input_q[indices])
            np.testing.assert_array_equal(observed, np.asarray(payload["intensity"])[indices])
            np.testing.assert_array_equal(sigma, np.asarray(payload["sigma"])[indices])
        if row["rank"] == 1:
            scores.append(
                dict(
                    side=row["side"],
                    source=row["best_source"],
                    logrmse=score,
                    native_points=len(q),
                    display_points=len(row["display_q"]),
                    combination=row["combination"],
                )
            )
    return scores


def check_gui_export(fitting, out, stage):
    """Exercise the ordinary Export path with deterministic dialog selections."""
    from PyQt5.QtWidgets import QDialog
    from src.gimap.features.fitting.presentation.bindings import fit_result_export
    from src.gimap.features.fitting.presentation.export_dialog import FittingExportSelection

    target = out / f"native_fit_{stage}_{time.time_ns()}.txt"
    original_dialog = fit_result_export.FittingDataExportDialog
    original_save = fit_result_export.QFileDialog.getSaveFileName

    class SelectedExport:
        def __init__(self, *_args):
            pass

        def exec_(self):
            return QDialog.Accepted

        def selection(self):
            return FittingExportSelection("Fitting Data", "raw")

    try:
        fit_result_export.FittingDataExportDialog = SelectedExport
        fit_result_export.QFileDialog.getSaveFileName = lambda *_a, **_k: (str(target), "")
        fitting._export_fitting_data()
    finally:
        fit_result_export.FittingDataExportDialog = original_dialog
        fit_result_export.QFileDialog.getSaveFileName = original_save
    text = target.read_text(encoding="utf-8")
    assert "# Parameter Source: native_v5_candidate_snapshot" in text
    assert "Fitting parameter export error" not in text
    side = None
    exported = {}
    for line in text.splitlines():
        if line.startswith("# Side: "):
            side = line.split(": ", 1)[1]
            exported[side] = {}
        elif side is not None and line.startswith("#   ") and " = " in line:
            key, value = line[4:].split(" = ", 1)
            if value not in ("True", "False"):
                exported[side][key] = float(value)
    meta = fitting.fitting["meta"]
    candidates = meta.get("side_candidates") or [meta["candidate"]]
    assert set(exported) == {row["side"] for row in candidates}
    for row in candidates:
        assert f"# Forward Version: {row['forward_version']}" in text
        expected = dict(row["global_params"])
        for index, component in enumerate(row["components"], 1):
            values = {
                **component["params"],
                **{key: component[key] for key in ("amplitude", "weight", "type_id")},
            }
            expected.update({f"component_{index}_{key}": value for key, value in values.items()})
        for key, value in expected.items():
            if value is not None:
                assert exported[row["side"]][key] == float(value), (row["side"], key)
    numeric_lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    actual = np.loadtxt(numeric_lines[1:])
    expected_q = fitting._convert_q_values_for_display(
        fitting.fitting["q"], source=meta.get("data_source")
    )
    np.testing.assert_allclose(actual[:, 0], expected_q, rtol=1e-6, atol=1e-10)
    np.testing.assert_allclose(actual[:, 1], fitting.fitting["I"], rtol=1e-6, atol=1e-10)
    return dict(path=str(target), sides=sorted(exported), candidate_parameters_preserved=True)


def run_worker_checks(out):
    from src.gimap.app.jobs import JobRequest
    from src.gimap.integrations.jobs import LocalProcessJobRunner

    out.mkdir(parents=True, exist_ok=True)
    payload = canonical_payload(out / "native_cbf")
    runner = LocalProcessJobRunner()

    def execute(request_payload, *, cancel=False):
        request = JobRequest(handler=HANDLER, payload=request_payload, timeout_seconds=900)

        def progress(value):
            print(value.message, flush=True)
            if cancel:
                runner.cancel(request.job_id)

        result = runner.run(request, on_progress=progress)
        if cancel:
            assert result.status == "cancelled", result
            return dict(status=result.status, seconds=result.elapsed_seconds)
        if not result.succeeded:
            raise RuntimeError(result.error)
        return result.value

    native = execute(payload)
    native_scores = check_rows(native["candidates"], payload=payload)
    assert {row["side"] for row in native_scores} == {"positive", "negative"}
    # Generous replay guard against routing back to the known poor frozen network;
    # this is not an acceptance cutoff for arbitrary experimental measurements.
    assert all(row["logrmse"] < 0.23 for row in native_scores), native_scores

    text_file = out / "current_cut_without_count_metadata.csv"
    np.savetxt(
        text_file,
        np.column_stack([payload[key] for key in ("q", "intensity", "sigma")]),
        delimiter=",",
        header="q_nm^-1,intensity,sigma",
    )
    batch_payload = dict(
        files=[str(out / "missing_curve.txt"), str(text_file)],
        output_dir=str(out / "text_batch"),
        options=dict(method="stable", render_points=500, components=[2]),
    )
    batch = execute(batch_payload)
    assert [record["status"] for record in batch["records"]] == ["failed", "complete"]
    batch_scores = check_rows(batch["candidates"])
    assert all(row["best_source"] == "stable_numerical_fallback" for row in batch["candidates"])
    cancellation = execute(
        {**batch_payload, "files": [str(text_file)], "output_dir": str(out / "cancelled")},
        cancel=True,
    )
    evidence = dict(
        passed=True,
        native=native_scores,
        text_batch=batch_scores,
        text_without_count_uses_fallback=True,
        batch_failure_isolation=True,
        cancellation=cancellation,
        native_seconds=native["summary"]["runtime_seconds"],
        text_batch_seconds=batch["summary"]["runtime_seconds"],
    )
    write_json(out / "VERIFIED.json", evidence)
    return evidence


def validated_band_rows():
    """Detector rows (first, last) of the band the validated fixture averaged."""
    source = ROOT / "validation/cause_audit_20260921/provenance.json"
    r0, r1, _x0, _x1 = json.loads(source.read_text(encoding="utf-8"))["mask"]["pixel_region"]
    return int(r0), int(r1)


def validated_q_max_angstrom(curve_q):
    """|q| limit (Å⁻¹) that keeps exactly the columns of the learned branch's validated profile.

    The profile is the canonical fixture (the former saved 1180-px cut, about
    4.23 nm⁻¹).  Analyze cuts the full detector width; the columns nearest the
    beam centre are the fixture's, so the limit lies between the last kept and
    the first dropped |q| (the user sets the Fitting range the same way).
    """
    source = ROOT / "validation/cause_audit_20260921/inputs.json"
    inputs = json.loads(source.read_text(encoding="utf-8"))
    columns = sum(len(inputs[side]["q"]) for side in ("positive", "negative"))
    magnitudes = np.sort(np.abs(np.asarray(curve_q, float)))
    magnitudes = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)]
    return float(0.5 * (magnitudes[columns - 1] + magnitudes[columns]))


def fixture_equivalence(request):
    """Analyze's observations against the validated fixture of the former detector path.

    The same detector columns must give the same intensities and pixel counts; q may
    move by less than one column (exact q model, symmetry-refined centre).
    """
    source = ROOT / "validation/cause_audit_20260921/inputs.json"
    inputs = json.loads(source.read_text(encoding="utf-8"))
    old_q = np.r_[-np.asarray(inputs["negative"]["q"]), np.asarray(inputs["positive"]["q"])]
    old_y = np.r_[inputs["negative"]["y"], inputs["positive"]["y"]]
    old_count = np.r_[inputs["negative"]["count"], inputs["positive"]["count"]]
    new_q = np.asarray(request["q"], float)
    new_y = np.asarray(request["intensity"], float)
    new_count = np.asarray(request["observation_metadata"]["valid_pixel_counts"], float)
    assert new_q.shape == old_q.shape, (new_q.shape, old_q.shape)
    old_order, new_order = np.argsort(old_q), np.argsort(new_q)
    np.testing.assert_allclose(new_y[new_order], old_y[old_order], rtol=1e-6, atol=1e-9)
    np.testing.assert_array_equal(new_count[new_order], old_count[old_order])
    spacing = float(np.median(np.diff(np.sort(old_q[old_q > 0]))))
    shift = float(np.max(np.abs(new_q[new_order] - old_q[old_order])) / spacing)
    assert shift < 1.0, shift
    return dict(
        columns=int(new_q.size),
        same_intensities=True,
        same_pixel_counts=True,
        max_q_shift_columns=shift,
    )


def saved_geometry_profile(saved):
    """The saved (former Cut & Fitting) geometry as the Analyze profile of the Pilatus frames."""
    from src.gimap.features.analyze.application import geometry_from_fitting_settings
    from src.gimap.shared.geometry import InstrumentProfile

    values = {("beam", key): value for key, value in saved["beam"].items()}
    for section in ("detector", "gisaxs_input"):
        for key, value in saved["fitting"].get(section, {}).items():
            values[("fitting", f"{section}.{key}")] = value
    geometry = geometry_from_fitting_settings(
        lambda section, key, default=None: values.get((section, key), default), 1679
    )
    assert geometry is not None, "The saved geometry is incomplete"
    return InstrumentProfile(
        "PILATUS 2M 1679×1475", geometry, "PILATUS 2M", (1679, 1475), source="saved geometry"
    )


def run_ui_checks(out, *, insitu=False):
    """Analyze opens the CBF, refines x by symmetry and sends the cut; Fitting predicts."""
    import shutil

    from PyQt5.QtGui import QFont, QFontDatabase
    from PyQt5.QtWidgets import QApplication, QMessageBox
    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )

    out.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    if not QFontDatabase().families():
        for name in ("arial.ttf", "segoeui.ttf", "msyh.ttc"):
            QFontDatabase.addApplicationFont(
                str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / name)
            )
        app.setFont(QFont("Segoe UI", 9))
    settings_path = ROOT / "config/user_parameters.json"
    saved_bytes = settings_path.read_bytes()
    saved = json.loads(saved_bytes)
    profile = saved_geometry_profile(saved)
    context = AppContext(
        settings=InMemorySettingsRepository({"beam": saved["beam"]}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=LocalProcessJobRunner(),
        instrument_profiles=InMemoryInstrumentProfileRepository([profile]),
    )
    # Analyze writes its curves next to the frame: work on a copy, never in TestSAXSdata.
    frames = out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    frame = frames / CBF.name
    shutil.copy2(CBF, frame)
    window = MainWindow(context)
    window.resize(1600, 1050)
    window.show()

    def until(check, timeout=90):
        end = time.monotonic() + timeout
        while not check():
            app.processEvents()
            time.sleep(0.02)
            if time.monotonic() > end:
                raise TimeoutError("Stable workflow UI smoke timeout")

    def fail(_parent, title, message, *_args):
        raise RuntimeError(f"{title}: {message}")

    old_warning, old_critical = QMessageBox.warning, QMessageBox.critical
    QMessageBox.warning = QMessageBox.critical = fail
    try:
        until(lambda: window._initialization_completed and hasattr(window, "runtime"))
        analyze = window.components.analyze_page
        analyze.add_paths([frame])
        assert analyze.tasks.wait(90)
        analysis = analyze.view_model.state.analysis
        assert analysis is not None and analysis.reduction is not None, analyze.status_text()
        assert analysis.kind == "gisaxs", f"Auto mode classified the frame as {analysis.kind}"
        initial_x = analysis.geometry.beam_center_x_px
        yoneda_band = analysis.reduction.markers["horizontal_band"]
        # Drag the horizontal band onto the rows of the validated profile (the former saved cut).
        first, last = validated_band_rows()
        analyze.detector_view.horizontalBandChanged.emit(float(first), float(last + 1))
        assert analyze.tasks.wait(90)
        band = analyze.view_model.state.analysis.reduction.curve("horizontal").region["rows"]
        assert tuple(band) == (first, last + 1), band
        analyze.refine_center_x()
        assert analyze.tasks.wait(90)
        refined_x = analyze.view_model.state.analysis.geometry.beam_center_x_px
        analyze.fit_button.click()
        fitting = window.runtime.fitting
        until(lambda: fitting._initialized and fitting.current_1d_data is not None)
        messages = []
        fitting.status_updated.connect(messages.append)
        # Limit the fitting range (|q| in the default folded view) to the validated profile.
        limit = validated_q_max_angstrom(fitting.current_1d_data["q"])
        window.fitFittingRegionMaxValue.setValue(fitting._roi_data_to_control_range(0.0, limit)[1])
        window.fitFittingRegionMaxValue.editingFinished.emit()
        app.processEvents()
        # The control shows the plot's unit (nm⁻¹); q_max is the data limit in Å⁻¹ as it rounds.
        q_max = fitting._roi_control_to_data_values(0.0, window.fitFittingRegionMaxValue.value())[1]
        assert abs(fitting._roi_max - q_max) < 1e-9, (fitting._roi_min, fitting._roi_max)
        assert window.fitFittingRegionMaxValue.suffix().strip() == "nm⁻¹"
        assert fitting._workflow_options()["method"] == "model"
        # The specialist requires an explicit composition prior; never select it
        # for unknown samples merely because they have the same detector grid.
        fitting._save_workflow_options(
            {**fitting._workflow_options(), "method": "stable", "components": [2]}
        )
        q, y, sigma = fitting._current_ai_curve_arrays()
        metadata = fitting._workflow_observation_metadata
        assert metadata["source"] == "native_cbf_columns", metadata.get("source")
        count = np.asarray(metadata["valid_pixel_counts"])
        assert count.shape == q.shape and np.all(count > 0)
        assert len(q) > 1000 and len(q) != 500
        assert metadata["input_selection"]["roi_abs"]
        assert np.any(q < 0) and np.any(q > 0)
        input_payload = dict(q=q.tolist(), intensity=y.tolist(), sigma=sigma.tolist())
        workspace = window.components.fitting_workspace
        workspace.show_fit_curve()
        window.fittingModeTabs.setCurrentIndex(3)
        tick = time.perf_counter()
        window.aiFittingFullAutoFitButton.click()
        until(
            lambda: (
                getattr(fitting, "_workflow_v5_dialog", None) is not None
                and fitting._ai_job_thread is None
            ),
            timeout=900,
        )
        dialog = fitting._workflow_v5_dialog
        scores = check_rows(dialog.rows, payload=input_payload)
        assert {row["side"] for row in scores} == {"positive", "negative"}
        request = json.loads((dialog.output_dir / "request.json").read_text(encoding="utf-8"))
        assert request["options"]["method"] == "stable"
        np.testing.assert_array_equal(request["observation_metadata"]["valid_pixel_counts"], count)
        equivalence = fixture_equivalence(request)
        app.processEvents()
        dialog.grab().save(str(out / "stable_prediction.png"))
        evidence = dict(
            passed=True,
            button="aiFittingFullAutoFitButton",
            method="stable",
            curve=str(fitting.current_1d_data["file_path"]),
            beam_center_x_px=dict(profile=initial_x, symmetry=refined_x),
            horizontal_band_rows=dict(yoneda=list(yoneda_band), used=[first, last + 1]),
            fitting_range_abs_q_max_angstrom=q_max,
            fixture_equivalence=equivalence,
            native_points=dict(
                positive=int(np.count_nonzero(q > 0)), negative=int(np.count_nonzero(q < 0))
            ),
            results=scores,
            seconds=time.perf_counter() - tick,
            fit_output=str(dialog.output_dir),
            count_metadata_preserved=True,
            saved_geometry_sha256=hashlib.sha256(saved_bytes).hexdigest(),
        )
        evidence["export"] = check_gui_export(fitting, out, "single")
        dialog.close()
        if insitu:
            from src.gimap.features.fitting.application.insitu_records import ManageInSituRecords
            from src.gimap.features.fitting.infrastructure.adapters.local_insitu_records import (
                LocalInSituRecordRepository,
            )

            class TestRecords(LocalInSituRecordRepository):
                def cache_directory(self):
                    return out / "cache"

            fitting.fitting_view_model.storage._insitu_records = ManageInSituRecords(TestRecords())
            curve = Path(fitting.current_1d_data["file_path"])
            workspace.show_context("insitu")
            page = window.fittingInsituSeriesPage
            page.ui.captureRecipeButton.click()
            recipe = fitting.fitting_view_model.insitu.recipe
            assert recipe is not None and recipe.model["workflow_v5"]["method"] == "stable"
            assert recipe.cut["source"] == "curve"
            widgets = page.workflow_widgets()
            widgets["run_mode"].setCurrentText("Process Existing Sequence")
            widgets["sequence_folder"].setText(str(curve.parent))
            widgets["sequence_pattern"].setText(curve.name)
            widgets["recursive"].setChecked(False)
            tick = time.perf_counter()
            widgets["process"].click()
            until(
                lambda: (
                    len(fitting._insitu_workflow_results) > 0
                    and not fitting._insitu_workflow_busy
                    and fitting._insitu_workflow_state == "Idle"
                ),
                timeout=900,
            )
            records = fitting._insitu_workflow_results
            assert len(records) == 1 and records[0]["fit_status"] == "ok", records
            insitu_rows = records[0]["v5_side_candidates"]
            assert len(insitu_rows) == 2
            evidence["insitu"] = dict(
                seconds=time.perf_counter() - tick,
                results=check_rows(insitu_rows, payload=input_payload),
                recipe=recipe.to_dict(),
                fit_output=records[0]["v5_output_dir"],
            )
            insitu_request = json.loads(
                (Path(records[0]["v5_output_dir"]) / "request.json").read_text(encoding="utf-8")
            )
            np.testing.assert_array_equal(
                insitu_request["observation_metadata"]["valid_pixel_counts"], count
            )
            evidence["insitu"]["same_native_observations_as_single"] = True
            for row in insitu_rows:
                side = row["side"]
                assert records[0]["v5_forward_versions"][side] == row["forward_version"]
                assert records[0]["v5_candidate_sources"][side] == row["best_source"]
                assert records[0]["v5_unit_contracts"][side] == row["unit_contract"]
                flat = json.loads(records[0]["fitted_parameters"])
                for index, component in enumerate(row["components"], 1):
                    for key in ("amplitude", "weight", "type_id"):
                        assert flat[f"{side}_component_{index}_{key}"] == component[key]
                for key, value in row["global_params"].items():
                    assert flat[f"{side}_{key}"] == value
            evidence["insitu"]["complete_scalar_parameters_persisted"] = True
            evidence["insitu"]["export"] = check_gui_export(fitting, out, "insitu")
            write_json(out / "completed_insitu_records.json", records)
            page.grab().save(str(out / "stable_insitu.png"))
            # Stop a newly queued sequence before its first load starts, ensuring
            # recipe/runtime state returns cleanly without touching the saved curve.
            widgets["process"].click()
            widgets["stop"].click()
            until(
                lambda: (
                    fitting._insitu_workflow_state == "Idle" and not fitting._insitu_workflow_busy
                )
            )
            assert not fitting._insitu_workflow_queue
            evidence["insitu"]["queued_sequence_cancelled"] = True
        (out / "messages.txt").write_text("\n".join(messages), encoding="utf-8")
        write_json(out / "VERIFIED.json", evidence)
        return evidence
    finally:
        window.close()
        app.processEvents()
        QMessageBox.warning, QMessageBox.critical = old_warning, old_critical
        assert settings_path.read_bytes() == saved_bytes, "Saved user settings were modified"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker", "ui", "insitu", "all"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/stable_release_20260922")
    args = parser.parse_args()
    results = {}
    if args.mode in ("worker", "all"):
        results["worker"] = run_worker_checks(args.output / "worker")
    if args.mode in ("ui", "insitu", "all"):
        results["ui"] = run_ui_checks(args.output / "ui", insitu=args.mode in ("insitu", "all"))
    write_json(
        args.output / "VERIFIED.json",
        dict(passed=True, runtime=runtime_provenance(), checks=results),
    )
    print(json.dumps(dict(passed=True, mode=args.mode, output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
