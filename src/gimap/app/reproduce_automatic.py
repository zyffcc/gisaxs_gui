"""Replay an automatic report on an isolated offscreen Analyze page and compare structural results."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import CancelledError
from pathlib import Path
import subprocess
import sys
import tempfile

from src.gimap.shared.reproduction import unpack_column, verified_path


def _map_paths(value, paths):
    if isinstance(value, dict):
        return {paths.get(key, key): _map_paths(item, paths) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_paths(item, paths) for item in value]
    return paths.get(value, value) if isinstance(value, str) else value


def _prepare(recipe, record_dir):
    paths = {}
    for name, ref in recipe["inputs"].items():
        path = str(verified_path(ref, record_dir).resolve())
        paths[name] = paths[ref["path"]] = path
    if plan := recipe.get("predict"):
        from src.gimap.features.fitting.domain.posterior import network

        for key, installed in (("weights", network.path()), ("network_record", network.path().with_suffix(".json"))):
            ref = plan[key]
            paths[ref["path"]] = str(verified_path(ref, record_dir, installed=installed).resolve())
        weights = Path(paths[plan["weights"]["path"]])
        if Path(paths[plan["network_record"]["path"]]) != weights.with_suffix(".json"):
            raise ValueError(f"Expected the network record beside its weights: {weights.with_suffix('.json')}")

    def require(path):
        if path and path not in paths:
            raise ValueError(f"No input file digest recorded for {path}.")

    require(recipe["frame"])
    for path in recipe["summed_files"]:
        require(path)
    for phase in recipe["phases"]:
        if phase["cif"]:
            require(phase["cif"])
            expected = recipe["inputs"][phase["cif"]]["sha256"]
            if phase["sha256"] != expected:
                raise ValueError(f"SHA-256 mismatch for {phase['cif']}: expected {phase['sha256']}, got {expected}")
    if not recipe["options"].get("geometry"):
        require(recipe["calibration_file"])
    corrections = recipe["settings"]["corrections"]
    for key in ("dark_path", "background_path", "flat_field_path", "mask_path"):
        require(corrections.get(key))
    if corrections.get("error_model") == "sigma":
        require(corrections.get("sigma_path"))
    if recipe.get("procedure", recipe["options"].get("technique")) == "giwaxs":
        require(corrections.get("film_cif_path"))
    for edit in corrections.get("mask_shapes") or ():
        require(edit.get("path"))
        for path in edit.get("sources", ()):
            require(path)
    ready = _map_paths(recipe, paths)
    profile = ready["settings"].get("profile")
    if profile:
        from src.gimap.features.analyze.application.geometry_files import LOADED_SOURCE, SIBLING_SOURCE

        for template in (LOADED_SOURCE, SIBLING_SOURCE):
            prefix = template.partition("{path}")[0]
            source = profile.get("source", "")
            if source.startswith(prefix) and source[len(prefix):] in paths:
                profile["source"] = template.format(path=paths[source[len(prefix):]])
    return ready, paths


def _run_worker(recipe, folder):
    from src.gimap.app.headless_assistant import LOAD_TIMEOUT_S, open_headless_session
    from src.gimap.app.headless_record import run_record
    from src.gimap.app.qt_plugins import ensure_qt_platform_plugin_path
    from src.gimap.features.analyze.presentation.automation_state import restore_automatic_state
    from src.gimap.features.assistant.application.pipeline import PipelineOptions, StandardPipeline
    from src.gimap.features.fitting.bootstrap import create_curve_fit
    from src.gimap.features.fitting.domain.posterior import network

    ensure_qt_platform_plugin_path()
    method = recipe.get("predict_method") or {}
    if plan := recipe.get("predict"):
        os.environ[network.VARIABLE] = plan["weights"]["path"]
        verified_path(plan["network_record"], folder)
    if expected := (method.get("network") or {}).get("sha256"):
        where = network.path().resolve()
        verified_path({"path": str(where), "sha256": expected}, where.parent)
    session = open_headless_session(recipe["frame"], also=tuple(recipe["summed_files"]),
                                    saved_profiles=False, out_dir=folder)
    try:
        page = session.page
        if plan:
            session.catalog.fitter = create_curve_fit(predict_settings=plan["options"])
        restore_automatic_state(page, recipe)
        page.tasks.wait(LOAD_TIMEOUT_S)

        page.indexing.set_settings(**recipe["indexing"])
        phases = recipe["phases"]
        page.automation().set_phases(phases)
        for index, phase in enumerate(phases):
            # A CIF cell or phase name can have been edited in Analyze after import.
            page.indexing.replace(index, cell=tuple(phase["cell"]), name=phase["name"], centring=phase["centring"])
        options = dict(recipe["options"], calibration=recipe["calibration_file"], phases=())
        # Geometry is selected from the recorded calibration, the reused batch geometry or the saved profile.
        # Notes still supply quantities and substrate composition, but cannot introduce a new calibration.
        if recipe["calibration_file"]:
            session.catalog._grant(recipe["calibration_file"], "file")
        pipe = StandardPipeline(session.catalog, PipelineOptions(**options), recorded_calibration=True)
        report = pipe.run()
        if not report.get("ok"):
            reason = report.get("failed") or " ".join(item["why"] for item in report.get("needs_attention", ()))
            raise ValueError(f"Automatic analysis stopped during replay: {reason}")
        report.update(run_record(session, report, options=options))
        actual_inputs = report["reproduction"]["inputs"].values()
        expected = {str(Path(ref["path"]).resolve()): ref["sha256"] for ref in recipe["inputs"].values()}
        for ref in actual_inputs:
            path = str(Path(ref["path"]).resolve())
            if path not in expected:
                raise ValueError(f"The detector reader used an input outside the record: {path}")
            if ref["sha256"] != expected[path]:
                raise ValueError(f"SHA-256 mismatch for {path}: expected {expected[path]}, got {ref['sha256']}")
        return report
    finally:
        session.close()


def _repeat(recipe, out, should_cancel):
    with tempfile.TemporaryDirectory(prefix="gimap-auto-") as tmp:
        work = Path(tmp)
        request = work / "recipe.json"
        request.write_text(json.dumps(recipe), encoding="utf-8")
        # Qt GUI objects must live on the main thread, even when Reproduce from Record runs on a worker.
        env = dict(os.environ, GIMAP_HOME=str(work / "home"), QT_QPA_PLATFORM="offscreen")
        worker = ["--automatic-replay-worker"] if getattr(sys, "frozen", False) else ["-m", "src.gimap.app.reproduce_automatic"]
        args = [sys.executable, *worker, str(request), str(out)]
        with subprocess.Popen(args, cwd=Path(__file__).resolve().parents[3], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
            try:
                while True:
                    if should_cancel and should_cancel():
                        raise CancelledError()
                    try:
                        _, error = process.communicate(timeout=0.2)
                        break
                    except subprocess.TimeoutExpired:
                        continue
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        if process.returncode:
            detail = error.decode("utf-8", errors="replace").strip().splitlines()
            raise ValueError(detail[-1] if detail else f"Automatic analysis exited with code {process.returncode}.")


def _sections(report):
    sections = {key: report[key] for key in ("peaks", "rings", "phase_assignment", "crystal_orientation",
                                              "cell_refinement") if key in report}
    sections["angles"] = {"incidence": report.get("incidence_measurement"), "roll": report.get("sample_roll_measurement"),
                         "incidence_deg": report["geometry"].get("incidence_deg"),
                         "sample_roll_deg": report["geometry"].get("sample_roll_deg")}
    if gisaxs := report.get("gisaxs"):
        sections["gisaxs_cut"] = {key: item for key, item in gisaxs.items() if key != "fit"}
        if fit := gisaxs.get("fit"):
            sections["fit"] = {**fit, "solutions": [{k: v for k, v in row.items() if k != "seconds"}
                                                  for row in fit["solutions"]]}
        if plan := report.get("reproduction", {}).get("predict"):
            sections["predict_options"] = plan["options"]
    return sections


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def replay_automatic(record, record_path, folder, should_cancel):
    from .reproduce import _compare_tables

    recipe, paths = _prepare(record["reproduction"], record_path.parent)
    out = folder / "reproduced_report.json"
    _repeat(recipe, out, should_cancel)
    actual = json.loads(out.read_bytes())
    expected_sections = _map_paths(_sections(record), paths)
    actual_sections = _sections(actual)
    sections = {}
    for key in dict.fromkeys([*expected_sections, *actual_sections]):
        original = _digest(expected_sections[key]) if key in expected_sections else None
        repeated = _digest(actual_sections[key]) if key in actual_sections else None
        sections[key] = {"status": "identical" if original == repeated else "different",
                         "expected_sha256": original, "reproduced_sha256": repeated}
    comparison = {"kind": "automatic", "file": out.name, "sections": sections,
                  "recorded_commit": recipe["commit"], "recorded_version": recipe["software_version"]}
    nested = record.get("analysis_record") or {}
    if nested.get("reproduction"):
        computed = actual["analysis_record"]["reproduction"]["reference"]
        tables = {key: {column: unpack_column(data) for column, data in columns.items()}
                  for key, columns in computed.items()}
        comparison.update(_compare_tables(nested, tables, record_path, folder))
    same = all(item["status"] == "identical" for item in sections.values())
    comparison["status"] = "identical" if same and comparison.get("status") != "different" else "different"
    return comparison


def worker_main(arguments):
    request, output = map(Path, arguments)
    try:
        result = _run_worker(json.loads(request.read_bytes()), request.parent)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=1, default=str)
    except (OSError, ValueError, RuntimeError) as error:
        if sys.stderr is not None:
            sys.stderr.buffer.write((str(error) + "\n").encode("utf-8"))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(worker_main(sys.argv[1:]))
