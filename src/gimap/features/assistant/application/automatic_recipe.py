"""Inputs to re-run the complete automatic procedure, before it changes Analyze's state."""

from dataclasses import asdict
from pathlib import Path

from src.gimap.shared.reproduction import file_reference, numerical_environment
from src.gimap.shared.revision import source_commit
from src.gimap.shared.version import SOFTWARE


def automatic_recipe(report, options, initial, final):
    inputs = {**initial["inputs"], **final["inputs"]}
    if options.notes_source:
        inputs[options.notes_source["path"]] = options.notes_source
    phases = final["phases"] if report.get("phase_assignment") else []
    for phase in phases:
        if phase["cif"]:
            # Use the hash computed when the phase was parsed, even if the file was edited later.
            inputs[phase["cif"]] = {**file_reference(phase["cif"]), "sha256": phase["sha256"]}

    calibration = None
    for step in report["steps"]:
        if step["tool"] == "use_geometry" and not step["error"]:
            record = report.get("calibration") or {}
            calibration = step["arguments"].get("path") or record.get("calibration_file") or record.get("source_image")
    if calibration and (not options.geometry or Path(calibration).is_file()):
        inputs.setdefault(calibration, file_reference(calibration))
    summed_files = list(dict.fromkeys([*initial["summed_files"], *final["summed_files"]]))
    fit = (report.get("gisaxs") or {}).get("fit") or {}
    return {
        "schema": 1, "kind": "automatic", "software_version": SOFTWARE, "commit": source_commit(),
        "runtime": numerical_environment(), "inputs": inputs, "frame": report["frame"], "procedure": report["procedure"],
        "frame_index": initial["frame_index"],
        "summed_files": summed_files,
        "import_selections": {k: v for k, v in initial["import_selections"].items()
                              if k == report["frame"] or k in summed_files},
        "options": asdict(options), "settings": initial["settings"], "calibration_file": calibration,
        "phases": phases, "indexing": final["indexing"], "predict_method": fit.get("method"),
        "angles": {"incidence": report.get("incidence_measurement"), "roll": report.get("sample_roll_measurement"),
                   "incidence_deg": report["geometry"].get("incidence_deg"),
                   "sample_roll_deg": report["geometry"].get("sample_roll_deg")},
    }
