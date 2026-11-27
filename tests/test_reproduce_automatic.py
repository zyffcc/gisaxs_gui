"""Replay the public automatic-analysis examples and check the structural results against fixed references."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from src.gimap.app.reproduce import reproduce_record
from src.gimap.features.fitting.domain.posterior.prediction import LOCAL_MAX


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "tests/data/external"
ARCHIVE = ROOT / "tests/data/automatic_replay"
ARCHIVE_PLATFORM = sys.platform == "win32" and sys.version_info[:3] == (3, 10, 18)


def fields(row, names):
    return {name: row[name] for name in names.split()}


def structural_results(report):
    angles = fields(report["geometry"], "incidence_deg sample_roll_deg")
    if report["gisaxs"]:
        fit = report["gisaxs"]["fit"]
        solutions = []
        for solution in fit["solutions"]:
            row = fields(solution, "model combination compared_by components globals chi2 "
                                   "least_squares_values intervals posterior_summary")
            diagnostics = solution["diagnostics"]
            row["composition_count"] = diagnostics["composition_count"]
            if solution["compared_by"] == "bic":
                row.update(bic_weight=solution["bic_weight"], composition_bic_weight=diagnostics["composition_bic_weight"])
            else:
                row["probability"] = solution["probability"]
                row.update(fields(diagnostics, "composition_probability composition_probability_mc68"))
            solutions.append(row)
        return dict(angles=angles, chosen=fit["chosen"], method=fit["method"], solutions=solutions)

    assignment = report["phase_assignment"]
    peaks = []
    for item in assignment["assignments"]:
        row = fields(item, "assigned phase hkl ambiguous tentative q q_hkl")
        row["alternatives"] = [fields(alt, "phase hkl q_hkl") for alt in item["alternatives"]]
        peaks.append(row)
    orientation = [fields(row, "phase determined reasons axes populations near_random fraction_iso "
                               "fraction_iso_err oriented_fraction reduced_chi2")
                   for row in report["crystal_orientation"]]
    cells = []
    for row in report["cell_refinement"]:
        cell = fields(row, "phase ok reason system values errors")
        if row["ok"]:
            cell.update(fields(row, "zero_shift zero_shift_error"))
        cells.append(cell)
    return dict(angles=angles, assignments=peaks, orientation=orientation, cells=cells)


def compare_archive(actual, expected, *, path=()):
    """Keep categorical outputs exact. Floating-point tolerances below apply only across platforms."""
    if ARCHIVE_PLATFORM:
        # JSON's round-trippable float representation distinguishes one ULP and the sign of zero.
        assert json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for k, value in expected.items():
            compare_archive(actual[k], value, path=(*path, k))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, value in enumerate(expected):
            compare_archive(actual[i], value, path=(*path, i))
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=0.01, abs=1e-10), path
    else:
        assert actual == expected, path


def compare_gisaxs(actual, expected):
    """Another platform's Predict result of a GALAXI frame against the archive's.

    The frame has several solutions of nearly the same probability. Which one ranks first, how many keep 1 %,
    and whether every pooled composition drew 10 effective samples (else they are compared by BIC) change with
    the last bits of the processor's and the libraries' arithmetic: on the CI computers the first solution and
    the method changed, while the best χ² stayed within 5 % of the archive's (full arm 1.089-1.090, positive arm
    1.171-1.226 against 1.173). The geometry, the choice, the method and the composition counts are exact."""
    for key in ("angles", "chosen", "method"):
        compare_archive(actual[key], expected[key], path=(key,))
    solutions = actual["solutions"]
    assert 1 <= len(solutions) <= LOCAL_MAX
    assert {s["composition_count"] for s in solutions} == {s["composition_count"] for s in expected["solutions"]}
    weights = []
    for solution in solutions:
        by_bic = solution["compared_by"] == "bic"
        assert ("bic_weight" in solution) == by_bic and ("probability" in solution) != by_bic
        weights.append(solution["bic_weight"] if by_bic else solution["probability"])
    assert weights == sorted(weights, reverse=True)
    best = min(s["chi2"] for s in expected["solutions"])
    assert min(s["chi2"] for s in solutions) == pytest.approx(best, rel=0.10)


def export_public_frame(case, folder, *extra_options):
    spec = json.loads((ARCHIVE / f"{case}.json").read_text(encoding="utf-8"))
    inputs = folder / "inputs"
    inputs.mkdir()
    for name, digest in spec["input_sha256"].items():
        source, dest = PUBLIC / spec["dataset"] / name, inputs / name
        if name == spec.get("mask"):
            # Mask the outer negative arm. The shorter measured reach makes the automatic
            # procedure fit the positive arm and provides an independent evidence-based case.
            mask = np.zeros((1043, 981), dtype=np.uint8)
            mask[:, :500] = 1
            np.save(dest, mask, allow_pickle=False)
        elif source.suffix in (".cif", ".poni"):
            # Preserve the exact archived input bytes across Git's checkout line-ending settings.
            dest.write_bytes(source.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        else:
            shutil.copyfile(source, dest)
        assert hashlib.sha256(dest.read_bytes()).hexdigest() == digest
    frame = inputs / spec["frame"]
    command = [sys.executable, "-m", "tools.gimap_agent", "auto", str(frame),
               "--no-saved-profiles", "--calibration", str(inputs / spec["calibration"]),
               "--incidence-deg", str(spec["reference_incidence_deg"]), "--out", str(folder / "export")]
    for cif in spec["phases"]:
        command += ["--phase", str(inputs / cif)]
    if "mask" in spec:
        command += ["--mask", str(inputs / spec["mask"])]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run([*command, *extra_options], cwd=ROOT, env=env, capture_output=True, encoding="utf-8")
    assert result.returncode in (0, 2), (result.stdout, result.stderr)
    path = folder / "export" / frame.stem / "report.json"
    # A collaborator receives the unchanged export and byte-identical input files beside it.
    for original in inputs.iterdir():
        original.rename(path.parent / original.name)
    return case, path, spec, env


@pytest.fixture(scope="module", params=["p08", "galaxi", "galaxi_positive"])
def exported(request, tmp_path_factory):
    return export_public_frame(request.param, tmp_path_factory.mktemp(request.param))


def replay_export(path, folder, env, monkeypatch):
    for name in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
        monkeypatch.setenv(name, env[name])
    return reproduce_record(path, folder)


def test_export_replays_structural_results_on_public_frame(exported, tmp_path, monkeypatch):
    case, path, archive, env = exported
    record = json.loads(path.read_text(encoding="utf-8"))
    recipe = record["reproduction"]
    assert recipe["kind"] == "automatic" and len(recipe["commit"]) == 40
    assert {ref["file"]: ref["sha256"] for ref in recipe["inputs"].values()} == archive["input_sha256"]
    assert recipe["settings"] and recipe["options"]["incidence_deg"] == archive["reference_incidence_deg"]
    assert recipe["angles"]["incidence"] == record["incidence_measurement"]
    assert recipe["angles"]["roll"] == record["sample_roll_measurement"]
    if case == "p08":
        assert recipe["options"]["assume_fibre_normal"] is False
        assert [phase["source_sha256"] for phase in record["phase_assignment"]["phases"]] == [
            archive["input_sha256"][cif] for cif in archive["phases"]]
    else:
        plan = recipe["predict"]
        assert plan["weights"]["sha256"] == archive["expected"]["method"]["network"]["sha256"]
        manifest = Path(plan["network_record"]["path"])
        assert plan["network_record"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert fields(plan["options"], "seed shortlist samples local_samples") == {
            "seed": 0, "shortlist": 4, "samples": 1000, "local_samples": 400}
    comparison = replay_export(path, tmp_path, env, monkeypatch)
    assert comparison["kind"] == "automatic"
    assert {ref["sha256"] for ref in comparison["relocated_inputs"].values()} == set(archive["input_sha256"].values())
    assert comparison["status"] == "identical", comparison.get("sections")
    if case != "p08":
        assert comparison["sections"]["predict_options"]["status"] == "identical"
    replay = json.loads((tmp_path / "reproduced_report.json").read_text(encoding="utf-8"))
    # Check both ends. The numerical archive is independent of this test run and of the replay comparator.
    for report in (record, replay):
        if ARCHIVE_PLATFORM or case == "p08":
            compare_archive(structural_results(report), archive["expected"])
        else:
            compare_gisaxs(structural_results(report), archive["expected"])


def test_p08_replays_with_fibre_normal_assumption(tmp_path, monkeypatch):
    _, path, _, env = export_public_frame("p08", tmp_path, "--assume-fibre-normal")
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["reproduction"]["options"]["assume_fibre_normal"] is True
    roll = record["sample_roll_measurement"]
    assert roll["method"] == "fibre assumption" and roll["significant"]
    # Geometry in the report keeps 15 significant digits of the measured angle.
    assert record["geometry"]["sample_roll_deg"] == pytest.approx(roll["deg"], rel=5e-15, abs=0)

    out = tmp_path / "replay"
    comparison = replay_export(path, out, env, monkeypatch)
    assert comparison["status"] == "identical", comparison.get("sections")
    replay = json.loads((out / "reproduced_report.json").read_text(encoding="utf-8"))
    assert replay["reproduction"]["options"]["assume_fibre_normal"] is True
    assert structural_results(replay) == structural_results(record)


def test_replay_rejects_changed_structural_input(exported, tmp_path):
    case, path, spec, _ = exported
    # Keep the module fixture usable by other consumers even if the test order changes.
    name = spec["phases"][0] if case == "p08" else spec["calibration"]
    source = path.parent / name
    original = source.read_bytes()
    try:
        source.write_bytes(original + b"\n")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            reproduce_record(path, tmp_path)
        assert not list(tmp_path.iterdir())
    finally:
        source.write_bytes(original)
