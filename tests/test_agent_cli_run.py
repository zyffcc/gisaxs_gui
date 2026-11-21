"""What a first run of ``tools/gimap_agent.py auto`` on the public frames must get right.

- frames of one name in different folders each keep their own folder and report;
- a pattern the shell did not expand and a folder stand for their files, and say how many;
- an output folder with an earlier run is not written into unless ``--overwrite``;
- ``--mask`` applies a mask as Analyze reads it, and every report says which mask was applied or that none was;
- report.json and summary.json identify the inputs (path, size, SHA-256), the time, the options and, for a
  model fit, the network, its weights and the seed; every CSV starts with the same ``#`` lines; a repeated run
  differs only in the time and the durations.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.gimap.app.headless_inputs import EARLIER_RUN, expand_frames, frame_folders, prepare_output, summary_reports
from src.gimap.app.headless_sources import GivenSource
from src.gimap.features.assistant.application import PipelineOptions
from src.gimap.shared.version import SOFTWARE

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data" / "external"
GALAXI = DATA / "gisaxs_galaxi"
XEUSS = DATA / "giwaxs_xeuss_pbte"
NETWORK = {"name": "flow_v1", "sha256": "ab" * 32}
"""An invented network record, as Fitting's ``create_curve_fit`` puts it on every row."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sources(*paths: str) -> list[GivenSource]:
    return [GivenSource(path, (path,)) for path in paths]


# -- frames: patterns, folders, folder names (no frame is read) -------------------------------------------


def test_frames_of_one_name_get_folders_of_their_own() -> None:
    assert frame_folders(_sources("a/film.tif", "b/other.tif")) == ["film", "other"]
    assert frame_folders(_sources("run1/raw/film.tif", "run2/raw/film.tif", "x/dots.tif")) == ["film_run1", "film_run2", "dots"]
    assert frame_folders(_sources("a/film.tif", "b/film.tif", "c/FILM.tif")) == ["film_a", "film_b", "FILM_c"]
    # No folder name tells them apart at one level (a/x, x/a against a/a): numbered, still all different.
    names = frame_folders(_sources("a/x/film.tif", "x/a/film.tif", "a/a/film.tif"))
    assert len({name.casefold() for name in names}) == 3 and all(name.startswith("film") for name in names)
    # A name made for one frame is not taken from another that is really called so.
    names = frame_folders(_sources("a/film.tif", "b/film.tif", "c/film_a.tif"))
    assert len({name.casefold() for name in names}) == 3


def test_a_pattern_and_a_folder_stand_for_their_files_in_natural_order(tmp_path: Path) -> None:
    folder = tmp_path / "data"
    folder.mkdir()
    for name in ("film_10.tif", "film_2.tif", "film_1.tif", "notes.txt", "mask.edf", "agbh.poni"):
        (folder / name).write_bytes(b"")
    (folder / "deeper").mkdir()
    (folder / "deeper" / "film_3.tif").write_bytes(b"")

    frames, notes = expand_frames([str(folder / "film_*.tif")])
    assert [Path(frame).name for frame in frames] == ["film_1.tif", "film_2.tif", "film_10.tif"]
    assert notes == [f"Pattern {folder / 'film_*.tif'}: 3 file(s) matched."]

    frames, notes = expand_frames([str(folder)], exclude=(str(folder / "mask.edf"), None))
    assert [Path(frame).name for frame in frames] == ["film_1.tif", "film_2.tif", "film_10.tif"]  # not the subfolder
    assert "3 detector image(s) found" in notes[0] and "subfolders are not searched" in notes[0]
    assert "1 file(s) given as calibration or mask left out" in notes[0]
    assert len(expand_frames([str(folder)])[0]) == 4  # the mask is a detector image unless it is named as the mask

    # A plain name is passed on as it is: a missing file is that frame's error, not the batch's.
    assert expand_frames(["missing.tif", str(folder / "film_1.tif")]) == (["missing.tif", str(folder / "film_1.tif")], [])
    with pytest.raises(FileNotFoundError, match="No file matches the pattern"):
        expand_frames([str(folder / "*.cbf")])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No detector image in the folder"):
        expand_frames([str(empty)])


def test_an_output_folder_with_an_earlier_run_is_refused_unless_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert prepare_output(out) == []  # nothing there yet
    for name in ("film_a", "film_b", "mine"):
        (out / name).mkdir(parents=True)
        (out / name / "report.md").write_text("old", encoding="utf-8")
    (out / "keep.txt").write_text("the person's own file", encoding="utf-8")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    summary = {"software_version": SOFTWARE, "frames": [
        {"frame": "a/film.tif", "folder": "film_a"}, {"frame": "b/film.tif", "folder": "film_b"},
        {"frame": "x.tif", "folder": "../elsewhere"}, {"frame": "gone.tif", "error": "No such file"},
    ]}
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (out / "summary.md").write_text("# old", encoding="utf-8")

    with pytest.raises(FileExistsError) as refused:
        prepare_output(out)
    assert str(refused.value) == EARLIER_RUN.format(out=out, found="summary.md, summary.json")
    assert "--overwrite" in str(refused.value) and (out / "film_a" / "report.md").is_file()  # nothing was touched

    removed = prepare_output(out, overwrite=True)
    assert sorted(folder.name for folder in removed) == ["film_a", "film_b"]
    # Only the frame folders the earlier summary lists, and only those directly in the output folder.
    assert sorted(path.name for path in out.iterdir()) == ["keep.txt", "mine", "summary.json", "summary.md"]
    assert outside.is_dir()

    # A summary written before 0.2.0 was the list alone; its folders are those of the outputs.
    (out / "film_a").mkdir()
    (out / "summary.json").write_text(json.dumps([{"frame": "a.tif", "outputs": [str(out / "film_a" / "report.json")]}]),
                                      encoding="utf-8")
    assert [folder.name for folder in prepare_output(out, overwrite=True)] == ["film_a"]
    assert summary_reports([{"frame": "a.tif"}]) == [{"frame": "a.tif"}] == summary_reports({"frames": [{"frame": "a.tif"}]})
    with pytest.raises(ValueError):
        summary_reports({"reports": []})


def test_the_command_expands_its_arguments_and_refuses_what_it_cannot_run(monkeypatch, tmp_path: Path, capsys) -> None:
    from src.gimap.app import headless_assistant
    from tools import gimap_agent

    folder = tmp_path / "data"
    folder.mkdir()
    for name in ("film_2.tif", "film_10.tif"):
        (folder / name).write_bytes(b"")
    seen: dict = {}

    def analyse_frames(frames, options, out, **kwargs):
        seen.update(kwargs, frames=list(frames), out=out)
        return [{"ok": True, "frame": frame, "procedure": "giwaxs", "needs_attention": [], "decisions": [], "steps": []}
                for frame in frames]

    monkeypatch.setattr(headless_assistant, "analyse_frames", analyse_frames)
    out = tmp_path / "out"
    code = gimap_agent.main(["auto", str(folder / "*.tif"), "--quiet", "--out", str(out), "--notes", "alpha_i 0.2 deg",
                             "--overwrite", "--language", "zh"])
    printed = capsys.readouterr().out
    assert code == 0 and [Path(frame).name for frame in seen["frames"]] == ["film_2.tif", "film_10.tif"]
    assert seen["overwrite"] is True and seen["mask"] is None and seen["language"] == "zh"
    assert f"Pattern {folder / '*.tif'}: 2 file(s) matched." in printed and seen["found"] == (f"Pattern {folder / '*.tif'}: 2 file(s) matched.",)
    # The options as they were given, for the record: the pattern itself, the notes, and no function.
    given = seen["record"]
    assert given["frames"] == [str(folder / "*.tif")] and given["notes"] == "alpha_i 0.2 deg" == given["notes_text"]
    assert given["command"] == "auto" and given["overwrite"] is True and "run" not in given
    json.dumps(given)

    seen.clear()
    assert gimap_agent.main(["auto", str(folder), "--quiet", "--out", str(out)]) == 0
    assert len(seen["frames"]) == 2 and seen["overwrite"] is False and seen["language"] == "en"

    # A pattern or folder without a frame, and a mask that is not there: an error before anything is analysed.
    seen.clear()
    assert gimap_agent.main(["auto", str(folder / "*.cbf"), "--quiet", "--out", str(out)]) == 1
    assert "No file matches the pattern" in capsys.readouterr().err
    assert gimap_agent.main(["auto", str(folder), "--mask", str(tmp_path / "no_mask.edf"), "--out", str(out)]) == 1
    assert "No such mask file" in capsys.readouterr().err and not seen
    text = " ".join(gimap_agent.parser()._subparsers._group_actions[0].choices["auto"].format_help().split())
    for option in ("--mask", "--overwrite", "--language", "a folder stands for every detector image"):
        assert option in text, option


# -- the public frames -------------------------------------------------------------------------------------


class Fitter:
    """Two solutions as Fitting's ``create_curve_fit`` gives them, named by network and seed (no network is run)."""

    def __call__(self, q, intensity, sigma, *, components=(), report=None, cancelled=None):
        from tests.test_guided_gisaxs import _row

        rows = [_row(q, intensity, sigma, 1, "sphere", 9.4, 53.4, 1.31, probability=0.9),
                _row(q, intensity, sigma, 2, "vertical_cylinder", 7.7, 30.8, 1.45, probability=0.1)]
        return [{**row, "network": dict(NETWORK), "seed": 0} for row in rows]


def _timeless(path: Path) -> str:
    """A JSON record without what may differ between two equal runs: the time and the durations. The ids of the
    operations (``op7``) are counted per process: equal for two command lines, not for two runs of one test."""
    def strip(value):
        if isinstance(value, dict):
            counted = ("id",) if str(value.get("id", "")).startswith("op") else ()
            return {key: strip(item) for key, item in value.items() if key not in ("created", "seconds", *counted)}
        return [strip(item) for item in value] if isinstance(value, list) else value

    return json.dumps(strip(json.loads(path.read_text(encoding="utf-8"))), sort_keys=True)


@pytest.fixture(scope="module")
def two_of_one_name(tmp_path_factory):
    """The public GALAXI frame under one name in two folders, analysed as one batch (with an invented fit)."""
    from src.gimap.app.headless_assistant import analyse_frames
    from src.gimap.features.fitting import bootstrap

    root = tmp_path_factory.mktemp("same_name")
    frames = []
    for name in ("a", "b"):
        (root / name).mkdir()
        frames.append(str(shutil.copy(GALAXI / "galaxi_data.tif", root / name / "galaxi_data.tif")))
    poni = GALAXI / "galaxi_bornagain.poni"
    given = {"command": "auto", "frames": [str(root / "*" / "galaxi_data.tif")], "calibration": str(poni),
             "notes": "sample B7, annealed", "notes_text": "sample B7, annealed"}
    patch = pytest.MonkeyPatch()
    patch.setattr(bootstrap, "create_curve_fit", lambda: Fitter())
    out = root / "out"

    def run(**extra):
        options = PipelineOptions(calibration=str(poni), incidence_deg=0.2, notes=given["notes"])
        return analyse_frames(frames, options, out, saved_profiles=False, record=given,
                              found=("Pattern x: 2 file(s) matched.",), **extra)

    try:
        reports = run()
        yield {"root": root, "out": out, "frames": frames, "poni": poni, "given": given, "reports": reports, "run": run}
    finally:
        patch.undo()


def test_frames_of_one_name_in_different_folders_keep_their_own_reports(two_of_one_name) -> None:
    out, reports = two_of_one_name["out"], two_of_one_name["reports"]
    assert [report["folder"] for report in reports] == ["galaxi_data_a", "galaxi_data_b"]
    assert sorted(path.name for path in out.iterdir()) == ["galaxi_data_a", "galaxi_data_b", "summary.json", "summary.md"]
    for report, frame in zip(reports, two_of_one_name["frames"]):
        folder = out / report["folder"]
        assert all(Path(path).parent == folder and Path(path).is_file() for path in report["outputs"])
        assert f"File: `{frame}`" in (folder / "report.md").read_text(encoding="utf-8")
        assert json.loads((folder / "report.json").read_text(encoding="utf-8"))["inputs"]["frame"][0]["path"] == frame
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "| galaxi_data.tif (galaxi_data_a) |" in summary and "| [report.md](galaxi_data_a/report.md) |" in summary
    assert "| galaxi_data.tif (galaxi_data_b) |" in summary and "| [report.md](galaxi_data_b/report.md) |" in summary
    assert "Pattern x: 2 file(s) matched." in summary and "需要处理" not in summary
    listed = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert [frame["folder"] for frame in listed["frames"]] == ["galaxi_data_a", "galaxi_data_b"]


def test_the_record_identifies_the_inputs_the_options_and_the_fit(two_of_one_name) -> None:
    out, frames, poni = two_of_one_name["out"], two_of_one_name["frames"], two_of_one_name["poni"]
    report = json.loads((out / "galaxi_data_a" / "report.json").read_text(encoding="utf-8"))
    digest = _sha256(Path(frames[0]))
    assert report["software_version"] == SOFTWARE
    assert report["created"].endswith("+00:00") and len(report["created"]) == len("2026-01-02T03:04:05+00:00")
    assert report["options"] == two_of_one_name["given"]  # as given: the pattern, the notes
    assert report["inputs"]["frame"] == [
        {"file": "galaxi_data.tif", "path": frames[0], "bytes": Path(frames[0]).stat().st_size, "sha256": digest}]
    assert report["inputs"]["calibration"] == {
        "file": poni.name, "path": str(poni), "bytes": poni.stat().st_size, "sha256": _sha256(poni)}
    assert report["inputs"]["mask"] is None and report["inputs"]["background"] is None
    # No mask was given: the report says so, and names the pixels found automatically apart from it.
    assert report["mask"]["applied"] is False and report["mask"]["file"] is None
    assert set(report["mask"]["bad_pixels_left_out"]) == {"hot", "dead", "defective_line_pixels"}
    text = (out / "galaxi_data_a" / "report.md").read_text(encoding="utf-8")
    assert "## Mask\n\n- mask applied: none (no mask file was given)\n- left out apart from a mask: the detector's gaps" in text
    # Analyze's own record of the frame: one vocabulary with the window's exports.
    record = report["analysis_record"]
    assert record["software"] == "GIMaP Analyze 1" and record["created"] == report["created"]
    assert record["input_sha256"] == [{"file": "galaxi_data.tif", "sha256": digest}]
    assert record["geometry"]["distance_m"] == pytest.approx(1.73) and record["error_model"] == "poisson"
    assert "updated_at" not in record["instrument_profile"]  # a profile the run made: no clock time of its own
    # The fit: the network by name and the SHA-256 of its weights, the seed, the solutions with their intervals.
    fit = report["gisaxs"]["fit"]
    assert fit["method"] == {"name": "Predict parameters", "network": NETWORK, "seed": 0}
    assert fit["solutions"][0]["model"] == "sphere" and fit["solutions"][0]["components"][0]["R"] == 9.4
    assert fit["solutions"][0]["intervals"]["R_1"]["low68"] == pytest.approx(0.97 * 9.4)
    assert f"Network: flow_v1 (weights SHA-256 {NETWORK['sha256']}), seed 0." in text

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert list(summary) == ["software_version", "created", "options", "arguments", "frames"]
    assert summary["software_version"] == SOFTWARE and summary["created"] == report["created"]
    assert summary["options"] == two_of_one_name["given"] and summary["arguments"] == ["Pattern x: 2 file(s) matched."]
    first = summary["frames"][0]
    assert first["inputs"] == report["inputs"] and "tables" not in first and "analysis_record" not in first


def test_every_csv_starts_with_the_same_lines_that_name_the_inputs(two_of_one_name) -> None:
    folder = two_of_one_name["out"] / "galaxi_data_a"
    digest = _sha256(Path(two_of_one_name["frames"][0]))
    tables = sorted(folder.glob("*.csv"))
    assert {path.name for path in tables} == {"horizontal.csv", "vertical.csv", "fit_curve.csv", "fit_solutions.csv"}
    heads = {path.name: path.read_text(encoding="utf-8").splitlines()[:4] for path in tables}
    assert len({tuple(head) for head in heads.values()}) == 1, heads
    head = heads["fit_solutions.csv"]
    assert head[0] == f"# {SOFTWARE} (command line); settings and decisions in report.json"
    assert head[1] == f"# input: galaxi_data.tif sha256 {digest}" and head[2].startswith("# frame: ")
    assert head[3].startswith("# geometry: distance 1730.0000 mm, direct beam (") and "alpha_i 0.2 deg" in head[3]
    solutions = (folder / "fit_solutions.csv").read_text(encoding="utf-8").splitlines()
    assert solutions[4].startswith("# GISAXS fit solutions [") and solutions[5].startswith("# units: ")
    assert solutions[6].startswith("rank,model,chi2,probability") and b"\r" not in (folder / "fit_solutions.csv").read_bytes()
    for path in tables:  # a reader that skips the comment lines still gets numbers
        if path.name != "fit_solutions.csv":
            assert np.loadtxt(path, delimiter=",", encoding="utf-8").shape[1] >= 3


def test_a_repeated_run_is_refused_then_identical_except_for_the_time(two_of_one_name) -> None:
    out, run = two_of_one_name["out"], two_of_one_name["run"]
    kept = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file() and path.suffix != ".json"}
    records = {path.relative_to(out): _timeless(path) for path in out.rglob("*.json")}
    stale = out / "galaxi_data_a" / "left_by_the_first_run.txt"
    stale.write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already holds the summary of an earlier run"):
        run()
    assert stale.is_file()

    run(overwrite=True)
    assert not stale.exists()  # the earlier run's frame folders went first
    again = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file() and path.suffix != ".json"}
    assert sorted(again) == sorted(kept)
    assert [str(name) for name in kept if kept[name] != again[name]] == []  # reports, tables and pictures: byte for byte
    assert {path.relative_to(out): _timeless(path) for path in out.rglob("*.json")} == records


def test_a_mask_is_applied_as_analyze_reads_it_and_the_report_says_which(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import analyse_frames

    frame, mask = XEUSS / "PbTe_3010_r1_PT_1_10_0_00028.edf", XEUSS / "mask_PbTe_3010.edf"
    options = PipelineOptions(calibration=str(XEUSS / "AgBe_250403_XENOCS.poni"), incidence_deg=0.2)
    before = sorted(path.name for path in XEUSS.iterdir())
    [report] = analyse_frames([str(frame)], options, tmp_path / "masked", saved_profiles=False, mask=str(mask))
    assert report["ok"] and report["mask"]["applied"] is True and report["mask"]["masked_pixels"] > 0
    assert report["mask"]["file"] == {"file": mask.name, "path": str(mask), "bytes": mask.stat().st_size,
                                      "sha256": _sha256(mask)} == report["inputs"]["mask"]
    assert report["analysis_record"]["corrections"]["mask_path"] == str(mask)
    assert report["analysis_record"]["drawn_mask_pixels"] == report["mask"]["masked_pixels"]
    text = (tmp_path / "masked" / frame.stem / "report.md").read_text(encoding="utf-8")
    assert f"- mask applied: `{mask}` (mask image: its non-zero pixels are left out; " in text
    assert f"{report['mask']['masked_pixels']} pixels masked; SHA-256 {_sha256(mask)})" in text
    assert sorted(path.name for path in XEUSS.iterdir()) == before  # nothing written next to the data

    # A mask of another detector: that frame fails with the application's reason, nothing is analysed with it.
    [refused] = analyse_frames([str(GALAXI / "galaxi_data.tif")], PipelineOptions(), tmp_path / "other",
                               saved_profiles=False, mask=str(mask))
    assert not refused["ok"] and mask.name in refused["error"] and "the frame is" in refused["error"]
    # A drawn mask (GIMaP's masks file) is applied too, and named by its file.
    shapes = tmp_path / "masks.json"
    shapes.write_text(json.dumps({"software": "GIMaP Analyze", "masks": [
        {"kind": "rectangle", "points": [[10, 10], [60, 40]]}]}), encoding="utf-8")
    [drawn] = analyse_frames([str(frame)], options, tmp_path / "drawn", saved_profiles=False, mask=str(shapes))
    assert drawn["mask"]["applied"] is True and drawn["mask"]["file"]["sha256"] == _sha256(shapes)
    assert drawn["mask"]["kind"] == "GIMaP masks file: 1 drawn shape(s) left out" and drawn["mask"]["masked_pixels"] > 0
    assert drawn["inputs"]["mask"] == drawn["mask"]["file"]


# -- what the run says about itself, and the documents ----------------------------------------------------


def test_the_fit_is_not_announced_with_a_time_that_is_not_true() -> None:
    """It said "about 25 s"; the public GALAXI cut takes well over a minute on a laptop. The time grows with the
    points and the compositions tried, so the run names the range."""
    import inspect

    from src.gimap.features.assistant.application import gisaxs_procedure, tool_specs
    from src.gimap.features.assistant.application.pipeline_progress import FIT_TAKES, SLOW

    assert FIT_TAKES == "some seconds to a few minutes" and SLOW["fit_horizontal_cut"] == f"takes {FIT_TAKES}"
    announced = inspect.getsource(gisaxs_procedure.GisaxsProcedureMixin._fit)
    assert f"predicting the particles of I(qy) ({FIT_TAKES})…" in announced and "25 s" not in announced
    fit = next(spec for spec in tool_specs(allow_images=False) if spec.name == "fit_horizontal_cut")
    assert "some seconds to a few minutes" in fit.description and "25 s" not in fit.description
    for path in ("README.md", "docs/User_Manual.md", "docs/User_Manual_zh.md", "docs/agents/gisaxs-playbook.md"):
        page = (ROOT / path).read_text(encoding="utf-8")
        assert "about 25 s with" not in page and "约 25 s" not in page and "about 25 s choosing" not in page, path


def test_the_documents_describe_the_tilt_the_record_and_the_checks_as_they_are() -> None:
    from src.gimap.features.fitting.domain.posterior.refine import SHORTLIST

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "folded into the beam centre" not in readme and "tilt is reported with its q error" not in readme
    assert "the rotations of a `.poni` file are part of the geometry" in readme
    install = readme.split("## Install")[1].split("\n## ")[0]
    assert "python tools/offscreen_smoke.py" in install
    for option in ("--mask FILE", "--overwrite", "SHA-256"):
        assert option in readme.split("## Command Line and MCP")[1].split("\n## ")[0], option
    playbook = (ROOT / "docs" / "agents" / "gisaxs-playbook.md").read_text(encoding="utf-8")
    assert "倾斜被略去" not in playbook and "geometry.flat_detector" not in playbook and "是几何的一部分" in playbook
    manual = (ROOT / "docs" / "User_Manual.md").read_text(encoding="utf-8")
    assert "do not name the release yet" not in manual and "`.poni` tilt left out" not in manual
    for name, section, solid in (("Physics_Manual.md", "### 14.10 The record of a run without the window", "(untilted)"),
                                 ("Physics_Manual_zh.md", "### 14.10 不用窗口运行时的记录", "（不倾斜）")):
        physics = (ROOT / "docs" / name).read_text(encoding="utf-8")
        record = physics.split(section)[1].split("\n## ")[0]
        for key in ("`software_version`", "`created`", "`options`", "`notes_text`", "`inputs.frame`", "`inputs.calibration`",
                    "`inputs.mask`", "`analysis_record`", "`gisaxs.fit.method`", "`seed`", "`summary.json`", "`frames`",
                    "fit_solutions.csv", "--overwrite"):
            assert key in record, (name, key)
        five = physics.split("### 5.2 ")[1].split("### 5.3 ")[0]
        assert "F_Ω = Ω_p / Ω_beam = (L / |t|)³" in five and solid in five and "F_Ω = cos³ 2θ" in five
        assert "1. The detector is flat and perpendicular to the direct beam." not in physics
        assert "gimap_agent.py" in physics.split("### 14.7 ")[1].split("### 14.8 ")[0]
    for module in ("src/gimap/shared/geometry/detector_tilt.py", "src/gimap/features/analyze/domain/intensity.py"):
        head = (ROOT / module).read_text(encoding="utf-8").split('"""')[1]
        assert "drops the rotations" not in head and "a flat detector perpendicular to the direct beam," not in head, module
    ml1d = (ROOT / "tools" / "ml1d" / "README.md").read_text(encoding="utf-8")
    assert SHORTLIST == 4 and "the head's four most probable" in ml1d and "three most probable" not in ml1d
    changes = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8").split("### Release preparation")[1].split("\n### ")[0]
    assert "tools/gimap_agent.py auto`): frames no longer overwrite each other" in changes


def test_cut_model_checks_keeps_its_stored_summary_when_a_comparison_was_skipped(monkeypatch, tmp_path: Path) -> None:
    import importlib

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # the script sets them: undone after
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr("sys.path", list(__import__("sys").path))
    checks = importlib.import_module("tools.benchmarks.cut_model_checks")
    assert checks.STORED.name == "cut_model_checks.json" and checks.STORED.is_file()
    assert checks.refuses(checks.STORED, [], force=False) == ""  # every comparison ran: the stored file may be written
    refusal = checks.refuses(checks.STORED, ["v5", "fit1d"], force=False)
    assert "v5 and fit1d skipped" in refusal and "--out PATH" in refusal and "--force" in refusal
    assert checks.refuses(checks.STORED, ["fit1d"], force=True) == ""
    assert checks.refuses(tmp_path / "mine.json", ["fit1d"], force=False) == ""  # another file is the person's own
    text = (ROOT / "tools" / "benchmarks" / "README.md").read_text(encoding="utf-8")
    assert "`--out PATH`" in text and "`--force`" in text and "refuses to" in " ".join(text.split())


def test_the_evaluation_reads_a_summary_of_either_form(tmp_path: Path) -> None:
    """summary.json became an object in 0.2.0; ``eval_giwaxs_agent.py --from`` reads it and the list it was."""
    from tools import eval_giwaxs_agent

    report = {"ok": True, "frame": "film.tif", "procedure": "giwaxs"}
    (tmp_path / "summary.json").write_text(json.dumps({"software_version": SOFTWARE, "frames": [report]}), encoding="utf-8")
    assert eval_giwaxs_agent._saved_reports(tmp_path) == [report]
    old = tmp_path / "old"
    old.mkdir()
    (old / "summary.json").write_text(json.dumps([report]), encoding="utf-8")
    assert eval_giwaxs_agent._saved_reports(old) == [report]
    (old / "summary.json").write_text(json.dumps({"software_version": SOFTWARE}), encoding="utf-8")
    with pytest.raises(eval_giwaxs_agent.NothingToEvaluate, match="not a list of reports"):
        eval_giwaxs_agent._saved_reports(old)
