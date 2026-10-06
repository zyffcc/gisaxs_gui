"""The baseline for command-line and MCP agents: which procedure runs, the calibrations and values the person
states, where files go, and what the exit codes of tools/gimap_agent.py mean.

- Without a technique the command line and MCP follow GIMaP's Auto detection once the geometry is applied
  (a GISAXS frame gets the GISAXS procedure); a technique given forces one; when Auto cannot classify the
  frame GIWAXS runs and the report asks for the technique (exit 2).
- A calibration given (--calibration) or named in the notes is tried before the profile and the search; a
  calibration given that is not used is flagged; the one calibrant the notes name is the standard.
- A pixel size given beats the image header, and the decision names both values.
- export_results in a session without a window writes under the output folder, never next to the data.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from src.gimap.features.assistant.application import (
    GOALS,
    PERMISSION_AUTO,
    AnalysisGoals,
    PipelineOptions,
    RunResults,
    StandardPipeline,
    ToolCatalog,
    pipeline_markdown,
)
from src.gimap.features.assistant.application.pipeline import DETECTED, FALLBACK, FORCED, NO_GEOMETRY
from src.gimap.features.assistant.application.pipeline_geometry import NAMED
from src.gimap.features.assistant.domain.notes import calibrants_in_notes, standard_from_notes
from src.gimap.features.assistant.infrastructure import LocalFileExplorer
from tests.assistant_fakes import FakeWorkbench
from tests.test_assistant_calibration import POOR_AGBH, FakeCalibrator, agbh_image, save_tiff, touch
from tests.test_assistant_pipeline import SeriesWorkbench

DATA = Path(__file__).parent / "data" / "external"
GALAXI = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))["gisaxs_galaxi"]


def _goals(notes: str = "") -> AnalysisGoals:
    return AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO, instructions=notes)


def _decisions(report: dict, what: str) -> list[dict]:
    return [item for item in report["decisions"] if item["what"] == what]


# -- (1) the technique ------------------------------------------------------------------------------------------


class AutoWorkbench(FakeWorkbench):
    """Analyze on Auto with an instrument profile; ``measurement``: what Auto made of the frame (None: nothing)."""

    def status(self) -> dict:
        return {**super().status(), "mode": "auto"}


def _run(measurement, **options) -> tuple[dict, AutoWorkbench]:
    workbench = AutoWorkbench(measurement=measurement)
    report = StandardPipeline(ToolCatalog(workbench, _goals(), RunResults()), PipelineOptions(**options)).run()
    return report, workbench


def test_without_a_technique_the_baseline_runs_what_auto_detected_and_says_so() -> None:
    report, workbench = _run("gisaxs", follow_detection=True)
    assert report["procedure"] == "gisaxs" and not workbench.calls  # no switch: Analyze already reduces it as GISAXS
    assert _decisions(report, "technique") == [{"what": "technique", "decision": "GISAXS", "why": DETECTED}]

    report, _workbench = _run("giwaxs", follow_detection=True)
    assert report["procedure"] == "giwaxs" and _decisions(report, "technique")[0]["why"] == DETECTED
    assert report["needs_attention"] == []


def test_a_technique_given_forces_the_procedure_and_the_switch_says_why() -> None:
    report, workbench = _run("gisaxs", technique="giwaxs", follow_detection=True)
    assert report["procedure"] == "giwaxs" and ("set_mode", "giwaxs") in workbench.calls
    assert _decisions(report, "technique")[0] == {"what": "technique", "decision": "GIWAXS", "why": FORCED}
    assert _decisions(report, "measurement")[0] == {
        "what": "measurement", "decision": "switched from gisaxs to GIWAXS", "why": FORCED}
    assert report["needs_attention"] == []


def test_when_auto_cannot_classify_the_frame_giwaxs_runs_and_the_technique_is_asked_for() -> None:
    report, workbench = _run(None, follow_detection=True)
    assert report["ok"] and report["procedure"] == "giwaxs" and ("set_mode", "giwaxs") in workbench.calls
    assert _decisions(report, "technique")[0]["why"] == FALLBACK
    [item] = report["needs_attention"]
    assert item["item"] == "technique" and item["option"] == "technique"
    text = pipeline_markdown(report)
    assert "NEEDS INPUT" in text and "--technique giwaxs|gisaxs" in text

    # Process with AI (no follow_detection) keeps GIWAXS and asks nothing: it names its technique itself.
    report, _workbench = _run(None)
    assert report["procedure"] == "giwaxs" and report["needs_attention"] == []


def test_the_command_line_follows_auto_unless_told_and_its_exit_codes_mean_what_the_help_says(monkeypatch, tmp_path) -> None:
    from src.gimap.app import headless_assistant
    from tools import gimap_agent

    seen: list[PipelineOptions] = []
    batch: list[dict] = []

    def analyse_frames(frames, options, out, **_kwargs):
        seen.append(options)
        return [dict(report) for report in batch]

    monkeypatch.setattr(headless_assistant, "analyse_frames", analyse_frames)

    def auto(*extra: str) -> int:
        return gimap_agent.main(["auto", "a.tif", "--quiet", "--out", str(tmp_path), *extra])

    done = {"ok": True, "frame": "a.tif", "procedure": "giwaxs", "needs_attention": [], "decisions": [], "steps": []}
    model = {"item": "model", "why": "2 solutions fit within 10% of the best χ².", "option": None, "hint": ""}
    no_geometry = {"item": "calibration", "why": "No calibration candidate gave a good geometry.", "option": "calibration", "hint": ""}
    batch[:] = [done]
    assert auto() == 0 and seen[-1].technique is None and seen[-1].follow_detection
    assert auto("--technique", "gisaxs") == 0 and seen[-1].technique == "gisaxs" and not seen[-1].follow_detection
    batch[:] = [done, {**done, "procedure": "gisaxs", "needs_attention": [model]}]  # a judgement, no flag
    assert auto() == 2
    batch[:] = [{**done, "ok": False, "needs_attention": [no_geometry]}]  # nothing analysed: no geometry
    assert auto() == 2
    batch[:] = [{**done, "needs_attention": [model]}, {"ok": False, "frame": "b.tif", "error": "unreadable", "needs_attention": []}]
    assert auto() == 1  # a failed frame wins over open items
    unread = {"item": "frame", "why": "b.tif is not a detector image GIMaP can read.", "option": None, "hint": ""}
    batch[:] = [{**done, "ok": False, "failed": unread["why"], "needs_attention": [unread]}]
    assert auto() == 1  # a file GIMaP cannot read failed: there is nothing to answer

    text = gimap_agent.parser().format_help()  # the module docstring, also `auto --help`
    for line in ("0  every frame was analysed", "2  at least one frame has an open item", "1  at least one frame failed",
                 "GISAXS model choice", "no geometry was found"):
        assert line in text, line


def test_the_agent_texts_name_both_techniques_both_playbooks_and_where_files_go() -> None:
    from src.gimap.app import headless_assistant
    from src.gimap.app.headless_assistant import MCP_INSTRUCTIONS, PLAYBOOKS, mcp_tools
    from src.gimap.features.assistant.application import pipeline
    from tools import gimap_agent

    root = Path(__file__).parents[1]
    assert all((root / playbook).is_file() for playbook in PLAYBOOKS)
    for text in (MCP_INSTRUCTIONS, gimap_agent.__doc__):
        assert all(playbook in text for playbook in PLAYBOOKS) and "GISAXS" in text and "GIWAXS" in text
    assert "technique='giwaxs' or 'gisaxs'" in MCP_INSTRUCTIONS and "never next to the data" in MCP_INSTRUCTIONS
    for module in (headless_assistant, pipeline):
        assert "Process with Claude" not in module.__doc__ and "Process with AI" in module.__doc__
    for word in ("αi", "energy", "pixel size", "calibration file", "calibrant"):
        assert word in gimap_agent.__doc__, word
    tools = {tool["name"]: tool for tool in mcp_tools()}
    assert "out_dir" in tools["open_frame"]["input_schema"]["properties"]
    assert "never next to the data" in tools["export_results"]["description"]


# -- (2) calibrations and the calibrant the person names; (4) a pixel size given --------------------------------


def _series(tmp_path: Path, notes: str = "", *, image: str = "AgBH_00001.tif", explorer=None, workbench=None):
    root = tmp_path / "beamtime" / "raw"
    frame = touch(root / "film" / "film_00001_m01.nxs", "x")  # the fake workbench never reads it
    save_tiff(root / "calib" / image, agbh_image())
    workbench = (workbench or SeriesWorkbench)(frame)
    calibrator = FakeCalibrator()
    catalog = ToolCatalog(
        workbench, _goals(notes), RunResults(), explorer=explorer or LocalFileExplorer(), calibrator=calibrator,
    )
    return catalog, workbench, calibrator


def test_a_calibration_named_in_the_notes_is_used_before_the_search(tmp_path: Path) -> None:
    poni = touch(tmp_path / "elsewhere" / "final" / "geometry.poni", POOR_AGBH)  # far from the frame: no search finds it
    notes = f"12.4 keV, alpha_i = 0.2 deg. Geometry: {poni}"
    catalog, _workbench, calibrator = _series(tmp_path, notes)
    report = StandardPipeline(catalog, PipelineOptions(notes=notes, follow_detection=True)).run()

    assert report["ok"] and report["needs_attention"] == [], report["needs_attention"]
    assert _decisions(report, "calibration candidates") == [
        {"what": "calibration candidates", "decision": "geometry.poni", "why": NAMED}]
    assert _decisions(report, "geometry")[0]["decision"] == "from geometry.poni"
    assert "find_calibration_files" not in [step["tool"] for step in report["steps"]] and calibrator.calls == []

    # An image of a standard named in the notes is fitted first too, before the one the search would find.
    image = tmp_path / "other" / "x" / "y" / "LaB6_redone_0001.tif"
    save_tiff(image, agbh_image())
    notes = f"12.4 keV, alpha_i = 0.2 deg. Calibration image: {image}"
    catalog, _workbench, calibrator = _series(tmp_path / "second", notes)
    report = StandardPipeline(catalog, PipelineOptions(notes=notes, follow_detection=True)).run()
    assert calibrator.calls[0][:2] == ("LaB6_redone_0001.tif", "lab6")
    assert _decisions(report, "calibration candidates")[0]["decision"] == "LaB6_redone_0001.tif"


def test_a_named_calibration_beats_the_instrument_profile(tmp_path: Path) -> None:
    poni = touch(tmp_path / "processed" / "geometry.poni", POOR_AGBH)
    notes = f"12.4 keV, use {poni}"

    class ProfileWorkbench(FakeWorkbench):  # a saved profile, possibly from another beamtime
        def __init__(self):
            super().__init__()
            self.frame = touch(tmp_path / "raw" / "film_0001.tif", "x")

        def status(self):
            return {**super().status(), "path": str(self.frame), "shape": [400, 400],
                    "geometry": {"instrument_profile": "P03 2019", "incidence_deg": 0.2, "pixel_size_um": [100.0, 100.0]}}

        def use_geometry(self, values, name, source):
            self.calls.append(("use_geometry", values))
            return self.status()

    workbench = ProfileWorkbench()
    catalog = ToolCatalog(workbench, _goals(notes), RunResults(), explorer=LocalFileExplorer(), calibrator=FakeCalibrator())
    report = StandardPipeline(catalog, PipelineOptions(notes=notes)).run()
    assert _decisions(report, "geometry")[0]["decision"] == "from geometry.poni"
    assert not any("kept the instrument profile" in item["decision"] for item in report["decisions"])


def test_a_calibration_given_that_does_not_fit_is_flagged_and_the_search_goes_on(tmp_path: Path) -> None:
    wrong = touch(tmp_path / "other_detector.poni", POOR_AGBH.replace("0.0001,", "0.000172,"))  # 172 µm pixels
    catalog, _workbench, calibrator = _series(tmp_path)
    options = PipelineOptions(calibration=str(wrong), energy_kev=12.4, incidence_deg=0.2, pixel_size_um=100.0)
    report = StandardPipeline(catalog, options).run()

    assert report["ok"] and _decisions(report, "geometry")[-1]["decision"] == "calibrated from AgBH_00001.tif"
    assert _decisions(report, "calibration")[0] == {
        "what": "calibration", "decision": "skipped other_detector.poni",
        "why": "made for 172 µm pixels, the given pixel size is 100 µm"}
    [item] = report["needs_attention"]  # exit 2: the person's calibration was not the one used
    assert item["item"] == "calibration" and item["option"] == "calibration"
    assert "skipped other_detector.poni" in item["why"] and "AgBH_00001.tif instead" in item["why"]
    assert calibrator.calls == [("AgBH_00001.tif", "agbh", 12.4, None, pytest.approx(100e-6))]

    # Nothing else fits either: the open item still says why the person's calibration was not used.
    frame = touch(tmp_path / "alone" / "a" / "b" / "c" / "film_00001_m01.nxs", "x")  # beyond the search's reach
    catalog = ToolCatalog(SeriesWorkbench(frame), _goals(), RunResults(), explorer=LocalFileExplorer(), calibrator=FakeCalibrator())
    report = StandardPipeline(catalog, dataclasses.replace(options, follow_detection=True)).run()
    assert not report["ok"] and any(
        item["item"] == "calibration" and "The calibration given was not used (skipped other_detector.poni" in item["why"]
        for item in report["needs_attention"]), report["needs_attention"]
    assert _decisions(report, "technique") == [{"what": "technique", "decision": "none", "why": NO_GEOMETRY}]
    text = pipeline_markdown(report)
    assert text.startswith("# GIMaP — film_00001_m01.nxs") and "NEEDS INPUT" in text and "Scherrer" not in text

    # Find Calibration Automatically runs no procedure: no technique decision, the heading its report renames.
    catalog = ToolCatalog(SeriesWorkbench(frame), _goals(), RunResults(), explorer=LocalFileExplorer(), calibrator=FakeCalibrator())
    report = StandardPipeline(catalog, dataclasses.replace(options, follow_detection=True, stop_after_geometry=True)).run()
    assert _decisions(report, "technique") == [] and pipeline_markdown(report).startswith("# GIWAXS — ")


def test_the_one_calibrant_the_notes_name_is_the_standard(tmp_path: Path) -> None:
    notes = "12.4 keV, alpha_i = 0.2 deg, calibrated with silver behenate"
    catalog, _workbench, calibrator = _series(tmp_path / "a", notes, image="calib_00001.tif")  # the name names no standard
    report = StandardPipeline(catalog, PipelineOptions(notes=notes)).run()
    assert report["ok"] and calibrator.calls == [("calib_00001.tif", "agbh", 12.4, None, None)]
    assert _decisions(report, "calibration standard")[0]["why"] == "from the notes (the only calibrant they name)"

    # Without the calibrant that image is not tried (no standard to fit).
    catalog, _workbench, calibrator = _series(tmp_path / "b", "12.4 keV, alpha_i = 0.2 deg", image="calib_00001.tif")
    assert not StandardPipeline(catalog, PipelineOptions(notes="12.4 keV, alpha_i = 0.2 deg")).run()["ok"]
    assert calibrator.calls == []

    # Two calibrants: neither is taken; the file name decides.
    notes = "12.4 keV, alpha_i = 0.2 deg, AgBH and LaB6 measured"
    catalog, _workbench, calibrator = _series(tmp_path / "c", notes)
    report = StandardPipeline(catalog, PipelineOptions(notes=notes)).run()
    assert report["ok"] and calibrator.calls[0][1] == "agbh"
    assert _decisions(report, "calibration standard")[0] == {
        "what": "calibration standard", "decision": "none from the notes", "why": "the notes name 2: agbh, lab6"}

    # The notes and the file name disagree: every standard is compared.
    notes = "12.4 keV, alpha_i = 0.2 deg, LaB6 calibration"
    catalog, _workbench, calibrator = _series(tmp_path / "d", notes)
    report = StandardPipeline(catalog, PipelineOptions(notes=notes)).run()
    assert {call[1] for call in calibrator.calls} == {"agbh", "lab6", "ceo2", "lab6_ceo2"}
    assert _decisions(report, "calibration standard")[0]["decision"] == "compared (AgBH_00001.tif)"


def test_calibrants_are_read_from_notes_only_when_one_is_named() -> None:
    for text, expected in (
        ("AgBH calibration", ["agbh"]), ("silver behenate", ["agbh"]), ("AgBe_250403.poni", ["agbh"]),
        ("用AgBH标定", ["agbh"]), ("LaB6+CeO2 mixture", ["lab6_ceo2"]), ("CeO2/LaB6", ["lab6_ceo2"]),
        ("calibrated with LaB6; CeO2 too", ["lab6", "ceo2"]), ("AgBH and LaB6", ["agbh", "lab6"]),
        ("in lab 6 we used the labels", []), ("ceria", ["ceo2"]),
        ("standards: AgBH, LaB6, CeO2", ["agbh", "lab6", "ceo2"]),  # a list, not the mixture
        ("E:/bt/calib/LaB6_CeO2_0001.tif", ["lab6_ceo2"]),
    ):
        assert calibrants_in_notes(text) == expected, text
        assert standard_from_notes(text) == (expected[0] if len(expected) == 1 else None), text


class HeaderWorkbench(SeriesWorkbench):
    """A frame whose header states a pixel size."""

    def status(self) -> dict:
        return {**super().status(), "header": {"pixel_size_um": [172.0, 172.0]}}


class HeaderExplorer(LocalFileExplorer):
    """Calibration images whose header states a pixel size too."""

    def image_header(self, path):
        return {**super().image_header(path), "pixel_size_um": [172.0, 172.0]}


def test_a_pixel_size_given_beats_the_header_and_the_decision_names_both(tmp_path: Path) -> None:
    catalog, _workbench, calibrator = _series(tmp_path, explorer=HeaderExplorer(), workbench=HeaderWorkbench)
    options = PipelineOptions(energy_kev=12.4, incidence_deg=0.2, pixel_size_um=100.0)
    report = StandardPipeline(catalog, options).run()
    assert report["ok"]
    assert _decisions(report, "pixel size") == [{
        "what": "pixel size", "decision": "100 µm",
        "why": "given; the image header says 172 µm, the given value is used"}]
    assert calibrator.calls == [("AgBH_00001.tif", "agbh", 12.4, None, pytest.approx(100e-6))]

    # A profile made for other pixels is not kept when a pixel size is given.
    class Profile(AutoWorkbench):
        def status(self):
            return {**super().status(), "geometry": {"instrument_profile": "old", "incidence_deg": 0.2, "pixel_size_um": [172.0, 172.0]}}

    report = StandardPipeline(ToolCatalog(Profile(), _goals(), RunResults()), PipelineOptions(pixel_size_um=100.0)).run()
    assert _decisions(report, "geometry")[0] == {
        "what": "geometry", "decision": "did not keep the instrument profile 'old'",
        "why": "made for 172 µm pixels, the given pixel size is 100 µm"}


# -- real frames: the technique Auto detects, and (3) nothing written next to the data ---------------------------


def _snapshot(folder: Path) -> list[tuple]:
    return sorted((str(path.relative_to(folder)), path.stat().st_size, path.stat().st_mtime_ns) for path in folder.rglob("*"))


def test_a_gisaxs_frame_gets_the_gisaxs_procedure_and_exports_stay_out_of_the_data_folder(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import open_headless_session

    frame = DATA / GALAXI["frame"]
    before = _snapshot(frame.parent)
    session = open_headless_session(frame, saved_profiles=False, out_dir=tmp_path / "out", kind="mcp")
    try:
        outcome = session.call("run_standard_pipeline", {
            "calibration": str(frame.parent / "galaxi_bornagain.poni"), "incidence_deg": GALAXI["incidence_deg"], "fit": False,
        })
        report = json.loads(outcome.content)
        assert not outcome.is_error and report["procedure"] == "gisaxs", report["decisions"]
        assert _decisions(report, "technique")[0] == {"what": "technique", "decision": "GISAXS", "why": DETECTED}

        exported = session.call("export_results", {"include_tables": True})
        payload = json.loads(exported.content)
        folder = tmp_path / "out" / "gimap_analysis"
        assert not exported.is_error and payload["folder"] == str(folder) and "never next to the data" in payload["note"]
        assert payload["written"] and all(Path(path).parent == folder and Path(path).is_file() for path in payload["written"])
        assert any(path.endswith("_assistant.json") for path in payload["written"])  # the tables too
    finally:
        session.close()
    assert _snapshot(frame.parent) == before


def test_a_file_gimap_cannot_read_fails_with_the_reason(tmp_path: Path, capsys) -> None:
    from src.gimap.app.headless_assistant import analyse_frames
    from tools import gimap_agent

    broken = tmp_path / "data" / "broken_0002.tif"
    broken.parent.mkdir()
    broken.write_text("not a tiff", encoding="utf-8")
    [report] = analyse_frames([str(broken)], PipelineOptions(), tmp_path / "out", saved_profiles=False)
    assert not report["ok"] and "not a detector image GIMaP can read" in report["failed"]
    [item] = report["needs_attention"]
    assert item["item"] == "frame" and item["why"] == report["failed"]
    text = (tmp_path / "out" / "broken_0002" / "report.md").read_text(encoding="utf-8")
    assert text.startswith("# GIMaP — broken_0002.tif") and "Status: **FAILED**" in text
    assert gimap_agent.exit_code([report]) == 1
    assert "failed: " in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")

    assert gimap_agent.main(["status", str(broken), "--no-saved-profiles"]) == 1
    assert "not a detector image GIMaP can read" in json.loads(capsys.readouterr().out)["message"]
    steps = tmp_path / "steps.json"
    steps.write_text(json.dumps([{"tool": "get_status", "args": {}}]), encoding="utf-8")
    assert gimap_agent.main(["call", str(broken), str(steps), "--no-saved-profiles"]) == 1
    assert sorted(path.name for path in broken.parent.iterdir()) == ["broken_0002.tif"]  # nothing written there


def test_the_public_giwaxs_frame_still_runs_giwaxs_without_a_technique(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import analyse_frames

    folder = DATA / "giwaxs_p08_mapi"
    before = _snapshot(folder)
    options = PipelineOptions(calibration=str(folder / "LaB6_2021_12_DESY_P08.poni"), incidence_deg=0.075)
    [report] = analyse_frames([str(folder / "S121_MAI_A2_00841.tif")], options, tmp_path, saved_profiles=False)
    assert report["ok"] and report["procedure"] == "giwaxs" and report["needs_attention"] == []
    assert _decisions(report, "technique")[0]["why"] == DETECTED
    assert (tmp_path / "S121_MAI_A2_00841" / "report.md").is_file()
    assert _snapshot(folder) == before
