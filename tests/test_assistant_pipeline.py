"""The standard GIWAXS pipeline: decisions in code, open questions named, same tools as the GUI."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from src.gimap.features.assistant.application import (
    GOALS,
    PERMISSION_AUTO,
    PERMISSION_CONFIRM,
    AnalysisGoals,
    PipelineOptions,
    RunResults,
    StandardPipeline,
    ToolCatalog,
    ToolOutcome,
    batch_markdown,
    pipeline_markdown,
)
from src.gimap.features.assistant.infrastructure import LocalFileExplorer, McpDispatcher, serve_stdio
from tests.assistant_fakes import FakeWorkbench, call
from tests.test_assistant_calibration import (
    ENERGY_KEV,
    POOR_AGBH,
    CalibrationWorkbench,
    Confirmer,
    FakeCalibrator,
    agbh_image,
    giwaxs_frame,
    save_tiff,
    touch,
)


class SeriesWorkbench(CalibrationWorkbench):
    """A series of 257 frames without geometry until ``use_geometry``."""

    def __init__(self, frame: Path, frames: int = 257):
        super().__init__(frame)
        self.frames, self.first, self.summed = frames, 1, 1

    def status(self) -> dict:
        status = super().status()
        status.update(frames=self.frames, frame=self.first, summed_frames=self.summed)
        return status

    def set_frame(self, frame_number, sum_count):
        self.first, self.summed = frame_number, sum_count
        return self._record("set_frame", frame_number, sum_count)


def setup(tmp_path: Path, *, notes: str = "", frames: int = 257):
    root = tmp_path / "beamtime" / "raw"
    frame = touch(root / "film" / "film_00001_m01.nxs", "x")  # the fake workbench never reads it
    save_tiff(root / "calib" / "AgBH_00001.tif", agbh_image())
    workbench = SeriesWorkbench(frame, frames)
    calibrator = FakeCalibrator()
    catalog = ToolCatalog(
        workbench, AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO, instructions=notes), RunResults(),
        explorer=LocalFileExplorer(), calibrator=calibrator,
    )
    return catalog, workbench, calibrator


def decision(report: dict, what: str) -> dict:
    return next(item for item in report["decisions"] if item["what"] == what)


def test_the_pipeline_finds_a_calibration_and_says_why_it_did_everything(tmp_path: Path) -> None:
    catalog, workbench, calibrator = setup(tmp_path, notes="P03 GIWAXS, alpha_i = 0.2 deg")
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, notes="P03 GIWAXS, alpha_i = 0.2 deg")).run()

    assert report["ok"] and report["needs_attention"] == []
    assert calibrator.calls == [("AgBH_00001.tif", "agbh", 12.4, None, None)]
    assert decision(report, "geometry")["decision"] == "calibrated from AgBH_00001.tif"
    assert decision(report, "geometry")["why"].startswith("good")
    assert decision(report, "incidence angle")["why"] == "from the notes"
    used = next(item for item in workbench.calls if item[0] == "use_geometry")
    assert used[1]["incidence_deg"] == pytest.approx(0.2)
    assert ("set_frame", 248, 10) in workbench.calls  # the last ten frames: the final state
    assert "the last frames show the final state" in decision(report, "frames")["why"]
    found = [peak["q"] for peak in report["peaks"]]
    for expected in (0.40, 0.80, 1.20, 1.65):
        assert min(abs(q - expected) for q in found) < 0.01, found
    assert len(report["rings"]) == 3 and report["calibration_quality"]["assessment"].startswith("good")
    assert [step["tool"] for step in report["steps"]][:6] == [
        "get_status", "find_calibration_files", "inspect_file", "calibrate_geometry", "use_geometry", "set_frame",
    ]


@pytest.mark.parametrize("frames, first, count", [
    (403, 394, 10), (50, 41, 10), (49, 41, 9), (10, 9, 2), (9, 9, 1), (4, 4, 1), (2, 2, 1),
])
def test_the_frames_summed_by_default_are_at_most_the_last_fifth_of_the_series(tmp_path: Path, frames, first, count) -> None:
    """A short series is not summed whole and called its final state: frame 1 of ten is the start."""
    catalog, workbench, _calibrator = setup(tmp_path, frames=frames)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4)).run()
    assert report["ok"] and ("set_frame", first, count) in workbench.calls
    assert report["frames"] == {"total": frames, "first": first, "summed": count}
    made = decision(report, "frames")
    assert made["why"].startswith(f"a series of {frames} frames (an in-situ run or a scan): ")
    if count == 1:
        assert made["decision"] == f"frame {first} of {frames}"
        assert "the last frame shows the final state" in made["why"] and "not summed" in made["why"]
        assert "sum_frames" in made["why"] and "--sum" in made["why"]  # how to sum, for a series that does not change
    else:
        assert made["decision"] == f"frames {first}–{frames} summed of {frames}"
        assert "the last frames show the final state" in made["why"]
    if 1 < count < 10:
        assert "the last fifth" in made["why"]


def test_a_block_from_the_start_that_covers_the_series_says_so(tmp_path: Path) -> None:
    """``frame=1, sum_frames=10`` on nine frames is the whole series (what "compare with the start" asked
    for before): the decision says so instead of only "as asked"; a start block without a count is the
    first fifth, the mirror of the final state."""
    catalog, workbench, _calibrator = setup(tmp_path, frames=9)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, frame=1, sum_frames=10)).run()
    assert ("set_frame", 1, 9) in workbench.calls
    made = decision(report, "frames")
    assert made["decision"] == "frames 1–9 summed of 9"
    assert made["why"] == "as asked: all 9 frames are summed, so this is the whole series, not one state of it"

    for frames, count in ((9, 1), (12, 2), (240, 10)):  # the start as the end is taken: the first fifth, at most ten
        catalog, workbench, _calibrator = setup(tmp_path / str(frames), frames=frames)
        report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, frame=1)).run()
        assert ("set_frame", 1, count) in workbench.calls and report["frames"]["summed"] == count


def test_frames_that_were_asked_for_are_used_and_a_whole_series_is_not_called_its_end(tmp_path: Path) -> None:
    catalog, workbench, _calibrator = setup(tmp_path, frames=10)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, sum_frames=10)).run()
    assert ("set_frame", 1, 10) in workbench.calls
    why = decision(report, "frames")["why"]
    assert why == "as asked: all 10 frames are summed, so this is the whole series, not its final state"

    catalog, workbench, _calibrator = setup(tmp_path / "three", frames=10)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, sum_frames=3)).run()
    assert ("set_frame", 8, 3) in workbench.calls
    assert decision(report, "frames")["why"] == "as asked: the last 3 of 10 frames are summed"

    catalog, workbench, _calibrator = setup(tmp_path / "one", frames=10)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, frame=2)).run()
    assert ("set_frame", 2, 2) in workbench.calls  # a frame without a count: the default count from that frame
    assert decision(report, "frames")["why"] == (
        "the frame as asked; 2 frames are summed from it by default (sum_frames=1 or --sum 1: that frame alone)"
    )
    catalog, workbench, _calibrator = setup(tmp_path / "both", frames=10)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, frame=2, sum_frames=1)).run()
    assert ("set_frame", 2, 1) in workbench.calls and decision(report, "frames")["why"] == "as asked"

    catalog, workbench, _calibrator = setup(tmp_path / "single", frames=1)
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4)).run()
    assert not any(step[0] == "set_frame" for step in workbench.calls)
    assert not any(item["what"] == "frames" for item in report["decisions"])


def test_the_pipeline_names_what_only_a_person_can_answer(tmp_path: Path) -> None:
    catalog, _workbench, calibrator = setup(tmp_path)
    stopped = StandardPipeline(catalog).run()
    assert not stopped["ok"] and calibrator.calls == []
    energy = next(item for item in stopped["needs_attention"] if item["item"] == "X-ray energy")
    assert energy["option"] == "energy_kev"

    catalog, _workbench, _calibrator = setup(tmp_path / "second")
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4)).run()
    assert report["ok"]
    question = next(item for item in report["needs_attention"] if item["option"] == "incidence_deg")
    assert "0°" in question["why"]
    text = pipeline_markdown(report)
    assert text.index("Needs attention") < text.index("Geometry") and "--incidence-deg" in text
    assert "NEEDS INPUT" in text


def test_the_baseline_is_an_optional_tool_and_every_other_tool_stays_available(tmp_path: Path) -> None:
    root = tmp_path / "beamtime" / "raw"
    frame = touch(root / "film" / "film_00001_m01.nxs", "x")
    save_tiff(root / "calib" / "AgBH_00001.tif", agbh_image())
    workbench = SeriesWorkbench(frame)
    goals = AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO, instructions="alpha_i = 0.2 deg, 12.4 keV", ring_q=0.8)
    catalog = ToolCatalog(workbench, goals, RunResults(), explorer=LocalFileExplorer(), calibrator=FakeCalibrator())
    assert "run_standard_pipeline" in [tool["name"] for tool in catalog.definitions()]

    outcome = catalog.execute(call("run_standard_pipeline"))
    baseline = json.loads(outcome.content)
    assert not outcome.is_error and outcome.summary.startswith("baseline done")
    assert baseline["ok"] and baseline["needs_attention"] == [] and "tables" not in baseline
    assert decision(baseline, "requested ring")["decision"] == "q ≈ 0.8 Å⁻¹"  # the user's ring, though not among the strongest
    assert any(abs(ring["q"] - 0.8) < 0.05 for ring in baseline["rings"])
    assert catalog.results.pipeline["tables"]["peaks"]  # the full tables stay in the results
    # Its choices are defaults: the start of the series is one call away.
    assert not catalog.execute(call("set_frame", frame=1, sum=10)).is_error and workbench.calls[-1] == ("set_frame", 1, 10)
    assert not catalog.execute(call("find_peaks", curve="out_of_plane")).is_error


def test_a_declined_geometry_stops_the_baseline_without_asking_again(tmp_path: Path) -> None:
    root = tmp_path / "beamtime" / "raw"
    frame = touch(root / "film" / "film_00001_m01.nxs", "x")
    save_tiff(root / "calib" / "AgBH_00001.tif", agbh_image())
    save_tiff(root / "calib" / "AgBH_00002.tif", agbh_image(seed=4))
    confirmer = Confirmer(answer=False)
    catalog = ToolCatalog(
        SeriesWorkbench(frame), AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_CONFIRM, instructions="12.4 keV"),
        RunResults(), explorer=LocalFileExplorer(), calibrator=FakeCalibrator(), confirmer=confirmer,
    )
    baseline = json.loads(catalog.execute(call("run_standard_pipeline")).content)
    assert not baseline["ok"] and len(confirmer.questions) == 1
    assert [item["item"] for item in baseline["needs_attention"]] == ["geometry"]


def test_a_saved_poni_is_used_before_fitting_images_and_the_frames_energy_wins(tmp_path: Path) -> None:
    catalog, workbench, calibrator = setup(tmp_path)
    touch(tmp_path / "beamtime" / "processed" / "agbh.poni", POOR_AGBH)  # 1 Å = 12.4 keV, 100 mm, 100 µm
    report = StandardPipeline(catalog, PipelineOptions(energy_kev=10.0, incidence_deg=0.2)).run()

    assert report["ok"] and calibrator.calls == []
    assert decision(report, "geometry")["decision"] == "from agbh.poni"
    values = next(item for item in workbench.calls if item[0] == "use_geometry")[1]
    assert values["distance_mm"] == pytest.approx(100.0) and values["wavelength_angstrom"] == pytest.approx(12.398419843320026 / 10.0)
    assert report["calibration_quality"]["assessment"].startswith("a saved calibration file")


def test_a_saved_instrument_profile_is_kept(tmp_path: Path) -> None:
    workbench = FakeWorkbench()
    catalog = ToolCatalog(workbench, AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO), RunResults())
    report = StandardPipeline(catalog).run()
    assert report["ok"] and "kept the instrument profile" in decision(report, "geometry")["decision"]
    assert not any(step["tool"] == "find_calibration_files" for step in report["steps"])
    assert report["needs_attention"] == []  # the profile has αi


def test_set_frame_counts_from_the_end_and_never_sums_past_the_last_frame(tmp_path: Path) -> None:
    catalog, workbench, _calibrator = setup(tmp_path, frames=257)
    for arguments, expected in (
        ({"frame": -1, "sum": 10}, (248, 10)),
        ({"frame": 254, "sum": 10}, (248, 10)),
        ({"frame": 1}, (1, 1)),
        ({"frame": 0, "sum": 5000}, (1, 257)),
        ({"frame": -3}, (255, 1)),
    ):
        outcome = catalog.execute(call("set_frame", **arguments))
        assert not outcome.is_error and workbench.calls[-1] == ("set_frame", *expected), (arguments, outcome.summary)
    assert outcome.summary == "frame 255 of 257"


def test_a_batch_summary_lists_lines_every_frame_shares(tmp_path: Path) -> None:
    reports = []
    for name in ("a", "b"):
        catalog, _workbench, _calibrator = setup(tmp_path / name)
        reports.append(StandardPipeline(catalog, PipelineOptions(energy_kev=12.4, incidence_deg=0.2)).run())
    text = batch_markdown(reports)
    assert "| film_00001_m01.nxs | OK | 248–257 of 257 |" in text
    assert "Lines at the same q" in text and "0.4" in text


def test_mcp_over_stdio_answers_line_by_line() -> None:
    seen = []

    def call_tool(name, arguments):
        seen.append((name, arguments))
        return ToolOutcome(json.dumps({"echo": arguments}), "ok")

    tools = [{"name": "echo", "description": "Echo.", "input_schema": {"type": "object"}}]
    dispatcher = McpDispatcher(tools, call_tool, instructions="Use echo.")
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "echo", "arguments": {"q": "1.65 Å⁻¹"}}},
    ]
    reader = io.BytesIO(b"".join(json.dumps(item, ensure_ascii=False).encode("utf-8") + b"\n" for item in messages) + b"not json\n")
    writer = io.BytesIO()
    serve_stdio(dispatcher, reader, writer)
    answers = [json.loads(line) for line in writer.getvalue().decode("utf-8").splitlines()]
    assert [answer.get("id") for answer in answers] == [1, 2, 3, None]
    assert answers[0]["result"]["instructions"] == "Use echo." and answers[1]["result"]["tools"][0]["name"] == "echo"
    assert json.loads(answers[2]["result"]["content"][0]["text"]) == {"echo": {"q": "1.65 Å⁻¹"}}
    assert answers[3]["error"]["code"] == -32700 and seen == [("echo", {"q": "1.65 Å⁻¹"})]


def test_the_command_line_pipeline_analyses_real_images_end_to_end(tmp_path: Path) -> None:
    from src.gimap.app.headless_assistant import analyse_frames, serve_mcp_stdio

    root = tmp_path / "beamtime" / "raw"
    frame = save_tiff(root / "P3HT_film" / "P3HT_00012.tif", giwaxs_frame())
    save_tiff(root / "calib" / "AgBH_00001.tif", agbh_image())
    notes = f"Energy: {ENERGY_KEV:.4f} keV, alpha_i = 0.2 deg, pixel size 100 um"
    out = tmp_path / "out"
    # The synthetic detector reaches only about 18° of 2θ, which Auto reads as small-angle: ask for GIWAXS.
    reports = analyse_frames([str(frame)], PipelineOptions(notes=notes, technique="giwaxs"), out, saved_profiles=False)

    report = reports[0]
    assert report["ok"] and report["needs_attention"] == [], report["needs_attention"]
    assert decision(report, "geometry")["decision"] == "calibrated from AgBH_00001.tif"
    assert decision(report, "technique")["why"].startswith("forced")
    assert report["measurement"] == "giwaxs" and report["geometry"]["incidence_deg"] == pytest.approx(0.2)
    found = [peak["q"] for peak in report["peaks"]]
    for expected in (0.40, 0.80, 1.10):
        assert min(abs(q - expected) for q in found) < 0.02, found
    folder = out / "P3HT_00012"
    for name in ("report.md", "report.json", "radial.csv", "qmap.png"):
        assert (folder / name).is_file(), name
    assert (out / "summary.md").is_file() and "P3HT_00012.tif | OK" in (out / "summary.md").read_text(encoding="utf-8")

    # The same tools over MCP stdio, as Codex or Claude Code would call them.
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "find_peaks", "arguments": {"curve": "radial"}}},
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "open_frame", "arguments": {"path": str(frame), "notes": notes}}},
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {
            "name": "run_standard_pipeline", "arguments": {"out_dir": str(tmp_path / "mcp_out"), "technique": "giwaxs"}}},
        # The baseline is a start, not the end: the other tools stay available on the same frame.
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "ring_orientation", "arguments": {"q_center": 1.1}}},
    ]
    reader = io.BytesIO(b"".join(json.dumps(item).encode("utf-8") + b"\n" for item in requests))
    writer = io.BytesIO()
    serve_mcp_stdio(saved_profiles=False, reader=reader, writer=writer)
    answers = {answer["id"]: answer for answer in map(json.loads, writer.getvalue().decode("utf-8").splitlines())}
    tools = {tool["name"]: tool for tool in answers[2]["result"]["tools"]}
    assert next(iter(tools)) == "open_frame" and "submit_report" not in tools
    assert "out_dir" in tools["run_standard_pipeline"]["inputSchema"]["properties"]
    assert answers[3]["result"]["isError"] and "open_frame" in answers[3]["result"]["content"][0]["text"]
    assert not answers[4]["result"]["isError"]
    pipeline = json.loads(answers[5]["result"]["content"][0]["text"])
    assert pipeline["ok"] and "tables" not in pipeline and pipeline["needs_attention"] == []
    assert (tmp_path / "mcp_out" / "report.md").is_file() and len(pipeline["outputs"]) >= 4
    ring = json.loads(answers[6]["result"]["content"][0]["text"])
    assert not answers[6]["result"]["isError"] and ring["coverage"] > 0.5
