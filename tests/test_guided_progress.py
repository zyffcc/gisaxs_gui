"""The automatic analysis says what it is doing, can be stopped between steps, and keeps what it found."""

from __future__ import annotations

import threading
from pathlib import Path

from src.gimap.features.assistant.application import PipelineOptions, StandardPipeline, step_text
from tests.test_assistant_pipeline import setup


def test_a_stop_ends_the_run_before_the_next_step_and_keeps_the_report(tmp_path: Path) -> None:
    catalog, _workbench, _calibrator = setup(tmp_path, notes="P03 GIWAXS, alpha_i = 0.2 deg")
    stop, events = threading.Event(), []

    def seen(event: dict) -> None:
        events.append(event)
        if event["state"] == "done" and event["tool"] == "calibrate_geometry":
            stop.set()  # pressed while the (slow) geometry fit was running: it finishes, nothing after it starts

    report = StandardPipeline(
        catalog, PipelineOptions(energy_kev=12.4, notes="P03 GIWAXS, alpha_i = 0.2 deg"), stop=stop, events=seen,
    ).run()
    tools = [step["tool"] for step in report["steps"]]
    assert report["stopped"] and report["stopped"] not in tools and not report["ok"]
    assert tools[-1] == "calibrate_geometry" and "find_peaks" not in tools
    assert report["tables"]["calibrations"]  # what was found (the fitted geometry) is kept in the report
    assert any(item["what"] == "stopped" for item in report["decisions"])
    starts = [event for event in events if event["state"] == "start"]
    assert len(starts) == len(tools) and all("seconds" in event for event in events if event["state"] == "done")
    full = StandardPipeline(catalog, PipelineOptions(energy_kev=12.4)).run()
    assert full["stopped"] is None


def test_step_texts_are_sentences() -> None:
    assert step_text("ring_orientation", {"q_center": 1.2345}) == "Orientation of the ring at q = 1.234 Å⁻¹"
    assert step_text("ring_orientation") == "Orientation of the ring"
    assert step_text("unknown_tool") == "Unknown tool"


def test_the_panel_follows_the_run_and_offers_to_keep_or_discard(tmp_path: Path) -> None:
    from src.gimap.features.assistant.presentation.guided_progress import GuidedProgressPanel
    from tests.test_analyze_workspace import _app

    _app()
    panel = GuidedProgressPanel()
    panel.start("Working…")
    assert not panel.isHidden() and panel.stop_button.isEnabled()
    panel.step({"state": "start", "tool": "get_status", "arguments": {}})
    panel.step({"state": "done", "tool": "get_status", "seconds": 0.2})
    panel.step({"state": "start", "tool": "calibrate_geometry", "arguments": {}})
    assert "Fitting the geometry" in panel.now_label.text()
    assert "✓" in panel.phases_label.text() and "Geometry" in panel.phases_label.text()
    stops = []
    panel.stopRequested.connect(lambda: stops.append(1))
    panel.stop_button.click()
    assert stops and not panel.stop_button.isEnabled()
    assert "waiting for" in panel.now_label.text() and "minute" in panel.now_label.text()  # a slow step: it says so
    panel.step({"state": "done", "tool": "calibrate_geometry", "seconds": 30.0})
    panel.finish({"stopped": "set_frame", "steps": []})
    assert "Stopped after 2 steps" in panel.now_label.text() and not panel.after_row.isHidden()
    discarded = []
    panel.discardRequested.connect(lambda: discarded.append(1))
    panel.discard_button.click()
    assert discarded
