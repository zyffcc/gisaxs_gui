"""Checks of the assistant area's cross-area work (review of the review).

A Clear while the automatic analysis waits for a re-analysis must end the run at once: Analyze drops
that re-analysis, so its answer never comes, and the run would otherwise hold Run Automatic Analysis
(and Analyze's own Run buttons) for the bridge's whole timeout. And a status that names no frame any
more (Analyze cleared) ends a run as a changed frame, like one that names another file.
"""

from __future__ import annotations

import time

from src.gimap.features.assistant.application import (
    GOALS,
    PERMISSION_AUTO,
    AnalysisGoals,
    PipelineOptions,
    RunResults,
    StandardPipeline,
    ToolCatalog,
)
from src.gimap.features.assistant.presentation import GuidedAnalysis
from src.gimap.features.assistant.presentation import guided_analysis as guided_module
from tests.assistant_fakes import FakeWorkbench
from tests.test_assistant_gui import _wait, analyze  # noqa: F401 - the fixture


def _goals() -> AnalysisGoals:
    return AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO)


def test_a_clear_while_the_run_waits_for_a_reanalysis_ends_the_run_at_once(analyze, monkeypatch) -> None:  # noqa: F811
    _window, page, _context = analyze
    automation = page.automation()
    waiting: list = []
    # A re-analysis that Analyze's Clear discards: its answer (``done``) never comes.
    monkeypatch.setattr(automation, "set_sector_widths", lambda _a, _b, done: waiting.append(done))
    outcome: dict = {}

    class Pipeline:  # one step: the re-analysis the Clear interrupts
        def __init__(self, catalog, options, progress, *, stop=None, events=None):
            self.catalog = catalog

        def run(self) -> dict:
            path = self.catalog.workbench.status()["path"]
            try:
                self.catalog.workbench.set_sector_widths(25.0, 25.0)
            except Exception as exc:
                outcome["error"] = type(exc).__name__
            return {"ok": False, "frame": path, "needs_attention": [], "decisions": [], "steps": []}

    monkeypatch.setattr(guided_module, "StandardPipeline", Pipeline)
    guided = GuidedAnalysis(page.automation)
    page.filesCleared.connect(guided.files_cleared)
    events: list[str] = []
    guided.started.connect(lambda _text: events.append("started"))
    guided.finished.connect(lambda _report: events.append("finished"))
    guided.failed.connect(lambda _message: events.append("failed"))

    guided.run()
    _wait(lambda: bool(waiting), 30)
    page.clear_files()
    cleared = time.monotonic()
    _wait(lambda: not guided._busy(), 15)  # not the bridge's 600 s

    assert time.monotonic() - cleared < 10 and outcome == {"error": "RunCancelled"}
    assert events == ["started", "finished"]  # Analyze hears the end and re-enables its buttons
    assert guided.report is None and guided.run_button.isEnabled()  # the run's results went with the Clear
    guided.results.deleteLater()


class _ClearedWorkbench(FakeWorkbench):
    """Analyze is cleared after the run read the frame: later statuses name no file. No reduction yet, so
    the run switches the mode, and that step's status is the one that names no file."""

    def __init__(self) -> None:
        super().__init__(measurement=None)
        self.statuses = 0

    def status(self) -> dict:
        self.statuses += 1
        status = super().status()
        return status if self.statuses == 1 else {**status, "file": None, "path": None}


def test_a_status_that_names_no_frame_any_more_ends_the_run() -> None:
    report = StandardPipeline(ToolCatalog(_ClearedWorkbench(), _goals(), RunResults()), PipelineOptions()).run()
    assert not report["ok"] and report["failed"].endswith("Analyze now shows no frame"), report["failed"]
    assert report["frame"] == "C:/data/synthetic.tif" and [step["tool"] for step in report["steps"]] == [
        "get_status", "set_measurement_mode"]  # nothing more was done
    # New statuses of the same file (a new dict each call) do not end a run.
    report = StandardPipeline(ToolCatalog(FakeWorkbench(), _goals(), RunResults()), PipelineOptions()).run()
    assert report["ok"] and report["failed"] is None and report["frame"] == "C:/data/synthetic.tif"
