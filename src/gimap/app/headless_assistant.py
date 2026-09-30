"""GIMaP's assistant tools without a window, for command-line agents and scripts.

The Analyze page runs offscreen, so every number is the one the GUI shows and
the tools are exactly those of Process with Claude.  The person's GIMaP data
folder is only read: instrument profiles saved there are used (a detector they
calibrated keeps its geometry), new ones live in memory, and settings are the
defaults so a run is reproducible.  Outputs go where the caller says, never
next to the data unless a tool is asked to export there.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..features.assistant.application import (
    GOALS,
    PERMISSION_AUTO,
    AnalysisGoals,
    Chooser,
    PipelineOptions,
    RunResults,
    StandardPipeline,
    ToolCall,
    ToolCatalog,
    ToolOutcome,
    batch_markdown,
    fit_curve_table,
    fit_solutions_csv,
    pipeline_markdown,
    ring_overlays,
    tool_specs,
)

LOAD_TIMEOUT_S = 900.0
"""Reading a large NeXus series (a few GB) can take minutes."""
CURVES = ("radial", "in_plane", "out_of_plane", "azimuthal", "horizontal", "vertical")
PLAYBOOK = "docs/agents/giwaxs-playbook.md"


_APPLICATION: list = []
"""Keeps the QApplication alive: one that is garbage-collected takes every widget with it."""


def _application():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    application = QApplication.instance() or QApplication([])
    if not _APPLICATION:
        _APPLICATION.append(application)
    return application


def _saved_profiles(data_dir: Optional[Path]) -> list:
    from ..integrations.state import JsonInstrumentProfileRepository, user_data_dir
    from ..integrations.state.user_store import PROFILES_FILE

    folder = Path(data_dir) if data_dir is not None else user_data_dir()
    try:
        return JsonInstrumentProfileRepository(folder / PROFILES_FILE).load_all()
    except (OSError, ValueError):
        return []


def default_output_folder() -> Path:
    import time

    from ..integrations.state import user_data_dir

    return user_data_dir() / "assistant_runs" / "cli" / time.strftime("%Y%m%d-%H%M%S")


@dataclass
class HeadlessSession:
    """One frame open in an offscreen Analyze page, with the assistant's tools on it."""

    page: Any
    catalog: ToolCatalog
    results: RunResults
    notes: str = ""
    """The person's notes given when the frame was opened; the pipeline reads αi, energy and pixel size from them."""
    calibration: Optional[dict] = None
    """The calibration the last pipeline run used (full precision)."""
    _calls: int = field(default=0, repr=False)

    def call(self, name: str, arguments: Optional[dict] = None) -> ToolOutcome:
        self._calls += 1
        return self.catalog.execute(ToolCall(f"cli{self._calls}", name, arguments or {}))

    def run_pipeline(self, options: PipelineOptions = PipelineOptions(), progress=lambda _text: None) -> dict:
        if not options.notes and self.notes:
            options = dataclasses.replace(options, notes=self.notes)
        pipeline = StandardPipeline(self.catalog, options, progress)
        report = pipeline.run()
        self.calibration = pipeline.calibration
        return report

    def curve(self, key: str) -> Optional[dict]:
        return self.page.automation().curve(key)

    def preview_png(self, max_size: int = 1100, rings=()) -> Optional[bytes]:
        return self.page.automation().preview_png(max_size, rings=rings)

    def close(self) -> None:
        self.page.tasks.wait(60.0)
        self.page.dispose()


def open_headless_session(
    frame: str | Path,
    *,
    notes: str = "",
    saved_profiles: bool = True,
    data_dir: Optional[Path] = None,
    chooser: Optional[Chooser] = None,
    load_timeout_s: float = LOAD_TIMEOUT_S,
) -> HeadlessSession:
    """Open ``frame`` offscreen; tools may read calibration material around it (read-only)."""
    _application()
    from ..features.analyze.bootstrap import create_analyze_view_model
    from ..features.analyze.presentation.page import AnalyzePage
    from ..features.assistant.infrastructure import LocalFileExplorer
    from ..features.assistant.presentation import GuiBridge, GuiWorkbench
    from ..features.calibration.bootstrap import create_headless_calibration
    from ..features.fitting.bootstrap import create_quick_fit
    from ..integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
        InMemoryUserPreferencesRepository,
    )
    from .context import AppContext

    path = Path(frame)
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    context = AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        instrument_profiles=InMemoryInstrumentProfileRepository(_saved_profiles(data_dir) if saved_profiles else []),
    )
    page = AnalyzePage(create_analyze_view_model(context))
    page.add_paths([str(path)])
    page.tasks.wait(load_timeout_s)
    results = RunResults()
    catalog = ToolCatalog(
        GuiWorkbench(page.automation(), GuiBridge()),
        AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO, instructions=notes),
        results,
        explorer=LocalFileExplorer(),
        calibrator=create_headless_calibration(),
        chooser=chooser,
        fitter=create_quick_fit(),
    )
    return HeadlessSession(page, catalog, results, notes=notes)


def write_outputs(session: HeadlessSession, report: dict, folder: Path) -> list[Path]:
    """report.json, report.md, the curves as CSV (GISAXS: the cuts and the fit) and the q-map as PNG."""
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    written = [folder / "report.json", folder / "report.md"]
    written[0].write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    written[1].write_text(pipeline_markdown(report), encoding="utf-8")
    for key in CURVES:
        curve = session.curve(key)
        if curve is None or len(curve["x"]) == 0:
            continue
        name = "azimuthal_last_ring.csv" if key == "azimuthal" else f"{key}.csv"
        with open(folder / name, "w", encoding="utf-8", newline="\n") as stream:
            np.savetxt(
                stream, np.column_stack([curve["x"], curve["y"], curve["sigma"]]), delimiter=",",
                header=f"{curve['x_label']}, I, sigma", comments="# ",
            )
        written.append(folder / name)
    written += _write_fit(((report.get("gisaxs") or {}).get("fit")), folder)
    png = session.preview_png(1100, rings=ring_overlays(report))
    if png:
        (folder / "qmap.png").write_bytes(png)
        written.append(folder / "qmap.png")
    return written


def _write_fit(fit: Optional[dict], folder: Path) -> list[Path]:
    """GISAXS: the fitted curve (data and best fit) and the solutions table."""
    import numpy as np

    if not fit:
        return []
    written = []
    header, columns = fit_curve_table(fit)
    if len(columns):
        written.append(folder / "fit_curve.csv")
        with open(written[-1], "w", encoding="utf-8", newline="\n") as stream:
            np.savetxt(stream, columns, delimiter=",", header=header, comments="# ")
    if fit.get("solutions"):
        written.append(folder / "fit_solutions.csv")
        written[-1].write_text(fit_solutions_csv(fit), encoding="utf-8")
    return written


def analyse_frames(
    frames: list[str],
    options: PipelineOptions,
    out: Path,
    *,
    progress: Callable[[str], None] = lambda _text: None,
    saved_profiles: bool = True,
    data_dir: Optional[Path] = None,
) -> list[dict]:
    """The standard pipeline on each frame; later frames of the same detector reuse the first calibration."""
    reports: list[dict] = []
    calibration: Optional[dict] = None
    for frame in frames:
        progress(f"== {frame}")
        try:
            session = open_headless_session(frame, notes=options.notes, saved_profiles=saved_profiles, data_dir=data_dir)
        except Exception as exc:  # one unreadable frame must not stop the batch
            reports.append({"ok": False, "frame": str(frame), "error": str(exc) or type(exc).__name__, "needs_attention": []})
            continue
        try:
            shape = (session.call("get_status").data or {}).get("shape")
            frame_options = options
            if calibration is not None and options.calibration is None and list(calibration.get("shape") or []) == list(shape or []):
                frame_options = dataclasses.replace(options, geometry=calibration)
            report = session.run_pipeline(frame_options, progress)
            calibration = session.calibration or calibration
            folder = out / Path(frame).stem
            report["outputs"] = [str(path) for path in write_outputs(session, report, folder)]
            reports.append(report)
        finally:
            session.close()
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(batch_markdown(reports), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps([{key: value for key, value in report.items() if key != "tables"} for report in reports],
                   indent=1, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return reports


# -- MCP over stdio -----------------------------------------------------------------------

MCP_INSTRUCTIONS = (
    "GIMaP GIWAXS tools. Start with open_frame(path, notes). run_standard_pipeline gives a baseline in "
    "one call (geometry, frames, peaks, sectors, rings, sizes, each decision with its reason, and "
    "needs_attention: values only the notes or the person know, with the option that supplies them). "
    "The baseline's choices are defaults: investigate further with the other tools when the question "
    "needs it — other frames of a series, lines every sample shares, peaks it skipped, custom sectors. "
    "Measured values come from the tools; show how derived values follow from them; mark "
    "interpretations as hypotheses; never write next to the data unless asked. "
    f"Guide: {PLAYBOOK} in the GIMaP repository."
)
OPEN_FRAME = {
    "name": "open_frame",
    "description": (
        "Open a detector image (a NeXus module file opens the whole series) offscreen in GIMaP's Analyze. "
        "notes: the person's beamtime notes (paths, αi, energy, calibrant) — paths named there become "
        "readable, and the baseline reads αi, the energy and the pixel size from them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}, "notes": {"type": ["string", "null"]}},
        "required": ["path"],
        "additionalProperties": False,
    },
}
OUT_DIR = {"type": ["string", "null"], "description": "Also write report.md/json, CSV curves and the q map here."}


def mcp_tools() -> list[dict]:
    """open_frame, then the assistant's tools; run_standard_pipeline can also write its files."""
    tools = [OPEN_FRAME]
    for spec in tool_specs(allow_images=False):
        if spec.name == "submit_report":  # the GUI's report; an agent here writes its own
            continue
        definition = spec.definition()
        if spec.name == "run_standard_pipeline":
            schema = dict(definition["input_schema"])
            schema["properties"] = {**schema["properties"], "out_dir": OUT_DIR}
            definition = {**definition, "input_schema": schema}
        tools.append(definition)
    return tools


def serve_mcp_stdio(*, saved_profiles: bool = True, data_dir: Optional[Path] = None, reader=None, writer=None) -> None:
    """Serve GIMaP's tools over stdio until the client closes the input."""
    from ..features.assistant.infrastructure import McpDispatcher, serve_stdio

    state: dict = {"session": None}

    def _outcome(data: Any, summary: str, error: bool = False) -> ToolOutcome:
        return ToolOutcome(json.dumps(data, ensure_ascii=False, default=str), summary, is_error=error)

    def call_tool(name: str, arguments: dict) -> ToolOutcome:
        if name == "open_frame":
            if state["session"] is not None:
                state["session"].close()
                state["session"] = None
            session = open_headless_session(
                arguments["path"], notes=arguments.get("notes") or "", saved_profiles=saved_profiles, data_dir=data_dir,
            )
            state["session"] = session
            return session.call("get_status")
        session = state["session"]
        if session is None:
            return _outcome({"error": "Open a frame first with open_frame(path)."}, "no frame", True)
        if name == "run_standard_pipeline" and "out_dir" in arguments:
            arguments = dict(arguments)
            folder = arguments.pop("out_dir")
            outcome = session.call(name, arguments)
            if folder and not outcome.is_error and session.results.pipeline is not None:
                written = [str(path) for path in write_outputs(session, session.results.pipeline, Path(folder))]
                data = {**json.loads(outcome.content), "outputs": written}
                return _outcome(data, outcome.summary)
            return outcome
        return session.call(name, arguments)

    dispatcher = McpDispatcher(mcp_tools(), call_tool, instructions=MCP_INSTRUCTIONS)
    out = writer or sys.stdout.buffer
    # Anything printed while a tool runs must not reach the protocol stream.
    sys.stdout = sys.stderr
    try:
        serve_stdio(dispatcher, reader or sys.stdin.buffer, out)
    finally:
        if state["session"] is not None:
            state["session"].close()


__all__ = [
    "HeadlessSession",
    "LOAD_TIMEOUT_S",
    "MCP_INSTRUCTIONS",
    "analyse_frames",
    "default_output_folder",
    "mcp_tools",
    "open_headless_session",
    "serve_mcp_stdio",
    "write_outputs",
]
