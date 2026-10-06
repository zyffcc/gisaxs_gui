"""GIMaP's assistant tools without a window, for command-line agents and scripts (GIWAXS and GISAXS).

The Analyze page runs offscreen, so every number is the one the GUI shows and
the tools are exactly those of Process with AI.  The person's GIMaP data
folder is only read: instrument profiles saved there are used (a detector they
calibrated keeps its geometry), new ones live in memory, and settings are the
defaults so a run is reproducible.  Two rules differ from the GUI:

- the standard pipeline follows GIMaP's Auto detection when no technique is
  given (the GISAXS procedure for a frame Auto reads as GISAXS);
- every file is written under the session's output folder (``--out``,
  ``out_dir``, else ``<GIMaP data folder>/assistant_runs/<cli|mcp>/<time>/<frame>``),
  never next to the data: ``export_results`` writes to ``gimap_analysis/`` there.
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
PLAYBOOKS = ("docs/agents/giwaxs-playbook.md", "docs/agents/gisaxs-playbook.md")
EXPORT_FOLDER = "gimap_analysis"
"""``export_results`` writes here inside the output folder (the GUI: next to the data)."""
EXPORT_NOTE = "written under this session's output folder, never next to the data (a session without a window)"


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


def default_output_folder(kind: str = "cli") -> Path:
    """``<GIMaP data folder>/assistant_runs/<kind>/<time>``: where a session writes when no folder is given."""
    import time

    from ..integrations.state import user_data_dir

    return user_data_dir() / "assistant_runs" / kind / time.strftime("%Y%m%d-%H%M%S")


@dataclass
class OutputFolder:
    """Where a session without a window writes: the folder given, else a new one in the GIMaP data folder."""

    path: Optional[Path] = None
    kind: str = "cli"
    stem: str = "frame"

    def resolve(self) -> Path:
        if self.path is None:
            self.path = default_output_folder(self.kind) / self.stem
        return Path(self.path)

    def exports(self) -> Path:
        return self.resolve() / EXPORT_FOLDER


class HeadlessCatalog(ToolCatalog):
    """The assistant's tools without a window: the baseline follows Auto unless a technique is given, and
    ``export_results`` writes under the output folder and says where."""

    def __init__(self, *args, output: OutputFolder, load_error: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output
        self.load_error = load_error
        """Why Analyze could not read the frame when it was opened (``None``: it was read)."""

    def unreadable(self, status: Optional[dict]) -> bool:
        """The frame was not read: Analyze reported an error and shows no frame of it."""
        status = status or {}
        return bool(self.load_error) and status.get("measurement") is None and not status.get("frames")

    def _tool_get_status(self) -> ToolOutcome:
        outcome = super()._tool_get_status()
        if not self.unreadable(self.results.status):
            return outcome
        status = {**self.results.status, "message": self.load_error}  # the reason, not "not analysed yet"
        self.results.status = status
        return self._ok(status, f"not read: {self.load_error}"[:160])

    def _tool_run_standard_pipeline(self, **arguments) -> ToolOutcome:
        if arguments.get("technique") is None:
            arguments["follow_detection"] = True  # the technique Auto detects once the geometry is applied
        return super()._tool_run_standard_pipeline(**arguments)

    def _tool_export_results(self, include_tables: bool) -> ToolOutcome:
        outcome = super()._tool_export_results(include_tables)
        folder = self.output.exports()
        payload = {**(outcome.data or {}), "folder": str(folder), "note": EXPORT_NOTE}
        return self._ok(payload, f"{outcome.summary} to {folder}")


def _headless_workbench(page, output: OutputFolder):
    from ..features.assistant.presentation import GuiBridge, GuiWorkbench

    class HeadlessWorkbench(GuiWorkbench):
        """Analyze's automation; its curves are exported under the output folder, not next to the data."""

        def export_curves(self) -> list[str]:
            return [str(path) for path in self._call(page.view_model.export, output.exports())]

    return HeadlessWorkbench(page.automation(), GuiBridge())


def _headless_store(output: OutputFolder):
    from ..features.assistant.infrastructure import JsonResultStore

    class HeadlessStore(JsonResultStore):
        """The tables of ``export_results`` next to its curves; tool requests are not kept (no data folder)."""

        def write_tables(self, folder: str, stem: str, payload: dict) -> str:
            return super().write_tables(str(output.exports()), stem, payload)

    return HeadlessStore(None)


@dataclass
class HeadlessSession:
    """One frame open in an offscreen Analyze page, with the assistant's tools on it."""

    page: Any
    catalog: ToolCatalog
    results: RunResults
    notes: str = ""
    """The person's notes given when the frame was opened: the pipeline reads αi, the energy, the pixel size,
    the calibrant and calibration files named in them."""
    output: OutputFolder = field(default_factory=OutputFolder)
    """Where this session writes (reports, ``export_results``)."""
    calibration: Optional[dict] = None
    """The calibration the last pipeline run used (full precision)."""
    _calls: int = field(default=0, repr=False)

    def call(self, name: str, arguments: Optional[dict] = None) -> ToolOutcome:
        self._calls += 1
        return self.catalog.execute(ToolCall(f"cli{self._calls}", name, arguments or {}))

    def run_pipeline(self, options: PipelineOptions = PipelineOptions(), progress=lambda _text: None) -> dict:
        """The standard pipeline; without a technique it follows Auto (``follow_detection``)."""
        if not options.notes and self.notes:
            options = dataclasses.replace(options, notes=self.notes)
        if options.technique is None:
            options = dataclasses.replace(options, follow_detection=True)
        pipeline = StandardPipeline(self.catalog, options, progress)
        report = pipeline.run()
        self.calibration = pipeline.calibration
        return report

    def unreadable(self) -> Optional[str]:
        """Why GIMaP could not read the frame (``None``: it was read): such a frame failed, it asks nothing."""
        catalog = self.catalog
        if isinstance(catalog, HeadlessCatalog) and catalog.unreadable(self.results.status or self.call("get_status").data):
            return catalog.load_error
        return None

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
    out_dir: Optional[str | Path] = None,
    kind: str = "cli",
) -> HeadlessSession:
    """Open ``frame`` offscreen; tools may read calibration material around it (read-only). Files go to
    ``out_dir`` (else ``default_output_folder(kind)/<frame stem>``, made only when something is written)."""
    _application()
    from ..features.analyze.bootstrap import create_analyze_view_model
    from ..features.analyze.presentation.page import AnalyzePage
    from ..features.assistant.infrastructure import LocalFileExplorer
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
    failures: list[str] = []

    def failed(message: str, *_details) -> None:  # why a file could not be read (status only says "not analysed")
        failures.append(str(message))

    page.analysisFailed.connect(failed)
    page.add_paths([str(path)])
    page.tasks.wait(load_timeout_s)
    page.analysisFailed.disconnect(failed)
    results = RunResults()
    output = OutputFolder(Path(out_dir) if out_dir else None, kind, path.stem)
    catalog = HeadlessCatalog(
        _headless_workbench(page, output),
        AnalysisGoals(goals=tuple(GOALS), permission=PERMISSION_AUTO, instructions=notes),
        results,
        explorer=LocalFileExplorer(),
        calibrator=create_headless_calibration(),
        chooser=chooser,
        fitter=create_quick_fit(),
        store=_headless_store(output),
        output=output,
        load_error=failures[-1] if failures else None,
    )
    return HeadlessSession(page, catalog, results, notes=notes, output=output)


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
    """The standard pipeline on each frame (files in ``out/<frame stem>/``); later frames of the same detector
    reuse the first calibration; without a technique each frame gets the procedure Auto detects for it."""
    reports: list[dict] = []
    calibration: Optional[dict] = None
    for frame in frames:
        progress(f"== {frame}")
        try:
            session = open_headless_session(
                frame, notes=options.notes, saved_profiles=saved_profiles, data_dir=data_dir,
                out_dir=out / Path(frame).stem,
            )
        except Exception as exc:  # one unreadable frame must not stop the batch
            reports.append({"ok": False, "frame": str(frame), "error": str(exc) or type(exc).__name__, "needs_attention": []})
            continue
        try:
            shape = (session.call("get_status").data or {}).get("shape")
            frame_options = options
            if calibration is not None and options.calibration is None and list(calibration.get("shape") or []) == list(shape or []):
                frame_options = dataclasses.replace(options, geometry=calibration)
            report = session.run_pipeline(frame_options, progress)
            if session.unreadable():  # a file GIMaP cannot read failed (exit 1); there is nothing to answer
                report["failed"] = session.unreadable()
            calibration = session.calibration or calibration
            report["outputs"] = [str(path) for path in write_outputs(session, report, session.output.resolve())]
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
    "GIMaP tools for grazing-incidence X-ray scattering frames, GIWAXS and GISAXS. Start with "
    "open_frame(path, notes, out_dir). run_standard_pipeline gives a baseline in one call: the geometry "
    "(calibrations the notes name first), the frames of a series, then the procedure of the technique "
    "GIMaP's Auto detection gives once the geometry is applied (technique='giwaxs' or 'gisaxs' forces "
    "one) — GIWAXS: peaks, sectors, ring orientation and sizes; GISAXS: the Yoneda cut, the symmetry "
    "axis, the halves, the in-plane spacing and a fit. Each decision comes with its reason; "
    "needs_attention lists values only the notes or the person know (with the argument that supplies "
    "them) and judgements left open, such as the GISAXS model. The baseline's choices are defaults: "
    "investigate further with the other tools when the question needs it — other frames of a series, "
    "lines every sample shares, peaks it skipped, custom sectors, the cut position. Measured values come "
    "from the tools; show how derived values follow from them; mark interpretations as hypotheses. "
    "Files are written only under the session's output folder (out_dir; export_results writes to "
    "gimap_analysis/ there), never next to the data. Guides in the GIMaP repository: "
    f"{PLAYBOOKS[0]} (GIWAXS) and {PLAYBOOKS[1]} (GISAXS)."
)
OPEN_FRAME = {
    "name": "open_frame",
    "description": (
        "Open a detector image (a NeXus module file opens the whole series) offscreen in GIMaP's Analyze. "
        "notes: the person's beamtime notes, verbatim — the baseline reads αi, the energy (keV) and the "
        "pixel size (µm) when the notes give exactly one value, uses a calibration file they name (.poni, "
        "GIMaP .json, an image of a standard) before searching, and the calibrant when they name exactly "
        "one; every path named there becomes readable. out_dir: where this session writes (default "
        "<GIMaP data folder>/assistant_runs/mcp/<time>/<frame>)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string"}, "notes": {"type": ["string", "null"]}, "out_dir": {"type": ["string", "null"]},
        },
        "required": ["path"],
        "additionalProperties": False,
    },
}
OUT_DIR = {
    "type": ["string", "null"],
    "description": "Also write report.md/json, CSV curves and the q map here; later exports of this frame go here too.",
}


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
                out_dir=arguments.get("out_dir"), kind="mcp",
            )
            state["session"] = session
            outcome = session.call("get_status")
            if session.unreadable():  # the client sees the reason as an error, not a frame waiting for analysis
                return ToolOutcome(outcome.content, outcome.summary, is_error=True, data=outcome.data)
            return outcome
        session = state["session"]
        if session is None:
            return _outcome({"error": "Open a frame first with open_frame(path)."}, "no frame", True)
        if name == "run_standard_pipeline" and "out_dir" in arguments:
            arguments = dict(arguments)
            folder = arguments.pop("out_dir")
            if folder:
                session.output.path = Path(folder)
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
    "EXPORT_FOLDER",
    "HeadlessCatalog",
    "HeadlessSession",
    "LOAD_TIMEOUT_S",
    "MCP_INSTRUCTIONS",
    "OutputFolder",
    "PLAYBOOKS",
    "analyse_frames",
    "default_output_folder",
    "mcp_tools",
    "open_headless_session",
    "serve_mcp_stdio",
    "write_outputs",
]
