"""The standard procedure in code, GIWAXS or GISAXS: a baseline any agent (or script) can start from.

``StandardPipeline`` drives the assistant's own tools, so the numbers are the
ones Process with AI and Run Automatic Analysis report, in a fixed order:
status → geometry → frames → the technique → for GIWAXS peaks, in-/out-of-plane,
ring orientation and crystallite size; for GISAXS the cut, symmetry, halves,
spacing and fit (``GisaxsProcedureMixin``). Every routine decision is taken the
same way each time and recorded with its reason (``decisions``).  What code
cannot decide — a value only the notes or the person know, a calibration that is
not good enough, a judgement such as the GISAXS model — goes into
``needs_attention`` with the option that supplies it (if any).

The decisions are defaults, not limits: an agent that can do more revisits
them with the other tools (another frame, another ring, custom sectors).
Nothing is written next to the data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..domain import caveat, energy_from_notes, incidence_from_notes, pixel_size_from_notes
from ..domain.notes import calibrants_in_notes
from .gisaxs_procedure import GisaxsProcedureMixin
from .models import ToolCall
from .pipeline_geometry import GeometryStepsMixin
from .pipeline_progress import FrameChanged, PipelineStopped, frame_changed, same_file
from .pipeline_report import calibration_quality, compact_report
from .pipeline_rows import peak_rows as _peak_rows
from .pipeline_rows import peak_search as _peak_search
from .pipeline_rows import ring_row as _ring_row
from .tools import ToolCatalog, clean, results_payload

SERIES_SUM = 10
"""Frames summed for the state of a series: the last ten show the final state of an in-situ run."""
TECHNIQUES = ("gisaxs", "giwaxs")
FORCED = "forced: the technique was given (--technique, technique=)"
DETECTED = "detected: GIMaP's Auto detection classified the frame once the geometry was applied"
CHOSEN = "the mode chosen in Analyze"
DEFAULT = "the default while Analyze is on Auto"
FALLBACK = "Auto could not classify the frame, so GIWAXS was assumed"
NO_GEOMETRY = "no geometry: Auto tells GISAXS from GIWAXS only once a geometry is applied"
"""Why the ``technique`` decision is what it is (the report's Decisions)."""
SWITCHED = (
    "the automatic detection chose otherwise; this is the GIWAXS procedure (choose GISAXS in Analyze, "
    "or --technique gisaxs, for the GISAXS one)"
)


@dataclass(frozen=True)
class PipelineOptions:
    calibration: Optional[str] = None
    """A calibration file (.poni, GIMaP calibration) or an image of a standard, tried first; then the
    calibration files the notes name, the instrument profile and the automatic search."""
    standard: Optional[str] = None
    """The standard an image shows (agbh, lab6, ceo2, lab6_ceo2); None: the one calibrant the notes name,
    else from the image's name, else compared (also when the notes and the name disagree)."""
    energy_kev: Optional[float] = None
    incidence_deg: Optional[float] = None
    pixel_size_um: Optional[float] = None
    """Detector pixel size (µm): the person's statement, used even when the image header gives another
    (the decision names both)."""
    frame: Optional[int] = None
    """1-based frame of a series, negative from the end; None: the last frames (the final state)."""
    sum_frames: Optional[int] = None
    rings: int = 3
    """How many of the strongest reliable peaks get an orientation distribution and a size."""
    ring_q: Optional[float] = None
    """A ring the person asked about (Å⁻¹): analysed as well, whatever its strength."""
    recalibrate: bool = False
    """Calibrate even when the detector already has an instrument profile."""
    notes: str = ""
    """Free text from the person (beamtime notes): αi, the energy and the pixel size (each when the notes give
    exactly one), the calibrant (when they name exactly one) and calibration files named in it are used."""
    geometry: Optional[dict] = None
    """A calibration result from an earlier frame of the same detector (a batch of samples)."""
    stop_after_geometry: bool = False
    """Only find and check the geometry (``Find Calibration Automatically``)."""
    technique: Optional[str] = None
    """``gisaxs`` or ``giwaxs`` forces that procedure; ``None``: the mode chosen in Analyze, else (Auto) see
    ``follow_detection``."""
    follow_detection: bool = False
    """With Analyze on Auto and no technique: the technique Auto detects once the geometry is applied (Run
    Automatic Analysis, the command line, MCP); GIWAXS, flagged in needs_attention, when Auto cannot classify
    the frame. Off: GIWAXS (Process with AI names its technique)."""
    fit: bool = True
    """GISAXS: fit the horizontal cut when a fitter is available."""


@dataclass
class _Run:
    ok: bool = False
    steps: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    attention: list = field(default_factory=list)
    calibration: Optional[dict] = None
    geometry_source: str = ""
    stopped: Optional[str] = None
    """The step that was not started because a stop was requested (``None``: ran to the end)."""
    failed: Optional[str] = None
    """Why the run ended early as a failure (the frame changed); ``None`` otherwise."""


class StandardPipeline(GeometryStepsMixin, GisaxsProcedureMixin):
    def __init__(
        self,
        catalog: ToolCatalog,
        options: PipelineOptions = PipelineOptions(),
        progress: Callable[[str], None] = lambda _text: None,
        *,
        stop=None,
        events: Optional[Callable[[dict], None]] = None,
    ):
        """``stop``: a ``threading.Event``; set, the run ends before the next tool call and reports what
        it found so far. ``events``: called with ``{"state": "start" | "done", "tool", …}`` around each call."""
        self.catalog = catalog
        self.options = options
        self.progress = progress
        self.stop = stop
        self.events = events
        self._run = _Run()
        self._energy: Optional[float] = None
        self._incidence: Optional[float] = None
        self._pixel: Optional[float] = None
        self._pixel_given = False
        """The pixel size is the person's (``pixel_size_um``): it beats image headers, files and profiles."""
        self._noted_standard: Optional[str] = None
        """The one calibrant the notes name."""
        self._given_rejected: Optional[str] = None
        """Why the calibration given (``calibration``) was not used, when another geometry was."""
        self._declined = False
        self._procedure = "giwaxs"
        self._start_path: Optional[str] = None
        """The file the run started on (its first ``get_status``): the report's ``frame``."""
        self._frame_status: Optional[dict] = None
        """The last status of that file: the report describes it even when the frame changed."""

    @property
    def calibration(self) -> Optional[dict]:
        """The calibration the run used (full precision), for later frames of the same detector."""
        return self._run.calibration

    # -- bookkeeping -----------------------------------------------------------------

    def _call(self, name: str, arguments: Optional[dict] = None) -> Optional[dict]:
        arguments = arguments or {}
        if self.stop is not None and self.stop.is_set():
            raise PipelineStopped(name)
        index = len(self._run.steps) + 1
        if self.events is not None:
            self.events({"state": "start", "tool": name, "arguments": dict(arguments), "index": index})
        started = time.monotonic()
        outcome = self.catalog.execute(ToolCall(f"p{index}", name, arguments))
        self._run.steps.append({"tool": name, "arguments": arguments, "summary": outcome.summary, "error": outcome.is_error})
        if self.events is not None:
            self.events({
                "state": "done", "tool": name, "arguments": dict(arguments), "index": index,
                "summary": outcome.summary, "error": outcome.is_error, "seconds": time.monotonic() - started,
            })
        self.progress(f"{name}: {outcome.summary}")
        data = outcome.data if isinstance(outcome.data, dict) else {}
        self._same_frame(data)
        if outcome.is_error:
            return None
        if data.get("declined"):  # the person said no: not done, and not asked again in this run
            self._declined = True
            self._decide("declined", name, str(data.get("message", "the person declined")))
            return None
        return data

    def _same_frame(self, data: dict) -> None:
        """End the run when Analyze no longer shows the file it started on (Analyze cleared, a project opened):
        a workbench that refused to act on another file says so (``frame_changed``), a new status names
        another file or none (Analyze cleared)."""
        if self._start_path is None:
            return
        if data.get("frame_changed"):
            raise FrameChanged(str(data.get("error") or frame_changed(self._start_path, None)))
        status = self.catalog.results.status
        if status is None or status is self._frame_status:  # no new status since the last check
            return
        if not same_file(status.get("path"), self._start_path):
            raise frame_changed(self._start_path, status.get("path"))
        self._frame_status = status

    def _decide(self, what: str, decision: str, why: str) -> None:
        self._run.decisions.append({"what": what, "decision": decision, "why": why})

    def _attention(self, item: str, why: str, option: Optional[str], hint: str) -> None:
        entry = {"item": item, "why": why, "option": option, "hint": hint}
        if entry not in self._run.attention:  # e.g. the same missing energy for every image tried
            self._run.attention.append(entry)

    # -- the procedure ---------------------------------------------------------------

    def run(self) -> dict:
        """The whole procedure; a stop request ends it between two steps with what was found so far, and a
        frame that changed (Analyze shows another file than the run's) ends it as a failure (``failed``)."""
        try:
            return self._procedure_run()
        except PipelineStopped as stopped:
            self._run.stopped = str(stopped) or "the next step"
            self._decide("stopped", f"before {self._run.stopped}", "the person pressed Stop; the results so far are kept")
            return self._report()
        except FrameChanged as changed:
            self._run.ok, self._run.failed = False, str(changed)
            self.catalog.results.status = self._frame_status  # the report describes the file the run started on
            self._decide("frame", "the run ended", f"{changed}; nothing more was done")
            self._attention("frame", f"{changed}.", None, "Show that file in Analyze again and run the analysis again.")
            return self._report()

    def _procedure_run(self) -> dict:
        status = self._call("get_status") or {}
        self._start_path, self._frame_status = status.get("path"), self.catalog.results.status
        if not status.get("path"):
            self._attention("frame", "No detector image is open.", None, "Give the path of a detector image.")
            return self._report()
        if status.get("measurement") is None and status.get("message") and not status.get("frames"):
            self._attention("frame", str(status["message"]), None, "Check that GIMaP can read this file.")
            return self._report()
        self._values(status)
        if not self._geometry(status):
            options = self.options
            if options.technique is None and options.follow_detection and not options.stop_after_geometry:
                self._decide("technique", "none", NO_GEOMETRY)
            return self._report()
        if self.options.stop_after_geometry:
            self._run.ok = True
            return self._report()
        self._frames()
        status = self.catalog.results.status or {}
        kind = status.get("measurement")
        technique, why = self._technique(status)
        self._decide("technique", technique.upper(), why)
        if kind != technique:
            if self._call("set_measurement_mode", {"mode": technique}) is None:
                self._attention(
                    "measurement", f"GIMaP reduces this frame as {kind} and could not switch to {technique.upper()}.",
                    None, "Check the frame in the GUI.",
                )
                return self._report()
            self._decide("measurement", f"switched from {kind or 'no reduction'} to {technique.upper()}",
                         SWITCHED if why == DEFAULT else why)
        self._procedure = technique
        self._analyse_gisaxs() if technique == "gisaxs" else self._analyse()
        self._run.ok = True
        return self._report()

    def _technique(self, status: dict) -> tuple[str, str]:
        """The procedure to run and why: the technique given, the mode chosen in Analyze, what Auto detected
        (``follow_detection``) or GIWAXS — flagged when Auto was to be followed but could not classify the frame."""
        options, mode, kind = self.options, status.get("mode"), status.get("measurement")
        detected = kind if kind in TECHNIQUES and mode not in TECHNIQUES else None
        if options.technique:
            return options.technique, DETECTED if options.follow_detection and detected == options.technique else FORCED
        if mode in TECHNIQUES:
            return mode, CHOSEN
        if not options.follow_detection:
            return "giwaxs", DEFAULT
        if detected:
            return detected, DETECTED
        self._attention(
            "technique", f"GIMaP's Auto detection could not classify this frame ({kind or 'no reduction'}), so the "
            "GIWAXS procedure ran.", "technique",
            "Give the technique (GIWAXS or GISAXS); the notes or the set-up (detector distance) say which.",
        )
        return "giwaxs", FALLBACK

    def _values(self, status: dict) -> None:
        options, header = self.options, status.get("header") or {}
        for value, source in (
            (options.energy_kev, "given"),
            (header.get("energy_kev"), "the image header"),
            (energy_from_notes(options.notes), "the notes"),
        ):
            if value:
                self._energy = float(value)
                self._decide("energy", f"{self._energy:g} keV", f"from {source}")
                break
        for value, source in ((options.incidence_deg, "given"), (incidence_from_notes(options.notes), "the notes")):
            if value is not None:
                self._incidence = float(value)
                self._decide("incidence angle", f"αi = {self._incidence:g}°", f"from {source}")
                break
        header_pixel = (header.get("pixel_size_um") or [None])[0]
        noted = pixel_size_from_notes(options.notes)
        if options.pixel_size_um:  # the person's statement: it beats the header
            self._pixel, self._pixel_given = float(options.pixel_size_um), True
            why = "from given (the image header has none)" if not header_pixel else (
                f"given; the image header says {header_pixel:g} µm, the given value is used"
                if abs(header_pixel - self._pixel) > 0.01 * header_pixel else "from given (the image header agrees)"
            )
            self._decide("pixel size", f"{self._pixel:g} µm", why)
        elif noted and not header_pixel:
            self._pixel = float(noted)
            self._decide("pixel size", f"{self._pixel:g} µm", "from the notes (the image header has none)")
        calibrants = calibrants_in_notes(options.notes)
        if len(calibrants) == 1:
            self._noted_standard = calibrants[0]
        elif calibrants and not options.standard:
            self._decide("calibration standard", "none from the notes", f"the notes name {len(calibrants)}: {', '.join(calibrants)}")

    def _frames(self) -> None:
        status = self.catalog.results.status or {}
        total = int(status.get("frames") or 1)
        if total <= 1:
            return
        frame = -1 if self.options.frame is None else int(self.options.frame)
        count = self.options.sum_frames or min(SERIES_SUM, total)
        if self._call("set_frame", {"frame": frame, "sum": count}) is None:
            return
        why = (
            f"a series of {total} frames (an in-situ run or a scan): the last frames show the final state, "
            "summed for statistics; frame 1 shows the start"
            if self.options.frame is None else "as asked"
        )
        self._decide("frames", self._run.steps[-1]["summary"], why)

    def _analyse(self) -> None:
        self._call("find_peaks", {"curve": "radial"})
        search = self.catalog.results.peak_searches.get("radial")
        if search is None or not search.peaks:
            reason = search.reason if search is not None else "the radial curve could not be searched"
            self._decide("peaks", "none", reason)
            return
        self._call("compare_sectors")
        reliable = [peak for peak in search.peaks if not caveat(peak)]
        chosen = sorted(reliable, key=lambda peak: -peak.area)[: max(0, int(self.options.rings))]
        skipped = [f"{peak.q:.4g} ({caveat(peak).split(':')[0]})" for peak in search.peaks if caveat(peak)]
        self._decide(
            "rings analysed", ", ".join(f"{peak.q:.4g}" for peak in sorted(chosen, key=lambda item: item.q)) or "none",
            "the strongest reliable peaks by area" + (f"; not analysed: {', '.join(skipped)}" if skipped else ""),
        )
        centres = [float(peak.q) for peak in chosen]
        asked = self.options.ring_q
        if asked and not any(abs(peak.q - asked) <= 0.5 * peak.fwhm for peak in chosen):
            centres.append(float(asked))
            self._decide("requested ring", f"q ≈ {asked:g} Å⁻¹", "the person asked for it")
        for q in sorted(centres):
            self._call("ring_orientation", {"q_center": q})
            if any(abs(peak.q - q) <= 0.5 * peak.fwhm for peak in reliable):
                self._call("crystallite_size", {"q_center": q})

    # -- report ----------------------------------------------------------------------

    def _report(self) -> dict:
        results, run = self.catalog.results, self._run
        status = results.status or {}
        return {
            "ok": run.ok,
            "stopped": run.stopped,
            "failed": run.failed,
            "procedure": "geometry" if self.options.stop_after_geometry else self._procedure,
            "gisaxs": self._gisaxs_report(),
            "frame": self._start_path or status.get("path"),
            "measurement": status.get("measurement"),
            "frames": {"total": status.get("frames"), "first": status.get("frame"), "summed": status.get("summed_frames")},
            "geometry": {"source": run.geometry_source, **clean(status.get("geometry") or {})},
            "calibration_quality": calibration_quality(run.calibration),
            "peaks": _peak_rows(results),
            "peak_search": _peak_search(results),
            "rings": [_ring_row(ring) for ring in results.rings],
            "series_hints": list(getattr(results.peak_searches.get("radial"), "series", ()) or ()),
            "decisions": run.decisions,
            "needs_attention": run.attention,
            "steps": run.steps,
            "calibration": clean(run.calibration),
            "tables": results_payload(results),
        }


__all__ = ["PipelineOptions", "SERIES_SUM", "StandardPipeline", "compact_report"]
