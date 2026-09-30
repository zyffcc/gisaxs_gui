"""The standard GIWAXS procedure in code: a baseline any agent (or script) can start from.

``StandardPipeline`` drives the assistant's own tools, so the numbers are the
ones Process with Claude reports, in a fixed order: status → geometry →
frames → peaks → in-/out-of-plane → ring orientation → crystallite size.
Every routine decision is taken the same way each time and recorded with its
reason (``decisions``).  What code cannot decide — a value only the notes or
the person know, a calibration that is not good enough — goes into
``needs_attention`` with the option that supplies it.

The decisions are defaults, not limits: an agent that can do more revisits
them with the other tools (another frame, another ring, custom sectors).
Nothing is written next to the data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..domain import (
    assessment,
    caveat,
    energy_from_notes,
    incidence_from_notes,
    pixel_size_from_notes,
    standard_from_name,
)
from .gisaxs_procedure import GisaxsProcedureMixin
from .models import ToolCall
from .pipeline_progress import PipelineStopped
from .pipeline_rows import peak_rows as _peak_rows
from .pipeline_rows import peak_search as _peak_search
from .pipeline_rows import ring_row as _ring_row
from .tools import ToolCatalog, clean, results_payload

SERIES_SUM = 10
"""Frames summed for the state of a series: the last ten show the final state of an in-situ run."""
MAX_FILES = 2
MAX_IMAGES = 3
"""Calibration files and standard images tried before giving up."""
ACCEPTED = ("good", "usable")


@dataclass(frozen=True)
class PipelineOptions:
    calibration: Optional[str] = None
    """A calibration file (.poni, GIMaP calibration) or an image of a standard; None: search for one."""
    standard: Optional[str] = None
    """The standard an image shows (agbh, lab6, ceo2, lab6_ceo2); None: from its name, else compared."""
    energy_kev: Optional[float] = None
    incidence_deg: Optional[float] = None
    pixel_size_um: Optional[float] = None
    """Detector pixel size, for images whose header has none (plain TIFF)."""
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
    """Free text from the person (beamtime notes): αi and the energy are read from it."""
    geometry: Optional[dict] = None
    """A calibration result from an earlier frame of the same detector (a batch of samples)."""
    stop_after_geometry: bool = False
    """Only find and check the geometry (``Find Calibration Automatically``)."""
    technique: Optional[str] = None
    """``gisaxs`` or ``giwaxs``; ``None``: the technique chosen in Analyze, GIWAXS when it is automatic."""
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


class StandardPipeline(GisaxsProcedureMixin):
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
        self._declined = False
        self._procedure = "giwaxs"

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
        if outcome.is_error:
            return None
        data = outcome.data if isinstance(outcome.data, dict) else {}
        if data.get("declined"):  # the person said no: not done, and not asked again in this run
            self._declined = True
            self._decide("declined", name, str(data.get("message", "the person declined")))
            return None
        return data

    def _decide(self, what: str, decision: str, why: str) -> None:
        self._run.decisions.append({"what": what, "decision": decision, "why": why})

    def _attention(self, item: str, why: str, option: Optional[str], hint: str) -> None:
        self._run.attention.append({"item": item, "why": why, "option": option, "hint": hint})

    # -- the procedure ---------------------------------------------------------------

    def run(self) -> dict:
        """The whole procedure; a stop request ends it between two steps with what was found so far."""
        try:
            return self._procedure_run()
        except PipelineStopped as stopped:
            self._run.stopped = str(stopped) or "the next step"
            self._decide("stopped", f"before {self._run.stopped}", "the person pressed Stop; the results so far are kept")
            return self._report()

    def _procedure_run(self) -> dict:
        status = self._call("get_status") or {}
        if not status.get("path"):
            self._attention("frame", "No detector image is open.", None, "Give the path of a detector image.")
            return self._report()
        if status.get("measurement") is None and status.get("message") and not status.get("frames"):
            self._attention("frame", str(status["message"]), None, "Check that GIMaP can read this file.")
            return self._report()
        self._values(status)
        if not self._geometry(status):
            return self._report()
        if self.options.stop_after_geometry:
            self._run.ok = True
            return self._report()
        self._frames()
        status = self.catalog.results.status or {}
        mode = status.get("measurement")
        chosen = status.get("mode") if status.get("mode") in ("gisaxs", "giwaxs") else None
        if (self.options.technique or chosen) == "gisaxs":
            if mode != "gisaxs" and self._call("set_measurement_mode", {"mode": "gisaxs"}) is None:
                self._attention("measurement", f"GIMaP reduces this frame as {mode} and could not switch to GISAXS.", None, "Check the frame in the GUI.")
                return self._report()
            self._procedure = "gisaxs"
            self._analyse_gisaxs()
            self._run.ok = True
            return self._report()
        if mode != "giwaxs":
            if self._call("set_measurement_mode", {"mode": "giwaxs"}) is None:
                self._attention("measurement", f"GIMaP reduces this frame as {mode} and could not switch to GIWAXS.", None, "Check the frame in the GUI.")
                return self._report()
            self._decide(
                "measurement", f"switched from {mode or 'no reduction'} to GIWAXS",
                "the automatic detection chose otherwise; this is the GIWAXS procedure (analyse GISAXS in the GUI)",
            )
        self._analyse()
        self._run.ok = True
        return self._report()

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
        for value, source in ((options.pixel_size_um, "given"), (pixel_size_from_notes(options.notes), "the notes")):
            if value and not header_pixel:
                self._pixel = float(value)
                self._decide("pixel size", f"{self._pixel:g} µm", f"from {source} (the image header has none)")
                break

    def _incidence_missing(self) -> None:
        self._attention(
            "incidence angle αi",
            "Neither the options, the notes nor an instrument profile give αi, so 0° is used. Ring "
            "positions |q| barely change, but qz shifts by about k·sin αi (≈0.04 Å⁻¹ at 0.4° and 12 keV) "
            "and the missing wedge moves.",
            "incidence_deg",
            "Beamtime notes, the logbook or the slides usually state it (typically 0.1–0.5°); otherwise ask.",
        )

    def _geometry(self, status: dict) -> bool:
        options = self.options
        existing = status.get("geometry")
        if options.geometry is not None:
            self.catalog.results.calibrations.append(options.geometry)
            origin = Path(str(options.geometry.get("source_image") or "")).name
            return self._use_calibration(len(self.catalog.results.calibrations) - 1, f"{origin} (reused from the first frame of this batch)")
        if existing and options.calibration is None and not options.recalibrate:
            name = existing.get("instrument_profile") or "saved"
            self._run.geometry_source = f"instrument profile '{name}'"
            self._decide(
                "geometry", f"kept the instrument profile '{name}' this detector already has",
                "a saved profile is the person's own calibration (recalibrate to replace it)",
            )
            if self._incidence is not None:
                self._call("set_incidence_angle", {"degrees": self._incidence})
            elif not existing.get("incidence_deg"):
                self._incidence_missing()
            return True
        for candidate in self._candidates():
            if self._try(candidate):
                return True
            if self._declined:
                self._attention(
                    "geometry", "The person declined saving the geometry, so nothing is in q.", None,
                    "Approve the geometry, or calibrate in Tools ▸ Geometry Calibration.",
                )
                return False
        if not any(item["item"].startswith(("calibration", "X-ray energy", "pixel size")) for item in self._run.attention):
            self._attention(
                "calibration",
                "No calibration candidate gave a good geometry (see decisions for each one).",
                "calibration",
                "Name the calibration image or file used at the beamtime; the notes usually say which.",
            )
        return False

    def _candidates(self) -> list[dict]:
        options = self.options
        if options.calibration:
            name = Path(options.calibration).name
            return [{"path": options.calibration, "standard_key": standard_from_name(name), "given": True}]
        listing = self._call("find_calibration_files", {"max_results": 12})
        if listing is None:
            self._attention("calibration", "The folders around the frame could not be searched.", "calibration", "Name the calibration file.")
            return []
        files = [dict(item, file=True) for item in listing.get("calibration_results", [])][:MAX_FILES]
        images = [item for item in listing.get("standard_images", []) if item.get("gimap_can_fit")][:MAX_IMAGES]
        if not files and not images:
            searched = len(listing.get("folders_searched", []))
            self._attention(
                "calibration",
                f"No calibration file and no image of a standard GIMaP can fit was found ({searched} folders searched).",
                "calibration",
                "An image of AgBh, LaB6, CeO2 or a LaB6+CeO2 mixture taken with this detector, or a .poni / "
                "GIMaP calibration file; a log or the beamtime notes usually name it.",
            )
        ranked = ", ".join(Path(item["path"]).name for item in files + images)
        if ranked:
            self._decide("calibration candidates", ranked, str(listing.get("ranking", "ranked by the search")))
        return files + images

    def _try(self, candidate: dict) -> bool:
        path, name = candidate["path"], Path(candidate["path"]).name
        info = self._call("inspect_file", {"path": path})
        if info is None:
            self._decide("calibration", f"skipped {name}", "it could not be read")
            return False
        frame = self.catalog.results.status or {}
        if candidate.get("file") or info.get("kind") in ("pyFAI calibration (.poni)", "GIMaP calibration"):
            return self._use_file(path, name, info, frame)
        if info.get("kind") != "detector image":
            self._decide("calibration", f"skipped {name}", f"it is a {info.get('kind')} file, not a calibration")
            return False
        if info.get("same_detector_size_as_frame") is False:
            self._decide("calibration", f"skipped {name}", f"{info.get('shape')} pixels, the frame has {frame.get('shape')}: another detector")
            return False
        energy = info.get("energy_kev") or self._energy
        if not energy:
            self._attention(
                "X-ray energy",
                "Neither the images' headers, the options nor the notes give the energy; calibration needs it.",
                "energy_kev",
                "Beamtime notes or the logbook; P03 GIWAXS is often 11.8 or 12.4 keV but never guess.",
            )
            return False
        standard = self.options.standard or candidate.get("standard_key") or info.get("standard") or "compare"
        arguments = {"path": path, "standard": standard, "energy_kev": float(energy)}
        if self._pixel and not (info.get("pixel_size_um") or [None])[0]:
            arguments["pixel_size_um"] = self._pixel
        fitted = self._call("calibrate_geometry", arguments)
        if fitted is None:
            failure = self._run.steps[-1]["summary"]
            self._decide("calibration", f"{name} ({standard}) failed", failure)
            if "pixel size" in failure.lower():
                self._attention(
                    "pixel size",
                    f"{name} has no pixel size in its header, so it cannot be calibrated.",
                    "pixel_size_um",
                    "The detector's pixel size: Pilatus 172 µm, Eiger 75 µm, Lambda 55 µm; the notes or the "
                    "detector name usually say which.",
                )
            return False
        if standard == "compare":
            verdict = str(fitted.get("verdict", ""))
            best = next((row for row in fitted.get("comparison", []) if row["calibration_index"] == fitted.get("best_calibration_index")), None)
            if best is None or not verdict.startswith(("clear", "probably")) or not str(best.get("assessment", "")).startswith(ACCEPTED):
                self._decide("calibration", f"rejected {name}", verdict or "no standard fitted")
                if verdict.startswith("ambiguous"):
                    self._attention("calibration standard", verdict, "standard", "The notes or the file name usually say which standard it is.")
                return False
            self._decide("calibration standard", str(best.get("standard")), verdict)
            index = int(fitted["best_calibration_index"])
        else:
            quality = str(fitted.get("assessment", ""))
            if not quality.startswith(ACCEPTED):
                self._decide("calibration", f"rejected {name} ({standard})", quality)
                return False
            index = int(fitted["calibration_index"])
        return self._use_calibration(index, f"{name}")

    def _use_calibration(self, index: int, origin: str) -> bool:
        result = self.catalog.results.calibrations[index]
        arguments: dict = {"source": "calibration", "calibration_index": index}
        fit_energy = result.get("energy_kev")
        if self._energy and fit_energy and abs(fit_energy - self._energy) > 2e-3 * self._energy:
            # The calibrant was measured at another energy: its distance and centre hold, the frame's energy is used.
            arguments = {
                "source": "values", "distance_mm": result["distance_mm"],
                "beam_center_x_px": result["beam_center_px"][0], "beam_center_y_px": result["beam_center_px"][1],
                "energy_kev": self._energy, "pixel_size_um": result["pixel_size_um"][0],
                "note": f"{origin} (calibrated at {fit_energy:g} keV; the frame's {self._energy:g} keV used)",
            }
        if self._incidence is not None:
            arguments["incidence_deg"] = self._incidence
        if self._call("use_geometry", arguments) is None:
            self._decide("geometry", f"could not use {origin}", self._run.steps[-1]["summary"])
            return False
        self._run.calibration = result
        self._run.geometry_source = origin
        if result.get("from_file"):
            self._decide("geometry", f"from {origin}", "a saved calibration (not re-checked against a standard image)")
        else:
            self._decide("geometry", f"calibrated from {origin}", assessment(result))
        if self._incidence is None:
            self._incidence_missing()
        return True

    def _use_file(self, path: str, name: str, info: dict, frame: dict) -> bool:
        header = frame.get("header") or {}
        frame_pixel = (header.get("pixel_size_um") or [None])[0]
        file_pixel = info.get("pixel_size_x_m")
        file_pixel = file_pixel * 1e6 if file_pixel else (info.get("pixel_size_um") or [None])[0]
        if frame_pixel and file_pixel and abs(file_pixel - frame_pixel) > 0.01 * frame_pixel:
            self._decide("calibration", f"skipped {name}", f"made for {file_pixel:g} µm pixels, the frame has {frame_pixel:g} µm: another detector")
            return False
        if info.get("shape") and frame.get("shape") and list(info["shape"]) != list(frame["shape"]):
            self._decide("calibration", f"skipped {name}", f"made for {info['shape']} pixels, the frame has {frame['shape']}")
            return False
        energy = self._energy or info.get("energy_kev")
        if not energy:
            self._attention("X-ray energy", f"{name} and the frame give no energy.", "energy_kev", "Beamtime notes or the logbook.")
            return False
        values = _file_geometry(path, info, file_pixel)
        arguments = {"source": "file", "path": path, "energy_kev": float(energy)}
        file_energy = info.get("energy_kev")
        if self._energy and file_energy and abs(file_energy - self._energy) > 2e-3 * self._energy:
            # Measured at another energy: the distance and centre hold, the frame's energy is used.
            arguments = {
                "source": "values", "distance_mm": values["distance_mm"],
                "beam_center_x_px": values["beam_center_px"][0], "beam_center_y_px": values["beam_center_px"][1],
                "energy_kev": self._energy, "note": f"{name} (made at {file_energy:g} keV; the frame's {self._energy:g} keV used)",
            }
            if file_pixel:
                arguments["pixel_size_um"] = float(file_pixel)
        if self._incidence is not None:
            arguments["incidence_deg"] = self._incidence
        if self._call("use_geometry", arguments) is None:
            self._decide("geometry", f"could not use {name}", self._run.steps[-1]["summary"])
            return False
        self._run.geometry_source = name
        self._run.calibration = values
        self._decide("geometry", f"from {name}", str(info.get("assessment") or info.get("kind")))
        if self._incidence is None:
            self._incidence_missing()
        return True

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
            "procedure": "geometry" if self.options.stop_after_geometry else self._procedure,
            "gisaxs": self._gisaxs_report(),
            "frame": status.get("path"),
            "measurement": status.get("measurement"),
            "frames": {"total": status.get("frames"), "first": status.get("frame"), "summed": status.get("summed_frames")},
            "geometry": {"source": run.geometry_source, **clean(status.get("geometry") or {})},
            "calibration_quality": _quality(run.calibration),
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


def compact_report(report: dict) -> dict:
    """The report without the full tables, the calibration record and the fitted arrays (kept in report.json)."""
    compact = {key: value for key, value in report.items() if key not in ("tables", "calibration")}
    fit = (compact.get("gisaxs") or {}).get("fit")
    if fit:
        compact["gisaxs"] = {**compact["gisaxs"], "fit": {
            key: value for key, value in fit.items() if key not in ("data", "best_curve", "curves", "native")}}
    if compact.get("peaks"):
        compact["peaks"] = [
            {**peak, "fit": {key: value for key, value in peak["fit"].items() if key not in ("x", "y", "sigma")}}
            if peak.get("fit") else peak for peak in compact["peaks"]
        ]
    return compact


def _file_geometry(path: str, info: dict, pixel_um: Optional[float]) -> dict:
    """A saved calibration (.poni or GIMaP) in the shape of a calibration result, for reuse in a batch."""
    if info.get("kind") == "GIMaP calibration":
        return {key: value for key, value in info.items() if key != "kind"}
    wavelength = info.get("wavelength_angstrom")
    return {
        "distance_mm": info["distance_mm"],
        "beam_center_px": [info["beam_center_x_px"], info["beam_center_y_px"]],
        "wavelength_angstrom": wavelength,
        "energy_kev": info.get("energy_kev"),
        "pixel_size_um": [pixel_um, pixel_um] if pixel_um else None,
        "source_image": path,
        "shape": None,
        "from_file": True,
    }


def _quality(result: Optional[dict]) -> Optional[dict]:
    if not result:
        return None
    if result.get("from_file"):
        return {"assessment": "a saved calibration file (not re-checked against a standard image)", "warnings": []}
    check = result.get("line_check") or {}
    return {
        "assessment": assessment(result),
        "standard": result.get("standard"),
        "lines_checked": check.get("lines_checked"),
        "mean_line_q_error_percent": None if check.get("mean_relative") is None else round(100 * check["mean_relative"], 3),
        "matched_rings": result.get("matched_rings"),
        "warnings": list(result.get("warnings") or ()),
    }


__all__ = ["PipelineOptions", "SERIES_SUM", "StandardPipeline", "compact_report"]
