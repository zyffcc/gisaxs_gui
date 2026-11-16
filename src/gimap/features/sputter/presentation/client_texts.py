"""Run-time texts of Analyze and of the main screen (templates the Chinese table holds; values filled in after).

Numbers, units, scan names, model families and the package's own lines are data: they are shown as they are.
"""

from __future__ import annotations

import re

from src.gimap.app.presentation.i18n import tr

from ..application import STEP_OF, STEP_REUSED, STEP_TEXT
from .texts import text

ENTRY_CANDIDATE = "Candidate {rank}"
ENTRY_REFINED = "Refined"
ENTRY_MODEL = "Model solution"
"""The one entry of a sample of a static series (the package labels it ``model``)."""
ENTRY_SEGMENT = "{name} · {deviation}"
ENTRY_TIP = "{name}: fit deviation {deviation}"
ENTRY_TIP_WEIGHT = "{name}: fit deviation {deviation}, weight of the network {weight}"
WINDOW_TEXT = "t = {time} s · {deposit} nm deposited"
WINDOW_TEXT_PLAIN = "t = {time} s"
WINDOW_TIP = "Time window {index} of {count}."
WINDOW_TIP_ONSET = "Time window {index} of {count}: {onset} s after the start of the deposition."
RESULT_NAME = "{scan} · {sample} · {family}"
MASK_FILE = "Pixel mask: {file}."
MASK_AUTO = ("Pixel mask: {file} and the automatic mask from the frames ({shadow} pixels of beamstop and holder, "
             "{dead} dead, {hot} hot).")
MASK_AUTO_ONLY = ("Pixel mask: the automatic mask from the frames ({shadow} pixels of beamstop and holder, {dead} dead, "
                  "{hot} hot); {file} does not fit this detector.")
WINDOW_ERROR = "This time window has no solution: {reason}"
WINDOW_CONTROLS = ("Previous time window", "Next time window", "Time window shown")
"""The tooltips of the arrows and of the slider for an in-situ scan …"""
SAMPLE_CONTROLS = ("Previous sample of the series", "Next sample of the series", "Sample shown")
"""… and for a series of static samples."""
SAMPLE_TEXT = "{label} · nominal {thickness} nm"
SAMPLE_TIP = "Sample {index} of {count}."
SAMPLE_TIP_FRAMES = "Sample {index} of {count}: {frames} frames."
SAMPLE_TIP_SUBSTRATE = "Substrate: {substrate}."
SAMPLE_ERROR = "This sample has no solution: {reason}"
TILE_RANGE = "Range: {low} – {high}"
RELATIVE_RADIUS_TIP = ("Radius of the resolvable islands. On a porous substrate it is a relative size: how it changes "
                       "over the scan counts, not its value.")
EMPTY_TITLE = "No result yet"
EMPTY_MESSAGE = ("In Situ: choose the folder of a scan recorded during the deposition, give the metal, the substrate "
                 "and the deposition rate, and press Analyze. After Deposition: choose the frames of a sample measured "
                 "afterwards and give its name and nominal thickness instead of a rate; frames of a bare piece of the "
                 "same substrate are optional. Open Result… shows a folder analysed before.")
EMPTY_ACTION = "Open Result…"
PACKAGE_TITLE = "The analysis package is needed"
PACKAGE_ACTION = "Choose Folder…"
PACKAGE_LINE = {"environment": "{folder} (from GIMAP_SPUTTER_CLIENT)", "chosen": "{folder}",
                "bundled": "{folder} (in the program folder)"}
PACKAGE_NONE = "not found"
GEOMETRY_ENERGY = "{energy} keV"
GEOMETRY_WAVELENGTH = "λ {wavelength} nm"
GEOMETRY_DISTANCE = "{distance} mm"
GEOMETRY_ALPHA = "αi {alpha}°"
GEOMETRY_CENTRE = "beam centre ({col}, {row}) px"
GEOMETRY_CENTRE_AUTO = "beam centre from the frames"
GEOMETRY_ROLL = "roll {roll}°"
GEOMETRY_TO_ENTER = "Geometry to enter: {names}"
GEOMETRY_MISSING = {"xray": "X-ray energy", "sdd": "sample-detector distance", "alpha": "incidence angle",
                    "beam": "both beam-centre coordinates"}
"""What the geometry line asks for (a field of the form that Analyze needs and that is empty)."""
WAITING_TO_ANALYZE = "Analyze starts once the header of the first frame is read…"
STARTING = "Starting the analysis…"
STOPPING_ANALYSIS = "Stopping…"
DONE = "Done: {windows} time windows."
SHOWN_AGAIN = "This request was analysed before: its result is shown ({windows} time windows)."
READY_TOAST = "The analysis of {name} is ready."
FAILED_TOAST = "The analysis failed."
CHOICE_NOT_SAVED = "The choice could not be saved: {reason}"
NO_REPORT = "This result has no report file."
OPENED_EARLIER = "This folder holds a result of the earlier method: it is shown in Details."
TEMPLATES = (ENTRY_CANDIDATE, ENTRY_REFINED, ENTRY_MODEL, ENTRY_TIP, ENTRY_TIP_WEIGHT, WINDOW_TEXT, WINDOW_TIP,
             WINDOW_TIP_ONSET, WINDOW_ERROR, *WINDOW_CONTROLS, *SAMPLE_CONTROLS, SAMPLE_TEXT, SAMPLE_TIP,
             SAMPLE_TIP_FRAMES, SAMPLE_TIP_SUBSTRATE, SAMPLE_ERROR, TILE_RANGE, RELATIVE_RADIUS_TIP,
             EMPTY_TITLE, EMPTY_MESSAGE, EMPTY_ACTION, PACKAGE_TITLE, PACKAGE_ACTION,
             *(line for line in PACKAGE_LINE.values() if line != "{folder}"), PACKAGE_NONE,
             GEOMETRY_CENTRE, GEOMETRY_CENTRE_AUTO, GEOMETRY_ROLL, GEOMETRY_TO_ENTER, *GEOMETRY_MISSING.values(),
             WAITING_TO_ANALYZE,
             STARTING, STOPPING_ANALYSIS, DONE, SHOWN_AGAIN, READY_TOAST, FAILED_TOAST, CHOICE_NOT_SAVED, NO_REPORT,
             OPENED_EARLIER, STEP_OF, STEP_REUSED, *STEP_TEXT.values(), MASK_FILE, MASK_AUTO, MASK_AUTO_ONLY)
"""The templates that need Chinese (the others hold only values, units and symbols)."""
_CANDIDATE = re.compile(r"candidate\s+(\d+)", re.IGNORECASE)
BREAK_AFTER = "\\/_-."
"""Characters of a folder after which a line may wrap."""
LONGEST_UNBROKEN = 12
"""Most characters of a folder without a place to wrap (a longer name gets one after so many)."""
ZERO_WIDTH_SPACE = "​"


def wrappable(text: str) -> str:
    """``text`` (a folder, or a line holding one) with a zero-width space after every separator, '_', '-' and '.', and
    after every LONGEST_UNBROKEN characters of a run without one: a label that wraps words can then break it at any
    font. A folder has no spaces: unbroken, its longest name sets the least width of the label and so of its column
    (macOS fonts are wider than Windows ones)."""
    out, run = [], 0
    for char in str(text):
        out.append(char)
        if char.isspace():
            run = 0
        elif char in BREAK_AFTER:
            out.append(ZERO_WIDTH_SPACE)
            run = 0
        else:
            run += 1
            if run >= LONGEST_UNBROKEN:
                out.append(ZERO_WIDTH_SPACE)
                run = 0
    return "".join(out)


def shown_number(value, digits: int = 3) -> str:
    """A value with ``digits`` significant digits ("" when it is missing)."""
    return "" if value is None else f"{float(value):.{digits}g}"


def entry_name(label: str) -> str:
    """The name of an entry in the interface language (a label the page does not know stays as it is)."""
    found = _CANDIDATE.fullmatch(label.strip())
    if found is not None:
        return text(ENTRY_CANDIDATE, rank=found.group(1))
    known = {"refined": ENTRY_REFINED, "model": ENTRY_MODEL}.get(label.strip().lower())
    return tr(known) if known else label


def entry_segment(entry) -> str:
    """What the selector says for an entry: its name and its fit deviation."""
    name = entry_name(entry.label)
    return name if entry.fit_deviation is None else ENTRY_SEGMENT.format(name=name,
                                                                        deviation=shown_number(entry.fit_deviation))


def entry_tip(entry) -> str:
    name, deviation = entry_name(entry.label), shown_number(entry.fit_deviation) or "–"
    if entry.weight is not None:
        return text(ENTRY_TIP_WEIGHT, name=name, deviation=deviation, weight=shown_number(entry.weight, 2))
    return text(ENTRY_TIP, name=name, deviation=deviation)


def window_text(window, static: bool = False) -> str:
    """What the selector says beside its slider: the time and the deposit of a time window, or — a series of static
    samples — the sample and its nominal thickness."""
    if static:
        label = window.label or shown_number(window.time_s)
        if window.nominal_nm is None:
            return label
        return text(SAMPLE_TEXT, label=label, thickness=shown_number(window.nominal_nm))
    time = shown_number(window.time_s, 4)
    if window.nominal_nm is None:
        return text(WINDOW_TEXT_PLAIN, time=time)
    return text(WINDOW_TEXT, time=time, deposit=shown_number(window.nominal_nm))


def window_tip(window, index: int, count: int, static: bool = False) -> str:
    if static:
        words = text(SAMPLE_TIP, index=index + 1, count=count) if window.frames is None else \
            text(SAMPLE_TIP_FRAMES, index=index + 1, count=count, frames=window.frames)
        return words + (" " + text(SAMPLE_TIP_SUBSTRATE, substrate=window.substrate) if window.substrate else "")
    if window.since_onset_s is None:
        return text(WINDOW_TIP, index=index + 1, count=count)
    return text(WINDOW_TIP_ONSET, index=index + 1, count=count, onset=shown_number(window.since_onset_s, 4))


def range_text(found, digits: int = 3) -> str:
    """The range of a value as the tooltip of its tile says it ("" without one)."""
    return "" if found is None else text(TILE_RANGE, low=shown_number(found[0], digits),
                                         high=shown_number(found[1], digits))


__all__ = ["CHOICE_NOT_SAVED", "DONE", "EMPTY_ACTION", "EMPTY_MESSAGE", "EMPTY_TITLE", "FAILED_TOAST",
           "GEOMETRY_ALPHA", "GEOMETRY_CENTRE", "GEOMETRY_CENTRE_AUTO", "GEOMETRY_DISTANCE", "GEOMETRY_ENERGY",
           "GEOMETRY_MISSING", "GEOMETRY_ROLL", "GEOMETRY_TO_ENTER", "GEOMETRY_WAVELENGTH", "NO_REPORT",
           "OPENED_EARLIER", "PACKAGE_ACTION", "PACKAGE_LINE", "PACKAGE_NONE", "PACKAGE_TITLE", "READY_TOAST",
           "MASK_AUTO", "MASK_AUTO_ONLY", "MASK_FILE", "RELATIVE_RADIUS_TIP", "RESULT_NAME", "SAMPLE_CONTROLS", "SAMPLE_ERROR", "SHOWN_AGAIN", "STARTING",
           "STOPPING_ANALYSIS", "TEMPLATES", "WAITING_TO_ANALYZE", "WINDOW_CONTROLS", "WINDOW_ERROR", "entry_name",
           "entry_segment", "entry_tip", "range_text", "shown_number", "window_text", "window_tip", "wrappable"]
