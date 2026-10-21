"""The sentences of the Series tab (English templates; ``tr`` / ``trf`` say them in the interface language).

What the tab says of a map is composed here, so the title line, the plot titles, the read-out and the
traces name a frame and a map in the same words: the frame by its number on the frame axis (the frame in its
file, or its place in the list), a row that sums frames as such, a short last sum as incomplete.
"""

from __future__ import annotations

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.stage_text import axis_symbol

from .source_texts import short_name

LISTED_FILES = "{n} files listed: Build Map stacks the chosen curve of every frame."
LISTED_FRAMES = "{n} frames listed: Build Map stacks the chosen curve of every frame."
LISTED_SUMS = "{frames} frames in {groups} sums of {n}: Build Map stacks one curve per sum."
LISTED_SUMS_SHORT = "{frames} frames in {groups} sums of {n}: Build Map stacks one curve per sum; the last sums only {last}."
NEEDS_GEOMETRY = "The frame needs a geometry before its curves can be stacked."
CUT_OF_SERIES = "Cut of the series — {cut}."
"""After the sentence of the listed frames: ``{cut}`` is ``view_model.series_cut_text()``."""
HOW_TO = (
    "{rows} frames × {points} points. Click or drag the horizontal band to pick a frame, drag the vertical "
    "band (and its edges) to pick a {axis} window; Open shows the frame in Analyze."
)
HOW_TO_SUMS = (
    "{rows} rows (sums of {n} frames) × {points} points. Click or drag the horizontal band to pick a row, drag "
    "the vertical band (and its edges) to pick a {axis} window; Open shows the row's frames in Analyze."
)
"""… of a map whose rows are sums of N frames (a row is no frame)."""
MAP_DONE_SUMS = "Series map: {n} rows (sums of {group} frames) of {curve}"
MAP_DONE_SUMS_FAILED = "Series map: {n} rows (sums of {group} frames) of {curve}; {failed} failed: {names}"
MAP_TITLE = "{curve} — {n} frames"
MAP_TITLE_SUMS = "{curve} — {n} rows (sums of {group})"
RANGE_ONLY = "Build Map takes frames {first} to {last} only (Options)."
RANGE_FROM = "Build Map takes frames {first} to the last only (Options)."
OPTIONS_TITLE = "Options: frames, cut, axes"
OPTIONS_RANGE = "Options: frames {first} to {last}, cut, axes"
OPTIONS_FROM = "Options: frames {first} to the last, cut, axes"
"""The title of the folded options while a frame range other than every frame is set (it holds every later map
until it is changed or the list is cleared)."""
SUM_SWITCHED_OFF = "Sum {n} of the automatic analysis was switched off for the series (Data ▸ Sum N to sum again)."
"""The same sentence as the Batch Export dialog says (``bindings/batch_export.py``)."""
NO_FRAMES_IN_RANGE = "Frames {first} to {last} hold fewer than two frames: a series map needs at least two."

MAP_FRAMES = "{n} frames of {name}"
MAP_FRAMES_OF = "{n} of {listed} frames of {name}"
MAP_ROWS = "{n} rows of {name}"
"""… of a map whose rows are sums of frames."""
MAP_SUMS = "each row the sum of {n} frames"
MAP_SUMS_SHORT = "each row the sum of {n} frames; the last row sums only {last} of {n} frames"
MAP_TIME = {
    "timestamps": "time from the frames' time stamps", "exposure": "time = frame × exposure",
    "start_exposure": "time = start of the file + frame × exposure",
}
VIEW_WORDS = {
    "positive": "{axis} > 0 half", "negative": "{axis} < 0 half", "folded": "both halves on |{axis}|",
    "mean": "mean of both halves",
}
"""The halves a map of a signed cut holds, in its title line (nothing for both halves as measured)."""
STALE_SETTINGS = "Built with other settings — Build Map again."
STALE_CURVE = "The Curve box shows another curve ({curve}) — Build Map stacks it."

FRAME_TITLE = "Frame {n}: {label}"
FRAME_OF_TITLE = "Frame {n} of {total}: {name}"
SUM_OF = "sum of {n}"
SUM_SHORT = "sum of {n} of {group}"
FRAME_NAME = "frame {n}"
READOUT = "frame {n} ({label})"
TRACE_INTENSITY = "I at {axis} = {centre} ± {half}"
TRACE_PEAK = "Peak in {low}–{high}"
TRACE_NEAREST = "I at {axis} = {x} (the point nearest to the window)"
"""A window narrower than the merged bins around it: the trace is of the nearest point, said so."""
TRACE_NO_POINT = "No measured point in this window (a detector gap, or beyond the data): move the band onto the data."
NO_STAGES = "No stages: {reason}"
AXIS_LABELS = {"frame": "frame", "time": "time (s)"}
"""The frame axis of the map and of the traces (figure content: English, like every axis label)."""
PEAK_LABELS = {"position": "position {unit}", "fwhm": "2.355 σ {unit}", "area": "area", "height": "height"}
"""The y axis of a trace of the peak in the window, short enough for a plot 100 px high (``fwhm``: the moment
width, 2.355 × the standard deviation inside the window, see ``track_peak``)."""
PEAK_NAMES = {"position": "position", "fwhm": "width", "area": "area", "height": "height"}
"""… and its name in the plot's legend (not “fwhm”: it is no fitted FWHM)."""
TRACE_WORDS = {
    "intensity": "mean intensity in the window {low} - {high} {unit}",
    "nearest": "intensity of the point nearest to the window {low} - {high} {unit} (no point inside it), at {x} {unit}",
    "position": "centroid of the intensity above a straight background in the window {low} - {high} {unit}",
    "fwhm": "moment width (2.355 x the standard deviation of the intensity above a straight background) in the window "
            "{low} - {high} {unit}; it grows with the window - a fitted FWHM: Batch Export > Fit > Peaks",
    "area": "intensity above a straight background integrated over the window {low} - {high} {unit}",
    "height": "largest intensity above a straight background in the window {low} - {high} {unit}",
    "change": "place of every frame along the main change of the series (first component of the stages' comparison)",
}
"""What a trace is, for the table written of it (English: a record)."""
NO_TIME = "The frames of this map carry no time (no time stamp per frame, no exposure time of one file)."
STAGES_NEED = "Stages need {n} frames or more (this map has {rows})."
STAGES_NEED_EXPORT = "Stages are found in a map of {n} frames or more (this map has {rows})."
TEMPLATES = (
    LISTED_FILES, LISTED_FRAMES, LISTED_SUMS, LISTED_SUMS_SHORT, NEEDS_GEOMETRY, CUT_OF_SERIES, HOW_TO,
    HOW_TO_SUMS, MAP_DONE_SUMS, MAP_DONE_SUMS_FAILED, MAP_TITLE, MAP_TITLE_SUMS, RANGE_ONLY, RANGE_FROM,
    OPTIONS_TITLE, OPTIONS_RANGE, OPTIONS_FROM,
    SUM_SWITCHED_OFF, NO_FRAMES_IN_RANGE, MAP_FRAMES, MAP_FRAMES_OF, MAP_ROWS, MAP_SUMS, MAP_SUMS_SHORT,
    *MAP_TIME.values(),
    *VIEW_WORDS.values(), STALE_SETTINGS, STALE_CURVE, FRAME_TITLE, FRAME_OF_TITLE, SUM_OF, SUM_SHORT, FRAME_NAME,
    READOUT, TRACE_INTENSITY, TRACE_PEAK, TRACE_NEAREST, TRACE_NO_POINT, NO_STAGES, NO_TIME, STAGES_NEED,
    STAGES_NEED_EXPORT,
)
"""Every sentence above (each has its Chinese in ``i18n/zh_series_map.py``)."""


def cut_sentence(cut_text) -> str:
    """``series_cut_text()`` in the interface language (empty when there is nothing to say)."""
    template, values = cut_text if cut_text else ("", {})
    if not template:
        return ""
    return trf(template, **values) if values else tr(template)


def sum_note(fact) -> str:
    """“sum of 5”, “sum of 2 of 5” for a row that sums frames (empty for a single frame)."""
    if fact is None or (fact.summed <= 1 and fact.group <= 1):
        return ""
    return trf(SUM_SHORT, n=fact.summed, group=fact.group) if fact.incomplete else trf(SUM_OF, n=fact.summed)


def row_fact(series, row: int):
    return series.facts[row] if len(series.facts) == series.rows else None


def frame_title(series, row: int) -> str:
    """The title of the frame plot: “Frame 7 of 12: scan_00007” for a frame of one multi-frame source (its
    number is the frame in the file), “Frame 4: film_003.tif” otherwise, and what a row sums. A long name is
    shortened in its middle (the plot is narrow; its tooltip has the whole name)."""
    fact, label = row_fact(series, row), series.labels[row]
    name = getattr(label, "name", "")
    if fact is None or not name:  # nothing known but the label
        return trf(FRAME_TITLE, n=series.number(row), label=short_name(str(label)))
    if fact.frame_count > 1 and fact.number == fact.frame_index + 1:
        text = trf(FRAME_OF_TITLE, n=fact.number, total=fact.frame_count, name=short_name(name))
    elif fact.frame_count > 1:  # one of several multi-frame files: the label says which frame of its file
        return trf(FRAME_TITLE, n=fact.number, label=short_name(str(label)))
    else:
        text = trf(FRAME_TITLE, n=fact.number, label=short_name(name))
    note = sum_note(fact) if fact.summed > 1 or fact.incomplete else ""
    return f"{text} — {note}" if note else text


def map_group(series) -> int:
    """Frames a row of the map sums (the Sum N it was built with; 1: rows are frames)."""
    return max([fact.group for fact in series.facts] or [1])


def map_title(curve: str, series) -> str:
    """The title of the map's picture: “I(q) — 6 frames”, “I(q) — 3 rows (sums of 5)”."""
    group = map_group(series)
    return trf(MAP_TITLE_SUMS, curve=curve, n=series.rows, group=group) if group > 1 else trf(
        MAP_TITLE, curve=curve, n=series.rows)


def range_words(first: int, last: int, *, options: bool = False) -> str:
    """The frame range Build Map takes, when it is not every frame (``last`` 0: to the last frame): under the
    buttons, or (``options``) as the title of the folded options. Empty for every frame (``options``: the
    usual title)."""
    if int(first) <= 1 and not last:
        return tr(OPTIONS_TITLE) if options else ""
    if last:
        return trf(OPTIONS_RANGE if options else RANGE_ONLY, first=int(first), last=int(last))
    return trf(OPTIONS_FROM if options else RANGE_FROM, first=int(first))


def halves_words(series) -> str:
    """The halves a map of a signed cut holds (“mean of both halves”); nothing for both halves as measured."""
    return trf(VIEW_WORDS[series.view], axis=axis_symbol(series.x_label)) if series.view in VIEW_WORDS else ""


def map_words(series, name: str, listed: int = 0) -> list[str]:
    """The parts of the title line after the curve: the halves, the frames, the sums, where the time is from."""
    parts = [halves_words(series)] if halves_words(series) else []
    groups = [fact.group for fact in series.facts if fact.group > 1]
    if groups:  # (rows that are sums are not frames: “4 of 20 frames” would be wrong)
        parts.append(trf(MAP_ROWS, n=series.rows, name=name))
    elif listed and listed > series.rows:
        parts.append(trf(MAP_FRAMES_OF, n=series.rows, listed=listed, name=name))
    else:
        parts.append(trf(MAP_FRAMES, n=series.rows, name=name))
    if groups:
        short = [fact.summed for fact in series.facts if fact.incomplete]
        parts.append(trf(MAP_SUMS_SHORT, n=max(groups), last=short[-1]) if short else trf(MAP_SUMS, n=max(groups)))
    return parts


__all__ = [
    "AXIS_LABELS", "CUT_OF_SERIES", "FRAME_NAME", "FRAME_OF_TITLE", "FRAME_TITLE", "HOW_TO", "HOW_TO_SUMS",
    "LISTED_FILES", "LISTED_FRAMES", "LISTED_SUMS", "LISTED_SUMS_SHORT", "MAP_DONE_SUMS", "MAP_DONE_SUMS_FAILED",
    "MAP_FRAMES", "MAP_FRAMES_OF", "MAP_ROWS", "MAP_SUMS", "MAP_SUMS_SHORT", "MAP_TIME", "MAP_TITLE", "MAP_TITLE_SUMS",
    "NEEDS_GEOMETRY", "NO_FRAMES_IN_RANGE", "NO_STAGES", "NO_TIME", "OPTIONS_FROM", "OPTIONS_RANGE", "OPTIONS_TITLE",
    "PEAK_LABELS", "PEAK_NAMES", "RANGE_FROM", "RANGE_ONLY", "READOUT", "STAGES_NEED",
    "STAGES_NEED_EXPORT", "STALE_CURVE", "STALE_SETTINGS", "SUM_OF", "SUM_SHORT", "SUM_SWITCHED_OFF", "TEMPLATES",
    "TRACE_INTENSITY", "TRACE_NEAREST", "TRACE_NO_POINT", "TRACE_PEAK", "TRACE_WORDS", "VIEW_WORDS", "cut_sentence",
    "frame_title", "halves_words", "map_group", "map_title", "map_words", "range_words", "row_fact", "sum_note",
]
