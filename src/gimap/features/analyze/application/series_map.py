"""A series as one image: the same curve of every frame, stacked (intensity against frame and x).

What a row is travels with its name (``RowLabel``: the file, the frame in it, the frames summed, its time),
so the frame axis, the marks of a short last sum and every export say the same. Every series export
starts with the same lines (``series_header``): which curve and halves it holds, which list it was built
from, the cut of the series, and — for the map — a JSON record with the settings next to it.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..domain import SeriesMap, q_from_two_theta, series_map
from ..domain.series_map import VIEW_BOTH, VIEW_FOLDED, VIEW_MEAN, VIEW_NEGATIVE, VIEW_POSITIVE, RowFacts
from .models import FrameAnalysis
from .ports import FRAME_OF, CurveWriter, source_name
from .use_cases import SOFTWARE

SERIES_TITLE = "GIMaP Analyze series map: rows = frames in list order, columns = x; mean intensity per bin, blank = no data"
FRAMES_OF = "frames {first}–{last} of {n}"
"""The frames of a multi-frame source summed into one row, counted from 1 (``frame_text``)."""
ALSO_SUMMED = " (+{n} summed)"
"""After the name of a row that sums further frames (as the tables of a Batch Export name their columns)."""
VIEW_WORDS = {
    VIEW_BOTH: "both halves, on the signed {axis}",
    VIEW_POSITIVE: "the {axis} > 0 half",
    VIEW_NEGATIVE: "the {axis} < 0 half, on |{axis}|",
    VIEW_FOLDED: "both halves on |{axis}| (every measured point)",
    VIEW_MEAN: "mean of both halves on |{axis}| (a point of one half alone where the other has none)",
}
"""What the rows of a map of a signed cut hold, for its exports (``{axis}``: qy)."""
TIME_WORDS = {
    "timestamps": "seconds since the first row, from the time stamps of the frames",
    "exposure": "seconds since the first row, frame number x exposure time (no time stamp per frame in the file)",
    "start_exposure": "seconds since the first row, start time of each file + frame number x exposure time (no "
                      "time stamp per frame in the files)",
}
WIDTH_NOTE = (
    "fwhm is the moment width: 2.355 x the standard deviation of the intensity above the background inside the "
    "window. It is the FWHM of a Gaussian peak well inside the window and grows with the window; a fitted FWHM "
    "with its error: Batch Export > Fit > Peaks of the ring regions"
)
"""Said with the peak table and at the trace (``track_peak``)."""
_FACTS = ("name", "source", "frame_index", "frame_count", "summed", "timestamp", "exposure_s", "frame_time_s")


class RowLabel(str):
    """What a row of a series is called (“scan_00007 frame 4 of 12 (+4 summed)”), with the facts of its frame as
    attributes. It is a string wherever a name is wanted — table columns, records, titles — and comes
    back whole from a worker process."""

    name: str = ""
    """The source without its frame (``ports.source_name``)."""
    source: str = ""
    frame_index: int = 0
    frame_count: int = 1
    summed: int = 1
    """Frames summed into the row, this one included."""
    timestamp: str = ""
    exposure_s: Optional[float] = None
    frame_time_s: Optional[float] = None

    def __new__(cls, text: str = "", **facts):
        label = super().__new__(cls, text)
        for key, value in facts.items():
            if key not in _FACTS:
                raise TypeError(f"RowLabel has no fact {key!r}")
            setattr(label, key, value)
        return label


def frame_text(frame_index: int, frame_count: int, summed: int = 1) -> str:
    """“frame 4 of 12”, “frames 6–10 of 12” for a sum, nothing for the one frame of a single-frame file."""
    if int(frame_count) <= 1:
        return ""
    first = int(frame_index) + 1
    if int(summed) <= 1:
        return FRAME_OF.format(i=first, n=int(frame_count))
    return FRAMES_OF.format(first=first, last=min(first + int(summed) - 1, int(frame_count)), n=int(frame_count))


def series_row(analysis: FrameAnalysis, key: str) -> tuple[np.ndarray, np.ndarray, str]:
    """``(x, intensity, x_label)`` of one curve of a frame; 2θ is converted to q (Å⁻¹)."""
    reduction = analysis.reduction
    curve = reduction.curve(key) if reduction is not None else None
    if curve is None or curve.is_empty:
        raise ValueError(f"{Path(analysis.path).name} has no '{key}' curve (it needs a geometry and this reduction).")
    x, label = np.asarray(curve.x, dtype=float), curve.x_label
    if label.startswith("2θ") and analysis.geometry is not None:
        x, label = q_from_two_theta(x, analysis.geometry.wavelength_angstrom), "q (Å⁻¹)"
    return x, np.asarray(curve.intensity, dtype=float), label


def row_label(analysis: FrameAnalysis) -> RowLabel:
    """What a frame is called as a row of the map: its file — a set of module files by its scan, without
    “_m01” —, the frame of a multi-frame source counted from 1 (“frame 4 of 12”), and how many further
    frames were summed with it. The same words as the columns of a Batch Export's tables."""
    metadata = analysis.metadata or {}
    name = source_name(analysis.path, metadata)
    count = int(analysis.frame_count or 1)
    summed = 1 + len(analysis.summed_frames or ())
    text = name + (" " + frame_text(analysis.frame_index, count) if count > 1 else "")
    text += ALSO_SUMMED.format(n=summed - 1) if summed > 1 else ""
    return RowLabel(
        text, name=name, source=str(analysis.path), frame_index=int(analysis.frame_index), frame_count=count,
        summed=summed, timestamp=str(metadata.get("timestamp") or ""), exposure_s=_number(metadata.get("exposure_time_s")),
        frame_time_s=_number(metadata.get("frame_time_s")),
    )


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def frame_key(path, frame_index: int) -> tuple[str, int]:
    """How a frame is looked up among the listed ones (the path whatever the case of its letters)."""
    return str(path).casefold(), int(frame_index)


def row_facts(rows: Sequence[tuple], *, numbers: Optional[Mapping] = None, group: int = 1) -> tuple[RowFacts, ...]:
    """``RowFacts`` of ``rows`` (``(x, intensity, x_label, label, ref)``): from each label (``RowLabel``) and
    ref. ``numbers``: ``frame_key`` → the frame's number in the list, from 1 (the row when it is not there);
    ``group``: the Sum N in force, so a row summing fewer frames is known as incomplete."""
    facts = []
    for row, (_x, _y, _x_label, label, ref) in enumerate(rows):
        path, index = (ref[0], int(ref[1])) if ref else (getattr(label, "source", ""), int(getattr(label, "frame_index", 0)))
        summed = int(getattr(label, "summed", group if not isinstance(label, RowLabel) else 1))
        facts.append(RowFacts(
            number=int((numbers or {}).get(frame_key(path, index), row + 1)), source=str(path), frame_index=index,
            frame_count=int(getattr(label, "frame_count", 1)), summed=summed, group=max(1, int(group)),
            timestamp=str(getattr(label, "timestamp", "") or ""), exposure_s=getattr(label, "exposure_s", None),
            frame_time_s=getattr(label, "frame_time_s", None),
        ))
    return tuple(facts)


_STAMP = re.compile(
    r"(?P<moment>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?)(?:[.,](?P<fraction>\d+))?\s*"
    r"(?P<zone>Z|[+-]\d{2}(?::?\d{2})?)?")


def _seconds(stamp: str) -> Optional[float]:
    """Seconds of an ISO time stamp, also as detectors write it: “+0200” for “+02:00” (the P03 Lambda), “Z”, a
    space for the “T”, a fraction of any length (Python 3.10 reads only 3 or 6 digits and “+HH:MM”)."""
    text = str(stamp or "").strip()
    found = _STAMP.fullmatch(text)
    if found is not None:
        zone = found["zone"] or ""
        zone = "+00:00" if zone == "Z" else (zone if len(zone) != 5 else f"{zone[:3]}:{zone[3:]}")
        zone = f"{zone}:00" if len(zone) == 3 else zone
        fraction = f".{found['fraction'][:6].ljust(6, '0')}" if found["fraction"] else ""
        text = found["moment"].replace(" ", "T") + fraction + zone
    try:
        return datetime.fromisoformat(text).timestamp() if text else None
    except ValueError:
        return None


def _per_frame(facts: Sequence[RowFacts], values: Sequence[float]) -> bool:
    """Times that are those of the frames: they differ between the rows of one source (a file's start time,
    the same for each of its frames, is no time of a frame) and never run backwards."""
    last: dict[str, float] = {}
    for fact, value in zip(facts, values):
        key = fact.source.casefold()
        if key in last and not value > last[key]:
            return False
        last[key] = value
    return bool(np.all(np.diff(np.asarray(values, dtype=float)) >= 0))


def series_times(facts: Sequence[RowFacts]) -> tuple[Optional[np.ndarray], str]:
    """``(seconds since the first row, where they come from)``, or ``(None, "")`` when the frames carry no
    time. In this order: a time of each frame (a per-frame dataset, or time stamps that differ between the
    rows of a source and never run backwards: ``timestamps``), else — for multi-frame sources with one
    exposure time — frame number × exposure (``exposure``, one source) or the start time of each file + frame
    number × exposure (``start_exposure``, several); both a lower bound: the read-out between frames is not
    in the file."""
    if len(facts) < 2:
        return None, ""
    starts = [_seconds(fact.timestamp) for fact in facts]
    for values in ([fact.frame_time_s for fact in facts], starts):
        if all(value is not None for value in values) and _per_frame(facts, values):
            times = np.asarray(values, dtype=float) - float(values[0])
            if times[-1] > 0:
                return times, "timestamps"
    exposures = {fact.exposure_s for fact in facts}
    if len(exposures) != 1 or exposures == {None} or float(next(iter(exposures))) <= 0:
        return None, ""
    if not any(fact.frame_count > 1 for fact in facts):
        return None, ""
    exposure = float(next(iter(exposures)))
    if len({fact.source.casefold() for fact in facts}) == 1:
        return np.asarray([(fact.frame_index - facts[0].frame_index) * exposure for fact in facts], dtype=float), "exposure"
    if any(start is None for start in starts):
        return None, ""
    times = np.asarray([start + fact.frame_index * exposure for fact, start in zip(facts, starts)], dtype=float)
    times -= times[0]
    if np.all(np.diff(times) >= 0) and times[-1] > 0:
        return times, "start_exposure"
    return None, ""


def build_series_map(
    rows: Sequence[tuple[np.ndarray, np.ndarray, str, str, tuple]], curve: str, *, view: str = VIEW_BOTH,
    title: str = "", numbers: Optional[Mapping] = None, group: int = 1, built_with: Optional[Mapping] = None,
) -> SeriesMap:
    """``rows``: ``(x, intensity, x_label, label, ref)`` in frame order. ``view``: the halves of a signed cut
    (``domain.series_map.VIEWS``); ``title``: the curve's English title; ``numbers`` and ``group``: see
    ``row_facts``; ``built_with``: what the exports say the map was built with."""
    if not rows:
        raise ValueError("No frame gave the curve.")
    facts = row_facts(rows, numbers=numbers, group=group)
    times, time_from = series_times(facts)
    return series_map(
        [(x, y) for x, y, _label, _name, _ref in rows], [name for *_rest, name, _ref in rows],
        x_label=rows[0][2], curve=curve, refs=[ref for *_rest, ref in rows], view=view, title=title, facts=facts,
        times=times, time_from=time_from, built_with=dict(built_with or {}),
    )


# -- what every export of a series says ---------------------------------------------------------------------------------


HALVES_CURVE = "horizontal"
"""The curve with halves: the horizontal GISAXS cut, I(qy) on both sides of the beam (I(χ) runs from −90° to 90°
and has none)."""


def axis_name(series: SeriesMap) -> str:
    """The x of a map in words for its records: “q (Å⁻¹)” → “q”, “|qy| (Å⁻¹)” → “qy”, “χ (°)” → “χ”."""
    return series.x_label.split(" (")[0].split(" or ")[0].strip().strip("|") or "x"


def _unit(series: SeriesMap) -> str:
    return series.x_label[series.x_label.find("("):] if "(" in series.x_label else ""


def view_words(series: SeriesMap) -> str:
    """The halves the rows hold, in words (English; nothing for a curve without halves)."""
    if series.curve != HALVES_CURVE:
        return ""
    signed = series.view != VIEW_BOTH or (series.x.size > 1 and float(np.nanmin(series.x)) < 0 < float(np.nanmax(series.x)))
    return VIEW_WORDS[series.view].format(axis=axis_name(series)) if signed else ""


def row_words(series: SeriesMap, row: int) -> str:
    """One row for a ``#`` line: its number, the file with its folder, the frame in it, the sum, the time."""
    label = series.labels[row]
    if len(series.facts) != series.rows:
        return f"frame {row + 1}: {label}"
    fact = series.facts[row]
    parts = [fact.source or str(label)]
    frames = frame_text(fact.frame_index, fact.frame_count, fact.summed)
    if frames:
        parts.append(frames)
    if fact.group > 1 or fact.summed > 1:
        parts.append(f"sum of {fact.summed} of {fact.group} (incomplete)" if fact.incomplete else f"sum of {fact.summed}")
    if series.times is not None:
        parts.append(f"t = {float(series.times[row]):.6g} s")
    return f"frame {fact.number}: " + " | ".join(parts)


def series_header(series: SeriesMap) -> list[str]:
    """The lines every export of a series starts with, after its own first line."""
    built = series.built_with
    lines = [f"software: {SOFTWARE}", f"curve: {series.curve}" + (f" - {series.title}" if series.title else "")]
    halves = view_words(series)
    if halves:
        lines.append(f"halves: {halves}")
    if built.get("name"):
        listed = f"; {series.rows} of {built['listed']} listed frames" if built.get("listed") else ""
        every = f", every {built['every']}. frame" if int(built.get("every") or 1) > 1 else ""
        lines.append(f"series: {built['name']}{listed}{every}")
    if built.get("cut_text"):
        lines.append(f"cut of the series: {built['cut_text']}")
    groups = {fact.group for fact in series.facts}
    if groups - {1}:
        short = [fact.number for fact in series.facts if fact.incomplete]
        lines.append(f"sum: each row is the sum of {max(groups)} frames (intensities are sums, not means)" + (
            f"; incomplete: frame {', '.join(str(number) for number in short)}" if short else ""))
    lines.append("frame: the number of the frame in the listed series, counted from 1 (for one file: the frame in the file)")
    if series.times is not None:
        lines.append(f"time: {TIME_WORDS.get(series.time_from, series.time_from)}")
    return lines


def series_record(series: SeriesMap) -> dict[str, Any]:
    """The JSON record written next to a series export: what the map holds and what it was built with."""
    built = series.built_with
    times = None if series.times is None else [float(value) for value in series.times]
    return {
        "software": SOFTWARE, "title": "Series map", "curve": series.curve, "curve_title": series.title,
        "x_label": series.x_label, "halves": series.view, "halves_in_words": view_words(series),
        "series": built.get("name"), "listed_frames": built.get("listed"), "every": built.get("every"),
        "cut_of_the_series": built.get("cut"), "cut_in_words": built.get("cut_text"),
        "time_from": series.time_from or None,
        "rows": [
            {"frame": fact.number, "label": str(series.labels[row]), "file": fact.source,
             "frame_index": fact.frame_index, "frame_count": fact.frame_count, "summed_frames": fact.summed,
             "sum_of": fact.group, "incomplete_sum": fact.incomplete,
             "time_s": None if times is None else times[row], "timestamp": fact.timestamp or None}
            for row, fact in enumerate(series.facts)
        ],
        "settings": built.get("settings"),
        "settings_of_first_frame": built.get("first_frame"),
    }


def _with_record(writer: CurveWriter, series: SeriesMap, path: Path, comments: list, header: list, rows: list,
                 **more) -> Path:
    """The table at ``path`` and, next to it, ``<name>.json``: what the map holds and the settings it was built
    with (``series_record``; ``more``: what this table adds). Every table of a series says where its numbers
    come from and how they were made."""
    path = Path(path)
    comments = [*comments, f"record with the settings: {path.with_suffix('.json').name}"]
    written = writer.write_table(path, comments, header, rows)
    writer.write_record(path.with_suffix(".json"), {**series_record(series), **more})
    return written


def export_series_track(writer: CurveWriter, series: SeriesMap, track, path: Path) -> Path:
    """The peak followed through the series (``track_peak``) as a CSV table, one row per frame."""
    unit = _unit(series)
    low, high = track.window
    comments = [
        "GIMaP Analyze series: the peak in one window of every frame (centroid and width above a straight background)",
        *series_header(series),
        f"window: {low:.6g} - {high:.6g} {unit}",
        f"position and fwhm in {unit.strip('()') or 'the unit of x'}; area: intensity x {unit.strip('()') or 'x'} "
        "(integrated over x); height: intensity above the background",
        WIDTH_NOTE,
    ]
    rows = [
        [series.number(index), label, float(track.position[index]), float(track.fwhm[index]), float(track.area[index]),
         float(track.height[index])]
        for index, label in enumerate(series.labels)
    ]
    return _with_record(writer, series, Path(path), comments, ["frame", "file", "position", "fwhm", "area", "height"],
                        rows, table="peak in a window", window=[float(low), float(high)], width=WIDTH_NOTE)


def export_series_map(writer: CurveWriter, series: SeriesMap, path: Path) -> Path:
    """CSV: the x of the map in the first row (as measured, not the picture's even grid), the frame number in
    the first column, a ``#`` line per frame; next to it ``<name>.json`` with the settings (``series_record``)."""
    comments = [SERIES_TITLE, *series_header(series), *(row_words(series, row) for row in range(series.rows)),
                f"rows: frame (first column); columns: {series.x_label} (first row)"]
    header = [f"frame \\ {series.x_label}", *(f"{value:.8g}" for value in series.x)]
    rows = [[series.number(row), *(float(value) for value in series.image[row])] for row in range(series.rows)]
    return _with_record(writer, series, Path(path), comments, header, rows, table="map")


def export_series_curve(writer: CurveWriter, series: SeriesMap, row: int, path: Path) -> Path:
    """The curve of one row as a table (x, intensity), as it is in the map."""
    row = int(np.clip(row, 0, series.rows - 1))
    comments = ["GIMaP Analyze series: the curve of one frame, as it is in the map (blank = no data)",
                *series_header(series), row_words(series, row)]
    values = series.profile(row)
    rows = [[float(x), float(value)] for x, value in zip(series.x, values)]
    return _with_record(writer, series, Path(path), comments, [series.x_label, "intensity"], rows,
                        table="curve of one frame", frame=series.number(row))


def export_series_trace(writer: CurveWriter, series: SeriesMap, name: str, values, path: Path, *, what: str) -> Path:
    """One value per frame (the trace under the map) as a table: frame, file, the value (``name``).
    ``what``: one English line saying how the value was made (the window, the kind)."""
    comments = ["GIMaP Analyze series: one value per frame (the trace of the map)", *series_header(series), what]
    header = ["frame", "file", name] + (["time_s"] if series.times is not None else [])
    rows = []
    for row, label in enumerate(series.labels):
        cells = [series.number(row), label, float(values[row])]
        rows.append(cells + ([float(series.times[row])] if series.times is not None else []))
    return _with_record(writer, series, Path(path), comments, header, rows, table="one value per frame", value=name,
                        what=what)


__all__ = [
    "ALSO_SUMMED", "FRAMES_OF", "HALVES_CURVE", "RowLabel", "SERIES_TITLE", "TIME_WORDS", "VIEW_WORDS", "WIDTH_NOTE",
    "axis_name", "build_series_map",
    "export_series_curve", "export_series_map", "export_series_trace", "export_series_track", "frame_key",
    "frame_text", "row_facts", "row_label", "row_words", "series_header", "series_record", "series_row",
    "series_times", "view_words",
]
