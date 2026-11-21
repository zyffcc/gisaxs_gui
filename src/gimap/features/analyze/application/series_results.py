"""The timeline of a Series map and its results database.

* **Timeline** (``domain/series_timeline``): the χ² of every frame to the frame before it and to a reference
  frame, with the frames where the series jumps; its table names the stage and the odd frames too, so a
  transition can be found in one look.
* **Results database** (``shared/series_store``): the curve of every row of a map, as it was reduced (both
  halves of a cut as measured), with its frame, file, time and the settings of the map; written as a map is
  built or a watched folder grows, read back to show the map again without its frames, or by Compare.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.gimap.shared.series_store import SeriesStore, read_curves
from src.gimap.shared.version import SOFTWARE as RELEASE

from ..domain.series_timeline import JUMP_LEAST, JUMP_SPREAD, Timeline, series_timeline
from .series_map import RowLabel, SeriesMap, _with_record, row_facts, series_header, series_times
from .ports import CurveWriter

TIMELINE_METHOD = (
    "chi2 of two frames: sum over the x columns both have of ((I_a - I_b) / s)^2 / (2 n), s the noise of the column "
    "from the second differences along the frames (1.4826 x median |I[i+1] - 2 I[i] + I[i-1]| / sqrt(6)), never "
    "below the usual column's: about 1 for frames that differ by noise alone; a jump: a local maximum of chi2 to the "
    f"previous frame above {JUMP_LEAST:g} and above the median by more than {JUMP_SPREAD:g} robust deviations"
)
"""How the timeline is made, for its table and record (Physics and Methods, chapter 11)."""


def timeline_of(series: SeriesMap, reference_row: int = 0, x_range: Optional[Sequence[float]] = None) -> Timeline:
    """The timeline of a map (``x_range``: only that part of its x)."""
    return series_timeline(series.x, series.image, reference_row=reference_row, x_range=x_range)


def export_series_timeline(writer: CurveWriter, series: SeriesMap, timeline: Timeline, path: Path, *,
                           stages=None) -> Path:
    """CSV: per frame its χ² to the frame before and to the reference frame, a jump or not, its stage and
    whether it is odd (``stages``: the map's ``SeriesStages`` when found), with the JSON record next to it."""
    reference = series.number(timeline.reference_row)
    odd = {frame.row: frame.reason for frame in getattr(stages, "odd", ())}
    comments = ["GIMaP Analyze series: the timeline - how far each frame is from the one before and from the "
                "reference frame", *series_header(series), f"reference frame: {reference}",
                f"method: {TIMELINE_METHOD}", f"noise of a column: {timeline.noise_from}; {timeline.columns} columns"]
    header = ["frame", "file", "chi2_to_previous", "chi2_to_reference", "jump", "stage", "odd"] + (
        ["time_s"] if series.times is not None else [])
    rows = []
    jumps = set(timeline.jumps)
    for row, label in enumerate(series.labels):
        stage = stages.stage_of(row) if stages is not None and stages.rows == series.rows else ""
        cells = [series.number(row), label, float(timeline.neighbour[row]), float(timeline.reference[row]),
                 row in jumps, stage, row in odd]
        rows.append(cells + ([float(series.times[row])] if series.times is not None else []))
    return _with_record(writer, series, Path(path), comments, header, rows, table="timeline", method=TIMELINE_METHOD,
                        reference_frame=reference, jumps=[series.number(row) for row in timeline.jumps],
                        noise=timeline.noise_from)


# -- the results database ----------------------------------------------------------------------------------------------


def store_rows(store: SeriesStore, name: str, rows: Sequence[tuple], curve: str, *, numbers: Optional[Mapping] = None,
               group: int = 1, only: Optional[set] = None) -> int:
    """Record the rows of a map (``(x, I, x_label, label, (path, frame_index))``, as reduced) under the series
    ``name``; ``only``: the ``(path, frame_index)`` of the rows to write (the new ones of a watch), else all.
    Each frame is written with its number and its time since the first row. Returns how many were written."""
    facts = row_facts(rows, numbers=numbers, group=group)
    times, _from = series_times(facts)
    items = []
    for row, ((x, y, x_label, label, ref), fact) in enumerate(zip(rows, facts)):
        key = (str(ref[0]), int(ref[1]))
        if only is not None and key not in only:
            continue
        items.append(dict(curve=curve, file=str(ref[0]), frame_index=int(ref[1]), frame=fact.number, x=x,
                          intensity=y, x_label=x_label, label=str(label), summed=fact.summed,
                          time_s=None if times is None else float(times[row]), timestamp=fact.timestamp))
    return store.put_curves(name, items)


def series_info(curve: str, title: str, built: Mapping[str, Any], kind: Optional[str]) -> dict:
    """What the database says a map was made with (its curve, cut, settings and software)."""
    return {"software_version": RELEASE, "kind": kind, "curve": curve, "curve_title": title,
            "cut": built.get("cut"), "cut_in_words": built.get("cut_text"), "sum_of": int(built.get("group") or 1),
            "settings": built.get("settings")}


def rows_from_store(path, series: Optional[str] = None, curve: Optional[str] = None) -> tuple[str, str, list, dict]:
    """``(series, curve, rows, info)``: the rows of a map recorded in a results database, as a map is built of them
    (``build_series_map``), and what the map was made with."""
    name, curve, curves = read_curves(path, series, curve)
    rows = []
    for item in curves:
        label = RowLabel(item.label or Path(item.file).name, name=Path(item.file).stem, source=item.file,
                         frame_index=item.frame_index, summed=item.summed, timestamp=item.timestamp)
        rows.append((np.asarray(item.x, dtype=float), np.asarray(item.intensity, dtype=float), item.x_label, label,
                     (Path(item.file), int(item.frame_index))))
    with SeriesStore(path, read_only=True) as store:
        info = store.info(name)
    return name, curve, rows, info


__all__ = ["TIMELINE_METHOD", "export_series_timeline", "rows_from_store", "series_info", "store_rows", "timeline_of"]
