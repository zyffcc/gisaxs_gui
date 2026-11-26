"""Events of a run — spray pulses, shutter openings, a temperature step — placed on the rows of a series map.

An events file is a CSV table with a header line and one event per line. Its time column is the first whose name is
``utc``, ``time``, ``timestamp``, ``datetime`` or ``date`` (any case), holding ISO times (``2027-01-20T14:00:08.55``,
with or without a time zone); an optional ``label`` (or ``event``, ``pulse``, ``name``) column names the event, else
it is numbered. A time without a zone is local time, like the time stamps of most detector headers and the
modification times of files.

An event lies on the first row measured at or after it: the row's time (``series_times``) from the absolute time of
the map's first row (its file time or header time stamp). An event before the first row or after the last row is
outside the map.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .series_times import _seconds

TIME_COLUMNS = ("utc", "time", "timestamp", "datetime", "date")
LABEL_COLUMNS = ("label", "event", "pulse", "name")
NO_TIME_COLUMN = "{name} has no time column (utc, time, timestamp, datetime or date) with ISO times."
NO_EVENTS = "{name} holds no event with a time."


@dataclass(frozen=True)
class Event:
    label: str
    seconds: float
    """Seconds since the epoch (a time without a zone taken as local time)."""
    text: str
    """The time as the file writes it."""


def read_events(path: Path) -> tuple[Event, ...]:
    """The events of a CSV file, in time order (``ValueError`` with the reason when there are none)."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(NO_EVENTS.format(name=path.name))
    header = [cell.strip().casefold() for cell in rows[0]]
    time_index = next((header.index(name) for name in TIME_COLUMNS if name in header), None)
    if time_index is None:
        raise ValueError(NO_TIME_COLUMN.format(name=path.name))
    label_index = next((header.index(name) for name in LABEL_COLUMNS if name in header), None)
    events = []
    for number, row in enumerate(rows[1:], start=1):
        if len(row) <= time_index:
            continue
        seconds = _seconds(row[time_index].strip())
        if seconds is None:
            continue
        label = row[label_index].strip() if label_index is not None and len(row) > label_index else ""
        events.append(Event(label or str(number), float(seconds), row[time_index].strip()))
    if not events:
        raise ValueError(NO_EVENTS.format(name=path.name))
    return tuple(sorted(events, key=lambda event: event.seconds))


def first_row_time(series) -> Optional[float]:
    """Seconds since the epoch of the map's first row, from its file time or header time stamp (``None``: the map
    has no time, or its first row carries none)."""
    if series.times is None or not series.facts:
        return None
    fact = series.facts[0]
    for text in ((fact.file_time if series.time_from == "file_times" else ""), fact.timestamp, fact.file_time):
        seconds = _seconds(text) if text else None
        if seconds is not None:
            return float(seconds)
    return None


def event_rows(series, events: Sequence[Event]) -> list[tuple[Event, Optional[int], Optional[float]]]:
    """``(event, row, seconds after the first row)`` of every event: the first row measured at or after it, or
    ``None`` when it lies before the first row or after the last one (or the map has no absolute time)."""
    origin = first_row_time(series)
    if origin is None:
        return [(event, None, None) for event in events]
    times = np.asarray(series.times, dtype=float)
    placed = []
    for event in events:
        after = event.seconds - origin
        rows = np.flatnonzero(times >= after - 1e-9)
        inside = after >= times[0] - 1e-9 and after <= times[-1] + 1e-9 and rows.size
        placed.append((event, int(rows[0]) if inside else None, after))
    return placed


def events_record(placed) -> list[dict]:
    """What a series export says of the events: each with its time, its row's frame and the seconds after row 1."""
    return [{"label": event.label, "time": event.text, "row": None if row is None else row + 1,
             "seconds_after_first_row": None if after is None else round(float(after), 6)}
            for event, row, after in placed]


__all__ = ["Event", "NO_EVENTS", "NO_TIME_COLUMN", "event_rows", "events_record", "first_row_time", "read_events"]
