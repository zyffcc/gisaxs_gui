"""What a running Analyze has found so far: the live view of the main screen, from the package's ``@live`` lines.

While the analysis package works, it writes lines ``@live {json}`` between its step lines (the package's module
``iampred.live`` is the contract). Each is an event of one step (``step``: ``intake``, ``predict``, ``present``) of a
kind (``kind``); one with pictures names a small ``.npz`` file (``arrays``), which the infrastructure reads in the
background (``LiveEvent.arrays``). A package that writes no such lines is followed by its step lines alone.

The events, by step:

* reading the frames (``intake``): ``centre`` — the beam centre found in the frames; ``detector`` — the summed frames
  the automatic mask was made from (log10 of the counts on blocks of pixels), the share of each block the mask leaves
  out, the beam centre and the sample horizon; ``frames`` — one chunk of frames read (how many of how many, their
  times, the count rate of each frame and their count rate on the grid of q_y and exit angle); ``imported`` — the
  scan is read (frames, times, the start of the deposition when it was found);
* finding the solutions (``predict``): ``models`` — the networks the analysis runs, the first being the current one;
  ``windows`` — a network starts on that many time windows; ``window`` — one window solved: its best candidate (and,
  when the mode refines, later its refined solution) with the quantities of the main screen, the fit deviation, and
  the measured image beside the model image; ``chosen`` — the network whose solutions fit the scan better;
* drawing the images (``present``): ``drawing`` — how many windows; ``drawn`` — one window drawn.

``LiveRun`` keeps what the page shows: the pictures of the latest events and every solution found so far, as a
``ClientResult`` (one window per time, its entries the solutions of the network) that the main screen's plots draw.
Nothing here is a result: the result is what the package writes at the end, and the page shows it then.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.gimap.shared.geometry.conventions import canonical_from_index_center

from ..domain import ClientEntry, ClientResult, ClientWindow, WindowPatterns
from ..domain.client import number

LIVE_PREFIX = "@live "
LIVE_STEPS = ("intake", "predict", "report", "present")


@dataclass(frozen=True)
class LiveEvent:
    """One ``@live`` line: its kind, its step, its fields and the arrays of its file (empty when it has none or they
    could not be read)."""

    kind: str
    step: str = ""
    fields: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)

    def get(self, key: str, default=None):
        return self.fields.get(key, default)

    def number(self, key: str) -> Optional[float]:
        return number(self.fields.get(key))


def parse_live(line: str) -> Optional[tuple]:
    """(the event's fields, the path of its arrays or None) of an ``@live`` line; None for any other line or one that
    cannot be read."""
    if not isinstance(line, str) or not line.startswith(LIVE_PREFIX):
        return None
    try:
        record = json.loads(line[len(LIVE_PREFIX):])
    except ValueError:
        return None
    if not isinstance(record, dict) or not isinstance(record.get("kind"), str):
        return None
    arrays = record.pop("arrays", None)
    return record, (Path(arrays) if isinstance(arrays, str) and arrays else None)


def live_event(record: dict, arrays: Optional[dict] = None) -> LiveEvent:
    fields = {key: value for key, value in record.items() if key not in ("kind", "step")}
    return LiveEvent(str(record["kind"]), str(record.get("step") or ""), fields, dict(arrays or {}))


@dataclass
class LiveRun:
    """Everything the live view shows of one run (fed event by event, in the GUI thread)."""

    centre: Optional[tuple] = None
    detector: Optional[LiveEvent] = None
    frames: Optional[LiveEvent] = None
    """The latest chunk of frames read."""
    frames_done: int = 0
    frames_total: int = 0
    trace: dict = field(default_factory=dict)
    """Time of a frame (s) -> its count rate per pixel."""
    imported: Optional[LiveEvent] = None
    families: list = field(default_factory=list)
    mode: str = ""
    family: str = ""
    """The network at work (or that worked last)."""
    totals: dict = field(default_factory=dict)
    """Network -> how many time windows it solves."""
    refine: int = 0
    solutions: dict = field(default_factory=dict)
    """Network -> time -> {entry label: (ClientEntry, since onset, nominal deposit)}."""
    latest: dict = field(default_factory=dict)
    """Network -> the latest window it solved (with its images)."""
    chosen: str = ""
    deviations: dict = field(default_factory=dict)
    drawing: int = 0
    drawn: int = 0

    def feed(self, event: LiveEvent) -> str:
        """Take one event; returns its kind when the view changes ("" for an event it does not know)."""
        kind = event.kind
        if kind == "centre":  # the package counts pixel centres from 0: GIMaP's pixel coordinates are shown
            col, row = event.number("col"), event.number("row")
            self.centre = canonical_from_index_center(col, row) if col is not None and row is not None else None
        elif kind == "detector":
            self.detector = event if "image" in event.arrays else None
        elif kind == "frames":
            self.frames_done = int(event.number("done") or self.frames_done)
            self.frames_total = int(event.number("total") or self.frames_total)
            for pair in event.get("trace") or ():
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    time_s, rate = number(pair[0]), number(pair[1])
                    if time_s is not None and rate is not None:
                        self.trace[time_s] = rate
            if "rate" in event.arrays:
                self.frames = event
        elif kind == "imported":
            self.imported = event
            self.frames_done = int(event.number("frames") or self.frames_done)
            self.frames_total = max(self.frames_total, self.frames_done)
        elif kind == "models":
            self.families = [str(name) for name in event.get("families") or () if name]
            self.mode = str(event.get("mode") or "")
        elif kind == "windows":
            self.family = str(event.get("family") or "")
            self.totals[self.family] = int(event.number("total") or 0)
            self.refine = int(event.number("refine") or 0)
            self.solutions.setdefault(self.family, {})
        elif kind == "window":
            if not self._solution(event):
                return ""
        elif kind == "chosen":
            self.chosen = str(event.get("family") or "")
            found = event.get("deviations")
            self.deviations = {str(k): number(v) for k, v in found.items()} if isinstance(found, dict) else {}
        elif kind == "drawing":
            self.drawing, self.drawn = int(event.number("total") or 0), 0
        elif kind == "drawn":
            self.drawn += 1
        else:
            return ""
        return kind

    def _solution(self, event: LiveEvent) -> bool:
        time_s, label = event.number("t"), str(event.get("label") or "")
        if time_s is None or not label:
            return False
        family = str(event.get("family") or self.family)
        values = event.get("values") if isinstance(event.get("values"), dict) else {}
        entry = ClientEntry(label, str(event.get("entry_kind") or "candidate"), fit_deviation=event.number("fit_deviation"),
                            values={key: value for key, value in ((k, number(v)) for k, v in values.items())
                                    if value is not None})
        self.solutions.setdefault(family, {}).setdefault(time_s, {})[label] = (
            entry, event.number("t_rel"), event.number("h_nom"))
        if {"q", "af", "measured", "model"} <= set(event.arrays):
            self.latest[family] = event
        return True

    # -- what the page shows ----------------------------------------------------------------------------------------
    def network_index(self) -> tuple:
        """(1-based place of the network at work among those the analysis runs, how many it runs)."""
        if self.family in self.families:
            return self.families.index(self.family) + 1, len(self.families)
        return (1, 1) if self.family else (0, len(self.families))

    def solved(self, family: Optional[str] = None) -> int:
        return len(self.solutions.get(self.family if family is None else family, {}))

    def trace_points(self) -> tuple:
        """(times, count rates) of the frames read so far, in the order of time."""
        times = sorted(self.trace)
        return times, [self.trace[time_s] for time_s in times]

    def result(self, folder: Path, family: Optional[str] = None) -> Optional[ClientResult]:
        """The solutions of a network so far as a result (one window per time; None before the first)."""
        family = self.family if family is None else family
        found = self.solutions.get(family)
        if not found:
            return None
        windows = []
        for time_s in sorted(found):
            entries = found[time_s]
            first = next(iter(entries.values()))
            windows.append(ClientWindow(time_s, first[1], first[2], entries=tuple(item[0] for item in entries.values())))
        return ClientResult(Path(folder), scan="", family=family, windows=tuple(windows))

    def shown_family(self) -> str:
        """The network whose solutions the view shows: the one the data chose, else the one at work."""
        return self.chosen or self.family

    def latest_patterns(self, family: Optional[str] = None) -> Optional[tuple]:
        """(WindowPatterns, entry label, time, the event) of the latest window a network solved (by default the one
        shown), or None."""
        event = self.latest.get(self.shown_family() if family is None else family)
        if event is None:
            return None
        try:
            arrays = event.arrays
            q, af = (_vector(arrays[key]) for key in ("q", "af"))
            measured, model = (_matrix(arrays[key]) for key in ("measured", "model"))
        except (KeyError, TypeError, ValueError):
            return None
        if measured.shape != (af.size, q.size) or model.shape != measured.shape or q.size < 2 or af.size < 2:
            return None
        label = str(event.get("label"))
        return WindowPatterns(q, af, measured, model[None], (label,)), label, event.number("t"), event


def _vector(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def _matrix(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


__all__ = ["LIVE_PREFIX", "LIVE_STEPS", "LiveEvent", "LiveRun", "live_event", "parse_live"]
