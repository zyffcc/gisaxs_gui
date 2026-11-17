"""What a running Analyze has found so far: the live view of the main screen, from the package's ``@live`` lines.

While the analysis package works, it writes lines ``@live {json}`` between its step lines (the package's module
``iampred.live`` is the contract). Each is an event of one step (``step``: ``intake``, ``predict``, ``present``) of a
kind (``kind``); one with pictures names a small ``.npz`` file (``arrays``), which the infrastructure reads in the
background (``LiveEvent.arrays``). A package that writes no such lines is followed by its step lines alone.

The events, by step:

* reading the frames (``intake``): ``centre`` — the beam centre found in the frames; ``detector`` — the summed frames
  the automatic mask was made from (log10 of the counts on blocks of pixels), the share of each block the mask leaves
  out, the beam centre and the sample horizon; ``frames`` — one chunk of frames read (how many of how many, their
  times, the count rate of each frame and their count rate on the grid of q_y and exit angle; a newer package adds
  ``frames``, ``read_s``, ``wall_s`` and ``mb``: the frames of the chunk, the seconds its process spent opening and
  reading their files from disk, its whole seconds and the MB read); ``reading`` — at the end of the import, how it
  spent its time (``frames_read``, ``frames_total``, ``workers``, ``seconds``, ``read_s_per_frame``, ``read_share``,
  ``mb_read``, ``slow``); ``imported`` — the scan is read (frames, times, the start of the deposition when it was
  found);
* finding the solutions (``predict``): ``models`` — the networks the analysis runs, the first being the current one;
  ``windows`` — a network starts on that many time windows; ``window`` — one window solved: its best candidate (and,
  when the mode refines, later its refined solution) with the quantities of the main screen, the fit deviation, and
  the measured image beside the model image; ``chosen`` — the network whose solutions fit the scan better;
  ``progress`` — how far a phase of a network has come, sent by each worker process for its own share (``done``,
  for the refinement also ``evaluations``: model images computed) at most once a second: ``features`` (the network
  input of each window), ``network`` (the network's candidates for all windows, 0 at its start and all at its end),
  ``render`` (the candidates of each window drawn by the model), ``refine`` (each window's best candidate refined),
  ``sequence`` (the whole-scan refinement: its iteration); ``preliminary`` — the network's result is written (its
  folder) while the refinement goes on; ``refine`` — the refinement of every window and the joint fit of the whole scan
  start (``iterations``); a newer package then says ``joint_stage`` (``stage`` ``pool``: the solutions of neighbouring
  windows are drawn at each window; ``fit``: the iterations start; ``windows``, ``total``) and, after every iteration,
  ``joint`` (``iteration``, ``total``, ``seconds`` since the fit started, ``median`` fit deviation and ``windows``: per
  window ``t`` and the quantities of the main screen of the current solution);
* drawing the images (``present``): ``drawing`` — how many windows; ``drawn`` — one window drawn.

``LiveRun`` keeps what the page shows: the pictures of the latest events and every solution found so far, as a
``ClientResult`` (one window per time, its entries the solutions of the network) that the main screen's plots draw,
and what the run is doing now (``activity``: the kind of the latest event, or ``step`` after a step line) with the
pace of its counts (``left``: the seconds still to go, estimated from the arrival times). Nothing here is a result: the
result is what the package writes at the end, and the page shows it then.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.gimap.shared.geometry.conventions import canonical_from_index_center

from ..domain import ClientEntry, ClientResult, ClientWindow, WindowPatterns
from ..domain.client import STATIC_KIND, number
from .live_progress import LiveProgressMixin

LIVE_PREFIX = "@live "
LIVE_STEPS = ("intake", "predict", "report", "present")
STATIC_LIVE_STEPS = ("intake", "predict", "report")
"""The steps of the analysis of a sample measured after its deposition (the package's static command)."""
PHASES = ("features", "network", "render", "refine", "sequence")
"""The phases of one network in the step that finds the solutions, in their order."""
PHASE_WEIGHTS = {"features": 1.0, "network": 0.3, "render": 2.0, "refine": 6.0, "sequence": 8.0}
"""The rough share of a network's time each phase takes (refining and the whole-scan refinement take most of it when
the mode has them): only for the bar of the job status, which says how far the analysis has come."""
SLOW_DISK_S_PER_FRAME, SLOW_DISK_SHARE = 0.15, 0.5
"""Reading the frames from disk is the slow part of the import when it takes at least this long per frame and at
least this share of the processes' time (the package's own thresholds)."""


def disk_slow(per_frame: Optional[float], share: Optional[float], said=None) -> bool:
    """The disk is the slow part of the import: the package says so (``said``), or the seconds per frame and the
    share of the time are both at their thresholds."""
    if said is True:
        return True
    return per_frame is not None and share is not None and per_frame >= SLOW_DISK_S_PER_FRAME and \
        share >= SLOW_DISK_SHARE


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
class LiveRun(LiveProgressMixin):
    """Everything the live view shows of one run (fed event by event, in the GUI thread)."""

    centre: Optional[tuple] = None
    detector: Optional[LiveEvent] = None
    frames: Optional[LiveEvent] = None
    """The latest chunk of frames read."""
    frames_done: int = 0
    frames_total: int = 0
    trace: dict = field(default_factory=dict)
    """Time of a frame (s) -> its count rate per pixel."""
    read_frames: int = 0
    """Frames of the chunks that said how long reading them from disk took (a newer package), and their sums:"""
    read_s: float = 0.0
    wall_s: float = 0.0
    read_mb: float = 0.0
    reading: Optional[LiveEvent] = None
    """How the import spent its time, said at its end (a newer package)."""
    workers: int = 0
    """The processes GIMaP asked the package to run with (0: not known)."""
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
    progress: dict = field(default_factory=dict)
    """(network, phase) -> {worker: its latest progress event's fields} (each worker counts its own share)."""
    phases: dict = field(default_factory=dict)
    """Network -> the latest of its phases that has begun."""
    drawing: int = 0
    drawn: int = 0
    activity: str = ""
    """What the run does now: the kind of the latest event the run took, or ``step`` when a step line came last."""
    preliminary: bool = False
    """The network's result was written (the page shows it) and the refinement goes on."""
    refining: Optional[LiveEvent] = None
    """The ``refine`` event: the refinement of every window and the joint fit of the whole scan have started."""
    joint_stage: Optional[LiveEvent] = None
    joint_stage_at: float = 0.0
    joint: Optional[LiveEvent] = None
    """The latest iteration of the joint fit of the whole scan (with its solution of every window)."""
    joint_at: float = 0.0
    paces: dict = field(default_factory=dict)
    """A count -> (time of its first event, its count then, time of its latest event, its count then): the
    ``frames`` read, a network's phase ``(family, phase)``, the windows ``drawn`` (monotonic seconds)."""
    # a sample measured after its deposition (the package's static command)
    kind: str = "insitu"
    """``insitu`` or ``static``."""
    reference_detector: Optional[LiveEvent] = None
    """The frames of the bare substrate summed (another piece of the same substrate, without metal)."""
    sample: Optional[LiveEvent] = None
    """The sample's frames read: how many, usable and left out, and the usable ones summed."""
    reference_sample: Optional[LiveEvent] = None
    frames_to_solve: int = 0
    frame_solutions: dict = field(default_factory=dict)
    """Frame number -> {stage: (ClientEntry, values)}."""
    latest_frame: Optional[LiveEvent] = None

    def feed(self, event: LiveEvent, now: Optional[float] = None) -> str:
        """Take one event (arrived at ``now``, monotonic seconds; by default now); returns its kind when the view
        changes ("" for an event it does not know)."""
        now = time.monotonic() if now is None else float(now)
        kind = self._take(event, now)
        if kind:
            self.activity = kind
        return kind

    def note_step(self) -> None:
        """A step line of the package came (a step started or ended): it says what the run does until the next
        event."""
        self.activity = "step"

    def _take(self, event: LiveEvent, now: float) -> str:
        kind = event.kind
        if kind in ("preliminary", "refine", "joint_stage", "joint"):
            return self._refinement(event, now)
        if kind == "centre":  # the package counts pixel centres from 0: GIMaP's pixel coordinates are shown
            col, row = event.number("col"), event.number("row")
            self.centre = canonical_from_index_center(col, row) if col is not None and row is not None else None
        elif kind == "detector":
            if "image" not in event.arrays:
                return ""
            if event.get("role") == "bare substrate":
                self.reference_detector = event
            else:
                self.detector = event
        elif kind == "sample":
            if event.get("role") == "bare substrate":
                self.reference_sample = event
            else:
                self.sample = event
        elif kind == "solving":
            self.frames_to_solve = int(event.number("total") or 0)
            self.refine = int(event.number("refine") or 0)
        elif kind == "frame":
            if not self._frame(event):
                return ""
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
            self._timing(event)
            self._pace("frames", self.frames_done, now)
        elif kind == "reading":
            self.reading = event
        elif kind == "imported":
            self.imported = event
            self.frames_done = int(event.number("frames") or self.frames_done)
            self.frames_total = max(self.frames_total, self.frames_done)
        elif kind == "models":
            self.families = [str(name) for name in event.get("families") or () if name]
            self.mode = str(event.get("mode") or "")
        elif kind == "windows":
            self.family = str(event.get("family") or "")
            if self.family in self.totals:  # the network starts again (refining after its preliminary result)
                self._restart(self.family)
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
        elif kind == "progress":
            phase, family = str(event.get("phase") or ""), str(event.get("family") or self.family)
            if phase not in PHASES:
                return ""
            self.progress.setdefault((family, phase), {})[str(event.get("worker") or "")] = dict(event.fields)
            if PHASES.index(phase) >= PHASES.index(self.phases.get(family, PHASES[0])):
                self.phases[family] = phase
            self._pace((family, phase), self.phase_count(phase, family)[0], now)
        elif kind == "drawing":
            self.drawing, self.drawn = int(event.number("total") or 0), 0
            self.paces.pop("drawn", None)
        elif kind == "drawn":
            self.drawn += 1
            self._pace("drawn", self.drawn, now)
        else:
            return ""
        return kind

    def _timing(self, event: LiveEvent) -> None:
        """The disk time of a chunk of frames, added up (a chunk without it, from an older package, adds nothing)."""
        frames, read_s = event.number("frames"), event.number("read_s")
        if frames is None or read_s is None or frames <= 0:
            return
        self.read_frames += int(frames)
        self.read_s += read_s
        self.wall_s += event.number("wall_s") or 0.0
        self.read_mb += event.number("mb") or 0.0

    # -- reading the frames from disk --------------------------------------------------------------------------------
    def disk_per_frame(self) -> Optional[float]:
        """Seconds a process spent reading one frame from disk (the import's own figure once it has ended)."""
        said = self.reading.number("read_s_per_frame") if self.reading is not None else None
        if said is not None:
            return said
        return self.read_s / self.read_frames if self.read_frames else None

    def disk_share(self) -> Optional[float]:
        """The share (0–1) of the processes' time spent reading the frames from disk."""
        said = self.reading.number("read_share") if self.reading is not None else None
        if said is not None:
            return said
        return min(1.0, self.read_s / self.wall_s) if self.read_frames and self.wall_s > 0 else None

    def disk_slow(self) -> bool:
        return disk_slow(self.disk_per_frame(), self.disk_share(),
                         self.reading.get("slow") if self.reading is not None else None)

    def processes(self) -> int:
        """The processes the run works with: as the import said them, else as GIMaP asked (0: not known)."""
        said = self.reading.number("workers") if self.reading is not None else None
        return int(said) if said else self.workers

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

    def _frame(self, event: LiveEvent) -> bool:
        number_ = event.number("frame")
        if number_ is None:
            return False
        frame, stage = int(number_), str(event.get("stage") or "network")
        values = event.get("values") if isinstance(event.get("values"), dict) else {}
        values = {key: value for key, value in ((k, number(v)) for k, v in values.items()) if value is not None}
        entry = ClientEntry(FRAME_LABEL.format(frame=frame), "refined" if stage == "refined" else "candidate",
                            fit_deviation=event.number("fit_deviation"), values=values)
        self.frame_solutions.setdefault(frame, {})[stage] = entry
        if {"q", "af", "measured", "model"} <= set(event.arrays):
            self.latest_frame = event
        return True

    # -- what the page shows ----------------------------------------------------------------------------------------
    def frames_solved(self) -> int:
        return len(self.frame_solutions)

    def frame_entry(self, frame: int) -> Optional[ClientEntry]:
        """The solution of a frame of the sample shown: refined when it is, else the network's."""
        stages = self.frame_solutions.get(frame) or {}
        return stages.get("refined") or stages.get("network")

    def static_result(self, folder: Path, series: Optional[ClientResult], label: str,
                      nominal_nm: Optional[float]) -> Optional[ClientResult]:
        """The series as far as it is known: its samples analysed before (``series``; the sample of this run is left
        out of it) and every frame of this sample solved so far, at the sample's nominal thickness (the plot joins
        the frames of one thickness through their median, as the result takes the median over the frames)."""
        windows = [window for window in (series.windows if series is not None else ())
                   if window.label != label and window.entries]
        for frame in sorted(self.frame_solutions):
            entry = self.frame_entry(frame)
            if entry is not None and nominal_nm is not None:
                windows.append(ClientWindow(float(nominal_nm), nominal_nm=float(nominal_nm), entries=(entry,),
                                            label=f"{label} {entry.label}"))
        if not windows:
            return None
        windows.sort(key=lambda window: (window.nominal_nm if window.nominal_nm is not None else window.time_s))
        return ClientResult(Path(folder), scan="", family="", windows=tuple(windows), kind=STATIC_KIND,
                            x_label=(series.x_label if series is not None and series.x_label else "nominal thickness"),
                            x_unit=(series.x_unit if series is not None and series.x_unit else "nm"))

    def frame_patterns(self) -> Optional[tuple]:
        """(WindowPatterns, entry label, frame number, the event) of the frame solved last, or None."""
        event = self.latest_frame
        found = _patterns(event, FRAME_LABEL.format(frame=int(event.number("frame") or 0))) if event is not None else None
        return None if found is None else (found, found.labels[0], event.number("frame"), event)

    def network_index(self) -> tuple:
        """(1-based place of the network at work among those the analysis runs, how many it runs)."""
        if self.family in self.families:
            return self.families.index(self.family) + 1, len(self.families)
        return (1, 1) if self.family else (0, len(self.families))

    def solved(self, family: Optional[str] = None) -> int:
        return len(self.solutions.get(self.family if family is None else family, {}))

    def refined(self, family: Optional[str] = None) -> int:
        """Windows of a network whose refined solution has arrived."""
        found = self.solutions.get(self.family if family is None else family, {})
        return sum(1 for entries in found.values() if any(entry.kind == "refined" for entry, *_ in entries.values()))

    def phase(self, family: Optional[str] = None) -> str:
        """The phase a network is in ("" before its first progress event or window): a solved window means it draws
        candidates at least, a refined one that it refines."""
        family = self.family if family is None else family
        found = self.phases.get(family, "")
        rank = PHASES.index(found) if found else -1
        if self.refined(family) and rank < PHASES.index("refine"):
            return "refine"
        if self.solved(family) and rank < PHASES.index("render"):
            return "render"
        return found

    def phase_count(self, phase: str, family: Optional[str] = None, key: str = "done") -> tuple:
        """(the sum over the workers of ``key``, the phase's total) of a network's phase ((0, 0) before any event)."""
        workers = self.progress.get((self.family if family is None else family, phase), {})
        done = sum(int(number(fields.get(key)) or 0) for fields in workers.values())
        total = max((int(number(fields.get("total")) or 0) for fields in workers.values()), default=0)
        return done, total

    def network_share(self, family: Optional[str] = None) -> float:
        """How much of a network's work is done (0–1), by its phases (``PHASE_WEIGHTS``; refining only when the mode
        refines, the whole-scan refinement only once it has begun)."""
        family = self.family if family is None else family
        phase = self.phase(family)
        steps = [name for name in PHASES if name not in ("refine", "sequence")]
        if self.refine > 0 or phase == "refine":
            steps.append("refine")
        if phase == "sequence" or self.mode == "full":
            steps.append("sequence")
        total = self.totals.get(family, 0)
        shares = {"features": self._share(*self.phase_count("features", family)),
                  "network": self._share(*self.phase_count("network", family)),
                  "render": self._share(max(self.solved(family), self.phase_count("render", family)[0]), total),
                  "refine": self._share(self.refined(family) + self._refining(family), total),
                  "sequence": self._share(*self.phase_count("sequence", family))}
        weight = sum(PHASE_WEIGHTS[name] for name in steps)
        return sum(PHASE_WEIGHTS[name] * shares[name] for name in steps) / weight if weight else 0.0

    def _refining(self, family: str) -> float:
        """How much of the windows being refined is done: each worker's model images of its window at work over about
        the most one window takes (``current`` / ``budget``; 0 between windows)."""
        found = 0.0
        for fields in self.progress.get((family, "refine"), {}).values():
            current, budget = number(fields.get("current")), number(fields.get("budget"))
            if current and budget:
                found += min(0.95, current / budget)  # a window is done when its refined solution arrives
        return found

    @staticmethod
    def _share(done, total) -> float:
        return min(1.0, done / total) if total else 0.0

    def step_share(self, step: str) -> Optional[float]:
        """How much of a step is done (0–1) as the events tell it, or None when they do not."""
        if step == "intake":
            if self.kind == "static":
                return 1.0 if self.sample is not None else None
            if self.imported is not None:
                return 1.0
            return self._share(self.frames_done, self.frames_total) if self.frames_total else None
        if step == "predict":
            if self.kind == "static":
                return self._share(self.frames_solved(), self.frames_to_solve) if self.frames_to_solve else None
            if self.chosen:
                return 1.0
            if not self.family:
                return None
            index, count = self.network_index()
            return (max(0, index - 1) + self.network_share()) / max(1, count)
        if step == "present" and self.drawing:
            return self._share(self.drawn, self.drawing)
        return None

    def trace_points(self) -> tuple:
        """(times, count rates) of the frames read so far, in the order of time."""
        times = sorted(self.trace)
        return times, [self.trace[time_s] for time_s in times]

    def result(self, folder: Path, family: Optional[str] = None) -> Optional[ClientResult]:
        """The solutions of a network so far as a result (one window per time; None before the first)."""
        family = self.shown_family() if family is None else family
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
        """The network whose solutions the view shows: the one the data chose, else the one at work — or, until that
        one has a solution, the one before it (the view does not go blank when the next network starts)."""
        if self.chosen:
            return self.chosen
        if self.solutions.get(self.family) or not self.family:
            return self.family
        earlier = [name for name in self.families if name != self.family and self.solutions.get(name)]
        return earlier[-1] if earlier else self.family

    def latest_patterns(self, family: Optional[str] = None) -> Optional[tuple]:
        """(WindowPatterns, entry label, time, the event) of the latest window a network solved (by default the one
        shown), or None."""
        event = self.latest.get(self.shown_family() if family is None else family)
        if event is None:
            return None
        label = str(event.get("label"))
        found = _patterns(event, label)
        return None if found is None else (found, label, event.number("t"), event)


FRAME_LABEL = "frame {frame}"
"""The label of the solution of one frame of a static sample (the page names it in its language)."""


def _patterns(event: LiveEvent, label: str) -> Optional[WindowPatterns]:
    """The measured and the model image of an event (q, af, measured, model), or None when they do not fit."""
    try:
        arrays = event.arrays
        q, af = (_vector(arrays[key]) for key in ("q", "af"))
        measured, model = (_matrix(arrays[key]) for key in ("measured", "model"))
    except (KeyError, TypeError, ValueError):
        return None
    if measured.shape != (af.size, q.size) or model.shape != measured.shape or q.size < 2 or af.size < 2:
        return None
    return WindowPatterns(q, af, measured, model[None], (label,))


def _vector(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def _matrix(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


__all__ = ["FRAME_LABEL", "LIVE_PREFIX", "LIVE_STEPS", "LiveEvent", "LiveRun", "PHASES", "PHASE_WEIGHTS",
           "SLOW_DISK_SHARE", "SLOW_DISK_S_PER_FRAME", "STATIC_LIVE_STEPS", "disk_slow", "live_event", "parse_live"]
