"""What a running Analyze is doing now, in one line (a mixin of ``SputterPage``, through ``LivePanelMixin``).

The analysis runs for minutes; its steps alone say little of that time. From the package's live events
(``application/client_live.py``: ``LiveRun.activity`` and the counts) one line says in plain words what happens now,
with its counts and, where the pace is known, the time still to go: reading the frames, each phase of a network,
the comparison of the networks, the refinement after the preliminary result, the joint fit of the whole scan (its
iteration, the median fit deviation, the time so far and — seconds per iteration times the iterations left — the time
left if it takes all of them) and drawing the main screen. Where the events say nothing more than the step lines, the
line is the step's (``ClientProgress.message``).

The line stands at the top of the live view, in the job status of the bar and — once the preliminary result has
replaced the live view — in the note line of the main screen. There, each iteration of the joint fit also draws its
current solution dashed over the evolution (``EvolutionPlot.set_overlay``), until the final result replaces it; the
tiles keep the values of the solution shown (the joint fit's values are only on the plot, so the tiles never mix two
results). A sample measured after its deposition keeps the step lines.
"""

from __future__ import annotations

import time
from typing import Optional

from src.gimap.app.presentation.i18n import tr

from ..application import LiveRun
from .client_texts import STARTING, shown_number
from .reading_texts import clock_time
from .texts import text

READING = "Reading frames: {done} of {total}"
READING_FIRST = "Reading the frames: finding the beam centre and the pixel mask"
IMPORTED = "{frames} frames read; the networks start"
STARTING_NETWORK = "Starting network {family}"
STARTING_NETWORKS = "Starting network {family} (the first of {count} to compare)"
NETWORK_WINDOWS = "Network {family}: starting on {total} time windows"
NETWORK_PHASES = {
    "features": "Network {family}: reading the network's input, window {done} of {total}",
    "network": "Network {family}: the network proposes candidates for {total} windows",
    "render": "Network {family}: drawing the candidates with the forward model, window {done} of {total}",
    "refine": "Network {family}: refining the best candidate of each window, {done} of {total}",
}
SEQUENCE = "Joint fit: iteration {done} of at most {total}"
COMPARING = "Comparing the networks: {scores} → {family}"
SOLVED = "Network {family}: every window solved"
PRELIMINARY_SHOWN = "Preliminary result shown; the refinement of every window starts"
REFINING = "Network {family}: refining every window"
REFINING_JOINT = "Network {family}: refining every window, then the joint fit of the whole scan"
JOINT_POOL = "Joint fit: preparing (each window tries its neighbours' solutions, {windows} windows)"
JOINT_FIRST = "Joint fit: working on iteration 1 of {total}"
JOINT = "Joint fit: iteration {iteration} of {total}"
MEDIAN = "median fit deviation {median}"
SO_FAR = "{elapsed} so far"
SO_FAR_LEFT = "{elapsed} so far, {left}"
LEFT_MINUTES = "about {minutes} min left"
LEFT_HOURS = "about {hours} h {minutes} min left"
LEFT_SOON = "less than a minute left"
JOINT_BATCH = "Joint fit: iteration {iteration} of {total} · {phase}: {done} of {windows} windows"
JOINT_POOL_BATCH = "Joint fit: preparing · {done} of {windows} windows"
JOINT_PHASES = {"derivatives": "derivatives", "step": "trial step"}
"""The phases of an iteration of the joint fit (``joint_progress``): the derivatives are its long part."""
DRAWING = "Drawing the main screen: window {drawn} of {total}"
JOINT_LABEL = "Joint fit, iteration {iteration} of {total}"
"""The overlay of the joint fit at work on the evolution plot."""
STATUS_FIELD = "{status}"
"""The job status holds the line as it is (already in the interface language)."""
TEMPLATES = (READING, READING_FIRST, IMPORTED, STARTING_NETWORK, STARTING_NETWORKS, NETWORK_WINDOWS,
             *NETWORK_PHASES.values(), SEQUENCE, COMPARING, SOLVED, PRELIMINARY_SHOWN, REFINING, REFINING_JOINT,
             JOINT_POOL, JOINT_FIRST, JOINT, MEDIAN, SO_FAR, SO_FAR_LEFT, LEFT_MINUTES, LEFT_HOURS, LEFT_SOON, DRAWING,
             JOINT_LABEL, JOINT_BATCH, JOINT_POOL_BATCH, *JOINT_PHASES.values())
"""The texts of the status line that need Chinese (numbers and network names are data)."""
AFTER_PRELIMINARY = {"preliminary", "refine", "windows", "progress", "window", "joint_stage", "joint", "joint_progress"}
"""What the run does after its preliminary result until the report: the job status's bar then follows the line."""
NETWORK_KINDS = {"windows", "progress", "window"}


def _count(value) -> str:
    return f"{int(value or 0):,}"


def left_words(seconds: Optional[float]) -> str:
    """"about 12 min left" ("" without an estimate)."""
    if seconds is None:
        return ""
    if seconds < 60:
        return tr(LEFT_SOON)
    minutes = max(1, int(round(seconds / 60.0)))
    if minutes < 60:
        return text(LEFT_MINUTES, minutes=minutes)
    return text(LEFT_HOURS, hours=minutes // 60, minutes=minutes % 60)


def _with_left(words: str, seconds: Optional[float]) -> str:
    left = left_words(seconds)
    return f"{words} · {left}" if left else words


def status_line(run: Optional[LiveRun], now: Optional[float] = None) -> tuple:
    """(what the run does now in the interface language — "" where the step line says as much —, the share of it
    that is done (0–1) or None)."""
    if run is None or run.kind == "static":
        return "", None
    now = time.monotonic() if now is None else now
    kind = run.activity
    if kind == "frames" and run.frames_total:
        words = text(READING, done=_count(run.frames_done), total=_count(run.frames_total))
        return _with_left(words, run.left("frames", run.frames_total, now)), run.frames_done / run.frames_total
    if kind in ("centre", "detector"):
        return tr(READING_FIRST), None
    if kind in ("reading", "imported"):
        frames = run.imported.number("frames") if run.imported is not None else None
        frames = frames if frames is not None else run.reading.number("frames_read") if run.reading else None
        return (text(IMPORTED, frames=_count(frames)), 1.0) if frames is not None else ("", None)
    if kind == "models" and run.families:
        family, count = run.families[0], len(run.families)
        return (text(STARTING_NETWORKS, family=family, count=count) if count > 1 else
                text(STARTING_NETWORK, family=family)), None
    if kind in NETWORK_KINDS:
        return _network_line(run, now)
    if kind == "chosen":
        return _chosen_line(run), None
    if kind == "preliminary":
        return tr(PRELIMINARY_SHOWN), None
    if kind == "refine" and run.refining is not None:
        family = str(run.refining.get("family") or run.family)
        joint = (run.refining.number("iterations") or 0) > 0
        return text(REFINING_JOINT if joint else REFINING, family=family), None
    if kind == "joint_progress" and run.joint_progress is not None:
        return _joint_batch_line(run, now)
    if kind in ("joint_stage", "joint"):
        return _joint_line(run, now)
    if kind in ("drawing", "drawn") and run.drawing:
        words = text(DRAWING, drawn=min(run.drawn + 1, run.drawing), total=run.drawing)
        return _with_left(words, run.left("drawn", run.drawing, now)), run.drawn / run.drawing
    return "", None


def _network_line(run: LiveRun, now: float) -> tuple:
    family, total = run.family, run.totals.get(run.family, 0)
    phase = run.phases.get(family, "")
    if phase == "sequence":
        done, most = run.phase_count("sequence", family)
        return text(SEQUENCE, done=done, total=most or "?"), (done / most if most else None)
    if not phase and run.solved(family):  # a package that does not count its phases: the windows solved
        phase = "render"
    if not phase:
        return (text(NETWORK_WINDOWS, family=family, total=_count(total)), 0.0) if family else ("", None)
    done, phase_total = run.phase_count(phase, family)
    if phase == "render" and not run.progress.get((family, "render")):
        done = run.solved(family)
    total = phase_total or total
    words = text(NETWORK_PHASES[phase], family=family, done=_count(done), total=_count(total))
    if phase == "network":
        return words, None
    return _with_left(words, run.left((family, phase), total, now)), (min(1.0, done / total) if total else None)


def _chosen_line(run: LiveRun) -> str:
    scores = [(name, value) for name, value in run.deviations.items() if value is not None]
    if len(scores) > 1:
        listed = ", ".join(f"{name} {shown_number(value)}" for name, value in scores)
        return text(COMPARING, scores=listed, family=run.chosen)
    return text(SOLVED, family=run.chosen) if run.chosen else ""


def _joint_line(run: LiveRun, now: float) -> tuple:
    stage, event = run.joint_stage, run.joint
    if event is None:
        if stage is None:
            return "", None
        total = int(stage.number("total") or 0)
        if stage.get("stage") != "fit":
            return text(JOINT_POOL, windows=_count(stage.number("windows"))), None
        words = text(JOINT_FIRST, total=total)
        elapsed = run.joint_elapsed(now)
        return (words if elapsed is None else f"{words} · {text(SO_FAR, elapsed=clock_time(elapsed))}"), 0.0
    iteration, total = int(event.number("iteration") or 0), int(event.number("total") or 0)
    parts = [text(JOINT, iteration=iteration, total=total)]
    if event.number("median") is not None:
        parts.append(text(MEDIAN, median=shown_number(event.number("median"))))
    elapsed, left = run.joint_elapsed(now), left_words(run.joint_left(now))
    if elapsed is not None:
        parts.append(text(SO_FAR_LEFT, elapsed=clock_time(elapsed), left=left) if left else
                     text(SO_FAR, elapsed=clock_time(elapsed)))
    return " · ".join(parts), (min(1.0, iteration / total) if total else None)


def _joint_batch_line(run: LiveRun, now: float) -> tuple:
    """"Joint fit: iteration 3 of 6 · derivatives: 12 of 20 windows · 6:12 so far, about 9 min left"."""
    event = run.joint_progress
    done, windows = _count(event.number("done")), _count(event.number("windows"))
    phase = str(event.get("phase") or "")
    if phase == "pool":
        words = text(JOINT_POOL_BATCH, done=done, windows=windows)
    else:
        words = text(JOINT_BATCH, iteration=int(event.number("iteration") or 0), total=int(event.number("total") or 0),
                     phase=tr(JOINT_PHASES.get(phase, phase)), done=done, windows=windows)
    elapsed, left = run.joint_elapsed(now), left_words(run.joint_batch_left(now))
    if elapsed is not None:
        words += " · " + (text(SO_FAR_LEFT, elapsed=clock_time(elapsed), left=left) if left else
                          text(SO_FAR, elapsed=clock_time(elapsed)))
    return words, run.joint_batch_share()


class LiveStatusMixin:
    """Needs ``_live``, ``_live_tracker``, ``_client_stop``, ``_client_status``, ``_client_status_parts``, the
    ``live_status`` label of the live view, ``run_note`` of the main screen and the two evolution plots."""

    def _live_status_words(self) -> str:
        """The line in the interface language: what the events say, else the step at work."""
        line, _share = status_line(self._live)
        if line:
            return line
        tracker = self._live_tracker()
        if tracker is not None and tracker.step:
            template, values = tracker.message()
            return text(template, **dict(values, stage=tr(str(values.get("stage", "")))))
        return tr(STARTING)

    def _live_show_status(self) -> None:
        """The line in the live view, in the note line of the main screen while it shows the preliminary result, and
        in the job status."""
        if self._live is None:
            return
        words = self._live_status_words()
        self.live_status.set_full_text(words)
        shown = self._preliminary_shown()
        if shown:
            self.run_note.set_full_text(words)
        self.run_note.setVisible(shown)
        self._show_steps_button(shown)  # Steps beside the note; back to the preliminary result in the live view
        self._live_show_tree()  # the line under the step at work
        if not self._live_static():
            self._live_job_status(self._live_tracker())

    def _live_job_status(self, tracker) -> None:
        """The job status of a running Analyze: what happens now (else the step), its bar the steps' — after the
        preliminary result, until the report, the share of what the line speaks of (None: busy)."""
        if tracker is None or self._client_stop:
            return
        run = self._live
        line, share = status_line(run)
        if line:
            parts = [(STATUS_FIELD, {"status": line})]
            own = run.preliminary and run.activity in AFTER_PRELIMINARY
            progress = share if own else tracker.fraction
        else:
            template, values = tracker.message()
            if not template:
                return
            parts, progress = [(template, values)], tracker.fraction
        if ("running", parts, progress) != self._client_status_parts:
            self._client_status("running", parts, progress)

    def _preliminary_shown(self) -> bool:
        """The main screen shows the preliminary result of the run (not the live view, not another result)."""
        preliminary = getattr(self, "_live_preliminary", None)
        return self._live is not None and preliminary is not None and self.client_result is preliminary and \
            self.result_stack.currentIndex() == 1

    def _live_show_joint(self) -> None:
        """The current solution of the joint fit, dashed over the evolution the screen shows: the main screen's while
        it shows the preliminary result, else the live view's (when it shows the same network)."""
        run = self._live
        if run is None or run.joint is None:
            return
        times, values = run.joint_points()
        if not times:
            return
        label = text(JOINT_LABEL, iteration=int(run.joint.number("iteration") or 0),
                     total=int(run.joint.number("total") or 0))
        overlay = (times, values, label)
        if self._preliminary_shown():
            self.evolution.set_overlay(overlay)
        elif str(run.joint.get("family") or run.family) == run.shown_family():
            self.live_evolution.set_overlay(overlay)

    def _live_status_end(self) -> None:
        """The run ended: the line and the joint fit's overlay go."""
        self.run_note.setVisible(False)
        self.live_status.set_full_text("")
        self.evolution.set_overlay(None)
        self.live_evolution.set_overlay(None)
        self._live_preliminary = None


__all__ = ["JOINT_LABEL", "LiveStatusMixin", "STATUS_FIELD", "TEMPLATES", "left_words", "status_line"]
