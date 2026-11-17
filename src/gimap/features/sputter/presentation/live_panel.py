"""The live view of the main screen while Analyze runs (a mixin of ``SputterPage``): what each step of the analysis
does and what it has found so far.

The analysis takes minutes (reading the frames alone several on a large scan). From its start to its end the main
screen shows the live view instead of a result (``views/sputter_live_view.py``): at the left the four steps — read the
frames, find the solutions, write the report, draw the images — each with its state and one line of what it found;
at the right the step at work, in one paragraph, with what it has found so far (``application/client_live.py``):

* reading the frames: the detector summed with the pixels the mask leaves out, the beam centre and the sample horizon;
  the frames read last on the grid of q_y and exit angle; the count rate of every frame read so far;
* finding the solutions (and the two steps after it): the measured image of the window solved last beside the model
  image of its solution, with its values in the line under them, and the radius, height, distance and coverage of
  every window solved so far (the network at work; once the data have chosen between two networks, the one chosen).

A newer package also says how long its processes spend reading the frames from disk: the line of the import step adds
the seconds per frame and their share of the time, and when the disk is the slow part a note above the pictures says
so, with what helps (``reading_texts.py``); the clock line says how many processes run. A click on a step that has
started shows what it found; a click on the step at work follows the analysis again. When
the analysis ends the result is shown (or, when it failed or was stopped, what was shown before). A package that
sends no live events is followed by its steps alone.

The events come from the background task with their arrays already read; here they only update ``LiveRun``. The
pictures are drawn by a timer at most every ``REDRAW_MS`` (many windows can be solved in a second), each only when
what it shows changed; the frames picture and the measured and model images only put new values into what is drawn.
"""

from __future__ import annotations

import time
from typing import Optional

from PyQt5.QtCore import QTimer

from src.gimap.app.presentation.i18n import tr

from ..application import LIVE_STEPS, STATIC_LIVE_STEPS, LiveEvent, LiveRun
from .client_texts import entry_name, shown_number
from .evolution_view import EvolutionPlot
from .live_pictures import FramesPicture, TracePlot
from .live_static import STATIC_EXPLAIN, STATIC_STEP_NAMES, LiveStaticMixin
from .live_status import LiveStatusMixin
from .pattern_view import PatternPair
from .reading_texts import PROCESSES, READING_LINE, SLOW_DISK_NOTE, disk_words
from .texts import text
from .views.sputter_live_view import LIVE_STEP_TITLES
from .views.sputter_main_view import TILES

LIVE_PAGE = 2
"""The live view's page in the main screen's stack (0: the empty state, 1: a result)."""
REDRAW_MS = 400
CLOCK_MS = 1000
STEP_NAMES = dict(LIVE_STEP_TITLES)
EXPLAIN = {
    "intake": ("Every detector frame is projected onto a grid of q_y and exit angle, without the pixels the mask leaves "
               "out, and the frames are summed in steps of 0.5 s. Left: the detector summed over sampled frames — red: "
               "left out by the mask, +: the direct beam, dashed: the sample horizon. Right: the frames read last. "
               "Below: the count rate of every frame read so far; it rises as the metal is deposited."),
    "predict": ("For every time window the network proposes candidate structures; the physical model draws each as an "
                "image and the one closest to the measured image is kept. First the network input of every window is "
                "prepared and the network runs once for all of them; when the mode refines, every window has its "
                "candidates before the refinements start. Above: the window solved last, measured and model image. "
                "Below: the sizes found so far."),
    "report": "The solutions of all windows go into a table, the plots of their evolution and a report.",
    "present": ("For every window the measured image and the image of every candidate are drawn for the main screen. "
                "The result opens when this step is done."),
}
"""What a step does, said once while it is shown."""
TITLE = "Step {index} of {count}: {name}"
CLOCK = "Running for {time}"
CENTRE_LINE = "Beam centre ({col}, {row}) px"
FRAMES_LINE = "{done} of {total} frames read"
IMPORTED_LINE = "{frames} frames, {first}–{last} s"
IMPORTED_ONSET = "{frames} frames, {first}–{last} s, deposition from {onset} s"
REUSED_LINE = "Read before for this scan"
WINDOWS_LINE = "{solved} of {total} time windows"
NETWORK_PREFIX = "Network {index} of {count}: {what}"
PHASE_LINES = {
    "features": "Network inputs ready for {done} of {total} time windows",
    "network": "The network proposes candidates for {total} time windows…",
    "render": "{solved} of {total} time windows solved ({images} model images computed)",
    "refine": "{solved} of {total} time windows solved, {done} refined ({images} model images computed)",
    "sequence": "Whole-scan refinement: iteration {done} of at most {total}",
}
"""What the network at work is doing, phase by phase (``LiveRun.phase``): the line of the step, and the message of
the pictures until the first window is solved."""
FIRST_WINDOW = "The measured and the model image of a time window appear here once the first one is solved."
CHOSEN_LINE = "{family} fits the data better"
CHOSEN_DEVIATION = "{family} fits the data better (fit deviation {deviation})"
DRAWN_LINE = "{drawn} of {total} windows drawn"
DETECTOR_CAPTION = ("{frames} frames summed · {masked} pixels left out by the mask (found in these frames: {dead} dead, "
                    "{hot} hot, {shadow} of beamstop and holder)")
WINDOW_CAPTION = "t = {time} s"
DEVIATION_PART = "fit deviation {deviation}"
WAITING_SOLUTIONS = "The first time windows are being solved…"
WAITING_FRAMES = "The frames are being read…"
NOTHING_LIVE = "This analysis package does not show what it finds while it works: its result opens at the end."
TEMPLATES = (*STEP_NAMES.values(), *EXPLAIN.values(), TITLE, CLOCK, CENTRE_LINE, FRAMES_LINE, IMPORTED_LINE,
             IMPORTED_ONSET, REUSED_LINE, NETWORK_PREFIX, *PHASE_LINES.values(), FIRST_WINDOW, CHOSEN_LINE,
             CHOSEN_DEVIATION, DRAWN_LINE, DETECTOR_CAPTION, DEVIATION_PART, WAITING_SOLUTIONS, WAITING_FRAMES,
             NOTHING_LIVE)
"""The templates of the live view that need Chinese (numbers, units and network names are data)."""
CAPTION_KEYS = ("R_nm", "H_nm", "D_nm", "coverage")
"""The quantities said under the images of the window solved last (those the solution has)."""
TILE_WORDS = {key: (caption, unit) for key, caption, unit, _tip in TILES}


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    return f"{hours}:{rest // 60:02d}:{rest % 60:02d}" if hours else f"{rest // 60:02d}:{rest % 60:02d}"


def _count(value) -> str:
    return f"{int(value or 0):,}"


class LivePanelMixin(LiveStatusMixin, LiveStaticMixin):
    """Needs the widgets of ``views/sputter_live_view.py``, ``result_stack``, ``_client_tracker``,
    ``_static_tracker`` and ``_show_client_empty``."""

    def _init_live(self) -> None:
        self._live: Optional[LiveRun] = None
        self._live_preliminary = None  # the preliminary result of the run, while the main screen shows it
        self._live_folder = None
        self._live_started = 0.0
        self._live_events = 0
        self._live_reused = False
        self._live_sample: dict = {}
        self._live_dirty: set = set()
        self._live_shown = ""
        self._live_pinned = ""
        self._live_redraw = QTimer(self)
        self._live_redraw.setSingleShot(True)
        self._live_redraw.setInterval(REDRAW_MS)
        self._live_redraw.timeout.connect(self._live_draw)
        self._live_clock = QTimer(self)
        self._live_clock.setInterval(CLOCK_MS)
        self._live_clock.timeout.connect(self._live_tick)
        self.live_frames = FramesPicture(self.live_frames_layout)
        self.live_trace = TracePlot(self.live_trace_layout)
        self.live_patterns = PatternPair(self.live_patterns_layout)
        self.live_evolution = EvolutionPlot(self.live_evolution_layout)
        self.live_steps.stepChosen.connect(self._live_step_chosen)

    # -- the run -------------------------------------------------------------------------------------------------
    def live_running(self) -> bool:
        """The live view is up (also asked while the page is built, before ``_init_live``)."""
        return getattr(self, "_live", None) is not None

    def _live_begin(self, folder, *, kind: str = "insitu", label: str = "", nominal_nm=None, series=None,
                    reference: bool = False, workers: Optional[int] = None) -> None:
        """Analyze started a run (not a result it had already): the live view replaces the main screen. ``kind``
        ``static``: one sample measured after its deposition (``label``, ``nominal_nm``; ``series``: the series as
        it was before; ``reference``: frames of a bare substrate were given). ``workers``: the processes the package
        was asked to run with."""
        self._live, self._live_folder, self._live_events, self._live_reused = LiveRun(kind=kind), folder, 0, False
        self._live.workers = int(workers or 0)
        self._live_preliminary = None
        self.live_evolution.set_overlay(None)
        self.live_disk_note.setVisible(False)
        self._live_sample = {"label": label, "nominal_nm": nominal_nm, "series": series, "reference": reference}
        self._live_started = time.monotonic()
        self._live_shown, self._live_pinned = "", ""
        self._live_titles()
        for key in LIVE_STEPS:
            self.live_steps.set_state(key, "pending", "")
        self.live_steps.set_step_visible("present", kind != "static")
        self.live_plot_host.setVisible(True)  # a static run hides it while its series has nothing
        self.live_patterns.show_message(tr(WAITING_SOLUTIONS))
        self.live_frames.show(None, None)
        self.live_trace.show([], [], None)
        self.live_evolution.draw(None, None, None)
        self.result_stack.setCurrentIndex(LIVE_PAGE)
        self._live_clock.start()
        self._live_tick()
        self._live_dirty = {"all"}
        self._live_draw()

    def _live_event(self, event: LiveEvent) -> None:
        if self._live is None:
            return
        self._live_events += 1
        changed = self._live.feed(event)
        if changed:
            self._live_changed(changed)
            self._live_update_bar()
            if changed == "joint":
                self._live_show_joint()

    def _live_update_bar(self) -> None:
        """The job status: how far the step at work has come, as the events tell it (``within``), and — Analyze of a
        scan — what happens now (``live_status.py``)."""
        tracker = self._live_tracker()
        if tracker is None or not tracker.step or self._live is None:
            return
        share = self._live.step_share(tracker.step)
        if not self._live_static():
            if share is not None:
                tracker.within = share
            self._live_job_status(tracker)
            return
        if share is None or (tracker.within is not None and abs(share - tracker.within) < 0.005):
            return
        tracker.within = share
        template, values = tracker.message()
        if template and not self._static_stop:
            self._static_status("running", [(template, values)], tracker.fraction)

    def _live_progress(self, tracker) -> None:
        """A step line of the package (``tracker`` has taken it): until the next event it says what the run does."""
        if self._live is None:
            return
        self._live.note_step()
        if tracker.step == "intake" and tracker.reused:
            self._live_reused = True
        self._live_changed("steps")

    def _live_changed(self, what: str = "steps") -> None:
        if self._live is None:
            return
        self._live_dirty.add(what)
        if not self._live_redraw.isActive():
            self._live_redraw.start()

    def _live_end(self) -> None:
        """The run ended (any way): the view goes; what follows (the result, or what was shown before) is the
        caller's."""
        self._live_redraw.stop()
        self._live_clock.stop()
        self._live_status_end()
        if self._live is None:
            return
        self._live = None
        if self.result_stack.currentIndex() == LIVE_PAGE:
            self._show_client_empty()  # the result shown before, else the empty state

    def _live_tick(self) -> None:
        if self._live is not None:
            words = text(CLOCK, time=_clock(time.monotonic() - self._live_started))
            processes = self._live.processes()
            self.live_clock.setText(words + " · " + text(PROCESSES, count=processes) if processes else words)
            self._live_show_status()  # the time so far and left move on

    def _live_step_chosen(self, key: str) -> None:
        """A click on a step: what it found (one that has started), or — the step at work — follow the analysis."""
        if self._live is None:
            return
        running, steps = self._live_running_step(), self._live_steps_of()
        if key not in steps or steps.index(key) > steps.index(running):
            if self._live_shown:  # not started: the step shown stays (and stays marked)
                self.live_steps.set_current(self._live_shown)
            return
        self._live_pinned = "" if key == running else key
        self._live_changed("all")

    # -- what is shown -------------------------------------------------------------------------------------------
    def _live_static(self) -> bool:
        return self._live is not None and self._live.kind == "static"

    def _live_steps_of(self) -> tuple:
        return STATIC_LIVE_STEPS if self._live_static() else LIVE_STEPS

    def _live_tracker(self):
        return getattr(self, "_static_tracker" if self._live_static() else "_client_tracker", None)

    def _live_titles(self) -> None:
        """The names of the steps for the kind of run (in the interface language)."""
        names = STATIC_STEP_NAMES if self._live_static() else STEP_NAMES
        for key, name in names.items():
            self.live_steps.set_title(key, tr(name))

    def _live_running_step(self) -> str:
        tracker, steps = self._live_tracker(), self._live_steps_of()
        step = tracker.step if tracker is not None else ""
        return step if step in steps else steps[0]

    def _live_draw(self) -> None:
        run = self._live
        if run is None:
            return
        dirty, self._live_dirty = self._live_dirty, set()
        self._live_show_status()
        self._live_show_steps(run)
        step = self._live_pinned or self._live_running_step()
        turned = step != self._live_shown or "all" in dirty
        if step != self._live_shown:
            self.live_steps.set_current(step)
        self._live_shown = step
        steps = self._live_steps_of()
        names, explain = (STATIC_STEP_NAMES, STATIC_EXPLAIN) if self._live_static() else (STEP_NAMES, EXPLAIN)
        self.live_title.setText(text(TITLE, index=steps.index(step) + 1, count=len(steps), name=tr(names[step])))
        self.live_explain.setText(tr(explain[step]))
        self._show_disk_note(run)
        if self._live_static():
            self._live_draw_static(run, step, turned, dirty)
            return
        if step == "intake":
            self.live_picture_host.setCurrentIndex(0)
            self.live_plot_host.setCurrentIndex(0)
            if turned or dirty & {"detector", "frames"}:
                self.live_frames.show(run.detector, run.frames)
            if turned or dirty & {"frames", "imported"}:
                times, rates = run.trace_points()
                self.live_trace.show(times, rates, run.imported.number("onset") if run.imported else None)
            self.live_caption.set_full_text(self._detector_caption(run))
            return
        self.live_picture_host.setCurrentIndex(1)
        self.live_plot_host.setCurrentIndex(1)
        latest = run.latest_patterns()
        if turned or "window" in dirty or (latest is None and "progress" in dirty):
            if latest is None:  # what the network does until its first window is solved
                doing = self._phase_line(run)
                self.live_patterns.show_message(doing + "\n" + tr(FIRST_WINDOW) if doing else tr(WAITING_SOLUTIONS))
            else:
                self.live_patterns.show(latest[0], latest[1], entry_name(latest[1]))
        if turned or dirty & {"window", "windows", "chosen"}:
            result = run.result(self._live_folder, run.shown_family())
            mark = result.nearest(latest[2]) if result is not None and latest is not None and latest[2] is not None \
                else None
            self.live_evolution.draw(result, None, mark)
        self.live_caption.set_full_text(self._window_caption(run, latest))

    def _show_disk_note(self, run: LiveRun) -> None:
        """The note that reading the frames from disk is the slow part (from then on, whatever step is shown)."""
        slow = run.disk_slow()
        if slow:
            self.live_disk_note.message_label.setText(tr(SLOW_DISK_NOTE))
        self.live_disk_note.setVisible(slow)

    def _live_show_steps(self, run: LiveRun) -> None:
        tracker, steps = self._live_tracker(), self._live_steps_of()
        running = self._live_running_step()
        for key in steps:
            failed = tracker is not None and key in tracker.failed
            done = tracker is not None and (key in tracker.done or bool(tracker.status))
            before = steps.index(key) < steps.index(running)
            if key == "predict" and run.families and not run.chosen and running == "predict":
                done = False  # the next network of the analysis is still at work
            state = "error" if failed else "ok" if done or before else "busy" if key == running else "pending"
            line = self._step_line(key, run, tracker)
            disk = disk_words(run.disk_per_frame(), run.disk_share()) if key == "intake" else ""
            self.live_steps.set_state(key, state, " · ".join(part for part in (line, disk) if part))

    def _step_line(self, key: str, run: LiveRun, tracker) -> str:
        if run.kind == "static":
            return self._static_step_line(key, run)
        if key == "intake":
            if self._live_reused:
                return tr(REUSED_LINE)
            imported = run.imported
            if imported is not None:
                first, last, onset = (imported.number(k) for k in ("t_first", "t_last", "onset"))
                values = {"frames": _count(imported.number("frames")), "first": shown_number(first, 4),
                          "last": shown_number(last, 4)}
                return text(IMPORTED_ONSET, onset=shown_number(onset, 4), **values) if onset is not None else \
                    text(IMPORTED_LINE, **values)
            if run.frames_total:  # a package that says the disk time: "Reading 312 of 1,241 frames · disk …"
                return text(READING_LINE if run.disk_per_frame() is not None else FRAMES_LINE,
                            done=_count(run.frames_done), total=_count(run.frames_total))
            if run.centre is not None:
                return text(CENTRE_LINE, col=f"{run.centre[0]:.1f}", row=f"{run.centre[1]:.1f}")
            return ""
        if key == "predict":
            if run.chosen:
                deviation = run.deviations.get(run.chosen)
                return text(CHOSEN_DEVIATION, family=run.chosen, deviation=shown_number(deviation)) \
                    if deviation is not None and len(run.deviations) > 1 else \
                    text(CHOSEN_LINE, family=run.chosen) if len(run.deviations) > 1 else ""
            return self._phase_line(run)
        if key == "present" and run.drawing:
            return text(DRAWN_LINE, drawn=run.drawn, total=run.drawing)
        return ""

    @staticmethod
    def _phase_line(run: LiveRun) -> str:
        """What the network at work is doing and how far it has come ("" before it starts)."""
        if not run.family:
            return ""
        phase, total = run.phase() or "render", run.totals.get(run.family, 0)
        images = run.phase_count("render", key="images")[0] + run.phase_count("refine", key="evaluations")[0]
        if phase in ("render", "refine") and not images:  # a package that does not count them
            words = text(WINDOWS_LINE, solved=run.solved(), total=total)
        elif phase in ("render", "refine"):
            words = text(PHASE_LINES[phase], solved=run.solved(), done=run.refined(), total=total,
                         images=_count(images))
        else:
            done, phase_total = run.phase_count(phase)
            words = text(PHASE_LINES[phase], done=_count(done), total=_count(phase_total or total))
        index, count = run.network_index()
        return text(NETWORK_PREFIX, index=index, count=count, what=words) if count > 1 else words

    def _detector_caption(self, run: LiveRun) -> str:
        event = run.detector
        if event is None:
            return tr(NOTHING_LIVE) if self._nothing_live() else tr(WAITING_FRAMES) if not run.frames_total else ""
        masked = (event.number("pixels") or 0) - (event.number("kept") or 0)
        words = text(DETECTOR_CAPTION, frames=_count(event.number("frames_summed")), masked=_count(masked),
                     dead=_count(event.number("dead")), hot=_count(event.number("hot")),
                     shadow=_count(event.number("shadow")))
        if run.centre is not None:
            words = text(CENTRE_LINE, col=f"{run.centre[0]:.1f}", row=f"{run.centre[1]:.1f}") + " · " + words
        return words

    def _window_caption(self, run: LiveRun, latest) -> str:
        if latest is None:
            return tr(NOTHING_LIVE) if self._nothing_live() else ""
        _patterns, label, time_s, event = latest
        values = event.get("values") if isinstance(event.get("values"), dict) else {}
        parts = [text(WINDOW_CAPTION, time=shown_number(time_s, 4)), entry_name(label)]
        for key in CAPTION_KEYS:
            value = values.get(key)
            if isinstance(value, (int, float)):
                caption, unit = TILE_WORDS[key]
                digits = 2 if key == "coverage" else 3
                parts.append(f"{tr(caption)} {shown_number(value, digits)} {unit}".rstrip())
        deviation = event.number("fit_deviation")
        if deviation is not None:
            parts.append(text(DEVIATION_PART, deviation=shown_number(deviation)))
        return " · ".join(parts)

    def _nothing_live(self) -> bool:
        """The package has sent no live event although its first step is over (an older package)."""
        tracker = self._live_tracker()
        return self._live_events == 0 and tracker is not None and \
            self._live_steps_of().index(self._live_running_step()) > 0

    # -- theme and language --------------------------------------------------------------------------------------
    def redraw_live_plots(self) -> None:
        for picture in (self.live_frames, self.live_trace, self.live_patterns, self.live_evolution):
            picture.redraw()

    def refresh_live_language(self) -> None:
        if self._live is None:
            return
        self._live_tick()
        self._live_titles()
        self.redraw_live_plots()
        self._live_show_joint()  # its label in the new language
        self._live_dirty.add("all")
        self._live_draw()


__all__ = ["CAPTION_KEYS", "EXPLAIN", "LIVE_PAGE", "LivePanelMixin", "REDRAW_MS", "STEP_NAMES", "TEMPLATES"]
