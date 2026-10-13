"""The Results step and tab follow the frame on screen.

The automatic analysis (or the AI) finds results for one file. They are kept per file for the
session, as the automatic analysis keeps its reports: a file shown again gets its Results step back
(its state and summary). A file without results says whose results were shown last and asks for a
run on this frame, and the right panel goes back to the curves. In a series (a file of several
frames) the results belong to the frames the run analysed: other frames of it say which frames the
results are for (the Results tab stays, as the automatic analysis keeps its report on screen), and the
run's frames get the step back. The frames the run itself reduces while it works change nothing.
Behaviour only: the widgets are in ``views/``.

The run works on the frame Analyze shows, so while it lasts that frame stays: the controls that open
or show another one are disabled, and a frame asked for from code (PgUp/PgDown, folder watching, a
drop) is not shown — new files are still listed. Clear (also before a project opens) stops the run
and drops what it finds: those results would belong to frames no longer listed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.i18n import tr, trf

RESULTS_INTRO = "Run the standard procedure: geometry, mask, cuts and results, each with its reason."
"""The Results step before any run."""
RESULTS_ELSEWHERE = "Results are for {name}; run again for this frame"
"""The same text as the automatic analysis's own line (one translation for both)."""
RESULTS_OTHER_FRAMES = "Results are for frames {a}–{b}; run again for this frame"
"""Other frames of the series the results are for: the automatic analysis's own line (one translation)."""
RUN_KEEPS_FRAME = "The automatic analysis works on this frame until it ends; stop it to show another one"
"""Another frame was asked for while the run works on the one shown."""
RUN_LISTED = "{n} new frame(s) listed; the automatic analysis works on this frame until it ends"
"""Files opened while the run works: listed, not shown."""
KEPT_RESULTS = 24
"""Files whose results are kept for this session (the automatic analysis keeps as many reports)."""


def results_key(path) -> Optional[str]:
    """A file's identity for its results: the absolute path (frames of one file share it)."""
    return os.path.normcase(os.path.abspath(str(path))) if path else None


def other_frames(frames, analysis) -> Optional[tuple[int, int]]:
    """``(first, last)`` frame (1-based) of the series the results are for, when ``analysis`` shows other
    frames of it; ``None`` when it shows the run's frames, the file is no series, or ``frames`` is unknown.

    ``frames``: the run's ``{"first": 1-based, "summed": n, "total": frames in the file}``; ``analysis``
    shows ``frame_index + 1`` and ``frame_total`` frames summed from it."""
    if not isinstance(frames, dict) or analysis is None:
        return None
    try:
        total = int(frames.get("total") or 1)
        first, summed = int(frames.get("first") or 1), int(frames.get("summed") or 1)
    except (TypeError, ValueError):
        return None
    if total <= 1 or (int(analysis.frame_index) + 1, int(analysis.frame_total)) == (first, summed):
        return None
    return first, first + summed - 1


class ResultsStateMixin:
    """Needs the step rail and intros (``set_step_state``, ``step_intro``, ``_intro``), ``show_right``,
    ``current_right``, ``show_step``, the command-bar buttons of the automatic analysis and ``_status``."""

    _results_shown: Optional[str] = None
    """The file (``results_key``) whose results the Results step shows, or showed last."""
    _results_where: Optional[str] = None
    """The file shown when the Results step was last looked at."""
    _results_elsewhere = False
    """The Results step does not show results of the file on screen."""
    _automatic_busy = False
    """A run works on the frame shown: Analyze keeps that frame until the run ends."""
    _automatic_dropped = False
    """The files of the run were cleared: its end restores the buttons and shows nothing."""
    _stop_pipeline: Optional[Callable[[], None]] = None
    _results_kept: Optional[dict] = None

    @property
    def _results_by_path(self) -> dict:
        """``results_key`` → ``(path, state, detail, frames)`` of the Results step, the newest last."""
        kept = self._results_kept
        if kept is None:
            kept = self._results_kept = {}
        return kept

    # -- the run (called by the component that runs the automatic analysis) --------------------

    def automatic_started(self, message: str) -> None:
        self._automatic_busy = True
        self._automatic_dropped = False
        self._keep_frame(True)
        self.run_pipeline_button.setEnabled(False)
        self.find_geometry_button.setEnabled(False)
        self.banner_find_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        self.stop_pipeline_button.setVisible(self._stop_pipeline is not None)
        self.stop_pipeline_button.setEnabled(True)
        self.stop_pipeline_button.setText(tr("Stop"))
        self.show_right("results")  # the progress of the run is at the top of the Results tab
        self.set_step_state("results", "busy", tr("Running…"))
        self._status(message)

    def automatic_progress(self, text: str) -> None:
        if not self._automatic_dropped:  # a run whose files were cleared says nothing over the new state
            self._status(text)

    def automatic_stopping(self) -> None:
        """Stop was asked for (here or on the progress panel): the run ends after the step in progress."""
        self.stop_pipeline_button.setEnabled(False)
        self.stop_pipeline_button.setText(tr("Stopping…"))
        self._status(tr("Stopping after the current step …"))

    def _stop_clicked(self) -> None:
        if self._stop_pipeline is not None:
            self._stop_pipeline()

    def automatic_finished(self, state: str, detail: str, *, show_results: bool = True, path=None,
                           frames: Optional[dict] = None) -> None:
        """``state``: ``ok`` or ``warn`` (questions only a person can answer). ``path``: the file the results
        belong to (the report's frame); by default the file shown. ``frames``: in a series, the frames the
        run analysed (``{"first": 1-based, "summed": n, "total": frames in the file}``; ``None``: the file).
        The outcome is shown as this frame's only when it is the file (and the frames) shown; results of
        another file are kept for it and named, those of other frames of the series say which frames."""
        if self._end_run():
            return
        analysis = self.view_model.state.analysis
        shown = results_key(analysis.path) if analysis is not None else None
        if path is None and analysis is not None:
            path = analysis.path
        key = results_key(path)
        if key is not None:
            kept = self._results_by_path
            kept.pop(key, None)
            kept[key] = (str(path), state, detail, dict(frames) if isinstance(frames, dict) else None)
            while len(kept) > KEPT_RESULTS:
                kept.pop(next(iter(kept)))
            self._results_shown = key
        self._results_where = shown
        self._results_elsewhere = False
        other = other_frames(frames, analysis) if key is not None and shown == key else None
        if other is not None:  # the run's frames are not the ones shown: they get the results when shown again
            self._say_other_frames(other)
            if show_results:  # the report stays in the tab, saying which frames it is for
                self.show_right("results")
            self._status(trf(RESULTS_OTHER_FRAMES, a=other[0], b=other[1]), "info")
            return
        if key is None or shown == key:  # this frame's results (or a run that names no file, e.g. none open)
            self.set_step_state("results", state, detail)
            self._intro("results", detail)
            if show_results:
                self.show_right("results")
            if state != "ok":
                self.show_step("results")
            self._status(detail, "ok" if state == "ok" else "warning")
            return
        if shown is None:  # nothing on screen: the step as before any run; the file gets them back when shown
            self.set_step_state("results", "pending")
            self._intro("results", RESULTS_INTRO)
            if self.current_right() == "results":
                self.show_right("curves")
            self._results_elsewhere = True
        else:  # the frame shown is another file's
            self._show_results_elsewhere()
        self._status(tr(RESULTS_ELSEWHERE).format(name=Path(str(path)).name), "info")

    def automatic_failed(self, message: str) -> None:
        if self._end_run():
            return
        self.set_step_state("results", "error", message)
        self._status(trf("The automatic analysis stopped: {error}", error=message), "error")

    def _end_run(self) -> bool:
        """The run ended: Analyze is free again. ``True`` when its files were cleared (nothing to show)."""
        dropped = self._automatic_dropped
        self._automatic_busy = self._automatic_dropped = False
        self._end_run_buttons()
        self._keep_frame(False)
        return dropped

    def _end_run_buttons(self) -> None:
        for button in (self.run_pipeline_button, self.find_geometry_button, self.banner_find_button):
            button.setEnabled(True)
        self.stop_pipeline_button.hide()
        self.progress_bar.setVisible(self.tasks.is_busy())

    def _drop_run(self) -> None:
        """The files of a running analysis are cleared (Clear, a project opened): stop it and drop its results.

        The run ends after the step in progress; Analyze is free at once (nothing is left to keep), the
        Run buttons only when the run has ended."""
        if not self._automatic_busy:
            return
        self._stop_clicked()
        self._automatic_busy, self._automatic_dropped = False, True
        self._keep_frame(False)

    # -- keeping the frame the run works on --------------------------------------------------

    def _keep_frame(self, keep: bool) -> None:
        """While a run works on the frame shown, the controls that show another frame are disabled. Watch
        stays: it only lists new frames (``_frame_kept`` keeps the one shown), and it must stay stoppable."""
        for widget in (self.file_list, self.frame_spin, self.open_files_button, self.clear_button):
            widget.setEnabled(not keep)
        self.open_folder_action.setEnabled(not keep)
        row, count = self.file_list.currentRow(), self.file_list.count()
        self.previous_file_button.setEnabled(not keep and row > 0)
        self.next_file_button.setEnabled(not keep and 0 <= row < count - 1)

    def _frame_kept(self) -> bool:
        """A run works on the frame shown: put the list back on it and say why (``True``); else ``False``."""
        if not self._automatic_busy:
            return False
        with QSignalBlocker(self.file_list):
            self.file_list.setCurrentRow(self.view_model.state.current_index)
        with QSignalBlocker(self.frame_spin):
            self.frame_spin.setValue(self.view_model.state.frame_index + 1)
        self._status(tr(RUN_KEEPS_FRAME), "warning")
        return True

    # -- following the frame shown ---------------------------------------------------------------

    def results_for(self) -> Optional[Path]:
        """The file whose results the Results step shows, or showed last (``None``: no results)."""
        kept = self._results_by_path.get(self._results_shown)
        return Path(kept[0]) if kept is not None else None

    def forget_results(self, path=None) -> None:
        """Forget the results of the file ``path`` (``None``: those the Results step shows), e.g. a stopped
        run the automatic analysis discarded; that file's Results step is as before any run again."""
        key = results_key(path) if path is not None else self._results_shown
        if key is None or self._results_by_path.pop(key, None) is None:
            return
        if key != self._results_shown:
            return
        self._results_shown = None
        self._results_elsewhere = True  # the frame shown has no results (any more)
        self.set_step_state("results", "pending")
        self._intro("results", RESULTS_INTRO)

    def _follow_results(self, analysis) -> None:
        """After a frame is shown (every time, also another frame of the same file): the Results step has
        this file's results, says which frames of the series they are for, or says whose they are."""
        if self._automatic_busy or analysis is None or not self._results_by_path:
            return
        key = results_key(analysis.path)
        moved = key != self._results_where
        self._results_where = key
        kept = self._results_by_path.get(key)
        if kept is not None:
            self._results_shown = key
            other = other_frames(kept[3], analysis)
            if other is not None:  # other frames of the series the results are for
                self._say_other_frames(other)
            elif moved or self._results_elsewhere:  # re-reductions of the same frames leave the step as it is
                _path, state, detail, _frames = kept
                self.set_step_state("results", state, detail)
                self._intro("results", detail)
                self._results_elsewhere = False
            return
        if moved or not self._results_elsewhere:
            self._show_results_elsewhere()

    def _say_results(self, text: str) -> None:
        """The Results step says ``text`` (whose results they are) instead of results of the frame shown."""
        self.set_step_state("results", "pending", text)
        self.step_intro["results"].setText(text)
        self._results_elsewhere = True

    def _say_other_frames(self, frames: tuple[int, int]) -> None:
        """Another frame of the series is shown: which frames the results are for. The Results tab stays (the
        automatic analysis keeps its report there, saying the same)."""
        self._say_results(trf(RESULTS_OTHER_FRAMES, a=frames[0], b=frames[1]))

    def _show_results_elsewhere(self) -> None:
        """The frame shown has no results: say whose results were shown last, and show the curves."""
        named = self._results_by_path.get(self._results_shown)
        if named is not None:
            self._say_results(trf(RESULTS_ELSEWHERE, name=Path(named[0]).name))
        else:  # those results were forgotten: nothing to name
            self.set_step_state("results", "pending")
            self._intro("results", RESULTS_INTRO)
        if self.current_right() == "results":
            self.show_right("curves")
        self._results_elsewhere = True

    def _results_language(self) -> None:
        """After a language switch: the Results step's own sentence (whose results, which frames) again.
        A run's outcome stays as the run wrote it; the right panel stays as it is."""
        analysis = self.view_model.state.analysis
        if self._automatic_busy or analysis is None or not self._results_elsewhere:
            return
        kept = self._results_by_path.get(results_key(analysis.path))
        other = other_frames(kept[3], analysis) if kept is not None else None
        named = self._results_by_path.get(self._results_shown)
        if other is not None:
            self._say_other_frames(other)
        elif kept is None and named is not None:
            self._say_results(trf(RESULTS_ELSEWHERE, name=Path(named[0]).name))

    def _clear_results(self) -> None:
        """No files any more: the Results step is as before any run (the automatic analysis forgets its
        reports too, ``filesCleared``), and the right panel shows the curves."""
        self._results_by_path.clear()
        self._results_shown = self._results_where = None
        self._results_elsewhere = False
        self.set_step_state("results", "pending")
        self._intro("results", RESULTS_INTRO)
        if self.current_right() == "results":
            self.show_right("curves")


__all__ = [
    "KEPT_RESULTS", "RESULTS_ELSEWHERE", "RESULTS_INTRO", "RESULTS_OTHER_FRAMES", "RUN_KEEPS_FRAME", "RUN_LISTED",
    "ResultsStateMixin", "other_frames", "results_key",
]
