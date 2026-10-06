"""Which file and frames the automatic analysis's results belong to (a mixin of ``GuidedAnalysis``).

The results belong to one file and, in a series, to the frames the run analysed: ``frame_shown(path)`` puts
away the results of another file (they come back with it) and turns off what acts on the frame while Analyze
shows other frames of the series, so a report is never saved or sent to Fitting next to another frame.
``files_cleared()`` (Clear, a project opened) forgets every report and earlier answer; ``file_removed(path)`` the
reports of one file taken off the list.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from src.gimap.app.presentation.i18n import tr, trf

from .guided_text import IDLE_TEXT, OTHER_FRAMES_TEXT, other_frames


def frame_key(path) -> Optional[str]:
    """A file's identity for the results: its absolute path (frames of one file share it)."""
    return os.path.normcase(os.path.abspath(str(path))) if path else None


class GuidedFramesMixin:
    """``frame_shown`` / ``files_cleared`` and what follows from them; needs the ``GuidedAnalysis`` state."""

    def frame_shown(self, path: str, frame: Optional[int] = None, summed: Optional[int] = None) -> None:
        """Analyze shows ``path`` (``analysisShown``); ``frame`` (1-based) and ``summed``: which frames of a
        series, when the caller knows (else Analyze's status says).

        Nothing changes during a run (the file is looked at when its result is handled). Results of another
        file are put away and come back with it. Other frames of the results' series keep them on screen,
        but Refine, Show in Fitting, Compare and Save Report are off until the run's frames are shown again.
        """
        key = frame_key(path)
        if key is None:
            return
        self._frames_given = (key, int(frame), int(summed or 1)) if frame is not None else None
        if self._busy():
            self._pending_path = key
            return
        if key == self._shown_path:
            self._follow_frames()  # the same file: perhaps other frames of it
            return
        self._shown_path = key
        shown = self.report
        if shown is not None and frame_key(shown.get("frame")) == key:
            self._follow_frames()
            return
        stored = self._reports.get(key)
        if stored is not None:
            self._restore(*stored)
        elif shown is not None:
            self._put_away(shown)
        self._follow_frames()

    def files_cleared(self) -> None:
        """Analyze ▸ Clear or a project opened: every report and earlier answer is forgotten (an answer would
        override a project's own αi and energy unseen); a run in progress is stopped and forgotten at its end."""
        if self._busy():
            self._clear_after_run, self._pending_path = True, None  # only a file shown after the Clear counts
            self.stop()
            return
        self._forget_all()

    def file_removed(self, path) -> None:
        """Analyze removed ``path`` from its list: its results are forgotten (they cannot come back with it)."""
        key = frame_key(path)
        if key is None or self._busy():  # the list is locked during a run, so no run is about this file
            return
        self._reports.pop(key, None)
        if self.report is not None and frame_key(self.report.get("frame")) == key:
            self.report = self.start_report = None
            self.results.hide()
            self.questions.hide()
            self.progress_panel.dismiss()
            self._say(lambda: tr(IDLE_TEXT))
            self._frames_elsewhere = False
        if self._shown_path == key:
            self._shown_path = None
        self._sync_actions()

    def _forget_all(self) -> None:
        self._clear_after_run = False
        self._reports.clear()
        self.report = self.start_report = None
        self._shown_path = self._pending_path = None
        self._frames_given = None
        self._frames_elsewhere = False
        self.questions.forget()
        self.results.hide()
        self.progress_panel.dismiss()
        self._say(lambda: tr(IDLE_TEXT))
        self._sync_actions()

    def _restore(self, report: dict, start_report: Optional[dict]) -> None:
        """The results of a file analysed before, as they were."""
        self.report, self.start_report = report, start_report
        self.progress_panel.dismiss()
        self._show_questions(report.get("needs_attention") or [])
        self.results.show_report(report, start_report)
        self._say(lambda: self._status_text(report))
        self._frames_elsewhere = False

    def _put_away(self, report: dict) -> None:
        """Hide the results of another file (kept: they come back with their file)."""
        self.report = self.start_report = None
        self.results.hide()
        self.questions.hide()
        self.progress_panel.dismiss()
        name = Path(str(report.get("frame") or "")).name
        self._say(lambda: trf("Results are for {name}; run again for this frame", name=name))
        self._frames_elsewhere = False

    def _frames_on_screen(self, key: Optional[str]) -> Optional[tuple[int, int]]:
        """(first frame, frames summed) Analyze shows of the file ``key``; ``None`` when unknown."""
        status = self._status()
        if key is not None and frame_key(status.get("path")) == key and status.get("frame") is not None:
            return int(status["frame"]), int(status.get("summed_frames") or 1)
        given = self._frames_given
        if given is not None and given[0] == key:
            return given[1], given[2]
        return None

    def _other_frames(self) -> Optional[tuple[int, int]]:
        """(first, last) frame of the series the report is for, while Analyze shows other frames of it."""
        if self.report is None or int((self.report.get("frames") or {}).get("total") or 1) <= 1:
            return None
        return other_frames(self.report, self._frames_on_screen(frame_key(self.report.get("frame"))))

    def _follow_frames(self) -> None:
        """The actions follow the frames shown; the status says when they are other frames of the series."""
        other = self._other_frames()
        if other is not None:
            self._say(lambda: trf(OTHER_FRAMES_TEXT, a=other[0], b=other[1]))
            self._frames_elsewhere = True
        elif self._frames_elsewhere:
            self._frames_elsewhere = False
            report = self.report
            if report is not None:
                self._say(lambda: self._status_text(report))
        self._sync_actions()

    def _for_this_frame(self) -> bool:
        """Whether the report is the one of the file Analyze shows (unknown: yes) and of its frames (series)."""
        report = self.report
        if report is None or (self._shown_path is not None and frame_key(report.get("frame")) != self._shown_path):
            return False
        return self._other_frames() is None

    def _sync_actions(self) -> None:
        current = self._for_this_frame()
        self.results.set_actions_enabled(current)
        self.progress_panel.save_button.setEnabled(current)

    def _frame_folder(self) -> str:
        """The folder of the frame Analyze shows (else of the results' frame): where a calibration file dialog starts."""
        path = self._status().get("path") or (self.report or {}).get("frame") or self._shown_path
        return str(Path(str(path)).parent) if path else ""


__all__ = ["GuidedFramesMixin", "frame_key"]
