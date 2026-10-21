"""The mode chosen for the frames (Auto, GISAXS, GIWAXS) and what a frame says about itself.

``choose_mode`` is the one way to change the mode from outside (the Start page's task cards, the
assistant): the control, the session state and the remembered choice change together, and a
frame on screen is reduced again. A mode pinned by hand that disagrees with what Auto would
choose for the frame is said in the status line and, once per file and mode, in a notice that
offers Auto. The reductions themselves never change behind the user's back.

A frame that cannot be read or reduced says why in the interface language, and takes the frame
shown before off the screen: its image, curves and steps must not stay under the new file's name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.shared.detector_io.nexus_modules import MODULES_FEWER

from ...application import AUTO, GISAXS, GIWAXS, MODES
from ..texts import message_text
from .data_step import SENTENCE
from .display import NOT_ANALYSED_TEXT

MODE_NAMES = {AUTO: "Auto", GISAXS: "GISAXS", GIWAXS: "GIWAXS"}
MODE_MISMATCH = "This frame looks like {detected} but {mode} is selected"


class FrameChecksMixin:
    """Needs ``mode_combo``, ``view_model``, ``run_analysis``, ``_remember`` and ``_status``."""

    _automatic_busy = False
    _failed: Optional[tuple] = None
    """``(message, forgotten, status text said)`` of the analysis that failed last, until another one starts, a
    frame is shown or the list is cleared: the English sentence is said again after a language switch
    (``_say_failure``), and when an automatic analysis is asked for a frame that could not be read."""
    _kept_mode: Optional[str] = None
    """The person's mode while the automatic analysis switched it for its run (``AnalyzeAutomation.set_mode(...,
    remember=False)``): saved instead of the switched one until the person chooses a mode."""

    # -- the mode --------------------------------------------------------------------------

    def choose_mode(self, mode: str) -> None:
        """Analyse this and the following frames as ``auto``, ``gisaxs`` or ``giwaxs`` (remembered for the next
        session); a frame on screen is reduced again at once."""
        if mode not in MODES:
            raise ValueError(f"Unknown analysis mode {mode!r}; expected one of {MODES}.")
        with QSignalBlocker(self.mode_combo):
            self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(mode)))
        self.view_model.set_mode(mode)
        self._kept_mode = None  # the person's choice: remembered as it is
        self._remember()
        if self.view_model.current_path is not None:
            self.run_analysis()

    def _mode_chosen(self, _index: int) -> None:
        self.choose_mode(self.mode_combo.currentData())

    def _mode_warnings(self) -> set:
        warned = getattr(self, "_mode_warned", None)
        if warned is None:
            warned = self._mode_warned = set()
        return warned

    def _mode_mismatch(self, analysis) -> Optional[str]:
        """The sentence saying the mode chosen by hand disagrees with the frame shown, or ``None``."""
        mode = self.view_model.state.mode
        detected = getattr(analysis, "detected_kind", None)
        # Only a frame reduced in the mode chosen now (a result of an earlier choice is about to be replaced).
        if mode == AUTO or detected is None or analysis.kind != mode or detected == analysis.kind:
            return None
        return tr(MODE_MISMATCH).format(
            detected=MODE_NAMES.get(detected, str(detected).upper()), mode=MODE_NAMES.get(mode, str(mode).upper()))

    def _check_mode(self, analysis) -> bool:
        """After a frame is shown: does the mode chosen by hand disagree with the frame? (``True``: it does.)"""
        text = self._mode_mismatch(analysis)
        if text is None:
            self._close_mode_notice()  # what it said no longer holds for the frame shown
            return False
        mode = self.view_model.state.mode
        self._status(" · ".join([text, *self._frame_messages(analysis)]), "warning")
        key = (str(analysis.path), mode)
        if key not in self._mode_warnings() and not self._automatic_busy:
            self._mode_warnings().add(key)
            self._mode_notice = show_toast(
                self.window(), text, level="warning", action=(tr("Use Auto"), lambda: self.choose_mode(AUTO)))
        return True

    def _close_mode_notice(self) -> None:
        notice, self._mode_notice = getattr(self, "_mode_notice", None), None
        if notice is not None:
            try:
                notice.close()
            except (RuntimeError, AttributeError):  # closed (and deleted) already
                pass

    # -- messages of a frame -----------------------------------------------------------------

    def _frame_messages(self, analysis) -> list[str]:
        """What is left out of the frame's source or may be wrong with it (a module file missing, modules with
        different numbers of frames, a file still written: ``bindings/data_step.py``), why the folder watch
        does not export it yet (``bindings/batch_watch.py``), then the frame's own messages (English in
        ``analysis.messages``) in the interface language (``texts.py``). The source comes first: a frame
        without a module has another size, which is why no instrument profile fits it. While the watch says
        that a set waits for module files, the reader's remark that the other scans have more is left out:
        the line would say it twice (the Data step shows it all the same)."""
        notes = self._watch_notes(analysis.path)
        source = self._source_warnings(analysis.path, without=(MODULES_FEWER,) if notes else ())
        source += [trf(SENTENCE, text=note) for note in notes]
        return source + [message_text(message) for message in analysis.messages]

    # -- a frame that could not be analysed -------------------------------------------------------

    def _analysis_failed(self, message: str, _details: str) -> None:
        """``message`` is the application's English sentence: shown in the interface language, emitted as it
        came (the assistant and the headless run read it)."""
        self._failed = (message, self._forget_frame_shown(), None)
        self._say_failure()
        self.analysisFailed.emit(message)

    def _say_failure(self) -> None:
        """Why the frame asked for is not shown: the Data step, the command bar and, forgotten, the Data card;
        the status line too, the first time and after a language switch while it still says so."""
        message, forgotten, said = self._failed
        shown = message_text(message)
        if said is None or self.status_text() == said:
            self._status(trf("Could not analyse the frame: {error}", error=shown), "error")
            self._failed = (message, forgotten, self.status_text())
        self.file_meta.setText(tr("could not be read"))
        self.set_step_state("data", "error", shown)
        if forgotten:  # the card described the frame before
            self.data_info_label.setText(shown)
            path = self.view_model.current_path  # what is known of the file asked for (its files, its warnings)
            self._show_source(self.view_model.describe(path) if path is not None else None, path)

    def _forget_frame_shown(self) -> bool:
        """The frame asked for could not be analysed. Unless it is the frame held (a setting that cannot be
        applied: its last good reduction stays on screen), nothing of the frame before stays under the new
        file's name: no image, no curves, no green steps, nothing to export or send to Fitting. ``True`` when
        the frame before was forgotten."""
        state, path = self.view_model.state, self.view_model.current_path
        held = state.analysis
        same_file = held is not None and path is not None and Path(held.path) == Path(path)
        if same_file and held.frame_index == state.frame_index:
            return False
        if not same_file:
            self._update_frame_controls(None)  # another file: how many frames it has is not known
        self.view_model.forget_analysis()
        self._shown_status = None
        self._empty_views(NOT_ANALYSED_TEXT)
        self.summary_label.setText(self.view_model.geometry_summary(tr))
        self._refresh_workspace(None)  # Geometry, Mask and Cuts no longer describe the frame before
        self._refresh_series_curves(None)  # nor does the Series tab offer its curves
        if self._automatic_busy:  # a run goes on: its step stays as it was
            self.set_step_state("results", "busy", tr("Running…"))
        elif self._results_by_path:
            self._show_results_elsewhere()  # results kept are another file's, and named so
        self.fit_button.setEnabled(False)
        self._update_center_control()
        self._forget_incidence()  # the field showed the αi of the frame before
        return True


__all__ = ["FrameChecksMixin", "MODE_MISMATCH", "MODE_NAMES"]
