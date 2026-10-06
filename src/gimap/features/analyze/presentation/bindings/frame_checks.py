"""The mode chosen for the frames (Auto, GISAXS, GIWAXS) and what a frame says about itself.

``choose_mode`` is the one way to change the mode from outside (the Start page's task cards, the
assistant): the control, the session state and the remembered choice change together, and a
frame on screen is reduced again. A mode pinned by hand that disagrees with what Auto would
choose for the frame is said in the status line and, once per file and mode, in a notice that
offers Auto. The reductions themselves never change behind the user's back.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import QSignalBlocker

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr

from ...application import AUTO, GISAXS, GIWAXS, MODES
from ..texts import message_text

MODE_NAMES = {AUTO: "Auto", GISAXS: "GISAXS", GIWAXS: "GIWAXS"}
MODE_MISMATCH = "This frame looks like {detected} but {mode} is selected"


class FrameChecksMixin:
    """Needs ``mode_combo``, ``view_model``, ``run_analysis``, ``_remember`` and ``_status``."""

    _automatic_busy = False
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
        """The frame's messages (English in ``analysis.messages``) in the interface language (``texts.py``)."""
        return [message_text(message) for message in analysis.messages]


__all__ = ["FrameChecksMixin", "MODE_MISMATCH", "MODE_NAMES"]
