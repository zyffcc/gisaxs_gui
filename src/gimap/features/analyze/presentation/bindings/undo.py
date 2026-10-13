"""Undo / Redo of the set-up in Analyze: the ↶ ↷ buttons of the command bar, Ctrl+Z and Ctrl+Shift+Z (Ctrl+Y).

Every manual change — a mask drawn, a region added or moved, a band dragged, the profile, αi or the
mode chosen, a settings file loaded, an AI card applied — ends in ``run_analysis``, which first lets
the history see the set-up (``setup_history.py``). Undo puts the previous set-up back, updates the
controls and analyses again; the tooltip says what the next Undo / Redo changes. A text field with
the focus keeps Ctrl+Z for its own text.
"""

from __future__ import annotations

import time

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QShortcut

from src.gimap.app.presentation.components import show_toast, tool_icon
from src.gimap.app.presentation.i18n import tr
from src.gimap.app.presentation.theme import theme_manager

from ..setup_history import SetupHistory, restore, snapshot


class UndoMixin:
    """Needs ``view_model``, ``undo_button``, ``redo_button``, ``run_analysis``, ``_sync_settings_widgets``."""

    def _connect_undo(self) -> None:
        self.setup_history = SetupHistory()
        self.undo_button.clicked.connect(self.undo_setup)
        self.redo_button.clicked.connect(self.redo_setup)
        for keys, slot in (("Ctrl+Z", self.undo_setup), ("Ctrl+Shift+Z", self.redo_setup), ("Ctrl+Y", self.redo_setup)):
            shortcut = QShortcut(QKeySequence(keys), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
        theme_manager().changed.connect(self._paint_undo_icons)
        self._paint_undo_icons()
        self._refresh_undo_buttons()

    def _dispose_undo(self) -> None:
        try:
            theme_manager().changed.disconnect(self._paint_undo_icons)
        except (TypeError, RuntimeError):
            pass

    def _paint_undo_icons(self, *_args) -> None:
        color = theme_manager().color("text")
        self.undo_button.setIcon(tool_icon("undo", color))
        self.redo_button.setIcon(tool_icon("redo", color))

    def _observe_setup(self) -> None:
        """Before an analysis: a set-up different from the last one is a step to undo."""
        if self.setup_history.observe(snapshot(self.view_model), time.monotonic()):
            self._refresh_undo_buttons()

    def undo_setup(self) -> bool:
        return self._step(self.setup_history.undo(), tr("Undone: {what}"), (tr("Redo"), self.redo_setup))

    def redo_setup(self) -> bool:
        return self._step(self.setup_history.redo(), tr("Redone: {what}"), (tr("Undo"), self.undo_setup))

    def _step(self, step, message: str, action) -> bool:
        if step is None:
            return False
        setup, what = step
        restore(self.view_model, setup)
        self._sync_settings_widgets()
        self._refresh_undo_buttons()
        self.run_analysis()
        text = message.format(what=_what(what))
        self._status(text, "ok")
        if self.isVisible():
            show_toast(self.window(), text, level="info", action=action, timeout_ms=2500)
        return True

    def _refresh_undo_buttons(self) -> None:
        history = self.setup_history
        undo, redo = history.next_undo(), history.next_redo()
        self.undo_button.setEnabled(undo is not None)
        self.redo_button.setEnabled(redo is not None)
        self.undo_button.setToolTip(
            tr("Undo: {what} (Ctrl+Z)").format(what=_what(undo)) if undo else tr("Nothing to undo (Ctrl+Z)"))
        self.redo_button.setToolTip(
            tr("Redo: {what} (Ctrl+Shift+Z)").format(what=_what(redo)) if redo else tr("Nothing to redo (Ctrl+Shift+Z)"))


def _what(what: str) -> str:
    """A step's name in the interface language (“mask and cut regions”: each part)."""
    if " and " in what:
        first, second = what.split(" and ", 1)
        return tr("{a} and {b}").format(a=tr(first), b=tr(second))
    return tr(what)


__all__ = ["UndoMixin"]
