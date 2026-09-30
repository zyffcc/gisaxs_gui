"""The Analyze command bar: never clips a control, drops the least important ones when narrow.

With the AI panel docked, or on a small screen, the bar can be ~1000 px wide. Below
``COMPACT_WIDTH`` it hides the file stepping arrows, the file details and the αi caption, and
shortens the two long button texts; the full texts stay in the tooltips. The file name is
elided in the middle rather than cut.
"""

from __future__ import annotations

from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFrame, QLabel, QSizePolicy, QWidget

from src.gimap.app.presentation.i18n import tr

COMPACT_WIDTH = 1180
"""Below this bar width the compact texts and the hidden extras apply."""


class ElidedLabel(QLabel):
    """A label that shows as much of its text as fits, with “…” in the middle; the tooltip has it all."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.setMinimumWidth(60)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt API
        self._full = str(text)
        self._refresh()

    def full_text(self) -> str:
        return self._full

    def sizeHint(self):  # noqa: N802 - Qt API
        hint = super().sizeHint()
        hint.setWidth(min(self.fontMetrics().horizontalAdvance(self._full) + 24, 320))
        return hint

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        room = max(20, self.width() - 16)
        shown = self.fontMetrics().elidedText(self._full, Qt.ElideMiddle, room)
        super().setText(shown)
        if shown != self._full and not self.toolTip():
            self.setToolTip(self._full)


class CommandBar(QFrame):
    """``set_compact_parts(hidden=..., texts=...)`` names what changes when narrow."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._hidden: tuple = ()
        self._texts: tuple = ()
        self.compact: Optional[bool] = None

    def set_compact_parts(self, *, hidden=(), texts=()) -> None:
        """``texts``: ``(button, full, short)``; buttons keep at least their text width."""
        self._hidden = tuple(hidden)
        self._texts = tuple(texts)
        self._apply(self.width() < COMPACT_WIDTH)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._apply(event.size().width() < COMPACT_WIDTH)

    def _apply(self, compact: bool) -> None:
        if compact == self.compact:
            return
        self.compact = compact
        for widget in self._hidden:
            widget.setVisible(not compact and widget.property("gimapWanted") is not False)
        for button, full, short in self._texts:
            button.setText(tr(short if compact else full))
            button.setToolTip(button.toolTip() or tr(full))


__all__ = ["COMPACT_WIDTH", "CommandBar", "ElidedLabel"]
