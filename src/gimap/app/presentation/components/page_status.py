"""A page's own line on a status bar that several pages share (the Labs pages: 2D Prediction and Trainset)."""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, pyqtSlot


class PageStatus(QObject):
    """Keeps the last status message of one page and says it again when the page is shown.

    2D Prediction and Trainset write to the same status bar under the Labs pages. Without this, opening
    one tool could leave the other tool's message there ("Batch results are ready in the workspace" from
    Prediction over Trainset). Showing the page puts back this page's own last message, or clears the bar
    when it has none yet. A spontaneous show (a window restored from the taskbar) changes nothing.
    """

    def __init__(self, page, status_signal, parent=None) -> None:
        super().__init__(parent)
        self.text = ""
        self._status_signal = status_signal
        status_signal.connect(self.remember)
        if page is not None:
            page.installEventFilter(self)

    @pyqtSlot(str)
    def remember(self, text: str) -> None:
        """The page said ``text`` (also from a worker thread: the slot runs on this object's thread)."""
        self.text = str(text or "")

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if event.type() == QEvent.Show and not event.spontaneous():
            self._status_signal.emit(self.text)
        return False


__all__ = ["PageStatus"]
