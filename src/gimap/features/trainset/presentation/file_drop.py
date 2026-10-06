"""A detector file dropped on a Trainset panel: the panel takes local paths and hands on the first one.

Taken here, a drop does not travel on to the main window (which would open it in Analyze): the binding loads a
file as the reference and answers a folder with a status message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt5.QtCore import QEvent, QObject
from PyQt5.QtWidgets import QWidget


def dropped_paths(mime) -> list[str]:
    """The local files and folders of a drag or a drop."""
    if mime is None or not mime.hasUrls():
        return []
    return [url.toLocalFile() for url in mime.urls() if url.isLocalFile() and Path(url.toLocalFile()).exists()]


class FileDropFilter(QObject):
    """Makes ``widget`` (and the children that do not take drops themselves) a drop target for files and folders."""

    def __init__(self, widget: QWidget, on_drop: Callable[[str], None]):
        super().__init__(widget)
        self._on_drop = on_drop
        widget.setAcceptDrops(True)
        widget.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        kind = event.type()
        if kind in (QEvent.DragEnter, QEvent.DragMove):
            if dropped_paths(event.mimeData()):
                event.acceptProposedAction()
            else:
                event.ignore()
            return True
        if kind == QEvent.Drop:
            paths = dropped_paths(event.mimeData())
            if not paths:
                event.ignore()
                return True
            event.acceptProposedAction()
            self._on_drop(paths[0])
            return True
        return False


__all__ = ["FileDropFilter", "dropped_paths"]
