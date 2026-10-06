"""Files dropped on the Fitting pages, and the width of their steps panel from one session to the next.

* Single analysis: a curve file (.dat, .txt: those the curve reader takes) dropped on the page opens it,
  as Open Curve… does.
* In-situ series: a folder dropped on the page lists its curves, as Choose Folder… does.

Anything else is refused while it is dragged (the cursor says so); a project (.gimap) is left to the main
window. The splitter between the steps and the plots is remembered per page (``_remember(splitter=…)``)
a moment after it was moved, and set again when the page is made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QTimer

from src.gimap.app.presentation.i18n import tr

from ...application import CURVE_SUFFIXES

SPLITTER_DELAY_MS = 400
"""How long after the last move of the splitter its widths are kept (one write per drag)."""


def dropped_paths(mime) -> list[Path]:
    """The local files and folders of a drag."""
    if mime is None or not mime.hasUrls():
        return []
    return [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile() and url.toLocalFile()]


def dropped_curves(mime) -> list[Path]:
    """The curve files of a drag (.dat, .txt)."""
    return [path for path in dropped_paths(mime) if path.suffix.lower() in CURVE_SUFFIXES and path.is_file()]


def dropped_folder(mime) -> Optional[Path]:
    """The first folder of a drag."""
    return next((path for path in dropped_paths(mime) if path.is_dir()), None)


class CurveDropMixin:
    """Single analysis: a dropped curve file opens. Needs ``open_curve`` and ``_status``; before ``QWidget``
    in the bases (its drag handlers replace the widget's)."""

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if dropped_curves(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        curves = dropped_curves(event.mimeData())
        if not curves:
            event.ignore()
            return
        event.acceptProposedAction()
        if self.open_curve(str(curves[0])) and len(curves) > 1:  # as Open Curve…: a path, no side
            name = curves[0].name
            self._status(lambda: tr("One curve at a time here: {name} is open. Fit ▸ Advanced ▸ Fit Many Curves… "
                                    "takes several.").format(name=name), "warning")


class FolderDropMixin:
    """In-situ series: a dropped folder is listed as the series. Needs ``open_series``; before ``QWidget``."""

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if dropped_folder(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        folder = dropped_folder(event.mimeData())
        if folder is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.open_series(folder)  # says itself when a series is running or nothing matches


class SplitterMemoryMixin:
    """Needs ``splitter``, ``_remember`` and ``_remembered``."""

    def _keep_splitter(self) -> None:
        """The widths of last time (the steps panel keeps its width when the window is resized), then
        remember them whenever the splitter is moved."""
        sizes = self._remembered("splitter")
        if isinstance(sizes, (list, tuple)) and len(sizes) == self.splitter.count() and all(
                isinstance(size, (int, float)) and size > 0 for size in sizes):
            self.splitter.setSizes([int(size) for size in sizes])
        if getattr(self, "_splitter_timer", None) is not None:  # restored again: connected already
            return
        self._splitter_timer = QTimer(self)
        self._splitter_timer.setSingleShot(True)
        self._splitter_timer.setInterval(SPLITTER_DELAY_MS)
        self._splitter_timer.timeout.connect(self._remember_splitter)
        self.splitter.splitterMoved.connect(lambda *_args: self._splitter_timer.start())

    def _remember_splitter(self) -> None:
        sizes = [int(size) for size in self.splitter.sizes()]
        if sizes and all(size > 0 for size in sizes):
            self._remember(splitter=sizes)


__all__ = ["CURVE_SUFFIXES", "CurveDropMixin", "FolderDropMixin", "SplitterMemoryMixin", "dropped_curves",
           "dropped_folder", "dropped_paths"]
