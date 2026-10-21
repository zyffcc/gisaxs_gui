"""Series read from curve files or dropped on the Compare page, and Remove All with an Undo.

Add ▸ Folder of Curves… takes a folder that is one series: one kind of curve, a file per frame. A folder
Analyze wrote — a Batch Export, its ``curves`` folder, the exports of single frames — holds several kinds
(I(q), I(χ), every cut region). They are never read together: the page asks which one is the series
(``choose_kind``, the last choice first), reads that kind alone and names the series after it; folders dropped
together are asked about once.

Reading the curve files of a series (``_read``: Add ▸ Folder of Curves… / Curve Files…, and a drop) says how far
it is: a thousand files take seconds, and the window stays drawn meanwhile. The comparison waits until the files
are in, so several folders dropped together are compared once.

Drop a folder of curve files (each folder: one series), or curve files (together: one series) anywhere on the
page — the same as Add ▸ Folder of Curves… / Curve Files…. Anything else dropped (a project, an image, a file
of another kind) is left out with a warning that says what Compare takes. Remove All empties the page at once
(no question first); its toast offers Undo, which puts the removed series back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from PyQt5 import sip
from PyQt5.QtCore import QEventLoop, Qt, QTimer
from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog

from src.gimap.app.presentation.components import show_toast
from src.gimap.app.presentation.i18n import tr, trf

from ..application import CURVE_SUFFIXES, Progress, SeriesData
from ..application.sources import FIT_INPUT_KIND, OTHER_KIND
from .errors import error_text

UNDO_MS = 10000
"""How long the toast of Remove All (with its Undo) stays."""
SHOWN_NAMES = 3
"""File names listed in the warning about dropped files that were left out."""
READING = "Reading curve files: {done} of {total} …"
"""The status line while the curve files of a series are read."""
KIND_TITLE = "Curves of the Folder"
KIND_QUESTION = "{folder} holds {n} kinds of curves. One kind is one series — which one?"
KIND_ITEM = "{key} — {title} · {n} frames"
KIND_ITEM_PLAIN = "{key} · {n} frames"
"""The question about a folder with several kinds of curves, and one of its answers (``choose_kind``)."""
FIT_INPUT_TITLE = "the curve for Fitting ({title})"
FIT_INPUT_TITLE_PLAIN = "the curve for Fitting"
OTHER_TITLE = "other files"
"""What the two kinds that are no curve of Analyze's are (``kind_title``); ``{title}``: the curve's own title."""
KIND_PREFERENCE = "kind"
FIRST_CHOICES = ("radial", "horizontal")
"""The curve offered first when none was chosen before: I(q) of a GIWAXS frame, the horizontal cut of a GISAXS one."""


class _ReadAbandoned(Exception):
    """The window was closed while curve files were read: the reading stops and nothing is added."""


def same_series(a: SeriesData, b: SeriesData) -> bool:
    """The same frames from the same place (a map sent twice, a folder added twice), whatever their names."""
    return (a.source == b.source and tuple(a.labels) == tuple(b.labels) and a.x.shape == b.x.shape
            and a.image.shape == b.image.shape and np.array_equal(a.x, b.x, equal_nan=True)
            and np.array_equal(a.image, b.image, equal_nan=True))


def dropped_paths(mime) -> list[Path]:
    """The local files and folders of a drag (``QMimeData``)."""
    if mime is None or not mime.hasUrls():
        return []
    return [Path(url.toLocalFile()) for url in mime.urls() if url.isLocalFile() and url.toLocalFile()]


def is_curve_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in CURVE_SUFFIXES and not path.name.startswith(".")


def _for_compare(paths: Sequence[Path]) -> bool:
    """A drag Compare takes: at least one folder or curve file."""
    return any(path.is_dir() or is_curve_file(path) for path in paths)


def kind_title(kind) -> str:
    """What a kind of curve is, in words: its title as its files or the batch's README say it (data, as it
    is) — and for the curves written for Fitting and the files that are none of Analyze's, the page's own
    words in the interface language."""
    if kind.key == FIT_INPUT_KIND:
        return trf(FIT_INPUT_TITLE, title=kind.title) if kind.title else tr(FIT_INPUT_TITLE_PLAIN)
    if kind.key == OTHER_KIND:
        return tr(OTHER_TITLE)
    return kind.title


def kind_text(kind) -> str:
    """One kind of curve as the question lists it: its key, what it is, how many frames. The key and the
    title are Analyze's own names (data), the rest is in the interface language."""
    title = kind_title(kind)
    if title:
        return trf(KIND_ITEM, key=kind.key, title=title, n=kind.frames)
    return trf(KIND_ITEM_PLAIN, key=kind.key, n=kind.frames)


def choose_kind(parent, folder: str, kinds: Sequence, last: Optional[str] = None) -> Optional[str]:
    """Ask which of the ``kinds`` of curves in ``folder`` is the series; the key chosen, ``None`` when the
    question is cancelled. Offered first: the kind chosen last (``last``), else I(q) or the horizontal cut."""
    keys = [kind.key for kind in kinds]
    first = next((key for key in (last, *FIRST_CHOICES) if key in keys), keys[0])
    items = [kind_text(kind) for kind in kinds]
    answer, accepted = QInputDialog.getItem(
        parent, tr(KIND_TITLE), trf(KIND_QUESTION, folder=folder, n=len(kinds)), items, keys.index(first), False)
    return keys[items.index(answer)] if accepted and answer in items else None


class CompareInputMixin:
    """Needs ``series``, ``service``, ``_q_range``, ``add_series``, ``_unique``, ``_emptied``, ``_status``,
    ``_remember``, ``_remembered``, ``refresh`` and ``_schedule`` (the ``ComparePage``); the page calls
    ``setAcceptDrops(True)`` and lets its comparison wait while ``_reading``."""

    # -- adding ----------------------------------------------------------------------------------

    _kind_asked: Optional[str] = None
    """The kind chosen at the question of this drop (folders dropped together are asked about once)."""
    _kind_last: Optional[str] = None
    """The kind chosen last, offered first at the next question (kept with the preferences when there are any)."""

    def add_folder(self, folder=None, kind: Optional[str] = None, *, prefer: Optional[str] = None) -> Optional[SeriesData]:
        """Add ▸ Folder of Curves…, or a folder given: its one kind of curve as a series. With several kinds
        (a folder Analyze wrote) the kind ``kind``; else ``prefer`` when the folder holds it; else the one the
        person chooses — nothing is added when the question is cancelled."""
        folder = folder or QFileDialog.getExistingDirectory(self, tr("Folder of Curve Files"), self._remembered("folder") or "")
        if not folder:
            return None
        self._remember(folder=str(folder))
        if kind is None:
            kind = self._kind_of(Path(folder), prefer)
            if kind is False:
                return None
        data = self._read(lambda progress: self.service.series_from_folder(Path(folder), kind=kind or None, progress=progress))
        return self.add_series(data) if data is not None else None

    def _kind_of(self, folder: Path, prefer: Optional[str] = None):
        """Which kind of curve of ``folder`` is the series: ``None`` when it holds one kind (or cannot be
        looked into: reading it says why), ``prefer`` when it holds that one, else the person's choice —
        ``False`` when the question is cancelled."""
        try:
            kinds = self.service.folder_kinds(folder)
        except (OSError, ValueError):
            return None
        if len(kinds) < 2:
            return None
        if prefer and prefer in [kind.key for kind in kinds]:
            return prefer
        chosen = choose_kind(self, folder.name, kinds, self._kind_last or self._remembered(KIND_PREFERENCE))
        if chosen is None:
            self._status(lambda name=folder.name: trf("Nothing added: no curve of {folder} was chosen.", folder=name))
            return False
        self._kind_asked = self._kind_last = chosen
        self._remember(**{KIND_PREFERENCE: chosen})
        return chosen

    def add_files(self, paths=None) -> Optional[SeriesData]:
        if paths is None:
            filters = tr("Curve files") + " (" + " ".join(f"*{suffix}" for suffix in CURVE_SUFFIXES) + ")"
            paths, _ = QFileDialog.getOpenFileNames(self, tr("Curve Files of One Series"), self._remembered("folder") or "", filters)
        if not paths:
            return None
        self._remember(folder=str(Path(paths[0]).parent))
        data = self._read(lambda progress: self.service.series_from_files([Path(path) for path in paths], progress=progress))
        return self.add_series(data) if data is not None else None

    # -- reading ---------------------------------------------------------------------------------

    _read_stopped = False
    """The window was closed while a series was read: the series still waiting (several folders dropped) are not
    read either, as long as it stays closed."""
    _reading = 0
    """Readings in progress. The comparison waits for them (``ComparePage.recompute``): its timer runs with the
    events of every count, and the folders of one drop are compared once, with one notice — not after each."""

    def _read(self, read: Callable[[Progress], SeriesData]) -> Optional[SeriesData]:
        """The series ``read(progress)`` makes from curve files; ``None``, with the error said, when they do not
        make one. The files are read on the interface thread, a thousand in seconds (longer the first time): the
        cursor waits, the status line counts the files, and at every count the window is drawn again. Clicks and
        keys wait until the reading has ended, so nothing else starts in between; a window closed meanwhile ends
        it, and nothing is added (no count is left on the status line either)."""
        if sip.isdeleted(self):  # gone with its window while an earlier folder of the same drop was read
            return None
        shown = self.window().isVisible()
        if self._read_stopped and not shown:
            return None
        self._read_stopped = False

        def progress(done: int, total: int) -> None:
            self._status(lambda: trf(READING, done=done, total=total))
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
            if sip.isdeleted(self) or (shown and not self.window().isVisible()):
                self._read_stopped = True
                raise _ReadAbandoned

        self._reading += 1
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            return read(progress)
        except _ReadAbandoned:
            if not sip.isdeleted(self):
                self._status("")  # “Reading … 50 of 300” would still stand there if the window is shown again
            return None
        except (ValueError, OSError) as exc:
            error = str(exc)
        finally:
            self._reading -= 1
            QApplication.restoreOverrideCursor()
        self._status(lambda: error_text(error), "error")
        return None

    # -- dropping --------------------------------------------------------------------------------

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        # Only a drag with a folder or a curve file is Compare's; anything else (a project, detector images)
        # is left to the window, which opens it as anywhere else. What does not fit in a mixed drag is said
        # at the drop, not with a “no” cursor.
        if _for_compare(dropped_paths(event.mimeData())):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if _for_compare(dropped_paths(event.mimeData())):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        paths = dropped_paths(event.mimeData())
        if not paths:
            event.ignore()
            return
        event.acceptProposedAction()
        # Read once the drop has returned: the file manager the files came from is not held while a big
        # folder is read (a thousand curve files: a second or more).
        QTimer.singleShot(0, lambda: None if sip.isdeleted(self) else self.add_dropped(paths))

    def add_dropped(self, paths: Sequence) -> list[SeriesData]:
        """Each folder as one series (``add_folder``), the curve files together as one (``add_files``); other
        paths are left out with a warning. Returns the series added."""
        paths = [Path(path) for path in paths]
        folders = [path for path in paths if path.is_dir()]
        curves = [path for path in paths if is_curve_file(path)]
        others = [path for path in paths if path not in folders and path not in curves]
        self._kind_asked = None  # folders of one drop: asked once, the others take the same kind when they hold it
        added = [self.add_folder(folder, prefer=self._kind_asked) for folder in folders]
        if curves:
            added.append(self.add_files([str(path) for path in curves]))
        if others and not sip.isdeleted(self):  # the page may have gone with its window while the files were read
            names = ", ".join(path.name or str(path) for path in others[:SHOWN_NAMES])
            names += " …" if len(others) > SHOWN_NAMES else ""
            self._status(lambda names=names: trf(
                "Not added: {names}. Compare takes a folder or curve files ({suffixes}).",
                names=names, suffixes=", ".join(CURVE_SUFFIXES)), "warning")
        return [item for item in added if item is not None]

    # -- Remove All, and its Undo ------------------------------------------------------------------

    def clear_all(self) -> None:
        """Remove All: every series out at once; the toast's Undo puts them back."""
        removed, q_range = list(self.series), self._q_range
        self.series.clear()
        self._emptied()  # the empty state; the status line stays empty (the toast is the notice)
        if removed and self.window().isVisible():
            show_toast(self.window(), trf("Removed {n} series.", n=len(removed)), level="info",
                       action=(tr("Undo"), lambda: self.undo_remove_all(removed, q_range)), timeout_ms=UNDO_MS)

    def undo_remove_all(self, removed: Sequence[SeriesData], q_range: Optional[tuple] = None) -> list[SeriesData]:
        """The series of a Remove All back, after any added since (one already here again is not doubled);
        the compared range of before when the page was still empty. Returns the series put back."""
        back = [item for item in removed if not any(same_series(item, kept) for kept in self.series)]
        if not back:
            return []
        if not self.series:
            self._q_range = q_range
        for item in back:
            self.series.append(item.renamed(self._unique(item.name)))
        self._announce = True  # one toast when they are compared again
        self._status(lambda count=len(back): trf("{n} series put back.", n=count))
        self.refresh()
        self._schedule()
        return back


__all__ = ["CompareInputMixin", "READING", "UNDO_MS", "dropped_paths", "is_curve_file", "kind_text", "kind_title",
           "same_series"]
