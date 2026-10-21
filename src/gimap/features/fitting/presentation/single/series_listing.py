"""The curves of the In-situ series: choosing the folder, listing what matches, the frames to fit.

A folder is opened with Choose Folder…, by dropping it, by Analyze ▸ Send Series to Fitting or by a
project. Its curve files that match the pattern are listed in natural order; when nothing matches, the
``fit_input`` folder Analyze writes below it is taken (the person may have chosen ``gimap_analysis``), and when
there are no curves at all but detector frames, the frames are listed and each is cut as in Analyze
(``series_frames``; a pattern such as ``*.cbf`` asks for them directly). The
frames to fit are first … last, every n-th. Fits that were not saved are asked about before another list
takes their place; listing the same curves again (the pattern field merely lost the focus) changes nothing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtWidgets import QFileDialog, QListWidgetItem

from src.gimap.app.presentation.i18n import tr, trf

from ...application import DiscoverInSituFramesRequest
from ...application.series_fit import file_name
from ..views.fit_series_view import NO_FOLDER
from .series_frames import FRAMES_LISTED
from .series_text import FIT_INPUT_FOLDER, series_name, short_names

CURVES_HINT = ("A folder of curves, e.g. the gimap_analysis/fit_input folder of Analyze's Batch Export or Send "
               "Series to Fitting, or a folder of detector frames (CBF, TIFF, EDF, NeXus): each frame is then cut "
               "as Analyze cuts it.")


class FitSeriesListingMixin:
    """Needs the series page: ``view_model``, ``running``, ``fits``, the widgets of the Curves step,
    ``_may_drop_results()``, ``_forget_halves()``, ``_probe_signed()``, ``_forget_stages()``, ``_mark_stages()``,
    ``refresh()``."""

    def choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, tr("Folder of the Curves"), self.folder or str(self._remembered("folder") or ""))
        if folder:
            self.open_series(folder)

    def open_series(self, folder, pattern: Optional[str] = None) -> bool:
        """List the curves of ``folder`` (Choose Folder…, a dropped folder, Analyze ▸ Send Series to Fitting)."""
        return self._open(folder, pattern, ask=True)

    def _open(self, folder, pattern: Optional[str] = None, *, ask: bool) -> bool:
        """``ask``: results that were not saved are asked about first (not when a project opens)."""
        if self.running:
            self._status(tr("Stop the running series first."), "warning")
            return False
        if ask and not self._may_drop_results():
            return False
        self.folder = str(folder)
        if pattern:
            self.pattern_edit.setText(pattern)
        self._forget_halves()  # a series opened anew: what was chosen for the one before is not its choice
        self._show_folder()
        return self._list_curves(again=True)

    def _folder_name(self) -> str:
        """What the series is called where the whole path has no room (``series_text.series_name``): in the
        line about the curves, the status line and the title of a saved trend. The whole path is on the chip."""
        return series_name(self.folder)

    def _show_folder(self) -> None:
        parts = Path(self.folder).parts
        self.folder_chip.setText(self.folder if len(parts) <= 4 else str(Path("…", *parts[-3:])))
        self.folder_chip.setToolTip(self.folder)
        self._remember(folder=self.folder, pattern=self.pattern_edit.text())

    def _discover(self, folder=None) -> list[str]:
        if self.frames_mode():  # detector frames, each cut as in Analyze (``series_frames``)
            return self.discover_frames(folder or self.folder)
        frames = self.view_model.storage.discover_insitu_frames(DiscoverInSituFramesRequest(
            Path(folder or self.folder), self.pattern_edit.text().strip() or "*_fit_input.dat", self.subfolders_check.isChecked()))
        return [str(frame.path) for frame in frames]

    def _curves_below(self) -> tuple:
        """Nothing matches in the chosen folder: ``(folder, curves)`` of its ``fit_input`` folder — one or
        two levels down — when exactly one of those holds curves, else ``(None, [])``; ``_below`` names the
        places that hold some (for the status line)."""
        root = Path(self.folder)
        try:
            children = sorted(child for child in root.iterdir() if child.is_dir())
        except OSError:  # a folder that cannot be looked into: only its own fit_input
            children = []
        places = [root / FIT_INPUT_FOLDER] + [child / FIT_INPUT_FOLDER for child in children]
        found = [(place, curves) for place in places if place.is_dir() for curves in [self._discover(place)] if curves]
        self._below = ", ".join(place.relative_to(root).as_posix() for place, _curves in found)
        return found[0] if len(found) == 1 else (None, [])

    def _list_curves(self, *, again: bool = False) -> bool:
        """The curves of the folder that match the pattern. ``again``: a folder opened anew — listed afresh
        also when the same files are there (they may have been written again)."""
        if not self.folder or self.running or self._asking:
            return False
        self._below, below = "", None
        try:
            paths = self._discover()
            if not paths and not self.subfolders_check.isChecked() and not self.frames_mode():
                below, paths = self._curves_below()
            if not paths and not self.frames_mode() and (pattern := self.frame_pattern_in(self.folder)):
                self.pattern_edit.setText(pattern)  # no curves, but detector frames: cut each as in Analyze
                paths = self._discover()
        except (OSError, ValueError, RuntimeError) as exc:
            self._status(lambda reason=str(exc): tr("Could not list the curves: {reason}").format(reason=reason), "error")
            return False
        if not again and paths == self.paths:  # the same curves (the field only lost the focus): nothing changes
            self._listing = (self.pattern_edit.text(), self.subfolders_check.isChecked())
            return bool(self.paths)
        if not again and not self._may_drop_results():  # the fits stay, and with them the curves as they are listed
            self.pattern_edit.setText(self._listing[0])
            self.subfolders_check.blockSignals(True)
            self.subfolders_check.setChecked(self._listing[1])
            self.subfolders_check.blockSignals(False)
            return bool(self.paths)
        if below is not None:  # the folder Analyze wrote the curves into becomes the folder of the series
            self.folder = str(below)
            self._show_folder()
        if paths != self.paths:
            self._forget_stages()  # those of the curves listed before (Start would leave out their odd frames)
        self.paths, self._names = paths, None
        self.fits, self._curves, self._selected, self._start_model = {}, {}, None, None
        self._preview_curves, self._unsaved, self._untouched_start = {}, False, False
        self._listing = (self.pattern_edit.text(), self.subfolders_check.isChecked())
        self._forget_halves(choice=False)  # nothing is known of these curves yet
        for spin in (self.first_spin, self.last_spin, self.every_spin):
            spin.blockSignals(True)
            spin.setRange(1, max(1, len(self.paths)))
        self.first_spin.setValue(1)
        self.last_spin.setValue(max(1, len(self.paths)))
        for spin in (self.first_spin, self.last_spin, self.every_spin):
            spin.blockSignals(False)
        self._frames_changed()
        self._select_first()
        self._probe_signed()  # the first frame may have lost a half, or be no curve: a look at a few more
        if not self.paths:
            self._clear_frame()  # not the first frame of the curves listed before
            if self._below:
                self._status(lambda places=self._below: trf(
                    "No curves here, but in {places}: choose one of them, or tick “Also in subfolders”.", places=places), "warning")
            else:
                self._status(lambda pattern=self.pattern_edit.text(), folder=self.folder: tr(
                    "No curves matching {pattern} in {folder}.").format(pattern=pattern, folder=folder), "warning")
        elif self._below:
            self._status(lambda count=len(self.paths), place=self._below: trf(
                "{count} curves listed, in {place} of the chosen folder.", count=count, place=place))
        elif self.frames_mode():
            self._status(lambda count=len(self.paths): trf(FRAMES_LISTED, count=count))
        else:
            self._status(lambda count=len(self.paths): tr("{count} curves listed.").format(count=count))
        self.refresh()  # the halves these curves were sent with (read with the first frame)
        return bool(self.paths)

    def _listed(self) -> list[int]:
        """The frames to fit (indices into ``paths``): first to last, every n-th."""
        if not self.paths:
            return []
        first, last = self.first_spin.value() - 1, min(self.last_spin.value(), len(self.paths))
        return list(range(first, last, max(1, self.every_spin.value())))

    def _frames_changed(self) -> None:
        self.frame_list.blockSignals(True)
        self.frame_list.clear()
        for _index in self._listed():
            self.frame_list.addItem(QListWidgetItem())
        self._mark_stages()  # every line: its text, stage and colour
        self._keep_selection()
        self.frame_list.blockSignals(False)
        self.refresh()

    def _short_name(self, index: int) -> str:
        """The frame's file without what every listed name shares (``series_text.short_names``)."""
        if self._names is None or len(self._names) != len(self.paths):
            self._names = short_names([file_name(path, self.folder) for path in self.paths])
        return self._names[index]

    def close_series(self) -> None:
        """No folder (a project without a series opens): the curves, fits, stages and halves of before go,
        and the page is as it starts."""
        self.folder, self.paths, self.fits, self.queue, self._names = "", [], {}, [], None
        self._curves, self._preview_curves, self._selected, self._start_model = {}, {}, None, None
        self._previous = self._last_done = self._drawn_with = None
        self._unsaved = self._untouched_start = False
        self._forget_halves()
        self.folder_chip.setText(tr(NO_FOLDER))
        self.folder_chip.setToolTip("")
        for spin in (self.first_spin, self.last_spin, self.every_spin):
            spin.blockSignals(True)
            spin.setRange(1, 1)
            spin.setValue(1)
            spin.blockSignals(False)
        self._clear_frame()
        self._status_text = None  # the line about the series of before is not said again
        self.status_label.setText("")
        self._forget_stages()
        self._frames_changed()  # the empty list, the steps, the empty table and trend
        self.show_step("curves")


__all__ = ["CURVES_HINT", "FIT_INPUT_FOLDER", "FitSeriesListingMixin"]
