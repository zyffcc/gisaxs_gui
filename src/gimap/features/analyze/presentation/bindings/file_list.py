"""The file list of the Data step and the file stepper of the command bar.

* **One file off the list** — right-click a file: Remove from List, Show in Folder, Copy Path; or select it and
  press Delete. The rest stays as it is: the file on screen (or, when it was the one removed, the file that took
  its row), the Series map without the removed file's frames, Batch Export with one file less. A right click
  opens the menu without showing that file. Refused while the automatic analysis keeps its frame (the list is
  disabled then) and while a batch runs (its frames were chosen when it started).
* **Delete** on the mask list and the region list removes the mask or region selected (their Remove buttons).
* **‹ 3 / 40 ›** — the previous / next file buttons are chevrons in the theme's text colour, at least 24 px, with
  the position in the list beside them once more than one file is listed (Page Up / Page Down step too). A
  narrow command bar leaves them out (``views/command_bar.py``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, QPointF, QSignalBlocker, Qt, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QMenu, QShortcut

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import theme_manager

from .results_state import RUN_KEEPS_FRAME

DELETE_KEYS = ("Delete", "Backspace")
"""Remove the selected row of a list (Backspace: the “delete” key of a Mac keyboard)."""
BATCH_KEEPS_LIST = "A batch is running: remove files from the list when it is done."
POSITION_TIP = "File {n} of {total} in the list — Page Up / Page Down show the previous / next one"


def chevron_icon(direction: str, color) -> QIcon:
    """A chevron pointing ``left`` or ``right``, drawn in ``color`` with the line of the undo / redo icons."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color))
    pen.setWidthF(3.4)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    tip, back = (11.0, 21.0) if direction == "left" else (21.0, 11.0)
    path = QPainterPath(QPointF(back, 6.0))
    path.lineTo(QPointF(tip, 16.0))
    path.lineTo(QPointF(back, 26.0))
    painter.drawPath(path)
    painter.end()
    return QIcon(pixmap)


class _RightPressKeepsRow(QObject):
    """A right press on the file list does not make that file current (which would analyse it): the menu acts
    on the row under the pointer instead."""

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if event.type() in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick) and event.button() == Qt.RightButton:
            return True
        return False


class FileListMixin:
    """Needs the Data-step widgets, the command bar's file widgets, ``view_model``, ``tasks``, ``_status`` and
    the page's other mixins (results, series, batch entry)."""

    def _connect_file_list(self) -> None:
        listing = self.file_list
        listing.customContextMenuRequested.connect(self._file_menu_requested)
        self._right_press = _RightPressKeepsRow(listing)
        listing.viewport().installEventFilter(self._right_press)
        for keys in DELETE_KEYS:
            for widget, slot in ((listing, lambda: self.remove_file()), (self.mask_list, self._delete_mask_key),
                                 (self.region_list, self._delete_region_key)):
                shortcut = QShortcut(QKeySequence(keys), widget)
                shortcut.setContext(Qt.WidgetShortcut)  # only while that list has the focus
                shortcut.activated.connect(slot)
        model = listing.model()
        for signal in (model.rowsInserted, model.rowsRemoved, model.modelReset):
            signal.connect(self._list_changed)
        theme_manager().changed.connect(self._paint_file_icons)
        self._paint_file_icons()
        self._show_file_position()

    def _dispose_file_list(self) -> None:
        try:
            theme_manager().changed.disconnect(self._paint_file_icons)
        except (TypeError, RuntimeError):
            pass

    # -- the stepper ---------------------------------------------------------------------

    def _list_changed(self, *_args) -> None:
        try:
            self._show_file_position()
        except RuntimeError:  # the list is being destroyed with the page
            pass

    def _paint_file_icons(self, *_args) -> None:
        color = theme_manager().color("text")
        self.previous_file_button.setIcon(chevron_icon("left", color))
        self.next_file_button.setIcon(chevron_icon("right", color))

    def _show_file_position(self) -> None:
        """The previous / next buttons and “3 / 40”: shown with more than one file (not on a narrow bar),
        enabled where there is a file to go to (none while the automatic analysis keeps its frame)."""
        count, row = self.file_list.count(), self.file_list.currentRow()
        wanted = count > 1
        compact = bool(getattr(self.command_bar, "compact", False))
        for widget in (self.previous_file_button, self.next_file_button, self.file_position_label):
            widget.setProperty("gimapWanted", wanted)
            widget.setVisible(wanted and not compact)
        free = not getattr(self, "_automatic_busy", False)
        self.previous_file_button.setEnabled(free and row > 0)
        self.next_file_button.setEnabled(free and 0 <= row < count - 1)
        position = f"{row + 1} / {count}" if wanted and row >= 0 else ""
        self.file_position_label.setText(position)
        self.file_position_label.setToolTip(trf(POSITION_TIP, n=row + 1, total=count) if position else "")

    # -- the menu ------------------------------------------------------------------------

    def _file_menu_requested(self, position) -> None:
        index = self.file_list.indexAt(position)
        if not index.isValid():
            return
        menu = self._file_menu(index.row())
        menu.aboutToHide.connect(menu.deleteLater)  # after its action ran (deleted on the next event loop turn)
        menu.popup(self.file_list.viewport().mapToGlobal(position))

    def _file_menu(self, row: int) -> QMenu:
        """The menu of one listed file (its name, greyed, on top): Remove from List, Show in Folder,
        Copy Path. Remove is disabled while the list cannot change (the automatic analysis, a batch)."""
        path = self.view_model.state.files[row]
        menu = QMenu(self.file_list)
        menu.addAction(path.name).setEnabled(False)  # which file (a name: never translated)
        menu.addSeparator()
        remove = menu.addAction(tr("Remove from List"), lambda: self.remove_file(row))
        remove.setShortcut(QKeySequence(DELETE_KEYS[0]))
        remove.setShortcutVisibleInContextMenu(True)
        remove.setEnabled(not self._list_kept())
        menu.addSeparator()
        menu.addAction(tr("Show in Folder"), lambda: self.show_file_in_folder(row))
        menu.addAction(tr("Copy Path"), lambda: self.copy_file_path(row))
        return menu

    def _list_kept(self) -> Optional[str]:
        """Why no file can leave the list now (English, a key of the table), or ``None``."""
        if getattr(self, "_automatic_busy", False):
            return RUN_KEEPS_FRAME
        if self.batch_running():
            return BATCH_KEEPS_LIST
        return None

    def show_file_in_folder(self, row: int) -> bool:
        """Open the folder of a listed file in the file manager."""
        files = self.view_model.state.files
        if not 0 <= row < len(files):
            return False
        return QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(files[row]).parent)))

    def copy_file_path(self, row: int) -> Optional[str]:
        """Put the full path of a listed file on the clipboard; returns it."""
        files = self.view_model.state.files
        if not 0 <= row < len(files):
            return None
        text = str(Path(files[row]))
        QApplication.clipboard().setText(text)
        self._status(trf("Copied the path of {name}", name=Path(text).name), "ok")
        return text

    # -- removing one file -----------------------------------------------------------------

    def remove_file(self, row: Optional[int] = None) -> bool:
        """Take one file off the list (``row``; by default the one selected). The file on screen stays shown;
        when it is the one removed, the file that took its row (else the one before) is shown. Its frames leave
        the Series map, its results are forgotten, Batch Export counts one file less. ``False``: no such row, or
        the list cannot change now (the status line says why)."""
        row = self.file_list.currentRow() if row is None else int(row)
        files = self.view_model.state.files
        if not 0 <= row < len(files):
            return False
        kept = self._list_kept()
        if kept is not None:
            self._status(tr(kept), "warning")
            return False
        path = Path(files[row])
        # Summed with the next listed files (single-frame files): every later group of the map changes.
        summed_across = self.view_model.sum_count > 1 and self.view_model.frame_count(path) == 1
        if len(files) == 1:
            self.clear_files()  # the last one: as Clear (nothing on screen, the automatic analysis forgets its reports)
        else:
            current = row == self.view_model.state.current_index
            self.view_model.remove_file(row)
            if current:
                self.tasks.cancel("analyze")  # an analysis of the removed file under way is not shown
                self._shown_status = None
            with QSignalBlocker(self.file_list):
                self.file_list.takeItem(row)
                self.file_list.setCurrentRow(self.view_model.state.current_index)
            self.forget_results(path)
            self._series_drop_file(path, summed_across=summed_across)
            self.refresh_batch_entry()
            self._show_file_position()
            if current:
                shown = self.view_model.current_path
                if shown is not None:
                    self._file_loading(shown)
                self.run_analysis()
            else:
                analysis = self.view_model.state.analysis
                if analysis is not None:
                    self._show_data_info(analysis, self.file_list.count())  # “Files listed: n”
                self._refresh_series_curves(analysis)  # the controls and the sentence of the Series tab
        self._status(trf("Removed {name} from the list", name=path.name), "ok")
        self.fileRemoved.emit(path)
        return True

    # -- Delete on the mask and region lists -------------------------------------------------

    def _delete_mask_key(self) -> None:
        if self.mask_list.currentRow() >= 0:  # never the last mask by default, as the button does
            self._remove_mask()

    def _delete_region_key(self) -> None:
        row = self._selected_row()
        if row is not None and row.editor == "generic":  # a region added, or the custom sector
            self._remove_region()


__all__ = ["BATCH_KEEPS_LIST", "DELETE_KEYS", "FileListMixin", "POSITION_TIP", "chevron_icon"]
