"""The file list of the Data step and the file stepper of the command bar.

* **One file off the list** — right-click a file: Remove from List, Show in Folder, Copy Path; or select it and
  press Delete. The rest stays as it is: the file on screen (or, when it was the one removed, the file that took
  its row), the Series map without the removed file's frames, Batch Export with one file less. A right click
  opens the menu without showing that file. Refused while the automatic analysis keeps its frame (the list is
  disabled then) and while a batch runs (its frames were chosen when it started).
* **Delete** on the mask list and the region list removes the mask or region selected (their Remove buttons).
* **‹ 3 / 40 ›** — the previous / next buttons are chevrons in the theme's text colour, at least 24 px, with
  the position in the list beside them once more than one file is listed (Page Up / Page Down step too). A
  narrow command bar leaves them out (``views/command_bar.py``). They step through the frames of the series:
  inside a source of several frames from frame to frame, then on to the neighbouring file. One file of several
  frames has “frame 5 / 12” beside them; with several files the place in the list stays there (“3 / 40”) and the
  frame is in the text after it (“frame 5 of 12”). The frame bar above the steps says and sets the frame too
  (``bindings/data_step.py``).
* **A row** — the name of the source and, when it holds several frames, how many (“scan_00007 · 240 frames”).
  A set of module files is one row named by its scan. The tooltip says what the source consists of, names
  its files and repeats its warnings (``source_texts.py``). Files that turn out to be one source after they
  were listed become one row (``_rows_regrouped``: a master that arrived after its data files, the first
  module of a set); a row is written again whenever its source changed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, QPointF, QSignalBlocker, Qt, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QIcon, QKeySequence, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import QApplication, QListWidgetItem, QMenu, QShortcut

from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import theme_manager

from ...application.ports import source_name
from ..source_texts import row_text, row_tooltip
from .results_state import RUN_KEEPS_FRAME

DELETE_KEYS = ("Delete", "Backspace")
"""Remove the selected row of a list (Backspace: the “delete” key of a Mac keyboard)."""
BATCH_KEEPS_LIST = "A batch is running: remove files from the list when it is done."
POSITION_TIP = "File {n} of {total} in the list — Page Up / Page Down show the previous / next one"
FRAME_POSITION = "frame {i} / {n}"
"""Beside ‹ ›, for one file of several frames: the frame shown."""
FRAME_POSITION_TIP = "Frame {i} of {n} of {name} — Page Up / Page Down show the previous / next frame"
STEP_TIPS = {False: ("Previous file (Page Up)", "Next file (Page Down)"),
             True: ("Previous frame (Page Up)", "Next frame (Page Down)")}
"""The tooltips of ‹ and ›; ``True``: the file shown holds several frames."""


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

    def _frames_shown(self) -> int:
        """Frames of the file shown, as last counted (1: a single frame, or nothing shown)."""
        path = self.view_model.current_path
        return self.view_model.known_frame_count(path) if path is not None else 1

    def _show_file_position(self) -> None:
        """The previous / next buttons and the position beside them (“3 / 40” in a list of files; “frame 5 / 12”
        for one file of several frames): shown with more than one file or one file of several frames (not on a
        narrow bar), enabled where there is a frame to go to (none while the automatic analysis keeps its
        frame)."""
        count, row = self.file_list.count(), self.file_list.currentRow()
        frames, frame = self._frames_shown(), self.view_model.state.frame_index
        wanted = count > 1 or (count == 1 and frames > 1)
        compact = bool(getattr(self.command_bar, "compact", False))
        for widget in (self.previous_file_button, self.next_file_button, self.file_position_label):
            widget.setProperty("gimapWanted", wanted)
            widget.setVisible(wanted and not compact)
        free = not getattr(self, "_automatic_busy", False)
        self.previous_file_button.setEnabled(free and (row > 0 or (row == 0 and frames > 1 and frame > 0)))
        self.next_file_button.setEnabled(free and (0 <= row < count - 1 or (row >= 0 and frame < frames - 1)))
        several = row >= 0 and frames > 1
        for button, tip in zip((self.previous_file_button, self.next_file_button), STEP_TIPS[several]):
            button.setToolTip(tr(tip))
        position = f"{row + 1} / {count}" if count > 1 and row >= 0 else ""
        if several:
            if count == 1:  # (with several files the text after it says the frame: the bar has no room for both)
                position = trf(FRAME_POSITION, i=frame + 1, n=frames)
            name = source_name(self.view_model.current_path, self.view_model.describe(self.view_model.current_path))
            self.file_position_label.setToolTip(trf(FRAME_POSITION_TIP, i=frame + 1, n=frames, name=name))
        else:
            self.file_position_label.setToolTip(trf(POSITION_TIP, n=row + 1, total=count) if position else "")
        self.file_position_label.setText(position)
        if hasattr(self, "frame_listing_model"):
            self._imported_frame_position()

    def _step_file(self, delta: int) -> None:
        """The previous / next file of the list (its first frame)."""
        if self._frame_kept():  # the automatic analysis works on the frame shown
            return
        count = self.file_list.count()
        if count:
            row = min(max(0, self.file_list.currentRow() + int(delta)), count - 1)
            self.file_list.setCurrentRow(row)

    def _step_frame(self, delta: int) -> None:
        """‹ ›, Page Up / Page Down: the previous / next frame of the series. Inside a file of several frames
        that is its neighbouring frame; at its ends, and for single frames, the neighbouring file (entered
        at its first frame going on, at its last going back: also from a single frame)."""
        if hasattr(self, "frame_listing_model"):
            self.imported_frame_step(delta)
            return
        row, frames = self.file_list.currentRow(), self._frames_shown()
        frame = self.view_model.state.frame_index + int(delta)
        if row >= 0 and frames > 1 and 0 <= frame < frames:
            self.show_frame(row, frame)
        elif delta < 0 and row > 0:  # (also from a single frame: the file before may hold several)
            self.show_frame(row - 1, self.view_model.frame_count(self.view_model.state.files[row - 1]) - 1)
        else:
            self._step_file(delta)

    def show_frame(self, row: int, frame_index: int = 0) -> bool:
        """Show one frame of a listed file (0-based, kept inside the file) with one analysis. ``False``: no
        such row, or the automatic analysis keeps the frame it works on."""
        if not 0 <= int(row) < self.file_list.count() or self._frame_kept():
            return False
        with QSignalBlocker(self.file_list):
            self.file_list.setCurrentRow(int(row))
        self.view_model.select(int(row))
        path = self.view_model.current_path
        count = self.view_model.frame_count(path)
        self.view_model.set_frame(min(max(0, int(frame_index)), count - 1))
        self._file_loading(path)
        self._show_frame_count(count, self.view_model.state.frame_index)
        self.run_analysis()
        return True

    # -- the rows ------------------------------------------------------------------------

    def _append_list_items(self, paths) -> None:
        previous = self._import_applying
        self._import_applying = True
        try:
            for path in paths:
                self.file_list.addItem(QListWidgetItem())
                self._write_row(self.file_list.count() - 1, path)
        finally:
            self._import_applying = previous
        if not previous and hasattr(self, "frame_listing_model"):
            self.refresh_frame_list()

    def _write_row(self, row: int, path) -> None:
        """The text and the tooltip of a row: what the source is called, how many frames it holds, and (the
        tooltip) what it consists of."""
        item = self.file_list.item(row)
        if item is None:
            return
        layout = self.view_model.describe(path)
        item.setText(row_text(path, layout, self.view_model.known_frame_count(path)))
        item.setToolTip(row_tooltip(path, layout))

    def _refresh_rows(self, paths=None) -> None:
        """Write the rows of ``paths`` again (every row when ``None``): a source grew, was completed by
        another file, or the interface language changed."""
        wanted = None if paths is None else {str(path).casefold() for path in paths}
        for row, path in enumerate(self.view_model.state.files):
            if wanted is None or str(path).casefold() in wanted:
                self._write_row(row, path)

    def _rows_regrouped(self) -> bool:
        """After files were listed (``add_paths``, a look of the folder watch): the rows of files that turned
        out to belong together. Rows whose file is a part of another listed source leave the list
        (``view_model.dropped``: the data files of a master that arrived after them) with what was kept for
        them; a row whose source is now named by another of its files (``view_model.replaced``: a module set
        whose first module arrived later, the first data file of that master) is written again. When the row
        on screen is one of them, its source is read again from all its files (``True``)."""
        model = self.view_model
        dropped, replaced = list(model.dropped), list(model.replaced)
        if not dropped and not replaced:
            return False
        shown = self.file_list.currentRow()
        read_again = any(row == shown for row, _path in dropped)
        with QSignalBlocker(self.file_list):
            for row, _path in dropped:  # (the highest row first)
                self.file_list.takeItem(row)
            self.file_list.setCurrentRow(model.state.current_index)
        for _row, path in dropped:
            self.forget_results(path)
            if not self.batch_running():  # (a map a batch is building holds the frames the batch started with)
                self._series_drop_file(path)
        for row, before, now in replaced:
            self._write_row(row, now)
            self.forget_results(before)
            read_again = read_again or row == model.state.current_index
        if dropped:
            self.refresh_batch_entry()
            self._show_file_position()
        read_again = read_again and model.current_path is not None and not getattr(self, "_automatic_busy", False)
        if read_again:
            self._file_loading(model.current_path)
            self.run_analysis()
        for _row, path in dropped:
            self.fileRemoved.emit(path)
        return read_again

    def _rows_listed_again(self, rows) -> None:
        """Files that are listed were opened again: their rows as the sources are now (one may have grown, or
        been completed by another file, since it was listed), and the frame on screen read again when its
        files changed."""
        files = self.view_model.state.files
        for row in rows:
            if 0 <= row < len(files):
                self._write_row(row, files[row])
        if not rows:
            return
        self.refresh_batch_entry()
        shown = self.view_model.state.current_index in rows and not getattr(self, "_automatic_busy", False)
        if shown and self.view_model.shown_changed():
            self.run_analysis()

    def _file_list_language(self) -> None:
        """After a switch of the interface language: the rows (“· 12 frames”), the stepper's position, and
        what the folder watch said last when the status line still says it (``bindings/batch_watch.py``)."""
        self._refresh_rows()
        self._show_file_position()
        self._watch_language()
        if hasattr(self, "frame_listing_model"):
            self._frame_import_language()

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
        name = source_name(path, self.view_model.describe(path))
        menu.addAction(name).setEnabled(False)  # which file (a name: never translated)
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
