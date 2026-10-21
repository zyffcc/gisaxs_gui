"""Responsive metadata import and the uniform, virtual frame list in Data."""
from pathlib import Path
from threading import Event
from PyQt5.QtCore import QObject, QSignalBlocker, Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QDialog, QShortcut
from src.gimap.app.presentation.i18n import tr, trf
from ...application.frame_import import ImportSource, prepare_frame_import
from ...application.ports import source_name
from ..frame_import_dialog import FrameImportDialog
from ..frame_listing import FrameListingModel
from ..frame_selection import FrameSelection
from ..source_texts import row_tooltip
from ..texts import message_text
from .file_list import DELETE_KEYS


class _ImportSignals(QObject):
    progress = pyqtSignal(object, int, int, str)


class FrameImportMixin:
    _import_applying = _importing = False
    def _connect_frame_import(self):
        self._frame_import_disposed = False
        self._importing = self._import_applying = False
        self._import_progress = None
        self._import_cancelled = Event()
        self._import_signals = _ImportSignals(self)
        self._import_signals.progress.connect(self._frame_import_progress)
        self.frame_listing_model = FrameListingModel(self.frame_list)
        self.frame_list.setModel(self.frame_listing_model)
        self.frame_list.selectionModel().currentChanged.connect(self._frame_row_selected)
        self.frame_list.customContextMenuRequested.connect(self._frame_list_menu)
        self.frame_list.viewport().installEventFilter(self._right_press)
        self.choose_frames_button.clicked.connect(self.choose_imported_frames)
        self.import_cancel_button.clicked.connect(self.cancel_frame_import)
        for keys in DELETE_KEYS:
            shortcut = QShortcut(QKeySequence(keys), self.frame_list)
            shortcut.setContext(Qt.WidgetShortcut)
            shortcut.activated.connect(self._remove_visible_source)
        self.analysisShown.connect(lambda _analysis: self._sync_frame_list())
        for signal in (self.file_list.model().rowsInserted, self.file_list.model().rowsRemoved,
                       self.file_list.model().dataChanged, self.file_list.model().modelReset):
            signal.connect(self._frame_list_model_changed)
        self.refresh_frame_list()

    def refresh_frame_list(self):
        if self._frame_import_disposed or self._import_applying:
            return
        model = self.view_model
        spans = [(path, model.selected_frame_indices(path), source_name(path, model.describe(path)),
                  model.known_frame_count(path), row_tooltip(path, model.describe(path))) for path in model.state.files]
        with QSignalBlocker(self.frame_list.selectionModel()):
            self.frame_listing_model.reset_sources(spans)
            self._sync_frame_list()
        count = self.frame_listing_model.rowCount()
        self.frames_imported_label.setText(trf("{n} frames imported", n=count))
        self.choose_frames_button.setEnabled(bool(model.state.files) and not self._importing)
        self._show_file_position()

    def _frame_list_model_changed(self, *_args):
        self.refresh_frame_list()

    def _dispose_frame_import(self):
        self.cancel_frame_import(silent=True)
        self._frame_import_disposed = True
        for signal in (self.file_list.model().rowsInserted, self.file_list.model().rowsRemoved,
                       self.file_list.model().dataChanged, self.file_list.model().modelReset):
            try:
                signal.disconnect(self._frame_list_model_changed)
            except (TypeError, RuntimeError):
                pass

    def _sync_frame_list(self):
        if not hasattr(self, "frame_listing_model") or self._frame_import_disposed:
            return
        state = self.view_model.state
        path = self.view_model.current_path
        row = self.frame_listing_model.row_for(path, state.frame_index) if path is not None else -1
        with QSignalBlocker(self.frame_list.selectionModel()):
            index = self.frame_listing_model.index(row, 0)
            self.frame_list.setCurrentIndex(index)
            if index.isValid():
                self.frame_list.scrollTo(index)

    def _frame_row_selected(self, current, _previous):
        if not current.isValid() or self._import_applying or self._frame_kept():
            if self._frame_kept():
                self._sync_frame_list()
            return
        path, frame = self.frame_listing_model.ref(current.row())
        keys = [str(source).casefold() for source in self.view_model.state.files]
        row = keys.index(str(path).casefold())
        self.show_frame(row, frame)

    def _frame_list_menu(self, position):
        index = self.frame_list.indexAt(position)
        if not index.isValid():
            return
        path, _frame = self.frame_listing_model.ref(index.row())
        keys = [str(source).casefold() for source in self.view_model.state.files]
        menu = self._file_menu(keys.index(str(path).casefold()))
        menu.actions()[2].setText(tr("Remove Source from List"))
        menu.aboutToHide.connect(menu.deleteLater)
        menu.popup(self.frame_list.viewport().mapToGlobal(position))

    def imported_frame_step(self, delta):
        model = self.frame_listing_model
        row = model.row_for(self.view_model.current_path, self.view_model.state.frame_index)
        wanted = row + int(delta)
        if 0 <= wanted < model.rowCount() and not self._frame_kept():
            self.frame_list.setCurrentIndex(model.index(wanted, 0))

    def _remove_visible_source(self):
        index = self.frame_list.currentIndex()
        if index.isValid():
            path, _frame = self.frame_listing_model.ref(index.row())
            keys = [str(source).casefold() for source in self.view_model.state.files]
            self.remove_file(keys.index(str(path).casefold()))

    def _imported_frame_position(self):
        model = self.frame_listing_model
        row = model.row_for(self.view_model.current_path, self.view_model.state.frame_index)
        count = model.rowCount()
        free = not getattr(self, "_automatic_busy", False)
        self.previous_file_button.setEnabled(free and row > 0)
        self.next_file_button.setEnabled(free and 0 <= row < count - 1)
        if len(self.view_model.state.files) != 1 or self.view_model.import_selection(self.view_model.current_path) is not None:
            self.file_position_label.setText(trf("{position} / {total}", position=row + 1, total=count)
                                             if row >= 0 and count > 1 else "")

    def _frame_is_available(self, frame):
        path = self.view_model.current_path
        if path is None or self.view_model.frame_is_imported(path, frame):
            return True
        with QSignalBlocker(self.frame_spin):
            self.frame_spin.setValue(self.view_model.state.frame_index + 1)
        self._status(trf("Frame {n} is not imported. Use Choose Frames to add it.", n=frame + 1), "warning")
        return False

    def import_paths(self, paths, *, interactive=True, selections=None, on_imported=None):
        """Inspect headers in the background, then import references; no stack is decoded here."""
        paths = list(paths)
        if not paths:
            return False
        if self._list_kept():
            self._status(tr("Wait for the current task before changing imported frames."), "warning")
            return False
        self.cancel_frame_import(silent=True)
        self._import_cancelled = cancelled = Event()
        self._importing = True
        self._import_progress = None
        self.choose_frames_button.setEnabled(False)
        self.import_status_label.setText(tr("Inspecting detector sources…"))
        self.import_status_label.show()
        self.import_cancel_button.show()
        self.show_step("data")
        self.frameImportStarted.emit()
        self.tasks.submit("import-frames", lambda: prepare_frame_import(
            self.view_model._frames, paths,
            progress=lambda n, total, name: self._import_signals.progress.emit(cancelled, n, total, name),
            cancelled=cancelled.is_set),
            on_done=lambda sources: self._frame_import_ready(sources, paths, interactive, selections, cancelled, on_imported),
            on_error=lambda message, _details: self._frame_import_failed(message, cancelled))
        return True

    def _frame_import_progress(self, cancelled, n, total, name):
        if self._importing and cancelled is self._import_cancelled and not cancelled.is_set():
            self._import_progress = n, total, name
            self.import_status_label.setText(trf("Inspecting source {n} of {total}: {name}",
                                                 n=n, total=total, name=name))

    def cancel_frame_import(self, *, silent=False):
        if not hasattr(self, "_import_cancelled"):
            return
        busy = self._importing
        self._import_cancelled.set()
        self.tasks.cancel("import-frames")
        self._importing = False
        self.import_cancel_button.hide()
        self.import_status_label.hide()
        self.choose_frames_button.setEnabled(bool(self.view_model.state.files))
        if busy and not silent:
            self._status(tr("Frame import cancelled."), "info")

    def _frame_import_failed(self, message, cancelled):
        if cancelled is not self._import_cancelled or cancelled.is_set():
            return
        self.cancel_frame_import(silent=True)
        self._status(trf("Frame import failed: {error}", error=message_text(message)), "error")

    def _frame_import_ready(self, sources, paths, interactive, selections, cancelled, on_imported=None):
        if cancelled is not self._import_cancelled or cancelled.is_set() or sources is None:
            return
        self._importing = False
        self.import_cancel_button.hide()
        self.import_status_label.hide()
        self.choose_frames_button.setEnabled(bool(self.view_model.state.files))
        if not sources:
            self._status(tr("No detector frames were found."), "warning")
            return
        if self._list_kept():
            self._status(tr("Frame import cancelled. Wait for the current task before changing imported frames."), "warning")
            return
        choices = dict(selections or {})
        if interactive and any(source.frames > 1 for source in sources):
            dialog = FrameImportDialog(sources, selections=choices, parent=self)
            try:
                if dialog.exec_() != QDialog.Accepted:
                    self._status(tr("Frame import cancelled."), "info")
                    return
                choices = dialog.selections()
            finally:
                dialog.deleteLater()
        if self._list_kept():
            self._status(tr("Frame import cancelled. Wait for the current task before changing imported frames."), "warning")
            return
        if self._apply_frame_import(sources, choices, on_imported=on_imported):
            self.remember_recent(paths)

    def _apply_frame_import(self, sources, choices, *, on_imported=None):
        self._import_applying = True
        model = self.view_model
        before = [(str(path), model.known_frame_count(path), model.import_selection(path)) for path in model.state.files]
        try:
            validated = {}
            for source in sources:
                selection = choices.get(str(source.path))
                if isinstance(selection, dict):
                    selection = FrameSelection.from_record(selection)
                elif isinstance(selection, range):
                    selection = FrameSelection((selection,))
                elif selection is not None and not isinstance(selection, FrameSelection):
                    selection = FrameSelection.explicit(selection)
                if selection is not None and any(part[-1] >= source.frames for part in selection.ranges if len(part)):
                    raise ValueError("Every frame number must be within this source.")
                validated[str(source.path)] = selection
            for source in sources:
                model.remember_frame_count(source.path, source.frames)
            added = model.add_import_sources([source.path for source in sources])
            self._rows_regrouped()
            self._append_list_items(added)
            for source in sources:
                model.set_import_selection(source.path, validated[str(source.path)])
            self._refresh_rows()
        except (ValueError, KeyError, TypeError) as exc:
            self._status(trf("Frame import failed: {error}", error=message_text(exc)), "error")
            return False
        finally:
            self._import_applying = False
        self.refresh_frame_list()
        self.refresh_batch_entry()
        after = [(str(path), model.known_frame_count(path), model.import_selection(path)) for path in model.state.files]
        if before != after and (self._series_map is not None or getattr(self, "_series_fit_run", None) is not None):
            self._reset_series()
        if on_imported is not None and model.imported_frame_count():
            on_imported()
        if not self._frame_kept():
            for source in sources:
                indices = model.selected_frame_indices(source.path)
                if len(indices):
                    row = model.state.files.index(source.path)
                    self.show_frame(row, indices[0])
                    break
        return True

    def choose_imported_frames(self):
        if self._list_kept():
            self._status(tr(self._list_kept()), "warning")
            return
        sources = [ImportSource(path, self.view_model.known_frame_count(path)) for path in self.view_model.state.files]
        choices = {str(source.path): self.view_model.import_selection(source.path) for source in sources}
        if not sources:
            return
        dialog = FrameImportDialog(sources, selections=choices, parent=self)
        try:
            if dialog.exec_() == QDialog.Accepted:
                self._apply_frame_import(sources, dialog.selections())
                self._reset_series()
        finally:
            dialog.deleteLater()

    def _frame_import_language(self):
        self.frame_listing_model.language_changed()
        self.frames_imported_label.setText(trf("{n} frames imported", n=self.frame_listing_model.rowCount()))
        if self._importing:
            if self._import_progress is None:
                self.import_status_label.setText(tr("Inspecting detector sources…"))
            else:
                n, total, name = self._import_progress
                self.import_status_label.setText(trf("Inspecting source {n} of {total}: {name}",
                                                    n=n, total=total, name=name))

