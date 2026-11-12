"""Cancellable table inspection and explicit column choices for curve imports."""

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from uuid import uuid4

from PyQt5 import sip
from PyQt5.QtCore import QSignalBlocker, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QListWidgetItem, QTableWidgetItem

from src.gimap.app.presentation.i18n import apply_to, current_language, language_changed, tr, trf
from src.gimap.app.presentation.task_runner import TaskRunner

from ..application.curve_sources import sources_from_table
from ..application.models import InSituSourceFrame
from .views.curve_import_view import CurveImportView


@dataclass(frozen=True)
class CurveImportPlan:
    sources: tuple[InSituSourceFrame, ...]
    independent: bool
    q_unit: str


class _InspectionLifetime:
    """Keep cancellation independent of the QWidget, including during cyclic GC."""

    def __init__(self, tasks, key, owns_tasks):
        self.token = Event()
        self.tasks, self.key, self.owns_tasks = tasks, key, owns_tasks
        if owns_tasks:
            tasks.busy_changed.connect(self.retire_if_idle)

    def cancel(self):
        self.token.set()
        if not sip.isdeleted(self.tasks):
            self.tasks.cancel(self.key)
            self.retire_if_idle()

    def retire_if_idle(self, *_args):
        if (self.owns_tasks and self.token.is_set() and not sip.isdeleted(self.tasks)
                and not self.tasks.is_busy()):
            self.tasks.deleteLater()


class CurveImportDialog(QDialog, CurveImportView):
    """``import_plan`` is populated only after every file was inspected and accepted.

    ``inspect_curve_table(path)`` is the view-model's read-only table inspector. It runs on a worker;
    layout changes and ``sources_from_table`` only construct source tokens, never read full curves.
    An optional shared TaskRunner must outlive this dialog.
    """

    def __init__(self, paths, inspect_curve_table, parent=None, *, task_runner=None, q_unit="angstrom"):
        super().__init__(parent)
        self.paths = tuple(Path(path) for path in paths)
        self.import_plan = None
        self._inspector = inspect_curve_table
        self._previews = [None] * len(self.paths)
        self._layouts = ["single"] * len(self.paths)
        self._errors = {}
        self._layout_notices = {}
        self._active_index = -1
        self._complete = False
        self._disposed = False
        self.tasks = task_runner or TaskRunner(QApplication.instance(), max_threads=1)
        self._task_key = "curve-import-" + uuid4().hex
        self._lifetime = _InspectionLifetime(self.tasks, self._task_key, task_runner is None)
        self.destroyed.connect(self._lifetime.cancel)
        self.setup_curve_import_view()
        self.q_unit_combo.setCurrentIndex(1 if q_unit in {"nm", "nm^-1"} else 0)
        for path in self.paths:
            item = QListWidgetItem(path.name, self.file_list)
            item.setToolTip(str(path))
        self.file_list.currentRowChanged.connect(self._show_current)
        self.layout_combo.currentIndexChanged.connect(self._layout_changed)
        self.apply_all_button.clicked.connect(self._apply_all)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.inspect_job.cancelRequested.connect(self.reject)
        language_changed().connect(self.refresh_language)
        self.file_list.setCurrentRow(0)
        self.refresh_language()
        self._start_timer = QTimer(self)
        self._start_timer.setSingleShot(True)
        self._start_timer.timeout.connect(self._inspect_next)
        self._start_timer.start(0)

    def _inspect_next(self):
        if self._disposed:
            return
        self._active_index += 1
        if self._active_index >= len(self.paths):
            self._complete = True
            self.refresh_language()
            return
        index = self._active_index
        inspector, path, token = self._inspector, self.paths[index], self._lifetime.token

        def inspect():
            if token.is_set():
                return None
            result = inspector(path)
            return None if token.is_set() else result

        self.refresh_language()
        self.tasks.submit(self._task_key, inspect, on_done=self._inspected, on_error=self._inspect_failed)

    def _inspected(self, preview):
        if self._disposed or self._lifetime.token.is_set() or preview is None:
            return
        self._previews[self._active_index] = preview
        self._layouts[self._active_index] = (
            "shared_q" if preview.suggested_layout == "shared_q" and preview.allow_shared_q else "single")
        try:
            self._sources(self._active_index)
        except ValueError as exc:
            self._errors[self._active_index] = str(exc)
        self._inspect_next()

    def _inspect_failed(self, message, _trace):
        if self._disposed or self._lifetime.token.is_set():
            return
        self._errors[self._active_index] = message
        self._inspect_next()

    def _sources(self, index):
        preview = self._previews[index]
        return sources_from_table(preview, self._layouts[index]) if preview is not None else ()

    def _same_columns(self):
        if not self._complete or self._errors or len(self.paths) < 2:
            return False
        keys = {(p.column_names, p.has_header, p.analyze_export) for p in self._previews if p is not None}
        return len(keys) == 1

    def _layout_changed(self):
        index = self.file_list.currentRow()
        if not 0 <= index < len(self.paths) or self._previews[index] is None:
            return
        preview = self._previews[index]
        layout = self.layout_combo.currentData()
        if layout == "shared_q" and not preview.allow_shared_q:
            self._show_current()
            return
        try:
            sources_from_table(preview, layout)
        except ValueError as exc:
            self._layout_notices[index] = str(exc)
            self.refresh_language()
            return
        self._layout_notices.pop(index, None)
        self._layouts[index] = layout
        self.refresh_language()

    def _apply_all(self):
        if not self._same_columns():
            return
        index = self.file_list.currentRow()
        layout = self._layouts[index]
        if layout == "shared_q" and any(not p.allow_shared_q for p in self._previews):
            return
        self._layouts = [layout] * len(self.paths)
        self._layout_notices.clear()
        self.refresh_language()

    def _show_current(self, *_args):
        if self._disposed:
            return
        index = self.file_list.currentRow()
        preview = self._previews[index] if 0 <= index < len(self.paths) else None
        self.preview_table.clear()
        self.preview_table.setRowCount(0)
        self.preview_table.setColumnCount(0)
        self.file_path_label.setText(str(self.paths[index]) if 0 <= index < len(self.paths) else "")
        self.layout_combo.setEnabled(preview is not None and index not in self._errors)
        self.apply_all_button.setEnabled(self._same_columns())
        self.apply_all_button.setToolTip(tr(
            "Apply this layout to files with identical column headings."
            if self._same_columns() else "Files with different columns must be reviewed individually."))
        if preview is None:
            self.layout_note.setText(trf("Could not inspect this file: {error}", error=tr(self._errors[index]))
                                     if index in self._errors else tr("Waiting for the file preview …"))
            self.unit_note.clear()
            return
        layout = self._layouts[index]
        with QSignalBlocker(self.layout_combo):
            self.layout_combo.setCurrentIndex(self.layout_combo.findData(layout))
            self.layout_combo.model().item(self.layout_combo.findData("shared_q")).setEnabled(preview.allow_shared_q)
        note = ""
        if index in self._errors:
            note = trf("Could not import this table: {error}", error=tr(self._errors[index]))
        elif index in self._layout_notices:
            note = trf("Layout not changed: {error}", error=tr(self._layout_notices[index]))
        elif preview.analyze_export:
            note = "Analyze export: q, I, σ and pixel counts describe one curve. Shared-q import is disabled."
        elif preview.uncertainty_columns:
            note = ("Named uncertainty columns cannot be imported as intensities. Only q, I and σ are used; "
                    "put curves with separate uncertainties in separate q, I, σ files.")
        elif preview.suggested_layout == "ambiguous":
            note = ("No unambiguous column roles were found. The default treats column 3 as σ, not another curve. "
                    "Choose Shared q if the remaining columns are intensities; values are not used to guess.")
        elif layout == "single" and preview.column_count > 3:
            note = "Only q, I and σ are used; columns after the third are ignored."
        elif layout == "shared_q":
            note = "Every column after q is a separate intensity curve. No uncertainty column is used."
        self.layout_note.setText(tr(note))
        self.layout_combo.setToolTip(tr(note) if not preview.allow_shared_q else "")
        self.preview_table.setColumnCount(preview.column_count)
        rows = preview.sample_rows[:10]
        self.preview_table.setRowCount(len(rows))
        roles = ["q"] + (["Intensity"] * (preview.column_count - 1) if layout == "shared_q"
                          else ["Intensity", "Uncertainty (σ)"] + ["Ignored"] * max(0, preview.column_count - 3))
        headers = [trf("{name}\nRole: {role}", name=name if preview.has_header else trf("Column {n}", n=i + 1),
                       role=tr(roles[i])) for i, name in enumerate(preview.column_names)]
        self.preview_table.setHorizontalHeaderLabels(headers)
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                self.preview_table.setItem(row, column, QTableWidgetItem(str(value)))
        unit_text = {"angstrom": "Å⁻¹", "nm": "nm⁻¹"}.get(preview.unit, preview.unit)
        self.unit_note.setText(trf("q unit from this file's header: {unit}", unit=unit_text)
                               if preview.unit else tr("No q unit in this header: the selected fallback unit will be used."))

    def refresh_language(self, *_args):
        if self._disposed:
            return
        apply_to(self, current_language())
        total = 0
        for index, path in enumerate(self.paths):
            if index in self._errors:
                state = tr("Read failed")
            elif self._previews[index] is not None:
                count = len(self._sources(index))
                total += count
                state = tr("1 curve") if count == 1 else trf("{count} curves", count=count)
            else:
                state = tr("Checking …" if index == self._active_index else "Waiting")
            self.file_list.item(index).setText(trf("{name} · {state}", name=path.name, state=state))
        checked = sum(preview is not None or index in self._errors
                      for index, preview in enumerate(self._previews))
        self.inspect_job.set_state(
            ("failed" if self._errors else "succeeded") if self._complete else "running",
            trf("Checked {done} of {total} files", done=checked, total=len(self.paths)),
            progress=checked / len(self.paths) if self.paths else 0)
        self.inspect_job.setVisible(not self._complete or bool(self._errors))
        self.q_unit_combo.setEnabled(any(p is None or not p.unit for p in self._previews))
        self.error_details.setVisible(bool(self._errors))
        self.error_details.setPlainText("\n".join(
            trf("{name}: {error}", name=self.paths[i].name, error=tr(message)) for i, message in self._errors.items()))
        if self._errors:
            summary = trf("{count} files could not be checked. Nothing will be imported; correct the files and try again.",
                          count=len(self._errors))
        elif not self.paths:
            summary = tr("No files selected.")
        elif not self._complete:
            summary = tr("Checking files before import …")
        else:
            summary = trf("Files: {files} · Curves to import: {count}", count=total, files=len(self.paths))
        self.summary_label.setText(summary)
        self.import_button.setText(tr("Import 1 curve") if total == 1 else trf("Import {count} curves", count=total))
        self.import_button.setEnabled(self._complete and not self._errors and total > 0)
        self._show_current()

    def accept(self):
        if not self._complete or self._errors or not self.paths or self._disposed:
            return
        sources = []
        for index in range(len(self.paths)):
            try:
                sources.extend(self._sources(index))
            except ValueError as exc:
                self._errors[index] = str(exc)
        if self._errors:
            self.refresh_language()
            return
        if not sources:
            return
        self.import_plan = CurveImportPlan(tuple(sources), bool(self.mode_combo.currentData()), self.q_unit_combo.currentData())
        self.dispose()
        super().accept()

    def reject(self):
        self.dispose()
        super().reject()

    def closeEvent(self, event):  # noqa: N802 - Qt API
        self.dispose()
        super().closeEvent(event)

    def dispose(self):
        if not self._disposed:
            self._disposed = True
            self._start_timer.stop()
            self._lifetime.cancel()


__all__ = ["CurveImportDialog", "CurveImportPlan"]
