"""Input Selection behavior for Format Converter."""

from __future__ import annotations

import os

from PyQt5.QtCore import Qt


from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QMessageBox,
    QTableWidgetItem,
    QTreeWidgetItem,
)


from src.gimap.app.presentation.i18n import tr, trf
from src.gimap.app.presentation.theme import set_role

from ..folder_import_dialog import FolderImportDialog
from ..display_formatting import INPUT_FILTER

UNSUPPORTED_CURRENT = "{name} is not an NXS, CBF or TIFF file: the converter cannot read it."
# Under the list: every input is one file, or some consist of several (module files, a master with data files).
INPUTS_NOTE = "{files} source file(s), {frames} available image(s) / frame(s)."
INPUTS_OF_FILES_NOTE = "{sources} inputs ({files} files), {frames} available image(s) / frame(s)."
SEVERAL_FILES = ("module_set", "chunked_master")
"""Layouts whose one input consists of several files: the list says so under the input."""


def source_lines(source) -> list[str]:
    """What a NeXus / HDF5 input holds, in the interface language: dataset, shape, frames; for an input
    made of several files what it consists of; then what the reader had to say about it."""
    shape = " × ".join(str(value) for value in source.dataset_shape) or tr("Unknown")
    lines = [
        trf("Dataset: {path}", path=source.dataset_path),
        trf("Shape: {shape}", shape=shape),
        trf("Frames: 1–{count}", count=source.frame_count),
    ]
    if source.summary is not None and source.layout in SEVERAL_FILES:
        lines.append(trf(source.summary[0], **source.summary[1]))
    lines += ["⚠ " + trf(template, **values) for template, values in source.warnings]
    return lines


class InputSelectionMixin:
    """Own input selection presentation behavior."""

    def current_file_problem(self) -> str:
        """Why the frame shown in Analyze cannot be an input ('' when it can, or when there is none)."""
        current = self.current_file
        if not current or self.view_model.supports_input_path(current):
            return ""
        return trf(UNSUPPORTED_CURRENT, name=os.path.basename(current))

    def use_current_file(self, path: str) -> str:
        """Tools ▸ Convert Current File…: make ``path`` (the frame Analyze shows) an input.

        Returns '' when it is an input now (or when there is no frame), else the reason the
        converter cannot read it. The window says that reason itself: under the input list, and
        as the tooltip of the disabled "Use currently opened file".
        """
        self.current_file = path or ""
        if self.current_file:
            self.add_paths([self.current_file])
        else:
            self._show_current_button()
        return self.current_file_problem()

    def _show_current_button(self) -> None:
        """"Use currently opened file" works only for a frame the converter can read."""
        problem = self.current_file_problem()
        self.current_button.setEnabled(bool(self.current_file) and not problem)
        self.current_button.setToolTip(problem)

    def add_paths(self, paths: list[str]) -> None:
        paths = [path for path in paths if path]
        # The frame of Analyze in a format the converter cannot read (EDF) is refused in words
        # under the input list, not with an error box about an input nobody chose.
        refused = self.current_file_problem()
        if refused:
            current = os.path.normcase(self.current_file)
            wanted = [path for path in paths if os.path.normcase(path) != current]
            refused = refused if len(wanted) != len(paths) else ""
            paths = wanted
        result = self.view_model.add_paths(paths)
        self._refresh_input_tree()
        self._refresh_selection_table()
        self._follow_default_destination()
        self._show_current_button()
        if refused:
            set_role(self.input_note, "warning")  # until the list is shown again
            self.input_note.setText(refused)
        if result.errors:
            QMessageBox.warning(
                self,
                tr("Some inputs could not be added"),
                "\n".join(result.errors[:12]),
            )
        elif not refused and not result.added and paths:
            self.input_note.setText(tr("The selected inputs are already in the task list."))

    def _follow_default_destination(self) -> None:
        """While the destination is the default, it is ``converted`` next to the first input."""
        if self.destination_edit.text().strip() != self._auto_destination:
            return  # chosen by the user: kept
        self._auto_destination = self.view_model.default_destination()
        self.destination_edit.setText(self._auto_destination)

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("Add detector images"), self.view_model.last_folder(), INPUT_FILTER
        )
        if paths:
            self.view_model.remember_folder(paths[0])
            self.add_paths(paths)

    def _choose_folder(self) -> None:
        dialog = FolderImportDialog(self, self.view_model, folder=self.view_model.last_folder())
        if dialog.exec_() != QDialog.Accepted:
            return
        self.view_model.remember_folder(dialog.path_edit.text().strip())
        paths = dialog.paths
        if not paths:
            QMessageBox.information(
                self, tr("Add Folder"), tr("No matching files were found in that folder.")
            )
            return
        if self._many_files_confirmed(len(paths)):
            self.add_paths(paths)

    def _many_files_confirmed(self, count: int) -> bool:
        if count <= 10_000:
            return True
        answer = QMessageBox.question(
            self,
            tr("Large folder selection"),
            # Also for a drop of files and folders: no "this folder".
            trf("{count} matching files were found. Add them to the task list?", count=f"{count:,}"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _refresh_input_tree(self) -> None:
        self.input_tree.clear()
        for index, source in enumerate(self.sources):
            root = QTreeWidgetItem(
                (
                    source.name,
                    source.file_type,
                    str(source.frame_count),
                    source.selection_summary,
                    tr(source.status),  # "Ready", "Failed" …: the model keeps the English
                )
            )
            root.setData(0, Qt.UserRole, index)
            # Every file of the input: the module files of a set, a master and its data files.
            root.setToolTip(0, "\n".join(source.members) if len(source.members) > 1 else source.path)
            self.input_tree.addTopLevelItem(root)
            if source.is_hdf5:
                for text in source_lines(source):
                    child = QTreeWidgetItem((text, "", "", "", ""))
                    child.setToolTip(0, text)
                    root.addChild(child)
                    child.setFirstColumnSpanned(True)  # a line of text, not a row of the table: the whole width
                root.setExpanded(True)
        total_frames = sum(source.frame_count for source in self.sources)
        files = sum(max(1, len(source.members)) for source in self.sources)
        set_role(self.input_note, "muted")
        if not self.sources:
            self.input_note.setText(tr("No input files yet."))
        elif files == len(self.sources):
            self.input_note.setText(trf(INPUTS_NOTE, files=files, frames=f"{total_frames:,}"))
        else:
            self.input_note.setText(
                trf(INPUTS_OF_FILES_NOTE, sources=len(self.sources), files=files, frames=f"{total_frames:,}")
            )

    def _input_current_changed(self, item: QTreeWidgetItem | None) -> None:
        while item is not None and item.parent() is not None:
            item = item.parent()
        index = item.data(0, Qt.UserRole) if item is not None else None
        source = (
            self.sources[index] if isinstance(index, int) and index < len(self.sources) else None
        )
        show = bool(source and source.is_hdf5 and len(source.dataset_paths) > 1)
        self.dataset_label.setVisible(show)
        self.dataset_combo.setVisible(show)
        self.dataset_note.setVisible(show)
        if show:
            self.dataset_combo.blockSignals(True)
            self.dataset_combo.clear()
            self.dataset_combo.addItems(source.dataset_paths)
            self.dataset_combo.setCurrentText(source.dataset_path or "")
            self.dataset_combo.setProperty("source_index", index)
            self.dataset_combo.blockSignals(False)

    def _dataset_changed(self, dataset_path: str) -> None:
        index = self.dataset_combo.property("source_index")
        if not dataset_path or not isinstance(index, int) or index >= len(self.sources):
            return
        try:
            self.view_model.select_dataset(self.sources[index], dataset_path)
            self._refresh_input_tree()
            self._refresh_selection_table()
        except Exception as exc:
            QMessageBox.warning(self, tr("Dataset selection"), str(exc))

    def _refresh_selection_table(self) -> None:
        self.selection_table.blockSignals(True)
        self.selection_table.setRowCount(len(self.sources))
        for row, source in enumerate(self.sources):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            use.setCheckState(Qt.Checked if source.included else Qt.Unchecked)
            use.setData(Qt.UserRole, row)
            self.selection_table.setItem(row, 0, use)
            name = QTableWidgetItem(source.name)
            name.setToolTip("\n".join(source.members) if len(source.members) > 1 else source.path)
            self.selection_table.setItem(row, 1, name)
            self.selection_table.setItem(row, 2, QTableWidgetItem(source.file_type))
            self.selection_table.setItem(row, 3, QTableWidgetItem(str(source.frame_count)))
            self.selection_table.setItem(row, 4, QTableWidgetItem(source.selection_summary))
        self.selection_table.blockSignals(False)
        self._filter_sources(self.filter_edit.text())

    def _set_all_included(self, included: bool) -> None:
        self.view_model.set_all_included(included)
        self._refresh_selection_table()

    def _include_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        index = item.data(Qt.UserRole)
        if isinstance(index, int) and index < len(self.sources):
            self.view_model.set_source_included(index, item.checkState() == Qt.Checked)

    def _selected_source_indices(self) -> list[int]:
        indices = []
        for model_index in self.selection_table.selectionModel().selectedRows():
            item = self.selection_table.item(model_index.row(), 0)
            source_index = item.data(Qt.UserRole) if item else None
            if isinstance(source_index, int):
                indices.append(source_index)
        return sorted(set(indices))

    def _remove_selected(self) -> None:
        selected = self._selected_source_indices()
        if not selected:
            return
        self.view_model.remove_indices(selected)
        self._refresh_input_tree()
        self._refresh_selection_table()
        self._follow_default_destination()

    def _sort_sources(self) -> None:
        self.view_model.sort_sources()
        self._refresh_input_tree()
        self._refresh_selection_table()
        self._follow_default_destination()

    def _filter_sources(self, text: str) -> None:
        needle = text.strip().lower()
        for row in range(self.selection_table.rowCount()):
            item = self.selection_table.item(row, 1)
            self.selection_table.setRowHidden(
                row, bool(needle and item and needle not in item.text().lower())
            )

    def _selection_current_changed(self) -> None:
        selected = self._selected_source_indices()
        if not selected:
            return
        source = self.sources[selected[0]]
        maximum = max(1, source.frame_count)
        self.range_start.setRange(1, maximum)
        self.range_end.setRange(1, maximum)
        self.range_start.setValue(1)
        self.range_end.setValue(maximum)
        self._start_preview(source)

    def _frame_mode_key(self) -> str:
        """The English mode key of the frame-mode combo; its displayed text may be translated."""
        return self.frame_mode.currentData() or "All"

    def _update_frame_editor(self, *_args) -> None:
        mode = self._frame_mode_key()
        range_enabled = mode == "Frame range"
        custom_enabled = mode == "Custom"
        nth_enabled = mode == "Every Nth frame"
        self.range_start.setEnabled(range_enabled)
        self.range_end.setEnabled(range_enabled)
        self.custom_frames.setEnabled(custom_enabled)
        self.nth_frame.setEnabled(nth_enabled)

    def _apply_frame_selection(self) -> None:
        indices = self._selected_source_indices()
        if not indices:
            QMessageBox.information(
                self, tr("Frame selection"), tr("Select one or more input rows first.")
            )
            return
        try:
            analyze = getattr(getattr(self.parent(), "components", None), "analyze_page", None)
            current_frame = getattr(
                getattr(analyze, "frame_spin", None),
                "value",
                lambda: 1,
            )()
            self.view_model.apply_frame_selection(
                indices,
                self._frame_mode_key(),
                current_file=self.current_file,
                current_frame=current_frame,
                range_start=self.range_start.value(),
                range_end=self.range_end.value(),
                custom_frames=self.custom_frames.text(),
                nth_frame=self.nth_frame.value(),
            )
            self._refresh_input_tree()
            self._refresh_selection_table()
            self._start_preview(self.sources[indices[0]])
        except ValueError as exc:
            QMessageBox.warning(self, tr("Frame selection"), tr(str(exc)))
