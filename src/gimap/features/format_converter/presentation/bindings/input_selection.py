"""Input Selection behavior for Format Converter."""

from __future__ import annotations


from PyQt5.QtCore import Qt


from PyQt5.QtWidgets import (
    QDialog,
    QFileDialog,
    QMessageBox,
    QTableWidgetItem,
    QTreeWidgetItem,
)


from ..folder_import_dialog import FolderImportDialog
from ..display_formatting import INPUT_FILTER


class InputSelectionMixin:
    """Own input selection presentation behavior."""

    def add_paths(self, paths: list[str]) -> None:
        result = self.view_model.add_paths([path for path in paths if path])
        self._refresh_input_tree()
        self._refresh_selection_table()
        if result.errors:
            QMessageBox.warning(
                self,
                "Some inputs could not be added",
                "\n".join(result.errors[:12]),
            )
        elif not result.added and paths:
            self.input_note.setText("The selected inputs are already in the task list.")

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add detector images", "", INPUT_FILTER)
        if paths:
            self.add_paths(paths)

    def _choose_folder(self) -> None:
        dialog = FolderImportDialog(self, self.view_model)
        if dialog.exec_() != QDialog.Accepted:
            return
        paths = dialog.paths
        if not paths:
            QMessageBox.information(
                self, "Add Folder", "No matching files were found in that folder."
            )
            return
        if len(paths) > 10_000:
            answer = QMessageBox.question(
                self,
                "Large folder selection",
                f"This folder contains {len(paths):,} matching files. Add them to the task list?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return
        self.add_paths(paths)

    def _refresh_input_tree(self) -> None:
        self.input_tree.clear()
        for index, source in enumerate(self.sources):
            root = QTreeWidgetItem(
                (
                    source.name,
                    source.file_type,
                    str(source.frame_count),
                    source.selection_summary,
                    source.status,
                )
            )
            root.setData(0, Qt.UserRole, index)
            self.input_tree.addTopLevelItem(root)
            if source.file_type == "NXS":
                shape = " × ".join(str(value) for value in source.dataset_shape) or "Unknown"
                root.addChild(QTreeWidgetItem((f"Dataset: {source.dataset_path}", "", "", "", "")))
                root.addChild(QTreeWidgetItem((f"Shape: {shape}", "", "", "", "")))
                root.addChild(QTreeWidgetItem((f"Frames: 1–{source.frame_count}", "", "", "", "")))
                root.setExpanded(True)
        total_frames = sum(source.frame_count for source in self.sources)
        self.input_note.setText(
            f"{len(self.sources)} source file(s), {total_frames:,} available image(s) / frame(s)."
            if self.sources
            else "No input files yet."
        )

    def _input_current_changed(self, item: QTreeWidgetItem | None) -> None:
        while item is not None and item.parent() is not None:
            item = item.parent()
        index = item.data(0, Qt.UserRole) if item is not None else None
        source = (
            self.sources[index] if isinstance(index, int) and index < len(self.sources) else None
        )
        show = bool(source and source.file_type == "NXS" and len(source.dataset_paths) > 1)
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
            QMessageBox.warning(self, "Dataset selection", str(exc))

    def _refresh_selection_table(self) -> None:
        self.selection_table.blockSignals(True)
        self.selection_table.setRowCount(len(self.sources))
        for row, source in enumerate(self.sources):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
            use.setCheckState(Qt.Checked if source.included else Qt.Unchecked)
            use.setData(Qt.UserRole, row)
            self.selection_table.setItem(row, 0, use)
            self.selection_table.setItem(row, 1, QTableWidgetItem(source.name))
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

    def _sort_sources(self) -> None:
        self.view_model.sort_sources()
        self._refresh_input_tree()
        self._refresh_selection_table()

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

    def _update_frame_editor(self) -> None:
        mode = self.frame_mode.currentText()
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
            QMessageBox.information(self, "Frame selection", "Select one or more input rows first.")
            return
        try:
            waxs = getattr(getattr(self.parent(), "components", None), "waxs_page", None)
            current_frame = getattr(
                getattr(waxs, "frame_spin", None),
                "value",
                lambda: 1,
            )()
            self.view_model.apply_frame_selection(
                indices,
                self.frame_mode.currentText(),
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
            QMessageBox.warning(self, "Frame selection", str(exc))
