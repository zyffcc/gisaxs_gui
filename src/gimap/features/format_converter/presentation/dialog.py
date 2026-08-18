"""Feature-owned PyQt presentation for detector image format conversion."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QImage, QPixmap
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QTableWidgetItem,
    QTreeWidgetItem,
    QWidget,
)

from src.gimap.app.bootstrap import create_standalone_legacy_context
from src.gimap.app.presentation import apply_design_system
from src.gimap.app.presentation.assets import app_icon
from src.gimap.app.presentation.section_bindings import (
    bind_advanced_section,
    bind_parameter_section,
)

from .views import (
    ConversionProgressDialogView,
    FolderImportDialogView,
    FormatConverterDialogView,
)


INPUT_FILTER = "Detector images (*.nxs *.cbf *.tif *.tiff);;NXS (*.nxs);;CBF (*.cbf);;TIFF (*.tif *.tiff)"


def _human_bytes(value: int) -> str:
    number = float(max(0, value))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if number < 1024.0 or unit == "TB":
            return f"{number:.1f} {unit}" if unit != "B" else f"{int(number)} B"
        number /= 1024.0
    return f"{number:.1f} TB"


def _duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _array_pixmap(data: np.ndarray, width: int = 210, height: int = 155) -> QPixmap:
    array = np.asarray(data, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size:
        low, high = np.percentile(finite, (1, 99))
        if high <= low:
            high = low + 1.0
        image = np.clip((np.nan_to_num(array, nan=low) - low) / (high - low), 0, 1)
    else:
        image = np.zeros(array.shape, dtype=np.float32)
    gray = np.ascontiguousarray(np.rint(image * 255).astype(np.uint8))
    qimage = QImage(gray.data, gray.shape[1], gray.shape[0], gray.strides[0], QImage.Format_Grayscale8).copy()
    return QPixmap.fromImage(qimage).scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class _PreviewWorker(QObject):
    finished = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, request_id: int, source, view_model):
        super().__init__()
        self.request_id = request_id
        self.source = source
        self.view_model = view_model

    def run(self) -> None:
        try:
            payload = self.view_model.load_preview(self.source)
            self.finished.emit(self.request_id, payload)
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


class _ConversionWorker(QObject):
    progress = pyqtSignal(int, int, str, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, options, view_model):
        super().__init__()
        self.options = options
        self.view_model = view_model

    def run(self) -> None:
        try:
            report = self.view_model.convert(self.options, self.progress.emit)
            self.finished.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.view_model.cancel()

    def set_paused(self, paused: bool) -> None:
        self.view_model.set_paused(paused)


class FolderImportDialog(QDialog, FolderImportDialogView):
    def __init__(self, parent: QWidget | None = None, view_model=None):
        super().__init__(parent)
        self.view_model = view_model or getattr(parent, "view_model", None)
        if self.view_model is None:
            from ..bootstrap import create_format_converter_view_model

            self.view_model = create_format_converter_view_model(
                create_standalone_legacy_context()
            )
        self.paths: list[str] = []
        self.setupUi(self)
        self.setWindowIcon(app_icon())
        self.browse_button.clicked.connect(self._browse)
        self.buttons.accepted.connect(self._accept_if_valid)
        self.buttons.rejected.connect(self.reject)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select input folder", self.path_edit.text())
        if folder:
            self.path_edit.setText(self.view_model.normalize_path(folder))

    def _accept_if_valid(self) -> None:
        if not any(check.isChecked() for check in (self.cbf, self.tiff, self.nxs)):
            QMessageBox.warning(self, "Add Folder", "Select at least one input format.")
            return
        try:
            self.paths = self.view_model.scan_folder(
                self.path_edit.text(),
                include_cbf=self.cbf.isChecked(),
                include_tiff=self.tiff.isChecked(),
                include_nxs=self.nxs.isChecked(),
                recursive=self.recursive.isChecked(),
            )
        except NotADirectoryError:
            QMessageBox.warning(self, "Add Folder", "Please select a valid folder.")
            return
        self.accept()


class ConversionProgressDialog(QDialog, ConversionProgressDialogView):
    def __init__(self, destination: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.destination = destination
        self.report_path = ""
        self.running = True
        self.setupUi(self)
        self.setWindowIcon(app_icon())
        self.setModal(False)
        self.job_status.set_actions_visible(details=False)
        self.bar = self.job_status.progress_bar
        self.pause_button = self.job_status.pause_button
        self.cancel_button = self.job_status.cancel_button
        self.open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.destination)))
        self.report_button.clicked.connect(self._open_report)
        self.close_button.clicked.connect(self.accept)
        apply_design_system(self)

    def complete(self, report) -> None:
        self.running = False
        self.report_path = report.report_path
        self.bar.setValue(self.bar.maximum())
        if report.cancelled:
            self.title.setText("Conversion cancelled")
            self.job_status.set_state("cancelled", "Conversion cancelled", progress=1.0)
        else:
            self.title.setText("Conversion completed")
            self.job_status.set_state("succeeded", "Conversion completed", progress=1.0)
        self.result.setText(f"{len(report.succeeded)} succeeded\n{len(report.failed)} failed")
        self.pause_button.hide()
        self.cancel_button.hide()
        self.open_button.show()
        self.report_button.setVisible(bool(self.report_path))
        self.close_button.show()

    def fail(self, message: str) -> None:
        self.running = False
        self.title.setText("Conversion could not be completed")
        self.result.setText(message)
        self.job_status.set_state("failed", message, progress=0.0)
        self.pause_button.hide()
        self.cancel_button.hide()
        self.open_button.show()
        self.close_button.show()

    def _open_report(self) -> None:
        if self.report_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.report_path))

    def closeEvent(self, event) -> None:
        if self.running:
            event.ignore()
            return
        event.accept()


class FormatConverterDialog(QDialog, FormatConverterDialogView):
    """Full converter. Inputs are detected; there is no single/batch mode."""

    def __init__(
        self,
        parent: QWidget | None = None,
        current_file: str = "",
        app_context=None,
        view_model=None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowIcon(app_icon())
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.app_context = (
            app_context
            or getattr(parent, "app_context", None)
            or getattr(view_model, "app_context", None)
            or create_standalone_legacy_context()
        )
        if view_model is None:
            # Compatibility construction path for the legacy class entry point.
            from ..bootstrap import create_format_converter_view_model

            view_model = create_format_converter_view_model(self.app_context)
        self.view_model = view_model
        self.sources = self.view_model.sources
        self.current_file = current_file
        self._preview_thread: Optional[QThread] = None
        self._preview_worker: Optional[_PreviewWorker] = None
        self._preview_request = 0
        self._pending_preview_source = None
        self._conversion_thread: Optional[QThread] = None
        self._conversion_worker: Optional[_ConversionWorker] = None
        self._progress_dialog: Optional[ConversionProgressDialog] = None
        self._conversion_started_at = 0.0
        self._paused = False
        self._bind_form()
        if current_file and self.view_model.supports_input_path(current_file):
            self.add_paths([current_file])

    def _bind_form(self) -> None:
        """Attach behavior and compatibility names to the Designer-owned form."""
        bind_parameter_section(
            self.format_input_section,
            self.formatInputTitle,
            self.formatInputDescription,
            self.formatInputContent,
            self.formatInputContentLayout,
        )
        bind_parameter_section(
            self.format_configure_section,
            self.formatConfigureTitle,
            self.formatConfigureDescription,
            self.formatConfigureContent,
            self.formatConfigureContentLayout,
        )
        bind_parameter_section(
            self.format_preview_panel,
            self.formatPreviewTitle,
            self.formatPreviewDescription,
            self.formatPreviewContent,
            self.formatPreviewContentLayout,
        )
        bind_parameter_section(
            self.format_output_section,
            self.formatOutputTitle,
            self.formatOutputDescription,
            self.formatOutputContent,
            self.formatOutputContentLayout,
        )
        bind_parameter_section(
            self.format_run_section,
            self.formatRunTitle,
            self.formatRunDescription,
            self.formatRunContent,
            self.formatRunContentLayout,
        )
        bind_advanced_section(
            self.frame_advanced_section,
            self.frameAdvancedToggle,
            self.frameAdvancedDescription,
            self.frameAdvancedContent,
            self.frameAdvancedContentLayout,
        )
        bind_advanced_section(
            self.format_output_advanced,
            self.formatOutputAdvancedToggle,
            self.formatOutputAdvancedDescription,
            self.formatOutputAdvancedContent,
            self.formatOutputAdvancedContentLayout,
        )

        self.step_labels = [
            self.step_input_label,
            self.step_configure_label,
            self.step_output_label,
        ]
        self.preview_captions = [
            self.first_preview_caption,
            self.middle_preview_caption,
            self.last_preview_caption,
        ]
        self.preview_labels = [
            self.first_preview_label,
            self.middle_preview_label,
            self.last_preview_label,
        ]
        self.current_button.setEnabled(bool(self.current_file))
        self.destination_edit.setText(str(Path.cwd() / "converted"))
        self.selection_splitter.setSizes((650, 340))
        self.input_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            self.input_tree.header().setSectionResizeMode(
                column,
                QHeaderView.ResizeToContents,
            )
        self.selection_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.Stretch,
        )

        self.format_group = QButtonGroup(self.output_format_group)
        self.format_buttons = {
            "TIFF": self.tiff_format_button,
            "CBF": self.cbf_format_button,
            "HDF5": self.hdf5_format_button,
            "NumPy": self.numpy_format_button,
        }
        for format_name, button in self.format_buttons.items():
            button.setProperty("format_name", format_name)
            self.format_group.addButton(button)
        for index, mode in enumerate(
            ("original", "float32", "scale_uint16", "clip_uint16")
        ):
            self.data_mode.setItemData(index, mode)

        self.add_files_button.clicked.connect(self._choose_files)
        self.add_folder_button.clicked.connect(self._choose_folder)
        self.current_button.clicked.connect(lambda: self.add_paths([self.current_file]))
        self.input_tree.currentItemChanged.connect(self._input_current_changed)
        self.dataset_combo.currentTextChanged.connect(self._dataset_changed)
        self.select_all_button.clicked.connect(lambda: self._set_all_included(True))
        self.select_none_button.clicked.connect(lambda: self._set_all_included(False))
        self.remove_selected_button.clicked.connect(self._remove_selected)
        self.sort_button.clicked.connect(self._sort_sources)
        self.filter_edit.textChanged.connect(self._filter_sources)
        self.selection_table.itemChanged.connect(self._include_changed)
        self.selection_table.itemSelectionChanged.connect(
            self._selection_current_changed
        )
        self.frame_mode.currentTextChanged.connect(self._update_frame_editor)
        self.apply_frames.clicked.connect(self._apply_frame_selection)
        self.destination_button.clicked.connect(self._choose_destination)
        self.naming_combo.currentTextChanged.connect(self._update_output_preview)
        self.format_group.buttonClicked.connect(self._update_output_preview)
        self.destination_edit.textChanged.connect(self._update_output_preview)
        self.container_check.toggled.connect(self._container_toggled)
        self.write_sidecar.toggled.connect(self._sidecar_toggled)
        self.data_mode.currentIndexChanged.connect(self._update_output_preview)
        self.preserve_values.toggled.connect(self._preserve_values_toggled)
        self.back_button.clicked.connect(self._back)
        self.next_button.clicked.connect(self._next)
        self.cancel_button.clicked.connect(self.close)

        self._update_frame_editor()
        self._update_step_header()
        apply_design_system(self)

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
            QMessageBox.information(self, "Add Folder", "No matching files were found in that folder.")
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
            root = QTreeWidgetItem((source.name, source.file_type, str(source.frame_count), source.selection_summary, source.status))
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
            if self.sources else "No input files yet."
        )

    def _input_current_changed(self, item: QTreeWidgetItem | None) -> None:
        while item is not None and item.parent() is not None:
            item = item.parent()
        index = item.data(0, Qt.UserRole) if item is not None else None
        source = self.sources[index] if isinstance(index, int) and index < len(self.sources) else None
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
            self.selection_table.setRowHidden(row, bool(needle and item and needle not in item.text().lower()))

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

    def _start_preview(self, source) -> None:
        self._preview_request += 1
        request_id = self._preview_request
        self._pending_preview_source = source
        self.preview_stats.setText(f"Loading preview for {source.name}…")
        for label in self.preview_labels:
            label.setText("Loading…")
            label.setPixmap(QPixmap())
        # Do not terminate a loader in native HDF5/Fabio code. Its late result is ignored.
        if self._preview_thread is not None and self._preview_thread.isRunning():
            return
        self._pending_preview_source = None
        self._preview_thread = QThread(self)
        self._preview_worker = _PreviewWorker(request_id, source, self.view_model)
        self._preview_worker.moveToThread(self._preview_thread)
        self._preview_thread.started.connect(self._preview_worker.run)
        self._preview_worker.finished.connect(self._preview_ready)
        self._preview_worker.failed.connect(self._preview_failed)
        self._preview_worker.finished.connect(self._preview_thread.quit)
        self._preview_worker.failed.connect(self._preview_thread.quit)
        self._preview_thread.finished.connect(self._preview_cleanup)
        self._preview_thread.start()

    def _preview_ready(self, request_id: int, payload: list[dict]) -> None:
        if request_id != self._preview_request:
            return
        statistics = []
        for index, item in enumerate(payload):
            self.preview_labels[index].setText("")
            self.preview_labels[index].setPixmap(_array_pixmap(item["data"]))
            self.preview_captions[index].setText(f"{item['label']} · frame {item['frame']}")
            if index == 0:
                minimum = "n/a" if item["minimum"] is None else f"{item['minimum']:.6g}"
                maximum = "n/a" if item["maximum"] is None else f"{item['maximum']:.6g}"
                statistics = [
                    f"Image size: {item['shape'][1]} × {item['shape'][0]}",
                    f"Data type: {item['dtype']}",
                    f"Min / max: {minimum} / {maximum}",
                    f"NaN/invalid: {item['nan_count']:,}",
                    f"Negative: {item['negative_count']:,}",
                    f"Pixels at maximum (possible saturation): {item['max_count']:,}",
                ]
        self.preview_stats.setText("\n".join(statistics))

    def _preview_failed(self, request_id: int, message: str) -> None:
        if request_id == self._preview_request:
            self.preview_stats.setText(f"Preview unavailable: {message}")

    def _preview_cleanup(self) -> None:
        self._preview_worker = None
        if self._preview_thread is not None:
            self._preview_thread.deleteLater()
        self._preview_thread = None
        pending = self._pending_preview_source
        self._pending_preview_source = None
        if pending is not None and self.isVisible():
            self._start_preview(pending)

    def _configure_output_formats(self) -> None:
        visibility = self.view_model.output_format_visibility(
            container=self.container_check.isChecked(),
        )
        for name, button in self.format_buttons.items():
            button.setVisible(visibility[name])
        checked = self.format_group.checkedButton()
        if checked is None or not checked.isVisible():
            for button in self.format_buttons.values():
                if button.isVisible():
                    button.setChecked(True)
                    break
        self._update_output_preview()

    def _selected_format(self) -> str:
        button = self.format_group.checkedButton()
        return str(button.property("format_name")) if button is not None else "TIFF"

    def _choose_destination(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", self.destination_edit.text())
        if folder:
            self.destination_edit.setText(self.view_model.normalize_path(folder))

    def _container_toggled(self, checked: bool) -> None:
        if checked:
            self.format_buttons["HDF5"].setVisible(True)
            self.format_buttons["HDF5"].setChecked(True)
        self.naming_combo.setEnabled(not checked)
        self.add_suffix.setEnabled(not checked)
        if checked:
            self._update_output_preview()
        else:
            self._configure_output_formats()

    def _sidecar_toggled(self, checked: bool) -> None:
        self.single_json.setEnabled(checked)
        self.per_image_json.setEnabled(checked)

    def _preserve_values_toggled(self, checked: bool) -> None:
        if checked:
            self.data_mode.setCurrentIndex(0)
        elif self.data_mode.currentData() == "original":
            self.data_mode.setCurrentIndex(1)
        self._update_output_preview()

    def _update_output_preview(self) -> None:
        try:
            options = self._options()
            preview = self.view_model.output_preview(options)
            self.naming_example.setText(f"Example: {preview.example}")
            self.output_summary.setText(
                f"Estimated output: {preview.image_count:,} image(s) in "
                f"{preview.file_count:,} file(s), approximately "
                f"{_human_bytes(preview.estimated_bytes)}"
            )
            self.dtype_warning.setText(preview.dtype_warning)
        except Exception:
            self.output_summary.setText("")
            self.dtype_warning.setText("")

    def _options(self):
        return self.view_model.make_options(
            output_format=self._selected_format(),
            destination=self.destination_edit.text().strip(),
            naming_template=self.naming_combo.currentText().strip() or "{source}_{frame:06d}",
            add_suffix=self.add_suffix.isChecked(),
            preserve_values=self.preserve_values.isChecked(),
            data_mode=str(self.data_mode.currentData()),
            preserve_metadata=self.preserve_metadata.isChecked(),
            write_sidecar=self.write_sidecar.isChecked(),
            single_metadata_file=self.single_json.isChecked(),
            container=self.container_check.isChecked(),
        )

    def _back(self) -> None:
        self.stack.setCurrentIndex(max(0, self.stack.currentIndex() - 1))
        self._update_step_header()

    def _next(self) -> None:
        index = self.stack.currentIndex()
        if index == 0:
            if not self.sources:
                QMessageBox.information(self, "Format Converter", "Add at least one input file.")
                return
            self._refresh_selection_table()
            self.stack.setCurrentIndex(1)
        elif index == 1:
            if not any(source.included and source.selected_frames for source in self.sources):
                QMessageBox.information(self, "Format Converter", "Select at least one image or frame to convert.")
                return
            self._configure_output_formats()
            self.stack.setCurrentIndex(2)
        else:
            self._review_and_convert()
            return
        self._update_step_header()

    def _update_step_header(self) -> None:
        current = self.stack.currentIndex()
        for index, label in enumerate(self.step_labels):
            if index == current:
                label.setStyleSheet("background: #2563eb; color: white; border-radius: 5px; font-weight: 600;")
            elif index < current:
                label.setStyleSheet("background: #dbeafe; color: #1d4ed8; border-radius: 5px;")
            else:
                label.setStyleSheet("background: #f1f5f9; color: #475569; border-radius: 5px;")
        self.back_button.setEnabled(current > 0)
        self.next_button.setText("Review & Convert" if current == 2 else "Next")

    def _review_and_convert(self) -> None:
        options = self._options()
        if not options.destination:
            QMessageBox.warning(self, "Output settings", "Choose an output destination.")
            return
        try:
            review = self.view_model.conversion_review(options)
        except Exception as exc:
            QMessageBox.warning(self, "Output settings", f"Invalid output settings:\n{exc}")
            return
        text = (
            f"Input:\n{review.input_summary}\n"
            f"{review.image_count:,} selected image(s) / frame(s)\n\n"
            f"Output:\n{options.output_format}\n"
            f"Destination: {review.destination}\nNaming: {review.naming}\n\n"
            f"Estimated output:\n{review.output_files:,} file(s), approximately "
            f"{_human_bytes(review.estimated_bytes)}"
        )
        if review.is_large_output:
            text += "\n\n⚠ Large output: conversion can take considerable time and disk space."
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Confirm conversion")
        dialog.setIcon(QMessageBox.Information)
        dialog.setText(text)
        back = dialog.addButton("Back", QMessageBox.RejectRole)
        convert = dialog.addButton("Convert", QMessageBox.AcceptRole)
        dialog.addButton("Cancel", QMessageBox.DestructiveRole)
        dialog.setDefaultButton(convert)
        dialog.exec_()
        if dialog.clickedButton() == convert:
            self._start_conversion(options)
        elif dialog.clickedButton() == back:
            return

    def _start_conversion(self, options) -> None:
        destination = self.view_model.normalize_path(options.destination)
        self._progress_dialog = ConversionProgressDialog(destination, self.parent())
        self._conversion_thread = QThread(self)
        self._conversion_worker = _ConversionWorker(options, self.view_model)
        self._conversion_worker.moveToThread(self._conversion_thread)
        self._conversion_thread.started.connect(self._conversion_worker.run)
        self._conversion_worker.progress.connect(self._conversion_progress)
        self._conversion_worker.finished.connect(self._conversion_finished)
        self._conversion_worker.failed.connect(self._conversion_failed)
        self._conversion_worker.finished.connect(self._conversion_thread.quit)
        self._conversion_worker.failed.connect(self._conversion_thread.quit)
        self._conversion_thread.finished.connect(self._conversion_cleanup)
        self._progress_dialog.pause_button.clicked.connect(self._toggle_pause)
        self._progress_dialog.cancel_button.clicked.connect(self._cancel_conversion)
        self._conversion_started_at = time.time()
        self._progress_dialog.show()
        self.hide()
        self._conversion_thread.start()

    def _conversion_progress(self, completed: int, total: int, source_name: str, frame_index: int) -> None:
        if self._progress_dialog is None:
            return
        total = max(1, total)
        self._progress_dialog.bar.setRange(0, total)
        self._progress_dialog.bar.setValue(completed)
        self._progress_dialog.title.setText(f"Converting {source_name}")
        self._progress_dialog.detail.setText(f"Frame {frame_index + 1} · {completed} / {total}")
        self._progress_dialog.job_status.set_state(
            "running",
            f"{source_name} · frame {frame_index + 1} · {completed} / {total}",
            progress=completed / total,
        )
        elapsed = time.time() - self._conversion_started_at
        remaining = (elapsed / completed * (total - completed)) if completed else 0
        remaining_text = _duration(remaining) if completed else "calculating…"
        self._progress_dialog.time_label.setText(f"Elapsed: {_duration(elapsed)}    Remaining: approximately {remaining_text}")

    def _toggle_pause(self) -> None:
        if self._conversion_worker is None or self._progress_dialog is None:
            return
        self._paused = not self._paused
        self._conversion_worker.set_paused(self._paused)
        self._progress_dialog.pause_button.setText("Resume" if self._paused else "Pause")
        if self._paused:
            self._progress_dialog.title.setText("Conversion paused")
            self._progress_dialog.job_status.set_state(
                "paused",
                "Conversion paused",
                progress=self._progress_dialog.bar.value()
                / max(1, self._progress_dialog.bar.maximum()),
            )
        else:
            self._progress_dialog.job_status.set_state(
                "running",
                "Conversion resumed",
                progress=self._progress_dialog.bar.value()
                / max(1, self._progress_dialog.bar.maximum()),
            )

    def _cancel_conversion(self) -> None:
        if self._conversion_worker is not None:
            self._conversion_worker.cancel()
        if self._progress_dialog is not None:
            self._progress_dialog.title.setText("Cancelling after the current image…")
            self._progress_dialog.job_status.set_state(
                "running",
                "Cancelling after the current image…",
                progress=self._progress_dialog.bar.value()
                / max(1, self._progress_dialog.bar.maximum()),
            )
            self._progress_dialog.cancel_button.setEnabled(False)

    def _conversion_finished(self, report) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.complete(report)

    def _conversion_failed(self, message: str) -> None:
        if self._progress_dialog is not None:
            self._progress_dialog.fail(message)

    def _conversion_cleanup(self) -> None:
        self._conversion_worker = None
        if self._conversion_thread is not None:
            self._conversion_thread.deleteLater()
        self._conversion_thread = None
        self.close()

    def closeEvent(self, event) -> None:
        if self._conversion_thread is not None and self._conversion_thread.isRunning():
            event.ignore()
            return
        # A native file read cannot be killed safely. Keep this dialog alive until it returns.
        if self._preview_thread is not None and self._preview_thread.isRunning():
            self.hide()
            self._preview_thread.finished.connect(self.close)
            event.ignore()
            return
        event.accept()
