"""Three-step, auto-detecting detector image format converter."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt5.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from calibration.image_loader import load_detector_image
from ui.app_assets import app_icon
from utils.format_converter import (
    ConversionEngine,
    ConversionOptions,
    InputSource,
    estimate_output,
    inspect_source,
    parse_custom_frames,
    scan_folder,
    select_dataset,
)
from utils.path_utils import normalize_path


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

    def __init__(self, request_id: int, source: InputSource):
        super().__init__()
        self.request_id = request_id
        self.source = source

    def run(self) -> None:
        try:
            frames = self.source.selected_frames or [0]
            picks = [frames[0], frames[len(frames) // 2], frames[-1]]
            payload = []
            for label, frame in zip(("First", "Middle", "Last"), picks):
                image = load_detector_image(
                    self.source.path,
                    frame_idx=frame,
                    dataset_path=self.source.dataset_path,
                )
                data = np.asarray(image.data)
                finite = data[np.isfinite(data)]
                minimum = float(np.min(finite)) if finite.size else None
                maximum = float(np.max(finite)) if finite.size else None
                max_count = int(np.count_nonzero(data == maximum)) if maximum is not None else 0
                payload.append({
                    "label": label,
                    "frame": frame + 1,
                    "data": data,
                    "shape": tuple(data.shape),
                    "dtype": str(data.dtype),
                    "minimum": minimum,
                    "maximum": maximum,
                    "nan_count": int(np.count_nonzero(~np.isfinite(data))),
                    "negative_count": int(np.count_nonzero(np.isfinite(data) & (data < 0))),
                    "max_count": max_count,
                })
            self.finished.emit(self.request_id, payload)
        except Exception as exc:
            self.failed.emit(self.request_id, str(exc))


class _ConversionWorker(QObject):
    progress = pyqtSignal(int, int, str, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, sources: list[InputSource], options: ConversionOptions):
        super().__init__()
        self.sources = sources
        self.engine = ConversionEngine(options)

    def run(self) -> None:
        try:
            report = self.engine.run(self.sources, self.progress.emit)
            self.finished.emit(report)
        except Exception as exc:
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.engine.cancel()

    def set_paused(self, paused: bool) -> None:
        self.engine.set_paused(paused)


class FolderImportDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Add Folder")
        self.setWindowIcon(app_icon())
        self.resize(560, 230)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(self)
        browse = QPushButton("Browse…", self)
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(browse)
        form.addRow("Folder:", path_row)
        include_row = QHBoxLayout()
        self.cbf = QCheckBox("CBF", self)
        self.tiff = QCheckBox("TIFF", self)
        self.nxs = QCheckBox("NXS", self)
        for check in (self.cbf, self.tiff, self.nxs):
            check.setChecked(True)
            include_row.addWidget(check)
        include_row.addStretch(1)
        form.addRow("Include:", include_row)
        self.recursive = QCheckBox("Include subfolders", self)
        self.recursive.setChecked(False)
        self.recursive.setToolTip("Disabled by default to avoid accidentally loading very large folder trees.")
        form.addRow("", self.recursive)
        layout.addLayout(form)
        note = QLabel("Subfolders are not scanned unless you explicitly enable the option.", self)
        note.setStyleSheet("color: #64748b;")
        layout.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(buttons)
        browse.clicked.connect(self._browse)
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select input folder", self.path_edit.text())
        if folder:
            self.path_edit.setText(normalize_path(folder))

    def _accept_if_valid(self) -> None:
        if not Path(self.path_edit.text().strip()).is_dir():
            QMessageBox.warning(self, "Add Folder", "Please select a valid folder.")
            return
        if not any(check.isChecked() for check in (self.cbf, self.tiff, self.nxs)):
            QMessageBox.warning(self, "Add Folder", "Select at least one input format.")
            return
        self.accept()


class ConversionProgressDialog(QDialog):
    def __init__(self, destination: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.destination = destination
        self.report_path = ""
        self.running = True
        self.setWindowTitle("Format Converter")
        self.setWindowIcon(app_icon())
        self.setModal(False)
        self.setMinimumWidth(570)
        layout = QVBoxLayout(self)
        self.title = QLabel("Preparing conversion…", self)
        self.title.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.detail = QLabel("", self)
        self.bar = QProgressBar(self)
        self.time_label = QLabel("Elapsed: 00:00:00", self)
        self.time_label.setStyleSheet("color: #64748b;")
        self.result = QLabel("", self)
        self.result.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.detail)
        layout.addWidget(self.bar)
        layout.addWidget(self.time_label)
        layout.addWidget(self.result)
        row = QHBoxLayout()
        self.pause_button = QPushButton("Pause", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.open_button = QPushButton("Open output folder", self)
        self.report_button = QPushButton("View report", self)
        self.close_button = QPushButton("Close", self)
        self.open_button.hide()
        self.report_button.hide()
        self.close_button.hide()
        row.addWidget(self.pause_button)
        row.addWidget(self.cancel_button)
        row.addStretch(1)
        row.addWidget(self.open_button)
        row.addWidget(self.report_button)
        row.addWidget(self.close_button)
        layout.addLayout(row)
        self.open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.destination)))
        self.report_button.clicked.connect(self._open_report)
        self.close_button.clicked.connect(self.accept)

    def complete(self, report) -> None:
        self.running = False
        self.report_path = report.report_path
        self.bar.setValue(self.bar.maximum())
        if report.cancelled:
            self.title.setText("Conversion cancelled")
        else:
            self.title.setText("Conversion completed")
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


class FormatConverterDialog(QDialog):
    """Full converter. Inputs are detected; there is no single/batch mode."""

    def __init__(self, parent: QWidget | None = None, current_file: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Format Converter")
        self.setWindowIcon(app_icon())
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.resize(1080, 760)
        self.setMinimumSize(920, 650)
        self.sources: list[InputSource] = []
        self.current_file = current_file
        self._preview_thread: Optional[QThread] = None
        self._preview_worker: Optional[_PreviewWorker] = None
        self._preview_request = 0
        self._pending_preview_source: Optional[InputSource] = None
        self._conversion_thread: Optional[QThread] = None
        self._conversion_worker: Optional[_ConversionWorker] = None
        self._progress_dialog: Optional[ConversionProgressDialog] = None
        self._conversion_started_at = 0.0
        self._paused = False
        self._build_ui()
        if current_file and Path(current_file).suffix.lower() in {".nxs", ".cbf", ".tif", ".tiff"}:
            self.add_paths([current_file])

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 16)
        self.step_header = QHBoxLayout()
        self.step_labels = []
        for number, text in ((1, "Choose input"), (2, "Select images / frames"), (3, "Output settings")):
            label = QLabel(f"{number}  {text}", self)
            label.setAlignment(Qt.AlignCenter)
            label.setMinimumHeight(38)
            self.step_header.addWidget(label)
            self.step_labels.append(label)
        outer.addLayout(self.step_header)
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self._build_input_page())
        self.stack.addWidget(self._build_selection_page())
        self.stack.addWidget(self._build_output_page())
        outer.addWidget(self.stack, 1)
        nav = QHBoxLayout()
        self.back_button = QPushButton("Back", self)
        self.next_button = QPushButton("Next", self)
        self.cancel_button = QPushButton("Cancel", self)
        nav.addWidget(self.back_button)
        nav.addStretch(1)
        nav.addWidget(self.cancel_button)
        nav.addWidget(self.next_button)
        outer.addLayout(nav)
        self.back_button.clicked.connect(self._back)
        self.next_button.clicked.connect(self._next)
        self.cancel_button.clicked.connect(self.close)
        self._update_step_header()

    def _build_input_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        intro = QLabel("Add any combination of NXS, CBF, and TIFF files. GIMaP detects single images and multi-frame inputs automatically.", page)
        intro.setWordWrap(True)
        layout.addWidget(intro)
        actions = QHBoxLayout()
        add_files = QPushButton("Add files", page)
        add_folder = QPushButton("Add folder", page)
        self.current_button = QPushButton("Use currently opened file", page)
        self.current_button.setEnabled(bool(self.current_file))
        actions.addWidget(add_files)
        actions.addWidget(add_folder)
        actions.addWidget(self.current_button)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.input_tree = QTreeWidget(page)
        self.input_tree.setColumnCount(5)
        self.input_tree.setHeaderLabels(("Source", "Type", "Images / frames", "Selection", "Status"))
        self.input_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.input_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            self.input_tree.header().setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(self.input_tree, 1)
        dataset_row = QHBoxLayout()
        self.dataset_label = QLabel("Dataset:", page)
        self.dataset_combo = QComboBox(page)
        self.dataset_note = QLabel("Recommended dataset is selected automatically.", page)
        self.dataset_note.setStyleSheet("color: #64748b;")
        dataset_row.addWidget(self.dataset_label)
        dataset_row.addWidget(self.dataset_combo, 1)
        dataset_row.addWidget(self.dataset_note)
        layout.addLayout(dataset_row)
        self.dataset_label.hide()
        self.dataset_combo.hide()
        self.dataset_note.hide()
        self.input_note = QLabel("No input files yet.", page)
        self.input_note.setStyleSheet("color: #64748b;")
        layout.addWidget(self.input_note)
        add_files.clicked.connect(self._choose_files)
        add_folder.clicked.connect(self._choose_folder)
        self.current_button.clicked.connect(lambda: self.add_paths([self.current_file]))
        self.input_tree.currentItemChanged.connect(self._input_current_changed)
        self.dataset_combo.currentTextChanged.connect(self._dataset_changed)
        return page

    def _build_selection_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        tools = QHBoxLayout()
        select_all = QPushButton("Select all", page)
        select_none = QPushButton("Select none", page)
        remove = QPushButton("Remove selected", page)
        sort_button = QPushButton("Sort by filename", page)
        self.filter_edit = QLineEdit(page)
        self.filter_edit.setPlaceholderText("Filter by filename…")
        tools.addWidget(select_all)
        tools.addWidget(select_none)
        tools.addWidget(remove)
        tools.addWidget(sort_button)
        tools.addStretch(1)
        tools.addWidget(self.filter_edit, 1)
        layout.addLayout(tools)
        splitter = QSplitter(Qt.Horizontal, page)
        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        self.selection_table = QTableWidget(0, 5, left)
        self.selection_table.setHorizontalHeaderLabels(("Use", "Source", "Type", "Images / frames", "Selection"))
        self.selection_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.selection_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.selection_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.selection_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        left_layout.addWidget(self.selection_table, 1)
        frame_group = QGroupBox("Frame selection for selected NXS", left)
        frame_layout = QGridLayout(frame_group)
        self.frame_mode = QComboBox(frame_group)
        self.frame_mode.addItems(("All", "Current frame", "Frame range", "Custom", "Every Nth frame"))
        self.range_start = QSpinBox(frame_group)
        self.range_end = QSpinBox(frame_group)
        self.custom_frames = QLineEdit("1, 5, 8–20", frame_group)
        self.nth_frame = QSpinBox(frame_group)
        self.nth_frame.setRange(1, 1_000_000)
        self.nth_frame.setValue(10)
        self.apply_frames = QPushButton("Apply selection", frame_group)
        frame_layout.addWidget(QLabel("Mode:"), 0, 0)
        frame_layout.addWidget(self.frame_mode, 0, 1, 1, 3)
        frame_layout.addWidget(QLabel("Range:"), 1, 0)
        frame_layout.addWidget(self.range_start, 1, 1)
        frame_layout.addWidget(QLabel("to"), 1, 2)
        frame_layout.addWidget(self.range_end, 1, 3)
        frame_layout.addWidget(QLabel("Custom:"), 2, 0)
        frame_layout.addWidget(self.custom_frames, 2, 1, 1, 3)
        frame_layout.addWidget(QLabel("Every N:"), 3, 0)
        frame_layout.addWidget(self.nth_frame, 3, 1)
        frame_layout.addWidget(self.apply_frames, 3, 3)
        left_layout.addWidget(frame_group)
        right_scroll = QScrollArea(splitter)
        right_scroll.setWidgetResizable(True)
        preview = QWidget(right_scroll)
        preview_layout = QVBoxLayout(preview)
        preview_layout.addWidget(QLabel("Preview", preview))
        self.preview_labels = []
        self.preview_captions = []
        for title in ("First", "Middle", "Last"):
            caption = QLabel(title, preview)
            caption.setStyleSheet("font-weight: 600;")
            image = QLabel("No preview", preview)
            image.setAlignment(Qt.AlignCenter)
            image.setMinimumSize(220, 150)
            image.setFrameShape(QFrame.StyledPanel)
            preview_layout.addWidget(caption)
            preview_layout.addWidget(image)
            self.preview_captions.append(caption)
            self.preview_labels.append(image)
        self.preview_stats = QLabel("Select an input to inspect its frames.", preview)
        self.preview_stats.setWordWrap(True)
        self.preview_stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
        preview_layout.addWidget(self.preview_stats)
        preview_layout.addStretch(1)
        right_scroll.setWidget(preview)
        splitter.addWidget(left)
        splitter.addWidget(right_scroll)
        splitter.setSizes((650, 340))
        layout.addWidget(splitter, 1)
        select_all.clicked.connect(lambda: self._set_all_included(True))
        select_none.clicked.connect(lambda: self._set_all_included(False))
        remove.clicked.connect(self._remove_selected)
        sort_button.clicked.connect(self._sort_sources)
        self.filter_edit.textChanged.connect(self._filter_sources)
        self.selection_table.itemChanged.connect(self._include_changed)
        self.selection_table.itemSelectionChanged.connect(self._selection_current_changed)
        self.frame_mode.currentTextChanged.connect(self._update_frame_editor)
        self.apply_frames.clicked.connect(self._apply_frame_selection)
        self._update_frame_editor()
        return page

    def _build_output_page(self) -> QWidget:
        page = QWidget(self)
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        layout = QVBoxLayout(content)
        format_group = QGroupBox("Output format", content)
        format_layout = QHBoxLayout(format_group)
        self.format_group = QButtonGroup(format_group)
        self.format_buttons = {}
        for text in ("TIFF", "CBF", "HDF5", "NumPy"):
            label = "NumPy (.npy)" if text == "NumPy" else text
            button = QRadioButton(label, format_group)
            button.setProperty("format_name", text)
            self.format_group.addButton(button)
            self.format_buttons[text] = button
            format_layout.addWidget(button)
        format_layout.addStretch(1)
        self.format_buttons["TIFF"].setChecked(True)
        layout.addWidget(format_group)
        destination_group = QGroupBox("Destination and naming", content)
        form = QFormLayout(destination_group)
        destination_row = QHBoxLayout()
        self.destination_edit = QLineEdit(str(Path.cwd() / "converted"), destination_group)
        destination_button = QPushButton("Browse…", destination_group)
        destination_row.addWidget(self.destination_edit, 1)
        destination_row.addWidget(destination_button)
        form.addRow("Destination:", destination_row)
        self.naming_combo = QComboBox(destination_group)
        self.naming_combo.setEditable(True)
        self.naming_combo.addItems(("{source}_{frame:06d}", "{source}_img_{frame:06d}"))
        form.addRow("Multi-frame template:", self.naming_combo)
        self.naming_example = QLabel("Example: scan_001_000123.tif", destination_group)
        self.naming_example.setStyleSheet("color: #475569;")
        form.addRow("", self.naming_example)
        self.add_suffix = QCheckBox("Add suffix automatically when names collide", destination_group)
        self.add_suffix.setChecked(True)
        form.addRow("", self.add_suffix)
        layout.addWidget(destination_group)
        values_group = QGroupBox("Pixel values and data type", content)
        values_layout = QVBoxLayout(values_group)
        self.preserve_values = QCheckBox("Preserve original values and data type when supported", values_group)
        self.preserve_values.setChecked(True)
        self.data_mode = QComboBox(values_group)
        self.data_mode.addItem("Preserve / use loader data type", "original")
        self.data_mode.addItem("Save as 32-bit float", "float32")
        self.data_mode.addItem("Convert to uint16 using the original data range", "scale_uint16")
        self.data_mode.addItem("Clip to the uint16 range", "clip_uint16")
        values_layout.addWidget(self.preserve_values)
        values_layout.addWidget(self.data_mode)
        self.dtype_warning = QLabel("", values_group)
        self.dtype_warning.setWordWrap(True)
        self.dtype_warning.setStyleSheet("color: #a16207;")
        values_layout.addWidget(self.dtype_warning)
        layout.addWidget(values_group)
        metadata_group = QGroupBox("Metadata", content)
        metadata_layout = QVBoxLayout(metadata_group)
        self.preserve_metadata = QCheckBox("Preserve metadata where supported", metadata_group)
        self.preserve_metadata.setChecked(True)
        self.write_sidecar = QCheckBox("Write metadata sidecar JSON", metadata_group)
        self.write_sidecar.setChecked(True)
        self.single_json = QRadioButton("One metadata file for the whole conversion", metadata_group)
        self.per_image_json = QRadioButton("One JSON file beside each output image", metadata_group)
        self.single_json.setChecked(True)
        metadata_layout.addWidget(self.preserve_metadata)
        metadata_layout.addWidget(self.write_sidecar)
        metadata_layout.addWidget(self.single_json)
        metadata_layout.addWidget(self.per_image_json)
        layout.addWidget(metadata_group)
        advanced = QGroupBox("Advanced", content)
        advanced_layout = QVBoxLayout(advanced)
        self.container_check = QCheckBox("Export as one NeXus/HDF5 container", advanced)
        self.container_check.setToolTip("Stores all selected images as compressed datasets in converted_images.h5.")
        advanced_layout.addWidget(self.container_check)
        layout.addWidget(advanced)
        self.output_summary = QLabel("", content)
        self.output_summary.setWordWrap(True)
        layout.addWidget(self.output_summary)
        layout.addStretch(1)
        scroll.setWidget(content)
        page_layout = QVBoxLayout(page)
        page_layout.addWidget(scroll)
        destination_button.clicked.connect(self._choose_destination)
        self.naming_combo.currentTextChanged.connect(self._update_output_preview)
        self.format_group.buttonClicked.connect(self._update_output_preview)
        self.destination_edit.textChanged.connect(self._update_output_preview)
        self.container_check.toggled.connect(self._container_toggled)
        self.write_sidecar.toggled.connect(self._sidecar_toggled)
        self.data_mode.currentIndexChanged.connect(self._update_output_preview)
        self.preserve_values.toggled.connect(self._preserve_values_toggled)
        return page

    def add_paths(self, paths: list[str]) -> None:
        existing = {os.path.normcase(source.path) for source in self.sources}
        added = 0
        errors = []
        for raw_path in paths:
            if not raw_path:
                continue
            normalized = normalize_path(raw_path)
            key = os.path.normcase(str(Path(normalized).resolve()))
            if key in existing:
                continue
            try:
                source = inspect_source(normalized)
                self.sources.append(source)
                existing.add(key)
                added += 1
            except Exception as exc:
                errors.append(f"{Path(normalized).name}: {exc}")
        self._refresh_input_tree()
        self._refresh_selection_table()
        if errors:
            QMessageBox.warning(self, "Some inputs could not be added", "\n".join(errors[:12]))
        elif not added and paths:
            self.input_note.setText("The selected inputs are already in the task list.")

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add detector images", "", INPUT_FILTER)
        if paths:
            self.add_paths(paths)

    def _choose_folder(self) -> None:
        dialog = FolderImportDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        paths = scan_folder(
            dialog.path_edit.text(),
            include_cbf=dialog.cbf.isChecked(),
            include_tiff=dialog.tiff.isChecked(),
            include_nxs=dialog.nxs.isChecked(),
            recursive=dialog.recursive.isChecked(),
        )
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
            select_dataset(self.sources[index], dataset_path)
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
        for source in self.sources:
            source.included = included
        self._refresh_selection_table()

    def _include_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        index = item.data(Qt.UserRole)
        if isinstance(index, int) and index < len(self.sources):
            self.sources[index].included = item.checkState() == Qt.Checked

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
        self.sources = [source for index, source in enumerate(self.sources) if index not in selected]
        self._refresh_input_tree()
        self._refresh_selection_table()

    def _sort_sources(self) -> None:
        self.sources.sort(key=lambda source: source.name.lower())
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
        mode = self.frame_mode.currentText()
        try:
            for index in indices:
                source = self.sources[index]
                if source.file_type != "NXS" or source.frame_count <= 1:
                    source.selected_frames = [0]
                    continue
                if mode == "All":
                    frames = list(range(source.frame_count))
                elif mode == "Current frame":
                    current = 1
                    if os.path.normcase(source.path) == os.path.normcase(self.current_file or ""):
                        waxs = getattr(getattr(self.parent(), "components", None), "waxs_page", None)
                        current = getattr(getattr(waxs, "frame_spin", None), "value", lambda: 1)()
                    frames = [max(0, min(source.frame_count - 1, int(current) - 1))]
                elif mode == "Frame range":
                    frames = list(range(self.range_start.value() - 1, self.range_end.value()))
                    if not frames:
                        raise ValueError("The frame range is empty.")
                elif mode == "Custom":
                    frames = parse_custom_frames(self.custom_frames.text(), source.frame_count)
                else:
                    frames = list(range(0, source.frame_count, self.nth_frame.value()))
                source.selected_frames = frames
            self._refresh_input_tree()
            self._refresh_selection_table()
            self._start_preview(self.sources[indices[0]])
        except ValueError as exc:
            QMessageBox.warning(self, "Frame selection", str(exc))

    def _start_preview(self, source: InputSource) -> None:
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
        self._preview_worker = _PreviewWorker(request_id, source)
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
        input_types = {source.file_type for source in self.sources if source.included}
        mapping = {"TIFF": "TIFF", "CBF": "CBF", "NXS": "HDF5"}
        for name, button in self.format_buttons.items():
            hide = len(input_types) == 1 and any(mapping.get(input_type) == name for input_type in input_types)
            # Keep HDF5 available for mixed/container tasks, where it is not a no-op.
            if name == "HDF5" and (len(input_types) > 1 or self.container_check.isChecked()):
                hide = False
            button.setVisible(not hide)
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
            self.destination_edit.setText(normalize_path(folder))

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
        suffix = {"TIFF": ".tif", "CBF": ".cbf", "HDF5": ".h5", "NumPy": ".npy"}[self._selected_format()]
        template = self.naming_combo.currentText().strip() or "{source}_{frame:06d}"
        try:
            example = template.format(source="scan_001", frame=123, img="img") + suffix
        except Exception:
            example = "Invalid naming template"
        if self.container_check.isChecked():
            example = "converted_images.h5"
        self.naming_example.setText(f"Example: {example}")
        try:
            options = self._options()
            count, size = estimate_output(self.sources, options)
            files = 1 if self.container_check.isChecked() and count else count
            self.output_summary.setText(
                f"Estimated output: {count:,} image(s) in {files:,} file(s), approximately {_human_bytes(size)}"
            )
        except Exception:
            self.output_summary.setText("")
        output_format = self._selected_format()
        if output_format == "CBF" and self.data_mode.currentData() in ("original", "float32"):
            self.dtype_warning.setText("CBF encoders may not preserve NaN values or every floating-point representation. A metadata sidecar is recommended.")
        else:
            self.dtype_warning.setText("")

    def _options(self) -> ConversionOptions:
        return ConversionOptions(
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
            Path(options.destination).expanduser().resolve()
            # Validate template before confirmation.
            options.naming_template.format(source="scan_001", frame=123, img="img")
        except Exception as exc:
            QMessageBox.warning(self, "Output settings", f"Invalid output settings:\n{exc}")
            return
        selected_sources = [source for source in self.sources if source.included]
        frame_count, estimated_bytes = estimate_output(selected_sources, options)
        type_counts = {}
        for source in selected_sources:
            type_counts[source.file_type] = type_counts.get(source.file_type, 0) + 1
        input_summary = ", ".join(f"{count} {kind} file(s)" for kind, count in sorted(type_counts.items()))
        naming = "converted_images.h5" if options.container else options.naming_template + options.suffix
        output_files = 1 if options.container and frame_count else frame_count
        text = (
            f"Input:\n{input_summary}\n{frame_count:,} selected image(s) / frame(s)\n\n"
            f"Output:\n{options.output_format}\nDestination: {Path(options.destination).resolve()}\nNaming: {naming}\n\n"
            f"Estimated output:\n{output_files:,} file(s), approximately {_human_bytes(estimated_bytes)}"
        )
        if frame_count > 10_000 or estimated_bytes > 20 * 1024**3:
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

    def _start_conversion(self, options: ConversionOptions) -> None:
        self._progress_dialog = ConversionProgressDialog(str(Path(options.destination).resolve()), self.parent())
        self._conversion_thread = QThread(self)
        self._conversion_worker = _ConversionWorker(self.sources, options)
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

    def _cancel_conversion(self) -> None:
        if self._conversion_worker is not None:
            self._conversion_worker.cancel()
        if self._progress_dialog is not None:
            self._progress_dialog.title.setText("Cancelling after the current image…")
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
