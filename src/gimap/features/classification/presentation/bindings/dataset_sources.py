"""Dataset Sources coordination for Classification."""

from __future__ import annotations


import os


from typing import Optional


from PyQt5.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.features.classification.application import (
    ClassificationPageState,
    DatasetSource,
)

from src.gimap.features.classification.presentation.workers import (
    ImportWorker,
)


class DatasetSourcesMixin:
    """Own dataset sources presentation behavior."""

    def _render_dataset_cards(self) -> None:
        page = self.page
        if page is None:
            return
        page.clear_dataset_cards()
        label_summary = self.classification_view_model.summarize_dataset(self.samples)
        for source in self.sources.values():
            card = self._create_dataset_card(source, label_summary.get(source.label, {}))
            page.add_dataset_card(card)

    def _create_dataset_card(self, source: DatasetSource, summary: dict[str, object]) -> QWidget:
        page = self.page
        card = QFrame(page)
        card.setProperty("classificationCard", True)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        top = QHBoxLayout()
        color = QLabel(card)
        color.setFixedSize(14, 14)
        color.setStyleSheet(f"background: {source.color}; border-radius: 7px;")
        title = QLabel(source.label, card)
        title.setStyleSheet("font-weight: 700;")
        status = QLabel(str(summary.get("status", "Empty")), card)
        status.setProperty("classificationBadge", True)
        top.addWidget(color)
        top.addWidget(title, 1)
        top.addWidget(status)
        layout.addLayout(top)

        details = QGridLayout()
        details.setHorizontalSpacing(8)
        detail_items = [
            ("Files", summary.get("files", 0)),
            ("Loaded", summary.get("loaded", 0)),
            ("Failed", summary.get("failed", 0)),
            ("Type", summary.get("data_type", "-")),
            ("Shape", summary.get("shape", "-")),
            ("Pattern", source.file_pattern or "*"),
            ("Path", self._short_paths(source.paths)),
        ]
        for row, (name, value) in enumerate(detail_items):
            details.addWidget(QLabel(str(name), card), row, 0)
            value_label = QLabel(str(value), card)
            value_label.setWordWrap(True)
            details.addWidget(value_label, row, 1)
        layout.addLayout(details)

        buttons = QHBoxLayout()
        choose_folder = QPushButton("Choose Folder", card)
        choose_files = QPushButton("Choose Files", card)
        edit = QPushButton("Edit", card)
        remove = QPushButton("Remove", card)
        rescan = QPushButton("Rescan", card)
        choose_folder.clicked.connect(lambda: self._choose_source_folder(source.label))
        choose_files.clicked.connect(lambda: self._choose_source_files(source.label))
        edit.clicked.connect(lambda: self._edit_source_dialog(source.label))
        remove.clicked.connect(lambda: self._remove_source(source.label))
        rescan.clicked.connect(lambda: self._start_import([source.label]))
        for button in (choose_folder, choose_files, edit, remove, rescan):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        return card

    def _add_class_dialog(self) -> None:
        source = self._source_dialog()
        if source is None:
            return
        self.sources[source.label] = source
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _edit_source_dialog(self, label: str) -> None:
        current = self.sources.get(label)
        if current is None:
            return
        updated = self._source_dialog(current)
        if updated is None:
            return
        if updated.label != label:
            self.sources.pop(label, None)
            for sample in self.samples:
                if sample.label == label:
                    sample.label = updated.label
        self.sources[updated.label] = updated
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _source_dialog(self, source: Optional[DatasetSource] = None) -> Optional[DatasetSource]:
        page = self.page
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Dataset Class")
        layout = QFormLayout(dialog)
        name_edit = QLineEdit(dialog)
        type_combo = QComboBox(dialog)
        type_combo.addItems(["Folder", "Files"])
        path_edit = QLineEdit(dialog)
        pattern_edit = QLineEdit(dialog)
        pattern_edit.setText("*")
        color_button = QPushButton("Color", dialog)
        selected_color = {"value": self._next_color(len(self.sources))}
        if source is not None:
            name_edit.setText(source.label)
            type_combo.setCurrentText("Files" if source.source_type == "files" else "Folder")
            path_edit.setText(";".join(source.paths))
            pattern_edit.setText(source.file_pattern or "*")
            selected_color["value"] = source.color
        color_button.setStyleSheet(f"background: {selected_color['value']};")

        def browse() -> None:
            if type_combo.currentText() == "Folder":
                folder = QFileDialog.getExistingDirectory(self.main_window, "Choose dataset folder")
                if folder:
                    path_edit.setText(folder)
                    if not name_edit.text().strip():
                        name_edit.setText(os.path.basename(folder.rstrip("/\\")))
            else:
                files, _ = QFileDialog.getOpenFileNames(
                    self.main_window,
                    "Choose dataset files",
                    "",
                    self._file_dialog_filter(),
                )
                if files:
                    path_edit.setText(";".join(files))
                    if not name_edit.text().strip():
                        name_edit.setText(os.path.basename(os.path.dirname(files[0])))

        def choose_color() -> None:
            color = QColorDialog.getColor(parent=dialog)
            if color.isValid():
                selected_color["value"] = color.name()
                color_button.setStyleSheet(f"background: {selected_color['value']};")

        browse_button = QPushButton("Browse", dialog)
        browse_button.clicked.connect(browse)
        color_button.clicked.connect(choose_color)
        path_row = QWidget(dialog)
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(path_edit, 1)
        path_layout.addWidget(browse_button)
        layout.addRow("Class name", name_edit)
        layout.addRow("Source type", type_combo)
        layout.addRow("Path", path_row)
        layout.addRow("File pattern", pattern_edit)
        layout.addRow("Color", color_button)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return None
        label = name_edit.text().strip()
        paths = [path for path in path_edit.text().split(";") if path.strip()]
        if not label or not paths:
            QMessageBox.warning(
                self.main_window, "Dataset Class", "Class name and path are required."
            )
            return None
        label = self._unique_label(label, existing=source.label if source else None)
        return DatasetSource(
            label=label,
            source_type="files" if type_combo.currentText() == "Files" else "folder",
            paths=paths,
            file_pattern=pattern_edit.text().strip() or "*",
            color=selected_color["value"],
        )

    def _choose_source_folder(self, label: str) -> None:
        folder = QFileDialog.getExistingDirectory(self.main_window, "Choose dataset folder")
        if not folder or label not in self.sources:
            return
        source = self.sources[label]
        source.source_type = "folder"
        source.paths = [folder]
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _choose_source_files(self, label: str) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self.main_window, "Choose dataset files", "", self._file_dialog_filter()
        )
        if not files or label not in self.sources:
            return
        source = self.sources[label]
        source.source_type = "files"
        source.paths = files
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _remove_source(self, label: str) -> None:
        if (
            QMessageBox.question(
                self.main_window, "Remove Class", f"Remove class '{label}' and its samples?"
            )
            != QMessageBox.Yes
        ):
            return
        self.sources.pop(label, None)
        self.samples = [sample for sample in self.samples if sample.label != label]
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _on_files_dropped(self, paths: list[str]) -> None:
        for path in paths:
            if os.path.isdir(path):
                label = self._unique_label(os.path.basename(path.rstrip("/\\")) or "Class")
                self.sources[label] = DatasetSource(
                    label=label,
                    source_type="folder",
                    paths=[path],
                    color=self._next_color(len(self.sources)),
                )
            elif os.path.isfile(path):
                folder = os.path.dirname(path)
                base_label = os.path.basename(folder) or "Files"
                label = base_label if base_label in self.sources else self._unique_label(base_label)
                if label not in self.sources:
                    self.sources[label] = DatasetSource(
                        label=label,
                        source_type="files",
                        paths=[],
                        color=self._next_color(len(self.sources)),
                    )
                self.sources[label].paths.append(path)
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _start_import(self, labels: Optional[list[str]] = None) -> None:
        if self.current_worker is not None:
            QMessageBox.information(
                self.main_window, "Classification", "A Classification task is already running."
            )
            return
        selected_sources = [
            self.sources[label]
            for label in labels or list(self.sources.keys())
            if label in self.sources
        ]
        if not selected_sources:
            QMessageBox.warning(
                self.main_window, "Classification", "Add at least one dataset class first."
            )
            return
        self._set_state(ClassificationPageState.IMPORTING)
        self.page.taskProgressBar.setValue(0)
        worker = ImportWorker(selected_sources, self.classification_view_model)
        self.current_worker = worker
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(lambda payload: self._on_import_finished(payload, labels))
        worker.signals.error.connect(self._on_worker_error)
        self.thread_pool.start(worker)

    def _on_import_finished(self, payload, labels: Optional[list[str]]) -> None:
        self.current_worker = None
        if not isinstance(payload, dict):
            self._on_worker_error("Import returned an invalid payload.")
            return
        new_samples = payload.get("samples", [])
        if labels:
            label_set = set(labels)
            self.samples = [
                sample for sample in self.samples if sample.label not in label_set
            ] + list(new_samples)
        else:
            self.samples = list(new_samples)
        self.summary = payload.get("summary") or self.classification_view_model.validate_dataset(
            self.samples
        )
        self._mark_results_outdated()
        self._refresh_everything()
        state = (
            ClassificationPageState.READY
            if self.summary.status == "Ready"
            else ClassificationPageState.SCANNED
        )
        self._set_state(state)
        self._persist_parameters()
        self.log(
            f"[Import] Loaded {self.summary.loaded_samples}/{self.summary.total_samples} files."
        )
