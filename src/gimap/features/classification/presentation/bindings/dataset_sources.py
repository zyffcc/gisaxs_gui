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
    QMenu,
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
        for source in self.sources.values():
            source_samples = [
                sample for sample in self.samples if sample.source_name == source.label
            ]
            loaded = [sample for sample in source_samples if sample.load_status == "loaded"]
            failed = [sample for sample in source_samples if sample.load_status == "failed"]
            shapes = sorted({sample.raw_shape for sample in loaded if sample.raw_shape})
            summary = {
                "files": len(source_samples),
                "loaded": len(loaded),
                "failed": len(failed),
                "data_type": "/".join(sorted({sample.data_type for sample in source_samples})) or "-",
                "shape": ", ".join(str(shape) for shape in shapes[:3]) or "-",
                "status": "Error" if failed else ("Ready" if loaded else "Empty"),
            }
            card = self._create_dataset_card(source, summary)
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
            (
                "Labels",
                "Accepted from source"
                if source.label_mode == "accepted"
                else (
                    "Suggested from source"
                    if source.label_mode == "provisional"
                    else "Unlabeled"
                ),
            ),
            ("Path", self._short_paths(source.paths)),
        ]
        for row, (name, value) in enumerate(detail_items):
            details.addWidget(QLabel(str(name), card), row, 0)
            value_label = QLabel(str(value), card)
            value_label.setWordWrap(True)
            details.addWidget(value_label, row, 1)
        layout.addLayout(details)

        buttons = QGridLayout()
        buttons.setHorizontalSpacing(6)
        buttons.setVerticalSpacing(6)
        choose_folder = QPushButton("Folder…", card)
        choose_files = QPushButton("Files…", card)
        edit = QPushButton("Edit", card)
        remove = QPushButton("Remove", card)
        rescan = QPushButton("Rescan", card)
        choose_folder.clicked.connect(lambda: self._choose_source_folder(source.label))
        choose_files.clicked.connect(lambda: self._choose_source_files(source.label))
        edit.clicked.connect(lambda: self._edit_source_dialog(source.label))
        remove.clicked.connect(lambda: self._remove_source(source.label))
        rescan.clicked.connect(lambda: self._start_import([source.label]))
        buttons.addWidget(choose_folder, 0, 0)
        buttons.addWidget(choose_files, 0, 1)
        buttons.addWidget(rescan, 0, 2)
        buttons.addWidget(edit, 1, 0, 1, 2)
        buttons.addWidget(remove, 1, 2)
        layout.addLayout(buttons)
        return card

    def _add_class_dialog(self) -> None:
        source = self._source_dialog(default_mode="accepted")
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
        self.sources.pop(label, None)
        for sample in self.samples:
            if sample.source_name != label:
                continue
            sample.source_name = updated.label
            if sample.label_source not in {"folder", "source"}:
                continue
            sample.suggested_label = None
            sample.suggestion_source = None
            if updated.label_mode == "accepted":
                sample.label = updated.label
                sample.label_status = "accepted"
            elif updated.label_mode == "provisional":
                sample.label = ""
                sample.label_status = "provisional"
                sample.suggested_label = updated.label
                sample.suggestion_source = "source"
            else:
                sample.label = ""
                sample.label_status = "unlabeled"
        self.sources[updated.label] = updated
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _source_dialog(
        self,
        source: Optional[DatasetSource] = None,
        *,
        default_mode: str = "accepted",
    ) -> Optional[DatasetSource]:
        page = self.page
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Dataset Class")
        layout = QFormLayout(dialog)
        name_edit = QLineEdit(dialog)
        type_combo = QComboBox(dialog)
        type_combo.addItems(["Folder", "Files"])
        label_mode_combo = QComboBox(dialog)
        label_mode_combo.addItem("Use source name as accepted label", "accepted")
        label_mode_combo.addItem("Keep samples unlabeled", "unlabeled")
        label_mode_combo.addItem("Use source name as suggestion", "provisional")
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
            index = label_mode_combo.findData(source.label_mode)
            label_mode_combo.setCurrentIndex(max(0, index))
        else:
            index = label_mode_combo.findData(default_mode)
            label_mode_combo.setCurrentIndex(max(0, index))
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
        layout.addRow("Source name", name_edit)
        layout.addRow("Label handling", label_mode_combo)
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
                self.main_window, "Dataset Source", "Source name and path are required."
            )
            return None
        label = self._unique_label(label, existing=source.label if source else None)
        return DatasetSource(
            label=label,
            source_type="files" if type_combo.currentText() == "Files" else "folder",
            paths=paths,
            file_pattern=pattern_edit.text().strip() or "*",
            color=selected_color["value"],
            label_mode=str(label_mode_combo.currentData()),
        )

    def _add_unlabeled_data_menu(self) -> None:
        menu = QMenu(self.page.addDataButton)
        files_action = menu.addAction("Add files")
        folder_action = menu.addAction("Add folder")
        action = menu.exec_(
            self.page.addDataButton.mapToGlobal(
                self.page.addDataButton.rect().bottomLeft()
            )
        )
        paths: list[str] = []
        source_type = "files"
        if action == files_action:
            paths, _ = QFileDialog.getOpenFileNames(
                self.main_window,
                "Add unlabeled data",
                "",
                self._file_dialog_filter(),
            )
        elif action == folder_action:
            folder = QFileDialog.getExistingDirectory(
                self.main_window, "Add unlabeled data folder"
            )
            if folder:
                paths = [folder]
                source_type = "folder"
        if not paths:
            return
        base = os.path.basename(paths[0].rstrip("/\\")) or "Unlabeled data"
        if source_type == "files":
            base = os.path.basename(os.path.dirname(paths[0])) or "Unlabeled data"
        name = self._unique_label(base)
        self.sources[name] = DatasetSource(
            label=name,
            source_type=source_type,
            paths=list(paths),
            color=self._next_color(len(self.sources)),
            label_mode="unlabeled",
        )
        self._refresh_everything()
        self._persist_parameters()
        self._start_import([name])

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
        self.samples = [sample for sample in self.samples if sample.source_name != label]
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()

    def _on_files_dropped(self, paths: list[str]) -> None:
        use_folder_labels = False
        if any(os.path.isdir(path) for path in paths):
            use_folder_labels = (
                QMessageBox.question(
                    self.main_window,
                    "Folder labels",
                    "Use each dropped folder name as an accepted training label?\n\n"
                    "Choose No to keep the samples unlabeled for exploration.",
                )
                == QMessageBox.Yes
            )
        affected: list[str] = []
        for path in paths:
            if os.path.isdir(path):
                label = self._unique_label(
                    os.path.basename(path.rstrip("/\\")) or "Data"
                )
                self.sources[label] = DatasetSource(
                    label=label,
                    source_type="folder",
                    paths=[path],
                    color=self._next_color(len(self.sources)),
                    label_mode="accepted" if use_folder_labels else "unlabeled",
                )
                affected.append(label)
            elif os.path.isfile(path):
                folder = os.path.dirname(path)
                base_label = os.path.basename(folder) or "Unlabeled files"
                label = next(
                    (
                        name
                        for name, source in self.sources.items()
                        if source.label_mode == "unlabeled"
                        and source.source_type == "files"
                        and source.label.startswith(base_label)
                    ),
                    self._unique_label(base_label),
                )
                if label not in self.sources:
                    self.sources[label] = DatasetSource(
                        label=label,
                        source_type="files",
                        paths=[],
                        color=self._next_color(len(self.sources)),
                        label_mode="unlabeled",
                    )
                if label not in affected:
                    affected.append(label)
                self.sources[label].paths.append(path)
        self._mark_results_outdated()
        self._refresh_everything()
        self._persist_parameters()
        if affected:
            self._start_import(affected)

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
        for sample in new_samples:
            override = self._label_overrides.get(sample.file_path)
            if override is None:
                continue
            sample.label = override.get("label", "")
            sample.label_status = override.get("label_status", "unlabeled")
            sample.label_source = override.get("label_source", "session")
        if labels:
            label_set = set(labels)
            self.samples = [
                sample for sample in self.samples if sample.source_name not in label_set
            ] + list(new_samples)
        else:
            self.samples = list(new_samples)
        self.summary = self.classification_view_model.validate_dataset(
            self.samples, require_labels=False, allow_mixed=True
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
