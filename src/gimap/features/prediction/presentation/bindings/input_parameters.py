"""Input Parameters coordination for Prediction."""

from __future__ import annotations

import os


from pathlib import Path

from typing import Dict, List, Optional, Tuple


from PyQt5.QtWidgets import (
    QFileDialog,
    QMessageBox,
)


from src.gimap.shared.file_paths import normalize_path


class InputParametersMixin:
    """Own input parameters presentation behavior."""

    def _set_default_parameters(self) -> None:
        self.current_parameters = {
            "framework": "tensorflow 2.15.0",
            "mode": "single_file",
            "input_file": "",
            "input_folder": "",
            "export_path": "",
            "stack_value": "1",
            "range_value": "",
            "showing_value": "",
            "auto_scale": True,
            "vmin": None,
            "vmax": None,
            "predict_auto_scale": True,
            "predict_vmin": None,
            "predict_vmax": None,
            "colormap": self._DEFAULT_COLORMAPS[0],
            "gisaxs_log_scale": False,
            "predict_log_scale": False,
            "predict_curve_logx": False,
            "predict_curve_logy": False,
            "predict_curve_autoscale": True,
            "predict_curve_xmin": None,
            "predict_curve_xmax": None,
            "predict_curve_ymin": None,
            "predict_curve_ymax": None,
            # module selection
            "module_name": "",
            "module_model_path": "",
        }

    def _persist_parameters(self) -> None:
        if self._synchronizing:
            return
        self._synchronizing = True
        try:
            self.prediction_view_model.save_settings(dict(self.current_parameters))
            self.parameters_changed.emit(dict(self.current_parameters))
        finally:
            self._synchronizing = False

    def get_parameters(self) -> Dict[str, object]:
        return dict(self.current_parameters)

    def set_parameters(self, parameters: Dict[str, object]) -> None:
        if not parameters:
            return
        self.current_parameters.update(parameters)
        if self._initialized:
            self._initialize_ui()

    def _choose_gisaxs_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self.main_window, "Select GISAXS Folder", "")
        if not folder:
            return
        folder = normalize_path(folder)
        self.current_parameters["input_folder"] = folder
        self._set_line_edit("gisaxsPredictChooseFolderValue", folder)
        self._scan_directory_for_cbf(folder)
        self._persist_parameters()
        self._refresh_predict_readiness()

    def _choose_gisaxs_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self.main_window,
            "Select GISAXS File",
            self.current_parameters.get("input_folder", ""),
            "GISAXS Files (*.cbf);;All Files (*)",
        )
        if file_path:
            file_path = normalize_path(file_path)
            self._handle_new_file_selection(file_path)

    def _choose_export_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self.main_window,
            "Select Export Folder",
            self.current_parameters.get("export_path", ""),
        )
        if not folder:
            return
        folder = normalize_path(folder)
        self.current_parameters["export_path"] = folder
        self._set_line_edit("gisaxsPredictExportFolderValue", folder)
        self._persist_parameters()
        self._append_status_message(f"Export folder selected: {folder}")

    def _prompt_export_folder(self, title: str = "Select Export Folder") -> str:
        folder = QFileDialog.getExistingDirectory(
            self.main_window,
            title,
            self.current_parameters.get("export_path", "") or "",
        )
        if not folder:
            return ""
        folder = normalize_path(folder)
        self.current_parameters["export_path"] = folder
        self._set_line_edit("gisaxsPredictExportFolderValue", folder)
        self._persist_parameters()
        return folder

    def _handle_file_line_edit_committed(self) -> None:
        widget = getattr(self.ui, "gisaxsPredictChooseGisaxsFileValue", None)
        if not widget:
            return
        text = normalize_path(widget.text())
        if not text:
            return
        if os.path.isabs(text) and os.path.exists(text):
            self._handle_new_file_selection(text)
            return

        folder = normalize_path(self.current_parameters.get("input_folder", ""))
        candidate = normalize_path(os.path.join(folder, text) if folder else text)
        if os.path.exists(candidate):
            self._handle_new_file_selection(candidate)
            return
        self._append_status_message(f"Unable to locate file: {text}", level="WARN")
        QMessageBox.warning(self.main_window, "File Not Found", f"Unable to locate file: {text}")

    def _on_import_images_clicked(self) -> None:
        # Behaves like pressing Enter in the file input: try to load the typed file
        self._sync_pending_text_fields()
        self._handle_file_line_edit_committed()

    def _handle_new_file_selection(self, file_path: str) -> None:
        file_path = normalize_path(file_path)
        if not os.path.exists(file_path):
            QMessageBox.warning(self.main_window, "File Not Found", file_path)
            return
        if not file_path.lower().endswith(".cbf"):
            QMessageBox.warning(
                self.main_window, "Unsupported Format", "Only CBF files are supported."
            )
            return

        folder = os.path.dirname(file_path)
        base_name = os.path.basename(file_path)
        index = self._extract_index(base_name)

        self.current_parameters.update(
            {
                "input_file": file_path,
                "input_folder": folder,
                "showing_value": str(index or ""),
            }
        )

        self._set_line_edit("gisaxsPredictChooseGisaxsFileValue", base_name)
        self._set_line_edit("gisaxsPredictChooseFolderValue", folder)
        self._set_line_edit(
            "gisaxsImageShowingValue", self.current_parameters.get("showing_value", "")
        )

        self._scan_directory_for_cbf(folder)
        if index is not None:
            self._current_file_index = index
        elif self._available_indices:
            self._current_file_index = self._available_indices[0]

        mode = self.current_parameters.get("mode", "single_file")
        if mode == "multi_files":
            typed_range = self._get_line_edit_text("gisaxsPredictStackValue")
            if typed_range.strip():
                self.current_parameters["range_value"] = typed_range
                self._set_line_edit("gisaxsPredictStackValue", typed_range)
            else:
                default_range = f"{self._current_file_index}-{self._current_file_index}"
                self.current_parameters["range_value"] = default_range
                self._set_line_edit("gisaxsPredictStackValue", default_range)
        else:
            stack_text = self._get_line_edit_text(
                "gisaxsPredictStackValue"
            ) or self.current_parameters.get("stack_value", "1")
            self.current_parameters["stack_value"] = stack_text or "1"
            self._set_line_edit("gisaxsPredictStackValue", stack_text or "1")

        self._update_range_tooltip()
        self._persist_parameters()
        self._trigger_data_reload()
        self._refresh_predict_readiness()

    def _scan_directory_for_cbf(self, folder: str) -> None:
        folder = normalize_path(folder)
        entries: List[Tuple[str, int]] = []
        index_to_file: Dict[int, str] = {}

        try:
            discovered = self.prediction_view_model.files.discover_numbered_files(
                Path(folder), ".cbf"
            )
            for item in discovered:
                entries.append((item.path.name, item.index))
                index_to_file[item.index] = str(item.path)

            if not entries:
                self._append_status_message(
                    "No numbered CBF files detected in the current folder", level="WARN"
                )

            self._folder_entries = entries
            self._index_to_file = index_to_file
            self._available_indices = sorted(index_to_file.keys())
            self._update_range_tooltip()
        except Exception as exc:
            self._append_status_message(f"Failed to scan folder: {exc}", level="ERROR")

    def _extract_index(self, file_name: str) -> Optional[int]:
        return self.prediction_view_model.files.file_index(file_name)

    def _update_range_tooltip(self) -> None:
        if not self._available_indices:
            tooltip = "No valid indices detected yet"
        else:
            tooltip = f"Available index range: {self._available_indices[0]} - {self._available_indices[-1]}"

        label = getattr(self.ui, "gisaxsPredictStackLabel", None)
        line_edit = getattr(self.ui, "gisaxsPredictStackValue", None)
        if label:
            label.setToolTip(tooltip)
        if line_edit:
            line_edit.setToolTip(tooltip)

    def _on_mode_changed(self) -> None:
        if self._ui_updating:
            return
        single_btn = getattr(self.ui, "gisaxsPredictSingleFileRadioButton", None)
        if single_btn is not None and single_btn.isChecked():
            self.current_parameters["mode"] = "single_file"
        else:
            self.current_parameters["mode"] = "multi_files"

        self._update_mode_controls(self.current_parameters["mode"])
        self._persist_parameters()
        self._refresh_predict_readiness()

    def _update_mode_controls(self, mode: str) -> None:
        label = getattr(self.ui, "gisaxsPredictStackLabel", None)
        stack_edit = getattr(self.ui, "gisaxsPredictStackValue", None)
        showing = getattr(self.ui, "gisaxsImageShowingValue", None)
        every_label = getattr(self.ui, "gisaxsPredictEveryLabel", None)
        every_value = getattr(self.ui, "gisaxsPredictEveryValue", None)

        if label:
            label.setText("Range:" if mode == "multi_files" else "Stack:")
        if stack_edit:
            text = (
                self.current_parameters.get("range_value", "")
                if mode == "multi_files"
                else self.current_parameters.get("stack_value", "1")
            )
            self._set_line_edit(
                "gisaxsPredictStackValue", text or ("1" if mode == "single_file" else "")
            )
        if showing:
            showing.setEnabled(mode == "multi_files")
        # Only show the "Every" controls in multi-file mode
        if every_label:
            every_label.setVisible(mode == "multi_files")
        if every_value:
            every_value.setVisible(mode == "multi_files")

        # 显示/隐藏多文件结果列表
        if self._multifile_results_widget:
            self._multifile_results_widget.setVisible(mode == "multi_files")

        # 在多文件模式下调整布局
        self._adjust_predict_layout_for_mode(mode)

    def _sync_pending_text_fields(self) -> None:
        """Apply user-typed values without triggering loads."""
        mode = self.current_parameters.get("mode", "single_file")
        stack_text = self._get_line_edit_text("gisaxsPredictStackValue")
        if mode == "multi_files":
            if stack_text.strip():
                self.current_parameters["range_value"] = stack_text
        else:
            if stack_text.strip():
                self.current_parameters["stack_value"] = stack_text

        if mode == "multi_files":
            showing_text = self._get_line_edit_text("gisaxsImageShowingValue")
            if showing_text.strip():
                self.current_parameters["showing_value"] = showing_text

    def _on_stack_field_committed(self) -> None:
        if self._ui_updating:
            return
        mode = self.current_parameters.get("mode", "single_file")
        text = self._get_line_edit_text("gisaxsPredictStackValue")
        if mode == "multi_files":
            self.current_parameters["range_value"] = text
            self._persist_parameters()
            self._trigger_data_reload()
            return

        try:
            count = max(1, int(text or "1"))
        except ValueError:
            count = 1
        self.current_parameters["stack_value"] = str(count)
        self._set_line_edit("gisaxsPredictStackValue", str(count))
        self._persist_parameters()
        self._trigger_data_reload()

    def _on_showing_value_committed(self) -> None:
        if self._ui_updating:
            return
        mode = self.current_parameters.get("mode", "single_file")
        if mode != "multi_files":
            return
        text = self._get_line_edit_text("gisaxsImageShowingValue")
        try:
            index = int(text)
        except ValueError:
            self._append_status_message("Showing Value must be a valid index", level="WARN")
            return
        if index not in self._index_to_file:
            self._append_status_message("Index is outside the available range", level="WARN")
            return
        self.current_parameters["showing_value"] = str(index)
        self._persist_parameters()
        self._start_image_loading(
            self._index_to_file[index], 1, {"mode": "multi_files", "index": index}
        )

    def _parse_range_text(self, text: str) -> List[int]:
        return self.prediction_view_model.files.index_range(text)

    def _trigger_data_reload(self) -> None:
        if self.current_parameters.get("mode", "single_file") == "single_file":
            self._load_single_stack()
        else:
            self._load_multi_sequence()

    def _load_single_stack(self) -> None:
        file_path = self.current_parameters.get("input_file")
        if not file_path:
            return
        try:
            stack = max(1, int(self.current_parameters.get("stack_value", "1")))
        except ValueError:
            stack = 1
            self.current_parameters["stack_value"] = "1"
        self._start_image_loading(file_path, stack, {"mode": "single_file", "stack": stack})

    def _load_multi_sequence(self) -> None:
        if not self._index_to_file:
            if self.current_parameters.get("input_folder"):
                self._scan_directory_for_cbf(self.current_parameters["input_folder"])
            if not self._index_to_file:
                return

        range_text = self.current_parameters.get("range_value") or self._get_line_edit_text(
            "gisaxsPredictStackValue"
        )
        indices = [idx for idx in self._parse_range_text(range_text) if idx in self._index_to_file]
        if not indices:
            if self._available_indices:
                indices = [self._available_indices[0]]
            else:
                self._append_status_message("No Multi File indices available", level="WARN")
                return

        self._sequence_indices = indices
        first = indices[0]
        self.current_parameters["range_value"] = range_text
        self.current_parameters["showing_value"] = str(first)
        self._set_line_edit("gisaxsImageShowingValue", str(first))
        self._persist_parameters()
        self._start_image_loading(
            self._index_to_file[first], 1, {"mode": "multi_files", "index": first}
        )
        self._refresh_predict_readiness()

    def _input_ready(self) -> bool:
        mode = self.current_parameters.get("mode", "single_file")
        if mode == "single_file":
            file_path = self.current_parameters.get("input_file")
            return bool(file_path and os.path.exists(file_path))
        folder = self.current_parameters.get("input_folder")
        if not folder or not os.path.isdir(folder):
            return False
        range_text = self.current_parameters.get("range_value") or self._get_line_edit_text(
            "gisaxsPredictStackValue"
        )
        return bool(range_text.strip() or self._available_indices or self._folder_entries)

    def _model_ready(self) -> bool:
        return self._current_model is not None and not self._model_loading

    def _refresh_predict_readiness(self) -> None:
        if not hasattr(self, "ui"):
            return
        input_ready = self._input_ready()
        model_ready = self._model_ready()
        framework_ready = self._framework_ready()
        mode = self.current_parameters.get("mode", "single_file")

        labels = {
            "gisaxsPredictInputReadyLabel": (
                "Input: Ready" if input_ready else "Input: Missing",
                input_ready,
            ),
            "gisaxsPredictModelReadyLabel": (
                "Model: Loaded" if model_ready else "Model: Not loaded",
                model_ready,
            ),
            "gisaxsPredictFrameworkReadyLabel": (
                "Framework: OK" if framework_ready else "Framework: Missing/Incompatible",
                framework_ready,
            ),
            "gisaxsPredictModeLabel": (
                f"Mode: {'Multi Files' if mode == 'multi_files' else 'Single File'}",
                True,
            ),
        }
        for name, (text, ok) in labels.items():
            label = getattr(self.ui, name, None)
            if label is not None:
                label.setText(text)
                label.setStyleSheet("color: #166534;" if ok else "color: #b91c1c;")

        btn = getattr(self.ui, "gisaxsPredictPredictButton", None)
        if btn is not None and not self._multifile_prediction_active:
            btn.setEnabled(input_ready and model_ready and framework_ready)
        stop_btn = getattr(self.ui, "gisaxsPredictStopButton", None)
        if stop_btn is not None:
            stop_btn.setEnabled(bool(self._multifile_prediction_active))

        for export_name in ("gisaxsImageExportButton", "predict2dExportButton"):
            export_btn = getattr(self.ui, export_name, None)
            if export_btn is not None:
                export_btn.setEnabled(bool(self.prediction_results))
