"""Prediction Execution coordination for Prediction."""

from __future__ import annotations

import os


from pathlib import Path

from typing import Dict, List, Optional


from PyQt5.QtWidgets import (
    QMessageBox,
)


class PredictionExecutionMixin:
    """Own prediction execution presentation behavior."""

    def _execute_prediction(self) -> None:
        self._update_parameters_from_ui()
        if not self._validate_parameters():
            return
        try:
            self.status_updated.emit("Starting GISAXS prediction...")
            self.progress_updated.emit(0)
            mode = self.current_parameters.get("mode", "single_file")
            if mode == "single_file":
                # Predict for the currently loaded image
                if self._current_image is None:
                    self._append_status_message("No image loaded for prediction", level="WARN")
                    return
                self.progress_updated.emit(10)
                inp = self._preprocess_for_module(self._current_image)
                if inp is None:
                    self._append_status_message("Preprocessing failed", level="ERROR")
                    return
                self.progress_updated.emit(40)
                outs = self._predict_with_current_model(inp)
                if not outs:
                    self._append_status_message("Prediction failed", level="ERROR")
                    self.progress_updated.emit(0)
                    return
                self.progress_updated.emit(70)
                self._display_prediction(outs)
                self.progress_updated.emit(100)
                self.status_updated.emit("GISAXS prediction finished!")
            else:
                # Multi-files: use new queue-based processing
                results = self._predict_multi_files()
                if results and results.get("processing_started"):
                    # 不需要等待完成，处理在后台进行
                    # progress和completion信号会由multifile_manager发出
                    pass
                else:
                    self.progress_updated.emit(0)
                    self.status_updated.emit("Failed to start multi-file prediction")
        except Exception as exc:  # pragma: no cover - runtime safety
            QMessageBox.critical(self.main_window, "Prediction Error", str(exc))
            self.status_updated.emit(f"GISAXS prediction error: {exc}")
            # 重置多文件预测状态
            if self._multifile_prediction_active:
                self._on_multifile_prediction_completed()

    def _update_parameters_from_ui(self) -> None:
        combo = getattr(self.ui, "gisaxsPredictFrameworkCombox", None)
        if combo is not None:
            self.current_parameters["framework"] = combo.currentText()
        export_edit = getattr(self.ui, "gisaxsPredictExportFolderValue", None)
        if export_edit is not None:
            text = export_edit.text().strip()
            if text:
                self.current_parameters["export_path"] = text

    def _validate_parameters(self) -> bool:
        mode = self.current_parameters.get("mode", "single_file")
        if mode == "single_file":
            file_path = self.current_parameters.get("input_file")
            if not file_path or not os.path.exists(file_path):
                QMessageBox.warning(
                    self.main_window, "Invalid Parameters", "Please select a valid input file"
                )
                return False
        else:
            folder = self.current_parameters.get("input_folder")
            if not folder or not os.path.exists(folder):
                QMessageBox.warning(
                    self.main_window, "Invalid Parameters", "Please select a valid folder"
                )
                return False
        if not self._framework_ready():
            QMessageBox.warning(
                self.main_window,
                "Framework",
                "The selected model requires a compatible installed framework.",
            )
            return False
        if not self._model_ready():
            QMessageBox.warning(
                self.main_window, "Model", "Please import a model before running prediction."
            )
            return False
        return True

    def _predict_single_file(self) -> Optional[Dict[str, object]]:
        file_path = self.current_parameters.get("input_file")
        if not file_path:
            return None
        self.status_updated.emit(f"Processing file: {os.path.basename(file_path)}")
        self.progress_updated.emit(25)
        results = {
            "file": file_path,
            "predictions": [],
            "confidence": 0.95,
            "processing_time": 1.5,
        }
        self.progress_updated.emit(75)
        return results

    def _predict_multi_files(self) -> Optional[Dict[str, object]]:
        """多文件预测 - 使用新的队列处理系统"""
        folder = self.current_parameters.get("input_folder")
        if not folder:
            self._append_status_message("No input folder selected", level="WARN")
            return None

        files = [
            str(path)
            for path in self.prediction_view_model.files.discover_files(
                Path(folder), (".cbf", ".tif", ".tiff")
            )
        ]
        if not files and self.prediction_view_model.state.error_message:
            self._append_status_message(
                f"Error scanning folder: {self.prediction_view_model.state.error_message}",
                level="ERROR",
            )
            return None

        if not files:
            self._append_status_message("No compatible image files found in folder", level="WARN")
            return None

        # 应用范围过滤
        range_text = self.current_parameters.get("range_value", "")
        if range_text:
            try:
                indices = self._parse_range_text(range_text)
                if indices:
                    self._scan_directory_for_cbf(folder)
                    missing = [idx for idx in indices if idx not in self._index_to_file]
                    files = [
                        self._index_to_file[idx] for idx in indices if idx in self._index_to_file
                    ]
                    if missing:
                        missing_text = ", ".join(f"{idx:05d}" for idx in missing[:10])
                        if len(missing) > 10:
                            missing_text += ", ..."
                        self._append_status_message(
                            f"Range skipped missing CBF indices: {missing_text}", level="WARN"
                        )
            except Exception as e:
                self._append_status_message(f"Error parsing range: {e}", level="WARN")

        if not files:
            self._append_status_message("No files selected by range", level="WARN")
            return None

        try:
            every = max(1, int(self._get_line_edit_text("gisaxsPredictEveryValue") or "1"))
        except ValueError:
            every = 1
            self._set_line_edit("gisaxsPredictEveryValue", "1")
            self._append_status_message("Every must be a positive integer; using 1.", level="WARN")

        if every > 1:
            batches = [
                list(batch)
                for batch in self.prediction_view_model.files.complete_batches(files, every)
            ]
            skipped = len(files) - (len(batches) * every)
            if skipped:
                self._append_status_message(
                    f"Skipped {skipped} trailing file(s) that do not make a full Every={every} stack.",
                    level="WARN",
                )
        else:
            batches = [[file_path] for file_path in files]
        self._multifile_batch_map = {batch[0]: batch for batch in batches if batch}
        files_to_process = list(self._multifile_batch_map.keys())
        if not files_to_process:
            self._append_status_message(
                "No complete multi-file stacks selected by range/every.", level="WARN"
            )
            return None
        if every > 1:
            self._append_status_message(
                f"Multi-file range grouped into {len(files_to_process)} batch(es), Every={every}.",
                level="INFO",
            )

        # 清空现有结果并添加新的待处理项目
        if self._multifile_results_widget:
            self._multifile_results_widget.clearResults()

            # 添加所有文件到结果列表
            for file_path in files_to_process:
                row = self._multifile_results_widget.addPredictResult(file_path)
                batch = self._multifile_batch_map.get(file_path, [])
                if len(batch) > 1:
                    result = self._multifile_results_widget.table_model.getResult(row)
                    if result is not None:
                        result.file_name = (
                            f"{os.path.basename(batch[0])} - {os.path.basename(batch[-1])}"
                        )
                        result.file_path = "\n".join(batch)
                        result.stack_count = len(batch)
                        self._multifile_results_widget.table_model.updateResult(row, result)
                        self._append_status_message(
                            f"Queued stack: {os.path.basename(batch[0])} - {os.path.basename(batch[-1])} ({len(batch)} files)",
                            level="INFO",
                        )
                elif batch:
                    result = self._multifile_results_widget.table_model.getResult(row)
                    if result is not None:
                        result.stack_count = 1
                        self._multifile_results_widget.table_model.updateResult(row, result)

        # 开始批量预测
        if self._multifile_manager:
            self._multifile_prediction_active = True
            self._show_multifile_results_window()
            self._multifile_manager.start_batch_prediction(
                files_to_process, self._predict_single_file_for_batch
            )

        # 立即返回，实际处理将在后台进行
        return {"folder": folder, "total_files": len(files_to_process), "processing_started": True}

    def _predict_single_file_for_batch(self, file_path: str) -> Dict[str, object]:
        """为批量处理执行单文件预测"""
        try:
            # 临时设置当前文件用于预测
            old_file = self.current_parameters.get("input_file", "")
            self.current_parameters["input_file"] = file_path
            batch = self._multifile_batch_map.get(file_path) or [file_path]
            if len(batch) > 1:
                self.status_updated.emit(
                    f"Predicting stack ({len(batch)} files): {os.path.basename(batch[0])} - {os.path.basename(batch[-1])}"
                )
            else:
                self.status_updated.emit(f"Predicting file: {os.path.basename(file_path)}")

            # 执行实际预测逻辑（这里需要调用真正的预测代码）
            result = self._execute_single_file_prediction(file_path, batch)

            # 恢复原来的文件设置
            self.current_parameters["input_file"] = old_file

            return result

        except Exception as e:
            # 恢复原来的文件设置
            if "old_file" in locals():
                self.current_parameters["input_file"] = old_file
            raise e

    def _execute_single_file_prediction(
        self, file_path: str, stack_files: Optional[List[str]] = None
    ) -> Dict[str, object]:
        """执行单个文件的预测逻辑 - 真正调用预测流程"""
        typed_module = (
            self._current_module.get("_prediction_module")
            if isinstance(self._current_module, dict)
            else None
        )
        model_path = str(self.current_parameters.get("module_model_path") or "")
        if typed_module is not None and model_path:
            paths = tuple(stack_files or [file_path])
            item = self.prediction_view_model.predict_file_batch(
                paths,
                typed_module,
                Path(model_path),
            )
            if item.status != "succeeded" or item.prediction is None:
                raise RuntimeError(item.error_message or "Prediction failed")
            return {
                "file": file_path,
                "stack_count": len(paths),
                "stack_files": list(paths),
                "prediction_data": dict(item.prediction.outputs),
            }
        try:
            # 保存原有参数和状态
            old_input_file = self.current_parameters.get("input_file", "")
            old_mode = self.current_parameters.get("mode", "single_file")
            old_current_image = self._current_image

            # 临时设置为单文件模式
            self.current_parameters["input_file"] = file_path
            self.current_parameters["mode"] = "single_file"

            # 加载图像（使用同步方法）
            image_data = (
                self._load_cbf_stack_sync(stack_files)
                if stack_files and len(stack_files) > 1
                else self._load_cbf_file_sync(file_path)
            )

            if image_data is None:
                raise Exception(f"Failed to load image: {file_path}")

            # 设置当前图像
            self._current_image = image_data

            # 执行真正的预测流程（与单文件相同）
            # 1. 预处理
            inp = self._preprocess_for_module(self._current_image)
            if inp is None:
                raise Exception("Preprocessing failed")

            # 2. 模型预测
            outs = self._predict_with_current_model(inp)
            if not outs:
                raise Exception("Prediction failed")

            # 恢复原有参数和状态
            self.current_parameters["input_file"] = old_input_file
            self.current_parameters["mode"] = old_mode
            self._current_image = old_current_image

            # 返回结果（只包含预测数据，预处理步骤按需计算）
            return {
                "file": file_path,
                "stack_count": len(stack_files) if stack_files else 1,
                "stack_files": list(stack_files) if stack_files else [file_path],
                "prediction_data": outs,  # 真正的预测结果
            }

        except Exception as e:
            # 确保恢复原有参数和状态
            if "old_input_file" in locals():
                self.current_parameters["input_file"] = old_input_file
            if "old_mode" in locals():
                self.current_parameters["mode"] = old_mode
            if "old_current_image" in locals():
                self._current_image = old_current_image
            raise e
