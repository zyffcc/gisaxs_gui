"""Multifile Results coordination for Prediction."""

from __future__ import annotations

import os

import datetime


from pathlib import Path

from typing import List, Optional

import numpy as np


from PyQt5.QtGui import QImage

from PyQt5.QtWidgets import (
    QMessageBox,
)


from src.gimap.features.prediction.application import (
    PredictionExportItem,
)


from src.gimap.features.prediction.presentation.multifile_results import (
    PredictResult,
    PredictStatus,
)


class MultifileResultsMixin:
    """Own multifile results presentation behavior."""

    def _on_multifile_result_selected(self, result: PredictResult) -> None:
        """多文件结果选中处理 - 双击显示单文件结果"""
        if result.status != PredictStatus.COMPLETED or not result.prediction_data:
            # 如果结果还未完成，只更新当前文件显示
            self._update_current_file_display(
                result.file_path.splitlines()[0], getattr(result, "stack_count", 1)
            )
            return

        try:
            # 更新当前文件显示
            self._update_current_file_display(
                result.file_path.splitlines()[0], getattr(result, "stack_count", 1)
            )

            # 获取预测结果数据
            prediction_data = result.prediction_data.get("prediction_data", {})

            if prediction_data:
                # 临时加载当前图像以支持预处理显示（按需计算）
                try:
                    temp_image = (
                        self._load_cbf_stack_sync(result.file_path.splitlines())
                        if "\n" in result.file_path
                        else self._load_cbf_file_sync(result.file_path)
                    )
                    if temp_image is not None:
                        # 临时设置当前图像用于预处理显示
                        old_current_image = self._current_image
                        self._current_image = temp_image

                        # 使用标准显示方法（会实时计算预处理步骤）
                        self._display_prediction(prediction_data)

                        # 恢复原来的图像
                        self._current_image = old_current_image
                    else:
                        # 如果无法加载图像，仅显示预测结果（无预处理tab）
                        self._current_image = None
                        self._display_prediction(prediction_data)
                except Exception as e:
                    # 如果图像加载失败，仍然显示预测结果
                    self._append_status_message(
                        f"Could not load image for preprocessing display: {e}", level="WARN"
                    )
                    self._current_image = None
                    self._display_prediction(prediction_data)

                # 设置当前参数
                self.current_parameters["input_file"] = result.file_path.splitlines()[0]

                # 切换到Predict-2D tab
                self._set_predict_main_tab("result")

                # 更新状态
                self._append_status_message(
                    f"Displaying results for: {os.path.basename(result.file_path.splitlines()[0])}",
                    level="INFO",
                )
            else:
                self._append_status_message(
                    f"No prediction data available for: {os.path.basename(result.file_path.splitlines()[0])}",
                    level="WARN",
                )

        except Exception as e:
            self._append_status_message(f"Error displaying result: {e}", level="ERROR")

    def _on_multifile_export_requested(self, config: dict, results: List[PredictResult]) -> None:
        """多文件导出请求处理"""
        if not results:
            QMessageBox.information(self.main_window, "Export", "No results to export.")
            return

        export_path = self._prompt_export_folder("Save Multi-File Prediction Output To")
        if not export_path:
            return

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # 过滤只导出已完成的结果
            completed_results = [r for r in results if r.status == PredictStatus.COMPLETED]
            if not completed_results:
                QMessageBox.information(
                    self.main_window, "Export", "No completed results to export."
                )
                return

            # 导出JSONL格式
            if config.get("jsonl", False):
                self._export_results_jsonl(completed_results, export_path, timestamp)

            # 导出JPG图像
            if config.get("jpg", False):
                self._export_results_jpg(completed_results, export_path, timestamp)

            # 导出ASCII 1D曲线
            if config.get("ascii", False):
                self._export_results_ascii(completed_results, export_path, timestamp)

            self._append_status_message(f"Export completed to {export_path}", level="INFO")

        except Exception as e:
            QMessageBox.critical(self.main_window, "Export Error", f"Export failed: {e}")
            self._append_status_message(f"Export error: {e}", level="ERROR")

    def _on_multifile_prediction_started(self) -> None:
        """多文件预测开始"""
        self._multifile_prediction_active = True
        # 禁用Predict按钮
        btn = getattr(self.ui, "gisaxsPredictPredictButton", None)
        if btn:
            btn.setEnabled(False)
            btn.setText("Predicting...")
        stop_btn = getattr(self.ui, "gisaxsPredictStopButton", None)
        if stop_btn:
            stop_btn.setEnabled(True)
            stop_btn.setVisible(True)
        self._render_prediction_workflow()

    def _on_multifile_prediction_completed(self) -> None:
        """多文件预测完成"""
        self._multifile_prediction_active = False
        # 重新启用Predict按钮
        btn = getattr(self.ui, "gisaxsPredictPredictButton", None)
        if btn:
            btn.setEnabled(True)
            btn.setText("Predict")
        stop_btn = getattr(self.ui, "gisaxsPredictStopButton", None)
        if stop_btn:
            stop_btn.setEnabled(False)
            stop_btn.setVisible(False)

        self._append_status_message("Multi-file prediction completed!", level="INFO")
        self._render_prediction_workflow()

    def _on_multifile_result_updated(self, index: int, update_data: dict) -> None:
        """多文件预测结果更新"""
        if self._multifile_results_widget:
            self._multifile_results_widget.updatePredictResult(index, **update_data)
            if update_data.get("status") == PredictStatus.RUNNING:
                result = self._multifile_results_widget.table_model.getResult(index)
                if result is not None:
                    first = (
                        result.file_path.splitlines()[0] if result.file_path else result.file_name
                    )
                    stack_count = max(1, int(getattr(result, "stack_count", 1)))
                    self._append_status_message(
                        f"Running stack ({stack_count} file{'s' if stack_count != 1 else ''}): {os.path.basename(first)}",
                        level="INFO",
                    )

    def _on_multifile_progress_updated(self, completed: int, total: int) -> None:
        """多文件预测进度更新"""
        if self._multifile_results_widget:
            self._multifile_results_widget.updateProgress(completed, total)

        # 更新主进度条
        if total > 0:
            progress = int((completed / total) * 100)
            self.progress_updated.emit(progress)

    def _export_results_jsonl(
        self, results: List[PredictResult], export_path: str, timestamp: str
    ) -> None:
        """导出JSONL格式结果"""
        exported = self.prediction_view_model.export_jsonl(
            self._prediction_export_items(results), Path(export_path), timestamp
        )
        if exported is None:
            raise OSError(
                self.prediction_view_model.state.error_message
                or "Failed to export prediction JSONL"
            )

    def _result_confidence(self, result: PredictResult) -> Optional[float]:
        """Return confidence when older/newer prediction payloads provide it."""
        value = getattr(result, "confidence", None)
        if isinstance(value, (int, float)):
            return float(value)
        payload = result.prediction_data if isinstance(result.prediction_data, dict) else {}
        value = payload.get("confidence")
        if isinstance(value, (int, float)):
            return float(value)
        inner = payload.get("prediction_data")
        if isinstance(inner, dict):
            value = inner.get("confidence")
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _export_results_jpg(
        self, results: List[PredictResult], export_path: str, timestamp: str
    ) -> None:
        """导出JPG图像到文件夹"""
        jpg_folder = os.path.join(export_path, f"prediction_images_{timestamp}")
        os.makedirs(jpg_folder, exist_ok=True)

        for i, result in enumerate(results):
            if not result.prediction_data:
                continue

            # 导出2D结果图像
            prediction_data = result.prediction_data.get("prediction_data", {})
            hr_data = prediction_data.get("hr")

            if isinstance(hr_data, np.ndarray):
                # 创建图像
                image_path = os.path.join(jpg_folder, f"{result.file_name}_{i:04d}_hr.jpg")
                self._save_array_as_image(hr_data, image_path)

    def _export_results_ascii(
        self, results: List[PredictResult], export_path: str, timestamp: str
    ) -> None:
        """导出ASCII 1D曲线数据"""
        exported = self.prediction_view_model.export_ascii(
            self._prediction_export_items(results), Path(export_path), timestamp
        )
        if exported is None and any(result.prediction_data for result in results):
            error = self.prediction_view_model.state.error_message
            if error:
                raise OSError(error)

    def _prediction_export_items(
        self, results: List[PredictResult]
    ) -> tuple[PredictionExportItem, ...]:
        return tuple(
            PredictionExportItem(
                filename=result.file_name,
                filepath=result.file_path,
                stack_count=max(1, int(getattr(result, "stack_count", 1))),
                timestamp=result.start_time.isoformat() if result.start_time else None,
                processing_time=result.processing_time,
                confidence=self._result_confidence(result),
                prediction_data=result.prediction_data,
            )
            for result in results
        )

    def _load_cbf_stack_sync(self, file_paths: Optional[List[str]]) -> Optional[np.ndarray]:
        if not file_paths:
            return None
        loaded = self.prediction_view_model.load_paths(file_paths)
        if loaded is not None:
            return loaded.image
        self._append_status_message(
            self.prediction_view_model.state.error_message or "Failed to load this stack.",
            level="ERROR",
        )
        return None

    def _load_cbf_file_sync(self, file_path: str) -> Optional[np.ndarray]:
        """同步加载CBF文件"""
        loaded = self.prediction_view_model.load_paths((file_path,))
        if loaded is not None:
            return loaded.image
        self._append_status_message(
            self.prediction_view_model.state.error_message
            or f"Failed to load CBF file {file_path}",
            level="ERROR",
        )
        return None

    def _save_array_as_image(self, array: np.ndarray, image_path: str) -> None:
        """将数组保存为图像文件"""
        try:
            # 标准化数组到0-255范围
            if array.dtype != np.uint8:
                array_norm = (array - array.min()) / (array.max() - array.min()) * 255
                array = array_norm.astype(np.uint8)

            # 创建QImage并保存
            height, width = array.shape
            qimage = QImage(array.data, width, height, width, QImage.Format_Grayscale8)
            qimage.save(image_path, "JPEG", 90)

        except Exception as e:
            self._append_status_message(f"Failed to save image {image_path}: {e}", level="WARN")
