"""Multifile Setup coordination for Prediction."""

from __future__ import annotations

import os


from PyQt5.QtWidgets import (
    QMessageBox,
    QLabel,
)

from src.gimap.features.prediction.presentation.multifile_results import (
    MultiFilePredictResultsWidget,
    MultiFilePredictManager,
)


class MultifileSetupMixin:
    """Own multifile setup presentation behavior."""

    def _setup_multifile_ui(self) -> None:
        """Embed the existing feature-owned result widget in the main workspace."""
        try:
            if self._multifile_results_widget is not None:
                return
            host = getattr(self.ui, "predictionBatchResultsHost", None)
            host_layout = getattr(self.ui, "predictionBatchResultsHostLayout", None)
            if host is None or host_layout is None:
                raise RuntimeError("Prediction batch result host is unavailable")

            self._current_file_label = getattr(
                self.ui, "predictionBatchCurrentFileLabel", None
            )
            if self._current_file_label is None:
                self._current_file_label = QLabel("No file selected", host)
            self._multifile_results_widget = MultiFilePredictResultsWidget(parent=host)
            self._multifile_results_widget.setVisible(True)
            host_layout.addWidget(self._multifile_results_widget)

            # 连接信号
            self._multifile_results_widget.result_selected.connect(
                self._on_multifile_result_selected
            )
            self._multifile_results_widget.result_double_clicked.connect(
                self._on_multifile_result_selected
            )
            self._multifile_results_widget.export_requested.connect(
                self._on_multifile_export_requested
            )

            # 创建多文件管理器
            if self._multifile_manager is None:
                self._multifile_manager = MultiFilePredictManager(self)
                self._multifile_manager.prediction_started.connect(
                    self._on_multifile_prediction_started
                )
                self._multifile_manager.prediction_completed.connect(
                    self._on_multifile_prediction_completed
                )
                self._multifile_manager.result_updated.connect(self._on_multifile_result_updated)
                self._multifile_manager.progress_updated.connect(
                    self._on_multifile_progress_updated
                )

            self._multifile_window = None
            self._append_status_message("Batch results are ready in the workspace", level="INFO")

        except Exception as e:
            self._append_status_message(f"Failed to setup multi-file UI: {e}", level="ERROR")

    def _show_multifile_results_window(self) -> None:
        if self._multifile_results_widget is None:
            self._setup_multifile_ui()
        section = getattr(self.ui, "predictionBatchResultsSection", None)
        workbench = getattr(self.ui, "predictionWorkbenchLayout", None)
        if section is None or self._multifile_results_widget is None:
            QMessageBox.information(
                self.main_window,
                "Multi-File Results",
                "The batch results panel is not available yet.",
            )
            return
        section.show()
        self._multifile_results_widget.show()
        if workbench is not None:
            workbench.focus_batch_results()

    def _clear_multifile_results(self) -> None:
        """清空所有多文件结果"""
        if self._multifile_results_widget:
            self._multifile_results_widget.clear_all_results()

    def _export_all_results(self) -> None:
        """导出所有结果"""
        if self._multifile_results_widget:
            all_results = self._multifile_results_widget.get_all_results()
            if all_results:
                self._multifile_results_widget.onExportClicked()
            else:
                QMessageBox.information(self.main_window, "Export", "No results to export.")

    def _stop_gisaxs_predict(self) -> None:
        if not self._multifile_prediction_active:
            self._append_status_message("No active multi-file prediction to stop.", level="INFO")
            return
        if self._multifile_manager:
            self._multifile_manager.cancel_prediction()
            self._append_status_message(
                "Stopping multi-file prediction after the current file...", level="WARN"
            )
        stop_btn = getattr(self.ui, "gisaxsPredictStopButton", None)
        if stop_btn:
            stop_btn.setEnabled(False)

    def _adjust_predict_layout_for_mode(self, mode: str) -> None:
        """根据模式调整预测布局"""
        # 更新当前文件标签的可见性
        if hasattr(self, "_current_file_label"):
            if mode == "multi_files":
                self._current_file_label.setVisible(True)
                if (
                    not self._current_file_label.text()
                    or self._current_file_label.text() == "Current: No file selected"
                ):
                    self._current_file_label.setText("No file selected")
            else:
                self._current_file_label.setVisible(False)

    def _update_current_file_display(self, file_path: str, stack_count: int = 1) -> None:
        """更新当前文件显示"""
        if hasattr(self, "_current_file_label"):
            if file_path:
                file_name = os.path.basename(file_path)
                suffix = (
                    f" ({stack_count} files stacked)"
                    if stack_count and stack_count > 1
                    else " (1 file)"
                )
                self._current_file_label.setText(f"{file_name}{suffix}")
                self._current_file_label.setToolTip(file_path)
            else:
                self._current_file_label.setText("No file selected")
                self._current_file_label.setToolTip("")

    def _connect_line_edit(self, name: str, slot) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        widget.returnPressed.connect(slot)

    def _connect_double_spin(self, name: str, slot) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        widget.editingFinished.connect(slot)
