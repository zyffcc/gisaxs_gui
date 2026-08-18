"""Results Widget for multi-file prediction."""

from __future__ import annotations

import os

import datetime


from typing import List, Optional


from PyQt5.QtCore import pyqtSignal, Qt, QModelIndex


from PyQt5.QtWidgets import (
    QWidget,
    QHeaderView,
    QMenu,
    QAction,
    QDialog,
    QMessageBox,
    QAbstractItemView,
)

from src.gimap.app.presentation.responsive_layout import move_window_to_cursor_screen

from .views import (
    MultiFileResultsWidgetView,
)


from .trend_windows import (
    DistributionHeatmapWindow,
    ParameterTrendWindow,
)


from .export_dialog import (
    ExportDialog,
)


from .result_types import (
    PredictResult,
    PredictStatus,
)


from .result_models import (
    PredictResultsFilterModel,
    PredictResultsTableModel,
)


class MultiFilePredictResultsWidget(QWidget, MultiFileResultsWidgetView):
    """多文件预测结果Widget"""

    # 信号
    result_selected = pyqtSignal(PredictResult)
    result_double_clicked = pyqtSignal(PredictResult)  # 双击信号
    export_requested = pyqtSignal(dict, list)  # config, results

    def __init__(self, parent=None):
        super().__init__(parent)
        self._heatmap_window: Optional[DistributionHeatmapWindow] = None
        self._parameter_trend_window: Optional[ParameterTrendWindow] = None
        self.setupUi(self)
        self._populate_form_controls()

        # 数据模型
        self.table_model = PredictResultsTableModel()
        self.filter_model = PredictResultsFilterModel()
        self.filter_model.setSourceModel(self.table_model)

        self.table_view.setModel(self.filter_model)
        self.setupTable()

        # 连接信号
        self.connectSignals()

        # 初始状态
        self.setVisible(False)  # 初始隐藏，只在multifile模式显示

    def _populate_form_controls(self) -> None:
        """Populate enum-backed controls after the Python View is created."""
        self.status_filter.addItem("All Status", None)
        for status in PredictStatus:
            self.status_filter.addItem(status.value, status)
        self.sort_combo.addItem("File Name", 0)
        self.sort_combo.addItem("Stack Files", 1)
        self.sort_combo.addItem("Status", 2)
        self.sort_combo.addItem("Duration", 3)

    def setupTable(self) -> None:
        """设置表格"""
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(True)

        header = self.table_view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)

        # 启用右键菜单
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)

    def connectSignals(self) -> None:
        """连接信号"""
        # 过滤和排序
        self.status_filter.currentTextChanged.connect(self.onStatusFilterChanged)
        self.filename_filter.textChanged.connect(self.onFilenameFilterChanged)
        self.sort_combo.currentTextChanged.connect(self.onSortChanged)
        self.sort_order_btn.clicked.connect(self.onSortOrderToggled)

        # 表格
        self.table_view.clicked.connect(self.onTableClicked)
        self.table_view.doubleClicked.connect(self.onTableDoubleClicked)  # 添加双击处理
        self.table_view.customContextMenuRequested.connect(self.showContextMenu)

        # 按钮
        self.heatmap_btn.clicked.connect(self.showDistributionHeatmap)
        self.parameter_trend_btn.clicked.connect(self.showParameterTrend)
        self.export_btn.clicked.connect(self.onExportClicked)
        self.clear_btn.clicked.connect(self.onClearClicked)

    def addPredictResult(self, file_path: str) -> int:
        """添加新的预测结果项"""
        result = PredictResult(
            file_path=file_path, file_name=os.path.basename(file_path), status=PredictStatus.PENDING
        )
        self.table_model.addResult(result)
        self.updateStats()
        return len(self.table_model.results) - 1

    def updatePredictResult(self, index: int, **kwargs) -> None:
        """更新预测结果"""
        result = self.table_model.getResult(index)
        if result is None:
            return

        # 更新字段
        for key, value in kwargs.items():
            if hasattr(result, key):
                setattr(result, key, value)

        # 特殊处理状态变更
        if "status" in kwargs:
            if kwargs["status"] == PredictStatus.RUNNING and result.start_time is None:
                result.start_time = datetime.datetime.now()
            elif kwargs["status"] in [
                PredictStatus.COMPLETED,
                PredictStatus.FAILED,
                PredictStatus.CANCELLED,
            ]:
                if result.start_time and result.end_time is None:
                    result.end_time = datetime.datetime.now()
                    result.processing_time = (result.end_time - result.start_time).total_seconds()

        self.table_model.updateResult(index, result)
        self.updateStats()
        self.refreshDistributionHeatmap()
        self.refreshParameterTrend()

    def updateProgress(self, completed: int, total: int) -> None:
        """更新总体进度"""
        if total > 0:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(completed)
        else:
            self.progress_bar.setVisible(False)

    def updateStats(self) -> None:
        """更新统计信息"""
        results = self.table_model.getAllResults()
        if not results:
            self.stats_label.setText("Ready")
            return

        total = len(results)
        completed = sum(1 for r in results if r.status == PredictStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == PredictStatus.FAILED)
        running = sum(1 for r in results if r.status == PredictStatus.RUNNING)

        text = f"Total: {total}, Completed: {completed}, Failed: {failed}"
        if running > 0:
            text += f", Running: {running}"

        self.stats_label.setText(text)

    def clearResults(self) -> None:
        """清空所有结果"""
        self.table_model.clear()
        self.progress_bar.setVisible(False)
        self.updateStats()
        self.refreshDistributionHeatmap()
        self.refreshParameterTrend()

    def clear_all_results(self) -> None:
        """清空所有结果（别名方法）"""
        self.clearResults()

    def get_all_results(self) -> List[PredictResult]:
        """获取所有结果"""
        return self.table_model.getAllResults()

    def get_completed_results(self) -> List[PredictResult]:
        """获取所有已完成的结果"""
        return [
            result for result in self.get_all_results() if result.status == PredictStatus.COMPLETED
        ]

    def update_status_summary(self) -> None:
        """更新状态摘要（别名方法）"""
        self.updateStats()

    def getSelectedResults(self) -> List[PredictResult]:
        """获取选中的结果"""
        selected_results = []
        selection = self.table_view.selectionModel()
        if selection:
            for index in selection.selectedRows():
                source_index = self.filter_model.mapToSource(index)
                result = self.table_model.getResult(source_index.row())
                if result:
                    selected_results.append(result)
        return selected_results

    def getCurrentDisplayResults(self) -> List[PredictResult]:
        """获取当前显示的结果（经过过滤）"""
        results = []
        for row in range(self.filter_model.rowCount()):
            source_index = self.filter_model.mapToSource(self.filter_model.index(row, 0))
            result = self.table_model.getResult(source_index.row())
            if result:
                results.append(result)
        return results

    def showDistributionHeatmap(self) -> None:
        if self._heatmap_window is None:
            self._heatmap_window = DistributionHeatmapWindow(self, self.window())
            self._heatmap_window.finished.connect(
                lambda _result: setattr(self, "_heatmap_window", None)
            )
        self._heatmap_window.refresh_components_and_plot()
        if not self._heatmap_window.isVisible():
            move_window_to_cursor_screen(self._heatmap_window)
        self._heatmap_window.show()
        self._heatmap_window.raise_()
        self._heatmap_window.activateWindow()

    def refreshDistributionHeatmap(self) -> None:
        if self._heatmap_window is not None and self._heatmap_window.isVisible():
            self._heatmap_window.refresh_components_and_plot()

    def showParameterTrend(self) -> None:
        if self._parameter_trend_window is None:
            self._parameter_trend_window = ParameterTrendWindow(self, self.window())
            self._parameter_trend_window.finished.connect(
                lambda _result: setattr(self, "_parameter_trend_window", None)
            )
        self._parameter_trend_window.refresh_parameters_and_plot()
        if not self._parameter_trend_window.isVisible():
            move_window_to_cursor_screen(self._parameter_trend_window)
        self._parameter_trend_window.show()
        self._parameter_trend_window.raise_()
        self._parameter_trend_window.activateWindow()

    def refreshParameterTrend(self) -> None:
        if self._parameter_trend_window is not None and self._parameter_trend_window.isVisible():
            self._parameter_trend_window.refresh_parameters_and_plot()

    def onStatusFilterChanged(self) -> None:
        """状态过滤改变"""
        current_data = self.status_filter.currentData()
        self.filter_model.setStatusFilter(current_data)

    def onFilenameFilterChanged(self) -> None:
        """文件名过滤改变"""
        self.filter_model.setFilenameFilter(self.filename_filter.text())

    def onSortChanged(self) -> None:
        """排序改变"""
        column = self.sort_combo.currentData()
        if column is not None:
            current_order = self.filter_model.sortOrder()
            self.filter_model.sort(column, current_order)

    def onSortOrderToggled(self) -> None:
        """切换排序顺序"""
        current_order = self.filter_model.sortOrder()
        new_order = Qt.DescendingOrder if current_order == Qt.AscendingOrder else Qt.AscendingOrder

        column = self.sort_combo.currentData()
        if column is not None:
            self.filter_model.sort(column, new_order)

        # 更新按钮文字
        self.sort_order_btn.setText("Desc" if new_order == Qt.DescendingOrder else "Asc")

    def onTableClicked(self, index: QModelIndex) -> None:
        """表格点击事件"""
        if index.isValid():
            source_index = self.filter_model.mapToSource(index)
            result = self.table_model.getResult(source_index.row())
            if result and result.status == PredictStatus.COMPLETED:
                self.result_selected.emit(result)

    def onTableDoubleClicked(self, index: QModelIndex) -> None:
        """表格双击事件 - 显示单文件预测结果"""
        if index.isValid():
            source_index = self.filter_model.mapToSource(index)
            result = self.table_model.getResult(source_index.row())
            if result:
                # 发射双击信号，无论状态如何都允许查看
                self.result_double_clicked.emit(result)

    def showContextMenu(self, position) -> None:
        """显示右键菜单"""
        index = self.table_view.indexAt(position)
        if not index.isValid():
            return

        menu = QMenu(self)

        # 仅导出此条
        export_action = QAction("Export This Result", self)
        export_action.triggered.connect(lambda: self.exportSingleResult(index))
        menu.addAction(export_action)

        # 重新预测（如果失败）
        source_index = self.filter_model.mapToSource(index)
        result = self.table_model.getResult(source_index.row())
        if result and result.status == PredictStatus.FAILED:
            retry_action = QAction("Retry Prediction", self)
            retry_action.triggered.connect(lambda: self.retryPrediction(result))
            menu.addAction(retry_action)

        menu.exec_(self.table_view.mapToGlobal(position))

    def exportSingleResult(self, index: QModelIndex) -> None:
        """导出单个结果"""
        source_index = self.filter_model.mapToSource(index)
        result = self.table_model.getResult(source_index.row())
        if result:
            # 使用默认配置导出单个结果
            config = {
                "range": -1,  # 特殊值表示单个结果
                "jsonl": True,
                "jpg": True,
                "ascii": True,
            }
            self.export_requested.emit(config, [result])

    def retryPrediction(self, result: PredictResult) -> None:
        """重试预测（这里只是重置状态，实际重试逻辑在控制器中）"""
        # 发送信号让控制器处理重试逻辑
        pass

    def onExportClicked(self) -> None:
        """导出按钮点击"""
        total_results = self.table_model.getAllResults()
        selected_results = self.getSelectedResults()
        current_results = self.getCurrentDisplayResults()

        if not total_results:
            QMessageBox.information(self, "Export", "No results to export.")
            return

        dialog = ExportDialog(len(total_results), len(selected_results), len(current_results), self)

        if dialog.exec_() == QDialog.Accepted:
            config = dialog.getExportConfig()

            # 根据选择确定要导出的结果
            if config["range"] == 0:  # All
                results_to_export = total_results
            elif config["range"] == 1:  # Selected
                results_to_export = selected_results
            else:  # Current display
                results_to_export = current_results

            if results_to_export:
                self.export_requested.emit(config, results_to_export)
            else:
                QMessageBox.information(self, "Export", "No results selected for export.")

    def onClearClicked(self) -> None:
        """清空按钮点击"""
        reply = QMessageBox.question(
            self,
            "Clear Results",
            "Are you sure you want to clear all results?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self.clearResults()
