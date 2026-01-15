"""多文件预测结果管理模块"""

from __future__ import annotations

import os
import json
import datetime
import threading
from enum import Enum
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor

from PyQt5.QtCore import QObject, pyqtSignal, QTimer, Qt, QSortFilterProxyModel, QAbstractTableModel, QModelIndex, QVariant
from PyQt5.QtGui import QFont, QBrush, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableView, QProgressBar, 
    QLabel, QPushButton, QComboBox, QLineEdit, QHeaderView, QMenu,
    QAction, QDialog, QDialogButtonBox, QRadioButton, QButtonGroup,
    QGroupBox, QCheckBox, QMessageBox, QApplication, QAbstractItemView
)


class PredictStatus(Enum):
    """预测状态枚举"""
    PENDING = "Pending"
    RUNNING = "Running" 
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass
class PredictResult:
    """单个预测结果数据类"""
    file_path: str
    file_name: str
    status: PredictStatus
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    processing_time: float = 0.0
    error_message: str = ""
    prediction_data: Optional[Dict[str, Any]] = None
    
    @property
    def duration_str(self) -> str:
        """格式化的处理时间"""
        if self.processing_time > 0:
            return f"{self.processing_time:.2f}s"
        return "-"
        
    @property
    def status_color(self) -> QColor:
        """状态对应的颜色"""
        color_map = {
            PredictStatus.PENDING: QColor(128, 128, 128),     # 灰色
            PredictStatus.RUNNING: QColor(0, 123, 255),       # 蓝色
            PredictStatus.COMPLETED: QColor(40, 167, 69),     # 绿色
            PredictStatus.FAILED: QColor(220, 53, 69),        # 红色
            PredictStatus.CANCELLED: QColor(255, 193, 7),     # 黄色
        }
        return color_map.get(self.status, QColor(0, 0, 0))


class PredictResultsTableModel(QAbstractTableModel):
    """预测结果表格模型"""
    
    COLUMNS = ['File Name', 'Status', 'Duration', 'Error']
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: List[PredictResult] = []
        
    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.results)
        
    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.COLUMNS)
        
    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> QVariant:
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.COLUMNS[section]
        return QVariant()
        
    def data(self, index: QModelIndex, role: int) -> QVariant:
        if not index.isValid() or index.row() >= len(self.results):
            return QVariant()
            
        result = self.results[index.row()]
        col = index.column()
        
        if role == Qt.DisplayRole:
            if col == 0:  # File Name
                return result.file_name
            elif col == 1:  # Status
                return result.status.value
            elif col == 2:  # Duration
                return result.duration_str
            elif col == 3:  # Error
                return result.error_message[:50] + "..." if len(result.error_message) > 50 else result.error_message
                
        elif role == Qt.ForegroundRole:
            if col == 1:  # Status column color
                return QBrush(result.status_color)
                
        elif role == Qt.FontRole:
            if col == 1 and result.status == PredictStatus.RUNNING:
                font = QFont()
                font.setBold(True)
                return font
                
        elif role == Qt.ToolTipRole:
            if col == 3 and result.error_message:
                return result.error_message
            elif col == 0:
                return result.file_path
                
        return QVariant()
        
    def addResult(self, result: PredictResult) -> None:
        """添加新结果"""
        self.beginInsertRows(QModelIndex(), len(self.results), len(self.results))
        self.results.append(result)
        self.endInsertRows()
        
    def updateResult(self, index: int, result: PredictResult) -> None:
        """更新结果"""
        if 0 <= index < len(self.results):
            self.results[index] = result
            model_index = self.index(index, 0)
            self.dataChanged.emit(model_index, self.index(index, self.columnCount() - 1))
            
    def getResult(self, index: int) -> Optional[PredictResult]:
        """获取结果"""
        if 0 <= index < len(self.results):
            return self.results[index]
        return None
        
    def getAllResults(self) -> List[PredictResult]:
        """获取所有结果"""
        return self.results.copy()
        
    def clear(self) -> None:
        """清空结果"""
        self.beginResetModel()
        self.results.clear()
        self.endResetModel()


class PredictResultsFilterModel(QSortFilterProxyModel):
    """预测结果过滤模型"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.status_filter: Optional[PredictStatus] = None
        self.filename_filter: str = ""
        
    def setStatusFilter(self, status: Optional[PredictStatus]) -> None:
        """设置状态过滤"""
        self.status_filter = status
        self.invalidateFilter()
        
    def setFilenameFilter(self, filename: str) -> None:
        """设置文件名过滤"""
        self.filename_filter = filename.lower()
        self.invalidateFilter()
        
    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        source_model = self.sourceModel()
        if not isinstance(source_model, PredictResultsTableModel):
            return True
            
        result = source_model.getResult(source_row)
        if result is None:
            return False
            
        # 状态过滤
        if self.status_filter is not None and result.status != self.status_filter:
            return False
            
        # 文件名过滤
        if self.filename_filter and self.filename_filter not in result.file_name.lower():
            return False
            
        return True


class ExportDialog(QDialog):
    """导出对话框"""
    
    def __init__(self, total_count: int, selected_count: int, current_count: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Prediction Results")
        self.setModal(True)
        self.setFixedSize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # 选择导出范围
        range_group = QGroupBox("Export Range")
        range_layout = QVBoxLayout(range_group)
        
        self.range_group = QButtonGroup(self)
        
        self.all_radio = QRadioButton(f"All Results ({total_count} items)")
        self.selected_radio = QRadioButton(f"Selected Results ({selected_count} items)")
        self.current_radio = QRadioButton(f"Current Display ({current_count} items)")
        
        self.all_radio.setChecked(True)
        
        self.range_group.addButton(self.all_radio, 0)
        self.range_group.addButton(self.selected_radio, 1)
        self.range_group.addButton(self.current_radio, 2)
        
        range_layout.addWidget(self.all_radio)
        range_layout.addWidget(self.selected_radio)
        range_layout.addWidget(self.current_radio)
        
        layout.addWidget(range_group)
        
        # 选择导出类型
        type_group = QGroupBox("Export Type")
        type_layout = QVBoxLayout(type_group)
        
        self.jsonl_check = QCheckBox("Structured JSONL/NDJSON")
        self.jsonl_check.setChecked(True)
        self.jpg_check = QCheckBox("JPG Images (in folder)")
        self.ascii_check = QCheckBox("1D Curve ASCII files")
        
        type_layout.addWidget(self.jsonl_check)
        type_layout.addWidget(self.jpg_check)
        type_layout.addWidget(self.ascii_check)
        
        layout.addWidget(type_group)
        
        # 按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        # 禁用选中范围选项如果没有选中项
        if selected_count == 0:
            self.selected_radio.setEnabled(False)
            
    def getExportConfig(self) -> Dict[str, Any]:
        """获取导出配置"""
        return {
            'range': self.range_group.checkedId(),  # 0: all, 1: selected, 2: current
            'jsonl': self.jsonl_check.isChecked(),
            'jpg': self.jpg_check.isChecked(),
            'ascii': self.ascii_check.isChecked()
        }


class MultiFilePredictResultsWidget(QWidget):
    """多文件预测结果Widget"""
    
    # 信号
    result_selected = pyqtSignal(PredictResult)
    result_double_clicked = pyqtSignal(PredictResult)  # 双击信号
    export_requested = pyqtSignal(dict, list)  # config, results
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUI()
        
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
        
    def setupUI(self) -> None:
        """设置UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # 顶部：进度条和统计信息
        top_layout = QHBoxLayout()
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        top_layout.addWidget(self.progress_bar)
        
        self.stats_label = QLabel("Ready")
        top_layout.addWidget(self.stats_label)
        
        layout.addLayout(top_layout)
        
        # 过滤控件
        filter_layout = QHBoxLayout()
        
        filter_layout.addWidget(QLabel("Filter:"))
        
        self.status_filter = QComboBox()
        self.status_filter.addItem("All Status", None)
        for status in PredictStatus:
            self.status_filter.addItem(status.value, status)
        filter_layout.addWidget(self.status_filter)
        
        self.filename_filter = QLineEdit()
        self.filename_filter.setPlaceholderText("Filter by filename...")
        filter_layout.addWidget(self.filename_filter)
        
        filter_layout.addWidget(QLabel("Sort:"))
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("File Name", 0)
        self.sort_combo.addItem("Status", 1) 
        self.sort_combo.addItem("Duration", 2)
        filter_layout.addWidget(self.sort_combo)
        
        self.sort_order_btn = QPushButton("↑")
        self.sort_order_btn.setFixedSize(30, 25)
        filter_layout.addWidget(self.sort_order_btn)
        
        layout.addLayout(filter_layout)
        
        # 结果表格
        self.table_view = QTableView()
        layout.addWidget(self.table_view)
        
        # 底部按钮
        button_layout = QHBoxLayout()
        
        self.export_btn = QPushButton("Export...")
        self.clear_btn = QPushButton("Clear")
        
        button_layout.addStretch()
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.clear_btn)
        
        layout.addLayout(button_layout)
        
    def setupTable(self) -> None:
        """设置表格"""
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(True)
        
        header = self.table_view.horizontalHeader()
        header.setStretchLastSection(True)
        header.resizeSection(0, 200)  # File Name
        header.resizeSection(1, 100)  # Status
        header.resizeSection(2, 80)   # Duration
        
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
        self.export_btn.clicked.connect(self.onExportClicked)
        self.clear_btn.clicked.connect(self.onClearClicked)
        
    def addPredictResult(self, file_path: str) -> int:
        """添加新的预测结果项"""
        result = PredictResult(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            status=PredictStatus.PENDING
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
        if 'status' in kwargs:
            if kwargs['status'] == PredictStatus.RUNNING and result.start_time is None:
                result.start_time = datetime.datetime.now()
            elif kwargs['status'] in [PredictStatus.COMPLETED, PredictStatus.FAILED, PredictStatus.CANCELLED]:
                if result.start_time and result.end_time is None:
                    result.end_time = datetime.datetime.now()
                    result.processing_time = (result.end_time - result.start_time).total_seconds()
                    
        self.table_model.updateResult(index, result)
        self.updateStats()
        
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
        
    def clear_all_results(self) -> None:
        """清空所有结果（别名方法）"""
        self.clearResults()
        
    def get_all_results(self) -> List[PredictResult]:
        """获取所有结果"""
        return self.table_model.getAllResults()
        
    def get_completed_results(self) -> List[PredictResult]:
        """获取所有已完成的结果"""
        return [result for result in self.get_all_results() 
                if result.status == PredictStatus.COMPLETED]
        
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
        self.sort_order_btn.setText("↓" if new_order == Qt.DescendingOrder else "↑")
        
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
                'range': -1,  # 特殊值表示单个结果
                'jsonl': True,
                'jpg': True,
                'ascii': True
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
            
        dialog = ExportDialog(
            len(total_results),
            len(selected_results),
            len(current_results),
            self
        )
        
        if dialog.exec_() == QDialog.Accepted:
            config = dialog.getExportConfig()
            
            # 根据选择确定要导出的结果
            if config['range'] == 0:  # All
                results_to_export = total_results
            elif config['range'] == 1:  # Selected
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
            self, "Clear Results",
            "Are you sure you want to clear all results?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.clearResults()


class MultiFilePredictManager(QObject):
    """多文件预测管理器"""
    
    # 信号
    prediction_started = pyqtSignal()
    prediction_completed = pyqtSignal()
    result_updated = pyqtSignal(int, dict)  # index, update_data
    progress_updated = pyqtSignal(int, int)  # completed, total
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.executor = ThreadPoolExecutor(max_workers=1)  # 单线程顺序处理
        self.current_futures = []
        self.is_running = False
        self.cancelled = False
        
    def start_batch_prediction(self, file_paths: List[str], predict_func: Callable) -> None:
        """开始批量预测"""
        if self.is_running:
            return
            
        self.is_running = True
        self.cancelled = False
        self.current_futures.clear()
        
        self.prediction_started.emit()
        
        # 提交批量任务
        future = self.executor.submit(self._batch_predict_worker, file_paths, predict_func)
        self.current_futures.append(future)
        
    def cancel_prediction(self) -> None:
        """取消预测"""
        self.cancelled = True
        for future in self.current_futures:
            future.cancel()
            
    def _batch_predict_worker(self, file_paths: List[str], predict_func: Callable) -> None:
        """批量预测工作线程"""
        total = len(file_paths)
        completed = 0
        
        try:
            for i, file_path in enumerate(file_paths):
                if self.cancelled:
                    break
                    
                # 更新状态为运行中
                self.result_updated.emit(i, {'status': PredictStatus.RUNNING})
                
                try:
                    # 执行预测
                    result_data = predict_func(file_path)
                    
                    # 更新完成状态
                    self.result_updated.emit(i, {
                        'status': PredictStatus.COMPLETED,
                        'prediction_data': result_data
                    })
                    completed += 1
                    
                except Exception as e:
                    # 更新失败状态
                    self.result_updated.emit(i, {
                        'status': PredictStatus.FAILED,
                        'error_message': str(e)
                    })
                    
                # 更新进度
                self.progress_updated.emit(completed, total)
                
        finally:
            self.is_running = False
            self.prediction_completed.emit()