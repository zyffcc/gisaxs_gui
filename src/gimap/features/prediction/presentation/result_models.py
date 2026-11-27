"""Result Models for multi-file prediction."""

from __future__ import annotations


from typing import Dict, List, Optional


from PyQt5.QtCore import Qt, QSortFilterProxyModel, QAbstractTableModel, QModelIndex, QVariant

from PyQt5.QtGui import QFont, QBrush

from src.gimap.app.presentation.i18n import language_changed, tr


from .result_types import (
    PredictResult,
    PredictStatus,
)


class PredictResultsTableModel(QAbstractTableModel):
    """预测结果表格模型"""

    COLUMNS = ["Input Stack", "Files", "Status", "Duration", "Error"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.results: List[PredictResult] = []
        self._counted: List[PredictStatus] = []
        """The state each row is counted in (``counts``): a live batch asks after every row and every change of state,
        and counting all the rows each time took 15 ms of the window at 50 000 rows."""
        self._counts: Dict[PredictStatus, int] = {}
        language_changed().connect(self._language_changed)

    def counts(self) -> Dict[PredictStatus, int]:
        """How many rows are in each state."""
        return dict(self._counts)

    def _count(self, index: int, status: PredictStatus) -> None:
        before = self._counted[index]
        if before != status:
            self._counts[before] -= 1
            self._counts[status] = self._counts.get(status, 0) + 1
            self._counted[index] = status

    def _language_changed(self, *_args) -> None:
        """The headers and the states are shown in the interface language (file names and errors as they are)."""
        self.headerDataChanged.emit(Qt.Horizontal, 0, len(self.COLUMNS) - 1)
        if self.results:
            self.dataChanged.emit(self.index(0, 2), self.index(len(self.results) - 1, 2))

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self.results)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(self.COLUMNS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> QVariant:
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return tr(self.COLUMNS[section])
        return QVariant()

    def data(self, index: QModelIndex, role: int) -> QVariant:
        if not index.isValid() or index.row() >= len(self.results):
            return QVariant()

        result = self.results[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:  # File Name
                return result.file_name
            elif col == 1:  # Stack count
                return f"{max(1, int(getattr(result, 'stack_count', 1)))}"
            elif col == 2:  # Status
                return tr(result.status.value)
            elif col == 3:  # Duration
                return result.duration_str
            elif col == 4:  # Error
                return (
                    result.error_message[:50] + "..."
                    if len(result.error_message) > 50
                    else result.error_message
                )

        elif role == Qt.ForegroundRole:
            if col == 2:  # Status column color
                return QBrush(result.status_color)

        elif role == Qt.FontRole:
            if col == 2 and result.status == PredictStatus.RUNNING:
                font = QFont()
                font.setBold(True)
                return font

        elif role == Qt.ToolTipRole:
            if col == 4 and result.error_message:
                return result.error_message
            elif col == 0:
                return result.file_path

        return QVariant()

    def addResult(self, result: PredictResult) -> None:
        """添加新结果"""
        self.beginInsertRows(QModelIndex(), len(self.results), len(self.results))
        self.results.append(result)
        self._counted.append(result.status)
        self._counts[result.status] = self._counts.get(result.status, 0) + 1
        self.endInsertRows()

    def updateResult(self, index: int, result: PredictResult) -> None:
        """更新结果"""
        if 0 <= index < len(self.results):
            self.results[index] = result
            self._count(index, result.status)
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
        self._counted.clear()
        self._counts.clear()
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
