"""Result table presentation with consistent empty-state overlay。"""

from __future__ import annotations

from collections.abc import Iterable

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

from ..styles import apply_design_system


class ResultTable(QTableWidget):
    """Presentation-only table; callers own result values and actions。"""

    def __init__(
        self,
        headers: Iterable[str] = (),
        parent: QWidget | None = None,
        *,
        empty_message: str = "No results yet",
    ) -> None:
        header_list = list(headers)
        super().__init__(0, len(header_list), parent)
        self.setProperty("gimapResultTable", True)
        self.setHorizontalHeaderLabels(header_list)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.empty_label = QLabel(empty_message, self.viewport())
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setProperty("gimapMeta", True)
        self.empty_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._position_empty_label()
        apply_design_system(self)

    def set_headers(self, headers: Iterable[str]) -> None:
        values = list(headers)
        self.setColumnCount(len(values))
        self.setHorizontalHeaderLabels(values)

    def set_empty_message(self, message: str) -> None:
        self.empty_label.setText(message)
        self._refresh_empty_state()

    def set_rows(self, rows: Iterable[Iterable[object]]) -> None:
        values = [list(row) for row in rows]
        self.setRowCount(len(values))
        for row_index, row in enumerate(values):
            for column_index, value in enumerate(row[: self.columnCount()]):
                self.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self._refresh_empty_state()

    def setRowCount(self, rows: int) -> None:  # noqa: N802 - Qt API compatibility
        super().setRowCount(rows)
        if hasattr(self, "empty_label"):
            self._refresh_empty_state()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_empty_label()

    def _position_empty_label(self) -> None:
        self.empty_label.setGeometry(self.viewport().rect())
        self._refresh_empty_state()

    def _refresh_empty_state(self) -> None:
        self.empty_label.setVisible(self.rowCount() == 0)
