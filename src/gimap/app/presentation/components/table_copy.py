"""Copy the rows of a table as text: Ctrl+C and a right-click menu, tab-separated with the header row.

``enable_table_copy(table)`` works on any ``QTableView`` / ``QTableWidget``. The text is what the table
shows (the header labels and the cells' display text), so a pasted table reads like the one on screen;
hidden columns are left out. A table with short headers can keep fuller names with units for the copy
(``Qt.UserRole`` of its header items). Ctrl+C copies the selected rows (every column of them) or, with nothing
selected, the whole table. A table that already has its own context menu keeps it; only Ctrl+C is added.
"""

from __future__ import annotations

from typing import Iterable, Optional

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAbstractItemView, QAction, QApplication

_ENABLED = "gimapTableCopy"


def _clean(text) -> str:
    return " ".join(str(text if text is not None else "").split())  # one line per row, one cell per column


def _copy_header(model, column: int):
    """The column's name for the copied text: a fuller name the table keeps for copying (``Qt.UserRole`` of the
    header, e.g. “q (Å⁻¹)” for a short “q” header), else the header shown."""
    fuller = model.headerData(column, Qt.Horizontal, Qt.UserRole)
    if isinstance(fuller, str) and fuller.strip():
        return fuller
    return model.headerData(column, Qt.Horizontal, Qt.DisplayRole)


def table_text(table: QAbstractItemView, rows: Optional[Iterable[int]] = None) -> str:
    """The header row and ``rows`` (all rows when ``None``), tab-separated, one line per row."""
    model = table.model()
    if model is None:
        return ""
    columns = [column for column in range(model.columnCount()) if not table.isColumnHidden(column)]
    if rows is None:
        rows = [row for row in range(model.rowCount()) if not table.isRowHidden(row)]
    header = [_clean(_copy_header(model, column)) for column in columns]
    lines = ["\t".join(header)] if any(header) else []
    for row in rows:
        lines.append("\t".join(_clean(model.data(model.index(row, column), Qt.DisplayRole)) for column in columns))
    return "\n".join(lines)


def selected_rows(table: QAbstractItemView) -> list[int]:
    """The rows that have a selected cell, top to bottom."""
    selection = table.selectionModel()
    if selection is None:
        return []
    return sorted({index.row() for index in selection.selectedIndexes() if not table.isRowHidden(index.row())})


def copy_rows(table: QAbstractItemView, *, whole: bool = False) -> str:
    """Put the selected rows (or the whole table) on the clipboard; returns the text."""
    rows = None if whole else (selected_rows(table) or None)
    text = table_text(table, rows)
    if text:
        QApplication.clipboard().setText(text)
    return text


class _CopyKey(QObject):
    """Ctrl+C on the table copies whole rows (Qt's own copies only the current cell)."""

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if event.type() in (QEvent.KeyPress, QEvent.ShortcutOverride) and event.matches(QKeySequence.Copy):
            if event.type() == QEvent.ShortcutOverride:
                event.accept()  # keep window shortcuts from taking Ctrl+C away from the table
                return True
            copy_rows(watched)
            return True
        return False


def enable_table_copy(table: QAbstractItemView) -> QAbstractItemView:
    """Ctrl+C and (unless the table has its own menu) Copy Rows / Copy Table on the right button."""
    from ..i18n import tr

    if table.property(_ENABLED):
        return table
    table.setProperty(_ENABLED, True)
    table._gimap_copy_key = _CopyKey(table)  # kept with the table
    table.installEventFilter(table._gimap_copy_key)
    if table.contextMenuPolicy() in (Qt.DefaultContextMenu, Qt.ActionsContextMenu):
        rows = QAction(tr("Copy Rows"), table)
        rows.setToolTip(tr("Copy the selected rows with the column names (tab-separated)"))
        rows.triggered.connect(lambda: copy_rows(table))
        whole = QAction(tr("Copy Table"), table)
        whole.setToolTip(tr("Copy every row with the column names (tab-separated)"))
        whole.triggered.connect(lambda: copy_rows(table, whole=True))
        table.addAction(rows)
        table.addAction(whole)
        table.setContextMenuPolicy(Qt.ActionsContextMenu)
    return table


__all__ = ["copy_rows", "enable_table_copy", "selected_rows", "table_text"]
