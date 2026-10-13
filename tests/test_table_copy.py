"""Tables copy whole rows with their column names: Ctrl+C, Copy Rows and Copy Table."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QTableWidget, QTableWidgetItem


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _table() -> QTableWidget:
    table = QTableWidget(3, 3)
    table.setHorizontalHeaderLabels(["q (Å⁻¹)", "d (nm)", "note"])
    for row, values in enumerate((("0.998", "0.630", "ring"), ("1.420", "0.442", "two\nlines"), ("2.010", "0.313", ""))):
        for column, value in enumerate(values):
            table.setItem(row, column, QTableWidgetItem(value))
    return table


def test_the_selected_rows_are_copied_with_the_header() -> None:
    from src.gimap.app.presentation.components.table_copy import copy_rows, enable_table_copy, table_text

    _app()
    table = enable_table_copy(_table())
    assert table.contextMenuPolicy() == Qt.ActionsContextMenu
    assert [action.text() for action in table.actions()] == ["Copy Rows", "Copy Table"]
    assert table_text(table).splitlines()[0] == "q (Å⁻¹)\td (nm)\tnote"
    table.selectRow(1)
    assert copy_rows(table) == "q (Å⁻¹)\td (nm)\tnote\n1.420\t0.442\ttwo lines"
    assert QApplication.clipboard().text() == "q (Å⁻¹)\td (nm)\tnote\n1.420\t0.442\ttwo lines"
    table.setColumnHidden(2, True)
    assert copy_rows(table, whole=True).splitlines() == ["q (Å⁻¹)\td (nm)", "0.998\t0.630", "1.420\t0.442", "2.010\t0.313"]
    enable_table_copy(table)  # twice: still one pair of actions
    assert len(table.actions()) == 2
    table.deleteLater()


def test_ctrl_c_copies_rows_and_nothing_selected_copies_all() -> None:
    from src.gimap.app.presentation.components.table_copy import enable_table_copy

    _app()
    table = enable_table_copy(_table())
    table.show()
    table.clearSelection()
    QApplication.clipboard().setText("")
    QTest.keySequence(table, QKeySequence(QKeySequence.Copy))
    assert QApplication.clipboard().text().count("\n") == 3
    table.selectRow(0)
    QTest.keySequence(table, QKeySequence(QKeySequence.Copy))
    assert QApplication.clipboard().text() == "q (Å⁻¹)\td (nm)\tnote\n0.998\t0.630\tring"
    table.close()
    table.deleteLater()


def test_short_headers_copy_their_full_names() -> None:
    from src.gimap.app.presentation.components.table_copy import enable_table_copy, table_text

    _app()
    table = enable_table_copy(_table())
    table.setHorizontalHeaderLabels(["q", "d", "note"])
    table.horizontalHeaderItem(0).setData(Qt.UserRole, "q (Å⁻¹)")  # what the copy says; the table shows “q”
    assert table.horizontalHeaderItem(0).text() == "q"
    assert table_text(table).splitlines()[0] == "q (Å⁻¹)\td\tnote"
    table.deleteLater()


def test_a_table_with_its_own_menu_keeps_it() -> None:
    from src.gimap.app.presentation.components.table_copy import enable_table_copy

    _app()
    table = _table()
    table.setContextMenuPolicy(Qt.CustomContextMenu)
    enable_table_copy(table)
    assert table.contextMenuPolicy() == Qt.CustomContextMenu and not table.actions()
    table.deleteLater()
