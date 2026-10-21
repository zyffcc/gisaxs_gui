"""Small widgets the steps of the single-curve page share: a muted line, an info card, a read-only table that
is as tall as its rows."""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QAbstractItemView, QFrame, QHeaderView, QLabel, QTableWidget, QVBoxLayout, QWidget

from src.gimap.app.presentation.components.table_copy import enable_table_copy


def muted(text: str, parent: QWidget) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    label.setProperty("gimapRole", "muted")
    return label


def info_card(parent: QWidget) -> tuple[QFrame, QLabel]:
    card = QFrame(parent)
    card.setProperty("gimapInfoCard", True)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 8, 10, 8)
    label = QLabel("", card)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    layout.addWidget(label)
    return card, label


def read_only_table(columns, parent: QWidget, name: str,
                    selection=QAbstractItemView.SingleSelection) -> QTableWidget:
    """Rows selected whole; Ctrl+C and the right button copy them with the column names (``enable_table_copy``).
    ``SingleSelection`` where the current row drives something (“Use This Solution”), else ``ExtendedSelection``."""
    table = QTableWidget(0, len(columns), parent)
    table.setObjectName(name)
    table.setHorizontalHeaderLabels(list(columns))
    table.verticalHeader().hide()
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(selection)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    enable_table_copy(table)
    return table


class _KeepToRows(QObject):
    """Sizes the table again once it is shown or restyled (its header's height is known only then)."""

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if event.type() in (QEvent.Show, QEvent.StyleChange, QEvent.FontChange):
            fit_to_rows(watched)
        return False


def fit_to_rows(table: QTableWidget) -> None:
    """As tall as its header and rows, with no scroll bar of its own: the step's page scrolls instead.
    Call again after the rows change."""
    if not table.property("gimapFitToRows"):
        table.setProperty("gimapFitToRows", True)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        table._gimap_keep_to_rows = _KeepToRows(table)
        table.installEventFilter(table._gimap_keep_to_rows)
    header = table.horizontalHeader()
    top = header.height() if header.isVisible() and header.height() > 0 else header.sizeHint().height()
    scroll = table.horizontalScrollBar()
    bottom = scroll.sizeHint().height() if table.horizontalScrollBarPolicy() != Qt.ScrollBarAlwaysOff \
        and scroll.isVisible() else 0
    table.setFixedHeight(top + table.verticalHeader().length() + bottom + 2 * table.frameWidth())


__all__ = ["fit_to_rows", "info_card", "muted", "read_only_table"]
