"""Static layout of the step tree of an Analyze run and of Details ▸ Storage (behaviour in
``presentation/live_steps_panel.py`` and ``presentation/storage_panel.py``).

**The live view** (``sputter_live_view.py``) puts the step tree (``step_tree.StepChecklist``, in a scroll area) at
its left when the package reports its steps, and a second page at its right for what a step found: its name with
*Back to the Analysis*, the whole line of what it found, a line of numbers and a picture (``step_pictures.py`` fills the host).

**Details ▸ Steps**: the same tree for the result shown (left: one line about it, then the tree) and what the chosen
step found (right).

**Details ▸ Storage** (collapsed): the import caches of Analyze — one row per scan import (its size, when it was last
used, where it is) —, the size of the results, *Remove Selected*, *Remove Unused For n Days*, and the limit of the
caches in GB (0: no limit).
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox,
                             QTableWidget, QVBoxLayout, QWidget)

from src.gimap.app.presentation.components import AdvancedSection
from src.gimap.app.presentation.components.table_copy import enable_table_copy

from ..step_tree import StepChecklist

STEPS_COLUMN = 380
"""The width of the tree in Details ▸ Steps."""
TAB_STEPS = "Steps"
STORAGE_TITLE = "Storage"
BACK_TIP = "Close what this step found and follow the analysis again."
REMOVE_SELECTED = "Remove Selected"
REMOVE_SELECTED_TIP = "Remove the imports chosen in the table (the scan is read again the next time it is analysed)."
REMOVE_UNUSED = "Remove Unused For"
REMOVE_UNUSED_TIP = "Remove the imports no analysis has used for at least this many days."
LIMIT_LABEL = "Limit of the imports"
LIMIT_TIP = ("When the imports together take more than this, the ones not used for the longest time are removed when "
             "an analysis starts and ends — never the one in use, never a result. 0: no limit.")
STORAGE_COLUMNS = ("Scan import", "Size", "Last used", "Where")
DAYS_SUFFIX = " days"
NO_LIMIT = "No limit"


def _scroll(widget: QWidget, parent: QWidget) -> QScrollArea:
    scroll = QScrollArea(parent)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    scroll.setWidget(widget)
    return scroll


def _summary(parent: QWidget) -> QLabel:
    """The whole line of what a step found, under its name (the tree cuts it to its width)."""
    label = QLabel("", parent)
    label.setWordWrap(True)
    label.setProperty("gimapMeta", True)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setVisible(False)
    return label


def _link(text: str, parent: QWidget) -> QPushButton:
    button = QPushButton(text, parent)
    button.setProperty("gimapRole", "link")
    button.setCursor(Qt.PointingHandCursor)
    return button


class SputterStepsView:
    def _live_tree_area(self, parent: QWidget) -> QScrollArea:
        """The step tree of the live view (hidden until the package reports its steps)."""
        holder = QWidget()
        column = QVBoxLayout(holder)
        column.setContentsMargins(0, 0, 0, 0)
        self.live_tree = StepChecklist(holder)
        self.live_tree.setObjectName("sputterLiveTree")
        column.addWidget(self.live_tree)
        column.addStretch(1)
        self.live_tree_area = _scroll(holder, parent)
        self.live_tree_area.setVisible(False)
        return self.live_tree_area

    def _live_detail_page(self, parent: QWidget) -> QWidget:
        """What a step found, in the live view: its name, Back to the Analysis, then the host of its picture."""
        page = QWidget(parent)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        top = QHBoxLayout()
        self.live_detail_title = QLabel("", page)
        self.live_detail_title.setObjectName("sputterLiveDetailTitle")
        self.live_detail_title.setProperty("gimapRole", "heading")
        self.live_detail_back = _link("Back to the Analysis", page)
        self.live_detail_back.setObjectName("sputterLiveDetailBack")
        self.live_detail_back.setToolTip(BACK_TIP)
        top.addWidget(self.live_detail_title, 1)
        top.addWidget(self.live_detail_back)
        layout.addLayout(top)
        self.live_detail_summary = _summary(page)
        layout.addWidget(self.live_detail_summary)
        self.live_detail_layout = QVBoxLayout()
        self.live_detail_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.live_detail_layout, 1)
        return page

    def _steps_tab(self) -> QWidget:
        """Details ▸ Steps: the tree of the result shown and what the chosen step found."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(12)
        left = QWidget(tab)
        left.setFixedWidth(STEPS_COLUMN)
        column = QVBoxLayout(left)
        column.setContentsMargins(0, 0, 0, 0)
        self.steps_note = QLabel("", left)
        self.steps_note.setObjectName("sputterStepsNote")
        self.steps_note.setWordWrap(True)
        self.steps_note.setProperty("gimapMeta", True)
        holder = QWidget()
        inner = QVBoxLayout(holder)
        inner.setContentsMargins(0, 0, 0, 0)
        self.steps_tree = StepChecklist(holder)
        self.steps_tree.setObjectName("sputterStepsTree")
        inner.addWidget(self.steps_tree)
        inner.addStretch(1)
        column.addWidget(self.steps_note)
        column.addWidget(_scroll(holder, left), 1)
        layout.addWidget(left)
        right = QVBoxLayout()
        right.setSpacing(6)
        self.steps_detail_title = QLabel("", tab)
        self.steps_detail_title.setObjectName("sputterStepsDetailTitle")
        self.steps_detail_title.setProperty("gimapRole", "heading")
        self.steps_detail_layout = QVBoxLayout()
        self.steps_detail_layout.setContentsMargins(0, 0, 0, 0)
        right.addWidget(self.steps_detail_title)
        self.steps_detail_summary = _summary(tab)
        right.addWidget(self.steps_detail_summary)
        right.addLayout(self.steps_detail_layout, 1)
        layout.addLayout(right, 1)
        return tab

    def _storage_section(self, parent: QWidget) -> AdvancedSection:
        """The import caches of Analyze and the size of the results (collapsed; the page fills it when opened)."""
        section = AdvancedSection(STORAGE_TITLE, "", parent, expanded=False)
        self.storage_summary = QLabel("", section)
        self.storage_summary.setObjectName("sputterStorageSummary")
        self.storage_summary.setWordWrap(True)
        self.storage_summary.setProperty("gimapMeta", True)
        self.storage_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.storage_table = QTableWidget(0, len(STORAGE_COLUMNS), section)
        self.storage_table.setObjectName("sputterStorageTable")
        self.storage_table.setHorizontalHeaderLabels(list(STORAGE_COLUMNS))
        self.storage_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.storage_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.storage_table.verticalHeader().setVisible(False)
        self.storage_table.horizontalHeader().setStretchLastSection(True)
        self.storage_table.setMinimumHeight(120)
        self.storage_table.setMaximumHeight(220)
        enable_table_copy(self.storage_table)
        self.storage_remove = QPushButton(REMOVE_SELECTED, section)
        self.storage_remove.setObjectName("sputterStorageRemove")
        self.storage_remove.setToolTip(REMOVE_SELECTED_TIP)
        self.storage_remove_unused = QPushButton(REMOVE_UNUSED, section)
        self.storage_remove_unused.setObjectName("sputterStorageRemoveUnused")
        self.storage_remove_unused.setToolTip(REMOVE_UNUSED_TIP)
        self.storage_days = QSpinBox(section)
        self.storage_days.setObjectName("sputterStorageDays")
        self.storage_days.setRange(1, 3650)
        self.storage_days.setValue(30)
        self.storage_days.setSuffix(DAYS_SUFFIX)
        self.storage_days.setToolTip(REMOVE_UNUSED_TIP)
        self.storage_limit = QSpinBox(section)
        self.storage_limit.setObjectName("sputterStorageLimit")
        self.storage_limit.setRange(0, 10000)
        self.storage_limit.setSuffix(" GB")
        self.storage_limit.setSpecialValueText(NO_LIMIT)
        self.storage_limit.setToolTip(LIMIT_TIP)
        limit_label = QLabel(LIMIT_LABEL, section)
        limit_label.setToolTip(LIMIT_TIP)
        limit_label.setBuddy(self.storage_limit)
        buttons = QWidget(section)
        row = QHBoxLayout(buttons)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(self.storage_remove)
        row.addStretch(1)
        row.addWidget(self.storage_remove_unused)
        row.addWidget(self.storage_days)
        limit = QWidget(section)
        line = QHBoxLayout(limit)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(6)
        line.addWidget(limit_label)
        line.addWidget(self.storage_limit)
        line.addStretch(1)
        section.add_widget(self.storage_summary)
        section.add_widget(self.storage_table)
        section.add_widget(buttons)
        section.add_widget(limit)
        self.storage_section = section
        return section


__all__ = ["DAYS_SUFFIX", "LIMIT_LABEL", "LIMIT_TIP", "NO_LIMIT", "REMOVE_SELECTED", "REMOVE_SELECTED_TIP",
           "REMOVE_UNUSED", "REMOVE_UNUSED_TIP", "STORAGE_COLUMNS", "STORAGE_TITLE", "SputterStepsView", "TAB_STEPS"]
