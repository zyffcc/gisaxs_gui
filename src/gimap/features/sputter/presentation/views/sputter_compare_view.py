"""Static layout of Compare Scans (behaviour in ``presentation/compare_dialog.py``; the button that opens it sits in the
input bar beside Export…).

Left, the results to compare: a list with a box and an editable name per result (the result shown on the main screen
and the others analysed under the results root), Add Result… and Remove. Right, the switch of the horizontal axis
(nominal deposit or time since the onset) and two tabs: **Curves** — one small plot per quantity, one colour per
result — and **Growth** — the growth descriptors of each result in a column. Below, a line on where the values come
from, Export… and Close.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QDialog, QHBoxLayout, QHeaderView, QLabel, QListWidget, QPushButton,
                             QSplitter, QTabWidget, QTableWidget, QVBoxLayout, QWidget)

from src.gimap.app.presentation.components import SegmentedControl

COMPARE_BUTTON = "Compare…"
COMPARE_TIP = ("Put the results of several scans side by side against the deposited amount: sizes, coverage, film "
               "thickness and the growth stages, for example the same metal on two polymers.")
AXIS_CHOICES = (("Nominal deposit", "nominal"), ("Time", "time"))
SOURCE_NOTE = ("Each line is the solution shown of each window (as chosen on the main screen); the film thickness, "
               "the grain height from its fringe, the optical density and the lateral scale are read from the data "
               "without the model (the report). GIMaP computes none of the values. Tick, rename or add results at "
               "the left.")
LIST_TIP = "Tick the results to compare; double-click a name to change it (the name is used in the plots and files)."


def compare_button(parent) -> QPushButton:
    """The button of the input bar that opens Compare Scans."""
    button = QPushButton(COMPARE_BUTTON, parent)
    button.setObjectName("sputterCompare")
    button.setToolTip(COMPARE_TIP)
    return button


class CompareDialogView:
    def setup_ui(self, dialog: QDialog) -> None:
        dialog.setWindowTitle("Compare Scans")
        dialog.setObjectName("sputterCompareDialog")
        dialog.resize(1180, 760)
        outer = QVBoxLayout(dialog)
        splitter = QSplitter(Qt.Horizontal, dialog)
        side = QWidget(splitter)
        left = QVBoxLayout(side)
        left.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Results", side)
        title.setProperty("gimapRole", "heading")
        left.addWidget(title)
        self.scan_list = QListWidget(side)
        self.scan_list.setObjectName("sputterCompareList")
        self.scan_list.setToolTip(LIST_TIP)
        self.scan_list.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.scan_list.setMinimumWidth(240)
        left.addWidget(self.scan_list, 1)
        buttons = QHBoxLayout()
        self.add_button = QPushButton("Add Result…", side)
        self.add_button.setToolTip("A result folder analysed before, also from another results folder.")
        self.remove_button = QPushButton("Remove", side)
        self.remove_button.setToolTip("Take the selected result off the list (its files stay).")
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        left.addLayout(buttons)
        main = QWidget(splitter)
        right = QVBoxLayout(main)
        right.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        axis_label = QLabel("Horizontal axis", main)
        self.axis_switch = SegmentedControl(main)
        self.axis_switch.setObjectName("sputterCompareAxis")
        for label, key in AXIS_CHOICES:
            self.axis_switch.addItem(label, key)
        self.axis_switch.setItemToolTip(0, "The nominal deposit of each window (rate × time): scans at different "
                                           "rates are compared at the same amount of metal.")
        self.axis_switch.setItemToolTip(1, "The time since the onset of the deposition.")
        top.addWidget(axis_label)
        top.addWidget(self.axis_switch)
        top.addStretch(1)
        right.addLayout(top)
        self.tabs = QTabWidget(main)
        self.curves_page = QWidget(self.tabs)
        self.curves_layout = QVBoxLayout(self.curves_page)
        self.curves_layout.setContentsMargins(0, 0, 0, 0)
        self.empty_label = QLabel(self.curves_page)
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setProperty("gimapMeta", True)
        self.curves_layout.addWidget(self.empty_label)
        self.tabs.addTab(self.curves_page, "Curves")
        self.growth_table = QTableWidget(self.tabs)
        self.growth_table.setObjectName("sputterCompareGrowth")
        self.growth_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.growth_table.verticalHeader().setVisible(False)
        self.growth_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tabs.addTab(self.growth_table, "Growth")
        self.tabs.setTabToolTip(1, "The package's growth descriptors of each result: the metal amount at coverage "
                                   "0.5 and at the film onset, the growth laws of size and distance, the stages.")
        right.addWidget(self.tabs, 1)
        splitter.addWidget(side)
        splitter.addWidget(main)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 900])
        outer.addWidget(splitter, 1)
        self.source_note = QLabel(SOURCE_NOTE, dialog)
        self.source_note.setWordWrap(True)
        self.source_note.setProperty("gimapMeta", True)
        outer.addWidget(self.source_note)
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.export_button = QPushButton("Export…", dialog)
        self.export_button.setToolTip("Write the curves and the growth table (CSV) and the plots (PNG, PDF) into a "
                                      "folder.")
        self.close_button = QPushButton("Close", dialog)
        bottom.addWidget(self.export_button)
        bottom.addWidget(self.close_button)
        outer.addLayout(bottom)


__all__ = ["AXIS_CHOICES", "COMPARE_BUTTON", "COMPARE_TIP", "CompareDialogView", "LIST_TIP", "SOURCE_NOTE",
           "compare_button"]
