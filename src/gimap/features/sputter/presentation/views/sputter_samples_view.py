"""Static layout of the sample list of After Deposition (behaviour in ``presentation/static_list.py``).

Left of the main screen and of Details while After Deposition is chosen: a card titled Sample List with the number of
samples, a menu (select all or none, remove the samples left out, remove all) and the list itself — one row per
series (its name, its bare substrate and how many of its samples are analysed) with its samples under it: a check box
(analysed by Analyze or not), the name, the nominal thickness in nm (both editable: double-click) and the state. While
the list is empty the card says how to fill it.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QLabel, QMenu, QSizePolicy,
                             QToolButton, QTreeWidget, QVBoxLayout, QWidget)

SAMPLES_TITLE = "Sample List"
SAMPLES_HINT = ("Add Samples… takes a folder of samples — each folder in it with CBF frames is one sample, however "
                "deep they lie — or one sample's folder. The nominal thickness is read from the name (10Au, H10Au "
                "for HiPIMS, 6.5nm); a folder named like the part before it (850 for 850_10Au) is the bare "
                "substrate.")
COLUMNS = ("Sample", "nm", "State")
EDIT_LIST = "Edit"
SELECT_ALL, SELECT_NONE = "Select All", "Select None"
REMOVE_LEFT_OUT, REMOVE_ALL = "Remove the Samples Not Selected", "Remove All"
LIST_WIDTH = 320
STATE_WIDTH = 92
"""The state column: short words (the reason of a failure or of a sample left out is in the tooltip)."""


class SputterSamplesView:
    """Builds ``samples_panel`` and its widgets on the page."""

    def _samples_panel(self, parent: QWidget) -> QFrame:
        panel = QFrame(parent)
        panel.setObjectName("sputterSamples")
        panel.setProperty("card", True)
        panel.setMinimumWidth(240)
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel(SAMPLES_TITLE, panel)
        title.setProperty("sectionTitle", True)
        self.samples_count = QLabel("", panel)
        self.samples_count.setProperty("gimapRole", "muted")
        self.samples_menu_button = QToolButton(panel)
        self.samples_menu_button.setObjectName("sputterSamplesMenu")
        self.samples_menu_button.setText(EDIT_LIST)
        self.samples_menu_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.samples_menu_button)
        self.samples_select_all = menu.addAction(SELECT_ALL)
        self.samples_select_none = menu.addAction(SELECT_NONE)
        menu.addSeparator()
        self.samples_remove_left_out = menu.addAction(REMOVE_LEFT_OUT)
        self.samples_remove_all = menu.addAction(REMOVE_ALL)
        self.samples_menu_button.setMenu(menu)
        header.addWidget(title)
        header.addWidget(self.samples_count, 1)
        header.addWidget(self.samples_menu_button)
        layout.addLayout(header)

        self.samples_tree = QTreeWidget(panel)
        self.samples_tree.setObjectName("sputterSamplesTree")
        self.samples_tree.setColumnCount(len(COLUMNS))
        self.samples_tree.setHeaderLabels(list(COLUMNS))
        self.samples_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.samples_tree.setEditTriggers(QAbstractItemView.NoEditTriggers)  # a double-click edits (static_list.py)
        self.samples_tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.samples_tree.setUniformRowHeights(True)
        self.samples_tree.setExpandsOnDoubleClick(False)
        self.samples_tree.setTextElideMode(Qt.ElideRight)  # whole in the tooltip
        header_view = self.samples_tree.header()
        header_view.setStretchLastSection(False)
        header_view.setSectionResizeMode(0, QHeaderView.Stretch)  # the name gets the room
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.Fixed)
        header_view.resizeSection(2, STATE_WIDTH)
        layout.addWidget(self.samples_tree, 1)

        self.samples_hint = QLabel(SAMPLES_HINT, panel)
        self.samples_hint.setObjectName("sputterSamplesHint")
        self.samples_hint.setProperty("gimapRole", "muted")
        self.samples_hint.setWordWrap(True)
        self.samples_hint.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.samples_hint, 1)
        self.samples_panel = panel
        return panel


__all__ = ["COLUMNS", "EDIT_LIST", "LIST_WIDTH", "REMOVE_ALL", "REMOVE_LEFT_OUT", "SAMPLES_HINT", "SAMPLES_TITLE",
           "SELECT_ALL", "SELECT_NONE", "STATE_WIDTH", "SputterSamplesView"]
