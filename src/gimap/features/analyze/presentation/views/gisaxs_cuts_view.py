"""Static layout of the GISAXS cuts in the Cuts step: the table of cuts, adding and drawing them, the editor.

The first two rows are the automatic cuts (the horizontal cut at the Yoneda band, which Fitting takes, and the
vertical cut at the beam centre); the cuts the person adds follow. A cut is added from the Add menu (typed: it starts
where it is easy to see) or drawn on the detector image or the q map (Horizontal / Vertical: one click; Box: two
corners; Sector: one click in its direction; Line: its two ends). The tick shows or hides a cut's curve and outline;
double-click a name to rename it; the editor under the table sets the position and width in the unit chosen (q,
exit angles or pixels). Behaviour: ``bindings/gisaxs_cuts.py`` and ``bindings/gisaxs_cut_shapes.py``.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QTableWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import FlowLayout

CUT_UNITS = (("q", "q (Å⁻¹)"), ("angle", "Angles (°)"), ("pixel", "Pixels (px)"))
"""The units a cut is typed and shown in (the tooltip says which angles and which pixels)."""
CUT_KIND_ITEMS = (
    ("horizontal", "Horizontal Band — I(qy)"),
    ("vertical", "Vertical Band — I(qz)"),
    ("box", "Box — I(qy) or I(qz)"),
    ("sector", "Sector — I(qyz) in a direction"),
    ("line", "Line in Any Direction — I(s)"),
)
"""The Add menu: each kind and what it gives."""
DRAW_BUTTONS = (
    ("horizontal", "Horizontal", "Click a height on the image or the q map: a horizontal band there, I(qy)"),
    ("vertical", "Vertical", "Click a position on the image or the q map: a vertical band there, I(qz)"),
    ("box", "Box", "Click two opposite corners: a box in qy × qz, cut along qy (or qz in the editor)"),
    ("sector", "Sector", "Click in the direction of the cut: every q within ±5° of it, from the direct beam"),
    ("line", "Line", "Click the two ends of a line in any direction: I along it, ±2 pixels across"),
)
CUT_TABLE_TIP = (
    "Every cut gives one curve: horizontal bands and boxes along qy in the upper plot, vertical bands along qz in the "
    "lower plot, sectors and lines in the lower plot's list. Tick a cut to show its curve and outline (detector and q "
    "map, same colour), double-click a name to rename it, select a cut to change it below. The first two rows are the "
    "automatic cuts; Fitting takes the horizontal one."
)
CUT_UNIT_TIP = (
    "The unit to type and show positions in. Cuts are kept in q, so they hold for every frame of this geometry. Exit "
    "angles: αf of qz, and 2θf of qy on the sample horizon; pixels: rows on the beam-centre column and columns on the "
    "horizon row."
)
CUT_NOTICE_TIP = "The cuts are in q: kept, they cut this frame at the same q; removed, only the automatic cuts stay."


def _spin(parent: QWidget) -> QDoubleSpinBox:
    spin = QDoubleSpinBox(parent)
    spin.setDecimals(4)
    spin.setRange(-1e6, 1e6)
    spin.setKeyboardTracking(False)
    spin.setMinimumWidth(64)
    return spin


class GisaxsCutsView:
    """Adds ``self.gisaxs_cuts_panel``: call inside the GISAXS part of the Cuts step."""

    def setup_gisaxs_cuts_panel(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        panel.setObjectName("analyzeGisaxsCutsPanel")
        self.gisaxs_cuts_panel = panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)
        header = QHBoxLayout()
        title = QLabel("List of cuts", panel)
        title.setProperty("gimapSectionTitle", True)
        self.cut_unit_combo = QComboBox(panel)
        self.cut_unit_combo.setObjectName("analyzeCutUnit")
        for key, text in CUT_UNITS:
            self.cut_unit_combo.addItem(text, key)
        self.cut_unit_combo.setToolTip(CUT_UNIT_TIP)
        self.cut_add_button = QToolButton(panel)
        self.cut_add_button.setObjectName("analyzeCutAdd")
        self.cut_add_button.setText("Add")
        self.cut_add_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.cut_add_button)
        self.cut_add_actions = {key: menu.addAction(text) for key, text in CUT_KIND_ITEMS}
        self.cut_add_button.setMenu(menu)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.cut_unit_combo)
        header.addWidget(self.cut_add_button)
        layout.addLayout(header)

        self.cut_notice = QFrame(panel)
        self.cut_notice.setObjectName("analyzeCutNotice")
        self.cut_notice.setProperty("gimapInfoCard", True)
        self.cut_notice.setToolTip(CUT_NOTICE_TIP)
        notice = QVBoxLayout(self.cut_notice)
        notice.setContentsMargins(10, 8, 10, 8)
        self.cut_notice_label = QLabel("", self.cut_notice)
        self.cut_notice_label.setWordWrap(True)
        notice.addWidget(self.cut_notice_label)
        choice = QHBoxLayout()
        self.cut_keep_button = QPushButton("Keep Cuts", self.cut_notice)
        self.cut_keep_button.setObjectName("analyzeCutKeep")
        self.cut_drop_button = QPushButton("Remove Cuts", self.cut_notice)
        self.cut_drop_button.setObjectName("analyzeCutDrop")
        self.cut_drop_button.setProperty("gimapDangerAction", True)
        choice.addWidget(self.cut_keep_button)
        choice.addWidget(self.cut_drop_button)
        choice.addStretch(1)
        notice.addLayout(choice)
        self.cut_notice.hide()  # placed at the top of the Cuts step (GISAXS and GIWAXS) by ``AnalyzeStepsView``

        prompt = QLabel("Draw on the image or the q map:", panel)
        prompt.setProperty("gimapRole", "muted")
        layout.addWidget(prompt)
        draws = FlowLayout(spacing=6)
        self.cut_draw_buttons: dict[str, QPushButton] = {}
        for key, text, tip in DRAW_BUTTONS:
            button = QPushButton(text, panel)
            button.setObjectName(f"analyzeCutDraw_{key}")
            button.setCheckable(True)
            button.setToolTip(tip)
            draws.addWidget(button)
            self.cut_draw_buttons[key] = button
        layout.addLayout(draws)

        self.cut_table = QTableWidget(0, 2, panel)
        self.cut_table.setObjectName("analyzeCutTable")
        self.cut_table.setHorizontalHeaderLabels(["Cut", "Kind and position"])
        self.cut_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cut_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.cut_table.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.cut_table.verticalHeader().hide()
        self.cut_table.setWordWrap(True)
        self.cut_table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        heading = self.cut_table.horizontalHeader()
        heading.setSectionResizeMode(0, QHeaderView.Interactive)
        heading.setSectionResizeMode(1, QHeaderView.Stretch)
        self.cut_table.setColumnWidth(0, 118)
        self.cut_table.setToolTip(CUT_TABLE_TIP)
        layout.addWidget(self.cut_table)

        actions = QHBoxLayout()
        self.cut_duplicate_button = QPushButton("Duplicate", panel)
        self.cut_duplicate_button.setObjectName("analyzeCutDuplicate")
        self.cut_duplicate_button.setToolTip("A copy of the selected cut, to move to another position")
        self.cut_remove_button = QPushButton("Remove", panel)
        self.cut_remove_button.setObjectName("analyzeCutRemove")
        self.cut_remove_button.setProperty("gimapDangerAction", True)
        self.cut_remove_button.setToolTip("Remove the selected cut (the automatic cuts stay; untick them to hide them)")
        actions.addWidget(self.cut_duplicate_button)
        actions.addWidget(self.cut_remove_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.cut_editor = QWidget(panel)
        self.cut_editor.setObjectName("analyzeCutEditor")
        grid = QGridLayout(self.cut_editor)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(4)
        self.cut_name_edit = QLineEdit(self.cut_editor)
        self.cut_name_edit.setObjectName("analyzeCutName")
        self.cut_name_caption = QLabel("Name", self.cut_editor)
        grid.addWidget(self.cut_name_caption, 0, 0)
        grid.addWidget(self.cut_name_edit, 0, 1, 1, 4)
        self.cut_row_captions: list[QLabel] = []
        self.cut_row_separators: list[QLabel] = []
        self.cut_row_units: list[QLabel] = []
        self.cut_spins: list[QDoubleSpinBox] = []
        for row in range(3):
            caption, separator, unit = QLabel("", self.cut_editor), QLabel("±", self.cut_editor), QLabel("", self.cut_editor)
            first, second = _spin(self.cut_editor), _spin(self.cut_editor)
            first.setObjectName(f"analyzeCutValue{2 * row}")
            second.setObjectName(f"analyzeCutValue{2 * row + 1}")
            grid.addWidget(caption, row + 1, 0)
            grid.addWidget(first, row + 1, 1)
            grid.addWidget(separator, row + 1, 2)
            grid.addWidget(second, row + 1, 3)
            grid.addWidget(unit, row + 1, 4)
            self.cut_row_captions.append(caption)
            self.cut_row_separators.append(separator)
            self.cut_row_units.append(unit)
            self.cut_spins.extend((first, second))
        self.cut_along_combo = QComboBox(self.cut_editor)
        self.cut_along_combo.setObjectName("analyzeCutAlong")
        self.cut_along_combo.addItem("Profile along qy (mean over qz)", "qy")
        self.cut_along_combo.addItem("Profile along qz (mean over qy)", "qz")
        grid.addWidget(self.cut_along_combo, 4, 0, 1, 5)
        self.cut_all_q_check = QCheckBox("Every q", self.cut_editor)
        self.cut_all_q_check.setObjectName("analyzeCutAllQ")
        self.cut_all_q_check.setToolTip("The sector reaches from the direct beam to the edge of the frame")
        grid.addWidget(self.cut_all_q_check, 5, 1, 1, 4)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        self.cut_range_label = QLabel("", self.cut_editor)
        self.cut_range_label.setObjectName("analyzeCutRange")
        self.cut_range_label.setProperty("gimapRole", "muted")
        self.cut_range_label.setWordWrap(True)
        grid.addWidget(self.cut_range_label, 6, 0, 1, 5)
        layout.addWidget(self.cut_editor)
        layout.addStretch(1)  # (room the step gives beyond the rows: below them, not between)
        return panel


__all__ = ["CUT_KIND_ITEMS", "CUT_NOTICE_TIP", "CUT_TABLE_TIP", "CUT_UNITS", "CUT_UNIT_TIP", "DRAW_BUTTONS",
           "GisaxsCutsView"]
