"""Static layout of the GIWAXS cut regions in the Cuts step: pick buttons, the list, the editor.

Every row is one region of the pattern: the full ring, the in-plane and out-of-plane bands and
the ring of I(χ) (the standard cuts), a custom sector when one is set, then the regions the
person added. A region is added by clicking on the image (Ring / Sector / Spot: snapped to the
peak there), by drawing a rectangle on the Cake view, or from the presets of the Add menu. The
tick in the first column shows or hides the region's curves and outline; the editor under the
list sets its centre and half width exactly. Behaviour: ``bindings/regions.py`` and
``bindings/region_pick.py``.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMenu,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.gimap.app.presentation.components import FlowLayout

EDITOR_PAGES = ("note", "width", "ring", "generic")
PICK_BUTTONS = (
    ("ring", "Ring", "Click a peak: its ring at every χ, for I(χ) (orientation). The q window is fitted to the peak."),
    ("sector", "Sector", "Click a direction: every q within a few degrees of that χ, for I(q) along it."),
    ("spot", "Spot", "Click a spot: the peak's window in q and in χ, for both profiles of that spot."),
)


def _spin(parent: QWidget, *, decimals: int, low: float, high: float, step: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox(parent)
    spin.setDecimals(decimals)
    spin.setRange(low, high)
    spin.setSingleStep(step)
    spin.setKeyboardTracking(False)
    spin.setMinimumWidth(64)
    return spin


def _q_center(parent: QWidget) -> QDoubleSpinBox:
    return _spin(parent, decimals=4, low=0.0, high=100.0, step=0.001)


def _q_half(parent: QWidget) -> QDoubleSpinBox:
    return _spin(parent, decimals=4, low=0.0001, high=50.0, step=0.001)


def _chi_center(parent: QWidget) -> QDoubleSpinBox:
    return _spin(parent, decimals=1, low=-90.0, high=90.0, step=1.0)


def _chi_half(parent: QWidget) -> QDoubleSpinBox:
    return _spin(parent, decimals=1, low=0.5, high=90.0, step=1.0)


def _plus_minus(first: QWidget, second: QWidget, parent: QWidget, unit: str) -> QHBoxLayout:
    """``centre ± half width unit`` (the unit once, after both numbers, so the numbers keep their room)."""
    row = QHBoxLayout()
    row.setSpacing(4)
    row.addWidget(first, 1)
    row.addWidget(QLabel("±", parent))
    row.addWidget(second, 1)
    row.addWidget(QLabel(unit, parent))
    return row


class RegionsView:
    """Adds ``self.regions_panel`` (call after ``build_option_sections``, which makes the width spins)."""

    def setup_regions_panel(self, parent: QWidget) -> QWidget:
        panel = QWidget(parent)
        panel.setObjectName("analyzeRegionsPanel")
        self.regions_panel = panel
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        header = QHBoxLayout()
        title = QLabel("Regions", panel)
        title.setProperty("gimapSectionTitle", True)
        self.region_add_button = QToolButton(panel)
        self.region_add_button.setObjectName("analyzeRegionAdd")
        self.region_add_button.setText("Add")
        self.region_add_button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.region_add_button)
        self.region_draw_action = menu.addAction("Draw on the Cake View")
        menu.addSeparator()
        self.region_add_ring_action = menu.addAction("Ring at the Strongest Peak")
        self.region_add_in_plane_action = menu.addAction("In-plane Band (|χ| 70–90°)")
        self.region_add_out_of_plane_action = menu.addAction("Out-of-plane Band (|χ| 0–20°)")
        self.region_add_whole_action = menu.addAction("Whole Pattern")
        menu.addSeparator()
        self.region_save_action = menu.addAction("Save Cuts…")
        self.region_load_action = menu.addAction("Load Cuts…")
        self.region_add_button.setMenu(menu)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.region_add_button)
        layout.addLayout(header)

        prompt = QLabel("Click on the image to add a cut:", panel)
        prompt.setProperty("gimapRole", "muted")
        layout.addWidget(prompt)
        picks = FlowLayout(spacing=6)
        self.region_pick_buttons: dict[str, QPushButton] = {}
        for key, text, tip in PICK_BUTTONS:
            button = QPushButton(text, panel)
            button.setObjectName(f"analyzeRegionPick_{key}")
            button.setCheckable(True)
            button.setToolTip(tip)
            picks.addWidget(button)
            self.region_pick_buttons[key] = button
        layout.addLayout(picks)

        self.region_list = QListWidget(panel)
        self.region_list.setObjectName("analyzeRegionList")
        self.region_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.region_list.setWordWrap(True)
        self.region_list.setToolTip("Tick a region to show its curves and outline; select it to change it below")
        layout.addWidget(self.region_list)

        self.region_editor = QStackedWidget(panel)
        self.region_editor.setObjectName("analyzeRegionEditor")
        self.region_editor_pages: dict[str, QWidget] = {}
        # note: the full ring
        note = QWidget(self.region_editor)
        note_layout = QVBoxLayout(note)
        note_layout.setContentsMargins(0, 0, 0, 0)
        self.region_note_label = QLabel("", note)
        self.region_note_label.setWordWrap(True)
        self.region_note_label.setProperty("gimapRole", "muted")
        note_layout.addWidget(self.region_note_label)
        # width: the in-plane and out-of-plane bands (the spins come from the options section)
        width = QWidget(self.region_editor)
        width_layout = QFormLayout(width)
        width_layout.setContentsMargins(0, 0, 0, 0)
        self.region_width_caption = QLabel("Half width", width)
        holder = QHBoxLayout()
        holder.addWidget(self.in_plane_spin)
        holder.addWidget(self.out_of_plane_spin)
        width_layout.addRow(self.region_width_caption, holder)
        # ring: the q window of I(χ)
        ring = QWidget(self.region_editor)
        ring_layout = QFormLayout(ring)
        ring_layout.setContentsMargins(0, 0, 0, 0)
        self.ring_q_center, self.ring_q_half = _q_center(ring), _q_half(ring)
        ring_layout.addRow("q", _plus_minus(self.ring_q_center, self.ring_q_half, ring, "Å⁻¹"))
        self.ring_auto_button = QPushButton("Strongest Ring", ring)
        self.ring_auto_button.setToolTip("Put the window back on the most prominent ring of I(q)")
        ring_layout.addRow("", self.ring_auto_button)
        # generic: a region the person added (or the custom sector)
        generic = QWidget(self.region_editor)
        generic_layout = QFormLayout(generic)
        generic_layout.setContentsMargins(0, 0, 0, 0)
        self.region_name_edit = QLineEdit(generic)
        self.region_name_edit.setObjectName("analyzeRegionName")
        self.region_q_center, self.region_q_half = _q_center(generic), _q_half(generic)
        self.region_all_q_check = QCheckBox("All q", generic)
        self.region_chi_center, self.region_chi_half = _chi_center(generic), _chi_half(generic)
        self.region_all_chi_check = QCheckBox("All χ", generic)
        self.region_both_check = QCheckBox("Both sides (±χ)", generic)
        self.region_both_check.setToolTip(
            "GIWAXS patterns are symmetric in ±q∥: take the χ range on both sides, folded onto |χ|"
        )
        chi_checks = QHBoxLayout()
        chi_checks.addWidget(self.region_all_chi_check)
        chi_checks.addWidget(self.region_both_check)
        chi_checks.addStretch(1)
        self.region_range_label = QLabel("", generic)
        self.region_range_label.setObjectName("analyzeRegionRange")
        self.region_range_label.setProperty("gimapRole", "muted")
        self.region_range_label.setWordWrap(True)
        self.region_snap_button = QPushButton("Snap to Peak", generic)
        self.region_snap_button.setObjectName("analyzeRegionSnap")
        self.region_snap_button.setToolTip(
            "Centre the window on the nearest peak and set its width from the peak (± FWHM)"
        )
        self.region_remove_button = QPushButton("Remove Region", generic)
        self.region_remove_button.setObjectName("analyzeRegionRemove")
        self.region_remove_button.setProperty("gimapDangerAction", True)
        actions = QHBoxLayout()
        actions.addWidget(self.region_snap_button)
        actions.addWidget(self.region_remove_button)
        actions.addStretch(1)
        generic_layout.addRow("Name", self.region_name_edit)
        generic_layout.addRow("q", _plus_minus(self.region_q_center, self.region_q_half, generic, "Å⁻¹"))
        generic_layout.addRow("", self.region_all_q_check)
        generic_layout.addRow("χ", _plus_minus(self.region_chi_center, self.region_chi_half, generic, "°"))
        generic_layout.addRow("", chi_checks)
        generic_layout.addRow("", self.region_range_label)
        generic_layout.addRow("", actions)
        for key, page in zip(EDITOR_PAGES, (note, width, ring, generic)):
            self.region_editor.addWidget(page)
            self.region_editor_pages[key] = page
        layout.addWidget(self.region_editor)
        hint = QLabel(
            "Drag a region on the Cake view (χ against q) to move or resize it. The q map shows it in the "
            "same colour, Sources shows its pixels on the detector.", panel,
        )
        hint.setWordWrap(True)
        hint.setProperty("gimapRole", "muted")
        hint.setAlignment(Qt.AlignLeft)
        layout.addWidget(hint)
        return panel


__all__ = ["EDITOR_PAGES", "PICK_BUTTONS", "RegionsView"]
