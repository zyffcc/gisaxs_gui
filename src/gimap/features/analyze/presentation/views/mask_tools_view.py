"""Controls for editing the ordered user mask in Analyze."""

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel, QListWidget,
    QPushButton, QSpinBox, QWidget,
)

from src.gimap.app.presentation.components import FlowLayout


SHAPES = {
    "rectangle": "Rectangle", "polygon": "Polygon", "circle": "Circle", "annulus": "Annulus",
    "q_annulus": "Beam-centred q annulus", "ellipse": "Ellipse", "brush": "Brush", "eraser": "Eraser",
}
RULES = {
    "above": "Above threshold", "below": "Below threshold", "saturated": "Saturated pixels",
    "negative": "Negative pixels", "grow": "Grow mask", "shrink": "Shrink mask", "invert": "Invert mask",
}


def _number(page, low, high, value, suffix=""):
    spin = QDoubleSpinBox(page)
    spin.setDecimals(6)
    spin.setRange(low, high)
    spin.setValue(value)
    spin.setSuffix(suffix)
    return spin


def add_mask_editor(view, page, layout):
    title = QLabel("User mask", page)
    title.setProperty("gimapSectionTitle", True)
    layout.addWidget(title)
    row = QHBoxLayout()
    view.mask_tool = QComboBox(page)
    for key, label in SHAPES.items():
        view.mask_tool.addItem(label, key)
    view.mask_tool.insertSeparator(view.mask_tool.count())
    for key, label in RULES.items():
        view.mask_tool.addItem(label, key)
    row.addWidget(view.mask_tool, 1)
    view.mask_draw = QPushButton("Draw", page)
    view.mask_draw.setCheckable(True)
    row.addWidget(view.mask_draw)
    layout.addLayout(row)
    view.mask_mode = QComboBox(page)
    view.mask_mode.addItem("Exclude pixels", "exclude")
    view.mask_mode.addItem("Include pixels", "include")
    layout.addWidget(view.mask_mode)
    view.mask_parameters = QWidget(page)
    form = QFormLayout(view.mask_parameters)
    form.setContentsMargins(0, 0, 0, 0)
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    view.mask_radius = _number(page, 0.01, 10000, 3, " px")
    view.mask_amount = QSpinBox(page)
    view.mask_amount.setRange(1, 10000)
    view.mask_amount.setSuffix(" px")
    view.mask_threshold = _number(page, -1e15, 1e15, 1)
    view.mask_q_min = _number(page, 0, 1000, 0.05, " Å⁻¹")
    view.mask_q_max = _number(page, 0, 1000, 0.1, " Å⁻¹")
    view.mask_fields = [
        ("Radius", view.mask_radius, ("brush", "eraser")),
        ("Distance", view.mask_amount, ("grow", "shrink")),
        ("Threshold", view.mask_threshold, ("above", "below", "saturated")),
        ("Inner q", view.mask_q_min, ("q_annulus",)), ("Outer q", view.mask_q_max, ("q_annulus",)),
    ]
    for label, spin, _kinds in view.mask_fields:
        form.addRow(label, spin)
    layout.addWidget(view.mask_parameters)
    view.mask_relative = QCheckBox("Multiple of the median", page)
    layout.addWidget(view.mask_relative)
    view.mask_preview_label = QLabel(page)
    view.mask_preview_label.setWordWrap(True)
    layout.addWidget(view.mask_preview_label)
    view.mask_preview = QPushButton("Preview mask", page)
    view.mask_apply = QPushButton("Apply mask edit", page)
    view.mask_cancel = QPushButton("Cancel", page)
    edit_row = FlowLayout()
    for button in (view.mask_preview, view.mask_apply, view.mask_cancel):
        edit_row.addWidget(button)
    layout.addLayout(edit_row)
    view.mask_list = QListWidget(page)
    view.mask_list.setObjectName("analyzeMaskList")
    view.mask_list.setMaximumHeight(96)
    layout.addWidget(view.mask_list)
    actions = FlowLayout()
    view.mask_load_button = QPushButton("Load…", page)
    view.mask_save_button = QPushButton("Save…", page)
    view.mask_remove_button = QPushButton("Remove Selected", page)
    view.mask_clear_button = QPushButton("Clear Masks", page)
    view.mask_clear_button.setToolTip("Clearing leaves 0 pixels in the user mask. Detector exclusions still apply.")
    for button in (view.mask_load_button, view.mask_save_button, view.mask_remove_button, view.mask_clear_button):
        actions.addWidget(button)
    layout.addLayout(actions)
