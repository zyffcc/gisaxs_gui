"""Shared sizing helpers for wrapper-built UI components."""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QWidget,
)


BUTTON_HEIGHT = 32
INPUT_HEIGHT = 28
SMALL_BUTTON_WIDTH = 36
CARD_MARGIN = 14
CARD_SPACING = 10
FORM_ROW_SPACING = 8
SECTION_MIN_WIDTH = 360


def normalize_button(button: QAbstractButton, compact: bool = False, wide: bool = False) -> None:
    """Apply consistent button sizing without changing object names."""
    if compact:
        button.setMinimumSize(SMALL_BUTTON_WIDTH, BUTTON_HEIGHT)
        button.setMaximumSize(SMALL_BUTTON_WIDTH, BUTTON_HEIGHT)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        return

    button.setMinimumHeight(BUTTON_HEIGHT)
    button.setMaximumHeight(BUTTON_HEIGHT + 4)
    button.setMinimumWidth(96 if wide else 72)
    button.setMaximumWidth(220 if wide else 160)
    horizontal = QSizePolicy.MinimumExpanding if wide else QSizePolicy.Preferred
    button.setSizePolicy(horizontal, QSizePolicy.Fixed)


def normalize_input(widget: QWidget) -> None:
    """Make input-like controls expand horizontally but not vertically."""
    widget.setMinimumHeight(INPUT_HEIGHT)
    widget.setMaximumHeight(INPUT_HEIGHT + 4)
    widget.setMinimumWidth(max(widget.minimumWidth(), 72))
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


def set_expanding_x(widget: QWidget) -> None:
    widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)


def normalize_checkbox(widget: QCheckBox) -> None:
    widget.setMinimumHeight(INPUT_HEIGHT)
    widget.setMaximumHeight(INPUT_HEIGHT + 6)
    widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)


def make_scroll_area(widget: QWidget, horizontal: bool = False) -> QScrollArea:
    """Wrap a dense widget so overflow scrolls instead of overlapping."""
    scroll_area = QScrollArea()
    scroll_area.setWidgetResizable(True)
    scroll_area.setHorizontalScrollBarPolicy(
        Qt.ScrollBarAsNeeded if horizontal else Qt.ScrollBarAlwaysOff
    )
    scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    scroll_area.setWidget(widget)
    return scroll_area


INPUT_WIDGET_TYPES = (QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox)


def collect_layout_diagnostics(root: QWidget) -> dict[str, list[str]]:
    """Return layout risks that should be reviewed during UI work."""
    oversized_fixed = []
    vertical_expanding_buttons = []

    for widget in root.findChildren(QWidget):
        name = widget.objectName() or widget.__class__.__name__
        size_policy = widget.sizePolicy()
        if isinstance(widget, QAbstractButton):
            if size_policy.verticalPolicy() == QSizePolicy.Expanding:
                vertical_expanding_buttons.append(name)
            if widget.maximumHeight() < 16777215 and widget.maximumHeight() > BUTTON_HEIGHT + 8:
                oversized_fixed.append(name)

    return {
        "fixed_oversized_dimensions": oversized_fixed,
        "vertical_expanding_buttons": vertical_expanding_buttons,
        "absolute_geometry": [
            "Generated ui/main_window.py contains setup-time geometry; wrapper code must not edit it directly."
        ],
    }
