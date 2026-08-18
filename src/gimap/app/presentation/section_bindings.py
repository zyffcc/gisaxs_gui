"""Behavior bindings for section-shaped widgets declared in Python Views."""

from __future__ import annotations

from PyQt5.QtCore import Qt


def bind_parameter_section(
    section,
    title_label,
    description_label,
    content,
    content_layout,
) -> None:
    """Expose the standard section contract on a Python View-owned frame."""
    section.title_label = title_label
    section.description_label = description_label
    section.content = content
    section.content_layout = content_layout


def bind_advanced_section(
    section,
    toggle_button,
    description_label,
    content,
    content_layout,
) -> None:
    """Bind collapsible behavior without rebuilding the Python View layout."""
    section.toggle_button = toggle_button
    section.description_label = description_label
    section.content = content
    section.content_layout = content_layout

    def set_expanded(expanded: bool) -> None:
        expanded = bool(expanded)
        toggle_button.blockSignals(True)
        toggle_button.setChecked(expanded)
        toggle_button.blockSignals(False)
        toggle_button.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        content.setVisible(expanded)
        description_label.setVisible(bool(description_label.text()) and expanded)

    section.is_expanded = toggle_button.isChecked
    section.set_expanded = set_expanded
    toggle_button.toggled.connect(set_expanded)
    set_expanded(toggle_button.isChecked())
