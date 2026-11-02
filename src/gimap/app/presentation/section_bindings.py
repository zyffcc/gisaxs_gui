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
    """Expose the standard section contract on a Python View-owned frame.

    As in ``ParameterSection``, the description is the title's tooltip and not a line on screen."""
    from .components.sections import describe_title

    section.title_label = title_label
    section.description_label = description_label
    section.content = content
    section.content_layout = content_layout
    if description_label is not None:
        if description_label.text() and not title_label.toolTip():
            describe_title(title_label, description_label.text())
        description_label.hide()


def bind_advanced_section(
    section,
    toggle_button,
    description_label,
    content,
    content_layout,
) -> None:
    """Bind collapsible behavior without rebuilding the Python View layout.

    As in ``AdvancedSection``, the description is the button's tooltip and not a line under it."""
    if description_label is not None:
        if description_label.text() and not toggle_button.toolTip():
            toggle_button.setToolTip(description_label.text())
        description_label.hide()
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

    section.is_expanded = toggle_button.isChecked
    section.set_expanded = set_expanded
    toggle_button.toggled.connect(set_expanded)
    set_expanded(toggle_button.isChecked())
