"""Shared pixel step control for the manual beam-centre editors."""

from __future__ import annotations

from PyQt5.QtWidgets import QDoubleSpinBox, QWidget

from src.gimap.app.presentation.i18n import tr


def make_center_step(
    parent: QWidget,
    x_spin: QDoubleSpinBox,
    y_spin: QDoubleSpinBox,
    initial_step: float = 1.0,
) -> QDoubleSpinBox:
    """Make one editable step for the two coordinates' arrows, keys and wheel."""
    step_spin = QDoubleSpinBox(parent)
    step_spin.setObjectName("analyzeCenterStep")
    step_spin.setDecimals(4)
    step_spin.setRange(0.0001, 1000.0)
    step_spin.setSingleStep(0.0001)
    step_spin.setSuffix(" px")
    step_spin.setKeyboardTracking(False)
    step_spin.setToolTip(tr(
        "Step used by the arrow buttons, Up/Down keys, and mouse wheel "
        "for both beam-centre coordinates."
    ))

    def apply_step(value: float) -> None:
        x_spin.setSingleStep(value)
        y_spin.setSingleStep(value)

    step_spin.valueChanged.connect(apply_step)
    step_spin.setValue(float(initial_step))
    # setValue can leave the minimum unchanged, without emitting valueChanged.
    apply_step(step_spin.value())
    return step_spin


__all__ = ["make_center_step"]
