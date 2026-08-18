"""Small typed widget factories for WAXS presentation."""

from __future__ import annotations


from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QSizePolicy,
)


def make_double_spin(minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setDecimals(9)
    spin.setSingleStep(0.1)
    spin.setValue(value)
    spin.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return spin
