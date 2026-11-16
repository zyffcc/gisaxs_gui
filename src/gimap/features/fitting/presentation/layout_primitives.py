"""Small widgets owned by the Fitting presentation."""

from __future__ import annotations

from PyQt5.QtGui import QValidator

from src.gimap.app.presentation import SafeWheelDoubleSpinBox


class ScientificDoubleSpinBox(SafeWheelDoubleSpinBox):
    """High-precision spin box with compact general/scientific display."""

    def textFromValue(self, value: float) -> str:
        precision = 12 if self.hasFocus() else 6
        return f"{float(value):.{precision}g}"

    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        self.lineEdit().setText(f"{float(self.value()):.12g}")
        self.lineEdit().selectAll()

    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        self.lineEdit().setText(self.textFromValue(self.value()))

    def valueFromText(self, text: str) -> float:
        cleaned = str(text).strip()
        value, accepted = self.locale().toDouble(cleaned)
        return float(value) if accepted else float(cleaned)

    def validate(self, text: str, position: int):
        cleaned = str(text).strip()
        if cleaned in {"", "+", "-", ".", "+.", "-."} or cleaned.lower().endswith(
            ("e", "e+", "e-")
        ):
            return QValidator.Intermediate, text, position
        try:
            value = self.valueFromText(cleaned)
        except ValueError:
            return QValidator.Invalid, text, position
        state = (
            QValidator.Acceptable
            if self.minimum() <= value <= self.maximum()
            else QValidator.Intermediate
        )
        return state, text, position


__all__ = ["ScientificDoubleSpinBox"]
