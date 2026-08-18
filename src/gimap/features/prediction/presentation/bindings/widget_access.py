"""Widget Access coordination for Prediction."""

from __future__ import annotations


from typing import Optional


from PyQt5.QtCore import QSignalBlocker


from PyQt5.QtWidgets import (
    QDoubleSpinBox,
)


class WidgetAccessMixin:
    """Own widget access presentation behavior."""

    def _set_line_edit(self, name: str, text: Optional[str]) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        blocker = QSignalBlocker(widget)
        widget.setText(text or "")
        del blocker

    def _get_line_edit_text(self, name: str) -> str:
        widget = getattr(self.ui, name, None)
        return widget.text().strip() if widget else ""

    def _set_checkbox(self, name: str, checked: bool) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None:
            return
        blocker = QSignalBlocker(widget)
        widget.setChecked(bool(checked))
        del blocker

    def _set_double_spin(self, name: str, value: Optional[float]) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None or value is None:
            return
        blocker = QSignalBlocker(widget)
        widget.setValue(float(value))
        del blocker

    def _configure_color_spin(self, name: str) -> None:
        widget = getattr(self.ui, name, None)
        if not isinstance(widget, QDoubleSpinBox):
            return
        widget.setDecimals(6)
        widget.setRange(-1e12, 1e12)
        widget.setSingleStep(0.1)

    def _set_combobox_text(self, name: str, text: str) -> None:
        widget = getattr(self.ui, name, None)
        if widget is None or text is None:
            return
        blocker = QSignalBlocker(widget)
        index = widget.findText(text)
        widget.setCurrentIndex(index if index >= 0 else 0)
        del blocker

    def _get_double_spin_value(self, name: str) -> Optional[float]:
        widget = getattr(self.ui, name, None)
        return float(widget.value()) if widget is not None else None

    def _append_status_message(self, message: str, level: str = "INFO") -> None:
        self.status_updated.emit(message)
        browser = getattr(self.ui, "predictStatusTextBrowser", None)
        line = f"[{level}] {message}"
        if browser is not None:
            browser.append(line)
        if self._status_text_window_browser is not None:
            self._status_text_window_browser.append(line)
