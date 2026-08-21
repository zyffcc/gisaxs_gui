"""Reusable numeric inputs that do not steal ordinary page scrolling."""

from __future__ import annotations

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QWidget,
)


WHEEL_INPUT_HINT = "Hold Alt/Option while scrolling to change this value."


def _nearest_scroll_area(widget: QWidget) -> QAbstractScrollArea | None:
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            return parent
        parent = parent.parentWidget()
    return None


def _scroll_ancestor(widget: QWidget, event) -> bool:
    scroll_area = _nearest_scroll_area(widget)
    if scroll_area is None:
        event.ignore()
        return True

    bar = scroll_area.verticalScrollBar()
    pixel_delta = event.pixelDelta().y()
    angle_delta = event.angleDelta().y()
    if pixel_delta:
        movement = pixel_delta
    elif angle_delta:
        movement = int((angle_delta / 120.0) * max(24, bar.singleStep() * 3))
    else:
        movement = 0
    bar.setValue(bar.value() - movement)
    event.accept()
    return True


def _wheel_may_edit(widget: QWidget, event) -> bool:
    return widget.hasFocus() and bool(event.modifiers() & Qt.AltModifier)


class SafeWheelInputFilter(QObject):
    """Require an explicit modifier before wheel events edit an input."""

    def eventFilter(self, watched, event):
        if event.type() != QEvent.Wheel:
            return False
        if not isinstance(watched, (QAbstractSpinBox, QComboBox)):
            return False
        if _wheel_may_edit(watched, event):
            return False
        return _scroll_ancestor(watched, event)


def install_safe_wheel_behavior(root: QWidget) -> SafeWheelInputFilter:
    """Protect all current spin boxes and combo boxes below ``root``.

    The returned filter is also retained on ``root``. Call this again for a
    dynamically inserted subtree; repeated installation on the same root is safe.
    """

    guard = getattr(root, "_gimap_safe_wheel_guard", None)
    if not isinstance(guard, SafeWheelInputFilter):
        guard = SafeWheelInputFilter(root)
        root._gimap_safe_wheel_guard = guard
    inputs = list(root.findChildren(QAbstractSpinBox))
    inputs.extend(root.findChildren(QComboBox))
    if isinstance(root, (QAbstractSpinBox, QComboBox)):
        inputs.append(root)
    for input_widget in inputs:
        input_widget.installEventFilter(guard)
        input_widget.setProperty("gimapSafeWheelInput", True)
        input_widget.setStatusTip(WHEEL_INPUT_HINT)
    return guard


class _SafeWheelMixin:
    def wheelEvent(self, event) -> None:
        if _wheel_may_edit(self, event):
            super().wheelEvent(event)
            return
        _scroll_ancestor(self, event)


class SafeWheelDoubleSpinBox(_SafeWheelMixin, QDoubleSpinBox):
    """Double spin box whose wheel editing requires Alt/Option and focus."""


class SafeWheelSpinBox(_SafeWheelMixin, QSpinBox):
    """Integer spin box whose wheel editing requires Alt/Option and focus."""


class SafeWheelComboBox(_SafeWheelMixin, QComboBox):
    """Combo box whose wheel selection requires Alt/Option and focus."""


__all__ = [
    "SafeWheelComboBox",
    "SafeWheelDoubleSpinBox",
    "SafeWheelInputFilter",
    "SafeWheelSpinBox",
    "install_safe_wheel_behavior",
]
