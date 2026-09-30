"""Short, non-blocking notices that float over a page and fade away ("Exported 6 curves …").

``show_toast(parent, text, level="ok", action=("Open Folder", callback))`` puts a card at the
bottom right of ``parent``; it disappears after a few seconds or when closed, never takes the
focus and never blocks work. Newer notices stack above older ones.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QEvent, QObject, Qt, QTimer
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolButton, QWidget

LEVELS = ("info", "ok", "warning", "error")
DEFAULT_TIMEOUT_MS = 5000
MARGIN = 16
SPACING = 8


class Toast(QWidget):
    def __init__(
        self,
        parent: QWidget,
        text: str,
        *,
        level: str = "info",
        action: Optional[tuple[str, Callable[[], None]]] = None,
        timeout_ms: int = DEFAULT_TIMEOUT_MS,
    ):
        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setProperty("gimapToast", True)
        self.setProperty("level", level if level in LEVELS else "info")
        self.setFocusPolicy(Qt.NoFocus)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 8, 10)
        layout.setSpacing(10)
        self.label = QLabel(text, self)
        self.label.setWordWrap(True)
        self.label.setProperty("gimapToastText", True)
        self.label.setMaximumWidth(420)
        layout.addWidget(self.label, 1)
        self.action_button: Optional[QPushButton] = None
        if action is not None:
            title, callback = action
            self.action_button = QPushButton(title, self)
            self.action_button.setProperty("gimapToastAction", True)
            self.action_button.setFocusPolicy(Qt.NoFocus)
            self.action_button.clicked.connect(callback)
            self.action_button.clicked.connect(self.close)
            layout.addWidget(self.action_button)
        close = QToolButton(self)
        close.setText("×")
        close.setAutoRaise(True)
        close.setFocusPolicy(Qt.NoFocus)
        close.setToolTip("Close")
        close.clicked.connect(self.close)
        layout.addWidget(close)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.close)
        if timeout_ms > 0:
            self._timer.start(int(timeout_ms))
        self.adjustSize()

    def text(self) -> str:
        return self.label.text()

    def enterEvent(self, event) -> None:  # keep it while the pointer is on it
        self._timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if not self._timer.isActive():
            self._timer.start(2000)
        super().leaveEvent(event)

    def closeEvent(self, event) -> None:
        stack = _stacks.get(id(self.parentWidget()))
        if stack is not None:
            stack.remove(self)
        super().closeEvent(event)


class _Stack(QObject):
    """Keeps the toasts of one parent in a column at its bottom-right corner."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.parent_widget = parent
        self.toasts: list[Toast] = []
        parent.installEventFilter(self)

    def add(self, toast: Toast) -> None:
        self.toasts.append(toast)
        self.layout()

    def remove(self, toast: Toast) -> None:
        if toast in self.toasts:
            self.toasts.remove(toast)
        self.layout()

    def layout(self) -> None:
        bottom = self.parent_widget.height() - MARGIN
        for toast in reversed(self.toasts):
            toast.adjustSize()
            width = min(toast.sizeHint().width(), max(200, self.parent_widget.width() - 2 * MARGIN))
            toast.resize(width, toast.sizeHint().height())
            bottom -= toast.height()
            toast.move(self.parent_widget.width() - MARGIN - toast.width(), bottom)
            toast.raise_()
            bottom -= SPACING

    def eventFilter(self, watched, event) -> bool:
        if watched is self.parent_widget and event.type() == QEvent.Resize:
            self.layout()
        return False


_stacks: dict[int, _Stack] = {}


def show_toast(
    parent: QWidget,
    text: str,
    *,
    level: str = "info",
    action: Optional[tuple[str, Callable[[], None]]] = None,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> Toast:
    stack = _stacks.get(id(parent))
    if stack is None:
        stack = _stacks[id(parent)] = _Stack(parent)
        parent.destroyed.connect(lambda *_args, key=id(parent): _stacks.pop(key, None))
    toast = Toast(parent, text, level=level, action=action, timeout_ms=timeout_ms)
    toast.show()
    stack.add(toast)
    return toast


def visible_toasts(parent: QWidget) -> list[Toast]:
    stack = _stacks.get(id(parent))
    return list(stack.toasts) if stack is not None else []


__all__ = ["Toast", "show_toast", "visible_toasts"]
