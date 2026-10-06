"""Short, non-blocking notices that float over a page and fade away ("Exported 6 curves …").

``show_toast(parent, text, level="ok", action=("Open Folder", callback))`` puts a card at the
bottom right of ``parent``; it disappears after a few seconds or when closed, never takes the
focus and never blocks work. Newer notices stack above older ones; at most ``MAX_TOASTS`` are open
(the oldest go first, errors last) and the same text is shown once. An error stays until it is closed, a
warning stays longer than other notices. Texts are translated here (toasts are child widgets: the
language walker, which translates windows as they are shown, never sees them).
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import QEvent, QObject, Qt, QTimer
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolButton, QWidget

LEVELS = ("info", "ok", "warning", "error")
DEFAULT_TIMEOUT_MS = 5000
WARNING_TIMEOUT_MS = 10000
MAX_TOASTS = 3
TEXT_WIDTH = 420
MARGIN = 16
SPACING = 8


def default_timeout(level: str) -> int:
    """How long a notice of ``level`` stays (ms; 0: until it is closed)."""
    return 0 if level == "error" else WARNING_TIMEOUT_MS if level == "warning" else DEFAULT_TIMEOUT_MS


class Toast(QWidget):
    def __init__(
        self,
        parent: QWidget,
        text: str,
        *,
        level: str = "info",
        action: Optional[tuple[str, Callable[[], None]]] = None,
        timeout_ms: Optional[int] = None,
    ):
        """``timeout_ms``: ``None`` for the level's default (``default_timeout``), 0 to stay until closed."""
        from ..i18n import tr

        super().__init__(parent)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self.setProperty("gimapToast", True)
        self.setProperty("level", level if level in LEVELS else "info")
        self.setFocusPolicy(Qt.NoFocus)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 10, 8, 10)
        layout.setSpacing(10)
        self.label = QLabel(tr(text), self)
        self.label.setWordWrap(True)
        self.label.setProperty("gimapToastText", True)
        self.label.setMaximumWidth(TEXT_WIDTH)
        self.label.ensurePolished()  # the style sheet's font, before measuring
        metrics = self.label.fontMetrics()
        widest = max((metrics.horizontalAdvance(line) for line in self.label.text().split("\n")), default=0)
        self.preferred_text_width = min(widest + 4, TEXT_WIDTH)
        """A short notice on one line, a long one wraps; a narrow parent narrows it (``fit_text``)."""
        self.label.setMinimumWidth(self.preferred_text_width)
        layout.addWidget(self.label, 1)
        self.action_button: Optional[QPushButton] = None
        if action is not None:
            title, callback = action
            self.action_button = QPushButton(tr(title), self)
            self.action_button.setProperty("gimapToastAction", True)
            self.action_button.setFocusPolicy(Qt.NoFocus)
            self.action_button.clicked.connect(callback)
            self.action_button.clicked.connect(self.close)
            layout.addWidget(self.action_button)
        close = QToolButton(self)
        close.setText("×")
        close.setAutoRaise(True)
        close.setFocusPolicy(Qt.NoFocus)
        close.setToolTip(tr("Close"))
        close.clicked.connect(self.close)
        layout.addWidget(close)
        self.close_button = close
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.close)
        self.timeout_ms = default_timeout(level) if timeout_ms is None else int(timeout_ms)
        if self.timeout_ms > 0:
            self._timer.start(self.timeout_ms)
        self.adjustSize()

    def text(self) -> str:
        return self.label.text()

    def fit_text(self, width: int) -> None:
        """The text on one line only while the whole toast fits in ``width``; narrower, it wraps (the
        buttons beside it are never drawn over it)."""
        layout = self.layout()
        margins = layout.contentsMargins()
        buttons = [button for button in (self.action_button, self.close_button) if button is not None]
        beside = margins.left() + margins.right() + sum(layout.spacing() + button.sizeHint().width() for button in buttons)
        self.label.setMinimumWidth(min(self.preferred_text_width, max(0, int(width) - beside)))

    def enterEvent(self, event) -> None:  # keep it while the pointer is on it
        self._timer.stop()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self.timeout_ms > 0 and not self._timer.isActive():  # one that stays until closed still stays
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
        """The same notice is shown once (the new one replaces it); at most ``MAX_TOASTS``: the oldest go
        first, notices before errors (an error stays until it is read)."""
        for old in [shown for shown in self.toasts if shown.text() == toast.text()]:
            self._close(old)
        while len(self.toasts) >= MAX_TOASTS:
            older = [shown for shown in self.toasts if shown.property("level") != "error"] or self.toasts
            self._close(older[0])
        self.toasts.append(toast)
        self.layout()

    def _close(self, toast: Toast) -> None:
        toast.close()
        if toast in self.toasts:  # ``closeEvent`` normally removes it
            self.toasts.remove(toast)

    def remove(self, toast: Toast) -> None:
        if toast in self.toasts:
            self.toasts.remove(toast)
        self.layout()

    def layout(self) -> None:
        bottom = self.parent_widget.height() - MARGIN
        available = max(200, self.parent_widget.width() - 2 * MARGIN)
        for toast in reversed(self.toasts):
            toast.fit_text(available)
            toast.adjustSize()
            width = min(toast.sizeHint().width(), available)
            height = toast.heightForWidth(width) if toast.hasHeightForWidth() else -1  # a wrapped text: its lines
            toast.resize(width, height if height > 0 else toast.sizeHint().height())
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
    timeout_ms: Optional[int] = None,
) -> Toast:
    """``timeout_ms``: ``None`` — until closed for an error, 10 s for a warning, 5 s otherwise; 0 — until closed."""
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


__all__ = ["MAX_TOASTS", "Toast", "default_timeout", "show_toast", "visible_toasts"]
