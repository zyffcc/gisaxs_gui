"""Short, non-blocking notices that float over a page and fade away ("Exported 6 curves …").

``show_toast(parent, text, level="ok", action=("Open Folder", callback))`` puts a card at the
bottom right of ``parent``; it disappears after a few seconds or when closed, never takes the
focus and never blocks work. The newest notice is in the corner and older ones stand to its left, in one
row along the bottom edge: a page keeps its status line there, under its plots, so the bottom axis and the
legend of the plot at the bottom right stay in view (notices stacked upwards covered them). A text is on
one line while the row has room for it; only a longer one wraps.

``set_toast_inset(parent, left, band=…)`` keeps the row off what stands at the left of ``parent`` (the main
window's sidebar) and tells it where to look for the status line of the page shown (``toast_band.py``): the
row never covers the progress bar and Cancel at the right end of that line — it ends before them, or lies
above them when they leave it no room — and keeps right of the line's text while there is room. The row
looks at the band again every ``LOOK_MS`` while notices are shown (a job starts, a sidebar collapses).

When the row is full the newest notice keeps its whole text; of the older ones, a plain confirmation goes
(its page's status line said the same), a warning or a notice with a button is shortened (its whole text in
the tooltip), or goes when not even that fits — a row above would lie over the plot's axis again. Only an
error never goes: it is shortened over the end of the status line, and one that does not fit even so starts
a row above. A status line that says what the newest notice says is no loss: the older ones may lie over it.
Without ``band`` the row looks at the parent's own status line (a page that shows notices itself); the
notices of a window and of a page in it keep off each other, whoever came first keeping the corner.

At most ``MAX_TOASTS`` are open (the oldest go first, errors last) and the same text is shown once. An error
stays until it is closed, a warning stays longer than other notices. Texts are translated here (toasts are
child widgets: the language walker, which translates windows as they are shown, never sees them).

``dismiss_toasts(parent)`` closes the plain confirmations (“Saved …”, “3 series compared”) when what they
were about is no longer shown — the main window calls it when the page changes; warnings, errors and
notices with a button (Undo, Open Folder, Details) stay.
"""

from __future__ import annotations

from typing import Callable, Iterable, Optional

from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, Qt, QTimer
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QToolButton, QWidget

from .toast_band import BandContent, band_content

LEVELS = ("info", "ok", "warning", "error")
DEFAULT_TIMEOUT_MS = 5000
WARNING_TIMEOUT_MS = 10000
MAX_TOASTS = 3
MIN_ROW = 200
"""The narrowest row: a parent narrower than this still gets a notice of this width."""
SHORTEST_TEXT = 96
"""The text a shortened notice keeps (pixels, “…” included)."""
BAND_HEIGHT = 28
"""The rows above the bottom of the row that are looked into for the status line (one line of a notice)."""
TEXT_STEP = 40
"""The end of the status text is taken in steps of this (a count in it does not move the notices each time)."""
LOOK_MS = 400
"""How often the row looks at the band again while notices are shown (a look: a few milliseconds)."""
WIDEST = 16777215  # (Qt's QWIDGETSIZE_MAX)
MARGIN = 16
"""To the left and right edges of the parent."""
BOTTOM_MARGIN = 3
"""To its bottom edge: a notice of one line is as tall as the band of a page's status line, and lies in it."""
SPACING = 8
BUTTON_HEIGHT = 22


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
        layout.setContentsMargins(12, 4, 6, 4)
        layout.setSpacing(8)
        self._text = tr(text)
        self.label = QLabel(self._text, self)
        self.label.setWordWrap(True)
        self.label.setProperty("gimapToastText", True)
        self.label.ensurePolished()  # the style sheet's font, before measuring
        metrics = self.label.fontMetrics()
        widest = max((metrics.horizontalAdvance(line) for line in self._text.split("\n")), default=0)
        self.preferred_text_width = widest + 4
        """The text on one line; a row narrower than that makes it wrap (``fit_text``) or shortens it."""
        self.label.setMinimumWidth(self.preferred_text_width)
        layout.addWidget(self.label, 1)
        self.action_button: Optional[QPushButton] = None
        if action is not None:
            title, callback = action
            self.action_button = QPushButton(tr(title), self)
            self.action_button.setProperty("gimapToastAction", True)
            self.action_button.setFocusPolicy(Qt.NoFocus)
            self.action_button.setFixedHeight(max(BUTTON_HEIGHT, self.label.fontMetrics().height() + 6))
            self.action_button.clicked.connect(callback)
            self.action_button.clicked.connect(self.close)
            layout.addWidget(self.action_button)
        close = QToolButton(self)
        close.setText("×")
        close.setAutoRaise(True)
        close.setFocusPolicy(Qt.NoFocus)
        close.setToolTip(tr("Close"))
        close.setFixedSize(BUTTON_HEIGHT, BUTTON_HEIGHT)
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
        """The whole text (also while the notice shows it shortened)."""
        return self._text

    def _beside(self) -> int:
        """The width of what stands beside the text: the margins and the buttons."""
        layout = self.layout()
        margins = layout.contentsMargins()
        buttons = [button for button in (self.action_button, self.close_button) if button is not None]
        return margins.left() + margins.right() + sum(layout.spacing() + button.sizeHint().width() for button in buttons)

    def fit_text(self, width: int) -> None:
        """The text on one line only while the whole toast fits in ``width``; narrower, it wraps (the
        buttons beside it are never drawn over it)."""
        self.label.setMinimumWidth(min(self.preferred_text_width, max(0, int(width) - self._beside())))

    def show_whole(self, available: int) -> int:
        """The whole text, on one line while it fits in ``available`` pixels, else wrapped; returns the width."""
        if self.label.text() != self._text or not self.label.wordWrap():
            self.label.setWordWrap(True)
            self.label.setText(self._text)
            self.setToolTip("")
        self.label.setMaximumWidth(WIDEST)
        self.fit_text(available)
        self.adjustSize()
        width = min(self.sizeHint().width(), available)
        height = self.heightForWidth(width) if self.hasHeightForWidth() else -1  # a wrapped text: its lines
        self.resize(width, height if height > 0 else self.sizeHint().height())
        return self.width()

    def shortest(self) -> int:
        """The narrowest the notice is shortened to (``show_short``)."""
        return self._beside() + SHORTEST_TEXT

    def show_short(self, width: int) -> int:
        """The text on one line, cut with “…” to make the notice ``width`` pixels wide (at least
        ``shortest``); the whole text is its tooltip. Returns the width."""
        room = max(SHORTEST_TEXT, int(width) - self._beside())
        line = " ".join(self._text.split())
        self.label.setWordWrap(False)
        self.label.setText(self.label.fontMetrics().elidedText(line, Qt.ElideRight, room - 4))
        self.label.setFixedWidth(room)
        self.setToolTip(self._text)
        self.adjustSize()
        self.resize(self.sizeHint())
        return self.width()

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


def _plain(toast: Toast) -> bool:
    """A confirmation that can go without loss: no warning or error, and no button that would go with it."""
    return toast.property("level") in ("info", "ok") and toast.action_button is None


class _Stack(QObject):
    """Keeps the toasts of one parent in a row along its bottom edge, the newest in the bottom-right corner."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.parent_widget = parent
        self.toasts: list[Toast] = []
        self.inset: Callable[[], int] = lambda: 0
        """Pixels at the left of the parent the row keeps off (``set_toast_inset``)."""
        self.band: Optional[Callable[[], Iterable[QWidget]]] = None
        """The widgets whose status line lies in the band of the row (``set_toast_inset``); ``None``: the
        parent's own (a page that shows notices itself keeps them off its own progress bar and Cancel)."""
        self._laid: Optional[tuple] = None
        """The bounds of the row the toasts were laid out in (``_bounds``)."""
        self._look = QTimer(self)
        self._look.setInterval(LOOK_MS)
        self._look.timeout.connect(self._look_again)
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
        self._look.start()

    def _close(self, toast: Toast) -> None:
        toast.close()
        if toast in self.toasts:  # ``closeEvent`` normally removes it
            self.toasts.remove(toast)

    def remove(self, toast: Toast) -> None:
        if toast in self.toasts:
            self.toasts.remove(toast)
        if not self.toasts:
            self._look.stop()
        self.layout()

    def _bounds(self) -> tuple[int, int, int, int]:
        """``(left, keep off up to, right edge, bottom)`` of the row: off the sidebar, ending before the
        controls at the right end of the band (or above them when they leave no room), right of the status
        text while there is room for it — unless that text is the newest notice's own."""
        parent = self.parent_widget
        try:
            left = MARGIN + max(0, int(self.inset()))
        except RuntimeError:  # what stood at the left is gone
            left = MARGIN
        edge, bottom = parent.width() - MARGIN, parent.height() - BOTTOM_MARGIN
        text_end = left
        try:
            roots = self.band() if self.band is not None else [parent]
            found = band_content(roots, parent, bottom - BAND_HEIGHT, bottom, self._others())
        except RuntimeError:  # a page that is gone with its window
            found = BandContent()
        if found.controls_left is not None:
            if found.controls_left - SPACING - left >= MIN_ROW:
                edge = found.controls_left - SPACING
            else:  # (above the band the status text is not under the row)
                return left, left, edge, found.controls_top - SPACING
        said = " ".join(found.text.split())
        if found.text_end is not None and not (self.toasts and said == " ".join(self.toasts[-1].text().split())):
            # (a status line that says what the newest notice says is no loss under the notices)
            text_end = max(left, -(-found.text_end // TEXT_STEP) * TEXT_STEP + SPACING)
        return left, text_end, edge, bottom

    def _others(self) -> list[QRect]:
        """The notices of the other widgets of this window (a page that shows its own, the window's), where
        they lie in the parent: the row keeps off them as off a Cancel (whoever was there first keeps the
        corner)."""
        parent = self.parent_widget
        window = parent.window()
        origin = parent.mapTo(window, QPoint(0, 0)) if parent is not window else QPoint(0, 0)
        return [QRect(toast.mapTo(window, QPoint(0, 0)) - origin, toast.size())
                for stack in list(_stacks.values()) if stack is not self
                for toast in stack.toasts if toast.isVisible() and toast.window() is window]

    def _look_again(self) -> None:
        """Lay the toasts out again when what they keep off has changed (a job started or ended, the status
        text grew, the sidebar was collapsed)."""
        if not self.toasts:
            self._look.stop()
        elif self.parent_widget.isVisible() and self._bounds() != self._laid:
            self.layout()

    def layout(self) -> None:
        """The newest toast in the bottom-right corner, whole; the older ones to its left on the same line
        (their bottoms level), right of the status text while there is room. One that does not fit there goes
        when it is a plain confirmation (the status line of its page said the same), else it is shortened to
        the room left of the newer ones, or goes when not even that is enough. An error never goes: it is
        shortened to its shortest over the end of the status text, else it starts a row above. A notice is
        shortened only as far as the errors older than it still find room at their shortest."""
        left, keep_off, edge, bottom = self._laid = self._bounds()
        available = max(MIN_ROW, edge - left)
        right, row_height = edge, 0
        crowded_out = []
        newest_first = list(reversed(self.toasts))
        for number, toast in enumerate(newest_first):
            width = toast.show_whole(available)
            # (the room the older errors need: they are not dropped, and a row above lies over the plots)
            reserved = sum(older.shortest() + SPACING for older in newest_first[number + 1:]
                           if older.property("level") == "error")
            room = right - keep_off - reserved
            error = toast.property("level") == "error"
            if number and width > room:
                if _plain(toast):
                    crowded_out.append(toast)
                    continue
                if room >= toast.shortest():
                    width = toast.show_short(room)
                elif not error:  # (it would lie over the status line: an older notice gives way to it)
                    crowded_out.append(toast)
                    continue
                elif right - left - reserved >= toast.shortest():
                    width = toast.show_short(toast.shortest())
                else:  # an error is never dropped: it starts a row above, whole
                    bottom -= row_height + SPACING
                    right, row_height, keep_off = edge, 0, left
            toast.move(right - width, bottom - toast.height())
            toast.raise_()
            right -= width + SPACING
            row_height = max(row_height, toast.height())
        for toast in crowded_out:  # (taken off the list first: closing one lays the others out again)
            self.toasts.remove(toast)
        for toast in crowded_out:
            toast.close()

    def eventFilter(self, watched, event) -> bool:
        if watched is self.parent_widget and event.type() == QEvent.Resize:
            self.layout()
        return False


_stacks: dict[int, _Stack] = {}


def _stack(parent: QWidget) -> _Stack:
    """The stack of ``parent`` (made at its first toast, gone with it)."""
    stack = _stacks.get(id(parent))
    if stack is None:
        stack = _stacks[id(parent)] = _Stack(parent)
        parent.destroyed.connect(lambda *_args, key=id(parent): _stacks.pop(key, None))
    return stack


def show_toast(
    parent: QWidget,
    text: str,
    *,
    level: str = "info",
    action: Optional[tuple[str, Callable[[], None]]] = None,
    timeout_ms: Optional[int] = None,
) -> Toast:
    """``timeout_ms``: ``None`` — until closed for an error, 10 s for a warning, 5 s otherwise; 0 — until closed."""
    stack = _stack(parent)
    toast = Toast(parent, text, level=level, action=action, timeout_ms=timeout_ms)
    toast.show()
    stack.add(toast)
    return toast


def set_toast_inset(parent: QWidget, left=0, *, band: Optional[Callable[[], Iterable[QWidget]]] = None) -> None:
    """Keep the toasts of ``parent`` off its left side: ``left`` pixels, or a function that gives them (the
    width of a sidebar, which can be collapsed). ``band``: a function that gives the widgets whose status line
    lies along the bottom of ``parent`` (the page shown): the toasts keep off its progress bar and Cancel,
    and right of its text while there is room (``toast_band.py``)."""
    stack = _stack(parent)
    stack.inset = left if callable(left) else (lambda: int(left))
    stack.band = band
    stack.layout()


def visible_toasts(parent: QWidget) -> list[Toast]:
    stack = _stacks.get(id(parent))
    return list(stack.toasts) if stack is not None else []


def dismiss_toasts(parent: QWidget, levels: tuple = ("info", "ok")) -> int:
    """Close the plain notices of ``parent`` (of ``levels``, without a button) and return how many went.

    For a confirmation of something the user no longer looks at (the page changed). A warning or an error
    is still to be read, and a notice with a button would take its function with it (Undo of a removal,
    Open Folder, Details): those stay."""
    stack = _stacks.get(id(parent))
    if stack is None:
        return 0
    plain = [toast for toast in stack.toasts if toast.property("level") in levels and toast.action_button is None]
    for toast in plain:
        stack._close(toast)
    return len(plain)


__all__ = ["LOOK_MS", "MAX_TOASTS", "Toast", "default_timeout", "dismiss_toasts", "set_toast_inset", "show_toast",
           "visible_toasts"]
