"""The tool windows of the Tools menu: one modeless instance each, where it was last closed and on the screen.

``menu_bar`` re-exports these names (``ToolWindows``, ``clamp_to_screen``, ``SCREEN_SHARE``, ``TOOL_GEOMETRY_KEY``).
"""

from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import QEvent, QObject, QRect
from PyQt5.QtWidgets import QApplication, QWidget

from .layout_metrics import available_geometry

TOOL_GEOMETRY_KEY = "tool_windows.{name}.geometry"
"""Preference: ``[x, y, width, height]`` of a tool window as it was last closed (``name``: calibration, converter …)."""
SCREEN_SHARE = 0.9
"""A tool window takes at most this share of the screen's available width and height."""
TITLE_ROOM = 32
"""Room above a window for its title bar, so it can always be grabbed and moved."""
_TOOL_NAME = "gimapToolWindow"


def clamp_to_screen(rect: QRect, area: QRect, *, share: float = SCREEN_SHARE) -> QRect:
    """``rect`` at most ``share`` of ``area`` wide and high, moved (not shrunk further) to lie inside ``area`` with
    room for the title bar above it."""
    width = max(1, min(rect.width(), int(area.width() * share)))
    height = max(1, min(rect.height(), int(area.height() * share)))
    left = min(max(rect.x(), area.left()), area.right() + 1 - width)
    top = area.top() + min(TITLE_ROOM, max(0, area.height() - height))
    return QRect(left, min(max(rect.y(), top), area.bottom() + 1 - height), width, height)


class ToolWindows(QObject):
    """Single-instance modeless tool windows, created on demand.

    A new window opens where it was last closed (``preferences``), else at its own size; either way at most
    about 90 % of the screen and on it (a 1280 × 820 window on a 1366 × 768 laptop would put its buttons
    under the task bar)."""

    def __init__(self, parent: QWidget, preferences=None):
        super().__init__(parent)
        self.owner = parent
        self.preferences = preferences
        self._open: dict[str, QWidget] = {}

    def get(self, name: str) -> QWidget | None:
        return self._open.get(name)

    def windows(self) -> list[QWidget]:
        """The tool windows open now (shown or waiting hidden for a job to end)."""
        return list(self._open.values())

    def show(self, name: str, factory: Callable[[QWidget], QWidget], *, reuse=None) -> QWidget:
        window = self._open.get(name)
        if window is None:
            window = factory(self.owner)
            # The dict, not self: as the main window goes, this object may be deleted before its tool windows.
            window.destroyed.connect(lambda *_args, key=name, opened=self._open: opened.pop(key, None))
            self._open[name] = window
            self.place(name, window)
        elif reuse is not None:
            reuse(window)
        if window.isMinimized():
            window.showNormal()
        else:
            window.show()
        self.keep_on_screen(window)
        window.raise_()
        window.activateWindow()
        return window

    # -- geometry -----------------------------------------------------------

    def place(self, name: str, window: QWidget) -> None:
        """Before ``window`` is first shown: its last geometry, or its own size, clamped to the screen; and
        remembered when it is hidden or closed."""
        window.setProperty(_TOOL_NAME, name)
        window.installEventFilter(self)
        saved = self._saved(name)
        if saved is not None:
            screen = QApplication.screenAt(saved.center())
            area = screen.availableGeometry() if screen is not None else available_geometry(self.owner)
            window.setGeometry(clamp_to_screen(saved, area))
            return
        area = available_geometry(self.owner)
        size = clamp_to_screen(QRect(area.topLeft(), window.size()), area).size()
        window.resize(size)  # a dialog is centred on the main window as it is shown, inside its screen

    def keep_on_screen(self, window: QWidget) -> None:
        """Move a shown window back inside its screen (title bar included) when it lies partly outside."""
        frame = window.frameGeometry()
        screen = QApplication.screenAt(frame.center()) or window.screen()
        if screen is None:
            return
        area = screen.availableGeometry()
        if area.contains(frame):
            return
        # move() places the frame; one larger than the screen (a large minimum size) keeps its top left corner.
        x = min(max(frame.x(), area.left()), max(area.left(), area.right() + 1 - frame.width()))
        y = min(max(frame.y(), area.top()), max(area.top(), area.bottom() + 1 - frame.height()))
        window.move(x, y)

    def remember(self, name: str, window: QWidget) -> None:
        if self.preferences is None:
            return
        rect = window.normalGeometry() if window.isMaximized() or window.isFullScreen() else window.geometry()
        if rect.isValid() and not rect.isEmpty():
            self.preferences.set(TOOL_GEOMETRY_KEY.format(name=name),
                                 [rect.x(), rect.y(), rect.width(), rect.height()])

    def remember_all(self) -> None:
        """The geometry of every tool window still shown (as the main window closes, before preferences are saved)."""
        for name, window in list(self._open.items()):
            try:
                if window.isVisible():
                    self.remember(name, window)
            except RuntimeError:  # already gone
                continue

    def _saved(self, name: str) -> QRect | None:
        if self.preferences is None:
            return None
        value = self.preferences.get(TOOL_GEOMETRY_KEY.format(name=name))
        if not isinstance(value, (list, tuple)):
            return None
        try:
            x, y, width, height = (int(part) for part in value)
        except (TypeError, ValueError):
            return None
        return QRect(x, y, width, height) if width > 0 and height > 0 else None

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        # Not when the window system hides it (minimised): its geometry then is not the one to come back to.
        if event.type() == QEvent.Hide and not event.spontaneous() and isinstance(watched, QWidget):
            name = watched.property(_TOOL_NAME)
            if name:
                self.remember(str(name), watched)
        return False


__all__ = ["SCREEN_SHARE", "TITLE_ROOM", "TOOL_GEOMETRY_KEY", "ToolWindows", "clamp_to_screen"]
