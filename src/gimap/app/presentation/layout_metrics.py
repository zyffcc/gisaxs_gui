"""Fixed layout metrics and window placement, in logical pixels.

Qt's high-DPI support converts logical pixels to device pixels for each
screen, so the application uses one set of sizes on every monitor instead of
per-resolution profiles.  The metrics are compact: the main window fits a
1280 x 720 logical screen, and dense panels scroll rather than shrink.
"""

from __future__ import annotations

from dataclasses import dataclass

from PyQt5.QtCore import QPoint, QRect, QSize
from PyQt5.QtGui import QCursor, QScreen
from PyQt5.QtWidgets import QApplication, QWidget


@dataclass(frozen=True)
class LayoutMetrics:
    key: str = "standard"
    min_window: QSize = QSize(1024, 640)
    default_window: QSize = QSize(1440, 900)
    content_min: int = 760
    workspace_min: int = 560
    preview_min: int = 360
    page_sizes: tuple[int, int] = (640, 420)
    work_sizes: tuple[int, int] = (720, 600)
    preview_sizes: tuple[int, int, int] = (280, 780, 150)


LAYOUT = LayoutMetrics()


def screen_at_cursor() -> QScreen | None:
    app = QApplication.instance()
    if app is None:
        return None
    return app.screenAt(QCursor.pos()) or app.primaryScreen()


def available_geometry(window: QWidget | None = None) -> QRect:
    screen = window.screen() if window is not None else screen_at_cursor()
    if screen is None:
        screen = screen_at_cursor()
    return screen.availableGeometry() if screen is not None else QRect(0, 0, 1280, 720)


def fit_to_screen(window: QWidget, size: QSize, *, margin: int = 24) -> QSize:
    """Resize ``window`` to ``size`` without exceeding its screen."""
    area = available_geometry(window)
    width = max(window.minimumWidth(), min(size.width(), area.width() - margin * 2))
    height = max(window.minimumHeight(), min(size.height(), area.height() - margin * 2))
    window.resize(width, height)
    return QSize(width, height)


def move_window_to_cursor_screen(window: QWidget, margin: int = 24) -> QScreen | None:
    """Centre a top-level window on the monitor under the mouse pointer."""
    screen = screen_at_cursor()
    if screen is None:
        return None
    area = screen.availableGeometry()
    size = window.size()
    if not size.isValid() or size.isEmpty():
        size = window.sizeHint()
    if not size.isValid() or size.isEmpty():
        size = QSize(900, 600)
    size = QSize(
        min(size.width(), max(320, area.width() - margin * 2)),
        min(size.height(), max(240, area.height() - margin * 2)),
    )
    window.resize(size)
    window.move(
        QPoint(
            area.x() + max(0, (area.width() - size.width()) // 2),
            area.y() + max(0, (area.height() - size.height()) // 2),
        )
    )
    return screen


__all__ = [
    "LAYOUT",
    "LayoutMetrics",
    "available_geometry",
    "fit_to_screen",
    "move_window_to_cursor_screen",
    "screen_at_cursor",
]
