"""A switch of theme or font size, applied to what changed and nothing else (``ThemeManager.apply``).

Setting the application style sheet again makes Qt 5 polish every widget once for each of its ancestors:
a theme switch of the main window took 20 s, the window frozen. The sheet of the theme goes on each window
without a parent instead (the main window, a dialog opened without one). Qt then polishes each widget below
once, and a window's sheet takes precedence over the application's, which keeps the theme of the start.
Dialogs, menus and tooltips with a parent take the sheet from their parent. A window without a parent that
is shown after the switch gets it then (``_Watcher``); back in the theme of the start the windows' sheets
are taken off again.

A feature sheet (``style_widget``) is rendered again only when its text changes, at once when its widget is
shown and else when it is shown next: a hidden page costs nothing at the switch.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from PyQt5.QtCore import QEvent, QObject, Qt
from PyQt5.QtWidgets import QWidget

from .engine import TEMPLATE_PROPERTY, _template

OWN_SHEET_PROPERTY = "gimapOwnSheet"
"""On a window without a parent: the sheet it has itself, kept under the theme's sheet of a switch."""
STAMP_PROPERTY = "gimapThemeStamp"
"""The theme (``mode:size``) a window's or a feature's sheet was last rendered for."""


def restyle_windows(manager, app) -> None:
    """The theme's sheet (``""`` in the theme of the start) on every window without a parent."""
    sheet = manager.window_sheet()
    for window in app.topLevelWidgets():
        if window.parent() is None:
            style_window(manager, window, sheet)
    if sheet:
        _watcher(manager).install(app)


def style_window(manager, window: QWidget, sheet: str) -> None:
    """``sheet`` on ``window``, followed by the sheet the window has itself (its own rules still win)."""
    own = window.property(OWN_SHEET_PROPERTY)
    if own is None:
        own = window.styleSheet()
        window.setProperty(OWN_SHEET_PROPERTY, own)
    combined = "\n".join(part for part in (sheet, own) if part)
    if window.styleSheet() != combined:
        window.setStyleSheet(combined)
    if not window.property(TEMPLATE_PROPERTY):  # (a feature window: its own sheet is rendered again next)
        window.setProperty(STAMP_PROPERTY, manager.stamp)


def restyle_features(manager, app) -> None:
    """Every feature sheet rendered for another theme: now when its widget is shown, else when it is shown."""
    for widget in app.allWidgets():
        path = widget.property(TEMPLATE_PROPERTY)
        if not path or widget.property(STAMP_PROPERTY) == manager.stamp:
            continue
        if widget.isVisible() or widget.isWindow():
            render_feature(manager, widget, Path(path))
        else:
            _watcher(manager).watch(widget)


def render_feature(manager, widget: QWidget, template: Path) -> None:
    """The feature sheet ``template`` rendered in the current theme on ``widget`` (set only when it changed)."""
    text = manager.render(_template(Path(template)))
    if widget.isWindow() and widget.parent() is None and manager.window_sheet():
        widget.setProperty(OWN_SHEET_PROPERTY, text)
        style_window(manager, widget, manager.window_sheet())
    else:
        widget.setProperty(OWN_SHEET_PROPERTY, None)
        if widget.styleSheet() != text:
            widget.setStyleSheet(text)
    widget.setProperty(STAMP_PROPERTY, manager.stamp)


@contextmanager
def waiting(app):
    """A wait cursor while the windows are restyled."""
    app.setOverrideCursor(Qt.WaitCursor)
    try:
        yield
    finally:
        app.restoreOverrideCursor()


class _Watcher(QObject):
    """Show events: a window without a parent opened after a switch, a feature widget hidden at the switch."""

    def __init__(self, manager) -> None:
        super().__init__()
        self.manager = manager
        self._installed = False

    def install(self, app) -> None:
        if not self._installed:
            app.installEventFilter(self)
            self._installed = True

    def watch(self, widget: QWidget) -> None:
        widget.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API
        if event.type() != QEvent.Show or not isinstance(watched, QWidget):
            return False
        manager = self.manager
        try:
            if watched.property(STAMP_PROPERTY) == manager.stamp:
                return False
            path = watched.property(TEMPLATE_PROPERTY)
            if path:
                render_feature(manager, watched, Path(path))
            elif watched.isWindow() and watched.parent() is None:
                sheet = manager.window_sheet()
                if sheet or watched.property(OWN_SHEET_PROPERTY) is not None:
                    style_window(manager, watched, sheet)
        except RuntimeError:  # a window being torn down
            pass
        return False


_WATCHER: _Watcher | None = None


def _watcher(manager) -> _Watcher:
    global _WATCHER
    if _WATCHER is None:
        _WATCHER = _Watcher(manager)
    return _WATCHER


__all__ = ["OWN_SHEET_PROPERTY", "STAMP_PROPERTY", "render_feature", "restyle_features", "restyle_windows",
           "style_window", "waiting"]
