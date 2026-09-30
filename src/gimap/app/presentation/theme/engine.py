"""Render the token style sheets and apply one theme to the whole application.

The application style sheet (``base.qss``) is set once on ``QApplication``;
widgets only carry semantic properties (``gimapRole="muted"``, ``card=true``
...), never colours.  A feature that needs rules of its own styles its root
widget with ``style_widget``; those sheets are re-rendered when the theme
changes.  Qt's high-DPI support scales every length, so the sheets use one set
of logical sizes on every screen.
"""

from __future__ import annotations

import gc
import re
import tempfile
from functools import lru_cache
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QEvent, QObject, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPalette
from PyQt5.QtWidgets import QApplication, QWidget

from .tokens import (
    DEFAULT_FONT_PT,
    DEFAULT_MODE,
    THEMES,
    font_tokens,
    normalized_font_pt,
    normalized_mode,
)

THEME_DIR = Path(__file__).resolve().parent
BASE_TEMPLATE = THEME_DIR / "base.qss"
ICON_TEMPLATES = THEME_DIR / "icons"
TEMPLATE_PROPERTY = "gimapStyleTemplate"
_MARKER = re.compile(r"@([a-z][a-z0-9_]*)@")
_COMMENT = re.compile(r"/\*.*?\*/", re.S)

# Icon tokens: ``@icon_chevron_down@`` is the chevron drawn in ``text_muted``.
ICONS = {
    "icon_chevron_down": ("chevron-down", "text_muted"),
    "icon_chevron_up": ("chevron-up", "text_muted"),
    "icon_chevron_down_disabled": ("chevron-down", "text_disabled"),
    "icon_chevron_up_disabled": ("chevron-up", "text_disabled"),
    "icon_chevron_down_on_accent": ("chevron-down", "on_accent"),
    "icon_check": ("check", "on_accent"),
    "icon_dash": ("dash", "on_accent"),
}


class ThemeManager(QObject):
    """Current theme of the application; ``changed`` fires after a switch."""

    changed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self.mode = DEFAULT_MODE
        self.font_pt = DEFAULT_FONT_PT
        self.applied = False

    @property
    def is_dark(self) -> bool:
        return self.mode == "dark"

    def tokens(self) -> dict[str, str]:
        values = dict(THEMES[self.mode])
        values.update(font_tokens(self.font_pt))
        for name, (icon, colour) in ICONS.items():
            values[name] = _tinted_icon(icon, values[colour])
        return values

    def color(self, name: str) -> QColor:
        return QColor(THEMES[self.mode][name])

    def render(self, template: str) -> str:
        values = self.tokens()

        def substitute(match: re.Match) -> str:
            name = match.group(1)
            if name not in values:
                raise KeyError(f"Unknown theme token @{name}@")
            return values[name]

        return _MARKER.sub(substitute, _COMMENT.sub("", template))

    def apply(self, mode: str | None = None, font_pt: float | None = None) -> None:
        """Apply (or re-apply) the theme to the running application."""
        if mode is not None:
            self.mode = normalized_mode(mode)
        if font_pt is not None:
            self.font_pt = normalized_font_pt(font_pt)
        app = QApplication.instance()
        if app is None:
            return
        if not self.applied:
            app.setStyle("Fusion")
        else:
            # Restyling touches every widget; widgets that only wait for the
            # garbage collector or a deferred delete must be gone first.
            gc.collect()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.setPalette(_palette(THEMES[self.mode]))
        font = QFont(app.font())
        font.setPointSizeF(self.font_pt)
        app.setFont(font)
        app.setStyleSheet(self.render(_template(BASE_TEMPLATE)))
        for widget in app.allWidgets():
            path = widget.property(TEMPLATE_PROPERTY)
            if path:
                widget.setStyleSheet(self.render(_template(Path(path))))
        self.applied = True
        self.changed.emit(self.mode)

    def style_widget(self, widget: QWidget, template: Path) -> None:
        """Give ``widget`` a feature style sheet that follows theme changes."""
        widget.setProperty(TEMPLATE_PROPERTY, str(template))
        widget.setStyleSheet(self.render(_template(Path(template))))


_MANAGER: ThemeManager | None = None


def theme_manager() -> ThemeManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ThemeManager()
    return _MANAGER


def apply_theme(mode: str | None = None, font_pt: float | None = None) -> ThemeManager:
    manager = theme_manager()
    manager.apply(mode, font_pt)
    return manager


def style_widget(widget: QWidget, template: Path) -> None:
    theme_manager().style_widget(widget, template)


def theme_color(name: str) -> QColor:
    return theme_manager().color(name)


def set_role(widget: QWidget, role: str | None) -> None:
    """Set the semantic ``gimapRole`` of a widget and refresh its style."""
    if widget.property("gimapRole") == role:
        return
    widget.setProperty("gimapRole", role)
    repolish(widget)


def set_state(widget: QWidget, name: str, value) -> None:
    """Update a dynamic style property (``statusKind``, ``level`` ...)."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    repolish(widget)


def repolish(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


@lru_cache(maxsize=None)
def _template(path: Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def _tinted_icon(name: str, colour: str) -> str:
    folder = Path(tempfile.gettempdir()) / "gimap-theme-icons"
    target = folder / f"{name}-{colour.lstrip('#')}.svg"
    if not target.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        text = (ICON_TEMPLATES / f"{name}.svg").read_text(encoding="utf-8")
        target.write_text(text.replace("@color@", colour), encoding="utf-8")
    return target.as_posix()


def _palette(tokens: dict[str, str]) -> QPalette:
    def colour(name: str) -> QColor:
        return QColor(tokens[name])

    palette = QPalette()
    roles = {
        QPalette.Window: "bg",
        QPalette.WindowText: "text",
        QPalette.Base: "surface",
        QPalette.AlternateBase: "surface_alt",
        QPalette.ToolTipBase: "tooltip_bg",
        QPalette.ToolTipText: "tooltip_text",
        QPalette.PlaceholderText: "text_faint",
        QPalette.Text: "text",
        QPalette.Button: "surface",
        QPalette.ButtonText: "text",
        QPalette.BrightText: "danger",
        QPalette.Highlight: "accent",
        QPalette.HighlightedText: "on_accent",
        QPalette.Link: "accent",
        QPalette.LinkVisited: "accent_pressed",
        QPalette.Light: "surface",
        QPalette.Midlight: "surface_alt",
        QPalette.Mid: "border_strong",
        QPalette.Dark: "border_strong",
        QPalette.Shadow: "rail",
    }
    for role, name in roles.items():
        palette.setColor(role, colour(name))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, colour("text_disabled"))
    palette.setColor(QPalette.Disabled, QPalette.Base, colour("disabled_bg"))
    return palette


__all__ = [
    "ICONS",
    "TEMPLATE_PROPERTY",
    "ThemeManager",
    "apply_theme",
    "repolish",
    "set_role",
    "set_state",
    "style_widget",
    "theme_color",
    "theme_manager",
]
