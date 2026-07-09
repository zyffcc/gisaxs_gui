"""Shared application asset helpers."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap


ASSET_ROOT = Path(__file__).resolve().parents[1] / "assets"
ICON_ROOT = ASSET_ROOT / "icons"

APP_ICON_PATH = ICON_ROOT / "GIMaP Logo.svg"
APP_LOGO_COLORED_PATH = ICON_ROOT / "GIMaP Logo_colored.png"


def app_icon() -> QIcon:
    """Return the monochrome rounded GIMaP icon for windows/taskbar use."""
    return QIcon(str(APP_ICON_PATH)) if APP_ICON_PATH.exists() else QIcon()


def app_colored_logo_pixmap(width: int, height: int | None = None) -> QPixmap:
    """Load the colored GIMaP logo as a pixmap, scaled for UI display."""
    pixmap = QPixmap(str(APP_LOGO_COLORED_PATH)) if APP_LOGO_COLORED_PATH.exists() else QPixmap()
    if pixmap.isNull() or width <= 0:
        return pixmap

    if height is None or height <= 0:
        return pixmap.scaledToWidth(width, Qt.SmoothTransformation)
    return pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
