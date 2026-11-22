"""The splash screen of the start-up: the name, the version and what is being loaded now.

Shown by the launcher within a second of the start, before the analysis modules (SciPy, Matplotlib, the
pages) are imported, and closed when the main window is on screen. Its texts are English here and shown in
the saved interface language (``translate``: the language is not applied to the application yet). Light
imports only: PyQt, the theme tokens, the version and the language tables.
"""

from __future__ import annotations

from PyQt5.QtCore import QRect, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPixmap
from PyQt5.QtWidgets import QApplication, QSplashScreen

from src.gimap.shared.version import APP_NAME, PRE_RELEASE, __version__

from .assets import app_colored_logo_pixmap
from .i18n import translate
from .theme.tokens import THEMES, normalized_mode

WIDTH, HEIGHT = 480, 270


def version_text(language: str = "en") -> str:
    """``v0.2.0`` or ``v0.2.0 (Pre-release)``, in ``language``."""
    template = "{name} v{version} (Pre-release)" if PRE_RELEASE else "{name} v{version}"
    shown = translate(template, language) if language != "en" else None
    return (shown or template).format(name=APP_NAME, version=__version__)


class StartupSplash(QSplashScreen):
    """``step(text)`` shows a step (English, translated here) and lets the splash paint it at once."""

    def __init__(self, language: str = "en", mode: str = "light"):
        self.language = language
        self.colours = THEMES[normalized_mode(mode)]
        super().__init__(self._picture())
        self.setWindowFlag(Qt.WindowStaysOnTopHint, False)  # a question asked meanwhile must not be hidden
        self.setObjectName("gimapSplash")

    def _picture(self) -> QPixmap:
        ratio = max(1.0, QApplication.primaryScreen().devicePixelRatio()) if QApplication.primaryScreen() else 1.0
        pixmap = QPixmap(int(WIDTH * ratio), int(HEIGHT * ratio))
        pixmap.setDevicePixelRatio(ratio)
        colours = self.colours
        pixmap.fill(QColor(colours["surface"]))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setPen(QColor(colours["border_strong"]))
        painter.drawRect(0, 0, WIDTH - 1, HEIGHT - 1)
        logo = app_colored_logo_pixmap(int(84 * ratio))
        if not logo.isNull():
            logo.setDevicePixelRatio(ratio)
            painter.drawPixmap(36, 52, logo)
        family = QApplication.font().family()
        painter.setPen(QColor(colours["text"]))
        painter.setFont(QFont(family, 26, QFont.DemiBold))
        painter.drawText(QRect(140, 48, WIDTH - 160, 48), Qt.AlignLeft | Qt.AlignVCenter, APP_NAME)
        painter.setPen(QColor(colours["text_muted"]))
        painter.setFont(QFont(family, 10))
        painter.drawText(QRect(142, 98, WIDTH - 160, 24), Qt.AlignLeft | Qt.AlignVCenter, version_text(self.language))
        painter.setPen(QColor(colours["accent"]))
        painter.drawLine(36, HEIGHT - 58, WIDTH - 36, HEIGHT - 58)
        painter.end()
        return pixmap

    def step(self, text: str) -> None:
        """Show ``text`` (English; translated when the saved language has it) and paint it now."""
        shown = translate(text, self.language) if self.language != "en" else None
        self.showMessage("  " + (shown or text), Qt.AlignBottom | Qt.AlignLeft, QColor(self.colours["text_muted"]))
        # Painted now, without handling the events the half-built window has queued (that would lay it out again
        # at every step).
        self.repaint()


__all__ = ["StartupSplash", "version_text"]
