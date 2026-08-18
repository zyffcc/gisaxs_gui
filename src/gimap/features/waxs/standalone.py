"""Standalone composition root for the feature-owned WAXS workspace."""

from __future__ import annotations

import sys
from collections.abc import Callable

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow

from src.gimap.app import AppContext
from src.gimap.app.bootstrap import create_app_context

from .bootstrap import create_waxs_view_model
from .presentation.page import InSituProcessingWidget


class WaxsStandaloneWindow(QMainWindow):
    """Host the same WAXS page used by the main GIMaP window."""

    def __init__(self, context: AppContext | None = None):
        super().__init__()
        self._owns_context = context is None
        self.app_context = context or create_app_context(restore_session=False)
        self.page = InSituProcessingWidget(
            self,
            view_model=create_waxs_view_model(self.app_context),
        )
        self.setCentralWidget(self.page)
        self.setWindowTitle("In-situ Data Processing")
        self.resize(1280, 800)
        self.page.statusChanged.connect(self.statusBar().showMessage)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._owns_context and self.app_context.jobs is not None:
            self.app_context.jobs.shutdown()
        super().closeEvent(event)


def launch_waxs(
    window_factory: Callable[[], QMainWindow] = WaxsStandaloneWindow,
) -> int:
    """Launch the standalone WAXS host while remaining safe for embedded use."""
    app = QApplication.instance()
    owns_application = app is None
    if app is None:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)

    font = QFont("Segoe UI", 9) if sys.platform.startswith("win") else QFont(app.font())
    font.setPointSize(9)
    app.setFont(font)

    window = window_factory()
    screen = app.primaryScreen()
    if screen is not None:
        available = screen.availableGeometry()
        window.resize(int(available.width() * 0.8), int(available.height() * 0.8))
        window.move(
            available.x() + int(available.width() * 0.1),
            available.y() + int(available.height() * 0.1),
        )
    window.show()
    if owns_application:
        return app.exec_()
    return 0


__all__ = ["WaxsStandaloneWindow", "launch_waxs"]
