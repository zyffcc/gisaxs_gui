"""Cross-suite cleanup for native GUI resources used by offscreen tests."""

from __future__ import annotations

import sys
import os
from pathlib import Path
import pytest


@pytest.fixture(scope="session", autouse=True)
def offscreen_fonts():
    """Windows offscreen Qt has no system font database; load real glyphs for UI checks."""
    if sys.platform != "win32" or os.environ.get("QT_QPA_PLATFORM") != "offscreen":
        yield
        return
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QFontDatabase, QFont
    app = QApplication.instance() or QApplication([])
    if not QFontDatabase().families():
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("arial.ttf", "segoeui.ttf", "msyh.ttc"):
            if (fonts / name).is_file():
                QFontDatabase.addApplicationFont(str(fonts / name))
        app.setFont(QFont("Segoe UI", 9))
    yield


def pytest_sessionfinish(session, exitstatus) -> None:
    """Release Qt/Matplotlib objects before pytest's final forced GC pass."""
    del session, exitstatus

    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")

    if "PyQt5.QtWidgets" not in sys.modules:
        return

    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    for widget in list(app.topLevelWidgets()):
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()
