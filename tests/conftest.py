"""Cross-suite cleanup for native GUI resources used by offscreen tests."""

from __future__ import annotations

import sys
import os
import tempfile
from pathlib import Path
import pytest


def pytest_configure(config) -> None:
    """Never touch the real user data folder: every test run gets its own."""
    del config
    os.environ["GIMAP_HOME"] = tempfile.mkdtemp(prefix="gimap-test-home-")


@pytest.fixture(scope="session", autouse=True)
def qt_application():
    """One QApplication with the production theme for every UI test.

    Windows offscreen Qt has no system font database, so real glyphs are
    loaded for layout checks.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtGui import QFont, QFontDatabase
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    if sys.platform == "win32" and not QFontDatabase().families():
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for name in ("arial.ttf", "segoeui.ttf", "msyh.ttc"):
            if (fonts / name).is_file():
                QFontDatabase.addApplicationFont(str(fonts / name))
        app.setFont(QFont("Segoe UI", 9))
    from src.gimap.app.presentation.theme import apply_theme

    apply_theme("light", 9)
    yield app


def _release_qt_widgets() -> None:
    """Close and delete every window the tests left behind."""
    if "matplotlib.pyplot" in sys.modules:
        sys.modules["matplotlib.pyplot"].close("all")

    if "PyQt5.QtWidgets" not in sys.modules:
        return

    from PyQt5.QtCore import QCoreApplication, QEvent
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    # Windows kept alive only by reference cycles are released the normal way
    # first: deleting them with deleteLater() while their Python wrappers still
    # reference embedded pyqtgraph views aborts the interpreter.
    import gc

    gc.collect()
    pyqtgraph = sys.modules.get("pyqtgraph")
    if pyqtgraph is not None:
        # Delete embedded pyqtgraph views before their parents (see above).
        for widget in list(app.allWidgets()):
            if isinstance(widget, pyqtgraph.GraphicsView) and widget.parent() is not None:
                widget.deleteLater()
    from PyQt5.QtWidgets import QMenu

    for widget in list(app.topLevelWidgets()):
        # Menus and other windows with a parent are deleted with that parent;
        # deleting them separately touches objects Qt already freed.
        if widget.parent() is not None or isinstance(widget, QMenu):
            continue
        widget.close()
        widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


@pytest.fixture(scope="module", autouse=True)
def release_windows_after_module():
    """Each test module starts without the previous module's windows.

    Thousands of leftover widgets make every later restyle (theme switch)
    and layout pass slow, and a restyle over half-destroyed windows can crash.
    """
    yield
    if not os.environ.get("GIMAP_TEST_KEEP_WINDOWS"):
        _release_qt_widgets()


def pytest_sessionfinish(session, exitstatus) -> None:
    """Release Qt/Matplotlib objects before pytest's final forced GC pass."""
    del session, exitstatus
    _release_qt_widgets()
