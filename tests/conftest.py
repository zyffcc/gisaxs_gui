"""Cross-suite cleanup for native GUI resources used by offscreen tests."""

from __future__ import annotations

import sys


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
