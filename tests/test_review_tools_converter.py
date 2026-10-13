"""Format Converter reopened from the Tools menu while an earlier preview read still runs."""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QCoreApplication, QEvent, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QWidget

from src.gimap.app import AppContext
from src.gimap.app.presentation.menu_bar import ToolWindows
from src.gimap.features.format_converter.domain import InputSource
from src.gimap.features.format_converter.presentation.dialog import FormatConverterDialog
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)

_APP = None


def _app() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _context() -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository(),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )


def _settle(app, seconds: float = 0.1) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        time.sleep(0.01)


def _slow_factory(release: threading.Event):
    def factory(parent):
        dialog = FormatConverterDialog(parent, app_context=_context())
        dialog.view_model.load_preview = lambda _source: (release.wait(10), [])[1]
        return dialog

    return factory


def _wait_for_thread_end(app, dialog) -> None:
    end = time.monotonic() + 10
    while dialog._preview_thread is not None and time.monotonic() < end:
        _settle(app, 0.05)
    _settle(app, 0.2)


def test_a_converter_reopened_before_the_preview_read_ends_stays_open(tmp_path) -> None:
    app = _app()
    main = QWidget()
    tools = ToolWindows(main)
    release = threading.Event()
    dialog = tools.show("converter", _slow_factory(release))
    events = []
    dialog.destroyed.connect(lambda *_: events.append("destroyed"))
    try:
        dialog._start_preview(InputSource(path=str(tmp_path / "frame.tif"), file_type="TIFF"))
        QTest.keyClick(dialog, Qt.Key_Escape)
        _settle(app, 0.2)
        assert not dialog.isVisible()  # waits hidden for the native read

        # The user opens the converter again (Tools menu) and adds a file before the read ends.
        again = tools.show("converter", _slow_factory(release))
        assert again is dialog and dialog.isVisible()
        added = InputSource(path=str(tmp_path / "a.cbf"), file_type="CBF")
        dialog.view_model.sources.append(added)

        release.set()
        _wait_for_thread_end(app, dialog)
        assert events == []
        assert tools.get("converter") is dialog
        assert dialog.isVisible()
        assert added in dialog.view_model.sources

        # An idle close still closes and deletes the window.
        QTest.keyClick(dialog, Qt.Key_Escape)
        _settle(app, 0.2)
        assert events == ["destroyed"]
        assert tools.get("converter") is None
    finally:
        release.set()
        main.deleteLater()


def test_escape_without_reopening_still_deletes_the_window_after_the_read() -> None:
    app = _app()
    main = QWidget()
    tools = ToolWindows(main)
    release = threading.Event()
    dialog = tools.show("converter", _slow_factory(release))
    events = []
    dialog.destroyed.connect(lambda *_: events.append("destroyed"))
    try:
        dialog._start_preview(InputSource(path="E:/nowhere/frame.tif", file_type="TIFF"))
        QTest.keyClick(dialog, Qt.Key_Escape)
        QTest.keyClick(dialog, Qt.Key_Escape)  # a second close while it waits changes nothing
        _settle(app, 0.2)
        assert events == [] and not dialog.isVisible()
        release.set()
        end = time.monotonic() + 10
        while "destroyed" not in events and time.monotonic() < end:
            _settle(app, 0.05)
        assert events == ["destroyed"]
        assert tools.get("converter") is None
    finally:
        release.set()
        main.deleteLater()
