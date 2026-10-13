"""Tool windows (Tools ▸ Geometry Calibration, Format Converter, XRR): at most about 90 % of the screen, on it,
and where each was last left (wave 2b, tools-15 part 3)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QApplication, QDialog, QWidget

from src.gimap.app.presentation.menu_bar import SCREEN_SHARE, TOOL_GEOMETRY_KEY, ToolWindows, clamp_to_screen
from src.gimap.integrations.state import InMemoryUserPreferencesRepository


def _app():
    return QApplication.instance() or QApplication([])


def _settle() -> None:
    for _ in range(5):
        QApplication.processEvents()


def _factory(width: int, height: int):
    def factory(parent):
        dialog = QDialog(parent)
        dialog.setModal(False)
        dialog.resize(width, height)  # as the tool views do (1180 × 760, 1080 × 760, 1280 × 820)
        return dialog

    return factory


def _close(tools, main) -> None:
    """As the Tools menu's windows end: closed, then the main window (deleted with the module's windows)."""
    for window in tools.windows():
        window.close()
    main.close()
    _settle()


def _area() -> QRect:
    return QApplication.primaryScreen().availableGeometry()


def test_clamp_keeps_a_rect_inside_the_area_with_room_for_the_title_bar() -> None:
    area = QRect(0, 0, 1366, 728)  # a 1366 × 768 laptop above its task bar
    inside = clamp_to_screen(QRect(40, 30, 1280, 820), area)
    assert inside.width() == int(1366 * SCREEN_SHARE) and inside.height() == int(728 * SCREEN_SHARE)
    assert area.contains(inside) and inside.top() >= 32
    moved = clamp_to_screen(QRect(3000, -400, 600, 400), area)  # left on a screen that is gone
    assert moved.size() == QRect(0, 0, 600, 400).size() and area.contains(moved) and moved.top() >= 32
    small = clamp_to_screen(QRect(100, 100, 300, 200), area)
    assert small == QRect(100, 100, 300, 200)  # a window that fits stays where it is


def test_a_large_tool_window_opens_inside_about_ninety_percent_of_the_screen() -> None:
    _app()
    main = QWidget()
    main.resize(700, 500)
    main.show()
    tools = ToolWindows(main)
    try:
        dialog = tools.show("xrr", _factory(1280, 820))
        _settle()
        area = _area()
        assert dialog.width() <= int(area.width() * SCREEN_SHARE) and dialog.height() <= int(area.height() * SCREEN_SHARE)
        assert area.contains(dialog.frameGeometry())
        assert tools.windows() == [dialog]
    finally:
        _close(tools, main)


def test_each_tool_window_comes_back_where_it_was_left_and_on_screen() -> None:
    _app()
    preferences = InMemoryUserPreferencesRepository()
    main = QWidget()
    main.show()
    tools = ToolWindows(main, preferences)
    try:
        dialog = tools.show("converter", _factory(500, 380))
        _settle()
        dialog.setGeometry(120, 90, 460, 330)
        dialog.close()  # hidden: its geometry is kept
        _settle()
        assert preferences.get(TOOL_GEOMETRY_KEY.format(name="converter")) == [120, 90, 460, 330]
        assert preferences.get(TOOL_GEOMETRY_KEY.format(name="xrr")) is None  # one entry per tool

        tools._open.clear()  # as after the window was deleted (WA_DeleteOnClose)
        again = tools.show("converter", _factory(500, 380))
        _settle()
        assert again is not dialog and again.geometry() == QRect(120, 90, 460, 330)

        # Left on a monitor that is no longer there, larger than this screen: inside it, at most 90 %.
        preferences.set(TOOL_GEOMETRY_KEY.format(name="calibration"), [4200, 1900, 2400, 1500])
        moved = tools.show("calibration", _factory(500, 380))
        _settle()
        area = _area()
        assert area.contains(moved.frameGeometry())
        assert moved.width() <= int(area.width() * SCREEN_SHARE) and moved.height() <= int(area.height() * SCREEN_SHARE)

        # Something unreadable in the preferences is ignored.
        preferences.set(TOOL_GEOMETRY_KEY.format(name="xrr"), "not a rect")
        assert tools.show("xrr", _factory(300, 200)).size().width() == 300
    finally:
        _close(tools, main)


def test_open_tool_windows_are_remembered_as_the_main_window_closes() -> None:
    _app()
    preferences = InMemoryUserPreferencesRepository()
    main = QWidget()
    main.show()
    tools = ToolWindows(main, preferences)
    try:
        dialog = tools.show("xrr", _factory(400, 300))
        _settle()
        dialog.setGeometry(60, 70, 380, 290)
        tools.remember_all()
        assert preferences.get(TOOL_GEOMETRY_KEY.format(name="xrr")) == [60, 70, 380, 290]
        dialog.move(5000, 5000)  # dragged off screen, then reopened from the Tools menu
        assert tools.show("xrr", _factory(400, 300)) is dialog
        _settle()
        assert _area().contains(dialog.frameGeometry())
    finally:
        _close(tools, main)


def test_the_main_window_restores_its_tool_windows_and_saves_them_on_close(monkeypatch) -> None:
    """The shell wiring: menus keep the geometry in the user's preferences; Analyze's Calibrate uses the same entry."""
    import time

    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (
        InMemoryInstrumentProfileRepository,
        InMemorySessionRepository,
        InMemorySettingsRepository,
    )

    _app()
    preferences = InMemoryUserPreferencesRepository()
    context = AppContext(settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
                         preferences=preferences, jobs=LocalProcessJobRunner(),
                         instrument_profiles=InMemoryInstrumentProfileRepository([]))
    window = MainWindow(context)
    window.show()
    end = time.monotonic() + 40
    while not window._initialization_completed and time.monotonic() < end:
        QApplication.processEvents()
        time.sleep(0.02)
    try:
        assert window.menus.tools.preferences is preferences
        dialog = window.menus.tools.show("probe", _factory(300, 200))
        _settle()
        dialog.setGeometry(50, 60, 310, 210)
        placed = []
        from src.gimap.features.calibration.presentation import dialog as calibration_dialog

        monkeypatch.setattr(calibration_dialog.GeometryCalibrationDialog, "exec_",
                            lambda self: placed.append(self.geometry()) or 0)
        preferences.set(TOOL_GEOMETRY_KEY.format(name="calibration"), [40, 50, 520, 400])
        window.components._calibrate_for_analyze(None)
        # Where it was left (its size: at least its own minimum, larger than this 800 × 600 offscreen screen).
        assert len(placed) == 1 and (placed[0].x(), placed[0].y()) == (40, 50)
    finally:
        window.close()  # nothing runs: no question
        _settle()
    assert preferences.get(TOOL_GEOMETRY_KEY.format(name="probe")) == [50, 60, 310, 210]
