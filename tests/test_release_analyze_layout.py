"""Release fixes of Analyze's layout at laptop sizes (R16, R17, R21, R75).

The command bar chooses its state from what its widgets need, so nothing is clipped: the short texts when the
full row does not fit (several files add the arrows, “2 / 3” and Batch Export…), and a second row for the
actions when the short row does not fit either. The Series tab scrolls when the window is too short for its
map and plots, which keep a least height, and is unchanged while it fits. The rows nested in the Batch Export
dialog keep a gap between their controls.

The page is laid out inside a host widget, as in the main window: there its width is the window's, not its
layout's least width, which is where the clipping came from.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QWheelEvent
from PyQt5.QtWidgets import QApplication, QBoxLayout, QDialog, QWidget

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application import BatchChoices, SeriesCorrection
from src.gimap.features.analyze.presentation.views.batch_export_view import ROW_SPACING
from src.gimap.features.analyze.presentation.views.command_bar import COMPACT_WIDTH, FULL_AGAIN, least_width
from src.gimap.features.analyze.presentation.views.series_view import (
    SERIES_MAP_HEIGHT,
    SERIES_PLOTS_HEIGHT,
    SeriesScroll,
)
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_review_analyze_run_and_clear import _frames, _profiles
from tests.test_series_map import _wait

MARGINS = 24
"""The page's left and right margins (px): the bar is this much narrower than the page."""


@pytest.fixture(autouse=True)
def english():
    apply_language("en")
    yield
    apply_language("en")


def _hosted(context, width: int = 1500, height: int = 760):
    """The page as a child of a host, like the main window's page: sized by the host, not by its layout."""
    host = QWidget()
    page = _page(context)
    page.setParent(host)
    host.resize(width, height)
    page.setGeometry(0, 0, width, height)
    host.show()
    _app().processEvents()
    return host, page


def _size(host, page, width: int, height: int = 760) -> None:
    host.resize(width, height)
    page.setGeometry(0, 0, width, height)
    for _ in range(4):  # a state change lays the bar out again
        _app().processEvents()


def _close(host, page) -> None:
    page.tasks.wait(30)
    page.dispose()
    host.close()
    host.deleteLater()


def _middle(widget) -> int:
    return widget.geometry().center().y()


def _cut(page) -> list[str]:
    """The widgets of the bar that are narrower than they need, or outside it."""
    bar = page.command_bar
    found = []
    for widget in bar._parts():
        if widget.isHidden():
            continue
        geometry = widget.geometry()
        if widget.width() < least_width(widget) - 1 or geometry.left() < 0 or geometry.right() >= bar.width():
            found.append(f"{widget.objectName() or type(widget).__name__} {widget.width()}/{least_width(widget)}")
    return found


# -- the command bar (R17) ------------------------------------------------------------------------------------------


def test_the_bar_computes_what_each_state_needs_from_its_widgets(tmp_path: Path) -> None:
    host, page = _hosted(_context(_profiles()), 2200)
    try:
        bar = page.command_bar
        page.set_automatic_analysis(run=lambda: None, find_geometry=lambda: None)  # Run Automatic Analysis is shown
        page.add_paths([str(path) for path in _frames(tmp_path, 3)])
        _done(page)
        _size(host, page, 2200)
        assert (bar.short, bar.compact, bar.wrapped) == (False, False, False)
        needs = full, brief, least = bar.row_need(), bar.row_need(short=True), bar.row_need(compact=True)
        assert bar.full_need() == full == bar.layout().minimumSize().width()  # exactly what the layout asks for
        assert least < brief < full
        assert not page.previous_file_button.isHidden() and not page.batch_export_button.isHidden()

        def state_needs() -> tuple:
            return bar.row_need(), bar.row_need(short=True), bar.row_need(compact=True)

        if max(COMPACT_WIDTH, brief) + 30 < full:  # the short texts alone are enough: the stepper stays
            _size(host, page, max(COMPACT_WIDTH, brief) + MARGINS + 30)
            assert (bar.short, bar.compact, bar.wrapped) == (True, False, False)
            assert state_needs() == needs  # the same numbers in every state
            assert bar.layout().minimumSize().width() == brief
            assert page.run_pipeline_button.text() == "Run Analysis" and page.fit_button.text() == "Fitting"
            assert not page.previous_file_button.isHidden() and page.file_position_label.text() == "1 / 3"

        _size(host, page, least + MARGINS + 40)  # not even with the short texts: the stepper and details go
        assert (bar.short, bar.compact, bar.wrapped) == (True, True, False)
        assert state_needs() == needs
        assert bar.layout().minimumSize().width() == least
        assert page.run_pipeline_button.text() == "Run Analysis" and page.previous_file_button.isHidden()

        _size(host, page, least + MARGINS - 80)  # not even the compact row: the actions on a second row
        assert (bar.short, bar.compact, bar.wrapped) == (True, True, True)
        assert state_needs() == needs
        assert _middle(page.fit_button) > _middle(page.mode_combo) + 20  # a row below
        assert abs(_middle(page.run_pipeline_button) - _middle(page.fit_button)) <= 3
        assert abs(_middle(page.mode_combo) - _middle(page.open_files_button)) <= 3  # the mode stays with the file
        assert bar.layout().minimumSize().width() < least - 200  # two rows need far less
        assert page.fit_button.geometry().right() > bar.width() - 20  # the actions stay at the right edge

        _size(host, page, 2200)
        assert (bar.short, bar.compact, bar.wrapped) == (False, False, False)
        assert abs(_middle(page.fit_button) - _middle(page.mode_combo)) <= 3  # one row again
        assert page.run_pipeline_button.text() == "Run Automatic Analysis"
        order = [widget for widget in bar._parts() if not widget.isHidden()]
        assert order[-1] is page.fit_button and order.index(page.mode_combo) < order.index(page.run_pipeline_button)
    finally:
        _close(host, page)


@pytest.mark.parametrize("language", ["en", "zh"])
def test_nothing_in_the_bar_is_clipped_from_a_wide_window_down_to_the_narrowest(tmp_path: Path, language) -> None:
    host, page = _hosted(_context(_profiles()), 1800)
    try:
        bar = page.command_bar
        page.set_automatic_analysis(run=lambda: None, find_geometry=lambda: None)
        apply_language(language, [host])
        page.refresh_language()
        widths = list(range(1800, 800, -37)) + [816 + MARGINS]  # down to the page of a 1024 px window
        for files in (0, 1, 3):
            if files:
                page.clear_files()
                page.add_paths([str(path) for path in _frames(tmp_path / str(files), files)])
                _done(page)
            states = []
            for width in widths + widths[::-1]:
                _size(host, page, width)
                state = (bar.short, bar.compact, bar.wrapped)
                assert _cut(page) == [], (language, files, width, state)
                states.append(state)
                room = bar.contentsRect().width()
                if room >= max(COMPACT_WIDTH, bar.row_need() + FULL_AGAIN):
                    assert state == (False, False, False), (files, width)
                if room < bar.row_need(compact=True):
                    assert state == (True, True, True), (files, width)
            narrowing = states[: len(widths)]
            assert narrowing == sorted(narrowing)  # full → short → compact → wrapped, never back while narrowing
            # On two rows nothing is clipped down to what they need (the AI panel docked beside a 1440 px page).
            _size(host, page, bar.row_need(compact=True) + MARGINS - 100)
            assert bar.wrapped
            _size(host, page, bar.layout().minimumSize().width() + MARGINS + 2)
            assert bar.wrapped and _cut(page) == [], (language, files)
            assert bar.width() < 700  # (about 630 px with the application's font)
        # The mode control is whole wherever the page is at least a 1024 px window's.
        assert page.mode_combo.width() >= page.mode_combo.sizeHint().width()
    finally:
        apply_language("en")
        _close(host, page)


def test_more_files_shorten_the_bar_without_a_resize(tmp_path: Path) -> None:
    host, page = _hosted(_context(_profiles()), 1500)
    try:
        bar = page.command_bar
        page.set_automatic_analysis(run=lambda: None, find_geometry=lambda: None)
        paths = _frames(tmp_path, 3)
        page.add_paths([str(paths[0])])
        _done(page)
        # The full row of one file fits with a little to spare (the default 1440 px window is such a width).
        _size(host, page, max(COMPACT_WIDTH, bar.row_need()) + MARGINS + 30)
        assert (bar.short, bar.compact) == (False, False) and _cut(page) == []
        page.add_paths([str(path) for path in paths[1:]])  # the arrows, “2 / 3” and Batch Export… appear
        _done(page)
        for _ in range(4):
            _app().processEvents()
        room = bar.contentsRect().width()
        assert bar.row_need() > room >= COMPACT_WIDTH - 2  # the full row no longer fits, above the fixed threshold
        assert bar.short is True and _cut(page) == []
        assert page.batch_export_button.text() == "Batch…" and page.run_pipeline_button.text() == "Run Analysis"
        if room >= bar.row_need(short=True):  # the short texts are enough: the stepper is still there
            assert bar.compact is False and not page.previous_file_button.isHidden()
        page.clear_files()  # … and without them the full row fits again
        for _ in range(4):
            _app().processEvents()
        assert (bar.short, bar.compact) == (False, False) and _cut(page) == []
        assert page.run_pipeline_button.text() == "Run Automatic Analysis"
    finally:
        _close(host, page)


# -- the Series tab (R16, R75) -------------------------------------------------------------------------------------


def _series_page(tmp_path: Path, width: int, height: int):
    host, page = _hosted(_context(_profiles()), width, height)
    page.set_mode_choice("giwaxs")
    page.add_paths([str(path) for path in _frames(tmp_path, 6)])
    _done(page)
    page.show_right("series")
    page.build_series_map()
    _wait(page, lambda: page._series_map is not None and page._series_map.rows == 6 and not page.batch_running())
    _done(page)
    for _ in range(6):
        _app().processEvents()
    return host, page


def _settle() -> None:
    for _ in range(8):
        _app().processEvents()


def test_the_series_tab_scrolls_in_a_laptop_window_and_keeps_its_map_and_plots_usable(tmp_path: Path) -> None:
    host, page = _series_page(tmp_path, 1072, 690)  # the page of a 1280 × 800 window
    try:
        scroll = page.series_scroll
        assert isinstance(scroll, SeriesScroll) and page.right_pages["series"] is scroll
        assert page.current_right() == "series"
        content, bar = scroll.widget(), scroll.verticalScrollBar()
        view = scroll.viewport()
        assert page.series_plots.stacked()  # a narrow panel: one plot above the other
        # The map's picture and each plot keep their least height …
        assert page.series_map_view.graphics.height() >= SERIES_MAP_HEIGHT
        for plot in (page.series_profile_plot, page.series_trace_plot):
            assert plot.height() >= SERIES_PLOTS_HEIGHT[1], plot.objectName()
        assert page.series_plots.height() >= 2 * SERIES_PLOTS_HEIGHT[1]
        # … so the tab is taller than the view and scrolls, down to its last plot.
        assert content.height() > view.height() and bar.maximum() == content.height() - view.height() > 0
        assert content.height() == scroll.least_height(view.width())
        # Never sideways: the content is exactly as wide as the view, and needs no more.
        assert content.width() == view.width() and scroll.horizontalScrollBar().maximum() == 0
        assert content.minimumSizeHint().width() <= view.width()
        panel = page.splitter.widget(2)
        assert scroll.minimumSizeHint().width() == content.minimumSizeHint().width() + bar.sizeHint().width()
        assert panel.minimumSizeHint().width() <= page.series_controls.minimumSizeHint().width() + 30
        for widget in (page.series_curve_combo, page.series_step_spin, page.series_build_button, page.series_batch_button):
            assert widget.geometry().right() < content.width(), widget.objectName()  # the controls are whole
        # The wheel over a combo that was not clicked scrolls the tab instead of changing the choice.
        combo = page.series_curve_combo
        chosen = combo.currentIndex()
        assert combo.count() > 1 and not combo.hasFocus() and bar.value() == 0
        wheel = QWheelEvent(QPointF(5, 5), QPointF(combo.mapToGlobal(QPoint(5, 5))), QPoint(0, -120), QPoint(0, -120),
                            Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        QApplication.sendEvent(combo, wheel)
        assert combo.currentIndex() == chosen and bar.value() > 0
        bar.setValue(bar.maximum())
        _settle()
        bottom = page.series_trace_plot.mapTo(scroll, page.series_trace_plot.rect().bottomLeft()).y()
        assert bottom <= scroll.height()  # the trace plot's lower edge is in view at the end of the scroll
        small = (content.height(), page.series_map_view.height(), page.series_plots.height())

        # A window with room: nothing scrolls, and the tab is laid out as without a scroll area.
        _size(host, page, 1700, 1300)
        _settle()
        view = scroll.viewport()
        assert bar.maximum() == 0 and not bar.isVisible()
        assert content.height() == view.height() and content.width() == view.width()
        assert page.series_map_view.graphics.height() > SERIES_MAP_HEIGHT
        assert page.series_plots.stacked() and page.series_plots.height() >= 2 * SERIES_PLOTS_HEIGHT[1]
        assert page.series_trace_plot.y() > page.series_profile_plot.y()
        assert page.series_trace_plot.width() == page.series_profile_plot.width()

        # Back in the small window: the same sizes as before (the content does not keep the large window's).
        _size(host, page, 1072, 690)
        _settle()
        assert (content.height(), page.series_map_view.height(), page.series_plots.height()) == small

        # Without a map the tab is short again and does not scroll.
        page.clear_files()
        _settle()
        assert bar.maximum() == 0 and content.height() == scroll.viewport().height()
        assert not page.series_empty.isHidden()
    finally:
        _close(host, page)


def test_a_wheel_over_the_series_tab_reaches_its_scroll_bar_once(tmp_path: Path) -> None:
    """Qt passes the scroll bars' own events through the scroll area's ``eventFilter``. Handing their wheel on
    to ``wheelEvent`` (which sends it to the bar) never ended: every scroll of the tab was a ``RecursionError``
    (an error window in the application). Only a combo or a spin box that was not clicked hands its wheel on."""
    from PyQt5.QtCore import QEvent, QObject

    class Wheels(QObject):
        seen = 0

        def eventFilter(self, _watched, event) -> bool:  # noqa: N802 - Qt API
            if event.type() == QEvent.Wheel:
                self.seen += 1
            return False

    def wheel(widget) -> None:
        point = widget.rect().center()
        QApplication.sendEvent(widget, QWheelEvent(
            QPointF(point), QPointF(widget.mapToGlobal(point)), QPoint(0, -120), QPoint(0, -120), Qt.NoButton,
            Qt.NoModifier, Qt.NoScrollPhase, False))

    host, page = _series_page(tmp_path, 1072, 690)
    wheels = Wheels()
    try:
        scroll = page.series_scroll
        bar = scroll.verticalScrollBar()
        assert bar.maximum() > 0 and bar.value() == 0
        bar.installEventFilter(wheels)
        wheel(scroll.viewport())  # the pointer over the tab itself
        assert wheels.seen == 1 and bar.value() > 0
        for control in (page.series_curve_combo, page.series_step_spin):  # … and over a control not clicked
            before, at = wheels.seen, bar.value()
            chosen = control.currentIndex() if control is page.series_curve_combo else control.value()
            assert not control.hasFocus()
            wheel(control)
            assert wheels.seen == before + 1 and bar.value() > at, control.objectName()
            assert chosen == (control.currentIndex() if control is page.series_curve_combo else control.value())
        wheel(bar)  # over the bar itself
        assert wheels.seen == 4
    finally:
        bar.removeEventFilter(wheels)
        _close(host, page)


def test_the_side_by_side_plots_keep_their_least_height_as_a_real_minimum() -> None:
    from PyQt5.QtWidgets import QPushButton

    from src.gimap.features.analyze.presentation.views.series_view import SideBySide

    _app()
    first, second = QPushButton("frame"), QPushButton("trace")
    plots = SideBySide(first, second, None, least_height=(200, 150))
    try:
        # A widget's own minimum, not only a size hint: a layout sizing by height-for-width keeps it.
        assert plots.minimumHeight() == plots.minimumSizeHint().height() == 200
        plots.resize(plots.side_by_side_width() - 20, 600)  # no room side by side: one above the other
        plots.show()
        _app().processEvents()
        assert plots.stacked() and second.y() > first.y()
        assert plots.minimumHeight() == plots.minimumSizeHint().height() == 2 * 150 + 6
        plots.resize(plots.side_by_side_width() + 200, 600)
        _app().processEvents()
        assert not plots.stacked() and second.y() == first.y()
        assert plots.minimumHeight() == plots.minimumSizeHint().height() == 200
    finally:
        plots.close()
        plots.deleteLater()


# -- the Batch Export dialog (R21) ----------------------------------------------------------------------------------


def _dialog():
    from src.gimap.features.analyze.presentation.batch_dialog import BatchExportDialog

    _app()
    return BatchExportDialog(
        frames=40, files=40, stem="film", settings_text="GIWAXS · profile “synthetic” · αi 0.2°",
        choices=BatchChoices(), destination=Path("."), subfolder=True, series=SeriesCorrection(),
        curve_options=[("radial", "I(q)"), ("azimuthal", "I(χ)")],
    )


def test_the_controls_of_a_row_in_batch_export_do_not_touch() -> None:
    dialog = _dialog()
    try:
        dialog.series_toggle.setChecked(True)  # the folded corrections too
        dialog.show()
        dialog.fit_to_screen(room=4000)
        for _ in range(4):
            _app().processEvents()
        rows = checked = 0
        for layout in dialog.findChildren(QBoxLayout):
            if layout.direction() not in (QBoxLayout.LeftToRight, QBoxLayout.RightToLeft):
                continue
            widgets = [layout.itemAt(index).widget() for index in range(layout.count())]
            pairs = [(a, b) for a, b in zip(widgets, widgets[1:])
                     if a is not None and b is not None and a.isVisible() and b.isVisible()]
            if not pairs:
                continue
            rows += 1
            for first, second in pairs:
                gap = second.geometry().left() - first.geometry().right() - 1
                assert gap >= 4, (first.objectName() or first.metaObject().className(),
                                  second.objectName() or second.metaObject().className(), gap)
                checked += 1
        assert rows >= 6 and checked >= 8  # Every [n] frames, Settings, Format ×3, Colour scale, All | None, Save to
        every = dialog.every_label.geometry().left() - dialog.every_spin.geometry().right() - 1
        assert every == ROW_SPACING  # “Every [1] frame: …” no longer reads as one word
        browse = dialog.browse_button.geometry().left() - dialog.destination_edit.geometry().right() - 1
        assert browse == ROW_SPACING
        both = dialog.no_curves_button.geometry().left() - dialog.all_curves_button.geometry().right() - 1
        assert both == ROW_SPACING
    finally:
        dialog.done(QDialog.Rejected)
        dialog.deleteLater()
