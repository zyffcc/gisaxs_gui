"""Shared widgets at narrow widths, in both themes and both languages: toasts, CurvePlot's header,
legend, window band and right-click menu, the compact detector view, StepRail, AdvancedSection and
JobStatus."""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtGui import QContextMenuEvent, QFontMetrics, QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QActionGroup, QApplication, QLabel, QLineEdit, QVBoxLayout, QWidget

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

TITLE = "Horizontal cut at qz = 0.035 Å⁻¹ (sum of 3 rows)"
_APP = None


def _app() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _settle(rounds: int = 15) -> None:
    for _ in range(rounds):
        QApplication.processEvents()
        time.sleep(0.005)


def _keys(monkeypatch, entries: dict) -> None:
    """Chinese entries the integrator adds to the tables; here only when they are not there yet."""
    for english, chinese in entries.items():
        if english not in i18n.ZH:
            monkeypatch.setitem(i18n.ZH, english, chinese)
            monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)


def _host(widget_factory, width: int, height: int = 320):
    host = QWidget()
    layout = QVBoxLayout(host)
    widget = widget_factory(host)
    layout.addWidget(widget)
    host.resize(width, height)
    host.show()
    _settle()
    return host, widget


def _signed_plot(parent, *, title: str = TITLE):
    from src.gimap.app.presentation.components import CurvePlot

    plot = CurvePlot(title, parent, log_x=False)
    plot.add_save_menu()
    plot.set_labels("qy (Å⁻¹)", "I (counts)")
    x = np.linspace(-1.5, 1.5, 200)
    plot.set_curves([("horizontal", x, 100 * np.exp(-x ** 2 / 0.1) + 5)])
    return plot


# -- toasts -----------------------------------------------------------------------------------


def _toast_parent():
    parent = QWidget()
    parent.resize(1280, 800)
    parent.show()
    _settle(3)
    return parent


def _advance(toasts, ms: int) -> None:
    """``ms`` of simulated time: every timer due by then fires."""
    for toast in toasts:
        timer = toast._timer
        if timer.isActive() and timer.remainingTime() <= ms:
            timer.timeout.emit()
    _settle(3)


def _words(metrics, low: int, high: int) -> str:
    """A sentence between ``low`` and ``high`` pixels wide in ``metrics``."""
    text = "Could not read frame 12:"
    while metrics.horizontalAdvance(text) < low:
        text += " the file is truncated"
    assert metrics.horizontalAdvance(text) < high
    return text


def test_a_short_toast_stays_on_one_line_and_a_long_one_while_the_row_takes_it() -> None:
    """A notice two lines tall rose above the band of the status line, over the bottom axis of the plot there:
    a long text stays on one line as long as the width of the row allows, and wraps only beyond it."""
    from src.gimap.app.presentation.components.toast import MARGIN, show_toast

    _app()
    parent = _toast_parent()
    toast = show_toast(parent, "Added run 1: 60 frames.", level="ok", action=("Open Folder", lambda: None))
    _settle(3)
    metrics = toast.label.fontMetrics()
    assert toast.label.width() >= metrics.horizontalAdvance(toast.text())
    assert toast.label.height() < 2 * metrics.lineSpacing()  # one line
    long = show_toast(parent, _words(metrics, 700, 1000), level="warning")  # (it wrapped beyond 600 px)
    _settle(3)
    assert long.label.height() < 2 * metrics.lineSpacing() and long.label.width() >= metrics.horizontalAdvance(long.text())
    longer = show_toast(parent, _words(metrics, 1400, 1800), level="error")
    _settle(3)
    assert longer.label.height() >= 2 * metrics.lineSpacing() and longer.width() <= parent.width() - 2 * MARGIN
    parent.close()


def test_a_toast_in_a_narrow_parent_wraps_beside_its_buttons() -> None:
    from src.gimap.app.presentation.components.toast import MARGIN, show_toast

    _app()
    parent = _toast_parent()
    toast = show_toast(parent, "Exported 6 curves to cuts_A.", level="ok", action=("Open Folder", lambda: None))
    _settle(3)
    try:
        lines = toast.label.fontMetrics().lineSpacing()
        assert toast.label.height() < 2 * lines
        narrow = toast.width() - 40  # 40 px short of the one-line toast (fonts differ between systems)
        parent.resize(narrow + 2 * MARGIN, 300)
        _settle(3)
        label, action, close = toast.label.geometry(), toast.action_button.geometry(), toast.close_button.geometry()
        assert toast.width() <= narrow and close.right() < toast.width()
        assert label.right() < action.left() and action.right() < close.left()  # nothing drawn over the text
        assert toast.label.height() >= 2 * lines and toast.height() >= toast.label.height()  # it wraps instead
        parent.resize(1280, 300)
        _settle(3)
        assert toast.label.height() < 2 * lines  # room again: one line
    finally:
        parent.close()


def test_at_most_three_toasts_and_one_per_text() -> None:
    from src.gimap.app.presentation.components.toast import MAX_TOASTS, show_toast, visible_toasts

    _app()
    parent = _toast_parent()
    toasts = [show_toast(parent, f"Saved file_{index}.csv", level="ok") for index in range(5)]
    _settle(3)
    shown = visible_toasts(parent)
    assert len(shown) <= MAX_TOASTS == 3
    assert [toast.text() for toast in shown] == [toast.text() for toast in toasts[-3:]]  # the oldest went first
    for toast in shown:
        toast.close()
    _settle(3)
    show_toast(parent, "Saved run.csv", level="ok")
    show_toast(parent, "Saved run.csv", level="ok")
    _settle(3)
    assert [toast.text() for toast in visible_toasts(parent)] == ["Saved run.csv"]
    parent.close()


def test_an_error_stays_until_closed_and_a_notice_goes() -> None:
    from src.gimap.app.presentation.components.toast import show_toast, visible_toasts

    _app()
    parent = _toast_parent()
    error = show_toast(parent, "Could not read frame 12.", level="error")
    ok = show_toast(parent, "Saved run.csv", level="ok")
    warning = show_toast(parent, "2 frames were skipped.", level="warning")
    assert error.timeout_ms == 0 and warning.timeout_ms == 10000 and ok.timeout_ms == 5000
    _advance([error, ok, warning], 6000)
    shown = visible_toasts(parent)
    assert error in shown and warning in shown and ok not in shown
    QApplication.sendEvent(error, QEvent(QEvent.Leave))  # the pointer leaving an error starts no timer
    assert not error._timer.isActive()
    explicit = show_toast(parent, "Undone.", level="error", timeout_ms=2500)  # an explicit value still counts
    assert explicit.timeout_ms == 2500
    _advance([explicit], 3000)
    assert explicit not in visible_toasts(parent)
    for index in range(3):  # more notices than room: the notices go before the error
        show_toast(parent, f"Saved file_{index}.csv", level="ok")
    _settle(3)
    assert error in visible_toasts(parent) and len(visible_toasts(parent)) == 3
    parent.close()


def test_toasts_speak_the_interface_language() -> None:
    from src.gimap.app.presentation.components.toast import show_toast

    _app()
    parent = _toast_parent()
    try:
        apply_language("zh")
        toast = show_toast(parent, "Saved {name} and its record.", level="ok", action=("Open Folder", lambda: None))
        assert toast.action_button.text() == "打开文件夹"
        assert toast.close_button.toolTip() == "关闭"
        assert toast.text() == i18n.ZH["Saved {name} and its record."]
    finally:
        apply_language(DEFAULT_LANGUAGE)
        parent.close()


# -- CurvePlot ----------------------------------------------------------------------------------


def test_a_narrow_plot_puts_its_title_on_a_row_of_its_own() -> None:
    _app()
    host, plot = _host(_signed_plot, 360)
    try:
        assert plot.minimumSizeHint().width() <= 340
        assert plot.header_layout.indexOf(plot.title_label) == 0 and plot.header_layout.indexOf(plot.save_button) == 1
        assert plot.compact_header.compact and plot.compact_header.wrapped
        assert plot.title_wrapped.isVisible() and plot.title_wrapped.text() == TITLE
        assert plot.title_wrapped.height() >= plot.title_wrapped.heightForWidth(plot.title_wrapped.width())
        assert plot.title_label.isHidden() and plot.side_control.isHidden() and plot.log_check.isHidden()
        assert plot.more_button.isVisible()
        host.resize(1000, 320)
        _settle()
        assert not plot.compact_header.compact and not plot.compact_header.wrapped
        assert plot.title_label.isVisible() and not plot.title_wrapped.isVisible()
        assert plot.side_control.isVisible() and plot.log_check.isVisible() and plot.log_x_check.isVisible()
        assert not plot.more_button.isVisible()
        assert plot.title_label.text() == TITLE and plot.figure_state()["title"] == TITLE
    finally:
        plot.dispose()
        host.close()


def test_the_compact_choice_depends_only_on_the_width() -> None:
    _app()
    from src.gimap.app.presentation.components.curve_plot_extras import HYSTERESIS

    host, plot = _host(_signed_plot, 1000)
    try:
        header = plot.compact_header
        title, controls = header.full_width(), header.controls_width()
        assert controls + 2 * HYSTERESIS < title  # two separate thresholds
        steps = lambda threshold: ((threshold + 1, False), (threshold - 1, True), (threshold + 1, True),  # noqa: E731
                                   (threshold + HYSTERESIS - 1, True), (threshold + HYSTERESIS, False),
                                   (threshold - 1, True))
        for width, wrapped in steps(title):  # the title's own row; the controls stay in the header
            for _ in range(3):  # the switch itself changes nothing the decision reads
                header.update(width)
                assert header.wrapped == wrapped and not header.compact, width - title
                assert header.full_width() == title and header.controls_width() == controls
        for width, compact in steps(controls):  # the halves and log toggles in the “⋯” menu
            for _ in range(3):
                header.update(width)
                assert header.compact == compact and header.wrapped, width - controls
                assert header.full_width() == title and header.controls_width() == controls
    finally:
        plot.dispose()
        host.close()


def test_the_more_menu_drives_the_same_controls() -> None:
    _app()
    host, plot = _host(_signed_plot, 360)
    try:
        menu = plot.more_button.menu()
        menu.aboutToShow.emit()
        texts = [action.text() for action in menu.actions()]
        assert "Log I" in texts and "Log q" in texts
        log_i = next(action for action in menu.actions() if action.text() == "Log I")
        assert log_i.isChecked() and plot.log_check.isChecked()
        log_i.trigger()
        assert not plot.log_check.isChecked() and not plot.plot.getAxis("left").logMode
        positive = next(action for action in menu.actions() if action.text().startswith("Only the positive"))
        positive.trigger()
        assert plot.side() == "positive" and plot.side_control.currentData() == "positive"
        for _ in range(4):  # opened again and again: one group of halves, no orphans
            menu.aboutToShow.emit()
        groups = plot.compact_header.findChildren(QActionGroup)
        assert not menu.findChildren(QActionGroup) and len(groups) == 1 and len(groups[0].actions()) == 4
        assert plot.figure_state()["curves"][0][1].min() > 0
        host.resize(1000, 320)
        _settle()
        assert plot.side_control.isVisible() and not plot.log_check.isChecked()  # the controls kept the choices
    finally:
        plot.dispose()
        host.close()


def test_a_hidden_log_toggle_stays_hidden_after_compact() -> None:
    from src.gimap.app.presentation.components import CurvePlot

    _app()

    def factory(parent):
        plot = CurvePlot("Change of each component", parent, sides=False)
        plot.log_check.hide()  # as the Compare page does: a component can be negative
        return plot

    host, plot = _host(factory, 200)
    try:
        host.resize(900, 320)
        _settle()
        assert plot.log_check.isHidden() and not plot.more_button.isVisible()
    finally:
        plot.dispose()
        host.close()


def test_the_legend_has_a_background_in_both_themes() -> None:
    from src.gimap.app.presentation.components import CurvePlot
    from src.gimap.app.presentation.components.curve_plot import LEGEND_ALPHA
    from src.gimap.app.presentation.theme import apply_theme, theme_manager

    _app()
    plot = CurvePlot("")
    x = np.linspace(0.1, 1.0, 20)
    plot.set_curves([("data", x, x)])
    try:
        for mode in ("dark", "light"):
            apply_theme(mode, 9.0)
            assert plot.legend.brush().color().alpha() == LEGEND_ALPHA < 200  # the data under it still shows
            assert plot.legend.brush().color().name() == theme_manager().color("plot_bg").name()
            assert plot.legend.pen().color().name() == theme_manager().color("plot_grid").name()
    finally:
        apply_theme("light", 9.0)
        plot.dispose()


def test_an_empty_name_has_no_legend_entry() -> None:
    from src.gimap.app.presentation.components import CurvePlot

    _app()
    plot = CurvePlot("")
    x = np.linspace(0.1, 1.0, 20)
    plot.set_curves([("", x, x), ("data", x, x * 2)])
    assert len(plot.legend.items) == 1 and plot.curve_count() == 2
    signed = np.linspace(-1, 1, 21)
    plot.set_curves([("", signed, signed ** 2 + 1)])
    plot.set_side("folded")
    assert len(plot.legend.items) == 0 and plot.curve_count() == 2  # both halves, still no entry
    plot.dispose()


def _legend_words(plot) -> list:
    """The legend's entries as they are read on screen (plain text, whatever the markup)."""
    return [label.item.toPlainText() for _sample, label in plot.legend.items]


def test_a_legend_name_with_markup_characters_is_shown_whole() -> None:
    """A legend entry is rich text: “measured (q < 0)” lost everything from the “<” on."""
    from src.gimap.app.presentation.components import CurvePlot

    _app()
    plot = CurvePlot("")
    x = np.linspace(0.1, 1.0, 20)
    names = ["measured (q > 0)", "measured (q < 0)", "a & b <i>not italic</i>", "plain"]
    plot.set_curves([(name, x, x * (index + 1)) for index, name in enumerate(names)])
    try:
        assert _legend_words(plot) == names
        assert [name for name, _x, _y in plot.figure_state()["curves"]] == names  # (the names themselves unchanged)
    finally:
        plot.dispose()


def test_an_exported_figure_has_its_legend_in_english_whatever_the_interface_language(tmp_path, monkeypatch) -> None:
    """Axis titles are figure content and stay English; the legend beside them was Chinese in a Chinese interface."""
    from src.gimap.app.presentation.components import CurvePlot, plot_export

    _app()
    _keys(monkeypatch, {"measured": "测量", "model": "模型"})
    host, plot = _host(lambda parent: CurvePlot("", parent), 700, 420)
    x = np.linspace(0.1, 1.0, 20)
    try:
        apply_language("zh")
        plot.set_labels("q (Å⁻¹)", "I (counts)")
        plot.set_curves([("measured", x, x), ("model", x, x * 2), ("q < 1", x, x * 3)])
        shown = _legend_words(plot)
        assert shown == [i18n.ZH["measured"], i18n.ZH["model"], "q < 1"]
        written, write = [], plot_export._write
        monkeypatch.setattr(plot_export, "_write", lambda item, path, width: (
            written.append(_legend_words(plot)), write(item, path, width))[1])
        path = plot_export.export_plot(plot, tmp_path / "figure.png")
        assert path.is_file() and written == [["measured", "model", "q < 1"]]  # as the axes: not translated
        assert _legend_words(plot) == shown  # … and on screen as before
        monkeypatch.setattr(plot_export, "_write", lambda *_args: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError):
            plot_export.export_plot(plot, tmp_path / "other.png")
        assert _legend_words(plot) == shown  # also when the export failed
    finally:
        apply_language(DEFAULT_LANGUAGE)
        plot.dispose()
        host.close()


def test_what_a_page_adds_to_the_curves_stays_when_the_plot_draws_them_again() -> None:
    """A theme change draws the curves again; the ±1σ bars a page had added to them went with the old ones."""
    import pyqtgraph as pg

    from src.gimap.app.presentation.components import CurvePlot
    from src.gimap.app.presentation.theme import apply_theme

    _app()
    plot = CurvePlot("")
    x = np.linspace(1.0, 20.0, 20)
    plot.set_curves([("R (nm)", x, x)], markers=[True])
    bars = plot.add_extra(plot.plot.plot(np.repeat(x, 2), np.column_stack([x - 1, x + 1]).ravel(),
                                         pen=pg.mkPen("#2563eb"), connect="pairs"))
    redrawn = []
    plot.curvesRedrawn.connect(lambda: redrawn.append(plot.curve_count()))
    try:
        apply_theme("dark", 9.0)
        assert bars in plot.plot.listDataItems() and redrawn == [1]  # the page is told, and its bars are there
        plot.log_check.setChecked(False)
        assert bars in plot.plot.listDataItems() and len(redrawn) == 2
        plot.set_curves([("R (nm)", x, 2 * x)], markers=[True])  # other curves: what belonged to the old ones goes
        assert bars not in plot.plot.listDataItems() and len(redrawn) == 2
        again = plot.add_extra(plot.plot.plot(x, x, pen=pg.mkPen("#f97316")))
        plot.clear_curves()
        assert again not in plot.plot.listDataItems()
    finally:
        apply_theme("light", 9.0)
        plot.dispose()


def test_toasts_sit_side_by_side_in_the_bottom_band_under_the_plots() -> None:
    """Two notices stacked upwards covered the frame axis and the legend of the plot at the bottom right."""
    from src.gimap.app.presentation.components.toast import (
        BOTTOM_MARGIN, MARGIN, SPACING, set_toast_inset, show_toast, visible_toasts)

    _app()
    parent = _toast_parent()
    try:
        first = show_toast(parent, "Series done: 300 frames fitted, 0 failed.", level="ok")
        second = show_toast(parent, "Saved series_fit.csv and its record.", level="ok", action=("Open Folder", lambda: None))
        _settle(3)
        a, b = first.geometry(), second.geometry()
        assert b.right() == parent.width() - MARGIN - 1 and b.bottom() == parent.height() - BOTTOM_MARGIN - 1
        assert a.bottom() == b.bottom() and a.right() == b.left() - SPACING - 1  # the older one to its left
        band = 40  # the status line of a page and the space around it: no plot reaches into it
        assert max(first.height(), second.height()) <= band, (first.height(), second.height())
        # The row keeps off what stands at the left of the window (a sidebar) …
        set_toast_inset(parent, lambda: parent.width() - first.width() - second.width() - 2 * MARGIN - SPACING)
        _settle(3)
        assert visible_toasts(parent) == [first, second] and first.x() == parent.width() - second.width() - first.width() - MARGIN - SPACING
        # … and when it is full the oldest plain confirmation goes to make room (its page's status line said it)
        set_toast_inset(parent, lambda: parent.width() - first.width() - second.width())
        _settle(3)
        assert visible_toasts(parent) == [second] and second.geometry().right() == parent.width() - MARGIN - 1
        # A warning and a notice with a button are shortened rather than put in a row above (over the bottom
        # axis of the plot there), their whole text in the tooltip; the newest is whole, in the corner. When not
        # even that makes room, the oldest goes
        set_toast_inset(parent, 0)
        warning = show_toast(parent, "Frame 12 could not be read: the file is truncated and its header names no detector.",
                             level="warning")
        third = show_toast(parent, "Exported 6 curves to cuts_A.", level="ok", action=("Open Folder", lambda: None))
        parent.resize(third.width() + SPACING + warning.shortest() + 40 + 2 * MARGIN, 400)
        _settle(3)
        assert visible_toasts(parent) == [warning, third]
        w, t = warning.geometry(), third.geometry()
        assert w.bottom() == t.bottom() == parent.height() - BOTTOM_MARGIN - 1  # one row
        assert t.right() == parent.width() - MARGIN - 1 and w.right() == t.left() - SPACING - 1 and w.left() >= MARGIN
        assert third.label.text() == third.text() and warning.toolTip() == warning.text() != warning.label.text()
        assert warning.label.text().endswith("…")
        # An error is shortened too, but it never goes …
        first_error = show_toast(parent, "Could not read frame 13: the file is truncated.", level="error")
        newest = show_toast(parent, "Exported 2 curves to cuts_B.", level="ok", action=("Open Folder", lambda: None))
        parent.resize(newest.width() + SPACING + first_error.shortest() + 30 + 2 * MARGIN, 400)
        _settle(3)
        assert visible_toasts(parent) == [first_error, newest] and first_error.label.text().endswith("…")
        assert first_error.geometry().bottom() == newest.geometry().bottom()
        # … only an error that has no room in the row at all starts a row above
        second_error = show_toast(parent, "Could not read frame 14.", level="error")
        parent.resize(second_error.width() + 2 * MARGIN + 20, 400)
        _settle(3)
        assert visible_toasts(parent) == [first_error, second_error]
        one, two = first_error.geometry(), second_error.geometry()
        assert two.bottom() == parent.height() - BOTTOM_MARGIN - 1 and one.bottom() == two.top() - SPACING - 1
    finally:
        parent.close()


def _band_parent(status: str):
    """A window whose page ends in a status line: its text, a progress bar and Cancel (both hidden)."""
    from PyQt5.QtWidgets import QHBoxLayout, QProgressBar, QSizePolicy, QToolButton

    parent = QWidget()
    parent.resize(1280, 800)
    outer = QVBoxLayout(parent)
    outer.setContentsMargins(0, 0, 0, 0)
    page = QWidget(parent)
    outer.addWidget(page)
    column = QVBoxLayout(page)
    column.setContentsMargins(12, 12, 12, 6)
    column.addWidget(QWidget(page), 1)  # (the plots)
    row = QHBoxLayout()
    label, bar, cancel = QLabel(status, page), QProgressBar(page), QToolButton(page)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)  # (a long line is cut, as on a page)
    bar.setMaximumWidth(220)
    cancel.setText("Cancel")
    row.addWidget(label, 1)
    row.addWidget(bar)
    row.addWidget(cancel)
    column.addLayout(row)
    bar.hide()
    cancel.hide()
    parent.show()
    _settle(3)
    return parent, page, label, bar, cancel


def _look() -> None:
    """Long enough for the notices to look at the band again."""
    from src.gimap.app.presentation.components.toast import LOOK_MS

    QTest.qWait(2 * LOOK_MS)
    _settle(3)


def _in(widget, parent):
    from PyQt5.QtCore import QRect

    return QRect(widget.mapTo(parent, QPoint(0, 0)), widget.size())


def test_toasts_keep_off_the_progress_bar_and_cancel_of_the_band_and_off_its_status_text() -> None:
    """A notice in the band of the status line hid Analyze's progress bar and Cancel while a batch ran, and the
    batch's own line under it."""
    from src.gimap.app.presentation.components.toast import MARGIN, set_toast_inset, show_toast, visible_toasts

    _app()
    parent, page, label, bar, cancel = _band_parent("Batch 0/8: mapi_insitu_scan0042.nxs frame 1 of 8 …")
    set_toast_inset(parent, 0, band=lambda: [page])
    try:
        error = show_toast(parent, "Could not compare: The series have different x axes: qy (Å⁻¹), qz (Å⁻¹)", level="error")
        undo = show_toast(parent, "Removed 2 series.", level="info", action=("Undo", lambda: None))
        _settle(3)
        assert undo.geometry().right() == parent.width() - MARGIN - 1  # nothing at the right end of the band
        bar.show()
        cancel.show()  # a job starts after the notices: they move off its controls
        _look()
        controls = _in(bar, parent).united(_in(cancel, parent))
        assert not any(toast.geometry().intersects(controls) for toast in (error, undo))
        assert undo.geometry().right() < controls.left() and error.geometry().right() < undo.geometry().left()
        text_end = _in(label, parent).left() + label.fontMetrics().horizontalAdvance(label.text())
        assert error.geometry().left() > text_end  # the job's own line stays readable: the older notice is shortened
        assert error.label.text().endswith("…") and error.toolTip() == error.text()
        assert error.text().startswith("Could not compare") and error.geometry().bottom() == undo.geometry().bottom()
        # A status line that takes the whole band: the error is not dropped, at its shortest over the line's end
        label.setText("Batch 3/8: " + "a_long_name_of_a_scan_" * 12 + "frame 4 of 8 …")
        _look()
        assert parent.width() == 1280 and visible_toasts(parent) == [error, undo]
        assert error.geometry().bottom() == undo.geometry().bottom() and error.label.text().endswith("…")
        assert not any(toast.geometry().intersects(controls) for toast in (error, undo))
        # The job ends: the corner again, and the whole text
        bar.hide()
        cancel.hide()
        label.setText("Exported 8 frames.")
        _look()
        assert undo.geometry().right() == parent.width() - MARGIN - 1 and error.label.text() == error.text()
        assert error.toolTip() == ""
        # A status line that says what the newest notice says is no loss: the older ones may lie over it
        undo.close()
        said = "Exported 8 frames to cuts_A: the tables, the records and the map of every frame."
        label.setText(said)
        other = show_toast(parent, "Saved the map as map_A.csv.", level="ok", action=("Open Folder", lambda: None))
        _look()
        assert error.label.text().endswith("…")  # the line says something else than the newest notice: kept readable
        other.close()
        newest = show_toast(parent, said, level="ok")
        _look()
        assert visible_toasts(parent) == [error, newest] and error.label.text() == error.text()
    finally:
        parent.close()


def test_a_page_that_shows_its_own_notices_keeps_them_off_its_cancel_and_off_the_windows() -> None:
    """Analyze shows some notices itself (Export, the end of a batch): they lay over its own Cancel, and under
    the window's notices in the same corner."""
    from src.gimap.app.presentation.components.toast import show_toast, visible_toasts

    _app()
    parent, page, label, bar, cancel = _band_parent("Batch 0/8: run_0042.nxs frame 1 of 8 …")
    bar.show()
    cancel.show()
    try:
        own = show_toast(page, "Could not write the table: the disk is full.", level="error")
        _look()
        controls = _in(bar, parent).united(_in(cancel, parent))
        assert not _in(own, parent).intersects(controls)
        window_notice = show_toast(parent, "Removed 2 series.", level="info", action=("Undo", lambda: None))
        _look()
        assert visible_toasts(page) == [own] and visible_toasts(parent) == [window_notice]
        assert not window_notice.geometry().intersects(_in(own, parent))  # side by side, the first one in the corner
        assert not window_notice.geometry().intersects(controls) and window_notice.geometry().bottom() == _in(own, parent).bottom()
    finally:
        parent.close()


def test_the_window_band_takes_a_colour() -> None:
    from src.gimap.app.presentation.components import CurvePlot

    _app()
    plot = CurvePlot("")
    plot.show_window(0.9, 1.1, color="#9333ea")
    assert all(line.pen.color().name() == "#9333ea" for line in plot.x_window.lines)
    assert plot.x_window.brush.color().name() == "#9333ea" and plot.x_window.brush.color().alpha() < 100
    plot.show_window(0.9, 1.1)
    assert all(line.pen.color().name() == "#f97316" for line in plot.x_window.lines)
    assert plot.x_window.brush.color().getRgb() == (249, 115, 22, 14)  # the band's own look
    assert plot.x_window.getRegion() == pytest.approx((0.9, 1.1))
    plot.dispose()


def _mouse(widget, kind, at: QPoint, button=Qt.NoButton, buttons=Qt.NoButton) -> None:
    QApplication.sendEvent(widget, QMouseEvent(kind, QPointF(at), QPointF(widget.mapToGlobal(at)), button, buttons,
                                               Qt.NoModifier))
    time.sleep(0.02)  # pyqtgraph drops mouse moves closer together than 10 ms


def _context(widget, at: QPoint, reason=QContextMenuEvent.Mouse) -> None:
    QApplication.sendEvent(widget, QContextMenuEvent(reason, at, widget.mapToGlobal(at)))


def test_right_click_opens_the_translated_plot_menu() -> None:
    _app()
    host, plot = _host(_signed_plot, 800)
    try:
        assert not plot.plot.menuEnabled() and not plot.plot.getViewBox().menuEnabled()
        viewport, menu = plot.plot_widget.viewport(), plot.plot_menu.menu
        at = QPoint(200, 120)
        _mouse(viewport, QEvent.MouseMove, at)
        _mouse(viewport, QEvent.MouseButtonPress, at, Qt.RightButton, Qt.RightButton)
        _context(viewport, at)  # macOS and Linux ask for a menu with the press …
        assert not menu.isVisible()  # … when no one knows yet whether a drag follows
        _mouse(viewport, QEvent.MouseButtonRelease, at, Qt.RightButton)
        _context(viewport, at)  # Windows asks with the release
        _settle(3)
        texts = [action.text() for action in menu.actions() if not action.isSeparator()]
        assert menu.isVisible()
        assert texts == ["Reset View", "Log I", "Log q", "Plot as Figure…", "Curves as Data…", "Copy Image", "Copy Data"]
        assert not any("Plot Options" in text for text in texts)
        log_i = next(action for action in menu.actions() if action.text() == "Log I")
        log_i.trigger()  # unchecks it
        assert not plot.log_check.isChecked() and not plot.plot.getAxis("left").logMode
        menu.hide()
        _context(plot.plot_widget, QPoint(5, 5), QContextMenuEvent.Keyboard)  # the menu key
        assert menu.isVisible()
        menu.hide()
        save_texts = [action.text() for action in plot.save_button.menu().actions()]
        assert "Copy Image" in save_texts and "Copy Data" in save_texts
    finally:
        plot.dispose()
        host.close()


def test_a_right_drag_zooms_and_opens_no_menu() -> None:
    _app()
    host, plot = _host(_signed_plot, 800)
    try:
        viewport, menu = plot.plot_widget.viewport(), plot.plot_menu.menu
        before = plot.plot.getViewBox().viewRange()
        start, end = QPoint(200, 120), QPoint(320, 60)
        _mouse(viewport, QEvent.MouseMove, start)
        _mouse(viewport, QEvent.MouseButtonPress, start, Qt.RightButton, Qt.RightButton)
        _context(viewport, start)  # macOS and Linux: with the press
        for step in range(1, 5):
            _mouse(viewport, QEvent.MouseMove, start + (end - start) * (step / 4), Qt.NoButton, Qt.RightButton)
        _mouse(viewport, QEvent.MouseButtonRelease, end, Qt.RightButton)
        _context(viewport, end)  # Windows: with the release
        _settle(3)
        assert not menu.isVisible()
        assert plot.plot.getViewBox().viewRange() != before  # pyqtgraph's right-drag zoom still works
    finally:
        plot.dispose()
        host.close()


def test_a_disposed_plot_still_in_a_layout_is_laid_out_quietly(monkeypatch) -> None:
    from PyQt5 import sip

    _app()
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda *info: errors.append(info))
    host, plot = _host(_signed_plot, 600)
    try:
        plot.dispose()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert sip.isdeleted(plot.plot_widget)
        for width in (420, 900, 360):
            host.resize(width, 320)
            _settle(3)
        assert plot.minimumSizeHint().width() > 0 and not errors
    finally:
        host.close()


def test_the_wrapped_title_keeps_the_title_tooltip() -> None:
    _app()
    host, plot = _host(_signed_plot, 360)
    try:
        sentence = "Horizontal cut at qz = 0.035 Å⁻¹: the sum of rows 410 to 412, both halves"
        plot.title_label.setToolTip(sentence)  # as Analyze does: a short title, the whole sentence here
        assert plot.compact_header.wrapped and plot.title_wrapped.toolTip() == sentence
        plot.set_title("Vertical cut")
        assert plot.title_wrapped.toolTip() == "Vertical cut" == plot.title_wrapped.text()
    finally:
        plot.dispose()
        host.close()


def test_copy_data_puts_the_curves_as_shown_on_the_clipboard() -> None:
    from src.gimap.app.presentation.components import CurvePlot

    _app()
    plot = CurvePlot("", log_y=False)
    plot.set_labels("q (Å⁻¹)", "I")
    plot.set_curves([("a", np.array([0.1, 0.2, 0.3]), np.array([10.0, np.nan, 30.0])),
                     ("b", np.array([0.5, 1.5]), np.array([1.25, 2.5]))])
    plot.copy_data()
    expected = ("a: q (Å⁻¹)\ta: I\tb: q (Å⁻¹)\tb: I\n"
                "0.1\t10.0\t0.5\t1.25\n"
                "0.2\t\t1.5\t2.5\n"
                "0.3\t30.0\t\t\n")
    assert QApplication.clipboard().text() == expected
    plot.copy_image()
    assert not QApplication.clipboard().pixmap().isNull()
    plot.dispose()


# -- the detector view, StepRail, AdvancedSection, JobStatus ----------------------------------


def test_a_narrow_detector_view_keeps_the_image_large() -> None:
    from src.gimap.app.presentation.components import DetectorView

    _app()
    view = DetectorView()
    frame = np.random.default_rng(1).gamma(2.0, 50.0, size=(200, 300)).astype(np.float32)
    view.set_image(frame, context="detector")
    view.resize(300, 300)
    view.show()
    _settle()
    try:
        assert view.color_bar.boundingRect().width() <= 60 and view.color_bar.compact
        assert view.plot.width() >= 200
        assert view.color_bar.vb.width() >= 15  # the handles stay usable
        assert not view.zoom_button.icon().isNull() and view.zoom_button.toolButtonStyle() == Qt.ToolButtonIconOnly
        assert view.plot.getAxis("bottom")._tickDensity == 0.5
        assert view.color_bar.toolTip() == "log₁₀ I" and not view.color_bar.axis.label.isVisible()
        view.levels.button.menu().aboutToShow.emit()
        view.levels.min_edit.setText("5")
        view.levels.max_edit.setText("500")
        view.levels._apply_typed()  # the Levels menu still sets the limits
        assert view.color_bar.levels() == pytest.approx((np.log10(5.0), np.log10(500.0)))
        view.resize(700, 400)
        _settle()
        assert not view.color_bar.compact and view.color_bar.boundingRect().width() > 60
        assert view.zoom_button.icon().isNull() and view.plot.getAxis("bottom")._tickDensity == 1.0
        assert view.color_bar.axis.label.isVisible()
    finally:
        view.dispose()
        view.close()


def test_tick_labels_of_a_short_axis_do_not_run_together() -> None:
    import pyqtgraph as pg

    from src.gimap.app.presentation.components.curve_plot_extras import roomy_ticks

    _app()
    axis = pg.AxisItem("bottom")
    original = axis.tickSpacing(0.0, 2048.0, 600.0)
    axis.tickSpacing = roomy_ticks(axis)
    assert axis.tickSpacing(0.0, 2048.0, 600.0) == original  # room enough: pyqtgraph's own ticks
    major = axis.tickSpacing(0.0, 2048.0, 70.0)[0][0]  # 70 px: “0500100015002000” before
    width = QFontMetrics(axis.font()).horizontalAdvance("1024")
    assert major in (1000.0, 2000.0, 5000.0) and major * 70.0 / 2048.0 >= width + 10
    axis.setTickSpacing(major=100, minor=50)  # an explicit choice is kept
    assert axis.tickSpacing(0.0, 2048.0, 70.0) == [(100, 0), (50, 0)]


def test_the_cleared_detector_view_speaks_the_language() -> None:
    from src.gimap.app.presentation.components import DetectorView

    _app()
    view = DetectorView()
    try:
        apply_language("zh")
        view.clear()
        assert view.readout_label.text() == i18n.ZH["Open a detector frame"]
    finally:
        apply_language(DEFAULT_LANGUAGE)
        view.dispose()


def test_a_step_shows_every_line_of_its_detail_and_tab_reaches_it() -> None:
    from src.gimap.app.presentation.components.step_rail import StepRail

    _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    edit = QLineEdit(host)
    layout.addWidget(edit)
    rail = StepRail((("data", "Data"), ("geometry", "Geometry")), host)
    layout.addWidget(rail)
    layout.addStretch(1)
    host.resize(220, 400)
    host.show()
    host.activateWindow()
    QApplication.setActiveWindow(host)
    rail.set_state("data", "ok", "PILATUS 2M · 1 frame · galaxi_data.tif, from the GALAXI beamline")
    _settle()
    try:
        step = rail._steps["data"]
        detail = step.detail
        needed = detail.heightForWidth(detail.width())
        assert needed >= 3 * detail.fontMetrics().lineSpacing() - 2  # three lines (or more) at this width
        assert detail.height() >= needed
        narrow = detail.height()
        host.resize(600, 400)
        _settle()
        assert detail.height() < narrow and detail.height() >= detail.heightForWidth(detail.width())  # no gap left
        host.resize(220, 400)
        _settle()
        assert detail.height() == narrow
        QTest.mouseClick(step, Qt.LeftButton)
        _settle(3)
        assert not step.hasFocus() and step.focusPolicy() == Qt.TabFocus
        edit.setFocus()
        _settle(3)
        QTest.keyClick(edit, Qt.Key_Tab)
        _settle(3)
        assert step.hasFocus()
    finally:
        host.close()


def test_the_whole_advanced_row_is_the_toggle() -> None:
    from src.gimap.app.presentation.components.sections import AdvancedSection

    _app()
    host, section = _host(lambda parent: AdvancedSection("Advanced", "", parent), 400, 200)
    section.add_widget(QLabel("content"))
    _settle()
    try:
        inner = section.contentsRect().width() - 16  # the section's own margins (8 + 8)
        assert section.toggle_button.width() >= inner - 1
        section.toggle_button.click()
        assert section.is_expanded()
    finally:
        host.close()


def test_job_status_texts_follow_the_language(monkeypatch) -> None:
    from src.gimap.app.presentation.components.feedback import JobStatus

    _app()
    _keys(monkeypatch, {"Running": "运行中", "Paused": "已暂停", "Succeeded": "已完成", "Queued": "排队中",
                        "Failed": "失败", "Cancelled": "已取消", "Timed out": "超时"})
    job = JobStatus()
    assert job.state_label.text() == "IDLE" and job.pause_button.text() == "Pause"
    job.set_state("running", "Working", progress=0.5)
    assert job.state_label.text() == "RUNNING"
    try:
        apply_language("zh")
        job = JobStatus()
        assert job.state_label.text() == i18n.ZH["Idle"] and job.message_label.text() == i18n.ZH["Ready"]
        for state in JobStatus.STATES:
            job.set_state(state)
            assert job.state_label.text() == i18n.ZH[JobStatus.STATE_TEXT[state]]
            assert not job.state_label.text().isascii()
        job.set_state("paused")
        assert job.pause_button.text() == i18n.ZH["Resume"]
        job.set_state("running")
        assert job.pause_button.text() == i18n.ZH["Pause"]
    finally:
        apply_language(DEFAULT_LANGUAGE)
