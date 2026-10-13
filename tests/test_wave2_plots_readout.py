"""CurvePlot's cursor readout and the text of an empty plot (audit wave 2: cross-8, cross-6).

The readout shows the values under the cursor in data units (10**v only on an axis that is on log), with
the quantity and unit of each axis label; it floats over a corner of the plot area, so the plot's size and
its header (compact or not) never change. The empty text is centred on the plot area while the plot has no
curves and follows the interface language."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

_APP = None
TITLE = "Horizontal cut at qz = 0.035 Å⁻¹ (sum of 3 rows)"
EMPTY = "Open a frame to see its curves here."


def _app() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _settle(rounds: int = 12) -> None:
    for _ in range(rounds):
        QApplication.processEvents()
        time.sleep(0.005)


def _plot(width: int = 640, height: int = 360, *, signed: bool = False, save: bool = False, **options):
    from src.gimap.app.presentation.components import CurvePlot

    _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    plot = CurvePlot(TITLE, host, **options)
    if save:
        plot.add_save_menu()
    layout.addWidget(plot)
    if signed:
        plot.set_labels("qy (Å⁻¹)", "I (counts)")
        x = np.linspace(-0.3, 0.3, 301)
        plot.set_curves([("horizontal", x, 1e3 * np.exp(-x ** 2 / 0.01) + 5)])
    host.resize(width, height)
    host.show()
    _settle()
    return host, plot


def _q_curve(plot) -> None:
    plot.set_labels("q (Å⁻¹)", "I (counts/pixel)")
    q = np.linspace(0.005, 0.3, 300)
    plot.set_curves([("sample", q, 1e4 * np.exp(-(q / 0.05) ** 2) + 10)])
    _settle()


def _scene(plot, x: float, y: float) -> QPointF:
    return plot.plot.getViewBox().mapViewToScene(QPointF(x, y))


def _hover(plot, x: float, y: float) -> None:
    """The cursor to the view position ``(x, y)`` (log10 on a log axis), through the scene's own signal."""
    plot.plot_widget.scene().sigMouseMoved.emit(_scene(plot, x, y))
    _settle()


def _leave(plot) -> None:
    QApplication.sendEvent(plot.plot_widget.viewport(), QEvent(QEvent.Leave))


def _close(host, plot) -> None:
    plot.dispose()
    host.close()


# -- the readout -----------------------------------------------------------------------------------


def test_axis_parts_takes_the_quantity_and_unit_of_a_label() -> None:
    from src.gimap.app.presentation.components.curve_plot_extras import axis_parts

    assert axis_parts("q (Å⁻¹)") == ("q", "Å⁻¹")
    assert axis_parts("|q| (nm⁻¹)") == ("|q|", "nm⁻¹")
    assert axis_parts("χ or |χ| (°)") == ("χ", "°")
    assert axis_parts("χ or |χ| (°)", folded=True) == ("|χ|", "°")
    assert axis_parts("qy (Å⁻¹)", folded=True) == ("|qy|", "Å⁻¹")
    assert axis_parts("Intensity", fallback="y") == ("I", "")
    assert axis_parts("I (counts/pixel)") == ("I", "counts/pixel")
    assert axis_parts("FWHM (Å⁻¹)") == ("FWHM", "Å⁻¹")
    assert axis_parts("Δ/σ") == ("Δ/σ", "")
    assert axis_parts("", fallback="y") == ("y", "") and axis_parts(None) == ("x", "")


def test_the_readout_shows_data_units_and_unlogs_only_a_log_axis() -> None:
    host, plot = _plot(log_x=False)  # log I on, log q off (with its toggle)
    try:
        _q_curve(plot)
        label = plot.readout.label
        assert label.isHidden()
        _hover(plot, 0.05, np.log10(400.0))
        assert not label.isHidden()
        assert plot.readout.text == "q = 0.05 Å⁻¹ · I = 400 counts/pixel"
        assert label.text() == plot.readout.text
        plot.log_x_check.setChecked(True)  # both axes on log
        _settle()
        _hover(plot, np.log10(0.05), np.log10(400.0))
        assert plot.readout.text == "q = 0.05 Å⁻¹ · I = 400 counts/pixel"
        plot.log_x_check.setChecked(False)
        plot.log_check.setChecked(False)  # both linear: the view values are the values
        _settle()
        _hover(plot, 0.12, 2500.0)
        assert plot.readout.text == "q = 0.12 Å⁻¹ · I = 2500 counts/pixel"
        plot.set_labels("frame", "")  # no unit, no y label (the left axis narrows)
        _settle()
        _hover(plot, 0.12, 2500.0)
        assert plot.readout.text == "frame = 0.12 · y = 2500"
    finally:
        _close(host, plot)


def test_the_readout_goes_when_the_cursor_leaves_the_plot_area() -> None:
    host, plot = _plot()
    try:
        _q_curve(plot)
        label = plot.readout.label
        _hover(plot, 0.1, 2.0)
        assert not label.isHidden()
        _leave(plot)
        assert label.isHidden()
        # A move still waiting in the rate-limiting proxy when the cursor leaves does not bring it back.
        plot.plot_widget.scene().sigMouseMoved.emit(_scene(plot, 0.1, 2.0))
        _leave(plot)
        _settle()
        assert label.isHidden()
        _hover(plot, 0.1, 2.0)
        assert not label.isHidden()
        host.hide()  # another page shown: no stale readout when this one comes back
        host.show()
        _settle()
        assert label.isHidden()
        _hover(plot, 0.1, 2.0)
        area = plot.plot.getViewBox().sceneBoundingRect()  # over the left axis, not the plot area
        plot.plot_widget.scene().sigMouseMoved.emit(QPointF(area.left() - 10, area.center().y()))
        _settle()
        assert label.isHidden()
    finally:
        _close(host, plot)


def test_a_plot_without_curves_has_no_readout() -> None:
    host, plot = _plot()
    try:
        _hover(plot, 0.5, 0.5)
        assert plot.readout.label.isHidden()
        _q_curve(plot)
        _hover(plot, 0.1, 2.0)
        assert not plot.readout.label.isHidden()
        plot.set_curves([])  # under a still cursor
        assert plot.readout.label.isHidden()
    finally:
        _close(host, plot)


def test_the_readout_names_the_halves_on_abs_x() -> None:
    host, plot = _plot(signed=True, log_x=False)
    try:
        _hover(plot, -0.1, 2.0)
        assert plot.readout.text.startswith("qy = -0.1 Å⁻¹ · I = 100 counts")
        plot.set_side("folded")
        _settle()
        _hover(plot, 0.1, 2.0)
        assert plot.readout.text.startswith("|qy| = 0.1 Å⁻¹ · I = 100 counts")
        plot.set_side("positive")
        _settle()
        _hover(plot, 0.1, 2.0)
        assert plot.readout.text.startswith("qy = 0.1 Å⁻¹")
    finally:
        _close(host, plot)


def test_the_readout_never_changes_the_plot_or_its_compact_header() -> None:
    from src.gimap.app.presentation.components.curve_plot_extras import view_rect

    host, plot = _plot(300, 320, signed=True, save=True, log_x=False)
    try:
        assert plot.compact_header.compact and plot.compact_header.wrapped
        header = plot.header_layout

        def layout_state():
            widgets = [header.itemAt(index).widget() for index in range(header.count())]
            return (plot.size(), plot.minimumSizeHint(), plot.plot_widget.geometry(), header.count(),
                    [(widget.objectName(), widget.isHidden(), widget.geometry()) for widget in widgets if widget],
                    plot.title_wrapped.geometry(), plot.compact_header.compact, plot.compact_header.wrapped)

        before = layout_state()
        label = plot.readout.label
        for x in (-0.25, 0.0, 0.25):
            _hover(plot, x, 1.5)
            assert not label.isHidden()
            assert layout_state() == before
        assert label.parent() is plot.plot_widget and label.testAttribute(Qt.WA_TransparentForMouseEvents)
        assert plot.layout().indexOf(label) == -1 and header.indexOf(label) == -1
        area = view_rect(plot)
        assert area.contains(label.geometry()), (area, label.geometry())  # shortened to the plot area
        assert label.text().endswith("…") or label.width() <= area.width()
        assert label.text().startswith("qy = 0.25") and plot.readout.text.startswith("qy = 0.25 Å⁻¹ · I =")
    finally:
        _close(host, plot)


def test_the_readout_moves_away_from_the_cursor() -> None:
    from src.gimap.app.presentation.components.curve_plot_extras import view_rect

    host, plot = _plot(700, 380)
    try:
        _q_curve(plot)
        label = plot.readout.label
        area = view_rect(plot)
        box, widget = plot.plot.getViewBox(), plot.plot_widget

        def at(x: int, y: int):  # the view position under a point of the plot widget
            return box.mapSceneToView(widget.mapToScene(QPoint(x, y) - widget.viewport().pos()))

        _hover(plot, 0.15, 3.0)
        assert label.geometry().center().x() < area.center().x()  # bottom left by default
        assert abs(label.geometry().bottom() - area.bottom()) <= 6
        corner = at(area.left() + 12, area.bottom() - 8)
        _hover(plot, corner.x(), corner.y())
        assert label.geometry().center().x() > area.center().x()  # out from under the cursor
        _hover(plot, 0.15, 3.0)
        assert label.geometry().center().x() > area.center().x()  # stays put while the cursor is elsewhere
        corner = at(area.right() - 12, area.bottom() - 8)
        _hover(plot, corner.x(), corner.y())
        assert label.geometry().center().x() < area.center().x()
    finally:
        _close(host, plot)


def test_a_zoom_under_a_still_cursor_updates_the_readout() -> None:
    host, plot = _plot(log_y=False)
    try:
        _q_curve(plot)
        position = _scene(plot, 0.1, 2000.0)
        plot.plot_widget.scene().sigMouseMoved.emit(position)
        _settle()
        assert plot.readout.text.startswith("q = 0.1 Å⁻¹")
        plot.plot.setXRange(0.2, 0.3, padding=0)
        _settle()
        x = plot.plot.getViewBox().mapSceneToView(position).x()
        assert 0.2 <= x <= 0.3
        assert plot.readout.text.startswith(f"q = {x:.4g} Å⁻¹")
    finally:
        _close(host, plot)


def test_copy_image_leaves_out_the_readout(monkeypatch) -> None:
    host, plot = _plot()
    try:
        _q_curve(plot)
        _hover(plot, 0.1, 2.0)
        label = plot.readout.label
        seen = []
        original = plot.plot_widget.grab
        monkeypatch.setattr(plot.plot_widget, "grab", lambda *args: (seen.append(label.isHidden()), original(*args))[1])
        plot.copy_image()
        assert seen == [True] and not label.isHidden()
        assert not QApplication.clipboard().pixmap().isNull()
    finally:
        _close(host, plot)


def test_the_readout_follows_the_theme() -> None:
    from src.gimap.app.presentation.theme import apply_theme, theme_manager

    host, plot = _plot()
    try:
        for mode in ("dark", "light"):
            apply_theme(mode, 9.0)
            for overlay in (plot.readout, plot.empty_overlay):
                sheet = overlay.label.styleSheet()
                assert theme_manager().color("text_muted").name() in sheet
                fill = theme_manager().color("plot_bg")
                assert f"rgba({fill.red()}, {fill.green()}, {fill.blue()}," in sheet
    finally:
        apply_theme("light", 9.0)
        _close(host, plot)


# -- the text of an empty plot ---------------------------------------------------------------------


def test_the_empty_text_shows_only_while_the_plot_has_no_curves() -> None:
    from src.gimap.app.presentation.components.curve_plot_extras import view_rect

    host, plot = _plot()
    try:
        label = plot.empty_overlay.label
        assert label.isHidden()  # no text: nothing
        plot.set_empty_text(EMPTY)
        _settle()
        assert not label.isHidden() and label.text() == EMPTY
        assert label.testAttribute(Qt.WA_TransparentForMouseEvents) and plot.layout().indexOf(label) == -1
        area, box = view_rect(plot), label.geometry()
        assert abs(box.center().x() - area.center().x()) <= 2 and abs(box.center().y() - area.center().y()) <= 2
        assert box.width() <= area.width()
        _q_curve(plot)
        assert label.isHidden() and plot.has_curves()
        plot.set_curves([])
        assert not label.isHidden() and not plot.has_curves()
        _q_curve(plot)
        plot.clear_curves()
        assert not label.isHidden()
        plot.set_curves([("nothing yet", np.array([]), np.array([]))])  # a curve without a value
        assert not label.isHidden()
        plot.set_empty_text("")
        assert label.isHidden()
    finally:
        _close(host, plot)


def test_an_empty_plot_with_a_text_has_no_meaningless_ticks_or_grid() -> None:
    host, plot = _plot()
    other_host, other = _plot()  # no empty text: the plot stays as it was
    try:
        axes = [plot.plot.getAxis(side) for side in ("left", "bottom")]

        def bare() -> bool:
            levels = [axis._tickLevels for axis in axes]  # [] — no ticks, values or grid lines; None — pyqtgraph's
            assert levels in ([[], []], [None, None]), levels
            return levels == [[], []]

        assert not bare()
        plot.set_empty_text(EMPTY)
        assert bare()
        _q_curve(plot)
        assert not bare() and all(axis.grid == int(0.2 * 255) and axis.style["showValues"] for axis in axes)
        plot.set_curves([])
        assert bare()
        plot.set_empty_text("")
        assert not bare()
        other.set_curves([])
        assert all(other.plot.getAxis(side)._tickLevels is None for side in ("left", "bottom"))
    finally:
        _close(host, plot)
        _close(other_host, other)


def test_the_empty_text_follows_the_plot_area() -> None:
    from src.gimap.app.presentation.components.curve_plot_extras import view_rect

    host, plot = _plot(900, 420)
    try:
        plot.set_empty_text(EMPTY)
        host.resize(420, 300)
        _settle()
        area, box = view_rect(plot), plot.empty_overlay.label.geometry()
        assert abs(box.center().x() - area.center().x()) <= 2 and box.width() <= area.width()
        assert box.height() >= plot.empty_overlay.label.heightForWidth(box.width())
    finally:
        _close(host, plot)


def test_the_empty_text_follows_the_language(monkeypatch) -> None:
    chinese = "打开一帧即可在此看到曲线。"
    if EMPTY not in i18n.ZH:
        monkeypatch.setitem(i18n.ZH, EMPTY, chinese)
        monkeypatch.setitem(i18n._TO_ENGLISH, chinese, EMPTY)
    chinese = i18n.ZH[EMPTY]
    host, plot = _plot()
    try:
        plot.set_empty_text(EMPTY)
        label = plot.empty_overlay.label
        apply_language("zh", [host])
        assert label.text() == chinese
        _q_curve(plot)
        plot.set_curves([])
        assert label.text() == chinese and not label.isHidden()
        apply_language(DEFAULT_LANGUAGE, [host])
        assert label.text() == EMPTY
        apply_language("zh")  # a window the switch does not walk: the plot still follows
        assert label.text() == chinese
    finally:
        apply_language(DEFAULT_LANGUAGE)
        _close(host, plot)


def test_the_empty_text_of_a_page_shown_later_is_centred() -> None:
    from PyQt5.QtWidgets import QStackedWidget

    from src.gimap.app.presentation.components import CurvePlot
    from src.gimap.app.presentation.components.curve_plot_extras import view_rect

    _app()
    stack, first, page = QStackedWidget(), QWidget(), QWidget()
    plot = CurvePlot(TITLE, page, log_y=False)
    plot.set_empty_text(EMPTY)  # before the plot was ever laid out
    QVBoxLayout(page).addWidget(plot)
    stack.addWidget(first)
    stack.addWidget(page)
    stack.resize(600, 400)
    stack.show()
    _settle()
    try:
        stack.setCurrentWidget(page)
        _settle()
        area, box = view_rect(plot), plot.empty_overlay.label.geometry()
        assert plot.empty_overlay.label.isVisible()
        assert abs(box.center().x() - area.center().x()) <= 2 and abs(box.center().y() - area.center().y()) <= 2
    finally:
        plot.dispose()
        stack.close()


def test_a_linked_plot_shows_no_readout_for_the_cursor_over_the_other() -> None:
    from src.gimap.app.presentation.components import CurvePlot

    host, plot = _plot(log_y=False)
    residual = CurvePlot("", host, log_y=False, log_x=None)  # as Fitting's residuals, X-linked to the main plot
    host.layout().addWidget(residual)
    try:
        _q_curve(plot)
        residual.set_labels("q (Å⁻¹)", "Δ/σ")
        residual.set_curves([("residual", np.linspace(0.005, 0.3, 50), np.sin(np.arange(50.0)))])
        residual.plot.setXLink(plot.plot)
        _settle()
        _hover(plot, 0.1, 2000.0)
        plot.plot.setXRange(0.05, 0.2, padding=0)  # a zoom under the cursor moves both
        _settle()
        assert not plot.readout.label.isHidden() and residual.readout.label.isHidden()
    finally:
        residual.dispose()
        _close(host, plot)


def test_a_disposed_plot_ignores_late_signals() -> None:
    from src.gimap.app.presentation.theme import apply_theme

    host, plot = _plot()
    try:
        _q_curve(plot)
        plot.set_empty_text(EMPTY)
        plot.dispose()
        _settle()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        _settle()
        assert not plot.readout.alive() and not plot.empty_overlay.alive()
        apply_theme("light", 9.0)  # the theme's and the language's signals reach the overlays: nothing breaks
        apply_language("zh")
        apply_language(DEFAULT_LANGUAGE)
        plot.readout.refresh()
        plot.empty_overlay.refresh()
    finally:
        apply_language(DEFAULT_LANGUAGE)
        host.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
