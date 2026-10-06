"""Review fixes of the shared components: Copy Data says when a half is drawn on |x|, the folded halves
differ in colour, an empty legend draws no box, a long title wraps without hiding the controls, and the
JobStatus chip follows a switch of the language."""

from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget

from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

_APP = None
LONG_TITLE = "Horizontal cut at qz = 0.035 Å⁻¹: the sum of rows 410 to 412, both halves of the frame"


def _app() -> QApplication:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _settle(rounds: int = 15) -> None:
    for _ in range(rounds):
        QApplication.processEvents()
        time.sleep(0.005)


def _signed_plot(title: str = ""):
    from src.gimap.app.presentation.components import CurvePlot

    plot = CurvePlot(title, log_y=False)
    plot.set_labels("qy (Å⁻¹)", "I")
    plot.set_curves([("I(qy)", np.array([-0.02, -0.01, 0.01, 0.02]), np.array([5.0, 6.0, 7.0, 8.0]))])
    return plot


# -- Copy Data ------------------------------------------------------------------------------------


def test_copy_data_labels_the_halves_drawn_on_abs_qy() -> None:
    _app()
    plot = _signed_plot()
    try:
        plot.set_side("folded")
        plot.copy_data()
        lines = QApplication.clipboard().text().splitlines()
        header = lines[0].split("\t")
        assert header == ["I(qy) (+): |qy| (Å⁻¹)", "I(qy) (+): I", "I(qy) (−): |qy| (Å⁻¹)", "I(qy) (−): I"]
        assert "|qy|" in header[2]  # the (−) column holds −qy: never under a plain “qy” header
        assert lines[1:] == ["0.01\t7.0\t0.01\t6.0", "0.02\t8.0\t0.02\t5.0"]  # as shown, in data units
        plot.set_side("negative")
        plot.copy_data()
        lines = QApplication.clipboard().text().splitlines()
        assert lines[0] == "I(qy) (−): qy (Å⁻¹)\tI(qy) (−): I"  # one half: the measured qy, and which half
        assert lines[1:] == ["-0.02\t5.0", "-0.01\t6.0"]
        plot.set_side("both")
        plot.copy_data()
        lines = QApplication.clipboard().text().splitlines()
        assert lines[0] == "I(qy): qy (Å⁻¹)\tI(qy): I" and len(lines) == 5
    finally:
        plot.dispose()


def test_folded_labels_keep_the_unit() -> None:
    from src.gimap.app.presentation.components.curve_plot_extras import axis_symbol, curves_text, folded_label

    assert folded_label("qy (Å⁻¹)") == "|qy| (Å⁻¹)"
    assert folded_label("χ or |χ| (°)") == "|χ| (°)" and axis_symbol("χ or |χ| (°)") == "χ"
    assert folded_label("") == "|x|" and folded_label("|qy| (Å⁻¹)") == "|qy| (Å⁻¹)"
    text = curves_text([("a", [1.0], [2.0]), ("b", [3.0], [4.0])], "q", "I", x_labels=["|q|", ""])
    assert text.splitlines()[0] == "a: |q|\ta: I\tb: q\tb: I"  # an empty entry: the plot's own label


# -- the folded halves in two shades --------------------------------------------------------------


def test_the_folded_negative_half_is_a_second_shade_of_its_curve() -> None:
    from src.gimap.app.presentation.components.curve_plot import CURVE_COLORS
    from src.gimap.app.presentation.theme import apply_theme

    _app()
    plot = _signed_plot()
    try:
        for mode, shade in (("light", QColor(CURVE_COLORS[0]).darker(170)), ("dark", QColor(CURVE_COLORS[0]).lighter(150))):
            apply_theme(mode, 9.0)
            plot.set_side("folded")
            colors = plot.figure_state()["colors"]
            assert colors == [CURVE_COLORS[0], shade.name()] and colors[1] not in CURVE_COLORS  # not another curve's colour
            assert [item.opts["pen"].style() for item in plot._items] == [Qt.SolidLine, Qt.DashLine]
            assert plot.curve_colors() == [CURVE_COLORS[0]]  # one per curve given (the Sources overlay pairs them)
        x = np.array([-0.02, -0.01, 0.01, 0.02])
        plot.set_curves([("data", x, x + 1), ("fit", x, x + 2)])
        assert plot.curve_colors() == list(CURVE_COLORS[:2]) and len(plot.figure_state()["colors"]) == 4
        assert plot.figure_state()["curves"][1][1].min() > 0  # values: only the drawing changes (−x on |x|)
    finally:
        apply_theme("light", 9.0)
        plot.dispose()


# -- the legend ---------------------------------------------------------------------------------


def test_an_empty_legend_draws_no_box_and_a_hidden_legend_stays_hidden() -> None:
    from src.gimap.app.presentation.components import CurvePlot
    from src.gimap.app.presentation.components.curve_plot import LEGEND_ALPHA

    _app()
    plot = CurvePlot("")
    x = np.linspace(0.1, 1.0, 20)
    try:
        assert plot.legend.brush().color().alpha() == 0 and plot.legend.pen().style() == Qt.NoPen
        plot.set_curves([("data", x, x)])
        assert plot.legend.brush().color().alpha() == LEGEND_ALPHA and plot.legend.pen().style() != Qt.NoPen
        plot.set_curves([("", x, x)])  # a guide line only: no entry, no box
        assert not plot.legend.items and plot.legend.brush().color().alpha() == 0 and plot.legend.pen().style() == Qt.NoPen
        plot.set_curves([("data", x, x)])
        plot.clear_curves()
        assert plot.legend.brush().color().alpha() == 0
        plot.legend.hide()  # as the Fitting residual plot does
        plot.set_curves([("data", x, x)])
        assert not plot.legend.isVisible()
    finally:
        plot.dispose()


# -- the header of a narrow plot ------------------------------------------------------------------


def test_a_long_title_wraps_but_the_halves_and_log_stay_in_the_header() -> None:
    from src.gimap.app.presentation.components import CurvePlot

    _app()
    host = QWidget()
    layout = QVBoxLayout(host)
    layout.setContentsMargins(0, 0, 0, 0)
    plot = CurvePlot(LONG_TITLE, host, log_x=False)
    plot.add_save_menu()
    plot.set_labels("qy (Å⁻¹)", "I (counts)")
    x = np.linspace(-1.5, 1.5, 200)
    plot.set_curves([("horizontal", x, 100 * np.exp(-x ** 2 / 0.1) + 5)])
    layout.addWidget(plot)
    host.resize(560, 320)
    host.show()
    _settle()
    try:
        header = plot.compact_header
        assert header.controls_width() < 560 < header.full_width()
        assert header.wrapped and not header.compact
        assert plot.title_wrapped.isVisible() and plot.title_wrapped.text() == LONG_TITLE
        assert plot.side_control.isVisible() and plot.log_check.isVisible() and plot.log_x_check.isVisible()
        assert not plot.more_button.isVisible()
        host.resize(300, 320)
        _settle()
        assert header.wrapped and header.compact
        assert plot.side_control.isHidden() and plot.log_check.isHidden() and plot.more_button.isVisible()
        host.resize(560, 320)
        _settle()
        assert not header.compact and plot.side_control.isVisible() and plot.log_check.isVisible()
    finally:
        plot.dispose()
        host.close()


# -- JobStatus ----------------------------------------------------------------------------------


def _chinese_states(monkeypatch) -> None:
    for english, chinese in {"Running": "运行中", "Idle": "空闲", "Ready": "就绪"}.items():
        if english not in i18n.ZH:
            monkeypatch.setitem(i18n.ZH, english, chinese)
            monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)


def test_the_job_chip_follows_a_switch_of_the_language(monkeypatch) -> None:
    from src.gimap.app.presentation.components.feedback import JobStatus

    _app()
    _chinese_states(monkeypatch)
    window = QWidget()
    window_layout = QVBoxLayout(window)
    started = JobStatus(window)  # built in English, as every workspace is before the saved language is applied
    started.set_state("idle", "Idle")
    window_layout.addWidget(started)
    elsewhere = JobStatus()  # in no window the switch walks
    elsewhere.set_state("running", "Working")
    try:
        assert started.state_label.text() == "IDLE"
        apply_language("zh", [window])
        assert started.state_label.text() == i18n.ZH["Idle"] and started.message_label.text() == i18n.ZH["Idle"]
        assert elsewhere.state_label.text() == i18n.ZH["Running"]
        started.set_state("running")
        apply_language(DEFAULT_LANGUAGE, [window])
        assert started.state_label.text() == "RUNNING" and elsewhere.state_label.text() == "RUNNING"  # not “Running”
    finally:
        apply_language(DEFAULT_LANGUAGE)


def test_the_language_signal_comes_once_per_switch() -> None:
    _app()
    seen = []
    signal = i18n.language_changed()
    signal.connect(seen.append)
    try:
        apply_language("zh")
        apply_language("zh")  # no switch
        apply_language(DEFAULT_LANGUAGE)
        assert seen == ["zh", DEFAULT_LANGUAGE]
    finally:
        signal.disconnect(seen.append)
        apply_language(DEFAULT_LANGUAGE)
