"""Calibration tool window: Esc, manual refinement, layout, Open filter, theme, run-time texts.

The AgBh tests calibrate the pyFAI Pilatus1M.edf frame once per module (about 5 s).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from matplotlib.backend_bases import MouseEvent
from PyQt5.QtCore import QCoreApplication, QEvent, QThread, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QAbstractButton,
    QApplication,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QStyle,
    QStyleOptionComboBox,
    QStyleOptionFrame,
    QWidget,
)

from src.gimap.app import AppContext
from src.gimap.features.calibration.domain import MANUAL_REFINEMENT_WARNING
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.integrations.state import (
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGBH = PROJECT_ROOT / "tests/data/external/saxs_agbh_pyfai/Pilatus1M.edf"


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _context() -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
    )


def _settle(app, seconds: float = 0.2) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _wait(app, condition, timeout: float) -> None:
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "timed out"
        _settle(app, 0.05)


def _clipped_texts(root: QWidget) -> list[tuple]:
    """Visible single-line labels, buttons, combo texts and placeholders wider than their room."""
    clipped = []
    for widget in root.findChildren(QWidget):
        if not widget.isVisibleTo(root) or widget.visibleRegion().isEmpty():
            continue
        metrics = widget.fontMetrics()
        if isinstance(widget, QLabel) and widget.text() and not widget.wordWrap():
            if metrics.horizontalAdvance(widget.text()) > widget.contentsRect().width() + 1:
                clipped.append(("label", widget.objectName(), widget.text()))
        elif isinstance(widget, QLineEdit) and not widget.text() and widget.placeholderText():
            option = QStyleOptionFrame()
            widget.initStyleOption(option)
            field = widget.style().subElementRect(QStyle.SE_LineEditContents, option, widget)
            margins = widget.textMargins()
            # QLineEdit keeps 2 px on each side of the text.
            room = field.width() - margins.left() - margins.right() - 4
            if metrics.horizontalAdvance(widget.placeholderText()) > room:
                clipped.append(("placeholder", widget.objectName(), widget.placeholderText()))
        elif isinstance(widget, QAbstractButton) and widget.text():
            if widget.sizeHint().width() > widget.width() + 1:
                clipped.append(("button", widget.objectName(), widget.text()))
        elif isinstance(widget, QComboBox) and widget.currentText():
            option = QStyleOptionComboBox()
            widget.initStyleOption(option)
            field = widget.style().subControlRect(
                QStyle.CC_ComboBox, option, QStyle.SC_ComboBoxEditField, widget
            )
            if metrics.horizontalAdvance(widget.currentText()) > field.width() + 1:
                clipped.append(("combo", widget.objectName(), widget.currentText()))
    return clipped


def test_escape_during_a_run_asks_before_closing(monkeypatch) -> None:
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.show()
    asked = []

    def question(_parent, _title, text, *_args):
        asked.append(text)
        return QMessageBox.No

    monkeypatch.setattr(QMessageBox, "question", question)
    running = QThread()
    running.start()
    dialog._cal_thread = running
    try:
        QTest.keyClick(dialog, Qt.Key_Escape)
        app.processEvents()
        assert asked == ["Cancel calibration and close?"]
        assert dialog.isVisible()
    finally:
        running.quit()
        running.wait(2000)
        dialog._cal_thread = None
    QTest.keyClick(dialog, Qt.Key_Escape)  # idle: Esc closes as before
    app.processEvents()
    assert not dialog.isVisible()


def test_a_reopened_window_does_not_close_itself_after_an_earlier_escape(monkeypatch) -> None:
    """The Tools menu reuses the window: Esc + Yes during a run must not stick to it."""
    app = _app()
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.Yes)
    for reopen_before_run_ends in (False, True):
        dialog = GeometryCalibrationDialog(app_context=_context())
        dialog.show()
        running = QThread()
        running.start()
        dialog._cal_thread = running
        try:
            QTest.keyClick(dialog, Qt.Key_Escape)
            app.processEvents()
            assert not dialog.isVisible() and dialog._close_when_idle
            if reopen_before_run_ends:
                dialog.show()
        finally:
            running.quit()
            running.wait(2000)
        dialog._cleanup_calibration()  # the run ends
        _settle(app, 0.1)
        assert dialog.isVisible() == reopen_before_run_ends
        assert not dialog._close_when_idle
        dialog.show()  # opened again from the Tools menu
        dialog._cleanup_loader()  # the next image load ends
        _settle(app, 0.1)
        assert dialog.isVisible()
        dialog.close()


def test_open_dialog_lists_every_format_the_loader_reads(monkeypatch) -> None:
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    filters = []

    def get_open_file_name(_parent, _title, _folder, file_filter):
        filters.append(file_filter)
        return "", ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", get_open_file_name)
    dialog.open_image_dialog()

    first = filters[0].split(";;")[0]
    for pattern in ("*.nxs", "*.cbf", "*.tif", "*.tiff", "*.edf"):
        assert pattern in first
    assert dialog.open_button.text() == "Open…"
    assert dialog.path_edit.placeholderText() == "Paste a path or use Open…"
    # The empty state is a Qt label over the canvas, never matplotlib text.
    assert dialog.preview_empty_label.text() == "Open a .nxs, .cbf, .tif or .edf calibration image"
    assert not dialog.preview_empty_label.isHidden()
    assert not dialog.axes.texts and not dialog.axes.axison
    dialog.close()


def test_empty_state_text_uses_qt_fonts_in_chinese(monkeypatch) -> None:
    from src.gimap.app.presentation import i18n

    app = _app()
    chinese = "打开 .nxs、.cbf、.tif 或 .edf 标定图像"
    monkeypatch.setitem(i18n.ZH, "Open a .nxs, .cbf, .tif or .edf calibration image", chinese)
    i18n.apply_language("zh")
    try:
        dialog = GeometryCalibrationDialog(app_context=_context())
        dialog.show()
        _settle(app, 0.2)
        assert dialog.preview_empty_label.text() == chinese
        assert dialog.preview_empty_label.isVisible()
        assert not dialog.axes.texts  # nothing for matplotlib's DejaVu Sans to draw as boxes
        dialog.close()
    finally:
        i18n.apply_language("en")


def test_empty_left_column_shows_whole_texts_and_placeholder(monkeypatch) -> None:
    from src.gimap.app.presentation import i18n

    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.show()
    try:
        for width, height in ((1180, 760), (900, 700)):
            dialog.resize(width, height)
            _settle(app, 0.4)
            assert not dialog.path_edit.text()
            assert _clipped_texts(dialog.calibrationLeftPane) == [], (width, height)
    finally:
        dialog.close()
    # The Chinese placeholder fits the narrowest window too.
    monkeypatch.setitem(i18n.ZH, "Paste a path or use Open…", "粘贴路径，或点击“打开…”")
    i18n.apply_language("zh")
    try:
        dialog = GeometryCalibrationDialog(app_context=_context())
        dialog.resize(900, 700)
        dialog.show()
        _settle(app, 0.4)
        assert dialog.path_edit.placeholderText() == "粘贴路径，或点击“打开…”"
        assert [item for item in _clipped_texts(dialog.calibrationLeftPane)
                if item[0] == "placeholder"] == []
        dialog.close()
    finally:
        i18n.apply_language("en")


def _hidden_toolbar_actions(dialog) -> list[str]:
    toolbar = dialog.toolbar
    hidden = []
    for action in toolbar.actions():
        widget = toolbar.widgetForAction(action)
        if action.isSeparator() or not action.text() or widget is None:
            continue
        if not widget.isVisible() or widget.visibleRegion().isEmpty():
            hidden.append(action.text())
    return hidden


def test_navigation_tools_stay_visible_in_narrow_windows() -> None:
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.show()
    try:
        for clean in (False, True):  # 'Clean image' / the wider 'Show overlays'
            dialog.clean_preview_button.setEnabled(True)
            dialog.clean_preview_button.setChecked(clean)
            own_rows = {}
            for width, height in ((900, 700), (1024, 700), (1180, 760), (1500, 950)):
                dialog.resize(width, height)
                _settle(app, 0.4)
                assert _hidden_toolbar_actions(dialog) == [], (clean, width)
                assert dialog.canvas.height() >= 240
                assert dialog.minimumSizeHint().height() <= 720
                own_rows[width] = (
                    dialog.previewToolbarRowLayout.indexOf(dialog.calibrationToolbarHost) >= 0
                )
            # Narrow windows give the toolbar its own row; wide ones keep a single row.
            assert own_rows[900] and not own_rows[1500]
    finally:
        dialog.close()


def test_toolbar_save_writes_matplotlib_colours_not_the_dark_theme(monkeypatch, tmp_path) -> None:
    from matplotlib import rcParams
    from matplotlib.colors import to_hex
    from matplotlib.image import imread

    from src.gimap.app.presentation.theme import apply_theme, theme_color

    _app()
    path = tmp_path / "preview.png"
    monkeypatch.setitem(rcParams, "savefig.directory", str(tmp_path))
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (str(path), "")
    )
    apply_theme("dark", 9)
    try:
        dialog = GeometryCalibrationDialog(app_context=_context())
        dialog.toolbar.save_figure()
        corner = imread(path)[1, 1, :3]
        assert to_hex(tuple(corner)) == to_hex(rcParams["figure.facecolor"])
        assert to_hex(dialog.figure.patch.get_facecolor()) == theme_color("plot_bg").name()
        dialog.close()
    finally:
        apply_theme("light", 9)


def test_left_column_titles_and_compact_combos() -> None:
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())

    assert dialog.calibration_input_section.title_label.text() == "Input"
    assert dialog.calibration_input_group.title() == "Experiment"
    assert dialog.detector_label.wordWrap()
    assert [dialog.range_combo.itemText(i) for i in range(4)] == [
        "Auto · 30–10000 mm",
        "SAXS · 500–10000 mm",
        "WAXS · 30–1500 mm",
        "Custom",
    ]
    for combo in (dialog.standard_combo, dialog.range_combo, dialog.detector_combo):
        assert combo.toolTip()
        assert combo.sizeAdjustPolicy() == QComboBox.AdjustToMinimumContentsLengthWithIcon
    # The run section is pinned outside the scroll area.
    assert not dialog.controls.isAncestorOf(dialog.calibration_run_section)
    assert dialog.calibrationLeftPane.isAncestorOf(dialog.calibration_run_section)
    dialog.close()


def test_runtime_texts_follow_the_interface_language(monkeypatch) -> None:
    from src.gimap.app.presentation import i18n

    _app()
    for english, chinese in (
        ("Finish manual", "完成手动"),
        ("Manual refine", "手动精修"),
        ("Show overlays", "显示叠加"),
        ("Show results", "显示结果"),
    ):
        monkeypatch.setitem(i18n.ZH, english, chinese)
    dialog = GeometryCalibrationDialog(app_context=_context())
    i18n.apply_language("zh")
    try:
        dialog.manual_group.setChecked(True)
        assert dialog.manual_refine_button.text() == "完成手动"
        dialog.manual_group.setChecked(False)
        assert dialog.manual_refine_button.text() == "手动精修"
        dialog.clean_preview_button.setChecked(True)
        assert dialog.clean_preview_button.text() == "显示叠加"
        dialog.expand_preview_button.click()
        assert dialog.expand_preview_button.text() == "显示结果"
        assert dialog.calibration_results_section.isHidden()
    finally:
        i18n.apply_language("en")
        dialog.close()


def test_dark_theme_colours_the_preview_and_follows_a_switch() -> None:
    from src.gimap.app.presentation.theme import apply_theme, theme_color

    _app()
    apply_theme("dark", 9)
    try:
        dialog = GeometryCalibrationDialog(app_context=_context())
        dialog.canvas.draw()
        pixel = np.asarray(dialog.canvas.buffer_rgba())[2, 2, :3]
        assert tuple(pixel) == theme_color("plot_bg").getRgb()[:3]
        assert not dialog.preview_empty_label.isHidden(), "the empty state is shown"
        assert not dialog.axes.axison, "no bare 0–1 axis"
        apply_theme("light", 9)
        dialog.canvas.draw()
        pixel = np.asarray(dialog.canvas.buffer_rgba())[2, 2, :3]
        assert tuple(pixel) == theme_color("plot_bg").getRgb()[:3]
        dialog.close()
    finally:
        apply_theme("light", 9)


@pytest.fixture(scope="module")
def calibrated_dialog():
    """The pyFAI AgBh Pilatus frame, loaded and auto-calibrated once for this module."""
    if not AGBH.is_file():
        pytest.skip("pyFAI AgBh test frame not available")
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.resize(1180, 760)
    dialog.show()
    _settle(app, 0.2)
    dialog.load_image(str(AGBH))
    _wait(app, lambda: dialog._load_thread is None and dialog.image is not None, 60)
    dialog.start_calibration()
    _wait(app, lambda: dialog._cal_thread is None and dialog.result is not None, 240)
    _settle(app, 0.3)
    yield dialog
    dialog.close()


def _mouse(dialog, kind: str, x: float, y: float) -> None:
    dialog.canvas.callbacks.process(kind, MouseEvent(kind, dialog.canvas, x, y, button=1))


def test_zoom_never_moves_the_fitted_center_and_reset_restores_it(calibrated_dialog) -> None:
    dialog = calibrated_dialog
    app = _app()
    fitted = dialog.result.selected_candidate
    fitted_x, fitted_y = fitted.center_x_px, fitted.center_y_px
    assert not dialog.manual_group.isChecked()
    assert dialog.manual_refine_button.text() == "Manual refine"

    # A zoom-rectangle drag starting at (600, 700) or on the center marker, manual mode off/on.
    for manual, start in (
        (False, (600.0, 700.0)),
        (True, (600.0, 700.0)),
        (True, (fitted_x, fitted_y)),
    ):
        dialog.manual_group.setChecked(manual)
        dialog.fit_preview_to_image()
        dialog.canvas.draw()
        dialog.toolbar.zoom()
        try:
            x, y = dialog.axes.transData.transform(start)
            _mouse(dialog, "button_press_event", x, y)
            _mouse(dialog, "motion_notify_event", x + 40, y - 40)
            _mouse(dialog, "button_release_event", x + 40, y - 40)
        finally:
            dialog.toolbar.zoom()  # zoom off again
        assert dialog.manual_x.value() == pytest.approx(fitted_x, abs=1e-3)
        assert dialog.manual_y.value() == pytest.approx(fitted_y, abs=1e-3)
    dialog.manual_group.setChecked(False)
    dialog._commit_manual_values()
    assert dialog.result.selected_candidate.center_x_px == pytest.approx(fitted_x)
    assert dialog.result.selected_candidate.center_y_px == pytest.approx(fitted_y)

    # Manual mode: only a press on the center marker drags it.
    dialog.manual_refine_button.click()
    assert dialog.manual_group.isChecked()
    dialog.fit_preview_to_image()
    dialog.canvas.draw()
    far_x, far_y = dialog.axes.transData.transform((600.0, 700.0))
    _mouse(dialog, "button_press_event", far_x, far_y)
    _mouse(dialog, "motion_notify_event", far_x + 30, far_y)
    _mouse(dialog, "button_release_event", far_x + 30, far_y)
    assert dialog.manual_x.value() == pytest.approx(fitted_x, abs=1e-3)
    marker_x, marker_y = dialog.axes.transData.transform((fitted_x, fitted_y))
    target_x, target_y = dialog.axes.transData.transform((fitted_x + 25.0, fitted_y + 15.0))
    _mouse(dialog, "button_press_event", marker_x + 3, marker_y - 2)
    _mouse(dialog, "motion_notify_event", target_x, target_y)
    _mouse(dialog, "button_release_event", target_x, target_y)
    assert dialog.manual_x.value() == pytest.approx(fitted_x + 25.0, abs=0.5)
    assert dialog.manual_y.value() == pytest.approx(fitted_y + 15.0, abs=0.5)

    dialog.reset_manual_button.click()
    assert dialog.manual_x.value() == pytest.approx(fitted_x, abs=1e-3)
    assert dialog.manual_y.value() == pytest.approx(fitted_y, abs=1e-3)
    dialog.manual_refine_button.click()
    assert not dialog.manual_group.isChecked()
    _settle(app, 0.1)


def _geometry(candidate) -> tuple:
    return (
        candidate.center_x_px,
        candidate.center_y_px,
        candidate.distance_mm,
        list(candidate.warnings),
    )


def test_reset_restores_the_fitted_solution_after_a_commit(calibrated_dialog, monkeypatch) -> None:
    dialog = calibrated_dialog
    app = _app()
    candidate = dialog.result.selected_candidate
    fitted = _geometry(candidate)
    # Export commits the manual values before its save dialog, which is cancelled here.
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: ("", ""))

    dialog.manual_refine_button.click()
    dialog.manual_x.setValue(fitted[0] + 25.0)
    dialog.export_result()
    assert candidate.center_x_px == pytest.approx(fitted[0] + 25.0, abs=1e-3)
    assert MANUAL_REFINEMENT_WARNING in candidate.warnings

    dialog.reset_manual_button.click()
    assert dialog.manual_x.value() == pytest.approx(fitted[0], abs=1e-3)
    assert _geometry(dialog.result.selected_candidate) == fitted

    # Committed, then 'Finish manual': the result is the fitted one again, also for Apply.
    dialog.manual_x.setValue(fitted[0] + 25.0)
    dialog.export_result()
    dialog.manual_refine_button.click()
    assert not dialog.manual_group.isChecked()
    assert _geometry(candidate) == fitted
    dialog._commit_manual_values()
    assert _geometry(candidate) == fitted
    _settle(app, 0.1)


def test_looking_at_the_manual_section_never_changes_the_fitted_solution(
    calibrated_dialog,
) -> None:
    dialog = calibrated_dialog
    app = _app()
    candidate = dialog.result.selected_candidate
    fitted = _geometry(candidate)

    dialog.calibrationManualToggle.click()  # open the section only to look
    assert dialog.calibration_manual_section.is_expanded()
    assert not dialog.manual_group.isChecked()
    dialog._commit_manual_values()  # as Apply and Export do
    assert _geometry(candidate) == fitted

    # 'Use manual values' with nothing edited: no rounding, no 'manually adjusted' warning.
    dialog.manual_group.setChecked(True)
    dialog._commit_manual_values()
    assert _geometry(candidate) == fitted
    dialog.manual_group.setChecked(False)
    assert dialog.calibration_manual_section.is_expanded()  # the user opened it, it stays
    dialog.calibrationManualToggle.click()
    assert not dialog.calibration_manual_section.is_expanded()
    assert not dialog.manual_group.isChecked()
    _settle(app, 0.1)


def test_layout_keeps_the_run_button_preview_and_results_usable(calibrated_dialog) -> None:
    dialog = calibrated_dialog
    app = _app()
    for width, height in ((1180, 760), (1500, 950)):
        dialog.resize(width, height)
        _settle(app, 0.4)
        assert not dialog.calibrate_button.visibleRegion().isEmpty()
        assert dialog.canvas.height() >= 240
        assert dialog.minimumSizeHint().height() <= 720
        # The solution grid (one or two title/value pairs per row): no two labels overlap.
        rects = [
            (label.objectName(), label.geometry())
            for row in dialog.result_rows
            for label in row
        ]
        for index, (name, rect) in enumerate(rects):
            for other_name, other in rects[index + 1:]:
                assert not rect.intersects(other), (name, other_name)
        if (width, height) == (1180, 760):
            assert _clipped_texts(dialog.calibrationLeftPane) == []
    # Focus image hides the whole Results section and gives its height to the preview.
    canvas_height = dialog.canvas.height()
    dialog.expand_preview_button.click()
    _settle(app, 0.3)
    assert dialog.calibration_results_section.isHidden()
    assert dialog.canvas.height() > canvas_height
    dialog.expand_preview_button.click()
    _settle(app, 0.3)
    assert not dialog.calibration_results_section.isHidden()
    dialog.resize(1180, 760)
    QCoreApplication.sendPostedEvents(None, QEvent.LayoutRequest)
