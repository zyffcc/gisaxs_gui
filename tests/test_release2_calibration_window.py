"""Geometry Calibration shows and accepts the beam centre in the pixels of Analyze.

The fit keeps pixel-centre indices (the centre of the first pixel is 0.0); the window, like Analyze and the
instrument profile that Apply saves, uses canonical detector pixels (pixel corners on integers, the centre of
the first pixel is 0.5). These tests calibrate the public pyFAI AgBh frame once in the window (about 5 s) and
compare what the window shows with what is saved and with what Analyze shows for the same frame.
"""

from __future__ import annotations

import ast
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from matplotlib.backend_bases import MouseEvent
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.gimap.app import AppContext
from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.components import visible_toasts
from src.gimap.features.calibration.domain import DetectorImage
from src.gimap.features.calibration.presentation import pixel_coordinates
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from src.gimap.shared.geometry import transmission_map

ROOT = Path(__file__).resolve().parents[1]
AGBH = ROOT / "tests/data/external/saxs_agbh_pyfai/Pilatus1M.edf"
PRESENTATION = ROOT / "src/gimap/features/calibration/presentation"
PIXEL_NOTE = (
    "Direct-beam position in detector pixels, the same as in Analyze: pixel j spans [j, j+1] "
    "(the centre of the first pixel is 0.5), x from the left edge and y from the top edge."
)
PIXEL_NOTE_ZH = (
    "直射束在探测器上的像素位置，与分析页相同：第 j 个像素覆盖 [j, j+1]"
    "（第一个像素的中心是 0.5），x 从左边缘算起，y 从上边缘算起。"
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _context(profiles=None) -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        instrument_profiles=profiles if profiles is not None else InMemoryInstrumentProfileRepository(),
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


def _dispose(dialog) -> None:
    """Close the window and delete it now, by the event loop (not inside a garbage collection)."""
    for name in ("_load_thread", "_cal_thread"):
        thread = getattr(dialog, name, None)
        if thread is not None:
            thread.wait(10000)
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _calibrated(context, frame: Path) -> GeometryCalibrationDialog:
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=context)
    dialog.resize(1180, 760)
    dialog.show()
    _settle(app, 0.2)
    dialog.load_image(str(frame))
    _wait(app, lambda: dialog._load_thread is None and dialog.image is not None, 60)
    dialog.start_calibration()
    _wait(app, lambda: dialog._cal_thread is None and dialog.result is not None, 240)
    _settle(app, 0.3)
    return dialog


@pytest.fixture(scope="module")
def frame(tmp_path_factory) -> Path:
    """A copy of the public frame: nothing of a test is ever written beside the repository's data."""
    if not AGBH.is_file():
        pytest.skip("pyFAI AgBh test frame not available")
    target = tmp_path_factory.mktemp("agbh") / AGBH.name
    shutil.copy2(AGBH, target)
    return target


@pytest.fixture(scope="module")
def window(frame):
    """The AgBh frame calibrated in the window, with an instrument-profile store of its own."""
    context = _context()
    dialog = _calibrated(context, frame)
    yield SimpleNamespace(dialog=dialog, context=context, frame=frame)
    _dispose(dialog)


@pytest.fixture
def yes(monkeypatch):
    """Apply may ask before it overwrites a profile that moved a lot: yes."""
    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: pytest.fail("no warning expected"))


def _apply(dialog):
    for toast in visible_toasts(dialog.right):
        toast.close()
    dialog.apply_result()
    _settle(_app(), 0.1)
    return dialog.view_model.last_profile


def _numbers(text: str) -> tuple[float, ...]:
    return tuple(float(part.replace("px", "").strip()) for part in text.split(","))


def _analyze_page(context):
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage

    _app()
    return AnalyzePage(create_analyze_view_model(context))


def _value_under(dialog, x: float, y: float) -> float:
    """The value the toolbar's readout gives at the position (x, y) of the axes.

    The readout asks the image with the mouse position on the screen; a real mouse event keeps whole screen
    pixels only (several detector pixels at this window size), so the exact position stands in for it.
    """
    screen_x, screen_y = dialog.axes.transData.transform((x, y))
    return float(dialog.axes.images[0].get_cursor_data(SimpleNamespace(x=screen_x, y=screen_y)))


# -- what the window shows is what is saved and what Analyze shows -----------------------------------


def test_every_number_of_the_window_is_the_centre_of_the_saved_profile(window, yes) -> None:
    dialog = window.dialog
    candidate = dialog.result.selected_candidate
    fitted = (candidate.center_x_px, candidate.center_y_px, candidate.distance_mm)
    profile = _apply(dialog)
    saved = (profile.geometry.beam_center_x_px, profile.geometry.beam_center_y_px)

    # The fit is what it was (pixel-centre indices); the profile is the same point in canonical pixels.
    assert (candidate.center_x_px, candidate.center_y_px, candidate.distance_mm) == fitted
    assert saved == (fitted[0] + 0.5, fitted[1] + 0.5)
    assert saved == pytest.approx((182.97, 264.84), abs=0.05)  # the documented direct beam is (180.0, 263.9)
    assert profile.geometry.distance_m == pytest.approx(fitted[2] * 1e-3)

    assert dialog.result_center_x.text() == f"{saved[0]:.3f} px"
    assert dialog.result_center_y.text() == f"{saved[1]:.3f} px"
    assert dialog.result_distance.text() == f"{fitted[2]:.3f} mm"
    assert (dialog.manual_x.value(), dialog.manual_y.value()) == pytest.approx(saved, abs=5e-4)
    assert dialog.manual_distance.value() == pytest.approx(fitted[2], abs=5e-4)
    assert dialog.candidate_table.item(0, 2).text() == f"{saved[0]:.1f}, {saved[1]:.1f}"
    for row, other in enumerate(dialog.result.candidates):
        assert _numbers(dialog.candidate_table.item(row, 2).text()) == pytest.approx(
            (other.center_x_px + 0.5, other.center_y_px + 0.5), abs=0.05
        )
    (toast,) = visible_toasts(dialog.right)
    assert f"centre ({saved[0]:.2f}, {saved[1]:.2f}) px, distance {fitted[2]:.2f} mm." in toast.text()

    # The settings other tools read keep the calibration's own numbers, as before.
    settings = window.context.settings
    assert (settings.get("detector", "beam_center_x"), settings.get("detector", "beam_center_y")) == fitted[:2]


def test_the_preview_is_drawn_in_the_same_pixels(window) -> None:
    dialog = window.dialog
    candidate = dialog.result.selected_candidate
    shown = (candidate.center_x_px + 0.5, candidate.center_y_px + 0.5)
    dialog.fit_preview_to_image()
    dialog.canvas.draw()
    rows, columns = dialog.image.data.shape

    # Pixel j spans j…j+1 and row 0 is at the top: the frame fills 0…columns, 0…rows.
    assert tuple(dialog.axes.images[0].get_extent()) == (0.0, float(columns), float(rows), 0.0)
    assert dialog.axes.get_xlim() == (0.0, float(columns))
    assert dialog.axes.get_ylim() == (float(rows), 0.0)

    vertical, horizontal = dialog.axes.lines
    assert tuple(vertical.get_xdata()) == (shown[0], shown[0])
    assert tuple(horizontal.get_ydata()) == (shown[1], shown[1])
    assert dialog.axes.patches, "the ring overlays are drawn"
    assert {tuple(patch.center) for patch in dialog.axes.patches} == {shown}
    # Lengths are no coordinates: the detected rings keep their radii.
    drawn = [patch.width / 2 for patch in dialog.axes.patches if patch.get_linestyle() == ":"]
    assert drawn
    for radius in drawn:
        assert any(radius == pytest.approx(detected) for detected in candidate.detected_peak_radii_px)

    # The value under the cursor anywhere inside pixel (row, column) is that pixel's.
    data = np.asarray(dialog.image.data, dtype=float)
    valid = np.isfinite(data) & (data > 0)
    if dialog.image.mask is not None:
        valid &= ~np.asarray(dialog.image.mask, dtype=bool)
    row, column = np.unravel_index(np.argmax(np.where(valid, data, -np.inf)), data.shape)
    brightest = np.log1p(data[row, column])  # the preview shows log intensity
    assert dialog.log_check.isChecked()
    for dx, dy in ((0.5, 0.5), (0.05, 0.05), (0.95, 0.95)):
        assert _value_under(dialog, column + dx, row + dy) == pytest.approx(brightest, rel=1e-6)
    assert _value_under(dialog, column - 0.05, row + 0.5) != pytest.approx(brightest, rel=1e-6)


def test_analyze_shows_the_numbers_of_the_window_for_the_applied_profile(window, yes) -> None:
    dialog = window.dialog
    profile = _apply(dialog)
    saved = (profile.geometry.beam_center_x_px, profile.geometry.beam_center_y_px)
    page = _analyze_page(window.context)
    try:
        page.add_paths([window.frame])
        assert page.tasks.wait(120)
        analysis = page.view_model.state.analysis
        assert analysis.resolution.profile.name == profile.name
        geometry = analysis.geometry
        assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == saved
        assert geometry.distance_m == profile.geometry.distance_m
        # The same digits on both screens: the window's table and Analyze's beam-centre button.
        table = dialog.candidate_table.item(0, 2).text()
        assert page.center_button.text() == f"Beam {table} px · profile"
        assert (dialog.result_center_x.text(), dialog.result_center_y.text()) == (
            f"{geometry.beam_center_x_px:.3f} px", f"{geometry.beam_center_y_px:.3f} px"
        )
    finally:
        page.dispose()
        page.close()


# -- what is typed or dragged in the window ---------------------------------------------------------


def test_a_centre_typed_in_the_window_lands_on_that_pixel_in_analyze(window, yes) -> None:
    dialog = window.dialog
    app = _app()
    fitted = dialog.result.selected_candidate
    before = (fitted.center_x_px, fitted.center_y_px, fitted.distance_mm)
    row, column = 300, 200
    try:
        dialog.manual_refine_button.click()
        assert dialog.manual_group.isChecked()
        # The centre of the pixel in row 300, column 200, as Analyze counts pixels.
        dialog.manual_x.setValue(column + 0.5)
        dialog.manual_y.setValue(row + 0.5)
        _settle(app, 0.2)  # the overlay follows the fields
        vertical, horizontal = dialog.axes.lines
        assert vertical.get_xdata()[0] == column + 0.5 and horizontal.get_ydata()[0] == row + 0.5

        profile = _apply(dialog)
        geometry = profile.geometry
        assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == (column + 0.5, row + 0.5)
        assert geometry.distance_m == pytest.approx(before[2] * 1e-3, abs=1e-6)  # the distance was not typed
        (toast,) = visible_toasts(dialog.right)
        assert "centre (200.50, 300.50) px" in toast.text()
        # Inside, the fit's own pixels: the index of that pixel.
        committed = dialog.result.selected_candidate
        assert (committed.center_x_px, committed.center_y_px) == (float(column), float(row))

        page = _analyze_page(window.context)
        try:
            page.add_paths([window.frame])
            assert page.tasks.wait(120)
            analysis = page.view_model.state.analysis
            assert page.center_button.text() == "Beam 200.5, 300.5 px · profile"
            # q = 0 at the centre of that pixel, and no other pixel is nearer to the direct beam.
            q = transmission_map(analysis.shape, analysis.geometry).q
            assert np.unravel_index(np.argmin(q), q.shape) == (row, column)
            assert q[row, column] == pytest.approx(0.0, abs=1e-12)
        finally:
            page.dispose()
            page.close()
    finally:
        dialog.manual_group.setChecked(False)  # back to the fitted solution
        dialog._commit_manual_values()
        _settle(app, 0.2)
    restored = dialog.result.selected_candidate
    assert (restored.center_x_px, restored.center_y_px, restored.distance_mm) == before


def test_a_typed_centre_is_saved_and_exported_as_the_same_point(window, yes, tmp_path, monkeypatch) -> None:
    import json

    dialog = window.dialog
    app = _app()
    fitted = dialog.result.selected_candidate
    before = (fitted.center_x_px, fitted.center_y_px, fitted.distance_mm)
    typed = (411.237, 512.864)  # no round numbers: nothing hides in a rounding
    target = tmp_path / "typed.gimap-calibration.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(target), ""))
    other = None
    try:
        dialog.manual_refine_button.click()
        dialog.manual_x.setValue(typed[0])
        dialog.manual_y.setValue(typed[1])
        _settle(app, 0.2)
        geometry = _apply(dialog).geometry
        assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == pytest.approx(typed, abs=1e-9)
        (toast,) = visible_toasts(dialog.right)
        assert "centre (411.24, 512.86) px" in toast.text()

        # The exported file holds the same point in its own convention (half a pixel lower), and says so.
        dialog.export_result()
        written = json.loads(target.read_text(encoding="utf-8"))
        assert written["metadata"]["center_convention"]["name"] == "index"
        stored = written["selected_candidate"]
        assert (stored["center_x_px"], stored["center_y_px"]) == pytest.approx((410.737, 512.364), abs=1e-9)
        assert "Geometry was manually adjusted after automatic calibration." in stored["warnings"]

        # Imported again, the window shows what was typed.
        other = GeometryCalibrationDialog(app_context=_context())
        other.import_result_from(str(target))
        assert (other.result_center_x.text(), other.result_center_y.text()) == ("411.237 px", "512.864 px")
        assert (other.manual_x.value(), other.manual_y.value()) == pytest.approx(typed, abs=5e-4)
    finally:
        if other is not None:
            _dispose(other)
        dialog.manual_group.setChecked(False)  # back to the fitted solution
        dialog._commit_manual_values()
        _settle(app, 0.2)
    assert (fitted.center_x_px, fitted.center_y_px, fitted.distance_mm) == before


def test_the_dragged_marker_and_the_fields_agree_with_the_axes(window) -> None:
    dialog = window.dialog
    app = _app()
    fitted = dialog.result.selected_candidate
    shown = (fitted.center_x_px + 0.5, fitted.center_y_px + 0.5)
    try:
        dialog.manual_refine_button.click()
        dialog.fit_preview_to_image()
        dialog.canvas.draw()
        assert (dialog.manual_x.value(), dialog.manual_y.value()) == pytest.approx(shown, abs=5e-4)

        def mouse(kind: str, x: float, y: float) -> None:
            dialog.canvas.callbacks.process(kind, MouseEvent(kind, dialog.canvas, x, y, button=1))

        marker = dialog.axes.transData.transform(shown)  # the marker is where the fields say
        target = dialog.axes.transData.transform((shown[0] + 40.0, shown[1] + 30.0))
        mouse("button_press_event", *marker)
        mouse("motion_notify_event", *target)
        mouse("button_release_event", *target)
        # The fields take the position under the mouse as the axes read it (3 decimals).
        under_mouse = dialog.axes.transData.inverted().transform(target)
        assert (dialog.manual_x.value(), dialog.manual_y.value()) == pytest.approx(tuple(under_mouse), abs=1e-3)
        assert dialog.manual_x.value() == pytest.approx(shown[0] + 40.0, abs=1e-3)
        _settle(app, 0.2)
        vertical, horizontal = dialog.axes.lines
        assert vertical.get_xdata()[0] == pytest.approx(dialog.manual_x.value())
        assert horizontal.get_ydata()[0] == pytest.approx(dialog.manual_y.value())
        # The preview candidate holds the fit's pixels: half a pixel lower.
        preview = dialog._display_candidate()
        assert (preview.center_x_px, preview.center_y_px) == pytest.approx(
            (dialog.manual_x.value() - 0.5, dialog.manual_y.value() - 0.5)
        )
    finally:
        dialog.manual_group.setChecked(False)
        dialog._commit_manual_values()
        _settle(app, 0.2)
    assert (fitted.center_x_px + 0.5, fitted.center_y_px + 0.5) == shown


def test_manual_mode_without_an_edit_keeps_the_fit_to_the_last_digit(window) -> None:
    dialog = window.dialog
    fitted = dialog.result.selected_candidate
    before = (fitted.center_x_px, fitted.center_y_px, fitted.distance_mm, list(fitted.warnings))
    try:
        dialog.manual_group.setChecked(True)
        dialog._commit_manual_values()  # as Apply and Export do
        assert (fitted.center_x_px, fitted.center_y_px, fitted.distance_mm, list(fitted.warnings)) == before
    finally:
        dialog.manual_group.setChecked(False)
        _settle(_app(), 0.1)


# -- a preview that keeps every second pixel ---------------------------------------------------------


def test_a_reduced_preview_draws_each_kept_pixel_where_it_is() -> None:
    app = _app()
    rows, columns = 1600, 1400  # more pixels than the preview draws: every second one is kept
    data = np.ones((rows, columns), dtype=np.float32)
    row, column = 400, 600
    data[row, column] = 5000.0
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.resize(1180, 760)
    dialog.show()
    try:
        dialog._image_loaded(DetectorImage(data, np.zeros(data.shape, dtype=bool), Path("large_detector.cbf")))
        _settle(app, 0.3)
        dialog.fit_preview_to_image()
        dialog.canvas.draw()
        shown = dialog.axes.images[0].get_array()
        assert shown.shape == (rows // 2, columns // 2)
        assert dialog.axes.get_xlim() == (0.0, float(columns))
        assert dialog.axes.get_ylim() == (float(rows), 0.0)
        bright = float(np.log1p(5000.0))
        # The block of the kept pixel is centred on that pixel's centre (column + 0.5, row + 0.5).
        assert _value_under(dialog, column + 0.5, row + 0.5) == pytest.approx(bright, rel=1e-6)
        for dx in (-0.4, 1.4):
            assert _value_under(dialog, column + dx, row + 0.5) == pytest.approx(bright, rel=1e-6)
        for dx in (-0.6, 1.6):
            assert _value_under(dialog, column + dx, row + 0.5) != pytest.approx(bright, rel=1e-6)
        for dy in (-0.6, 1.6):
            assert _value_under(dialog, column + 0.5, row + dy) != pytest.approx(bright, rel=1e-6)
    finally:
        _dispose(dialog)


def test_the_cursor_readout_counts_pixels_as_the_fields_do() -> None:
    app = _app()
    data = np.arange(40 * 50, dtype=np.float32).reshape(40, 50)  # every pixel has a value of its own
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.resize(1180, 760)
    dialog.show()
    try:
        dialog._image_loaded(DetectorImage(data, np.zeros(data.shape, dtype=bool), Path("small_detector.cbf")))
        _settle(app, 0.3)
        dialog.fit_preview_to_image()
        dialog.canvas.draw()
        # The mouse over the centre of the pixel in row 7, column 12.
        screen = dialog.axes.transData.transform((12.5, 7.5))
        dialog.canvas.callbacks.process(
            "motion_notify_event", MouseEvent("motion_notify_event", dialog.canvas, *screen)
        )
        readout = dialog.preview_cursor_label.text()
        assert readout.startswith("(x, y) = (12.5, 7.5)")
        assert readout.endswith(f"[{np.log1p(data[7, 12]):.2f}]")  # the preview shows log intensity
    finally:
        _dispose(dialog)


def test_extent_and_limits_of_the_preview() -> None:
    assert pixel_coordinates.image_extent((1043, 981)) == (0.0, 981.0, 1043.0, 0.0)
    assert pixel_coordinates.image_limits(981, 1043) == ((0.0, 981.0), (1043.0, 0.0))
    # Every third pixel of 10: the kept pixels 0, 3, 6, 9 are drawn 3 px wide around their centres.
    left, right, bottom, top = pixel_coordinates.image_extent((4, 4), 3)
    assert (left, top) == (-1.0, -1.0) and (right, bottom) == (11.0, 11.0)
    centres = [left + (index + 0.5) * 3 for index in range(4)]
    assert centres == [0.5, 3.5, 6.5, 9.5]


# -- one way to the screen -------------------------------------------------------------------------


def test_shown_and_fitted_centres_are_the_shared_conventions() -> None:
    from src.gimap.shared.geometry.conventions import canonical_from_index_center, index_center_from_canonical

    candidate = SimpleNamespace(center_x_px=182.25, center_y_px=264.75)
    assert pixel_coordinates.shown_center(candidate) == canonical_from_index_center(182.25, 264.75) == (182.75, 265.25)
    assert pixel_coordinates.fitted_center(182.75, 265.25) == index_center_from_canonical(182.75, 265.25)
    assert pixel_coordinates.fitted_center(*pixel_coordinates.shown_center(candidate)) == (182.25, 264.75)


def test_the_window_reads_a_fitted_centre_only_through_pixel_coordinates() -> None:
    """A centre written to the screen straight from ``center_x_px`` would be half a pixel off again."""
    bookkeeping = {"_remember_fitted_geometry", "_restore_fitted_geometry"}  # copies, never shown
    found = []
    for path in sorted(PRESENTATION.rglob("*.py")):
        if path.name == "pixel_coordinates.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) or function.name in bookkeeping:
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Attribute) and node.attr in ("center_x_px", "center_y_px"):
                    found.append(f"{path.name}:{node.lineno} {function.name}")
    assert found == []


# -- the window says which pixels ----------------------------------------------------------------------


def test_the_centre_fields_say_which_pixels_they_are(monkeypatch) -> None:
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    try:
        noted = (
            dialog.resultCenterXTitle, dialog.result_center_x, dialog.resultCenterYTitle, dialog.result_center_y,
            dialog.manualXLabel, dialog.manual_x, dialog.manualYLabel, dialog.manual_y,
        )
        assert {widget.toolTip() for widget in noted} == {PIXEL_NOTE}
        monkeypatch.setitem(i18n.ZH, PIXEL_NOTE, PIXEL_NOTE_ZH)
        monkeypatch.setitem(i18n._TO_ENGLISH, PIXEL_NOTE_ZH, PIXEL_NOTE)
        i18n.apply_language("zh", [dialog])
        _settle(app, 0.05)
        assert {widget.toolTip() for widget in noted} == {PIXEL_NOTE_ZH}
        i18n.apply_language("en", [dialog])
        assert {widget.toolTip() for widget in noted} == {PIXEL_NOTE}
    finally:
        i18n.apply_language("en")
        _dispose(dialog)


def test_the_chinese_table_has_the_new_texts_once_they_are_merged() -> None:
    """After the integrator's merge both new texts of this round are in the Chinese table (none before it)."""
    from src.gimap.features.calibration.domain.models import UNKNOWN_CENTER_CONVENTION

    new_texts = (PIXEL_NOTE, UNKNOWN_CENTER_CONVENTION)
    missing = [text for text in new_texts if text not in i18n.ZH]
    if len(missing) == len(new_texts):
        pytest.skip("the entries of this round are not merged into i18n/zh_tools.py yet")
    assert missing == []


def test_export_from_the_window_and_import_show_the_same_numbers(window, tmp_path, monkeypatch) -> None:
    app = _app()
    dialog = window.dialog
    shown = (
        dialog.result_center_x.text(), dialog.result_center_y.text(), dialog.result_distance.text(),
        [dialog.candidate_table.item(row, 2).text() for row in range(dialog.candidate_table.rowCount())],
    )
    target = tmp_path / "agbh.gimap-calibration.json"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(target), ""))
    dialog.export_result()
    assert target.is_file()

    profiles = InMemoryInstrumentProfileRepository()
    other = GeometryCalibrationDialog(app_context=_context(profiles))
    other.resize(1180, 760)
    other.show()
    try:
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(target), ""))
        monkeypatch.setattr(QMessageBox, "warning", lambda *_a, **_k: pytest.fail("the import failed"))
        other.import_result()
        _settle(app, 0.3)
        assert (
            other.result_center_x.text(), other.result_center_y.text(), other.result_distance.text(),
            [other.candidate_table.item(row, 2).text() for row in range(other.candidate_table.rowCount())],
        ) == shown
        assert (other.manual_x.value(), other.manual_y.value()) == (dialog.manual_x.value(), dialog.manual_y.value())
        # Applied from the imported file: the profile the first window saves.
        monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.Yes)
        imported = _apply(other).geometry
        original = dialog.result.selected_candidate
        assert (imported.beam_center_x_px, imported.beam_center_y_px, imported.distance_m) == (
            original.center_x_px + 0.5, original.center_y_px + 0.5, float(original.distance_mm) * 1e-3,
        )
    finally:
        _dispose(other)
