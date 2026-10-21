"""The XRR Series Extractor shows and accepts positions in the pixels of Analyze.

The extraction keeps pixel-centre indices (the centre of the first pixel is 0.0): its geometry, its points,
the settings Geometry Calibration's Apply writes, the exported CSV and its JSON record hold those numbers. The
window, like Analyze and the Geometry Calibration window, uses canonical detector pixels (pixel corners on
integers, the centre of the first pixel is 0.5). These tests type, pick and load positions in the window and
follow them into the extraction, the preview, the table and the exported files. One of them calibrates the
public pyFAI AgBh frame in the Geometry Calibration window (a few seconds) and compares the three windows.
"""

from __future__ import annotations

import ast
import csv
import json
import math
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
from src.gimap.app.presentation.theme import apply_theme
from src.gimap.features.xrr.application import (
    ExportXrrCurve,
    ExtractXrrSeries,
    LoadLastCalibrationGeometry,
    XrrDetectorFrame,
    XrrExtractionProgress,
    XrrExtractionResult,
    XrrFrameRef,
    XrrPoint,
    XrrSeriesInspection,
)
from src.gimap.features.xrr.application.export_record import CENTER_CONVENTION_KEY, INDEX_CENTER_CONVENTION
from src.gimap.features.xrr.domain import extract_circular_roi
from src.gimap.features.xrr.infrastructure import (
    LocalXrrCurveExportAdapter,
    LocalXrrRecordAdapter,
    SettingsXrrGeometryAdapter,
)
from src.gimap.features.xrr.infrastructure.serialization import decode_preview, encode_preview
from src.gimap.features.xrr.presentation import pixel_coordinates
from src.gimap.features.xrr.presentation.dialog import XrrSeriesDialog
from src.gimap.features.xrr.presentation.view_model import XrrViewModel
from src.gimap.integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)
from src.gimap.shared.geometry.conventions import canonical_from_index_center, index_center_from_canonical

ROOT = Path(__file__).resolve().parents[1]
AGBH = ROOT / "tests/data/external/saxs_agbh_pyfai/Pilatus1M.edf"
PRESENTATION = ROOT / "src/gimap/features/xrr/presentation"
PIXEL_NOTE = (
    "Direct-beam position in detector pixels, the same as in Analyze: pixel j spans [j, j+1] "
    "(the centre of the first pixel is 0.5), x from the left edge and y from the top edge."
)
ROI_PIXEL_NOTE = (
    "Centre of the ROI in detector pixels, the same as in Analyze: pixel j spans [j, j+1] (the centre of the "
    "first pixel is 0.5). The exported CSV counts pixel centres from 0: its roi_center values are 0.5 px lower."
)
ROI_PIXEL_NOTE_ZH = (
    "ROI 中心在探测器上的像素位置，与分析页相同：第 j 个像素覆盖 [j, j+1]（第一个像素的中心是 0.5）。"
    "导出的 CSV 按像素中心从 0 数：其中的 roi_center 值小 0.5 px。"
)
FITTED = (182.47412173380593, 264.3439878683439)  # the fit of the public AgBh frame, as indices
ROI_COLUMNS = (4, 5)  # "ROI x" and "ROI y" of the results table


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _settle(app, seconds: float = 0.1) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _dispose(dialog) -> None:
    """Close the window and delete it now, by the event loop (not inside a garbage collection)."""
    for name in ("_load_thread", "_cal_thread", "_inspect_thread", "_extract_thread"):
        thread = getattr(dialog, name, None)
        if thread is not None:
            thread.wait(10000)
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


class _Frames:
    """A series held in memory: ``frames[k]`` is frame k."""

    def __init__(self, *frames):
        self.frames = frames

    def discover(self, _spec):
        return tuple(XrrFrameRef(Path(f"frame_{index:05d}.cbf"), 0, index) for index in range(len(self.frames)))

    def load_frame(self, ref):
        return XrrDetectorFrame(np.asarray(self.frames[ref.sequence_index], dtype=np.float32), None, {})


class _Direct:
    """The extraction in this thread, without the worker process."""

    def __init__(self, repository):
        self.workflow = ExtractXrrSeries(repository)

    def execute(self, request, *, on_progress=None):
        return self.workflow.execute(request, on_progress=on_progress)

    def cancel(self):
        return True


class _Fixed:
    def __init__(self, value):
        self.value = value

    def last_calibration(self):
        return self.value


class _Unused:
    def execute(self, *_args, **_kwargs):
        raise AssertionError("not used here")

    def cancel(self):
        return True


def _view_model(*, calibration=None, frames=None) -> XrrViewModel:
    return XrrViewModel(
        inspect_series=_Unused(),
        run_extraction=_Direct(_Frames(*frames)) if frames is not None else _Unused(),
        export_curve=ExportXrrCurve(LocalXrrCurveExportAdapter(), LocalXrrRecordAdapter()),
        last_calibration=LoadLastCalibrationGeometry(_Fixed(calibration)),
    )


def _inspection(shape=(101, 81), metadata=None, data=None) -> XrrSeriesInspection:
    frame = np.zeros(shape, dtype=np.float32) if data is None else np.asarray(data, dtype=np.float32)
    return XrrSeriesInspection(
        frame_count=1,
        first_ref=XrrFrameRef(path=Path("series/f_00001.cbf"), frame_index=0, sequence_index=0),
        first_frame=XrrDetectorFrame(frame, None, dict(metadata or {})),
    )


def _numbered(rows: int = 40, columns: int = 50) -> np.ndarray:
    """A frame in which every pixel has a value of its own: 1 + row * columns + column."""
    return np.arange(1, rows * columns + 1, dtype=np.float32).reshape(rows, columns)


def _extract(dialog, tmp_path: Path) -> XrrExtractionResult:
    """Run the extraction of the window's request in this thread, as the worker would report it."""
    dialog.source_picker.set_path(str(tmp_path))
    result = dialog.view_model.extract(dialog._request(), on_progress=dialog._on_extraction_progress)
    dialog._on_extraction_finished(result)
    return result


def _value_under(dialog, x: float, y: float) -> float:
    """What the preview shows at the position (x, y) of its axes (the exact position, not a screen pixel)."""
    axis = dialog.plotter.live_axis
    screen_x, screen_y = axis.transData.transform((x, y))
    return float(axis.images[0].get_cursor_data(SimpleNamespace(x=screen_x, y=screen_y)))


def _roi_pixels(shape, center_x: float, center_y: float, radius: int) -> set:
    """The pixels (row, column) the extraction reads for an index centre, found one by one."""
    left, top, right, bottom = extract_circular_roi(
        np.ones(shape, dtype=np.float32), None, center_x, center_y, radius, "sum"
    ).bounds
    probe = np.zeros(shape, dtype=np.float32)
    found = set()
    for row in range(top, bottom):
        for column in range(left, right):
            probe[row, column] = 1.0
            if extract_circular_roi(probe, None, center_x, center_y, radius, "sum").intensity == 1.0:
                found.add((row, column))
            probe[row, column] = 0.0
    return found


@pytest.fixture
def window(tmp_path):
    """The window with a 50 × 40 px frame of numbered pixels loaded, and that frame as its series."""
    app = _app()
    frame = _numbered()
    dialog = XrrSeriesDialog(view_model=_view_model(frames=[frame]), app_context=object())
    dialog.resize(1280, 820)
    dialog.show()
    dialog._on_inspected(_inspection(data=frame))
    _settle(app)
    yield SimpleNamespace(dialog=dialog, frame=frame, folder=tmp_path)
    _dispose(dialog)


# -- one way to the screen and back ------------------------------------------------------------------------


def test_shown_and_extraction_positions_are_the_shared_conventions() -> None:
    assert pixel_coordinates.shown_coordinate(182.25) == canonical_from_index_center(182.25, 0.0)[0] == 182.75
    point = SimpleNamespace(roi_center_x_px=182.25, roi_center_y_px=264.75)
    assert pixel_coordinates.shown_roi_center(point) == canonical_from_index_center(182.25, 264.75) == (182.75, 265.25)
    assert pixel_coordinates.extraction_coordinate(182.75, 2) == index_center_from_canonical(182.75, 0.0)[0] == 182.25
    # The middle of a frame: between the two middle pixels of an even side, the centre of the middle pixel
    # of an odd one.
    assert pixel_coordinates.shown_image_center((1043, 981)) == (490.5, 521.5)
    assert pixel_coordinates.shown_image_center((100, 80)) == (40.0, 50.0)


def test_a_centre_of_two_decimals_reaches_the_extraction_as_the_float_of_its_index() -> None:
    """What the former window, which took indices, gave when the index was typed: to the last bit."""
    for hundredths in range(-30_000, 450_001):  # every index from -300.00 to 4500.00
        index = float(f"{hundredths / 100.0:.2f}")
        shown = float(f"{hundredths / 100.0 + 0.5:.2f}")  # what the field holds for that point
        assert pixel_coordinates.extraction_coordinate(shown, 2) == index, index
    # Just below a power of two a plain subtraction is one unit in the last place off.
    assert float("64.49") - 0.5 != float("63.99")
    assert pixel_coordinates.extraction_coordinate(float("64.49"), 2) == float("63.99")


def test_extent_limits_and_step_of_the_preview() -> None:
    assert pixel_coordinates.image_extent((1043, 981), (1043, 981)) == (0.0, 981.0, 1043.0, 0.0)
    assert pixel_coordinates.image_limits((1043, 981)) == ((0.0, 981.0), (1043.0, 0.0))
    # Every third pixel of 10: the kept pixels 0, 3, 6, 9 are drawn 3 px wide around their centres.
    left, right, bottom, top = pixel_coordinates.image_extent((4, 4), (10, 10))
    assert (left, top) == (-1.0, -1.0) and (right, bottom) == (11.0, 11.0)
    assert [left + (index + 0.5) * 3 for index in range(4)] == [0.5, 3.5, 6.5, 9.5]


@pytest.mark.parametrize(
    "shape",
    [(1043, 981), (981, 1043), (1679, 1475), (4727, 3142), (2167, 2070), (360, 360), (361, 200), (100, 100),
     (721, 3), (1081, 1080), (3001, 3002), (4, 7)],
)
def test_the_step_of_the_preview_is_the_one_the_worker_used(shape) -> None:
    """The window derives the step from the two shapes the worker sends; it is the step of ``encode_preview``."""
    preview, full_shape = decode_preview(encode_preview(np.zeros(shape, dtype=np.float32)))
    used = max(1, math.ceil(max(shape) / 360))
    assert preview.shape == np.zeros(shape)[::used, ::used].shape and full_shape == shape
    assert pixel_coordinates.preview_stride(preview.shape, full_shape) == used


def test_the_window_reaches_the_extraction_positions_only_through_pixel_coordinates() -> None:
    """A position written to the screen straight from ``roi_center_x_px`` would be half a pixel off again."""
    positions = {"roi_center_x_px", "roi_center_y_px", "beam_center_x_px", "beam_center_y_px"}
    read, converting = [], []
    for path in sorted(PRESENTATION.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.endswith("geometry.conventions"):
                converting.append(path.name)
            if isinstance(node, ast.Attribute) and node.attr in positions and path.name != "pixel_coordinates.py":
                read.append(f"{path.name}:{node.lineno}")
    assert read == []
    assert converting == ["pixel_coordinates.py"]


# -- the centre of the last calibration ---------------------------------------------------------------------


def _applied_calibration(settings, fitted=FITTED):
    """Apply a calibration result as the Geometry Calibration window does; return what Analyze gets."""
    from src.gimap.features.calibration.domain import CalibrationCandidate, CalibrationResult
    from src.gimap.features.calibration.domain.instrument_profile import calibrated_geometry
    from src.gimap.features.calibration.infrastructure.adapters import SettingsGeometryAdapter

    candidate = CalibrationCandidate("agbh", fitted[0], fitted[1], 1575.478, matched_ring_count=4)
    result = CalibrationResult(
        "Pilatus1M.edf", 10, 20, "abc", 12.0, 12.398419843320026 / 12.0, "Pilatus 1M", 172e-6, 172e-6,
        candidate, [candidate], "2026-10-13T08:00:00+00:00", metadata={"image_shape": [1043, 981]},
    )
    SettingsGeometryAdapter(settings).apply(result)
    return calibrated_geometry(result)


def test_the_centre_of_the_last_calibration_is_the_centre_analyze_gets() -> None:
    _app()
    settings = InMemorySettingsRepository({})
    profile_geometry = _applied_calibration(settings)
    # The settings keep the fit's own numbers; the instrument profile of Analyze is half a pixel higher.
    stored = (settings.get("detector", "beam_center_x"), settings.get("detector", "beam_center_y"))
    assert stored == FITTED
    assert (profile_geometry.beam_center_x_px, profile_geometry.beam_center_y_px) == (FITTED[0] + 0.5, FITTED[1] + 0.5)

    view_model = _view_model(calibration=SettingsXrrGeometryAdapter(settings).last_calibration())
    dialog = XrrSeriesDialog(view_model=view_model, app_context=object())
    try:
        assert (dialog.center_x_spin.text(), dialog.center_y_spin.text()) == (
            f"{profile_geometry.beam_center_x_px:.2f}", f"{profile_geometry.beam_center_y_px:.2f}"
        ) == ("182.97", "264.84")
        assert dialog.geometry_sources()["center_x"] == dialog.geometry_sources()["center_y"] == "calibration"
        # The extraction gets the stored index, with the two decimals of the field (as it always did).
        geometry = dialog._geometry()
        assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == (182.47, 264.34)
        # A frame of the calibrated size keeps that centre; nothing was typed.
        dialog._on_inspected(_inspection(shape=(1043, 981)))
        assert (dialog.center_x_spin.text(), dialog.center_y_spin.text()) == ("182.97", "264.84")
        assert dialog._geometry() == geometry
    finally:
        _dispose(dialog)
    # Opening the window and loading a frame wrote nothing.
    assert (settings.get("detector", "beam_center_x"), settings.get("detector", "beam_center_y")) == FITTED


@pytest.mark.parametrize(
    "fitted",
    [(0.015, 0.045), (255.505, 255.995), (1023.505, 2047.755), (182.125, 264.375), (0.0, 737.0)],
)
def test_a_stored_centre_between_two_steps_of_the_field_reads_as_analyze_reads_it(fitted) -> None:
    """A stored index can lie exactly between two values of a two-decimal field (x.xx5).

    The XRR field rounds the number Analyze holds (index + 0.5), which Analyze's Beam centre ▸ Enter Coordinates
    shows to four decimals, as a two-decimal Qt field rounds it, and the extraction gets what the field shows,
    half a pixel lower. (The former window rounded the index itself: on such a tie, and only there, its last
    digit could differ, 0.01 px.)
    """
    from PyQt5.QtWidgets import QDoubleSpinBox

    from src.gimap.features.analyze.presentation.bindings.beam_center import CenterDialog

    _app()
    settings = InMemorySettingsRepository({})
    profile_geometry = _applied_calibration(settings, fitted)
    analyze = CenterDialog((profile_geometry.beam_center_x_px, profile_geometry.beam_center_y_px))
    view_model = _view_model(calibration=SettingsXrrGeometryAdapter(settings).last_calibration())
    dialog = XrrSeriesDialog(view_model=view_model, app_context=object())
    try:
        # Analyze holds and shows the number itself (four decimals are enough for these).
        held = (profile_geometry.beam_center_x_px, profile_geometry.beam_center_y_px)
        assert analyze.center() == held
        assert (analyze.x_spin.text(), analyze.y_spin.text()) == (f"{held[0]:.4f}", f"{held[1]:.4f}")
        # The XRR field shows that number as a field of two decimals reads it, digit for digit.
        two_decimals = QDoubleSpinBox()
        two_decimals.setDecimals(2)
        two_decimals.setRange(-1e6, 1e6)
        expected = []
        for value in analyze.center():
            two_decimals.setValue(value)
            expected.append((two_decimals.text(), two_decimals.value()))
        two_decimals.deleteLater()
        assert (dialog.center_x_spin.text(), dialog.center_y_spin.text()) == (expected[0][0], expected[1][0])
        assert (dialog.center_x_spin.value(), dialog.center_y_spin.value()) == (expected[0][1], expected[1][1])
        geometry = dialog._geometry()
        assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == (
            round(dialog.center_x_spin.value() - 0.5, 2), round(dialog.center_y_spin.value() - 0.5, 2)
        )
        # Never further from the stored point than the half step a field of two decimals allows.
        assert abs(geometry.beam_center_x_px - fitted[0]) <= 0.005 + 1e-9
        assert abs(geometry.beam_center_y_px - fitted[1]) <= 0.005 + 1e-9
    finally:
        analyze.deleteLater()
        _dispose(dialog)


# -- a centre typed by hand ---------------------------------------------------------------------------------


def _type(spin, text: str) -> None:
    """As a person types: the text into the field, then Enter (the fields act when the edit is finished)."""
    spin.lineEdit().setText(text)
    spin.interpretText()


def test_a_centre_typed_into_the_fields_reaches_the_extraction_half_a_pixel_lower() -> None:
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    try:
        for typed_x, typed_y, index in (
            ("182.97", "264.84", (182.47, 264.34)),  # copied from Analyze or Geometry Calibration: as it is
            ("0.50", "0.5", (0.0, 0.0)),  # the centre of the first pixel
            ("64.49", "4096", (63.99, 4095.5)),
            ("-12.25", "100000", (-12.75, 99999.5)),
        ):
            _type(dialog.center_x_spin, typed_x)
            _type(dialog.center_y_spin, typed_y)
            geometry = dialog._geometry()
            assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == index, (typed_x, typed_y)
            assert dialog.center_x_spin.text() == typed_x  # the field keeps what was typed
        assert dialog.geometry_sources()["center_x"] == dialog.geometry_sources()["center_y"] == "typed"
    finally:
        _dispose(dialog)



@pytest.mark.parametrize(
    "inside",
    [(0.5, 0.5), (0.05, 0.05), (0.95, 0.95), (0.05, 0.95)],  # the middle of the pixel and near its corners
)
def test_a_typed_centre_names_the_pixel_it_lies_in(window, inside) -> None:
    dialog, frame = window.dialog, window.frame
    row, column = 7, 12
    dialog.center_x_spin.setValue(column + inside[0])
    dialog.center_y_spin.setValue(row + inside[1])
    dialog.theta_start_spin.setValue(0.0)  # the first frame: the ROI is on the direct beam
    dialog.radius_spin.setValue(0)
    assert dialog.geometry_sources()["center_x"] == "typed"
    (point,) = _extract(dialog, window.folder).points
    assert point.intensity == frame[row, column] and point.valid_pixels == 1
    # The cross of the preview lies in that pixel too, where the fields say.
    cross = dialog.plotter.live_axis.lines[0]
    assert (cross.get_xdata()[0], cross.get_ydata()[0]) == (dialog.center_x_spin.value(), dialog.center_y_spin.value())
    assert (int(cross.get_xdata()[0]), int(cross.get_ydata()[0])) == (column, row)


@pytest.mark.parametrize(
    "index",
    [(737.0, 737.0), (182.47, 264.34), (0.0, 0.0), (-0.5, -0.5), (490.0, 521.0), (255.7, 511.99),
     (127.63, 63.99), (1023.75, 2047.61), (-12.34, 4095.5), (99999.25, -99999.25), (0.3, 0.7)],
)
def test_the_same_point_typed_half_a_pixel_higher_gives_the_former_geometry(index) -> None:
    """Who typed the index X before types X + 0.5 now: the extraction's geometry is the same to the last bit."""
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    try:
        dialog.center_x_spin.setValue(index[0] + 0.5)
        dialog.center_y_spin.setValue(index[1] + 0.5)
        geometry = dialog._geometry()
        assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == index
    finally:
        _dispose(dialog)


# -- the frame gives the centre -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, shown, index",
    [((101, 81), (40.5, 50.5), (40.0, 50.0)),  # odd sides: the centre of the middle pixel
     ((100, 80), (40.0, 50.0), (39.5, 49.5)),  # even sides: the corner between the four middle pixels
     ((1043, 981), (490.5, 521.5), (490.0, 521.0)),
     ((1679, 1475), (737.5, 839.5), (737.0, 839.0))],
)
def test_the_image_centre_is_the_middle_of_the_frame_as_drawn(shape, shown, index) -> None:
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    try:
        dialog._on_inspected(_inspection(shape=shape))
        assert dialog.geometry_sources()["center_x"] == dialog.geometry_sources()["center_y"] == "image center"
        assert (dialog.center_x_spin.value(), dialog.center_y_spin.value()) == shown == (shape[1] / 2, shape[0] / 2)
        geometry = dialog._geometry()
        # The point the former window used: ((columns - 1) / 2, (rows - 1) / 2) as indices.
        assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == index == ((shape[1] - 1) / 2, (shape[0] - 1) / 2)
        cross = dialog.plotter.live_axis.lines[0]
        assert (cross.get_xdata()[0], cross.get_ydata()[0]) == shown
        limits = dialog.plotter.live_axis
        assert sum(limits.get_xlim()) / 2 == shown[0] and sum(limits.get_ylim()) / 2 == shown[1]
    finally:
        _dispose(dialog)


def test_a_centre_in_the_file_is_read_as_before_and_shown_in_the_pixels_of_the_window() -> None:
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    try:
        dialog.center_y_spin.setValue(600.0)  # typed: a file never replaces it
        dialog._on_inspected(_inspection(metadata={"beam_center_x_px": 700.0, "beam_center_y_px": 500.0}))
        assert dialog.geometry_sources()["center_x"] == "file"
        assert dialog.center_x_spin.value() == 700.5 and dialog.center_y_spin.value() == 600.0
        geometry = dialog._geometry()
        assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == (700.0, 599.5)
        # The sentence gives the position of the file in the pixels of the fields: typing it gives that point.
        assert "The file says: beam center Y 500.50 px" in dialog.series_summary.text()
        dialog.center_y_spin.setValue(500.5)
        assert dialog._geometry().beam_center_y_px == 500.0
    finally:
        _dispose(dialog)


# -- Pick ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("inside", [(0.5, 0.5), (0.2, 0.2), (0.8, 0.8), (0.95, 0.05)])
def test_pick_takes_the_point_clicked_and_the_extraction_reads_the_pixel_clicked(window, inside) -> None:
    dialog, frame = window.dialog, window.frame
    row, column = 7, 12
    click = (column + inside[0], row + inside[1])
    dialog.pick_center_button.setChecked(True)
    dialog._on_detector_click(SimpleNamespace(inaxes=dialog.plotter.live_axis, xdata=click[0], ydata=click[1]))
    assert not dialog.pick_center_button.isChecked()
    assert (dialog.center_x_spin.value(), dialog.center_y_spin.value()) == pytest.approx(click)
    assert dialog.geometry_sources()["center_x"] == "typed"
    cross = dialog.plotter.live_axis.lines[0]
    assert (cross.get_xdata()[0], cross.get_ydata()[0]) == pytest.approx(click)
    dialog.radius_spin.setValue(0)
    (point,) = _extract(dialog, window.folder).points
    assert point.intensity == frame[row, column]  # the former window read the next pixel for (0.5, 0.5)


def test_a_mouse_click_on_the_preview_gives_the_position_the_axes_read(window) -> None:
    dialog = window.dialog
    canvas, axis = dialog.plotter.live_canvas, dialog.plotter.live_axis
    canvas.draw()
    target = axis.transData.transform((30.5, 20.5))
    under_mouse = axis.transData.inverted().transform(target)
    dialog.pick_center_button.setChecked(True)
    canvas.callbacks.process("button_press_event", MouseEvent("button_press_event", canvas, *target, button=1))
    assert (dialog.center_x_spin.value(), dialog.center_y_spin.value()) == pytest.approx(tuple(under_mouse), abs=5e-3)
    assert dialog.center_x_spin.value() == pytest.approx(30.5, abs=5e-3)
    geometry = dialog._geometry()
    assert (math.floor(geometry.beam_center_x_px + 0.5), math.floor(geometry.beam_center_y_px + 0.5)) == (30, 20)


# -- the preview --------------------------------------------------------------------------------------------


def test_the_preview_is_drawn_in_the_pixels_of_the_fields(window) -> None:
    dialog, frame = window.dialog, window.frame
    rows, columns = frame.shape
    axis = dialog.plotter.live_axis
    dialog.center_x_spin.setValue(12.3)
    dialog.center_y_spin.setValue(27.6)
    dialog.theta_start_spin.setValue(0.02)  # the reflection is above the direct beam
    dialog.distance_spin.setValue(1000.0)
    dialog.radius_spin.setValue(2)
    (point,) = _extract(dialog, window.folder).points
    dialog.plotter.live_canvas.draw()

    # Pixel j spans j…j+1 and row 0 is at the top: the frame fills 0…columns, 0…rows.
    assert tuple(axis.images[0].get_extent()) == (0.0, float(columns), float(rows), 0.0)
    assert axis.get_xlim() == (0.0, float(columns)) and axis.get_ylim() == (float(rows), 0.0)
    shown = np.asarray(axis.images[0].get_array())
    for row, column in ((0, 0), (7, 12), (rows - 1, columns - 1)):
        for dx, dy in ((0.5, 0.5), (0.05, 0.05), (0.95, 0.95)):
            assert _value_under(dialog, column + dx, row + dy) == pytest.approx(float(shown[row, column]), rel=1e-6)
    assert _value_under(dialog, 12 - 0.05, 7.5) == pytest.approx(float(shown[7, 11]), rel=1e-6)

    # The cross is where the fields say; the ROI circle is where the table says.
    cross = axis.lines[0]
    assert (cross.get_xdata()[0], cross.get_ydata()[0]) == (12.3, 27.6)
    (circle,) = axis.patches
    assert tuple(circle.center) == (point.roi_center_x_px + 0.5, point.roi_center_y_px + 0.5)
    assert circle.center[0] == pytest.approx(12.3) and circle.center[1] < 27.6 - 4.0
    assert circle.radius == 2.0
    assert (dialog.results_table.item(0, 4).text(), dialog.results_table.item(0, 5).text()) == (
        f"{circle.center[0]:.2f}", f"{circle.center[1]:.2f}"
    )
    # The pixels the extraction read are those whose centres, as drawn, lie inside the circle as drawn.
    read = _roi_pixels(frame.shape, point.roi_center_x_px, point.roi_center_y_px, 2)
    inside = {
        (row, column)
        for row in range(rows)
        for column in range(columns)
        if (column + 0.5 - circle.center[0]) ** 2 + (row + 0.5 - circle.center[1]) ** 2 <= circle.radius**2
    }
    assert read == inside and len(read) == point.valid_pixels > 1
    assert point.intensity == float(sum(frame[row, column] for row, column in read))


def test_a_reduced_preview_draws_each_kept_pixel_where_it_is() -> None:
    """The worker sends every third pixel of a 981 × 1043 px frame; the ROI is computed on the whole frame."""
    app = _app()
    frame = np.ones((1043, 981), dtype=np.float32)
    row, column = 300, 450  # a pixel the reduced preview keeps
    frame[row, column] = 5000.0
    preview, full_shape = decode_preview(encode_preview(frame))
    assert preview.shape == (348, 327) and preview[row // 3, column // 3] == 5000.0
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    dialog.resize(1280, 820)
    dialog.show()
    try:
        point = XrrPoint(0, "frame_00000.cbf", 0, 0.0, 0.0, 5000.0, float(column), float(row), 1)
        dialog.view_model.state.result = XrrExtractionResult((point,))
        dialog._on_extraction_progress(XrrExtractionProgress(1, 1, point, preview=preview, preview_shape=full_shape))
        _settle(app)
        dialog.plotter.live_canvas.draw()
        axis = dialog.plotter.live_axis
        assert axis.get_xlim() == (0.0, 981.0) and axis.get_ylim() == (1043.0, 0.0)
        bright = float(np.asarray(axis.images[0].get_array()).max())
        # The block of the kept pixel is centred on that pixel's centre (column + 0.5, row + 0.5), 3 px wide.
        assert _value_under(dialog, column + 0.5, row + 0.5) == pytest.approx(bright)
        for offset in (-0.9, 1.9):
            assert _value_under(dialog, column + offset, row + 0.5) == pytest.approx(bright)
            assert _value_under(dialog, column + 0.5, row + offset) == pytest.approx(bright)
        for offset in (-1.1, 2.1):
            assert _value_under(dialog, column + offset, row + 0.5) != pytest.approx(bright)
            assert _value_under(dialog, column + 0.5, row + offset) != pytest.approx(bright)
        # The ROI of that pixel is drawn on its centre, and the table names it in the same pixels.
        (circle,) = axis.patches
        assert tuple(circle.center) == (column + 0.5, row + 0.5)
        assert (dialog.results_table.item(0, 4).text(), dialog.results_table.item(0, 5).text()) == ("450.50", "300.50")
    finally:
        _dispose(dialog)


# -- the table and the exported files -------------------------------------------------------------------------


def test_the_table_shows_the_pixels_of_the_window_and_the_files_keep_the_numbers_of_the_extraction(
    window, monkeypatch
) -> None:
    dialog, frame = window.dialog, window.frame
    dialog.center_x_spin.setValue(12.5)  # the centre of the pixel in column 12
    dialog.center_y_spin.setValue(27.5)
    dialog.theta_start_spin.setValue(0.0)
    dialog.radius_spin.setValue(1)
    dialog._run_geometry_sources = dialog.geometry_sources()
    (point,) = _extract(dialog, window.folder).points
    assert (point.roi_center_x_px, point.roi_center_y_px) == (12.0, 27.0)
    assert (dialog.results_table.item(0, 4).text(), dialog.results_table.item(0, 5).text()) == ("12.50", "27.50")
    assert point.valid_pixels == 5  # the pixel and its four neighbours
    assert point.intensity == float(frame[27, 12] + frame[26, 12] + frame[28, 12] + frame[27, 11] + frame[27, 13])
    headers = [dialog.results_table.horizontalHeaderItem(column) for column in range(7)]
    assert [header.text() for header in headers[4:6]] == ["ROI x", "ROI y"]
    assert [header.toolTip() for header in headers] == ["", "", "", "", ROI_PIXEL_NOTE, ROI_PIXEL_NOTE, ""]

    target = window.folder / "out" / "numbered_xrr.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(target), "CSV (*.csv)"))
    dialog._export_curve()
    with target.open(encoding="utf-8") as handle:
        (written,) = list(csv.DictReader(handle))
    assert (float(written["roi_center_x_px"]), float(written["roi_center_y_px"])) == (12.0, 27.0)
    record = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    geometry = record["geometry"]
    assert (geometry["beam_center_x_px"], geometry["beam_center_y_px"]) == (12.0, 27.0)
    assert geometry["sources"]["center_x"] == "typed"
    # What every earlier record said, word for word, and the convention by name.
    assert geometry["pixel_convention"] == "numpy indices of the loaded frame: x = column, y = row, row 0 at the top"
    assert record["format"] == "gimap-xrr-curve-record" and record["format_version"] == 1
    assert record[CENTER_CONVENTION_KEY] == INDEX_CENTER_CONVENTION
    assert record[CENTER_CONVENTION_KEY]["name"] == "index"
    for said in ("beam_center_x_px", "roi_center_x_px", "(0.0, 0.0)", "(0.5, 0.5)", "0.5 px higher"):
        assert said in record[CENTER_CONVENTION_KEY]["description"]
    assert "pixel centres counted from 0" in record["columns"]["roi_center_x_px"]
    assert "pixel centres counted from 0" in record["columns"]["roi_center_y_px"]


# -- the window says which pixels -----------------------------------------------------------------------------


def test_the_centre_fields_say_which_pixels_they_are() -> None:
    app = _app()
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    try:
        noted = (
            dialog.center_x_spin, dialog.center_y_spin,
            dialog.geometry_field_labels["center_x"], dialog.geometry_field_labels["center_y"],
        )
        assert [label.text() for label in noted[2:]] == ["Beam center X (px)", "Beam center Y (px)"]
        assert {widget.toolTip() for widget in noted} == {PIXEL_NOTE} == {pixel_coordinates.PIXEL_NOTE}
        # The sentence of the Geometry Calibration window, word for word: its Chinese serves both windows.
        # (Read from that window's source as Python reads it, however the sentence is broken into lines.)
        calibration_texts = ast.parse((
            ROOT / "src/gimap/features/calibration/presentation/views/geometry_calibration_translations.py"
        ).read_text(encoding="utf-8"))
        sentences = {
            node.value for node in ast.walk(calibration_texts)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert PIXEL_NOTE in sentences
        assert PIXEL_NOTE in i18n.ZH
        i18n.apply_language("zh", [dialog])
        _settle(app, 0.05)
        assert {widget.toolTip() for widget in noted} == {i18n.ZH[PIXEL_NOTE]}
        i18n.apply_language("en", [dialog])
        assert {widget.toolTip() for widget in noted} == {PIXEL_NOTE}
    finally:
        i18n.apply_language("en")
        _dispose(dialog)


def test_the_note_of_the_roi_columns_follows_the_language(monkeypatch) -> None:
    app = _app()
    assert pixel_coordinates.ROI_PIXEL_NOTE == ROI_PIXEL_NOTE
    monkeypatch.setitem(i18n.ZH, ROI_PIXEL_NOTE, ROI_PIXEL_NOTE_ZH)
    monkeypatch.setitem(i18n._TO_ENGLISH, ROI_PIXEL_NOTE_ZH, ROI_PIXEL_NOTE)
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    try:
        headers = [dialog.results_table.horizontalHeaderItem(column) for column in (4, 5)]
        assert {header.toolTip() for header in headers} == {ROI_PIXEL_NOTE}
        i18n.apply_language("zh", [dialog])
        _settle(app, 0.05)
        assert {header.toolTip() for header in headers} == {ROI_PIXEL_NOTE_ZH}
        assert [header.text() for header in headers] == ["ROI x", "ROI y"]  # names of quantities stay
        i18n.apply_language("en", [dialog])
        assert {header.toolTip() for header in headers} == {ROI_PIXEL_NOTE}
    finally:
        i18n.apply_language("en")
        _dispose(dialog)


def test_a_window_opened_while_the_interface_is_chinese_says_which_pixels_in_chinese(monkeypatch) -> None:
    """As the Tools menu opens it in a Chinese session: nothing switches the language afterwards."""
    app = _app()
    monkeypatch.setitem(i18n.ZH, ROI_PIXEL_NOTE, ROI_PIXEL_NOTE_ZH)
    monkeypatch.setitem(i18n._TO_ENGLISH, ROI_PIXEL_NOTE_ZH, ROI_PIXEL_NOTE)
    i18n.apply_language("zh")
    dialog = None
    try:
        dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
        dialog.show()
        _settle(app, 0.05)
        noted = (
            dialog.center_x_spin, dialog.center_y_spin,
            dialog.geometry_field_labels["center_x"], dialog.geometry_field_labels["center_y"],
        )
        headers = [dialog.results_table.horizontalHeaderItem(column) for column in ROI_COLUMNS]
        assert {widget.toolTip() for widget in noted} == {i18n.ZH[PIXEL_NOTE]}
        assert {header.toolTip() for header in headers} == {ROI_PIXEL_NOTE_ZH}
        i18n.apply_language("en", [dialog])
        assert {widget.toolTip() for widget in noted} == {PIXEL_NOTE}
        assert {header.toolTip() for header in headers} == {ROI_PIXEL_NOTE}
    finally:
        i18n.apply_language("en")
        if dialog is not None:
            _dispose(dialog)


def test_the_chinese_table_has_the_note_of_the_roi_columns_once_it_is_merged() -> None:
    """After the integrator's merge the new text of this round is in the Chinese table (not before it)."""
    if ROI_PIXEL_NOTE not in i18n.ZH:
        pytest.skip("the entry of this round is not merged into i18n/zh_tools.py yet")
    chinese = i18n.ZH[ROI_PIXEL_NOTE]
    for kept in ("ROI", "[j, j+1]", "0.5", "CSV", "roi_center"):
        assert kept in chinese


# -- language and theme ---------------------------------------------------------------------------------------


def test_language_and_theme_change_no_number(window) -> None:
    dialog = window.dialog
    app = _app()
    dialog.center_x_spin.setValue(12.3)
    dialog.center_y_spin.setValue(27.6)
    dialog.theta_start_spin.setValue(0.02)
    dialog.distance_spin.setValue(1000.0)
    dialog.radius_spin.setValue(2)
    _extract(dialog, window.folder)

    def numbers():
        axis = dialog.plotter.live_axis
        return (
            dialog.center_x_spin.text(), dialog.center_y_spin.text(),
            dialog.results_table.item(0, 4).text(), dialog.results_table.item(0, 5).text(),
            tuple(axis.images[0].get_extent()), axis.lines[0].get_xdata()[0], axis.lines[0].get_ydata()[0],
            tuple(axis.patches[0].center), dialog._geometry(),
        )

    before = numbers()
    assert before[:2] == ("12.30", "27.60") and before[2] == "12.30"
    try:
        for language, theme in (("zh", "dark"), ("en", "dark"), ("zh", "light"), ("en", "light")):
            i18n.apply_language(language, [dialog])
            apply_theme(theme, 9)
            _settle(app, 0.15)
            dialog._redraw_detector()
            assert numbers() == before, (language, theme)
    finally:
        i18n.apply_language("en")
        apply_theme("light", 9)


# -- the three windows on the public AgBh frame -------------------------------------------------------------


def _wait(app, condition, timeout: float) -> None:
    end = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < end, "timed out"
        _settle(app, 0.05)


def test_the_public_agbh_calibration_reads_the_same_in_calibration_analyze_and_xrr(tmp_path, monkeypatch) -> None:
    if not AGBH.is_file():
        pytest.skip("pyFAI AgBh test frame not available")
    from src.gimap.features.analyze.bootstrap import create_analyze_view_model
    from src.gimap.features.analyze.presentation.page import AnalyzePage
    from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog

    monkeypatch.setattr(QMessageBox, "question", lambda *_args, **_kwargs: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_args, **_kwargs: pytest.fail("no warning expected"))
    app = _app()
    frame = tmp_path / AGBH.name  # a copy: nothing of a test is written beside the repository's data
    shutil.copy2(AGBH, frame)
    context = AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        jobs=SimpleNamespace(run=None, cancel=lambda _job: False),  # no extraction is run here
        instrument_profiles=InMemoryInstrumentProfileRepository(),
    )
    calibration = GeometryCalibrationDialog(app_context=context)
    calibration.resize(1180, 760)
    calibration.show()
    xrr = page = None
    try:
        calibration.load_image(str(frame))
        _wait(app, lambda: calibration._load_thread is None and calibration.image is not None, 60)
        # The standard and the rough distance are known for this frame (the manual advises to give them).
        calibration.standard_combo.setCurrentIndex(calibration.standard_combo.findData("agbh"))
        calibration.estimated_distance_spin.setValue(1600.0)
        calibration.start_calibration()
        _wait(app, lambda: calibration._cal_thread is None and calibration.result is not None, 240)
        calibration.apply_result()
        _settle(app, 0.2)
        fitted = calibration.result.selected_candidate
        profile = calibration.view_model.last_profile.geometry
        saved = (profile.beam_center_x_px, profile.beam_center_y_px)
        assert saved == (fitted.center_x_px + 0.5, fitted.center_y_px + 0.5)
        assert saved == pytest.approx((182.97, 264.84), abs=0.05)  # the documented direct beam is (180.0, 263.9)
        stored = (context.settings.get("detector", "beam_center_x"), context.settings.get("detector", "beam_center_y"))
        assert stored == (fitted.center_x_px, fitted.center_y_px)

        # The XRR window as the Tools menu opens it: its view model from the application context.
        xrr = XrrSeriesDialog(app_context=context)
        xrr.show()
        _settle(app, 0.1)
        shown = (xrr.center_x_spin.text(), xrr.center_y_spin.text())
        assert shown == (f"{saved[0]:.2f}", f"{saved[1]:.2f}")
        # Geometry Calibration shows three decimals of the same numbers.
        assert (calibration.result_center_x.text(), calibration.result_center_y.text()) == (
            f"{saved[0]:.3f} px", f"{saved[1]:.3f} px"
        )
        assert tuple(f"{float(text.split()[0]):.2f}" for text in (
            calibration.result_center_x.text(), calibration.result_center_y.text()
        )) == shown
        assert xrr.center_x_spin.toolTip() == calibration.manual_x.toolTip() == PIXEL_NOTE
        geometry = xrr._geometry()
        assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == (round(stored[0], 2), round(stored[1], 2))
        assert geometry.distance_m == pytest.approx(fitted.distance_mm * 1e-3, abs=1e-6)  # the field: 0.001 mm

        # Analyze, for the same frame: the centre of the profile, which is what the XRR fields show.
        page = AnalyzePage(create_analyze_view_model(context))
        page.add_paths([frame])
        assert page.tasks.wait(120)
        analysis = page.view_model.state.analysis.geometry
        assert (analysis.beam_center_x_px, analysis.beam_center_y_px) == saved
        assert page.center_button.text() == f"Beam {saved[0]:.4f}, {saved[1]:.4f} px · profile"
        assert (f"{analysis.beam_center_x_px:.2f}", f"{analysis.beam_center_y_px:.2f}") == shown
        # The settings other tools read still hold the calibration's own numbers.
        assert (
            context.settings.get("detector", "beam_center_x"), context.settings.get("detector", "beam_center_y")
        ) == stored
    finally:
        if page is not None:
            page.dispose()
            page.close()
        if xrr is not None:
            _dispose(xrr)
        _dispose(calibration)
