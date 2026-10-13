"""Geometry Calibration in the pixels of Analyze, where half a pixel is easiest to lose.

The first and the last pixel of the frame, a beam centre outside the detector, a value typed on an edge, the
profile as the next start of the program reads it from the user folder, a switch of language and theme, and
what a calibration file may state about its beam centres. Every result here is made up (no calibration run):
the fit keeps pixel-centre indices (the centre of the first pixel is 0.0), the window shows and takes
canonical pixels (the centre of the first pixel is 0.5), and the saved profile is the same point.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

from src.gimap.app import AppContext
from src.gimap.app.presentation import i18n
from src.gimap.app.presentation.components import visible_toasts
from src.gimap.app.presentation.theme import apply_theme
from src.gimap.features.calibration.domain import (
    CalibrationCandidate,
    CalibrationResult,
    DetectorImage,
    MatchedRing,
)
from src.gimap.features.calibration.infrastructure.adapters import JsonCalibrationStorageAdapter
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
    JsonInstrumentProfileRepository,
)

ROWS, COLUMNS = 1043, 981
UNKNOWN = "This calibration file states a beam-centre convention this version does not know."


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _settle(app, seconds: float = 0.15) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _context(profiles=None) -> AppContext:
    return AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        instrument_profiles=profiles if profiles is not None else InMemoryInstrumentProfileRepository(),
    )


def _candidate(x: float, y: float, distance: float = 1500.0) -> CalibrationCandidate:
    return CalibrationCandidate(
        "agbh", x, y, distance, matched_ring_count=2, rms_residual_px=0.8, confidence="High",
        matched_rings=[MatchedRing(0, 0.1076, 150.2, 150.0, 0.2)], detected_peak_radii_px=[150.2, 300.3],
    )


def _result(source: Path, x: float, y: float) -> CalibrationResult:
    selected = _candidate(x, y)
    return CalibrationResult(
        str(source), 10, 20, "abc", 12.0, 12.398419843320026 / 12.0, "Pilatus 1M", 172e-6, 172e-6,
        selected, [selected], "2026-10-13T08:00:00+00:00", metadata={"image_shape": [ROWS, COLUMNS]},
    )


def _former_payload(result: CalibrationResult) -> dict:
    """The record as the first release wrote it: no word about the convention of its centres."""
    payload = asdict(result)
    payload["format"] = "gimap-geometry-calibration"
    payload["format_version"] = 1
    return payload


def _show(dialog, result: CalibrationResult) -> None:
    """Put a result into the window as a finished calibration does."""
    dialog.result = result
    dialog._remember_fitted_geometry()
    dialog._populate_candidates()
    dialog.manual_group.setChecked(False)
    dialog._show_candidate(result.selected_candidate)
    _settle(_app())


def _dispose(dialog) -> None:
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _apply(dialog):
    for toast in visible_toasts(dialog.right):
        toast.close()
    dialog.apply_result()
    _settle(_app(), 0.1)
    return dialog.view_model.last_profile


@pytest.fixture
def window(tmp_path, monkeypatch):
    """The window with a frame of ones (981 × 1043 px) and nothing calibrated yet."""
    monkeypatch.setattr(QMessageBox, "question", lambda *_a, **_k: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *_a, **_k: pytest.fail("no warning expected"))
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.resize(1180, 760)
    dialog.show()
    data = np.ones((ROWS, COLUMNS), dtype=np.float32)
    dialog._image_loaded(DetectorImage(data, np.zeros(data.shape, dtype=bool), tmp_path / "frame.cbf"))
    _settle(app, 0.2)
    yield dialog
    _dispose(dialog)


# -- where the fit puts the centre ------------------------------------------------------------------


@pytest.mark.parametrize(
    "fitted",
    [
        (0.0, 0.0),  # the centre of the first pixel
        (-0.5, -0.5),  # the corner of the frame
        (COLUMNS - 1.0, ROWS - 1.0),  # the centre of the last pixel
        (-250.3, 400.0),  # left of the detector (a GIWAXS frame with the direct beam off the module)
        (412.0, -37.25),  # above it
        (99999.25, -99999.25),  # as far as the fields go
    ],
)
def test_a_fitted_centre_anywhere_is_shown_and_saved_as_the_same_point(window, tmp_path, fitted) -> None:
    dialog = window
    shown = (fitted[0] + 0.5, fitted[1] + 0.5)
    _show(dialog, _result(tmp_path / "frame.cbf", *fitted))

    assert dialog.result_center_x.text() == f"{shown[0]:.3f} px"
    assert dialog.result_center_y.text() == f"{shown[1]:.3f} px"
    assert dialog.candidate_table.item(0, 2).text() == f"{shown[0]:.1f}, {shown[1]:.1f}"
    vertical, horizontal = dialog.axes.lines
    assert (vertical.get_xdata()[0], horizontal.get_ydata()[0]) == shown
    assert tuple(dialog.axes.images[0].get_extent()) == (0.0, float(COLUMNS), float(ROWS), 0.0)

    # Manual mode on, nothing edited, Apply: the fit is the fit to the last digit, the profile the same point.
    dialog.manual_group.setChecked(True)
    _settle(_app())
    assert (dialog.manual_x.value(), dialog.manual_y.value()) == pytest.approx(shown, abs=5e-4)
    profile = _apply(dialog)
    selected = dialog.result.selected_candidate
    assert (selected.center_x_px, selected.center_y_px) == fitted and selected.warnings == []
    assert (profile.geometry.beam_center_x_px, profile.geometry.beam_center_y_px) == shown
    (toast,) = visible_toasts(dialog.right)
    assert f"centre ({shown[0]:.2f}, {shown[1]:.2f}) px" in toast.text()
    # The settings another tool reads keep the fit's own numbers, as in every earlier version.
    settings = dialog.app_context.settings
    assert (settings.get("detector", "beam_center_x"), settings.get("detector", "beam_center_y")) == fitted


@pytest.mark.parametrize(
    "typed",
    [
        (0.0, 0.0),  # the top-left corner of the frame
        (0.5, 0.5),  # the centre of the first pixel
        (float(COLUMNS), float(ROWS)),  # the bottom-right corner
        (COLUMNS - 0.5, ROWS - 0.5),  # the centre of the last pixel
        (-12.345, 2000.001),  # outside the frame, left and below
    ],
)
def test_a_centre_typed_on_an_edge_is_saved_exported_and_read_back_as_typed(window, tmp_path, typed) -> None:
    dialog = window
    result = _result(tmp_path / "frame.cbf", 400.0, 500.0)
    _show(dialog, result)
    dialog.manual_group.setChecked(True)
    dialog.manual_x.setValue(typed[0])
    dialog.manual_y.setValue(typed[1])
    _settle(_app(), 0.2)  # the overlay follows the fields

    vertical, horizontal = dialog.axes.lines
    assert (vertical.get_xdata()[0], horizontal.get_ydata()[0]) == pytest.approx(typed, abs=1e-9)
    geometry = _apply(dialog).geometry
    assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == pytest.approx(typed, abs=1e-9)
    stored = dialog.result.selected_candidate  # the fit's pixels: half a pixel lower
    assert (stored.center_x_px, stored.center_y_px) == pytest.approx((typed[0] - 0.5, typed[1] - 0.5), abs=1e-9)

    target = tmp_path / "typed.gimap-calibration.json"
    JsonCalibrationStorageAdapter().save(dialog.result, target)
    other = GeometryCalibrationDialog(app_context=_context())
    try:
        other.import_result_from(str(target))
        assert (other.result_center_x.text(), other.result_center_y.text()) == (
            f"{typed[0]:.3f} px", f"{typed[1]:.3f} px"
        )
        assert (other.manual_x.value(), other.manual_y.value()) == pytest.approx(typed, abs=5e-4)
    finally:
        _dispose(other)

    # 'Finish manual': the fitted solution again, also for Apply.
    dialog.manual_group.setChecked(False)
    dialog._commit_manual_values()
    assert (stored.center_x_px, stored.center_y_px) == (400.0, 500.0)


# -- the user folder ---------------------------------------------------------------------------------


def test_the_next_start_reads_the_centre_the_window_showed(tmp_path, monkeypatch) -> None:
    """Apply writes the profile file; another start reads it, and applying the same file again asks nothing."""
    asked = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **_k: (asked.append(args[2]), QMessageBox.Yes)[1])
    monkeypatch.setattr(QMessageBox, "warning", lambda *_a, **_k: pytest.fail("no warning expected"))
    store = tmp_path / "home" / "instrument_profiles.json"
    store.parent.mkdir()
    fitted = (182.47412173380593, 264.3439878683439)  # many digits: nothing may be rounded on the way
    shown = (fitted[0] + 0.5, fitted[1] + 0.5)
    exported = tmp_path / "first.gimap-calibration.json"

    first = GeometryCalibrationDialog(app_context=_context(JsonInstrumentProfileRepository(store)))
    try:
        _show(first, _result(tmp_path / "missing.edf", *fitted))
        name = _apply(first).name
        JsonCalibrationStorageAdapter().save(first.result, exported)
        window_text = (first.result_center_x.text(), first.result_center_y.text())
    finally:
        _dispose(first)
    assert asked == []  # a new profile: nothing to overwrite
    (written,) = json.loads(store.read_text(encoding="utf-8"))["profiles"]
    assert (written["geometry"]["beam_center_x_px"], written["geometry"]["beam_center_y_px"]) == shown
    assert window_text == (f"{shown[0]:.3f} px", f"{shown[1]:.3f} px")

    # The next start: a repository of its own on the same file.
    again = JsonInstrumentProfileRepository(store)
    geometry = again.find(name).geometry
    assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == shown
    second = GeometryCalibrationDialog(app_context=_context(again))
    try:
        second.import_result_from(str(exported))
        assert (second.result_center_x.text(), second.result_center_y.text()) == window_text
        _apply(second)
    finally:
        _dispose(second)
    assert asked == []  # the same geometry: no "moves the beam centre … significantly"
    (rewritten,) = json.loads(store.read_text(encoding="utf-8"))["profiles"]
    assert rewritten["geometry"] == written["geometry"]


# -- language and theme -----------------------------------------------------------------------------


def test_language_and_theme_change_no_number(window, tmp_path) -> None:
    dialog = window
    app = _app()
    _show(dialog, _result(tmp_path / "frame.cbf", 400.0, 500.0))

    def numbers():
        vertical, horizontal = dialog.axes.lines
        return (
            dialog.result_center_x.text(), dialog.result_center_y.text(), dialog.candidate_table.item(0, 2).text(),
            dialog.manual_x.value(), dialog.manual_y.value(), tuple(dialog.axes.images[0].get_extent()),
            vertical.get_xdata()[0], horizontal.get_ydata()[0],
        )

    before = numbers()
    assert before[:3] == ("400.500 px", "500.500 px", "400.5, 500.5")
    try:
        for language, theme in (("zh", "dark"), ("en", "dark"), ("zh", "light"), ("en", "light")):
            i18n.apply_language(language, [dialog])
            apply_theme(theme, 9)
            _settle(app, 0.2)
            assert numbers() == before, (language, theme)
            _apply(dialog)
            (toast,) = visible_toasts(dialog.right)
            assert "(400.50, 500.50) px" in toast.text(), (language, theme)
    finally:
        i18n.apply_language("en")
        apply_theme("light", 9)


# -- reduced previews of large detectors ----------------------------------------------------------------


@pytest.mark.parametrize("shape", [(2162, 2068), (2001, 2003), (3001, 3002)])  # every 2nd, 2nd and 3rd pixel
def test_a_reduced_preview_keeps_every_drawn_pixel_under_the_overlays(shape) -> None:
    app = _app()
    rows, columns = shape
    stride = int(np.ceil(np.sqrt(rows * columns / 1_400_000)))
    assert stride > 1
    data = np.ones(shape, dtype=np.float32)
    kept = [(0, 0), (stride * 100, stride * 150), ((rows - 1) // stride * stride, (columns - 1) // stride * stride)]
    for row, column in kept:
        data[row, column] = 5000.0
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.resize(1180, 760)
    dialog.show()
    try:
        dialog._image_loaded(DetectorImage(data, np.zeros(shape, dtype=bool), Path("large_detector.cbf")))
        _settle(app, 0.2)
        dialog.fit_preview_to_image()
        dialog.canvas.draw()
        assert dialog.axes.get_xlim() == (0.0, float(columns)) and dialog.axes.get_ylim() == (float(rows), 0.0)
        image = dialog.axes.images[0]
        to_screen = dialog.axes.transData.transform
        for row, column in kept:  # the value drawn over the centre of a kept pixel is that pixel's
            x, y = to_screen((column + 0.5, row + 0.5))
            assert float(image.get_cursor_data(SimpleNamespace(x=x, y=y))) == pytest.approx(np.log1p(5000.0), rel=1e-6)
        # A centre marked on a kept pixel lies on its block, whatever the size of the frame.
        row, column = kept[1]
        _show(dialog, _result(Path("large_detector.cbf"), float(column), float(row)))
        vertical, horizontal = dialog.axes.lines
        assert (vertical.get_xdata()[0], horizontal.get_ydata()[0]) == (column + 0.5, row + 0.5)
    finally:
        _dispose(dialog)


# -- what a file may state ------------------------------------------------------------------------------


@pytest.mark.parametrize("stated", ["", {}, {"name": None}, {"name": ["index"]}, "INDEX", " index", 1, True, []])
def test_a_convention_that_is_not_spelled_as_known_is_refused(tmp_path, stated) -> None:
    payload = _former_payload(_result(tmp_path / "missing.edf", 182.25, 264.75))
    payload["metadata"]["center_convention"] = stated
    path = tmp_path / "stated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError) as refused:
        JsonCalibrationStorageAdapter().load(path)
    assert str(refused.value) == UNKNOWN


@pytest.mark.parametrize("change", ["metadata_null", "no_metadata", "no_candidates"])
def test_an_older_file_with_less_in_it_reads_as_indices_and_is_written_with_the_convention(tmp_path, change) -> None:
    payload = _former_payload(_result(tmp_path / "missing.edf", 182.25, 264.75))
    if change == "metadata_null":
        payload["metadata"] = None
    elif change == "no_metadata":
        del payload["metadata"]
    else:
        del payload["candidates"]
    path = tmp_path / "older.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    storage = JsonCalibrationStorageAdapter()
    loaded = storage.load(path)
    assert (loaded.selected_candidate.center_x_px, loaded.selected_candidate.center_y_px) == (182.25, 264.75)
    storage.save(loaded, tmp_path / "again.json")
    written = json.loads((tmp_path / "again.json").read_text(encoding="utf-8"))
    assert written["metadata"]["center_convention"]["name"] == "index"
    assert written["selected_candidate"]["center_x_px"] == 182.25
