"""XRR Tools window (wave 2): the JSON record of an export, its default name and toast, the
geometry from the last calibration, values typed by the user kept on Load first frame, the
summary of where each value came from, the remembered folder and dropped series."""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QApplication, QFileDialog

from src.gimap.app.presentation.components import visible_toasts
from src.gimap.features.calibration.domain import CalibrationCandidate, CalibrationResult
from src.gimap.features.calibration.infrastructure.adapters import SettingsGeometryAdapter
from src.gimap.features.xrr.application import (
    ExportXrrCurve,
    ExportXrrCurveRequest,
    LoadLastCalibrationGeometry,
    SpecularGeometry,
    XrrCalibrationGeometry,
    XrrDetectorFrame,
    XrrExtractionRequest,
    XrrExtractionResult,
    XrrExtractionSettings,
    XrrFrameRef,
    XrrPoint,
    XrrSeriesInspection,
    XrrSeriesSpec,
    default_export_path,
    record_path_for,
)
from src.gimap.features.xrr.domain import qz_from_theta
from src.gimap.features.xrr.infrastructure import (
    LocalXrrCurveExportAdapter,
    LocalXrrRecordAdapter,
    PreferencesXrrFolderAdapter,
    SettingsXrrGeometryAdapter,
)
from src.gimap.features.xrr.presentation.dialog import XrrSeriesDialog
from src.gimap.features.xrr.presentation.view_model import XrrViewModel
from src.gimap.integrations.state import InMemorySettingsRepository, InMemoryUserPreferencesRepository


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _dispose(dialog) -> None:
    """Close the window and delete it now, by the event loop: a parentless window left to the
    garbage collector is deleted inside a collection, which can take the interpreter down."""
    for name in ("_preview_thread", "_conversion_thread", "_load_thread", "_cal_thread",
                 "_inspect_thread", "_extract_thread"):
        thread = getattr(dialog, name, None)
        if thread is not None:
            thread.wait(10000)
    dialog.close()
    dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def _settle(app, seconds: float = 0.05) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        time.sleep(0.01)


def _point(index: int, energy_kev: float = 12.0) -> XrrPoint:
    theta = 0.05 * (index + 1)
    return XrrPoint(
        sequence_index=index,
        source_name=f"frame_{index:05d}.cbf",
        frame_index=0,
        theta_deg=theta,
        qz_inv_angstrom=qz_from_theta(theta, energy_kev),
        intensity=1000.0 / (index + 1),
        roi_center_x_px=400.5,
        roi_center_y_px=300.0 - index,
        valid_pixels=13,
    )


def _request(source: Path) -> XrrExtractionRequest:
    return XrrExtractionRequest(
        series=XrrSeriesSpec(source_path=source, source_kind="cbf", theta_start_deg=0.05, theta_step_deg=0.05),
        geometry=SpecularGeometry(
            distance_m=1.2345,
            energy_kev=12.0,
            pixel_size_x_m=172e-6,
            pixel_size_y_m=172e-6,
            beam_center_x_px=400.5,
            beam_center_y_px=310.25,
            vertical_direction=-1,
        ),
        extraction=XrrExtractionSettings(radius_px=3, aggregation="mean"),
    )


class _Unused:
    def execute(self, *_args, **_kwargs):
        raise AssertionError("not used here")

    def cancel(self):
        return True


class _Fixed:
    def __init__(self, value):
        self.value = value

    def last_calibration(self):
        return self.value


def _view_model(*, calibration=None, preferences=None) -> XrrViewModel:
    return XrrViewModel(
        inspect_series=_Unused(),
        run_extraction=_Unused(),
        export_curve=ExportXrrCurve(LocalXrrCurveExportAdapter(), LocalXrrRecordAdapter()),
        last_calibration=LoadLastCalibrationGeometry(_Fixed(calibration)),
        input_folders=PreferencesXrrFolderAdapter(preferences or InMemoryUserPreferencesRepository()),
    )


def _inspection(shape=(101, 81), metadata=None) -> XrrSeriesInspection:
    return XrrSeriesInspection(
        frame_count=3,
        first_ref=XrrFrameRef(path=Path("series/f_00001.cbf"), frame_index=0, sequence_index=0),
        first_frame=XrrDetectorFrame(np.zeros(shape, dtype=np.float32), None, dict(metadata or {})),
    )


# -- application and adapters ---------------------------------------------------------------------


def test_export_writes_the_csv_and_a_json_record_of_every_setting_that_fixes_qz(tmp_path):
    source = tmp_path / "scan"
    source.mkdir()
    request = _request(source)
    points = tuple(_point(index) for index in range(4))
    csv_path = tmp_path / "out" / "scan_xrr.csv"

    exported = ExportXrrCurve(LocalXrrCurveExportAdapter(), LocalXrrRecordAdapter()).execute(
        ExportXrrCurveRequest(
            csv_path,
            XrrExtractionResult(points),
            settings=request,
            geometry_sources={"energy": "file", "center_x": "typed"},
        )
    )

    assert exported.csv_path == csv_path
    assert exported.record_path == tmp_path / "out" / "scan_xrr.json"
    with csv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # The science is unchanged: qz is written as computed.
    assert [float(row["qz_inv_angstrom"]) for row in rows] == [point.qz_inv_angstrom for point in points]
    record = json.loads(exported.record_path.read_text(encoding="utf-8"))
    assert record["format"] == "gimap-xrr-curve-record"
    assert record["csv"] == "scan_xrr.csv"
    assert record["points"] == 4
    assert record["series"]["source_path"] == str(source)
    assert record["series"]["theta_start_deg"] == 0.05
    assert record["series"]["theta_step_deg"] == 0.05
    geometry = record["geometry"]
    assert geometry["distance_m"] == 1.2345
    assert geometry["energy_kev"] == 12.0
    assert geometry["wavelength_angstrom"] == pytest.approx(12.398419843320026 / 12.0)
    assert geometry["pixel_size_x_m"] == 172e-6
    assert geometry["beam_center_x_px"] == 400.5
    assert geometry["beam_center_y_px"] == 310.25
    assert geometry["vertical_direction"] == -1
    assert geometry["sources"] == {"energy": "file", "center_x": "typed"}
    assert record["extraction"] == {"roi_radius_px": 3, "aggregation": "mean"}
    assert "4·π·sin(θ)/λ" in record["formulas"]["qz"]
    assert record["theta_deg"] == {"first": points[0].theta_deg, "last": points[-1].theta_deg}
    # Every CSV column is explained; no value came from a calibration, so none is named.
    assert set(record["columns"]) == set(rows[0])
    assert "last_calibration" not in geometry


def test_the_record_names_the_calibration_values_came_from(tmp_path):
    calibration = XrrCalibrationGeometry(
        distance_mm=1234.5, source_image="E:/data/AgBh_00001.cbf", timestamp="2026-10-01T12:03:00+00:00",
        image_shape=(1679, 1475), detector="Pilatus",
    )
    exported = ExportXrrCurve(LocalXrrCurveExportAdapter(), LocalXrrRecordAdapter()).execute(
        ExportXrrCurveRequest(
            tmp_path / "scan_xrr.csv",
            XrrExtractionResult((_point(0),)),
            settings=_request(tmp_path),
            geometry_sources={"distance": "calibration", "energy": "file"},
            calibration=calibration,
        )
    )
    record = json.loads(exported.record_path.read_text(encoding="utf-8"))
    assert record["geometry"]["last_calibration"] == {
        "source_image": "E:/data/AgBh_00001.cbf",
        "timestamp": "2026-10-01T12:03:00+00:00",
        "detector": "Pilatus",
        "image_shape": [1679, 1475],
    }
    # The values written are the ones the extraction used, not those of the calibration.
    assert record["geometry"]["distance_m"] == 1.2345


def test_record_and_default_names(tmp_path):
    assert record_path_for(tmp_path / "a.csv") == tmp_path / "a.json"
    assert record_path_for(tmp_path / "a.json") == tmp_path / "a.record.json"
    nxs = tmp_path / "scan_00012.nxs"
    nxs.write_bytes(b"")
    assert default_export_path(nxs) == tmp_path / "scan_00012_xrr.csv"
    folder = tmp_path / "cbf_run"
    folder.mkdir()
    assert default_export_path(folder) == folder / "cbf_run_xrr.csv"


def test_the_last_applied_calibration_is_read_back_in_xrr_units():
    settings = InMemorySettingsRepository({})
    # The built-in detector values are not a calibration.
    settings.set("detector", "distance", 2000.0)
    assert SettingsXrrGeometryAdapter(settings).last_calibration() is None

    candidate = CalibrationCandidate("agbh", 812.25, 640.5, 1575.48, matched_ring_count=4)
    result = CalibrationResult(
        "E:/data/AgBh_00001.cbf", 10, 20, "abc", 10.0, 1.2398419843320026, "Pilatus", 75e-6, 75e-6,
        candidate, [candidate], "2026-10-01T12:03:00+00:00", metadata={"image_shape": [1679, 1475]},
    )
    SettingsGeometryAdapter(settings).apply(result)

    geometry = SettingsXrrGeometryAdapter(settings).last_calibration()
    assert geometry == XrrCalibrationGeometry(
        distance_mm=1575.48,
        energy_kev=10.0,
        pixel_size_x_um=pytest.approx(75.0),
        pixel_size_y_um=pytest.approx(75.0),
        beam_center_x_px=812.25,
        beam_center_y_px=640.5,
        source_image="E:/data/AgBh_00001.cbf",
        timestamp="2026-10-01T12:03:00+00:00",
        image_shape=(1679, 1475),
        detector="Pilatus",
    )


# -- the window ---------------------------------------------------------------------------------


CALIBRATION = XrrCalibrationGeometry(
    distance_mm=1575.48,
    energy_kev=10.0,
    pixel_size_x_um=75.0,
    pixel_size_y_um=75.0,
    beam_center_x_px=812.25,
    beam_center_y_px=640.5,
    source_image="E:/data/AgBh_00001.cbf",
    timestamp="2026-10-01T12:03:00+00:00",
)


def test_geometry_starts_from_the_last_calibration_and_says_so():
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(calibration=CALIBRATION), app_context=object())
    try:
        assert dialog.distance_spin.value() == pytest.approx(1575.48)
        assert dialog.energy_spin.value() == pytest.approx(10.0)
        assert dialog.pixel_x_spin.value() == pytest.approx(75.0)
        assert dialog.center_x_spin.value() == pytest.approx(812.25)
        assert dialog.center_y_spin.value() == pytest.approx(640.5)
        assert set(dialog.geometry_sources().values()) == {"calibration"}
        tag = dialog.geometry_source_labels["distance"]
        assert tag.text() == "(from last calibration)"
        assert not tag.isHidden()
        assert "AgBh_00001.cbf" in tag.toolTip()
        assert not dialog.geometry_note.isHidden()
        assert "AgBh_00001.cbf" in dialog.geometry_note.text()
    finally:
        _dispose(dialog)


def test_without_a_calibration_the_values_are_marked_as_defaults():
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    try:
        assert dialog.distance_spin.value() == pytest.approx(5000.0)
        assert set(dialog.geometry_sources().values()) == {"default"}
        tag = dialog.geometry_source_labels["energy"]
        assert tag.text() == "built-in default"
        assert tag.property("gimapRole") == "warning"
        assert dialog.geometry_note.isHidden()
    finally:
        _dispose(dialog)


def test_load_first_frame_keeps_a_typed_beam_center_and_reports_each_source():
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    try:
        dialog.center_x_spin.setValue(321.0)  # as the user types it
        assert dialog.geometry_sources()["center_x"] == "typed"
        assert dialog.geometry_source_labels["center_x"].isHidden()

        dialog._on_inspected(_inspection(metadata={"energy_kev": 9.5, "distance_m": 2.5}))

        assert dialog.center_x_spin.value() == pytest.approx(321.0)  # kept
        assert dialog.center_y_spin.value() == pytest.approx(50.0)  # (101 - 1) / 2: image centre
        assert dialog.energy_spin.value() == pytest.approx(9.5)
        assert dialog.distance_spin.value() == pytest.approx(2500.0)
        sources = dialog.geometry_sources()
        assert sources["center_x"] == "typed"
        assert sources["center_y"] == "image center"
        assert sources["energy"] == "file" and sources["distance"] == "file"
        assert sources["pixel_x"] == "default"
        summary = dialog.series_summary.text()
        assert "3 frame(s) · first: f_00001.cbf · shape 81 × 101" in summary
        assert "From the file: distance, energy" in summary
        assert "Image center (the file has no beam center): beam center Y" in summary
        assert "Defaults, check them: pixel size" in summary
        assert "Kept as typed: beam center X" in summary
        assert dialog.geometry_source_labels["center_y"].text() == "image center"
    finally:
        _dispose(dialog)


def test_a_beam_center_in_the_file_never_replaces_a_typed_one_but_is_reported():
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(calibration=CALIBRATION), app_context=object())
    try:
        dialog.center_y_spin.setValue(600.0)
        dialog._on_inspected(_inspection(metadata={"beam_center_x_px": 700.0, "beam_center_y_px": 500.0}))
        assert dialog.center_y_spin.value() == pytest.approx(600.0)
        assert dialog.center_x_spin.value() == pytest.approx(700.0)  # the file over the calibration
        assert dialog.energy_spin.value() == pytest.approx(10.0)  # the calibration stays
        assert "The file says: beam center Y 500.00 px" in dialog.series_summary.text()
        # The calibration value the file replaced is named too.
        assert "The last calibration says: beam center X 812.25 px" in dialog.series_summary.text()
        assert "From the last calibration: distance, energy, pixel size" in dialog.series_summary.text()

        # A next series without metadata: the values of the previous file go back to the calibration.
        dialog._on_inspected(_inspection())
        assert dialog.center_x_spin.value() == pytest.approx(812.25)
        assert dialog.geometry_sources()["center_x"] == "calibration"
        assert dialog.center_y_spin.value() == pytest.approx(600.0)
        assert "says" not in dialog.series_summary.text()
    finally:
        _dispose(dialog)


def test_a_file_value_equal_to_the_calibration_is_not_reported_as_replaced():
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(calibration=CALIBRATION), app_context=object())
    try:
        dialog._on_inspected(_inspection(metadata={"energy_kev": 10.0, "distance_m": 2.0}))
        summary = dialog.series_summary.text()
        assert "From the file: distance, energy" in summary
        assert "The last calibration says: distance 1575.480 mm" in summary
        assert "energy 10" not in summary
    finally:
        _dispose(dialog)


def test_the_value_fields_line_up_whatever_their_tags_say():
    app = _app()
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    dialog.resize(1280, 820)
    dialog.show()
    try:
        dialog.center_y_spin.setValue(321.0)  # typed: no tag
        dialog._on_inspected(_inspection(metadata={"energy_kev": 9.5}))
        _settle(app)
        spins = [dialog.distance_spin, dialog.energy_spin, dialog.pixel_x_spin, dialog.center_x_spin, dialog.center_y_spin]
        assert len({spin.width() for spin in spins}) == 1
        assert dialog.geometry_source_labels["center_y"].isHidden()

        # No tag at all: the values take the whole row again.
        for spin in (dialog.distance_spin, dialog.energy_spin, dialog.pixel_x_spin, dialog.pixel_y_spin, dialog.center_x_spin):
            spin.setValue(spin.value() + 1.0)
        _settle(app)
        assert dialog.center_y_spin.width() > dialog.direction_combo.width() - 4
    finally:
        _dispose(dialog)


def test_a_calibration_of_frames_of_another_shape_gives_only_its_energy():
    _app()
    calibration = XrrCalibrationGeometry(**{**CALIBRATION.__dict__, "image_shape": (1043, 981)})
    dialog = XrrSeriesDialog(view_model=_view_model(calibration=calibration), app_context=object())
    try:
        assert dialog.distance_spin.value() == pytest.approx(1575.48)  # nothing loaded yet
        dialog._on_inspected(_inspection(shape=(101, 81)))
        sources = dialog.geometry_sources()
        assert sources["energy"] == "calibration"
        assert dialog.energy_spin.value() == pytest.approx(10.0)
        assert sources["distance"] == "default" and dialog.distance_spin.value() == pytest.approx(5000.0)
        assert sources["pixel_x"] == "default" and dialog.pixel_x_spin.value() == pytest.approx(172.0)
        assert sources["center_x"] == "image center" and dialog.center_x_spin.value() == pytest.approx(40.0)
        assert (
            "The last calibration is of 981 × 1043 frames, not of this series: only its energy is used."
            in dialog.series_summary.text()
        )
        # A series of the calibrated shape takes the whole calibration again.
        dialog._on_inspected(_inspection(shape=(1043, 981)))
        assert set(dialog.geometry_sources().values()) == {"calibration"}
        assert "not of this series" not in dialog.series_summary.text()
    finally:
        _dispose(dialog)


def test_export_proposes_the_source_folder_writes_the_record_and_shows_open_folder(tmp_path, monkeypatch):
    app = _app()
    view_model = _view_model()
    dialog = XrrSeriesDialog(view_model=view_model, app_context=object())
    dialog.resize(1280, 820)
    dialog.show()
    try:
        source = tmp_path / "run_07"
        source.mkdir()
        dialog.center_x_spin.setValue(400.5)
        dialog._run_geometry_sources = dialog.geometry_sources()
        view_model.state.request = _request(source)
        view_model.state.result = XrrExtractionResult(tuple(_point(index) for index in range(3)))
        asked = {}

        def save_name(_parent, _title, start, _filter):
            asked["start"] = start
            return start, "CSV (*.csv)"

        monkeypatch.setattr(QFileDialog, "getSaveFileName", save_name)
        dialog._export_curve()
        _settle(app)

        assert asked["start"] == str(source / "run_07_xrr.csv")
        assert (source / "run_07_xrr.csv").is_file()
        record = json.loads((source / "run_07_xrr.json").read_text(encoding="utf-8"))
        assert record["geometry"]["sources"]["center_x"] == "typed"
        toasts = visible_toasts(dialog)
        assert len(toasts) == 1
        assert "run_07_xrr.csv" in toasts[0].text() and "run_07_xrr.json" in toasts[0].text()
        assert toasts[0].action_button is not None and toasts[0].action_button.text() == "Open Folder"
        assert dialog.job_status.message_label.text() == "Exported run_07_xrr.csv"
    finally:
        _dispose(dialog)


def test_browse_starts_in_the_remembered_folder_and_remembers_the_choice(tmp_path, monkeypatch):
    _app()
    remembered = tmp_path / "earlier"
    remembered.mkdir()
    preferences = InMemoryUserPreferencesRepository({"xrr.last_folder": str(remembered)})
    dialog = XrrSeriesDialog(view_model=_view_model(preferences=preferences), app_context=object())
    try:
        chosen = tmp_path / "new" / "scan.nxs"
        chosen.parent.mkdir()
        chosen.write_bytes(b"")
        asked = {}

        def open_name(_parent, _title, start, _filter):
            asked["start"] = start
            return str(chosen), ""

        monkeypatch.setattr(QFileDialog, "getOpenFileName", open_name)
        dialog._browse_source()
        assert asked["start"] == str(remembered)
        assert dialog.source_picker.path() == str(chosen)
        assert preferences.get("xrr.last_folder") == str(chosen.parent)
    finally:
        _dispose(dialog)


def _mime(*paths: Path) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path)) for path in paths])
    return mime


def test_a_dropped_cbf_folder_becomes_the_series_and_its_first_frame_is_read(tmp_path, monkeypatch):
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    try:
        assert dialog.acceptDrops()
        folder = tmp_path / "cbf_series"
        folder.mkdir()
        text = tmp_path / "notes.txt"
        text.write_text("x", encoding="utf-8")
        inspected = []
        monkeypatch.setattr(dialog, "_inspect_series", lambda: inspected.append(dialog.source_picker.path()))
        dialog.source_kind_combo.setCurrentIndex(1)  # NXS: a folder cannot be one

        text_mime = _mime(text)  # the event does not own its data: keep it alive
        refused = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, text_mime, Qt.LeftButton, Qt.NoModifier)
        dialog.dragEnterEvent(refused)
        assert not refused.isAccepted()
        mime = _mime(folder)
        enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        dialog.dragEnterEvent(enter)
        assert enter.isAccepted()
        drop = QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        dialog.dropEvent(drop)

        assert inspected == [str(folder)]
        assert dialog.source_kind_combo.currentIndex() == 2  # CBF series
        assert dialog.view_model.last_folder() == str(folder)
    finally:
        _dispose(dialog)


def test_switching_the_language_after_the_window_is_gone_is_safe():
    from src.gimap.app.presentation.i18n import apply_language

    app = _app()
    dialog = XrrSeriesDialog(view_model=_view_model(calibration=CALIBRATION), app_context=object())
    dialog.show()
    dialog.close()
    _settle(app, 0.1)
    try:
        apply_language("zh")
        apply_language("en")
    finally:
        apply_language("en")
    survivor = XrrSeriesDialog(view_model=_view_model(calibration=CALIBRATION), app_context=object())
    try:
        apply_language("zh")
        assert survivor.geometry_source_labels["distance"].text() != ""
    finally:
        apply_language("en")
        assert survivor.geometry_source_labels["distance"].text() == "(from last calibration)"
        _dispose(survivor)


def test_the_summary_is_composed_again_in_the_new_language_as_it_was_when_read(monkeypatch):
    from src.gimap.app.presentation import i18n

    for english, chinese in (
        ("From the file: {fields}", "来自文件：{fields}"),
        ("The file says: {values}", "文件中的值：{values}"),
        ("energy", "能量"),
        ("beam center Y", "光束中心 Y"),
    ):
        monkeypatch.setitem(i18n.ZH, english, chinese)
        monkeypatch.setitem(i18n._TO_ENGLISH, chinese, english)
    _app()
    dialog = XrrSeriesDialog(view_model=_view_model(), app_context=object())
    try:
        dialog.center_y_spin.setValue(600.0)
        dialog._on_inspected(_inspection(metadata={"energy_kev": 9.5, "beam_center_y_px": 500.0}))
        dialog.distance_spin.setValue(1234.0)  # typed after the frame was read
        before = dialog.series_summary.text()
        i18n.apply_language("zh")
        summary = dialog.series_summary.text()
        assert "来自文件：能量" in summary
        assert "文件中的值：光束中心 Y 500.00 px" in summary
        i18n.apply_language("en")
        assert dialog.series_summary.text() == before  # the distance typed later is not in it
        assert "Kept as typed: beam center Y" in before and "distance" not in before.split("Kept as typed:")[1]
    finally:
        i18n.apply_language("en")
        _dispose(dialog)


def test_the_fake_view_model_of_older_tests_still_works():
    _app()
    view_model = SimpleNamespace(state=SimpleNamespace(result=None), cancel=lambda: True)
    dialog = XrrSeriesDialog(view_model=view_model, app_context=object())
    try:
        assert dialog._start_folder() == ""
        assert set(dialog.geometry_sources().values()) == {"default"}
    finally:
        _dispose(dialog)
