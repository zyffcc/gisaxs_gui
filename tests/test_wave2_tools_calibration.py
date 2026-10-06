"""Geometry Calibration (wave 2): Apply confirms with one toast (the overwrite question stays),
Export shows a toast with Open Folder, the file dialogs start in the remembered folder, and a
detector image or an exported calibration can be dropped on the window."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox

from src.gimap.app import AppContext
from src.gimap.app.presentation.components import visible_toasts
from src.gimap.features.calibration.domain import CalibrationCandidate, CalibrationResult
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGBH = PROJECT_ROOT / "tests/data/external/saxs_agbh_pyfai/Pilatus1M.edf"


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
        time.sleep(0.01)


def _context(preferences=None) -> AppContext:
    context = AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=preferences or InMemoryUserPreferencesRepository(),
    )
    context.instrument_profiles = InMemoryInstrumentProfileRepository()
    return context


def _result(source: Path, center=(123.5, 234.5), distance=1456.7) -> CalibrationResult:
    candidate = CalibrationCandidate("agbh", center[0], center[1], distance, matched_ring_count=3)
    return CalibrationResult(
        str(source), 10, 20, "abc", 12.0, 12.398419843320026 / 12.0, "Pilatus 1M", 172e-6, 172e-6,
        candidate, [candidate], datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def no_information_box(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("Apply must confirm with a toast, not a message box")

    monkeypatch.setattr(QMessageBox, "information", refuse)


def test_apply_confirms_once_with_a_toast_naming_the_instrument_profile(tmp_path, no_information_box):
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.resize(1180, 760)
    dialog.show()
    try:
        dialog.result = _result(tmp_path / "scan_agbh.cbf")
        dialog.apply_result()
        _settle(app)
        toasts = visible_toasts(dialog.right)
        assert len(toasts) == 1
        text = toasts[0].text()
        assert "Geometry calibration applied: centre (123.50, 234.50) px, distance 1456.70 mm." in text
        assert "Saved as instrument profile" in text and dialog.view_model.last_profile.name in text
        assert not visible_toasts(dialog)  # over the preview and results, not over Apply and Close
        # The toast never covers the footer buttons.
        footer_top = dialog.calibration_export_section.mapTo(dialog, QPoint(0, 0)).y()
        toast_bottom = toasts[0].mapTo(dialog, QPoint(0, toasts[0].height())).y()
        assert toast_bottom <= footer_top
    finally:
        _dispose(dialog)


def test_the_overwrite_question_stays_and_no_toast_follows_a_no(tmp_path, monkeypatch, no_information_box):
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.show()
    try:
        dialog.result = _result(tmp_path / "scan_agbh.cbf")
        dialog.apply_result()
        _settle(app)
        for toast in visible_toasts(dialog.right):
            toast.close()
        asked = []
        monkeypatch.setattr(
            QMessageBox, "question", lambda *args, **_kwargs: asked.append(args[2]) or QMessageBox.No
        )
        dialog.result = _result(tmp_path / "scan_agbh.cbf", center=(400.0, 500.0), distance=900.0)
        dialog.apply_result()
        _settle(app)
        assert len(asked) == 1 and "Overwrite the profile?" in asked[0]
        assert not visible_toasts(dialog.right)
    finally:
        _dispose(dialog)


def test_export_writes_the_file_and_offers_open_folder(tmp_path, monkeypatch):
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.show()
    try:
        dialog.result = _result(tmp_path / "scan_agbh.cbf")
        target = tmp_path / "out" / "agbh.gimap-calibration.json"
        target.parent.mkdir()
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: (str(target), ""))
        dialog.export_result()
        _settle(app)
        assert target.is_file()
        toasts = visible_toasts(dialog.right)
        assert len(toasts) == 1
        assert toasts[0].text() == "Calibration exported: agbh.gimap-calibration.json"
        assert toasts[0].action_button.text() == "Open Folder"
    finally:
        _dispose(dialog)


def test_open_and_import_start_in_the_remembered_folder(tmp_path, monkeypatch):
    _app()
    preferences = InMemoryUserPreferencesRepository({"calibration.last_folder": str(tmp_path)})
    dialog = GeometryCalibrationDialog(app_context=_context(preferences))
    try:
        starts = []

        def open_name(_parent, _title, start, _filter):
            starts.append(start)
            return "", ""

        monkeypatch.setattr(QFileDialog, "getOpenFileName", open_name)
        dialog.open_image_dialog()
        dialog.import_result()
        assert starts == [str(tmp_path), str(tmp_path)]
        # With an image path in the window, its folder comes first.
        other = tmp_path / "other"
        other.mkdir()
        dialog.path_edit.setText(str(other / "frame.cbf"))
        dialog.open_image_dialog()
        assert starts[-1] == str(other)
    finally:
        _dispose(dialog)


@pytest.mark.skipif(not AGBH.is_file(), reason="pyFAI AgBh test frame not available")
def test_a_loaded_image_is_remembered_for_the_next_open():
    app = _app()
    preferences = InMemoryUserPreferencesRepository()
    dialog = GeometryCalibrationDialog(app_context=_context(preferences))
    dialog.show()
    try:
        dialog.load_image(str(AGBH))
        end = time.monotonic() + 60
        while (dialog._load_thread is not None or dialog.image is None) and time.monotonic() < end:
            _settle(app, 0.05)
        assert dialog.image is not None
        assert Path(preferences.get("calibration.last_folder")) == AGBH.parent
    finally:
        _dispose(dialog)
        _settle(app, 0.1)


def _mime(path: Path) -> QMimeData:
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(path))])
    return mime


def test_dropping_an_image_loads_it_and_a_json_imports_it(tmp_path, monkeypatch):
    _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    try:
        assert dialog.acceptDrops()
        image = tmp_path / "AgBh_00001.cbf"
        record = tmp_path / "AgBh.gimap-calibration.json"
        text = tmp_path / "notes.txt"
        for path in (image, record, text):
            path.write_bytes(b"")
        loaded, imported = [], []
        monkeypatch.setattr(dialog, "load_image", lambda path, *_args: loaded.append(path))
        monkeypatch.setattr(dialog, "import_result_from", lambda path: imported.append(path))

        text_mime = _mime(text)
        refused = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, text_mime, Qt.LeftButton, Qt.NoModifier)
        dialog.dragEnterEvent(refused)
        assert not refused.isAccepted()
        for path in (image, record):
            mime = _mime(path)
            enter = QDragEnterEvent(QPoint(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
            dialog.dragEnterEvent(enter)
            assert enter.isAccepted()
            dialog.dropEvent(QDropEvent(QPointF(5, 5), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
        assert [Path(path) for path in loaded] == [image]
        assert [Path(path) for path in imported] == [record]
    finally:
        _dispose(dialog)
