"""Calibration review fixes: an imported result's fitted copy, the results layout at the
default size, one empty-state message, and deleting a per-use window once it is idle.

The module calibrates the pyFAI AgBh Pilatus1M.edf frame once (about 5 s) and exports it.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QCoreApplication, QEvent, QPoint, QThread
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget

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


def _geometry(candidate) -> tuple:
    return (
        candidate.center_x_px,
        candidate.center_y_px,
        candidate.distance_mm,
        list(candidate.warnings),
    )


@pytest.fixture(scope="module")
def calibration_json(tmp_path_factory) -> Path:
    """An exported AgBh calibration, as Export Calibration writes it."""
    if not AGBH.is_file():
        pytest.skip("pyFAI AgBh test frame not available")
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.show()
    dialog.load_image(str(AGBH))
    _wait(app, lambda: dialog._load_thread is None and dialog.image is not None, 60)
    dialog.start_calibration()
    _wait(app, lambda: dialog._cal_thread is None and dialog.result is not None, 240)
    path = tmp_path_factory.mktemp("calibration") / "agbh.gimap-calibration.json"
    dialog.view_model.export_result(path)
    dialog.close()
    _settle(app, 0.1)
    return path


def _import(dialog, path, monkeypatch) -> None:
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_a, **_k: (str(path), ""))
    dialog.import_result()


def test_reset_and_finish_manual_restore_an_imported_solution(calibration_json, monkeypatch):
    app = _app()
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.show()
    _settle(app, 0.2)
    _import(dialog, calibration_json, monkeypatch)
    _settle(app, 0.3)
    result = dialog.result
    selected = result.selected_candidate
    # The case of the finding: the imported selected solution is an object of its own.
    assert all(selected is not candidate for candidate in result.candidates)
    imported = _geometry(selected)
    assert MANUAL_REFINEMENT_WARNING not in imported[3]
    others = [_geometry(candidate) for candidate in result.candidates]
    # Export commits the manual values before its save dialog, which is cancelled here.
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_a, **_k: ("", ""))

    dialog.manual_refine_button.click()
    dialog.manual_x.setValue(imported[0] + 25.0)
    dialog.export_result()
    assert selected.center_x_px == pytest.approx(imported[0] + 25.0, abs=1e-3)
    assert MANUAL_REFINEMENT_WARNING in selected.warnings

    dialog.reset_manual_button.click()
    assert dialog.manual_x.value() == pytest.approx(imported[0], abs=1e-3)
    assert result.selected_candidate is selected  # never swapped for candidate 0
    assert _geometry(selected) == imported
    assert dialog.result_warning.text() == (" ".join(imported[3]) or "None")

    # Committed again, then 'Finish manual' and Apply's commit: still the imported solution.
    dialog.manual_x.setValue(imported[0] + 25.0)
    dialog.export_result()
    dialog.manual_refine_button.click()
    assert not dialog.manual_group.isChecked()
    assert _geometry(selected) == imported
    dialog._commit_manual_values()
    assert _geometry(selected) == imported
    assert dialog._display_candidate().center_x_px == pytest.approx(imported[0])
    assert [_geometry(candidate) for candidate in result.candidates] == others
    dialog.close()


def _rows_outside_viewport(dialog) -> list[str]:
    viewport = dialog.result_scroll.viewport()
    outside = []
    for title, value in dialog.result_rows:
        for label in (title, value):
            corner = label.mapTo(viewport, QPoint(0, 0))
            if corner.y() < 0 or corner.y() + label.height() > viewport.height():
                outside.append(label.objectName())
            # The text's right end, not the label's (a value cell may be wider than its text).
            text_width = min(label.width(), label.fontMetrics().horizontalAdvance(label.text()))
            if corner.x() + text_width > viewport.width() + 1:
                outside.append(f"{label.objectName()} (right)")
    return outside


def _clipped_labels(dialog) -> list[str]:
    clipped = []
    for title, value in dialog.result_rows:
        for label in (title, value):
            if label.wordWrap():
                continue
            if label.fontMetrics().horizontalAdvance(label.text()) > label.width() + 1:
                clipped.append(label.objectName())
    return clipped


@pytest.mark.parametrize("language", ["en", "zh"])
def test_every_solution_row_shows_at_the_default_size(calibration_json, monkeypatch, language):
    from src.gimap.app.presentation import i18n

    app = _app()
    i18n.apply_language(language)
    try:
        dialog = GeometryCalibrationDialog(app_context=_context())
        dialog.resize(1180, 760)  # the default size
        dialog.show()
        _settle(app, 0.4)
        # The empty preview has one message: the centred one over the canvas.
        assert dialog.preview_empty_label.isVisible()
        assert not dialog.preview_info_label.isVisible()

        _import(dialog, calibration_json, monkeypatch)
        _settle(app, 0.4)
        assert dialog.preview_info_label.isVisible()
        assert dialog.result_warning.text() not in ("", "—", "None")  # a wrapped warning
        assert dialog._result_two_columns
        assert dialog.result_scroll.verticalScrollBar().maximum() == 0
        assert _rows_outside_viewport(dialog) == []
        assert _clipped_labels(dialog) == []
        assert dialog.canvas.height() >= 240
        assert dialog.candidates_group.width() >= 240

        # Switched to the other language while open (wider English titles, taller Chinese
        # lines): every row still shows whole.
        other = "zh" if language == "en" else "en"
        i18n.apply_language(other, [dialog])
        _settle(app, 0.3)
        assert _clipped_labels(dialog) == []
        assert _rows_outside_viewport(dialog) == []
        i18n.apply_language(language, [dialog])
        _settle(app, 0.3)

        # A narrow window: one pair per row, so neither the labels nor the table are squeezed.
        dialog.resize(900, 600)
        _settle(app, 0.4)
        assert not dialog._result_two_columns
        assert _clipped_labels(dialog) == []
        dialog.resize(1180, 760)
        _settle(app, 0.4)
        assert dialog._result_two_columns
        dialog.close()
    finally:
        i18n.apply_language("en")


def _deleted_events(dialog) -> list[str]:
    events = []
    dialog.destroyed.connect(lambda *_: events.append("destroyed"))
    return events


def _flush_deletes(app) -> None:
    for _ in range(5):
        app.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        time.sleep(0.01)


def test_dispose_when_idle_deletes_a_closed_window_at_once() -> None:
    app = _app()
    parent = QWidget()
    dialog = GeometryCalibrationDialog(parent, app_context=_context())
    events = _deleted_events(dialog)
    dialog.show()
    _settle(app, 0.1)
    dialog.close()  # the window that exec_() showed was closed while idle
    _flush_deletes(app)
    assert events == []  # closed windows are kept without dispose_when_idle
    dialog.dispose_when_idle()
    _flush_deletes(app)
    assert events == ["destroyed"]
    assert parent.findChildren(GeometryCalibrationDialog) == []
    parent.deleteLater()


def test_dispose_when_idle_waits_for_a_running_image_read() -> None:
    app = _app()
    parent = QWidget()
    dialog = GeometryCalibrationDialog(parent, app_context=_context())
    events = _deleted_events(dialog)
    dialog.show()
    _settle(app, 0.1)
    running = QThread()
    running.start()
    dialog._load_thread = running
    try:
        dialog.close()  # hides and waits for the read
        assert dialog._close_when_idle and not dialog.isVisible()
        dialog.dispose_when_idle()
        _flush_deletes(app)
        assert events == []  # the thread still runs: nothing is deleted under it
    finally:
        running.quit()
        running.wait(2000)
    dialog._load_thread = None
    dialog._cleanup_loader()  # the read ends: the deferred close accepts and deletes
    _settle(app, 0.1)
    _flush_deletes(app)
    assert events == ["destroyed"]
    parent.deleteLater()
