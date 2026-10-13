"""Exported calibrations: the same numbers as ever, with the convention of the beam centre named.

``center_x_px`` / ``center_y_px`` in a ``*.gimap-calibration.json`` are pixel-centre indices (the centre of
the first pixel is 0.0), half a pixel lower than the window, Analyze and the instrument profiles show. The
file now says so in ``metadata["center_convention"]``. A file from before that field reads to the same
geometry, a file written now reads back identically, and a reader from before the field still reads it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QCoreApplication, QEvent
from PyQt5.QtWidgets import QApplication, QMessageBox

from src.gimap.app import AppContext
from src.gimap.app.presentation import i18n
from src.gimap.features.calibration.bootstrap import create_headless_calibration
from src.gimap.features.calibration.domain import (
    CalibrationCandidate,
    CalibrationResult,
    MatchedRing,
    calibrated_geometry,
)
from src.gimap.features.calibration.infrastructure.adapters import JsonCalibrationStorageAdapter
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.integrations.state import (
    InMemoryInstrumentProfileRepository,
    InMemorySessionRepository,
    InMemorySettingsRepository,
    InMemoryUserPreferencesRepository,
)

UNKNOWN = "This calibration file states a beam-centre convention this version does not know."
UNKNOWN_ZH = "这个标定文件注明的光束中心坐标约定，这个版本不认识。"
FORMER_KEYS = {
    "source_image", "source_image_size", "source_image_mtime_ns", "image_sha256", "energy_kev",
    "wavelength_angstrom", "detector_name", "pixel_size_x_m", "pixel_size_y_m", "selected_candidate",
    "candidates", "calibration_timestamp", "software_version", "metadata", "format", "format_version",
}
"""Every top-level key of an export of the first release: none is added, none is renamed."""


def _candidate(key: str, x: float, y: float, distance: float) -> CalibrationCandidate:
    return CalibrationCandidate(
        key, x, y, distance, matched_ring_count=2, rms_residual_px=0.8, confidence="High",
        matched_rings=[MatchedRing(0, 0.1076, 150.2, 150.0, 0.2), MatchedRing(1, 0.2152, 300.3, 300.5, -0.2)],
        detected_peak_radii_px=[150.2, 300.3, 451.0],
    )


def _result(tmp_path: Path) -> CalibrationResult:
    """A result as the engine records it (no detector file: only the record is looked at)."""
    best = _candidate("agbh", 182.47412173380593, 264.3439878683439, 1575.4781391138824)
    other = _candidate("lab6", 176.4, 264.1, 420.5)
    return CalibrationResult(
        str(tmp_path / "missing_agbh.edf"), 10, 20, "abc", 12.0, 12.398419843320026 / 12.0, "Pilatus 1M",
        172e-6, 172e-6, best, [best, other], "2026-10-13T08:00:00+00:00",
        metadata={
            "loader": {"format": "edf", "header": {"Dim_1": "981"}},
            "image_shape": [1043, 981],
            "estimated_distance_mm": None,
            "distance_range_mm": [30.0, 10000.0],
        },
    )


def _former_payload(result: CalibrationResult) -> dict:
    """The record exactly as ``CalibrationResult.to_dict`` wrote it before the convention was named."""
    payload = asdict(result)
    payload["format"] = "gimap-geometry-calibration"
    payload["format_version"] = 1
    return payload


def _former_reader(payload: dict) -> CalibrationResult:
    """``CalibrationResult.from_dict`` of the first release: every top-level key but two is a field."""
    values = {key: value for key, value in payload.items() if key not in {"format", "format_version"}}
    values["selected_candidate"] = CalibrationCandidate.from_dict(values["selected_candidate"])
    values["candidates"] = [CalibrationCandidate.from_dict(item) for item in values.get("candidates", [])]
    return CalibrationResult(**values)


def _centres(result: CalibrationResult) -> list[tuple[float, float, float]]:
    return [(c.center_x_px, c.center_y_px, c.distance_mm) for c in (result.selected_candidate, *result.candidates)]


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# -- what is written -------------------------------------------------------------------------------


def test_an_export_keeps_its_numbers_and_keys_and_names_the_convention(tmp_path: Path) -> None:
    result = _result(tmp_path)
    target = tmp_path / "agbh.gimap-calibration.json"
    JsonCalibrationStorageAdapter().save(result, target)
    written = json.loads(target.read_text(encoding="utf-8"))
    former = _former_payload(result)

    assert set(written) == FORMER_KEYS == set(former)
    # The file opens with what it is and how its centres are counted.
    assert list(written)[:3] == ["format", "format_version", "metadata"]
    assert list(written["metadata"])[0] == "center_convention"
    convention = written["metadata"].pop("center_convention")
    assert convention["name"] == "index"
    assert "(0.0, 0.0)" in convention["description"] and "(0.5, 0.5)" in convention["description"]
    assert "center_x_px" in convention["description"] and "0.5 px higher" in convention["description"]
    # It names the windows that show the other numbers: not every tool window of GIMaP counts pixels that way.
    assert "Geometry Calibration and Analyze windows" in convention["description"]
    # Apart from that one field, the record is the former one, value for value.
    assert written == former
    assert written["format_version"] == 1
    assert written["selected_candidate"]["center_x_px"] == 182.47412173380593
    assert written["selected_candidate"]["center_y_px"] == 264.3439878683439
    assert [item["center_x_px"] for item in written["candidates"]] == [182.47412173380593, 176.4]
    # The result in memory does not carry the field: only the file does.
    assert "center_convention" not in result.metadata


def test_a_file_written_now_reads_back_identically(tmp_path: Path) -> None:
    result = _result(tmp_path)
    target = tmp_path / "agbh.gimap-calibration.json"
    storage = JsonCalibrationStorageAdapter()
    storage.save(result, target)
    loaded = storage.load(target)

    assert loaded == result
    assert loaded.to_dict() == result.to_dict()
    assert "center_convention" not in loaded.metadata
    storage.save(loaded, tmp_path / "again.json")  # written again: the same file
    assert (tmp_path / "again.json").read_text(encoding="utf-8") == target.read_text(encoding="utf-8")


# -- files of an older version ---------------------------------------------------------------------


def test_a_file_from_before_the_field_reads_to_the_same_geometry(tmp_path: Path) -> None:
    result = _result(tmp_path)
    former = _write(tmp_path / "former.gimap-calibration.json", _former_payload(result))
    assert "center_convention" not in former.read_text(encoding="utf-8")
    loaded = JsonCalibrationStorageAdapter().load(former)

    assert loaded == result
    assert _centres(loaded) == _centres(result)
    # The same instrument profile as from the result itself: the index + 0.5 in canonical pixels.
    geometry = calibrated_geometry(loaded)
    assert geometry == calibrated_geometry(result)
    assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == (182.97412173380593, 264.8439878683439)
    assert geometry.distance_m == pytest.approx(1.5754781391138824)


def test_a_reader_from_before_the_field_still_reads_a_file_written_now(tmp_path: Path) -> None:
    """Why the field lives in ``metadata``: that reader takes every other top-level key for a field."""
    result = _result(tmp_path)
    target = tmp_path / "agbh.gimap-calibration.json"
    JsonCalibrationStorageAdapter().save(result, target)
    payload = json.loads(target.read_text(encoding="utf-8"))

    read = _former_reader(payload)
    assert _centres(read) == _centres(result)
    assert read.metadata["image_shape"] == [1043, 981]
    assert read.metadata["center_convention"]["name"] == "index"  # kept as a note, true for that reader too
    # … and when that version exports it again, this one reads its file.
    again = JsonCalibrationStorageAdapter().load(_write(tmp_path / "again.json", _former_payload(read)))
    assert again == result


def test_the_assistant_reads_both_kinds_of_file_to_the_same_canonical_centre(tmp_path: Path) -> None:
    result = _result(tmp_path)
    now = tmp_path / "now.gimap-calibration.json"
    JsonCalibrationStorageAdapter().save(result, now)
    former = _write(tmp_path / "former.gimap-calibration.json", _former_payload(result))
    headless = create_headless_calibration()

    read_now, read_former = headless.read_result(str(now)), headless.read_result(str(former))
    assert read_now == read_former
    assert read_now["beam_center_px"] == [182.97412173380593, 264.8439878683439]
    (other,) = [item for item in read_now["alternatives"] if item["standard"] == "lab6"]
    assert other["beam_center_px"] == [176.9, 264.6]
    assert read_now["distance_mm"] == 1575.4781391138824


# -- a stated convention is believed ---------------------------------------------------------------


@pytest.mark.parametrize("stated", [{"name": "index"}, "index", {"name": "index", "description": "other words"}])
def test_index_centres_are_taken_as_they_are(tmp_path: Path, stated) -> None:
    result = _result(tmp_path)
    payload = _former_payload(result)
    payload["metadata"]["center_convention"] = stated
    loaded = JsonCalibrationStorageAdapter().load(_write(tmp_path / "stated.json", payload))
    assert loaded == result


@pytest.mark.parametrize("stated", [{"name": "canonical"}, "canonical"])
def test_canonical_centres_in_a_file_are_converted_to_the_fit_s_pixels(tmp_path: Path, stated) -> None:
    result = _result(tmp_path)
    payload = _former_payload(result)
    payload["metadata"]["center_convention"] = stated
    for item in (payload["selected_candidate"], *payload["candidates"]):
        item["center_x_px"] += 0.5  # the numbers a person reads in the window
        item["center_y_px"] += 0.5
    loaded = JsonCalibrationStorageAdapter().load(_write(tmp_path / "canonical.json", payload))

    assert _centres(loaded) == pytest.approx(_centres(result))
    assert [c.distance_mm for c in loaded.candidates] == [c.distance_mm for c in result.candidates]
    assert "center_convention" not in loaded.metadata
    geometry = calibrated_geometry(loaded)
    assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == pytest.approx(
        (182.97412173380593, 264.8439878683439)
    )


@pytest.mark.parametrize("stated", [{"name": "fit2d"}, "one-based", {"description": "no name"}, 0.5])
def test_an_unknown_convention_is_refused_and_not_guessed(tmp_path: Path, stated) -> None:
    payload = _former_payload(_result(tmp_path))
    payload["metadata"]["center_convention"] = stated
    path = _write(tmp_path / "unknown.json", payload)
    with pytest.raises(ValueError) as refused:
        JsonCalibrationStorageAdapter().load(path)
    assert str(refused.value) == UNKNOWN


def test_the_window_says_so_in_the_interface_language(tmp_path: Path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    payload = _former_payload(_result(tmp_path))
    payload["metadata"]["center_convention"] = {"name": "fit2d"}
    path = _write(tmp_path / "unknown.json", payload)
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, title, text, *_a, **_k: shown.append((title, text)))
    context = AppContext(
        settings=InMemorySettingsRepository({}),
        session=InMemorySessionRepository(),
        preferences=InMemoryUserPreferencesRepository(),
        instrument_profiles=InMemoryInstrumentProfileRepository(),
    )
    dialog = GeometryCalibrationDialog(app_context=context)
    try:
        dialog.import_result_from(str(path))
        assert shown == [("Import Calibration", UNKNOWN)]
        assert dialog.result is None and not dialog.apply_button.isEnabled()  # nothing half-read is shown
        monkeypatch.setitem(i18n.ZH, UNKNOWN, UNKNOWN_ZH)
        monkeypatch.setitem(i18n._TO_ENGLISH, UNKNOWN_ZH, UNKNOWN)
        i18n.apply_language("zh", [dialog])
        dialog.import_result_from(str(path))
        assert shown[-1][1] == UNKNOWN_ZH
    finally:
        i18n.apply_language("en")
        dialog.close()
        dialog.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
