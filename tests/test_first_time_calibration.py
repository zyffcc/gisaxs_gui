"""A first calibration with the public AgBh frame and its documented test wavelength."""

from pathlib import Path

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation import i18n
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.shared.detector_io.metadata import extract_cbf_metadata, extract_edf_metadata
from src.gimap.shared.geometry.poni import read_poni
from tests.test_calibration_presentation import _context
from tests.test_release2_calibration_window import _dispose, _wait

AGBH = Path(__file__).parent / "data/external/saxs_agbh_pyfai/Pilatus1M.edf"


def enter(spin, value):
    QTest.mouseClick(spin.lineEdit(), Qt.LeftButton)
    QTest.keyClick(spin, Qt.Key_A, Qt.ControlModifier)
    QTest.keyClicks(spin, value)
    QTest.keyClick(spin, Qt.Key_Tab)


def test_wavelength_to_checked_calibration_and_direct_poni(qt_application, tmp_path, monkeypatch):
    dialog = GeometryCalibrationDialog(app_context=_context())
    dialog.resize(900, 560)
    dialog.show()
    try:
        assert not dialog.export_poni_button.isEnabled()
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(AGBH), ""))
        QTest.mouseClick(dialog.open_button, Qt.LeftButton)
        _wait(qt_application, lambda: dialog.image is not None and dialog.open_button.isEnabled(), 30)
        assert "from frame size" in dialog.pixel_label.text()
        assert "Check the pixel size" in dialog.detector_label.text()
        assert not dialog.calibrate_button.isEnabled()
        viewport = dialog.calibrationControlsScroll.viewport()
        assert viewport.rect().contains(dialog.energy_spin.mapTo(viewport, dialog.energy_spin.rect().center()))

        enter(dialog.wavelength_spin, "0.01")
        assert dialog.energy_spin.value() == 0 and not dialog.calibrate_button.isEnabled()
        assert "exceeds 200 keV" in dialog.energy_source_label.text()
        enter(dialog.wavelength_spin, "1.0")
        assert dialog.energy_spin.value() == pytest.approx(12.39842)
        assert dialog.stage_label.text() == "Image loaded. Click Auto Calibration."
        QTest.mouseClick(dialog.calibrate_button, Qt.LeftButton)
        assert not dialog.wavelength_spin.isEnabled()
        _wait(qt_application, lambda: dialog.export_poni_button.isEnabled(), 90)
        candidate = dialog.result.selected_candidate
        assert candidate.matched_ring_count == 5
        assert not candidate.warnings
        assert "Check the ring overlay before Apply." in dialog.stage_label.text()
        assert candidate.confidence in dialog.stage_label.text()
        _wait(qt_application, lambda: dialog.stage_label.height() >= dialog.stage_label.heightForWidth(
            dialog.stage_label.width()), 3)
        assert dialog.result.metadata["energy_source"] == "entered by user"
        i18n.apply_language("zh", [dialog])
        assert "匹配环数：5" in dialog.stage_label.text()
        assert "根据图像尺寸推定" in dialog.pixel_label.text()
        i18n.apply_language("en", [dialog])

        def save(_parent, _title, filename, formats):
            assert filename.endswith("Pilatus1M.gimap-calibration.poni")
            assert formats == "pyFAI geometry (*.poni)"
            return str(tmp_path / "geometry"), formats

        monkeypatch.setattr(QFileDialog, "getSaveFileName", save)
        QTest.mouseClick(dialog.export_poni_button, Qt.LeftButton)
        poni = read_poni((tmp_path / "geometry.poni").read_text(encoding="utf-8"))
        assert poni.tilt.beam_distance_m == pytest.approx(candidate.distance_mm / 1000)
        assert poni.wavelength_angstrom == pytest.approx(dialog.result.wavelength_angstrom)
    finally:
        i18n.apply_language("en", [dialog])
        _dispose(dialog)


@pytest.mark.parametrize(("reader", "stated"), [
    (extract_edf_metadata, {"PSize_1": "0.0001", "PSize_2": "0.00012"}),
    (extract_cbf_metadata, {"_array_data.header_contents": "# Pixel_size 0.0001 m x 0.00012 m"}),
])
def test_inferred_and_stated_pixel_sizes_have_distinct_sources(reader, stated):
    assert reader({}, (1043, 981))["pixel_size_source"] == "frame shape"
    metadata = reader(stated, (1043, 981))
    assert metadata.get("pixel_size_source") != "frame shape"
    assert (metadata["pixel_size_x_m"], metadata["pixel_size_y_m"]) == pytest.approx((0.0001, 0.00012))
