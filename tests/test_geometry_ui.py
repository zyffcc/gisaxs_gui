"""The geometry in the windows: the calibration energy is never made up, the Detector geometry dialog says where
its wavelength comes from and has the tilt, Load / Export .poni, and frames of another detector are not appended."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.features.calibration.domain import DetectorImage
from src.gimap.features.calibration.presentation.dialog import GeometryCalibrationDialog
from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from src.gimap.shared.geometry.poni import format_poni, read_poni
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_calibration_presentation import _context as _calibration_context

EXTERNAL = Path(__file__).resolve().parent / "data" / "external"
GALAXI = EXTERNAL / "gisaxs_galaxi" / "galaxi_data.tif"
requires_public = pytest.mark.skipif(not GALAXI.exists(), reason="public test data")


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_the_calibration_energy_is_required_when_the_file_states_none() -> None:
    _app()
    dialog = GeometryCalibrationDialog(app_context=_calibration_context())
    try:
        image = DetectorImage(np.ones((32, 40), np.float32), np.zeros((32, 40), bool), Path("standard.cbf"))
        dialog._image_loaded(image)
        assert dialog.energy_spin.value() == 0 and dialog.energy_spin.text() == "Required"  # no 12 keV made up
        assert not dialog.calibrate_button.isEnabled()
        assert dialog.energy_source_label.text().startswith("Not in the file")
        assert dialog.energy_source_label.property("gimapRole") == "warning"
        dialog.energy_spin.setValue(10.0)  # typed by the person
        assert dialog.calibrate_button.isEnabled() and dialog.run_energy_source() == "entered by user"
        assert dialog.energy_source_label.text() == "Entered by you."
        dialog._image_loaded(image)  # the next image without energy: the value is kept, and said to be checked
        assert dialog.energy_spin.value() == 10.0 and dialog.energy_source_label.text().startswith("Not in this file")
        stated = DetectorImage(np.ones((32, 40), np.float32), np.zeros((32, 40), bool), Path("s.cbf"), energy_kev=8.04)
        dialog._image_loaded(stated)
        assert dialog.energy_spin.value() == pytest.approx(8.04) and dialog.run_energy_source() == "file header"
        assert dialog.tilt_check.isChecked() and dialog.run_tilt_mode() == "auto"
    finally:
        dialog.close()


def _galaxi_profile() -> InstrumentProfile:
    geometry = DetectorGeometry(172e-6, 172e-6, 1.73, 597.1, 719.6, 1.34, 0.463, rot1_rad=math.radians(0.3))
    return InstrumentProfile("GALAXI", geometry, None, (1043, 981), source="test")


@requires_public
def test_the_geometry_dialog_has_the_tilt_and_says_where_the_wavelength_comes_from(tmp_path: Path) -> None:
    frame = tmp_path / GALAXI.name
    shutil.copy2(GALAXI, frame)  # no .poni beside it, and its TIFF states no wavelength
    context = _context(InMemoryInstrumentProfileRepository([_galaxi_profile()]))
    page = _page(context)
    try:
        page.add_paths([str(frame)])
        _done(page)
        page._enter_geometry()
        dialog = page._geometry_dialog
        assert dialog.rot1_spin.value() == pytest.approx(0.3) and dialog.rot2_spin.value() == 0.0
        assert dialog.wavelength_spin.value() == pytest.approx(1.34)
        assert dialog.energy_spin.value() == pytest.approx(12.398419843320026 / 1.34, rel=1e-4)
        assert dialog.wavelength_source_label.text() == "Wavelength of profile “GALAXI”."
        dialog.energy_spin.setValue(10.0)
        assert dialog.wavelength_spin.value() == pytest.approx(1.23984, abs=1e-5)
        assert dialog.wavelength_source_label.text() == "Wavelength entered by you."
        dialog.wavelength_spin.setValue(0.0)
        assert dialog.wavelength_source_label.property("gimapRole") == "warning"
        dialog.save_button.click()  # refused: no wavelength
        assert "Enter the wavelength or the energy first" in dialog.status_label.text()
        with pytest.raises(ValueError):
            dialog.geometry()
        dialog.wavelength_spin.setValue(1.34)
        dialog.rot2_spin.setValue(-0.25)
        assert math.degrees(dialog.geometry().rot2_rad) == pytest.approx(-0.25)
        written = page.export_poni(str(tmp_path / "out.poni"), dialog=dialog)
        poni = read_poni(written.read_text(encoding="utf-8"))
        assert math.degrees(poni.tilt.rot2_rad) == pytest.approx(-0.25) and poni.shape == (1043, 981)
        assert poni.geometry().beam_center_x_px == pytest.approx(dialog.center_x_spin.value(), abs=1e-6)
        dialog.close()
    finally:
        page.dispose()


@requires_public
def test_load_poni_makes_it_the_profile_of_this_detector(tmp_path: Path, monkeypatch) -> None:
    from PyQt5.QtWidgets import QMessageBox

    frame = tmp_path / GALAXI.name
    shutil.copy2(GALAXI, frame)
    page = _page(_context(InMemoryInstrumentProfileRepository([])))
    try:
        page.add_paths([str(frame)])
        _done(page)
        assert page.view_model.state.analysis.geometry is None
        asked = []
        monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: asked.append(args[2]) or QMessageBox.No)
        other = tmp_path / "perkin.poni"  # a file that states another frame size
        other.write_text(format_poni(DetectorGeometry(2e-4, 2e-4, 0.81, 545.5, 1826.9, 0.6888), shape=(2048, 2048)),
                         encoding="utf-8")
        assert not page.load_poni(str(other))  # asked, and not taken
        assert asked and page.view_model.state.analysis.geometry is None
        assert page.load_poni(str(EXTERNAL / "gisaxs_galaxi" / "galaxi_bornagain.poni"))
        _done(page)
        analysis = page.view_model.state.analysis
        assert analysis.geometry is not None and analysis.resolution.how == "chosen"
        assert analysis.geometry.distance_m == pytest.approx(1.73)
        assert ".poni loaded:" in analysis.resolution.profile.source
    finally:
        page.dispose()


@requires_public
def test_a_profile_of_another_detector_is_offered_in_the_banner_and_used_only_when_asked(tmp_path: Path) -> None:
    frame = tmp_path / GALAXI.name
    shutil.copy2(GALAXI, frame)
    perkin = InstrumentProfile("Perkin", DetectorGeometry(2e-4, 2e-4, 0.81, 545.5, 1826.9, 0.6888), None, (2048, 2048))
    page = _page(_context(InMemoryInstrumentProfileRepository([perkin])))
    try:
        page.view_model.set_profile("Perkin")  # picked for the frames before (a GIWAXS project)
        page.add_paths([str(frame)])
        _done(page)
        analysis = page.view_model.state.analysis
        assert analysis.geometry is None and analysis.resolution.refused_profile == "Perkin"
        assert not page.banner.isHidden() and not page.banner_anyway_button.isHidden()
        assert page.banner_anyway_button.text() == "Use “Perkin” Anyway"
        assert "is from another detector, so it is not used" in page.banner_label.text()
        page.banner_anyway_button.click()
        _done(page)
        analysis = page.view_model.state.analysis
        assert analysis.geometry is not None and analysis.resolution.how == "chosen"
        assert any("Profile “Perkin” is for 2048×2048 frames" in message for message in analysis.messages)
    finally:
        page.dispose()


@requires_public
def test_frames_of_another_detector_are_refused_when_added(tmp_path: Path, monkeypatch) -> None:
    from src.gimap.features.analyze.presentation.bindings import frame_import

    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    shutil.copy2(GALAXI, first / GALAXI.name)
    data = np.ones((64, 80), np.float32)
    from PIL import Image

    Image.fromarray(data, mode="F").save(second / "small_0001.tif")
    page = _page(_context(InMemoryInstrumentProfileRepository([])))
    toasts = []
    monkeypatch.setattr(frame_import, "show_toast", lambda parent, text, **options: toasts.append(text))
    try:
        assert page.import_paths([str(first)])
        _done(page)
        assert len(page.view_model.state.files) == 1 and page.view_model.state.analysis is not None
        monkeypatch.setattr(type(page), "ask_replace_or_add", lambda self, paths: "add")
        assert page.import_paths([str(second)])
        _done(page)
        assert len(page.view_model.state.files) == 1  # refused: a list never mixes two detectors
        assert toasts and toasts[-1].startswith("These frames are 64×80")
        monkeypatch.setattr(type(page), "ask_replace_or_add", lambda self, paths: "replace")
        assert page.import_paths([str(second)])
        _done(page)
        assert [path.name for path in page.view_model.state.files] == ["small_0001.tif"]
        monkeypatch.setattr(type(page), "ask_replace_or_add", lambda self, paths: None)
        assert not page.import_paths([str(first)])  # cancelled: nothing changes
        assert [path.name for path in page.view_model.state.files] == ["small_0001.tif"]
    finally:
        page.dispose()
