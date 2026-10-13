"""Release fixes of Analyze's geometry (R1, R6, R28).

A geometry saved after a session beam centre is the one used: Edit / Enter Geometry, Use Previous Geometry,
the automatic analysis's Find Calibration and the Calibration window's Apply all forget the centre set
earlier in the session for that detector — a centre typed into the dialog used to be saved and then ignored,
with a green status line. A centre set on another detector stays. The automatic analysis and Find Geometry
say “Open a frame first.” with no frame open, instead of leaving an empty outcome on the next frame.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from src.gimap.features.analyze.presentation.geometry_dialog import PROFILE_NAME_WIDTH, GeometryDialog
from src.gimap.shared.geometry import DetectorGeometry, InstrumentProfile
from tests.test_analyze_frames_and_center import SHAPE, FakeFrames, _geometry, _symmetric_frame, _view_model
from tests.test_analyze_workspace import _app, _context, _done, _page
from tests.test_review_analyze_run_and_clear import _frames, _profiles


def _shown(tmp_path: Path):
    model = _view_model(FakeFrames({"a.cbf": [_symmetric_frame()]}))
    model.add_paths([tmp_path / "a.cbf"])
    model.select(0)
    model.set_mode("gisaxs")
    model.accept(model.analyze(model.request()))
    return model


def _centre(model) -> tuple[float, float]:
    geometry = model.state.analysis.geometry
    return geometry.beam_center_x_px, geometry.beam_center_y_px


def test_a_geometry_saved_after_a_session_centre_is_the_one_used(tmp_path: Path) -> None:
    model = _shown(tmp_path)
    model.set_beam_center(150.0, 60.0)  # dragged, typed, or refined by an automatic analysis
    model.accept(model.analyze(model.request()))
    assert _centre(model) == (150.0, 60.0) and model.center_state()["overridden"]

    saved = _geometry().with_beam_center(201.25, 111.5)  # typed into Edit Geometry… and saved
    model.save_profile("Test", saved, source="entered by hand")
    assert model.state.beam_center is None and model.request().beam_center is None
    model.accept(model.analyze(model.request()))
    assert _centre(model) == (201.25, 111.5)
    state = model.center_state()
    assert not state["overridden"] and state["source"] == "profile" and state["profile_center"] == (201.25, 111.5)


def test_a_centre_set_on_another_detector_survives_a_geometry_saved_for_this_one(tmp_path: Path) -> None:
    model = _shown(tmp_path)
    model.set_beam_center(545.5, 1826.9, shape=(2048, 2048))  # set on frames of another detector
    model.save_profile("Test", _geometry().with_beam_center(201.25, 111.5), source="entered by hand")
    assert model.state.beam_center == (545.5, 1826.9) and model.state.beam_center_shape == (2048, 2048)
    model.accept(model.analyze(model.request()))
    assert _centre(model) == (201.25, 111.5)  # not used for this frame: the saved geometry is

    model.set_beam_center(150.0, 60.0, shape=None)  # a centre without a shape (a settings file): every frame
    model.save_profile("Test", _geometry().with_beam_center(202.0, 112.0), source="entered by hand")
    assert model.state.beam_center is None


def test_edit_geometry_in_the_page_saves_the_centre_typed_and_uses_it(tmp_path: Path, monkeypatch) -> None:
    paths = _frames(tmp_path, 1)
    page = _page(_context(_profiles()))
    try:
        page.add_paths([str(path) for path in paths])
        _done(page)
        page.set_session_center(150.0, 200.0)  # the cross dragged
        _done(page)
        assert page.center_button.property("centerSource") == "session"

        def accept(dialog) -> int:  # the dialog is pre-filled with the centre in use; the person types another
            assert dialog.center_x_spin.value() == 150.0 and dialog.center_y_spin.value() == 200.0
            dialog.center_x_spin.setValue(201.5)
            dialog.center_y_spin.setValue(251.25)
            return GeometryDialog.Accepted

        monkeypatch.setattr(GeometryDialog, "exec_", accept)
        page._enter_geometry()
        _done(page)
        geometry = page.view_model.state.analysis.geometry
        assert (geometry.beam_center_x_px, geometry.beam_center_y_px) == (201.5, 251.25)
        assert page.center_button.property("centerSource") == "profile"
        assert page.status_level() == "ok"  # … and the green line is now true
    finally:
        page.tasks.wait(30)
        page.dispose()
        page.close()


def test_a_calibration_applied_is_the_geometry_used(tmp_path: Path) -> None:
    """The contract with the Calibration window: after Apply it calls ``AnalyzePage.geometry_calibrated()``."""
    paths = _frames(tmp_path, 1)
    profiles = _profiles()
    page = _page(_context(profiles))
    try:
        assert page.geometry_calibrated() is None  # no frame open: nothing to analyse, no error
        page.add_paths([str(path) for path in paths])
        _done(page)
        page.set_session_center(150.0, 200.0)
        _done(page)
        shown = []
        page.analysisShown.connect(shown.append)
        # What the Calibration window does on Apply: its profile is saved in the store, then Analyze is told.
        calibrated = profiles.load_all()[0].geometry.with_beam_center(203.0, 252.0)
        profiles.save(InstrumentProfile("calibrated here", calibrated, None, profiles.load_all()[0].detector_shape))
        page.geometry_calibrated()
        assert page.view_model.state.beam_center is None
        assert page.profile_combo.findData("calibrated here") >= 0  # the profiles were read again
        _done(page)
        assert len(shown) == 1  # the frame shown was analysed again
        assert not page.view_model.center_state()["overridden"]
        assert page.center_button.property("centerSource") == "profile"
    finally:
        page.tasks.wait(30)
        page.dispose()
        page.close()


def test_run_and_find_geometry_ask_for_a_frame_first() -> None:
    page = _page(_context())
    called = []
    page.set_automatic_analysis(run=lambda: called.append("run"), find_geometry=lambda: called.append("find"))
    try:
        for button in (page.run_pipeline_button, page.find_geometry_button, page.banner_find_button):
            page._status("Ready")
            button.click()
            assert called == []
            assert page.status_text() == "Open a frame first." and page.status_level() == "warning"
        assert page.step_rail.state("results") == "pending" and page.current_step() == "data"
        page.view_model.state.files.append(Path("C:/data/frame.tif"))  # a frame is listed and chosen
        page.view_model.select(0)
        page.run_pipeline_button.click()
        page.find_geometry_button.click()
        assert called == ["run", "find"]
    finally:
        page.dispose()
        page.close()


def test_the_geometry_dialog_shows_a_long_profile_name_from_its_start() -> None:
    _app()
    name = "GALAXI Pilatus 1M (BornAgain example)"
    defaults = dict(distance_mm=1730.0, pixel_x_um=172.0, pixel_y_um=172.0, wavelength_angstrom=1.34,
                    center_x=597.1, center_y=719.6, incidence_deg=0.463)
    dialog = GeometryDialog(name, defaults, allow_delete=True)
    try:
        edit = dialog.name_edit
        assert edit.minimumWidth() == PROFILE_NAME_WIDTH >= 260 and edit.cursorPosition() == 0
        dialog.show()
        _app().processEvents()
        assert edit.width() >= PROFILE_NAME_WIDTH and edit.text() == name
        assert edit.cursorPosition() == 0  # the start of the name is in view, not “… example)”
        room = edit.width() - 16
        assert edit.fontMetrics().horizontalAdvance(name) <= room or edit.cursorPosition() == 0
    finally:
        dialog.close()
        dialog.deleteLater()


@pytest.mark.parametrize("shape", [SHAPE])
def test_the_profile_saved_is_for_the_frame_shown(tmp_path: Path, shape) -> None:
    """Unchanged by the fix: the profile is saved for the detector of the frame shown and matched automatically."""
    model = _shown(tmp_path)
    model.set_profile("Test")
    profile = model.save_profile("Other", DetectorGeometry(100e-6, 100e-6, 2.0, 190.0, 100.0, 1.0, 0.3), source="x")
    assert profile.detector_shape == shape and model.state.profile_name is None
