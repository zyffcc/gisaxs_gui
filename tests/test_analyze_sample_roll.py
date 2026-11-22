"""The roll shown in Analyze survives file export, session restore and asynchronous measurement."""

from dataclasses import replace
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application import analysis_metadata, read_poni_file, settings_from_record, settings_record
from src.gimap.features.analyze.presentation.bindings import sample_roll
from src.gimap.shared.geometry.sample_roll import RollEstimate, RollMeasurement
from tests.test_analyze_geometry_adjustment import page  # noqa: F401  (workspace with two memory-backed sources)


def _incidence():
    estimate = {"deg": 0.50, "error_deg": 0.012}
    return SimpleNamespace(as_dict=lambda: dict(estimate, horizon=estimate, specular=None, yoneda=None, method="horizon"))


@pytest.mark.parametrize(("value", "record", "expected"), [
    (.35, dict(deg=.35, error_deg=.013, decision="agreement"), "0.35"),
    (.35, dict(deg=.35, error_deg=.013, decision="user_override"), "0.35"),
    (.27, dict(deg=.35, error_deg=.013, decision="measured"), "0.27"),
    (.15, dict(deg=.15, error_deg=.072, origin="measured", significant=False), "0.150"),
    (.15, dict(deg=.15, error_deg=.072, origin="automatic", significant=False), "0.15"),
])
def test_only_an_applied_estimate_supplies_the_angle_precision(value, record, expected):
    from src.gimap.shared.numbers import applied_angle

    assert applied_angle(value, record) == expected


def test_measure_roll_on_worker_and_round_trip(page, monkeypatch, tmp_path):
    main = get_ident()
    result = RollMeasurement(RollEstimate(1.234, .12), RollEstimate(1.236, .1), 1.235619, .1, "combined", "specular rod")

    def measure(analysis, source, *, use_resolved_roll, critical_angle_deg, yoneda_material):
        assert use_resolved_roll
        assert yoneda_material is None
        assert critical_angle_deg == pytest.approx(.134)
        assert source is page.view_model.frames
        assert get_ident() != main and analysis is page.view_model.state.analysis
        return result, _incidence()

    monkeypatch.setattr(sample_roll, "measure_frame_angles", measure)
    page.yoneda_angle_spin.setValue(.134)
    assert page.tasks.wait(10)
    page._use_incidence(.075)
    assert page.tasks.wait(10)
    page.roll_step_button.click()
    page.measure_roll_button.click()
    assert page.tasks.wait(10)
    model = page.view_model
    current = model.state.analysis
    assert current.geometry.sample_roll_deg == result.deg
    assert "1.24 ± 0.10°" in page.roll_estimate_label.text()
    assert page.roll_step_button.text() == "ψ 1.24°"
    assert "ψ = 1.24°" in page.summary_label.text()
    assert "αi = 0.500°" in page.summary_label.text()
    assert "0.500 ± 0.012°" in page.incidence_spin.toolTip()
    assert current.sample_roll_measurement["method"] == "combined"
    assert current.geometry.incidence_deg == pytest.approx(.5)
    assert "0.075°" in page.incidence_estimate_label.text() and "3σ" in page.incidence_estimate_label.text()
    assert page.incidence_spin.value() == pytest.approx(.5)
    record = analysis_metadata(current)["incidence_measurement"]
    assert record["reference_deg"] == pytest.approx(.075)
    assert record["reference_source"] == "entered" and record["decision"] == "disagreement"
    assert record["horizon"] == _incidence().as_dict()["horizon"]
    assert record["calibration"]["beam_center_y_px"] == current.geometry.beam_center_y_px
    assert analysis_metadata(current)["sample_roll_measurement"]["horizon"] == result.as_dict()["horizon"]
    saved = settings_from_record(settings_record(model.current_settings()))
    assert saved.incidence_measurement == record
    assert saved.yoneda_critical_deg == pytest.approx(.134)
    assert current.geometry.critical_angle_deg == 0
    model.set_incidence_measurement(None)
    model.set_sample_roll(0)
    model.apply_settings(saved)
    assert model.state.incidence_measurement == record
    assert model.state.sample_roll_measurement == current.sample_roll_measurement
    assert model.state.sample_roll_deg == result.deg
    assert all(req.sample_roll_deg == result.deg for req in model.batch_requests())
    written = page.export_poni(str(tmp_path / "example.poni"))
    poni = read_poni_file(written)
    assert poni.geometry().sample_roll_deg == result.deg
    assert poni.sample_roll_measurement == current.sample_roll_measurement
    page._enter_geometry()
    assert page._geometry_dialog.geometry().sample_roll_deg == result.deg
    profile_angle = current.resolution.profile.geometry.incidence_deg
    assert page._geometry_dialog.geometry().incidence_deg == pytest.approx(profile_angle)
    assert page.load_poni(str(written), dialog=page._geometry_dialog)
    assert page._geometry_dialog.geometry().incidence_deg == pytest.approx(profile_angle)
    try:
        apply_language("zh", [page])
        page.refresh_language()
        assert "镜面杆" in page.roll_estimate_label.text()
        assert page.measure_roll_button.text() == "从当前帧测量"
        assert "参考值（0.075°）" in page.incidence_estimate_label.text()
    finally:
        apply_language("en", [page])


def test_roll_measurement_finishing_after_edit_is_dropped(page, monkeypatch):
    started, release = Event(), Event()
    def measure(_analysis, _source, **_options):
        started.set()
        assert release.wait(5)
        return RollMeasurement(RollEstimate(1., .1), None, 1., .1, "horizon"), _incidence()
    monkeypatch.setattr(sample_roll, "measure_frame_angles", measure)
    page.measure_roll_button.click()
    try:
        assert started.wait(2)
        page.sample_roll_spin.setValue(-2.0)
    finally:
        release.set()
    assert page.tasks.wait(10)
    assert page.view_model.state.analysis.geometry.sample_roll_deg == -2.
    assert page.view_model.state.sample_roll_measurement is None
    assert page.view_model.state.incidence_measurement is None


def test_incidence_override_scope_and_sample_reset(page, monkeypatch, tmp_path):
    model = page.view_model
    monkeypatch.setattr(sample_roll, "measure_frame_angles", lambda *_, **_options: (
        RollMeasurement(None, None, None, None, "unavailable"), _incidence()))
    page._use_incidence(.075)
    assert page.tasks.wait(10)
    page.incidence_lock_check.click()
    assert page.tasks.wait(10)
    page.measure_roll_button.click()
    assert page.tasks.wait(10)
    assert model.state.analysis.geometry.incidence_deg == pytest.approx(.075)
    assert model.state.analysis.incidence_measurement["decision"] == "user_override"
    page.incidence_lock_check.click()
    assert page.tasks.wait(10)
    measured = model.state.analysis
    assert measured.geometry.incidence_deg == pytest.approx(.5)
    original = model.request()
    other = model.analyze(replace(original, frame_index=1))
    assert other.incidence_measurement is None and other.geometry.incidence_deg == pytest.approx(.075)
    changed = model.analyze(replace(original, beam_center=(200., 105.)))
    assert changed.incidence_measurement is None and changed.geometry.incidence_deg == pytest.approx(.075)
    remasked = model.analyze(replace(original, corrections=replace(original.corrections, maximum=40.)))
    assert remasked.incidence_measurement is None and remasked.geometry.incidence_deg == pytest.approx(.075)
    for stale, expected in ((False, .5), (True, .075)):
        observation = original.incidence_measurement
        if stale:
            observation = dict(observation, frame_files={"changed input": {"sha256": "different"}})
        fast = model.analyze(replace(original, with_record=False, incidence_measurement=observation))
        assert fast.geometry.incidence_deg == pytest.approx(expected)
    # A project or batch request can substitute a nominal angle while retaining the observed pixels.
    agreed = model.analyze(replace(original, incidence_deg=.495))
    assert agreed.incidence_measurement["decision"] == "agreement"
    assert agreed.incidence_measurement["override_deg"] == pytest.approx(.495)
    assert agreed.geometry.incidence_deg == pytest.approx(.495)
    locked = model.analyze(replace(original, incidence_locked=True))
    assert locked.incidence_measurement["decision"] == "user_override"
    assert locked.incidence_measurement["applied_deg"] is None
    assert locked.geometry.incidence_deg == pytest.approx(.075)
    page.incidence_spin.setValue(.2)
    assert page.tasks.wait(10)
    assert model.state.analysis.geometry.incidence_deg == pytest.approx(.2)
    assert model.state.analysis.incidence_measurement["decision"] == "user_override"
    assert model.state.analysis.incidence_measurement["reference_deg"] == pytest.approx(.075)
    assert model.state.analysis.incidence_measurement["override_deg"] == pytest.approx(.2)
    assert page.incidence_lock_check.isChecked()
    profile_angle = model.state.analysis.resolution.profile.geometry.incidence_deg
    saved = page.export_poni(str(tmp_path / "geometry.poni"))
    assert page.load_poni(saved)
    assert page.tasks.wait(10)
    assert model.state.analysis.resolution.profile.geometry.incidence_deg == pytest.approx(profile_angle)
    page._reset_sample_choices()
    assert model.state.incidence_measurement is None
    assert not model.state.incidence_locked and model.state.incidence_deg is None


def test_yoneda_material_and_roll_must_match_saved_measurement(page):
    from src.gimap.features.analyze.application.incidence_record import frame_incidence_record

    model = page.view_model
    analysis = model.state.analysis
    estimate = {"deg": .45, "error_deg": .01}
    measured = dict(estimate, method="yoneda", yoneda=estimate, critical_angle_deg=.14,
                    critical_angle_source="Yoneda critical angle", roll_deg=0.)
    record = frame_incidence_record(analysis, measured, origin="measured", reference_deg=.075,
                                    reference_source="entered", applied_deg=.45, decision="disagreement")
    request = replace(model.request(), incidence_measurement=record, yoneda_critical_deg=.14)
    assert model.analyze(request).geometry.incidence_deg == pytest.approx(.45)
    assert model.analyze(replace(request, yoneda_critical_deg=.18)).incidence_measurement is None
    assert model.analyze(replace(request, sample_roll_deg=3.)).incidence_measurement is None


def test_automatic_roll_is_scoped_to_original_sum(page):
    model = page.view_model
    req = model.request()
    record = dict(origin="automatic", path=str(req.path), frame_index=req.frame_index, summed_frames=[])
    model.set_sample_roll(1.5, record)
    assert model.request().sample_roll_deg == 1.5
    other = replace(model.request(), summed_frames=((req.path, 1),))
    assert other.with_scoped_roll().sample_roll_deg is None
    assert model.analyze(other).geometry.sample_roll_deg == 0


def test_invalid_roll_measurement_in_saved_settings(page):
    record = settings_record(page.view_model.current_settings())
    record["sample_roll_measurement"] = {"method": "horizon", "horizon": "bad data"}
    with pytest.raises(ValueError, match="Invalid sample roll"):
        settings_from_record(record)


def test_invalid_incidence_settings(page):
    record = settings_record(page.view_model.current_settings())
    with pytest.raises(ValueError, match="Yoneda critical angle"):
        settings_from_record(dict(record, yoneda_critical_deg=-.2))
    with pytest.raises(ValueError, match="Invalid incidence measurement"):
        settings_from_record(dict(record, incidence_measurement={"method": "specular", "deg": .3}))
    with pytest.raises(ValueError, match="Unknown Yoneda material"):
        settings_from_record(dict(record, yoneda_material="unknown"))


def test_material_selection_uses_frame_wavelength_and_preserves_manual_entry(page):
    from src.gimap.shared.material_optics import critical_angle_deg

    model = page.view_model
    page.yoneda_angle_spin.setValue(.134)
    assert page.tasks.wait(10)
    combo = page.yoneda_material_combo
    combo.setCurrentIndex(combo.findData("SiO2"))
    assert page.tasks.wait(10)
    expected = critical_angle_deg("SiO2", model.state.analysis.geometry.wavelength_angstrom / 10)
    assert f"{expected:.4f}°" in page.yoneda_material_value.text()
    assert model.request().yoneda_critical_deg is None
    page.measure_roll_button.click()
    assert page.tasks.wait(10)
    measured = model.state.analysis.incidence_measurement
    assert measured["critical_angle_source"] == "material optical constants"
    assert measured["critical_angle_deg"] == pytest.approx(expected)
    assert measured["yoneda_material"] == "SiO2" and measured["material_wavelength_angstrom"] == pytest.approx(1.)
    assert model.state.analysis.geometry.critical_angle_deg == 0
    assert model.analyze(replace(model.request(), yoneda_material="Au")).incidence_measurement is None
    saved = settings_from_record(settings_record(model.current_settings()))
    assert saved.yoneda_material == "SiO2" and saved.yoneda_critical_deg == pytest.approx(.134)
    combo.setCurrentIndex(0)
    assert page.tasks.wait(10)
    assert model.request().yoneda_material is None and model.request().yoneda_critical_deg == pytest.approx(.134)
    model.apply_settings(saved)
    profile = model.state.analysis.resolution.profile
    model._profiles.save(replace(profile, geometry=replace(profile.geometry, wavelength_angstrom=4.)))
    page.run_analysis()
    assert page.tasks.wait(10)
    assert combo.currentData() == "SiO2" and "Outside the optical table" in page.yoneda_material_value.text()
    assert model.state.analysis.incidence_measurement is None
    page.measure_roll_button.click()
    assert page.tasks.wait(10)
    unavailable = model.state.analysis.incidence_measurement
    assert unavailable["material_optics_error"] and unavailable["yoneda"] is None
    assert unavailable["material_wavelength_angstrom"] == pytest.approx(4.)
    assert model.state.analysis.geometry.critical_angle_deg == 0
