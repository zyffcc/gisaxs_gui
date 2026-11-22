"""The roll shown in Analyze survives file export, session restore and asynchronous measurement."""

from dataclasses import replace
from threading import Event, get_ident

import pytest

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application import analysis_metadata, read_poni_file, settings_from_record, settings_record
from src.gimap.features.analyze.presentation.bindings import sample_roll
from src.gimap.shared.geometry.sample_roll import RollEstimate, RollMeasurement
from tests.test_analyze_geometry_adjustment import page  # noqa: F401  (workspace with two memory-backed sources)


def test_measure_roll_on_worker_and_round_trip(page, monkeypatch, tmp_path):
    main = get_ident()
    result = RollMeasurement(RollEstimate(1.234, .12), RollEstimate(1.236, .1), 1.235619, .1, "combined", "specular rod")

    def measure(analysis, source):
        assert source is page.view_model.frames
        assert get_ident() != main and analysis is page.view_model.state.analysis
        return result

    monkeypatch.setattr(sample_roll, "measure_frame_roll", measure)
    page.roll_step_button.click()
    page.measure_roll_button.click()
    assert page.tasks.wait(10)
    model = page.view_model
    current = model.state.analysis
    assert current.geometry.sample_roll_deg == result.deg
    assert "1.24 ± 0.10°" in page.roll_estimate_label.text()
    assert current.sample_roll_measurement["method"] == "combined"
    assert analysis_metadata(current)["sample_roll_measurement"]["horizon"] == result.as_dict()["horizon"]
    saved = settings_from_record(settings_record(model.current_settings()))
    model.set_sample_roll(0)
    model.apply_settings(saved)
    assert model.state.sample_roll_measurement == current.sample_roll_measurement
    assert model.state.sample_roll_deg == result.deg
    assert all(req.sample_roll_deg == result.deg for req in model.batch_requests())
    written = page.export_poni(str(tmp_path / "example.poni"))
    poni = read_poni_file(written)
    assert poni.geometry().sample_roll_deg == result.deg
    assert poni.sample_roll_measurement == current.sample_roll_measurement
    page._enter_geometry()
    assert page._geometry_dialog.geometry().sample_roll_deg == result.deg
    try:
        apply_language("zh", [page])
        page.refresh_language()
        assert "镜面杆" in page.roll_estimate_label.text()
        assert page.measure_roll_button.text() == "从当前帧测量"
    finally:
        apply_language("en", [page])


def test_roll_measurement_finishing_after_edit_is_dropped(page, monkeypatch):
    started, release = Event(), Event()
    def measure(_analysis, _source):
        started.set()
        assert release.wait(5)
        return RollMeasurement(RollEstimate(1., .1), None, 1., .1, "horizon")
    monkeypatch.setattr(sample_roll, "measure_frame_roll", measure)
    page.measure_roll_button.click()
    try:
        assert started.wait(2)
        page.sample_roll_spin.setValue(-2.0)
    finally:
        release.set()
    assert page.tasks.wait(10)
    assert page.view_model.state.analysis.geometry.sample_roll_deg == -2.
    assert page.view_model.state.sample_roll_measurement is None


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
