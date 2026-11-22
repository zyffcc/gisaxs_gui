"""Calibrate through Analyze, reuse the profile and export sample intensities and provenance."""

import json
from dataclasses import replace

import h5py
import numpy as np
import pytest

from src.gimap.features.analyze.application import Normalisation, settings_from_record, settings_record
from src.gimap.features.analyze.application.absolute_scale import StandardRequest
from src.gimap.features.analyze.application.series_map import build_series_map
from src.gimap.features.analyze.bootstrap import create_analyze_view_model
from src.gimap.features.analyze.presentation.absolute_scale_dialog import AbsoluteScaleDialog
from src.gimap.integrations.state import JsonInstrumentProfileRepository
from src.gimap.shared.geometry import InstrumentProfile
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_normalisation import GISAXS_GEOMETRY, GISAXS_SHAPE, _counts_frame


def test_profile_selected_without_an_open_sample_and_failed_save(tmp_path, monkeypatch):
    repo = JsonInstrumentProfileRepository(tmp_path / "profiles.json")
    profile = InstrumentProfile("Bench", GISAXS_GEOMETRY)
    repo.save(profile)
    vm = create_analyze_view_model(_context(repo))
    vm.set_profile(profile.name)
    source = tmp_path / "standard.txt"
    np.savetxt(source, [[0.1, 324], [0.2, 326]])
    result = vm.standard_calibrator()(StandardRequest("curve", source, thickness_mm=.8))
    prior = vm.state.corrections.normalisation

    def failed_save(_profile):
        raise OSError("disk full")

    with monkeypatch.context() as failure:
        failure.setattr(repo, "save", failed_save)
        with pytest.raises(OSError, match="disk full"):
            vm.apply_standard_calibration(profile, result.record, 2, save=True)
    assert vm.state.corrections.normalisation == prior
    assert repo.find(profile.name).standard_calibration is None
    vm.apply_standard_calibration(profile, result.record, 2, save=True)
    saved = repo.find(profile.name)
    assert saved.standard_calibration == result.record
    # A change in wavelength invalidates the stored intensity calibration.
    assert saved.updated(replace(profile.geometry, wavelength_angstrom=1.25), source="new energy").standard_calibration is None


def test_standard_to_profile_sample_and_exports(tmp_path, monkeypatch):
    repo = JsonInstrumentProfileRepository(tmp_path / "profiles.json")
    profile = InstrumentProfile("Synthetic bench", GISAXS_GEOMETRY, detector_shape=GISAXS_SHAPE)
    repo.save(profile)
    sample = _counts_frame(tmp_path)
    source = tmp_path / "synthetic_water.dat"
    np.savetxt(source, [[.02, 410], [.04, 400], [.06, 390]])
    page = _page(_context(repo))
    vm = page.view_model
    try:
        page.add_paths([str(sample)])
        _done(page)
        plain = vm.state.analysis
        vm.set_normalisation(Normalisation(exposure="value", exposure_s=4, transmission=.5))
        page._sync_normalisation_widgets()

        def fit_and_apply(dialog):
            fit = dialog.preview(StandardRequest("curve", source, exposure_s=2, thickness_mm=5))
            dialog.apply(fit, 2.5)
            dialog.accept()

        monkeypatch.setattr(AbsoluteScaleDialog, "exec_", fit_and_apply)
        page.absolute_scale_dialog()
        page._options_timer.stop()
        page.run_analysis()
        _done(page)
        calibrated = vm.state.analysis
        stored = repo.find(profile.name).standard_calibration
        factor = .01632 / 400
        assert stored["factor"] == pytest.approx(factor)
        assert stored["standard"] == "Water" and stored["source_path"] == str(source)
        assert stored["date_utc"] and stored["sigma_factor"] > 0
        assert stored["q_range_A_inv"] == [.02, .06]
        assert calibrated.normalisation.unit == "cm⁻¹"
        assert page._read_normalisation().standard_calibration == stored
        c = factor / (4 * .5 * .25)
        before, after = (a.reduction.curve("horizontal") for a in (plain, calibrated))
        np.testing.assert_allclose(after.intensity, c * before.intensity, rtol=1e-6)
        np.testing.assert_allclose(after.sigma, c * before.sigma, rtol=1e-6)
        restored = settings_from_record(json.loads(json.dumps(settings_record(vm.current_settings()))))
        assert restored.corrections.normalisation.standard_calibration == stored
        out = tmp_path / "out"
        vm._export(calibrated, out, nxcansas=True)
        norm = json.loads((out / "film_001_analysis.json").read_text(encoding="utf-8"))["normalisation"]
        assert norm["standard_calibration"] == stored
        assert norm["thickness_cm"]["value"] == .25
        assert norm["absolute_factor"] == {"value": stored["factor"], "source": "secondary standard"}
        assert "I (cm^-1)" in (out / "film_001_horizontal.csv").read_text(encoding="utf-8")
        row = (after.x, after.intensity, after.x_label, sample.name, (sample, 0))
        series = build_series_map([row], "horizontal", built_with={"settings": settings_record(vm.current_settings())})
        csv = vm.export_series_map(series, out / "series.csv")
        assert "# curve intensity: cm^-1" in csv.read_text(encoding="utf-8")
        with h5py.File(out / "film_001_nxcansas.h5") as file:
            units = []
            file.visititems(lambda k, v: units.append(v.attrs["units"]) if k.endswith("/I") else None)
            assert units and all(unit in ("1/cm", b"1/cm") for unit in units)
        with pytest.raises(ValueError, match="another instrument set-up"):
            vm.analyze(replace(vm.request(), distance_m=GISAXS_GEOMETRY.distance_m * 2))

        # Temporarily switching A off does not reinterpret the same displayed A in millimetres.
        page.absolute_check.setChecked(False)
        page.absolute_check.setChecked(True)
        assert page._read_normalisation().standard_calibration == stored
        page.absolute_edit.setText("3")
        page.absolute_edit.editingFinished.emit()
        assert vm.state.corrections.normalisation.standard_calibration is None

        def reuse(dialog):
            assert dialog.saved_record == stored
            dialog.use_saved(1.0)
            dialog.accept()

        monkeypatch.setattr(AbsoluteScaleDialog, "exec_", reuse)
        page.absolute_scale_dialog()
        assert vm.state.corrections.normalisation.thickness_mm == 1.0
        assert vm.state.corrections.normalisation.standard_calibration == stored
        repo.save(replace(profile, name="Second bench"))
        vm.set_profile("Second bench")
        assert vm.state.corrections.normalisation.absolute_factor is None
    finally:
        page._options_timer.stop()
        page.dispose()
        page.close()
