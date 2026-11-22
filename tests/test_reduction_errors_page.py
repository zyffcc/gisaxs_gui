"""Reference and detector settings applied through Analyze's controls."""

from dataclasses import replace

import numpy as np
from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.shared.geometry.detector_response import DetectorResponse
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_assistant_calibration import SHAPE
from tests.test_review_analyze_run_and_clear import _frames, _profiles


def _activate(combo, key):
    combo.setCurrentIndex(combo.findData(key))
    combo.activated.emit(combo.currentIndex())


def test_load_edit_and_save_response_for_a_matched_profile(tmp_path, monkeypatch):
    profiles = _profiles()
    original = profiles.find("synthetic")
    profiles.save(replace(original, detector_response=DetectorResponse(2.5, 12, "scatter")))
    page = _page(_context(profiles))
    try:
        page.add_paths(_frames(tmp_path, 1))
        _done(page)
        assert page.use_profile_response_check.isChecked() and not page.gain_spin.isEnabled()
        assert page.error_model_combo.currentData() == "scatter" and page.gain_spin.value() == 2.5
        page.use_profile_response_check.setChecked(False)
        assert page.offset_spin.value() == 12 and page.gain_spin.isEnabled()
        page.gain_spin.setValue(3.5)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: ("", ""))
        _activate(page.error_model_combo, "sigma")
        assert page.error_model_combo.currentData() == "scatter"
        sigma = tmp_path / "uncertainty.npy"
        np.save(sigma, np.full(SHAPE, 4.0))
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *_: (str(sigma), ""))
        _activate(page.error_model_combo, "sigma")
        assert page.sigma_label.text() == sigma.name
        assert not page.pool_scatter_check.isEnabled() and not page.poisson_variance_combo.isEnabled()
        page.save_response_button.click()
        assert profiles.find("synthetic").detector_response == DetectorResponse(3.5, 12, "sigma", str(sigma))
        assert page.use_profile_response_check.isChecked() and page.gain_spin.value() == 3.5
        page._options_timer.stop()
        page.run_analysis()
        _done(page)
        assert page.view_model.state.analysis.corrections.sigma_path == str(sigma)
        apply_language("zh", [page])
        assert page.detector_errors_section.toggle_button.text() == "探测器与曲线误差"
        assert page.error_model_combo.currentText() == "指定 σ 帧"
    finally:
        apply_language("en")
        page.dispose()
        page.close()


def test_reference_and_curve_error_controls_follow_restored_settings():
    page = _page(_context(_profiles()))
    try:
        vm = page.view_model
        original = vm.state.corrections
        assert original.background_dark_scale is None and page.background_dark_auto.isChecked()
        assert page.pool_scatter_check.isChecked()
        page.background_dark_auto.setChecked(False)
        page.background_dark_spin.setValue(1.5)
        page.background_dark_check.setChecked(True)
        page.flat_dark_check.setChecked(True)
        page.flat_floor_spin.setValue(0.2)
        page.pool_scatter_check.setChecked(False)
        _activate(page.poisson_variance_combo, "smoothed")
        assert vm.state.corrections == replace(
            original, background_dark_scale=1.5, background_already_dark_subtracted=True, flat_field_floor=0.2,
            flat_already_dark_subtracted=True, pooled_scatter=False, poisson_variance="smoothed",
        )
        assert not page.background_dark_spin.isEnabled()
        vm.state.corrections = original
        page._sync_settings_widgets()
        assert page.background_dark_auto.isChecked() and page.background_dark_spin.isHidden()
        assert page.flat_floor_spin.value() == 0.05 and not page.flat_dark_check.isChecked()
        assert page.pool_scatter_check.isChecked() and page.poisson_variance_combo.currentData() == "observed"
    finally:
        page._options_timer.stop()
        page.dispose()
        page.close()
