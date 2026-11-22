"""The displayed standard fit, sample thickness and saved scale in the Analyze dialog."""

from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt5.QtWidgets import QDialog

from src.gimap.app.presentation.i18n import apply_language
from src.gimap.features.analyze.application.absolute_scale import CalibrateStandard
from src.gimap.features.analyze.domain import Corrections
from src.gimap.features.analyze.presentation.absolute_scale_dialog import AbsoluteScaleDialog
from src.gimap.shared.geometry.detector_geometry import DetectorGeometry


def test_curve_fit_then_apply_with_the_sample_thickness(tmp_path):
    path = tmp_path / "standard.dat"
    np.savetxt(path, [[0.1, 1500], [0.2, 1501], [0.3, 1499], [0.4, 1500]])
    applied = []
    geometry = DetectorGeometry(1e-4, 1e-4, 2.0, 128.5, 128.5, 1.0)
    worker = CalibrateStandard(None, SimpleNamespace(name="Bench"), geometry, None, Corrections(), "Transmission")
    dialog = AbsoluteScaleDialog(worker, lambda result, d: applied.append((result, d)), profile_name="Bench")
    dialog.show()
    dialog.source_kind.setCurrentIndex(1)
    dialog.source_path.setText(str(path))
    dialog.source_unit.setCurrentText("nm⁻¹")
    dialog.std_thickness.setText("2")
    dialog.fit_button.click()
    assert dialog.tasks.wait()
    assert dialog.calibration is not None, dialog.feedback.text()
    assert dialog.calibration.fit.q == pytest.approx([.01, .02, .03, .04])
    assert dialog.calibration.fit.factor == pytest.approx(.01632 / (1500 / .2))
    assert len(dialog.overlay.plot.listDataItems()) == 2
    assert len(dialog.residual.plot.listDataItems()) == 2
    dialog.q_min.setText(".015")
    assert not dialog.apply_button.isEnabled()
    assert not dialog.overlay.plot.listDataItems()
    dialog.q_max.setText(".035")
    dialog.fit_button.click()
    assert dialog.tasks.wait()
    dialog.apply_button.click()
    assert not applied
    assert "sample thickness" in dialog.feedback.text()
    dialog.sample_thickness.setText("0.0000015")
    dialog.apply_button.click()
    assert "six decimal places" in dialog.feedback.text()
    assert not applied
    dialog.sample_thickness.setText("0.45")
    dialog.apply_button.click()
    assert dialog.result() == QDialog.Accepted
    (result, thickness), = applied
    assert thickness == 0.45
    assert result.fit.q == pytest.approx([.02, .03])


def test_changing_a_fit_setting_discards_an_in_flight_result(qt_application):
    started, finish = Event(), Event()
    stopped = []

    def wait_for_edit(_request, *, cancelled):
        started.set()
        finish.wait(5)
        stopped.append(cancelled())
        raise ValueError("stale error")

    dialog = AbsoluteScaleDialog(wait_for_edit, None)
    dialog.source_path.setText("invented-standard.cbf")
    dialog.std_thickness.setText("1")
    dialog.fit_button.click()
    try:
        assert started.wait(3)
        dialog.frame_number.setValue(2)
    finally:
        finish.set()
        assert dialog.tasks.wait()
    assert stopped == [True]
    assert not dialog.apply_button.isEnabled()
    assert not dialog.feedback.text()
    dialog.reject()


def test_saved_standard_can_be_reused_without_refitting(qt_application):
    received = []
    saved = {"factor": 2.5e-6, "sigma_factor": 1e-8, "date_utc": "2025-01-01", "standard": "Water"}
    dialog = AbsoluteScaleDialog(None, None, profile_name="Synthetic instrument", saved_record=saved,
                                 use_saved=received.append, sample_thickness_mm=.7)
    dialog.show()
    try:
        apply_language("zh", [dialog])
        assert dialog.saved_button.text() == "使用已保存的比例因子"
        assert "水" in dialog.saved_label.text()
        assert dialog.profile_label.text() == "Synthetic instrument"
        dialog.saved_button.click()
        assert received == [.7]
        assert dialog.result() == QDialog.Accepted
    finally:
        apply_language("en")
        dialog.close()
