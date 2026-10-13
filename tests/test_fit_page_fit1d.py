"""The actual single-curve page uses the v2 engine and keeps its scientific result intact."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
from PyQt5.QtWidgets import QApplication

from src.gimap.app.jobs import JobError, JobResult
from src.gimap.features.fitting.application.single_fit import FitModel, evaluate, new_component
from src.gimap.features.fitting.application.workflow_v5 import fit1d_kit_home
from tests.test_fit_page import _page, _wait

needs_kit = pytest.mark.skipif(
    not (fit1d_kit_home() / "gisaxs1d_kit" / "core.py").is_file(),
    reason="the gisaxs1d kit (GISAXS1D_KIT_HOME or utils/gisaxs1d_gui_kit) is not installed",
)


def _synthetic_file(tmp_path, *, sigma=True):
    q = np.linspace(0.06, 1.8, 48)
    y = 5.0 + 300.0 * np.exp(-1.4 * q)
    errors = np.sqrt(y)
    path = tmp_path / "synthetic.dat"
    np.savetxt(path, np.column_stack([q / 10, y] + ([errors] if sigma else [])))
    return path


@needs_kit
@pytest.mark.parametrize("network,mode,count", [("general", "gisaxs", 1), ("au_np_si", "gisaxs", 2),
                                              ("general", "saxs", 3), ("au_np_si", "saxs", 4)])
def test_main_fit_runs_without_tf_and_preserves_native_result(tmp_path, monkeypatch, network, mode, count):
    page = _page()
    try:
        assert page.open_curve(_synthetic_file(tmp_path, sigma=count != 3)), page.status_label.text()
        families = ["sphere", "vertical_cylinder", "sphere", "vertical_cylinder"][:count]
        page.set_model(FitModel(tuple(new_component(family) for family in families)))
        page.method_buttons["fit1d"].setChecked(True)
        page.fit1d_network.setCurrentIndex(page.fit1d_network.findData(network))
        page.fit1d_mode.setCurrentIndex(page.fit1d_mode.findData(mode))
        page.fit1d_policy.setCurrentIndex(page.fit1d_policy.findData("fast"))
        page.fit1d_use_qz.setChecked(count != 2)
        page.fit1d_qz.setValue(0.42)
        page.fit1d_sigma_floor.setValue(3.0)
        assert "ln I" not in page.sigma_note.text()
        if count == 3:
            assert "Poisson errors" in page.sigma_note.text()
        # This path must not even ask for the old V5 model directory.
        monkeypatch.setattr("src.gimap.features.fitting.presentation.single.run.bundled_workflow",
                            lambda: pytest.fail("fit1d must not require V5/TensorFlow assets"))
        page.run_fit()
        _wait(page)
        native = page.session.native_result
        assert native is not None, page.status_label.text()
        assert page.session.native_is_current() and page.session.result is None
        assert native["summary"]["model"] == network
        assert native["summary"]["mode"] == mode
        assert native["summary"]["qz_nm_inv"] == pytest.approx(0.42 if mode == "gisaxs" and count != 2 else 0)
        if count == 2:
            assert any("No qz" in warning for warning in native["warnings"])
        if count == 3:
            assert any("Poisson errors" in warning for warning in native["warnings"])
            assert not any("Relative σ" in warning for warning in native["warnings"])
        assert len(native["populations"]) == count
        assert native["conventions"]["physics"].startswith("v2")
        curves = native["curves"]
        q, y = np.asarray(curves["q_nm_inv"]), np.asarray(curves["intensity"])
        assert q == pytest.approx(np.linspace(0.06, 1.8, 48))
        assert curves["sigma"] == pytest.approx(np.hypot(np.sqrt(y), 0.03 * y))
        terms = sum(np.asarray(term) for term in curves["components"].values())
        assert terms == pytest.approx(curves["model"])
        # The mapped manual model is an initial guess; its forward curve must never replace v2.
        assert not np.allclose(evaluate(page.session.model, q), curves["model"], rtol=1e-4)
        page.terms_check.setChecked(True)
        shown = page.plot._source  # the scientific arrays supplied to the plot, before logarithmic display
        assert len(shown) >= count + 2
        assert any(np.allclose(np.asarray(item[2]), curves["model"]) for item in shown if len(item[2]) == len(q))
        assert page.parameters_table.rowCount() == len(native["table"])
        assert "χ²/point" in page.quality_label.text()
        assert "starting point" in page.warnings_label.text()
        output = tmp_path / "native_fit.csv"
        assert page.export_data_dialog(output) == output
        exported = np.loadtxt(output, delimiter=",", skiprows=1)
        assert exported[:, 3] == pytest.approx(curves["sigma"], rel=1e-7)
        assert exported[:, 4] == pytest.approx(curves["model"], rel=1e-7)
        assert exported[:, 5] == pytest.approx(curves["residual"], abs=1e-7)
        record = json.loads(output.with_suffix(".json").read_text())
        assert record["backend"] == "fit1d_physics_v2"
        assert record["native_result"]["warnings"] == native["warnings"]
        mapped = page.session.model
        page.set_model(mapped.with_values({(0, "R"): mapped.components[0].value("R") * 1.02}))
        assert not page.session.native_is_current()
        assert "manual starting model" in page.warnings_label.text()
        assert "previous fit1d result" in page.quality_label.text()
        assert "original fit1d v2 curve is shown" not in page.warnings_label.text()
        page.undo()
        assert page.session.native_is_current()
        page.fit1d_sigma_floor.setValue(4.0)
        assert not page.session.native_is_current()
        assert page.export_data_dialog(tmp_path / "changed_floor.csv") is None
        page.fit1d_sigma_floor.setValue(3.0)
        assert page.session.native_is_current()
        for widget in (page.fit1d_mode, page.fit1d_network, page.fit1d_policy):
            original = widget.currentIndex()
            widget.setCurrentIndex((original + 1) % widget.count())
            assert not page.session.native_is_current()
            assert page.export_data_dialog(tmp_path / "changed_settings.csv") is None
            widget.setCurrentIndex(original)
            assert page.session.native_is_current()
        checked = page.fit1d_use_qz.isChecked()
        page.fit1d_use_qz.setChecked(not checked)
        assert not page.session.native_is_current()
        assert page.export_data_dialog(tmp_path / "changed_qz.csv") is None
        page.fit1d_use_qz.setChecked(checked)
        assert page.session.native_is_current()
        page.set_range((0.2, 1.0))
        assert not page.session.native_is_current()
        assert page.export_data_dialog(tmp_path / "stale.csv") is None
    finally:
        page.dispose()


def test_fit1d_rejects_unknown_families_and_five_populations(tmp_path):
    page = _page()
    try:
        page.open_curve(_synthetic_file(tmp_path))
        for families in (["cylinder"], ["sphere"] * 5):
            page.set_model(FitModel(tuple(new_component(f) for f in families)))
            page.method_buttons["fit1d"].setChecked(True)
            page.run_fit()
            assert not page.fit_running() and page.session.native_result is None
            assert "1–4 particles" in page.status_label.text()
    finally:
        page.dispose()


def test_stop_reaches_jobrunner_without_a_new_result(tmp_path):
    class BlockingRunner:
        def __init__(self):
            self.started = threading.Event()
            self.stopped = threading.Event()
            self.request = None

        def run(self, request, on_progress):
            self.request = request
            self.started.set()
            assert self.stopped.wait(5), "Stop did not reach JobRunner"
            return JobResult(request.job_id, "cancelled", error=JobError("cancelled", "Cancelled"))

        def cancel(self, job_id):
            assert self.request.job_id == job_id
            self.stopped.set()
            return True

    page = _page()
    runner = BlockingRunner()
    page.view_model.context.jobs = runner
    try:
        page.open_curve(_synthetic_file(tmp_path))
        page.method_buttons["fit1d"].setChecked(True)
        page.run_fit()
        end = time.monotonic() + 5
        while not runner.started.is_set() and time.monotonic() < end:
            QApplication.processEvents()
            time.sleep(0.01)
        assert runner.started.is_set()
        page.stop_fit()
        _wait(page)
        assert runner.stopped.is_set() and page.session.native_result is None
        assert "Stopped" in page.status_label.text()
    finally:
        page.dispose()
