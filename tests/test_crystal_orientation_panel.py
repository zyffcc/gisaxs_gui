"""The phase fit in the Pole Figure tab, including its curves, language and export."""

import csv
import json
from threading import Event

from PyQt5.QtWidgets import QFileDialog

from src.gimap.app.presentation.i18n import apply_language
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_indexing_panel import _close
from tests.test_review_analyze_run_and_clear import _frames, _profiles


def test_phase_model_shows_joint_fit_and_reflection_curves(tmp_path, monkeypatch):
    from src.gimap.features.analyze.application import crystal_orientation

    app = _page(_context(_profiles()))
    cat = app.automation()
    release_worker = Event()
    try:
        app.set_mode_choice("giwaxs")
        app.add_paths([str(_frames(tmp_path, 1)[0])])
        _done(app)
        panel = app.open_indexing()
        combo = panel.model_combo
        assert combo.findData("crystal_orientation") == -1
        cat.set_phases([{"name": "Test crystal", "cell": (5, 5, 8)}])
        combo.setCurrentIndex(combo.findData("crystal_orientation"))
        assert panel.fit_button.text() == "Fit Crystal Orientation"
        assert panel.direction_combo.isHidden() and panel.background_spin.isHidden()

        def wait_for_cancel(analysis, phases, *, chi_max, frame_source, check_cancel):
            assert release_worker.wait(10)
            check_cancel()
            raise AssertionError("A cancelled fit should stop.")

        monkeypatch.setattr(crystal_orientation, "frame_crystal_orientation", wait_for_cancel)
        panel.fit_button.click()
        panel.cancel_crystal_button.click()
        release_worker.set()
        _done(app)
        assert panel.result_label.text() == "Cancelled"
        assert not panel.export_button.isEnabled()

        ring = {"q": 1.1, "q_window": [1.08, 1.12], "chi_deg": [20., 21., 22., 23.],
                "fitted": [4., 5., 6., 7.], "observed": [4.2, 5., None, 6.7], "sigma": [1., 1., None, 1.],
                "measured": [True, True, False, True], "reduced_chi2": .99, "herman_measured": .31,
                "components": [{"hkl": "110", "weight": .4, "values": [1., 2., 3., 4.]},
                               {"hkl": "002", "weight": .6, "values": [3., 3., 3., 3.]}]}
        phase = {"phase": "Test crystal", "axes": [[1, 1, 0], [1, 1, 0]], "determined": True,
                 "populations": [
                     {"axis": [1, 1, 0], "fwhm_deg": 25., "fwhm_err_deg": .4, "fraction": .56, "fraction_err": .03},
                     {"axis": [1, 1, 0], "fwhm_deg": 6., "fwhm_err_deg": .2, "fraction": .14, "fraction_err": .02}],
                 "runner_up_axes": [[0, 0, 1]], "delta_bic": 15., "reduced_chi2": 1., "reasons": [],
                 "ok": True, "family_weights": "kinematic", "fraction_iso": .3, "fraction_iso_err": .04,
                 "film_optics": {"available": True, "delta": 3.1e-6, "beta": 4.3e-7,
                                 "model": "refracted waves", "depth_source": "thick limit"},
                 "oriented_fraction": .7, "oriented_fraction_err": .04, "herman_crystal": .65,
                 "rings": [ring, dict(ring, q=2., q_window=[1.98, 2.02], fit_mask=[True, True, False, False])]}
        monkeypatch.setattr(crystal_orientation, "frame_crystal_orientation", lambda *_a, **_kw: [phase])
        panel.fit_button.click()
        _done(app)
        assert panel.crystal_ring_combo.count() == 2 and panel.export_button.isEnabled()
        assert "1.100" in panel.crystal_ring_combo.currentText()
        curves = panel.pole_plot.figure_state()["curves"]
        assert [curve[0] for curve in curves] == ["Measured N(χ)", "Joint fit", "110 (40.0%)", "002 (60.0%)"]
        assert list(curves[0][1]) == [20., 21., 23.]
        assert "Orientation determined" in panel.result_label.text()
        text = panel.result_label.text()
        for width, fraction in (("25.00 ± 0.40", "56.0 ± 3.0"), ("6.00 ± 0.20", "14.0 ± 2.0")):
            assert f"(1 1 0) ∥ substrate · mosaic FWHM {width}° · {fraction}% of phase" in text
        assert "Total oriented 70.0 ± 4.0%" in text
        assert "Refracted film waves · δ 3.1e-06 · β 4.3e-07" in text
        from src.gimap.features.assistant.application.pipeline_crystal import crystal_lines

        report = "\n".join(crystal_lines([phase]))
        assert "refraction and attenuation of the incident and exit waves" in report
        assert "| (1 1 0) | 25.00(40) | 0.560(30) |" in report
        assert "| (1 1 0) | 6.00(20) | 0.140(20) |" in report
        from src.gimap.features.analyze.presentation.crystal_orientation_panel import _summary

        conditional = dict(phase, determined=False, reasons=["joint or ring reduced χ² exceeds 5"])
        assert "conditional on the preferred candidate" in _summary(conditional, ring)
        assert "conditional on the preferred candidate" in "\n".join(crystal_lines([conditional]))
        panel.crystal_ring_combo.setCurrentIndex(1)
        assert "1.980" in panel.pole_plot.figure_state()["title"]
        fitted, excluded = panel.pole_plot.figure_state()["curves"][:2]
        assert list(fitted[1]) == [20., 21.]
        assert excluded[0] == "not fitted" and list(excluded[1]) == [23.] and list(excluded[2]) == [6.7]
        target = tmp_path / "result_crystal_orientation.csv"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
        panel.export_button.click()
        lines = target.read_text(encoding="utf-8").splitlines()
        rows = list(csv.reader(line for line in lines if not line.startswith("#")))
        assert rows[0][-2:] == ["110", "002"] and rows[3][1:3] == ["", ""]
        assert rows[0][3:5] == ["measured", "fit_mask"] and rows[4][3:5] == ["True", "False"]
        record = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
        assert record["crystal_orientation"] == [phase] and record["analysis"]["geometry"]
        apply_language("zh", [panel])
        assert "已确定取向" in panel.result_label.text() and "晶体取向（依据物相）" == combo.currentText()
        assert "薄膜内的折射波" in panel.result_label.text()
        assert "镶嵌展宽（半高宽）6.00 ± 0.20° · 占该物相 14.0 ± 2.0%" in panel.result_label.text()
        panel.chi_max_spin.setValue(75.)
        assert not panel.export_button.isEnabled() and not panel.crystal_ring_combo.count()
        cat.set_phases([])
        assert combo.findData("crystal_orientation") == -1 and not panel.direction_combo.isHidden()
    finally:
        release_worker.set()
        apply_language("en")
        _close(app)
