"""Index a synthetic detector stack, track its cell through a missing frame, and export from Analyze."""

import csv
import json

import numpy as np
from PyQt5.QtWidgets import QFileDialog

from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_indexing_panel import _close
from tools.benchmarks.cell_series import make_stack


def test_tracked_singleton_is_excluded_when_later_cell_brings_in_another_family():
    from src.gimap.features.analyze.application.cell_series import _fit_tracks
    from src.gimap.features.analyze.application.indexing import PhaseSpec
    from src.gimap.features.analyze.domain.crystal.refinement import CellPeak

    spec = PhaseSpec("Evolving cell", (6., 6., 8.40, 90., 90., 90.))
    indexed = {spec.name: {"system": "tetragonal"}}
    hkls = [(0,0,1), (1,0,0), (1,0,1), (1,1,1), (2,1,0), (1,0,2), (1,1,0)]
    def radius(h, k, l):
        return 2*np.pi*np.sqrt((h*h+k*k)/6.018**2 + l*l/8.485**2) + .0015

    centres = [radius(*h) for h in hkls]
    centres[-1] = .5*(centres[-1] + radius(0,0,2))
    peaks = [CellPeak(h, q, .0003) for h, q in zip(hkls, centres)]
    tracks = [dict(phase=spec.name, hkl=list(p.hkl), q=p.q, sigma=p.sigma, fwhm=.024) for p in peaks]
    fit = _fit_tracks([spec], indexed, {spec.name: peaks}, tracks, lambda: None)[spec.name]
    assert fit["ok"] and len(fit["residuals"]) == 6
    assert {entry["hkl"] for entry in fit["excluded"][0]["candidates"]} == {"110", "002"}
    np.testing.assert_allclose([fit["values"]["a"], fit["values"]["c"]], [6.018, 8.485])


def test_indexed_detector_series_recovers_cell_and_keeps_missing_frame(tmp_path, monkeypatch):
    path, times, a, c, profile, phase, _, _ = make_stack(tmp_path, blank=5)
    page = _page(_context(InMemoryInstrumentProfileRepository([profile])))
    try:
        page.set_mode_choice("giwaxs")
        page.add_paths([str(path)])
        _done(page)
        panel = page.open_indexing()
        panel.model.add(phase)
        page.indexing_changed()
        panel.tabs.setCurrentWidget(panel.cell_tab)
        _done(page)
        assert panel.cell_series_button.isEnabled(), panel.cell_summary.text()
        cell_path = tmp_path / "frame_cells.json"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(cell_path), ""))
        panel.cell_export_button.click()
        single = json.loads(cell_path.read_text(encoding="utf-8"))
        assert single["analysis"]["source_file"] == str(path)
        assert single["cell_refinement"][0]["covariance"] and single["settings"]
        panel.cell_series_button.click()
        _done(page)
        dialog = panel._cell_series_dialog
        record = dialog._record
        assert record and record["time_from"] == "timestamps"
        assert record["indexing"]["analysis"]["source_file"] == str(path)
        fits = [r["fits"][phase.name] for r in record["frames"]]
        np.testing.assert_equal([r["time_s"] for r in record["frames"]], times)
        assert not fits[5]["ok"] and fits[5]["reason"]
        valid = np.arange(len(fits)) != 5
        for key, truth in (("a", a), ("c", c), ("V", a*a*c)):
            actual = [fit["values"][key] for fit in fits if fit["ok"]]
            np.testing.assert_allclose(actual, truth[valid], rtol=.002)
            assert all(fit["errors"][key] > 0 for fit in fits if fit["ok"])
        assert len(dialog.length_plot._extras) == len(dialog.volume_plot._extras) == 1
        dialog.parameter_combo.setCurrentIndex(dialog.parameter_combo.findData("c"))
        assert dialog.length_plot.figure_state()["curves"][0][0] == "c"
        out = tmp_path / "tracked.csv"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_: (str(out), ""))
        dialog.export_button.click()
        rows = list(csv.DictReader(line for line in out.read_text(encoding="utf-8").splitlines()
                                   if not line.startswith("#")))
        assert rows[5]["V_A3"] == "" and rows[5]["reason"]
        assert float(rows[6]["sigma_a_A"]) > 0 and rows[6]["frame_index"] == "6"
        companion = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
        assert companion["frames"][6]["analysis"]["input_sha256"]
        assert companion["frames"][6]["fits"][phase.name]["residuals"]
        dialog.close()
    finally:
        _close(page)
