"""Index a synthetic detector stack, track its cell through a missing frame, and export from Analyze."""

import csv
import json

import numpy as np
from PyQt5.QtWidgets import QFileDialog

from src.gimap.integrations.state import InMemoryInstrumentProfileRepository
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_indexing_panel import _close
from tools.benchmarks.cell_series import make_stack


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
