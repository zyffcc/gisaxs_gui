"""First-time GIWAXS controls use the state the scientist sees on screen."""

import csv
import io
import json
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QAbstractButton, QDialog, QFileDialog, QLabel

from src.gimap.features.assistant.application.pipeline_report import pipeline_markdown
from src.gimap.features.assistant.presentation.guided_analysis import GuidedAnalysis
from src.gimap.features.assistant.presentation.guided_details import PeakDetails
from tests.test_analyze_workspace import _context, _done, _page
from tests.test_crystal_indexing import MAPI_CIF, PBI2_CIF
from tests.test_review_analyze_run_and_clear import _frames, _profiles


def test_load_several_candidate_cifs_and_hide_one_by_clicking_the_show_cell(tmp_path, monkeypatch):
    frame = _frames(tmp_path, 1)[0]
    paths = [tmp_path / name for name in ("first.cif", "invalid.cif", "second.cif")]
    # With no listed operations the second CIF has a long note about generated symmetry.
    cell, _operations, atoms = PBI2_CIF.split("\nloop_")
    for path, content in zip(paths, (MAPI_CIF, "", cell + "\nloop_" + atoms)):
        path.write_text(content, encoding="utf-8")
    starts = []

    def choose(parent, title, folder, filters):
        starts.append(Path(folder))
        return [str(path) for path in paths], filters

    monkeypatch.setattr(QFileDialog, "getOpenFileNames", choose)
    page = _page(_context(_profiles()))
    try:
        page.add_paths([str(frame)])
        _done(page)
        page.show()
        QTest.mouseClick(page.phases_button, Qt.LeftButton)
        assert page.phases_button.isChecked()
        dialog = page.findChild(QDialog, "analyzeIndexingWindow")
        QTest.mouseClick(dialog.add_cif_button, Qt.LeftButton)
        assert starts == [frame.parent] and dialog.phase_table.rowCount() == 2
        assert all(path.name in dialog.status_label.text() for path in paths)
        assert "from the space group" in dialog.symmetry_label.text()
        assert "generated from" in dialog.symmetry_label.toolTip()
        check = dialog.phase_table.cellWidget(0, 0)
        QTest.mouseClick(check, Qt.LeftButton, pos=check.rect().center())
        assert not page.indexing.phases[0].visible and page.indexing.phases[1].visible
        assert all(dialog.reflection_table.item(i, 0).text() == "PbI2" for i in range(dialog.reflection_table.rowCount()))
    finally:
        page.dispose()
        page.close()


def test_peak_export_writes_report_values_and_keeps_ambiguity(tmp_path, monkeypatch):
    path = tmp_path / "peaks.csv"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(path), "CSV (*.csv)"))
    guided = GuidedAnalysis(lambda: None, save_text=lambda name, text: Path(name).write_text(text, encoding="utf-8"))
    report = {
        "ok": True, "procedure": "giwaxs", "frame": str(tmp_path / "measurement.tif"),
        "analysis_record": {"geometry": {"distance_mm": 401}}, "peak_search": {"max_peaks": 5},
        "peaks": [
            {"q": 1.00000029, "d_A": 6.28, "fwhm": .015, "fit": {"area": 22, "q_err": .002},
             "phase": "Candidate, one", "hkl": "110", "ambiguous": True, "flags": ["poor_fit", "overlap"],
             "alternatives": [{"phase": "Other", "hkl": "002"}]},
            {"q": 2.1, "d_A": 2.99, "fwhm": .02, "phase": None, "flags": ["at_edge"]},
        ],
        "rings": [{"q": 1.00000029, "strongest_by_area": True, "reason": "Coverage is incomplete."}],
    }
    try:
        guided.report = report
        guided.results.show_report(report)
        button = guided.results.findChild(QAbstractButton, "guidedSavePeaks")
        labels = [x.text() for x in guided.results.findChildren(QLabel)]
        assert "Strongest reliable ring by fitted area" in labels
        assert any("Coverage is incomplete" in line for line in labels)
        section = guided.results.peak_details
        if not section.is_expanded():
            QTest.mouseClick(section.toggle_button, Qt.LeftButton)
        warning = section.findChild(PeakDetails).warnings.text()
        assert "poor fit" in warning and "overlaps a neighbouring peak" in warning
        QTest.mouseClick(section.toggle_button, Qt.LeftButton)
        QTest.mouseClick(button, Qt.LeftButton)
        contents = path.read_text(encoding="utf-8")
        assert json.dumps(report["frame"]) in contents and '"distance_mm": 401' in contents
        rows = list(csv.DictReader(io.StringIO("\n".join(s for s in contents.splitlines() if not s.startswith("#")))))
        assert len(rows) == 2 and float(rows[0]["q (A^-1)"]) == report["peaks"][0]["q"]
        assert rows[0]["phase"] == "Candidate, one" and rows[0]["hkl"] == "110"
        assert rows[0]["sigma_q (A^-1)"] == "0.002" and rows[0]["ambiguous"] == "True"
        assert json.loads(rows[0]["alternatives"]) == report["peaks"][0]["alternatives"]
        assert rows[1]["flags"] == "at_edge" and rows[1]["phase"] == ""
        guided.results.set_actions_enabled(False)
        assert not button.isEnabled()
    finally:
        guided.results.close()
        guided.controls.close()
        guided.progress_panel.close()


def test_orientation_checks_do_not_ask_for_an_unchanged_run():
    report = {"procedure": "giwaxs", "ok": True, "needs_attention": [
        {"item": "pole figure", "why": "Poor fit.", "option": None},
        {"item": "orientation factor", "why": "Limited angular coverage.", "option": None},
    ]}
    text = pipeline_markdown(report)
    assert "COMPLETED WITH CHECKS" in text and "## Orientation checks" in text and "Poor fit." in text
    assert "## 取向核查" in pipeline_markdown(report, language="zh")
    report["needs_attention"].append({"item": "incidence angle", "why": "Unknown.", "option": "incidence_deg"})
    text = pipeline_markdown(report)
    assert "NEEDS INPUT" in text and "--incidence-deg" in text
