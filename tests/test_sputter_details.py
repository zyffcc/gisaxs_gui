"""The Sputter page after the earlier method went: Details ▸ Windows, Report and Log about the result shown, the unit of
the deposition rate (with the question about an unusual rate and the nominal thickness at the last frame), the joint
fit of the whole scan as a solution, and the memory of an older page read without harm.

Offline, with a fake of the package's process and invented results (``tests/sputter_client_fixtures.py``).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QMessageBox

from src.gimap.integrations.state import InMemoryUserPreferencesRepository
from tests.sputter_client_fixtures import FakeClientRunner, entry, window, write_result
from tests.sputter_fixtures import scan_folder
from tests.test_sputter_main_screen import _close, _fill, _page, _settle


@pytest.fixture(autouse=True)
def _no_package_of_this_computer(monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_CLIENT", raising=False)


def _cells(table) -> list:
    return [[table.item(row, column).text() for column in range(table.columnCount())] for row in range(table.rowCount())]


def _headers(table) -> list:
    return [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())]


# -- Details ▸ Windows, Report and Log ------------------------------------------------------------------------------


def test_details_shows_the_windows_the_report_and_the_logs_of_the_result_shown(tmp_path):
    page = _page(tmp_path)
    page.show()
    page.show_details(True)
    assert [page.tabs.tabText(index) for index in range(page.tabs.count())] == ["Windows", "Report", "Logs"]
    for tab in (page.tab_windows, page.tab_report, page.tab_log):  # no result: one line says what to do
        assert tab.currentIndex() == 0 and tab.widget(0).text() == "Analyze a scan or open a result"
    out = write_result(tmp_path / "out")
    (out / "REPORT.md").write_text("# An invented report\n\nThe *invented* summary.\n", encoding="utf-8")
    (out / "analyze_predict.log").write_text("an invented line of the network\n", encoding="utf-8")
    (out / "analyze_intake.log").write_text("an invented line of the import\n", encoding="utf-8")
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    analysis.update(families_tried={"zz_family": 4.4, "zz_other_family": 6.25},
                    mask={"file": "zz_mask.npy", "automatic": False})
    (out / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    page.open_client_result(out)
    page.show_details(True)
    for tab in (page.tab_windows, page.tab_report, page.tab_log):
        assert tab.currentIndex() == 1
    # Windows: one row per window, the solution the main screen shows (its tiles only), with copy
    table = page.windows_table
    assert _headers(table) == ["Time (s)", "Nominal thickness (nm)", "Radius R (nm)", "Height H (nm)",
                               "Distance D (nm)", "Gap (nm)", "Size distribution width", "Coverage", "Fit deviation",
                               "Solution"]
    rows = _cells(table)
    assert rows[0] == ["2", "0.4", "1.5", "2.25", "6", "3", "0.21", "0.3", "4.4", "Candidate 2"]
    assert rows[2][0] == "6" and rows[2][-1] == "no solution: no frames in this time window" and rows[2][2] == ""
    assert rows[3][3] == ""  # a solution without a height
    page.entry_control.button(3).click()  # the refined solution on the main screen: the row follows
    assert _cells(table)[0][-1] == "Refined" and _cells(table)[0][-2] == "2.1"
    # Report: the facts of analysis.json above the report
    facts = page.report_facts.text()
    assert "Network used: zz_family" in facts and "Mode: quick" in facts and "Run time: 4 s" in facts
    assert "Networks tried (fit deviation): zz_family (4.4), zz_other_family (6.25)" in facts
    assert "Pixel mask: zz_mask.npy." in facts
    assert page.report_view.toPlainText().startswith("An invented report") and "invented" in page.report_view.toPlainText()
    # Log: every analyze_*.log, the steps' order first
    names = [page.log_choice.itemText(index) for index in range(page.log_choice.count())]
    assert names == ["analyze_intake.log", "analyze_predict.log"]
    assert page.log_view.toPlainText() == "an invented line of the import\n"
    page.log_choice.setCurrentIndex(1)
    assert page.log_view.toPlainText() == "an invented line of the network\n"
    page.show_client_result(None)
    assert page.tab_windows.currentIndex() == 0 and page.windows_table.rowCount() == 0
    _close(page)


def test_a_result_without_report_and_logs_says_so(tmp_path):
    page = _page(tmp_path)
    page.open_client_result(write_result(tmp_path / "out", report=False))
    assert page.report_view.toPlainText() == "This result has no report (REPORT.md)."
    assert page.log_view.toPlainText() == "This result has no log files." and not page.log_choice.isVisibleTo(page)
    _close(page)


def test_the_windows_table_can_be_copied(tmp_path):
    from src.gimap.app.presentation.components.table_copy import copy_rows

    page = _page(tmp_path)
    page.open_client_result(write_result(tmp_path / "out"))
    page.windows_table.selectRow(0)
    lines = copy_rows(page.windows_table).splitlines()
    assert lines[0].split("\t")[:3] == ["Time (s)", "Nominal thickness (nm)", "Radius R (nm)"]
    assert lines[1].split("\t")[:3] == ["2", "0.4", "1.5"] and len(lines) == 2
    _close(page)


def test_the_tabs_speak_chinese(tmp_path):
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

    page = _page(tmp_path)
    page.show()
    page.open_client_result(write_result(tmp_path / "out"))
    try:
        apply_language("zh", roots=(page,))
        page.refresh_language()
        assert [page.tabs.tabText(index) for index in range(3)] == ["时间窗", "报告", "日志"]
        assert _headers(page.windows_table)[-2:] == ["拟合偏差", "解"]
        assert _cells(page.windows_table)[0][-1] == "候选 2"
        assert "所用网络：zz_family" in page.report_facts.text() and "运行时间：4 s" in page.report_facts.text()
    finally:
        apply_language(DEFAULT_LANGUAGE, roots=(page,))
        page.refresh_language()
    assert _headers(page.windows_table)[-1] == "Solution"
    _close(page)


# -- the joint fit of the whole scan --------------------------------------------------------------------------------


def test_the_joint_fit_of_the_whole_scan_is_shown_first_and_named(tmp_path):
    windows = [window(t, [entry("candidate 1", 3.0, 1.0 + t, rank=1, weight=0.8),
                          entry("candidate 2", 5.0, 1.5 + t, rank=2, weight=0.2),
                          entry("refined", 2.5, 1.1 + t, kind="refined"),
                          entry("joint", 2.8, 1.2 + t, kind="joint")],
                   pick="refined", smooth="joint")
               for t in (2.0, 4.0)]
    page = _page(tmp_path)
    page.show()
    assert page.open_client_result(write_result(tmp_path / "out", windows=windows))
    first = page.client_result.windows[0]
    joint = first.entry("joint")
    assert (joint.kind, joint.weight, joint.rank) == ("joint", None, None)
    assert page.client_entry_label() == "joint" and page.entry_control.currentData() == "joint"
    names = [page.entry_control.itemText(index) for index in range(page.entry_control.count())]
    assert names == ["Candidate 1 · 3", "Candidate 2 · 5", "Refined · 2.5", "Joint fit (whole scan) · 2.8"]
    assert page.entry_control.button(3).toolTip() == "Joint fit (whole scan): fit deviation 2.8"  # no network weight
    assert page.patterns.titles[1] == "Model image: Joint fit (whole scan)"
    assert page.tile_values["R_nm"].text() == "3.2" and _cells(page.windows_table)[0][-1] == "Joint fit (whole scan)"
    page.entry_control.button(2).click()
    assert page.default_button.isVisible()  # the joint fit is the default: going back to it
    page.default_button.click()
    assert page.client_entry_label() == "joint"
    from src.gimap.app.presentation.i18n import translate

    assert translate("Joint fit (whole scan)", "zh") == "联合精修解"
    _close(page)


def test_a_preliminary_result_replaces_the_live_view_until_the_final_one(tmp_path):
    from src.gimap.features.sputter.application import LiveEvent

    page = _page(tmp_path)
    page.show()
    out = write_result(tmp_path / "out")
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    (out / "analysis.json").write_text(json.dumps(dict(analysis, status="preliminary")), encoding="utf-8")
    page._client_event(LiveEvent(kind="preliminary", step="predict", fields={"folder": str(out)}))
    assert page.client_result is not None and page.result_stack.currentIndex() == 1
    assert page.result_name.text().endswith("network result: the joint fit of the whole scan is running and replaces "
                                            "it when it is done")
    page._client_event(LiveEvent(kind="preliminary", step="predict", fields={"folder": str(tmp_path / "zz_none")}))
    assert page.client_result is not None  # a folder without a result leaves what is shown
    _close(page)


def test_a_result_whose_joint_fit_did_not_finish_says_so(tmp_path):
    page = _page(tmp_path)
    page.show()
    out = write_result(tmp_path / "out")
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    (out / "analysis.json").write_text(json.dumps(dict(analysis, joint_fit={"done": False, "workers": 2, "iterations": 6})),
                                       encoding="utf-8")
    assert page.open_client_result(out)
    assert page.client_result.joint_failed
    assert "the joint fit of the whole scan did not finish" in page.result_name.text()
    assert page.result_name.property("gimapRole") == "warning"
    good = write_result(tmp_path / "good")
    assert page.open_client_result(good)
    assert not page.client_result.joint_failed and "did not finish" not in page.result_name.text()
    assert page.result_name.property("gimapRole") == "muted"
    _close(page)


# -- the deposition rate and its unit -----------------------------------------------------------------------------------


def test_the_rate_is_taken_in_angstrom_per_second_unless_nm_per_second_is_chosen(tmp_path):
    runner = FakeClientRunner()
    preferences = InMemoryUserPreferencesRepository()
    page = _page(tmp_path, client=runner, preferences=preferences)
    assert page.rate_unit.currentData() == "A/s" and page.rate_unit.currentText() == "Å/s"  # what a QCM shows
    scan = scan_folder(tmp_path, "zz_invented_a")
    _fill(page, scan, rate="2.2", unit="A/s")
    page.analyze()
    _settle(lambda: not page.analyzing())
    assert runner.requests[0].rate_nm_s == 0.22  # carried as nm/s, without a rounding tail
    other = scan_folder(tmp_path, "zz_invented_b")
    _fill(page, other, rate="0.3", unit="nm/s")
    page.analyze()
    _settle(lambda: not page.analyzing())
    assert runner.requests[1].rate_nm_s == 0.3
    # each folder gets its own unit back with its rate; a new page starts with the unit used last
    page.set_scan_folder(scan)
    _settle(lambda: page.header_ready())
    assert (page.sample_rate.text(), page.rate_unit.currentData()) == ("2.2", "A/s")
    _close(page)
    fresh = _page(tmp_path, preferences=preferences)
    assert fresh.rate_unit.currentData() == "nm/s"
    _close(fresh)


def test_a_rate_an_older_page_remembered_is_in_nm_per_second(tmp_path):
    scan = scan_folder(tmp_path)
    from src.gimap.features.sputter.application import folder_key

    preferences = InMemoryUserPreferencesRepository(
        {"sputter_page": {"client_folders": {folder_key(scan): {"substrate": "PS", "rate": 0.25, "qcm": False}}}})
    page = _page(tmp_path, preferences=preferences)
    page.set_scan_folder(scan)
    _settle(lambda: page.header_ready())
    assert (page.sample_rate.text(), page.rate_unit.currentData()) == ("0.25", "nm/s")
    # a project of an older page, too
    state = page.project_state()
    state["client"].pop("unit")
    state["client"]["fields"]["sample_rate"] = "0.4"
    other = _page(tmp_path / "other")
    other.apply_project_state(state)
    assert (other.sample_rate.text(), other.rate_unit.currentData()) == ("0.4", "nm/s")
    _close(page)
    _close(other)


@pytest.mark.parametrize("rate, unit, qcm, asked", [("2.2", "nm/s", False, True), ("2.2", "A/s", False, False),
                                                     ("0.04", "A/s", False, True), ("1.5", "nm/s", True, False),
                                                     ("1.8", "nm/s", True, True)])
def test_an_unusual_rate_at_the_sample_asks_whether_its_unit_is_right(tmp_path, monkeypatch, rate, unit, qcm, asked):
    """Above 1 nm/s or below 0.005 nm/s at the sample (a QCM reading times 0.6): Analyze Anyway or Cancel."""
    said, answer = [], {"button": "Cancel"}

    def exec_(box):
        said.append((box.windowTitle(), box.text()))
        wanted = next(button for button in box.buttons() if answer["button"] in button.text())
        wanted.click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec_", exec_)
    runner = FakeClientRunner()
    page = _page(tmp_path, client=runner)
    _fill(page, scan_folder(tmp_path), rate=rate, unit=unit)
    page.qcm_check.setChecked(qcm)
    page.analyze()
    _settle(lambda: not page.analyzing())
    if not asked:
        assert said == [] and len(runner.requests) == 1
        _close(page)
        return
    # Qt on macOS ignores a message box's window title (Apple's guidelines), so the title reads empty there
    title = "" if sys.platform == "darwin" else "Check the Deposition Rate"
    assert len(said) == 1 and said[0][0] == title and runner.requests == []  # cancelled
    if (rate, unit, qcm) == ("2.2", "nm/s", False):
        assert said[0][1] == ("This rate is unusual for sputtering (2.2 nm/s = 22 Å/s at the sample). Is the unit "
                              "right?")
    answer["button"] = "Analyze Anyway"
    page.analyze()
    _settle(lambda: not page.analyzing())
    assert len(said) == 2 and len(runner.requests) == 1
    _close(page)


def test_the_scan_line_says_the_nominal_thickness_at_the_last_frame(tmp_path):
    """3 invented frames every 0.04 s: 0.12 s of deposition from the first frame."""
    page = _page(tmp_path)
    _fill(page, scan_folder(tmp_path), rate="5", unit="A/s")
    assert page.scan_summary.text().endswith(" · ≈ 0.06 nm deposited at the last frame (0.5 nm/s at the sample)")
    page.qcm_check.setChecked(True)
    assert "≈ 0.036 nm deposited at the last frame (0.3 nm/s at the sample)" in page.scan_summary.text()
    page.onset.setText("0.04")
    page.duration.setText("0.05")
    assert "≈ 0.015 nm deposited" in page.scan_summary.text()
    page.sample_rate.setText("")
    assert "deposited" not in page.scan_summary.text() and "3 frames" in page.scan_summary.text()
    _close(page)


# -- what an older page left ----------------------------------------------------------------------------------------------


OLDER_MEMORY = {"sputter_page": {
    "last_result": "C:/invented/zz_old_result", "python": "C:/invented/python.exe", "advanced_open": True,
    "results_root": "", "last": {"energy_kind": "energy", "energy": 9.5, "sdd": 1234.0, "alpha": 0.3, "metal": "Au",
                                 "polymer": "PS", "window": 0.5, "mask": ""},
    "folders": {"c:\\invented\\zz_scan": {"rate": 0.2, "t0": 1.0, "t_end": 8.0, "polymer": "PS", "sdd": 1234.0}}}}
"""The memory of the page with the earlier method (Predict): its result folder, its polymer, time window and rate."""


def test_the_memory_of_an_older_page_opens_and_its_predict_values_are_dropped(tmp_path):
    preferences = InMemoryUserPreferencesRepository(json.loads(json.dumps(OLDER_MEMORY)))
    runner = FakeClientRunner()
    page = _page(tmp_path, client=runner, preferences=preferences)
    page.show()
    assert page.client_result is None and page.sample_metal.currentData() == "Au" and page.sdd.text() == "1234"
    assert page.options_section.is_expanded()
    _fill(page, scan_folder(tmp_path))
    page.analyze()
    _settle(lambda: not page.analyzing())
    stored = preferences.get("sputter_page")
    assert "last_result" not in stored and "python" not in stored and len(runner.requests) == 1
    _close(page)


def test_an_older_project_opens_without_its_earlier_result(tmp_path):
    page = _page(tmp_path)
    older = {"fields": {"energy_value": "9", "sdd": "1000", "alpha": "0.3", "beam_col": "", "beam_row": "",
                        "roll": "", "rate": "0.2", "t0": "", "t_end": ""},
             "energy_kind": "energy", "metal": "Cu", "polymer": "PS", "window": 0.5, "scan_folder": "",
             "result_folder": str(tmp_path / "zz_no_such_result"), "variant": "physical", "log_scale": True}
    assert page.apply_project_state(older) == []
    assert page.sdd.text() == "1000" and page.sample_metal.currentData() == "Cu" and page.client_result is None
    _close(page)
