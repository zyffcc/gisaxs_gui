"""Sputter ▸ Compare… (Compare Scans): several Analyze results side by side against the nominal deposit.

Offline, with invented results (``tests/sputter_client_fixtures.py``): two scans of one metal on two invented polymers,
sputtered at different rates, each with a process analysis and a report table of model-free readings.
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QFileDialog

from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.features.sputter.domain.client import client_result
from src.gimap.features.sputter.domain.compare import (ComparedScan, compare_csv, curves, default_name, growth_rows,
                                                       report_readings, series)
from tests.sputter_client_fixtures import FakeClientRunner, entry, window, write_result
from tests.sputter_fixtures import FakeInspector


@pytest.fixture(autouse=True)
def _no_package_of_this_computer(monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_CLIENT", raising=False)
    monkeypatch.setenv("GIMAP_SPUTTER_NO_INSTALL", "1")


REPORT_HEAD = ["t", "t_rel", "h_nom", "stage", "stage_source", "fit_dev5", "fit_grade"] + \
    [f"{key}{suffix}" for key in ("d_film_nm", "h_grain_nm", "yoneda_rel", "L_lateral_nm")
     for suffix in ("", "_low", "_high", "_source")]


def _report(rows) -> str:
    """report.csv text: rows of (t, h_nom, d_film, d_source, yoneda_rel)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(REPORT_HEAD)
    for t, h_nom, film, source, yoneda in rows:
        line = [t, t - 0.5, h_nom, "islands", "reading", 3.0, "good"]
        line += [film, "", "", source] + ["", "", "", ""] + [yoneda, "", "", "reading"] + ["", "", "", ""]
        writer.writerow(line)
    return out.getvalue()


def _scan(folder: Path, *, rate: float, polymer: str, radius: float, film_from: float, stamp: float) -> Path:
    """An invented result: three windows at 2, 4 and 8 s with the joint fit's solution, a process analysis and a
    report table; ``stamp`` is the modification time of its main.json (newest first in the list)."""
    windows = [window(t, [entry("candidate 1", 4.0, radius * t, rank=1, weight=1.0),
                          entry("joint", 3.0, radius * t, kind="joint", base_nm=0.1 * t)],
                      pick="joint", smooth="joint", h_nom=rate * t) for t in (2.0, 4.0, 8.0)]
    out = write_result(folder, windows=windows, scan="gui_zz_invented_" + polymer, metal="Cu", family="zz_family",
                       substrate=polymer, wanted=[{"key": "R_nm"}, {"key": "D_nm"}, {"key": "coverage"},
                                                  {"key": "base_nm", "label": "continuous layer", "unit": "nm"}])
    main = out / "main" / "main.json"
    record = json.loads(main.read_text(encoding="utf-8"))
    record["process"] = {"source": "joint fit of the whole scan",
                         "descriptors": {"h_cov50": 0.5 * film_from, "h_base": film_from, "p_D": 0.4},
                         "stages": [{"stage": "islands", "t_start": 2, "t_end": 4, "h_start": 2 * rate, "h_end": 4 * rate},
                                    {"stage": "film", "t_start": 8, "t_end": 8, "h_start": film_from,
                                     "h_end": film_from}]}
    main.write_text(json.dumps(record), encoding="utf-8")
    (out / "report.csv").write_text(_report([(2.0, 2 * rate, "", "", 0.2), (4.0, 4 * rate, 4 * rate + 0.4, "reading", 0.3),
                                             (8.0, 8 * rate, 8 * rate + 0.9, "estimate", 0.7)]), encoding="utf-8")
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    analysis["files"]["report_csv"] = str(out / "report.csv")
    (out / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    os.utime(main, (stamp, stamp))
    return out


def _results(tmp_path) -> tuple:
    root = tmp_path / "results"
    first = _scan(root / "zz_scan_a-0a1b2c3d" / "iampred-11111111", rate=0.1, polymer="PEO", radius=0.5, film_from=0.7,
                  stamp=time.time() - 100)
    second = _scan(root / "zz_scan_b-4e5f6a7b" / "iampred-22222222", rate=0.2, polymer="Polymer", radius=0.3,
                   film_from=1.4, stamp=time.time() - 10)
    return root, first, second


def _compared(folder: Path, name: str = "") -> ComparedScan:
    result = client_result(folder, json.loads((folder / "main" / "main.json").read_text(encoding="utf-8")),
                           json.loads((folder / "analysis.json").read_text(encoding="utf-8")))
    return ComparedScan(name or default_name(result), result,
                        report_readings((folder / "report.csv").read_text(encoding="utf-8")))


# -- the values (domain) -----------------------------------------------------------------------------------------------


def test_curves_are_against_the_nominal_deposit_and_readings_keep_only_values_read_from_the_data(tmp_path):
    _root, first, second = _results(tmp_path)
    a, b = _compared(first), _compared(second)
    assert a.name == "zz_invented_PEO · PEO" and b.name == "zz_invented_Polymer · Polymer"
    assert series(a, "R_nm") == ([0.2, 0.4, 0.8], [1.0, 2.0, 4.0])      # the joint fit's solution, at rate x time
    assert series(b, "R_nm") == ([0.4, 0.8, 1.6], [0.6, 1.2, 2.4])
    assert series(a, "R_nm", "time") == ([1.5, 3.5, 7.5], [1.0, 2.0, 4.0])  # since the onset
    # the film thickness: the window without a reading is left out, the estimate too
    assert series(a, "d_film_nm") == ([0.4], [0.8])
    assert series(a, "yoneda_rel")[1] == [0.2, 0.3, 0.7]
    assert series(a, "base_nm") == ([0.2, 0.4, 0.8], [0.2, 0.4, 0.8])  # listed among the wanted quantities
    assert [key for key, *_rest in curves([a, b])] == ["R_nm", "D_nm", "coverage", "base_nm", "d_film_nm", "yoneda_rel"]


def test_the_growth_table_puts_the_descriptors_and_the_stage_onsets_of_each_scan_side_by_side(tmp_path):
    _root, first, second = _results(tmp_path)
    rows = growth_rows([_compared(first), _compared(second)])
    named = {name.format(**fields): values for name, fields, _unit, values in rows}
    assert named["Metal amount at which the film forms"] == (0.7, 1.4)
    assert named["Metal amount at coverage 0.5"] == (0.35, 0.7)
    assert named["Continuous film starts at (metal amount, nm)"] == (0.7, 1.4)
    assert named["Separate islands starts at (metal amount, nm)"] == (0.2, 0.4)


def test_the_exported_table_says_where_every_value_comes_from(tmp_path):
    _root, first, second = _results(tmp_path)
    text = compare_csv([_compared(first, "PEO"), _compared(second, "BCP")])
    head = [line for line in text.splitlines() if line.startswith("#")]
    assert any("PEO: folder" in line and "substrate PEO" in line for line in head)
    assert any("GIMaP computes none" in line for line in head)
    rows = list(csv.DictReader(line for line in text.splitlines() if not line.startswith("#")))
    film = [row for row in rows if row["quantity"] == "Film thickness (fringes)"]
    assert [(row["scan"], row["source"], row["Nominal deposit (nm)"], row["value"]) for row in film] == \
        [("PEO", "reading", "0.4", "0.8"), ("BCP", "reading", "0.8", "1.2")]


# -- the dialog (presentation) -----------------------------------------------------------------------------------------


def _page(tmp_path, root: Path):
    from src.gimap.features.sputter.bootstrap import create_client_service, create_sputter_service
    from src.gimap.features.sputter.infrastructure import ClientFolderReader
    from src.gimap.features.sputter.presentation.page import SputterPage

    service = create_sputter_service(
        inspector=FakeInspector(), results_root=lambda: root,
        client=create_client_service(runner=FakeClientRunner(), reader=ClientFolderReader(lambda: tmp_path / "user")))
    page = SputterPage(service, task_runner=TaskRunner())
    page.resize(1256, 869)
    return page


def _close(page) -> None:
    page.dispose()
    page.tasks.wait(10)
    page.close()
    page.deleteLater()


def test_compare_lists_the_analysed_results_and_ticks_the_shown_one_and_the_newest_other(tmp_path):
    root, first, second = _results(tmp_path)
    page = _page(tmp_path, root)
    page.show()
    assert page.compare_button.isEnabled() and page.compare_button.text() == "Compare…"
    assert page.open_client_result(first)
    dialog = page.open_compare()
    QApplication.processEvents()
    names = [dialog.scan_list.item(row).text() for row in range(dialog.scan_list.count())]
    assert names == ["zz_invented_PEO · PEO", "zz_invented_Polymer · Polymer"]  # the shown one first
    assert [scan.name for scan in dialog.ticked()] == names
    assert set(dialog.plots.drawn) == {"R_nm", "D_nm", "coverage", "base_nm", "d_film_nm", "yoneda_rel"}
    assert dialog.plots.drawn["R_nm"]["zz_invented_Polymer · Polymer"] == ([0.4, 0.8, 1.6], [0.6, 1.2, 2.4])
    table = dialog.growth_table
    assert [table.horizontalHeaderItem(column).text() for column in range(table.columnCount())] == \
        ["Quantity", "Unit", "zz_invented_PEO · PEO", "zz_invented_Polymer · Polymer"]
    film = next(row for row in range(table.rowCount())
                if table.item(row, 0).text() == "Metal amount at which the film forms")
    assert (table.item(film, 2).text(), table.item(film, 3).text()) == ("0.7", "1.4")
    # the time axis
    dialog.axis_switch.setCurrentIndex(1)
    dialog.axis_switch.activated.emit(1)
    assert dialog.plots.drawn["R_nm"]["zz_invented_PEO · PEO"][0] == [1.5, 3.5, 7.5]
    # Compare… again brings the same dialog to the front
    assert page.open_compare() is dialog
    _close(page)
    assert not dialog.isVisible()


def test_a_renamed_unticked_result_leaves_the_plots_and_export_writes_tables_and_figures(tmp_path):
    root, first, second = _results(tmp_path)
    page = _page(tmp_path, root)
    page.show()
    page.open_client_result(second)
    dialog = page.open_compare()
    QApplication.processEvents()
    item = dialog.scan_list.item(0)
    item.setText("Cu on the block copolymer")
    assert "Cu on the block copolymer" in dialog.plots.drawn["R_nm"]
    dialog.scan_list.item(1).setCheckState(Qt.Unchecked)
    assert list(dialog.plots.drawn["R_nm"]) == ["Cu on the block copolymer"]
    written = dialog.export(tmp_path / "exported")
    assert [path.name for path in written] == ["compare_curves.csv", "compare_growth.csv", "compare.png", "compare.pdf"]
    assert all(path.stat().st_size > 0 for path in written)
    growth = (tmp_path / "exported" / "compare_growth.csv").read_text(encoding="utf-8")
    assert "quantity,unit,Cu on the block copolymer" in growth
    _close(page)


def test_add_result_takes_a_folder_outside_the_results_root_and_says_when_it_holds_none(tmp_path, monkeypatch):
    root, first, _second = _results(tmp_path)
    elsewhere = _scan(tmp_path / "elsewhere" / "zz_scan_c-99999999" / "iampred-33333333", rate=0.3, polymer="PS",
                      radius=0.2, film_from=2.0, stamp=time.time())
    page = _page(tmp_path, root)
    page.show()
    page.open_client_result(first)
    dialog = page.open_compare()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(elsewhere))
    dialog.add_button.click()
    assert dialog.scan_list.count() == 3 and "zz_invented_PS · PS" in [scan.name for scan in dialog.ticked()]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not dialog.add_folder(empty)
    assert dialog.scan_list.item(dialog.scan_list.count() - 1).checkState() == Qt.Unchecked
    _close(page)


def test_compare_in_chinese(tmp_path):
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

    root, first, _second = _results(tmp_path)
    page = _page(tmp_path, root)
    page.show()
    page.open_client_result(first)
    dialog = page.open_compare()
    try:
        apply_language("zh", roots=(page, dialog))
        QApplication.processEvents()
        assert dialog.windowTitle() == "比较扫描" and dialog.tabs.tabText(1) == "生长"
        assert page.compare_button.text() == "比较…"
        assert dialog.growth_table.horizontalHeaderItem(0).text() == "量"
        rows = [dialog.growth_table.item(row, 0).text() for row in range(dialog.growth_table.rowCount())]
        assert "成膜时的金属量" in rows and "连续膜开始于（金属量，nm）" in rows
        titles = [ax.get_title() for ax in dialog.plots.figure.axes if ax.get_visible()]
        assert "半径 R (nm)" in titles and "膜厚（条纹） (nm)" in titles
    finally:
        apply_language(DEFAULT_LANGUAGE, roots=(page, dialog))
    _close(page)
