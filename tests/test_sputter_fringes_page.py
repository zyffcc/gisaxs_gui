"""The island height H from the q_z fringes on the Sputter page: read fresh outputs at once, measure in the
background otherwise, show a Size Trend panel, two Windows columns and a Reliability paragraph; never fail the page.

Invented results only (``tests/sputter_fixtures.py``, ``tests/sputter_fringe_data.py``); the fringe tool is faked
except in the last test, which runs it on a small synthetic folder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.task_runner import TaskRunner
from tests.sputter_fixtures import FakeInspector, FakeInstaller, FakeRunner, result_folder

COLUMNS = ("time_s", "t_lo_s", "t_hi_s", "status", "method", "H_nm", "H_lo68_nm", "H_hi68_nm", "H_upper_bound_nm")
ROWS = [  # invented: the fixture's windows are at 0.5, 1.0, 1.5 and 2.0 s
    dict(time_s="0.5", t_lo_s="0.25", t_hi_s="0.75", status="below_range", H_upper_bound_nm="3.49"),
    dict(time_s="1.0", t_lo_s="0.75", t_hi_s="1.25", status="measured", method="minima", H_nm="4.2",
         H_lo68_nm="4.0", H_hi68_nm="4.4"),
    dict(time_s="1.5", t_lo_s="1.25", t_hi_s="1.75", status="one_side_only", method="minima", H_nm="4.6",
         H_lo68_nm="4.3", H_hi68_nm="4.9"),
    dict(time_s="2.0", t_lo_s="1.75", t_hi_s="2.25", status="not_detected"),
]


@pytest.fixture(autouse=True)
def _no_pip(monkeypatch):
    monkeypatch.setenv("GIMAP_SPUTTER_NO_INSTALL", "1")


def _settle(condition, seconds: float = 30.0) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _binned(folder: Path) -> Path:
    binned = folder / "binned"
    binned.mkdir(parents=True, exist_ok=True)
    (binned / "run.json").write_text(json.dumps({"chunks": [], "metal": "Ag"}), encoding="utf-8")
    return folder


def _outputs(out: Path, result: Path, digest: str, rows=ROWS) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "fringe_heights.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"tool_sha256": digest, "status": "measured", "metal": "Ag", "group_seconds": 0.5,
               "sources": {"run.json": _sha(result / "binned" / "run.json"), "beam_off_frames": []},
               "alpha_c": {"used_deg": 0.271, "error_deg": 0.005, "source": "phase-locked fits"},
               "growth": {"fitted": True, "v_nm_s": 0.8, "v_error_nm_s": 0.1, "t0_s": -4.2},
               "ranges": {"clean_q_inside_nm_inv": {"0": [[0.85, 1.5], [1.57, 1.8]]},
                          "H_upper_bound_below_range_nm": 3.49}}
    (out / "fringe_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return out


class FakeFringeTool:
    """Writes the invented outputs (or fails) instead of running the tool."""

    def __init__(self, *, fail: bool = False):
        self.fail, self.calls = fail, []

    def problem(self) -> str:
        return ""

    def tool_digest(self) -> str:
        return "invented-digest"

    def run(self, result_folder, out_folder, metal, on_line, should_stop) -> None:
        from src.gimap.features.sputter.application import FringesFailed

        self.calls.append((Path(result_folder), Path(out_folder), metal))
        if self.fail:
            raise FringesFailed("The island height from the q_z fringes could not be measured (exit code 2): "
                                "invented failure")
        _outputs(Path(out_folder), Path(result_folder), self.tool_digest())


def _service(tmp_path, tool=None):
    from src.gimap.features.sputter.application import FringeService
    from src.gimap.features.sputter.infrastructure import FringeFolderStore

    return FringeService(tool or FakeFringeTool(), FringeFolderStore(), lambda: tmp_path / "home")


def _page(tmp_path, fringes, *, details: bool = True):
    from src.gimap.features.sputter.bootstrap import create_sputter_service
    from src.gimap.features.sputter.presentation.page import SputterPage

    service = create_sputter_service(runner=FakeRunner(), installer=FakeInstaller(), inspector=FakeInspector(),
                                     results_root=lambda: tmp_path / "results", fringes=fringes)
    page = SputterPage(service, task_runner=TaskRunner())
    page.resize(1400, 900)
    page.trend_details.setChecked(details)
    return page


def _close(page) -> None:
    page.dispose()
    page.tasks.wait(10)
    page.close()
    page.deleteLater()


def _titles(page) -> list:
    return [ax.get_title(loc="left") for ax in page.trend.figure.axes]


def _column(page, header: str) -> list:
    headers = [page.table.horizontalHeaderItem(i).text() for i in range(page.table.columnCount())]
    j = headers.index(header)
    return [page.table.item(i, j).text() for i in range(page.table.rowCount())]


def test_fresh_outputs_show_the_panel_the_columns_and_the_paragraph(tmp_path):
    folder = _binned(result_folder(tmp_path / "res", physical=True))
    tool = FakeFringeTool()
    _outputs(folder / "fringes_v1", folder, tool.tool_digest())
    for details in (False, True):  # a direct measurement: in the simple view too
        page = _page(tmp_path, _service(tmp_path, tool), details=details)
        try:
            page.open_results(folder)
            assert page.fringes is not None and not tool.calls  # read, not measured again
            assert "H, island height from q_z fringes (nm)" in _titles(page)
            ax = next(a for a in page.trend.figure.axes if a.get_ylabel() == "H (nm)")
            labels = [t.get_text() for t in ax.get_legend().get_texts()]
            assert "H from q_z fringes (68%)" in labels and "H, one side of the beam only" in labels
            assert "below the range: H < 3.49 nm" in labels and "nominal δ = rate × (t − t0)" in labels
            assert _column(page, "H fringes (nm)") == ["< 3.49", "4.2", "4.6", ""]
            assert _column(page, "H fringes 68% (nm)") == ["", "4–4.4", "4.3–4.9", ""]
            report = page.report.toPlainText()
            assert "Island height H from the q_z fringes." in report and "αc = 0.271° ± 0.005°" in report
            assert "H is measured in 2 of 4 time groups" in report
        finally:
            _close(page)


def test_without_outputs_the_height_is_measured_in_the_background(tmp_path):
    folder = _binned(result_folder(tmp_path / "res", physical=True))
    tool = FakeFringeTool()
    page = _page(tmp_path, _service(tmp_path, tool))
    try:
        page.open_results(folder)
        _settle(lambda: page.fringes is not None)
        assert tool.calls == [(folder, folder / "fringes_v1", "Ag")]
        assert "H, island height from q_z fringes (nm)" in _titles(page)
        assert "H fringes (nm)" in [page.table.horizontalHeaderItem(i).text() for i in range(page.table.columnCount())]
        page.open_results(folder)  # shown again: the outputs are fresh now
        assert len(tool.calls) == 1 and page.fringes is not None
    finally:
        _close(page)


def test_a_failure_is_logged_and_the_page_goes_on(tmp_path):
    folder = _binned(result_folder(tmp_path / "res", physical=True))
    tool = FakeFringeTool(fail=True)
    page = _page(tmp_path, _service(tmp_path, tool))
    try:
        page.open_results(folder)
        _settle(lambda: "invented failure" in page.log.toPlainText())
        assert "invented failure" in page.log.toPlainText() and page.fringes is None
        assert "H, island height from q_z fringes (nm)" not in _titles(page) and page.result is not None
        page.open_results(folder)  # shown again: the failed run is not repeated
        QApplication.processEvents()
        assert len(tool.calls) == 1
    finally:
        _close(page)


def test_a_result_without_binned_cuts_is_not_measured(tmp_path):
    tool = FakeFringeTool()
    page = _page(tmp_path, _service(tmp_path, tool))
    try:
        page.open_results(result_folder(tmp_path / "res", physical=True))
        QApplication.processEvents()
        assert not tool.calls and page.fringes is None
    finally:
        _close(page)


def test_outputs_go_to_the_user_data_folder_when_the_result_cannot_take_them_and_go_stale(tmp_path, monkeypatch):
    from src.gimap.features.sputter.infrastructure import FringeFolderStore, StructureFolderReader

    folder = _binned(result_folder(tmp_path / "res", physical=True))
    service = _service(tmp_path)
    monkeypatch.setattr(FringeFolderStore, "writable", lambda self, path: Path(folder) not in Path(path).parents)
    result = StructureFolderReader().read(folder, "physical")
    heights = service.measure(result)
    assert heights.folder == tmp_path / "home" / "sputter_fringes" / folder.name
    assert not (folder / "fringes_v1").exists() and service.cached(result) is not None
    (folder / "binned" / "run.json").write_text(json.dumps({"chunks": [], "metal": "Au"}), encoding="utf-8")
    assert service.cached(result) is None  # the binned cuts changed: measured again


def test_the_bundled_tool_runs_through_gimaps_runner(tmp_path):
    from src.gimap.features.sputter.application import FringeService
    from src.gimap.features.sputter.infrastructure import FringeFolderStore, FringeTool, StructureFolderReader
    from tests.sputter_fixtures import PHYSICAL_SUMMARY, write_table
    from tests.sputter_fringe_data import write_result

    folder = write_result(tmp_path / "res", lambda t: 0.0, windows=40, metal="Ag")
    write_table(folder, "structure_physical", [dict(frame=f"bin_{i:06d}", time_s=str((i + .5) * .5), metal="Ag",
                                                    beam_off_sample="false") for i in range(40)], PHYSICAL_SUMMARY)
    service = FringeService(FringeTool(), FringeFolderStore(), lambda: tmp_path / "home")
    result = StructureFolderReader().read(folder, "physical")
    assert service.can_measure(result) and service.cached(result) is None
    heights = service.measure(result)
    assert heights.status == "not_detected" and not heights.measured()
    assert heights.folder == folder / "fringes_v1" and service.cached(result) is not None
