"""The Sputter page's Details: the scan folder and the geometry form (where each value comes from, presets), the
memory of each folder, the language of the page and closing the window while Analyze runs.

Offline, with a fake of the scan header and of the package's process, and invented data only.
"""

from __future__ import annotations

import ast
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.integrations.state import InMemoryUserPreferencesRepository
from tests.sputter_client_fixtures import FakeClientRunner, write_result
from tests.sputter_fixtures import FakeInspector, scan_folder

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_package_of_this_computer(monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_CLIENT", raising=False)


def _settle(condition=lambda: False, seconds: float = 10.0) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()


def _page(tmp_path, *, inspector=None, preferences=None, client=None):
    from src.gimap.features.sputter.bootstrap import create_client_service, create_sputter_service
    from src.gimap.features.sputter.infrastructure import ClientFolderReader
    from src.gimap.features.sputter.presentation.page import SputterPage

    service = create_sputter_service(
        inspector=inspector or FakeInspector(), results_root=lambda: tmp_path / "results",
        client=create_client_service(runner=client or FakeClientRunner(),
                                     reader=ClientFolderReader(lambda: tmp_path / "user")))
    page = SputterPage(service, task_runner=TaskRunner(), preferences=preferences)
    page.resize(1400, 900)
    return page


def _fill(page, folder) -> None:
    page.set_scan_folder(folder)
    _settle(lambda: page.scan_info is not None)
    page.sdd.setText("1000")
    page.alpha.setText("0.3")


def _close(page) -> None:
    page.dispose()
    page.tasks.wait(10)
    page.close()
    page.deleteLater()


# -- Details: what it holds ------------------------------------------------------------------------------------------


def test_details_holds_the_geometry_the_options_of_analyze_static_frames_and_more_options(tmp_path):
    from PyQt5.QtWidgets import QLabel

    page = _page(tmp_path)
    page.show()
    page.show_details(True)
    sections = [page.geometry_section, page.analysis_section, page.static_section, page.options_section]
    column = page.details_scroll.widget().layout()
    assert [column.itemAt(index).widget() for index in range(len(sections))] == sections
    assert not page.static_section.is_expanded() and not page.options_section.is_expanded()
    page.options_section.set_expanded(True)
    shown = {label.text() for label in page.options_section.findChildren(QLabel) if label.isVisible()}
    assert {"Mask", "Sample roll (°)", "Save results in"} <= shown
    for gone in ("predict_button", "job_status", "environment_label", "choice_box", "polymer", "bin_seconds", "t0",
                 "t_end", "merge_button", "open_button", "metal", "trend", "literature", "sketch"):
        assert not hasattr(page, gone), gone
    assert [page.tabs.tabText(index) for index in range(page.tabs.count())] == ["Windows", "Report", "Logs"]
    _close(page)


# -- the folder and the form ---------------------------------------------------------------------------------

def test_the_header_fills_the_form_and_each_value_says_where_it_comes_from(tmp_path):
    page = _page(tmp_path, inspector=FakeInspector())
    page.set_scan_folder(tmp_path / "zz_invented_scan")
    page.sdd.setText("1500")
    page.sdd.textEdited.emit("1500")  # typed while the header is read: kept
    _settle(lambda: page.scan_info is not None)
    assert page.energy_value.text() == "9" and page.energy_kind.currentData() == "energy"
    assert page.source_labels["xray"].text() == "from the file header"
    assert page.sdd.text() == "1500" and page.source_labels["sdd"].text() == "entered"
    assert page.source_labels["alpha"].text() == "not in the files"
    assert "enter it" in page.source_labels["alpha"].toolTip() and "enter it" in page.alpha.toolTip()
    assert page.source_labels["beam"].text() == "estimated from the frames"
    assert "pixel coordinates" in page.source_labels["beam"].toolTip()
    assert page.source_labels["alpha"].property("gimapRole") is None  # a plain source line, not a warning
    summary = page.scan_summary.text()
    assert "3 frames" in summary and "PILATUS 1M" in summary and "2030-01-02 03:04:05" in summary
    assert "S/N" not in summary and "keV" not in summary  # needed once: the tooltip; the energy beside X-rays
    tip = page.scan_summary.toolTip()
    assert "S/N 99-0001" in tip and "172 µm pixels" in tip and "flat field FF_x99_E9000_T4500.tif" in tip
    _close(page)


def test_an_empty_sample_roll_says_that_0_is_assumed(tmp_path):
    """The roll is optional and 0 is used when it is empty: the line says so (as a value to check), not "enter"."""
    page = _page(tmp_path)
    label = page.source_labels["roll"]
    assert page.roll.text() == "" and page._values()["roll"] == 0.0
    assert label.text() == "⚠ not in the files: 0 assumed — please check"
    assert label.property("gimapRole") == "warning" and not label.property("gimapMeta")
    page.roll.setText("0.5")
    page.roll.textEdited.emit("0.5")
    assert label.text() == "entered" and label.property("gimapRole") is None and label.property("gimapMeta")
    page.roll.setText("")
    page.roll.textEdited.emit("")
    assert label.text() == "⚠ not in the files: 0 assumed — please check" and page._values()["roll"] == 0.0
    _close(page)


def test_a_one_line_source_has_no_empty_line_under_it(tmp_path):
    """In the scroll area Qt would size a wrapped line for a narrower width: a one-line source got an empty second
    line, and a height set once never shrank again when the panel was wider."""
    page = _page(tmp_path, inspector=FakeInspector(energy_hint_source="flat_field"))
    page.show()
    page.show_details()  # the geometry fields are in Details (the page opens on the main screen)
    page.set_scan_folder(tmp_path / "zz_invented_scan")
    _settle(lambda: page.scan_info is not None)
    _settle(seconds=0.3)
    label = page.source_labels["xray"]
    one_line = label.fontMetrics().lineSpacing()
    assert label.text() and label.height() < 2 * one_line
    label.resize(60, label.height())  # narrow: the warning wraps and the line grows ...
    assert label.height() >= 2 * one_line
    label.resize(400, label.height())  # ... and wide again: back to one line
    assert label.height() < 2 * one_line
    _close(page)


def test_a_flat_field_energy_is_shown_beside_an_empty_field(tmp_path):
    page = _page(tmp_path, inspector=FakeInspector(energy_hint_source="flat_field"))
    page.set_scan_folder(tmp_path / "zz_invented_scan")
    _settle(lambda: page.scan_info is not None)
    assert page.energy_value.text() == ""
    label = page.source_labels["xray"]
    assert label.text() == "not in the files · ⚠ flat-field file name: E = 9 keV"  # one line, a warning
    assert label.property("gimapRole") == "warning"
    assert "calibration energy" in label.toolTip() and "calibration energy" in page.energy_value.toolTip()
    assert "keV" not in page.scan_summary.text()  # said once, beside the field
    _close(page)


def test_a_preset_needs_a_name(monkeypatch):
    from PyQt5.QtCore import QDate
    from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QLineEdit

    import src.gimap.features.sputter.presentation.preset_dialog as dialog

    seen = []

    def look(self):
        save = self.findChild(QDialogButtonBox).button(QDialogButtonBox.Save)
        seen.append(save.isEnabled())
        self.findChild(QLineEdit).setText("invented beamtime")
        seen.append(save.isEnabled())
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec_", look)
    answer = dialog.ask_preset(None, serial="99-0001", flat_field="", date=QDate(2030, 1, 2))
    assert seen == [False, True] and answer[0] == "invented beamtime"


def test_a_folder_gets_its_own_values_back_and_another_one_last_used(tmp_path):
    preferences = InMemoryUserPreferencesRepository()
    runner = FakeClientRunner()
    page = _page(tmp_path, preferences=preferences, client=runner)
    first, second = scan_folder(tmp_path, "zz_invented_a"), scan_folder(tmp_path, "zz_invented_b")
    _fill(page, first)
    page.energy_kind.setCurrentIndex(page.energy_kind.findData("wavelength"))
    page.energy_value.setText("0.11")
    page.sample_rate.setText("2.5")
    page.analyze()
    _settle(lambda: not page.analyzing())
    assert len(runner.requests) == 1 and runner.requests[0].wavelength_nm == 0.11
    page.set_scan_folder(second)
    _settle(lambda: page.scan_info is not None and Path(page.scan_info.folder) == second)
    assert page.sdd.text() == "1000" and page.source_labels["sdd"].text() == "⚠ last used — please check"
    assert page.source_labels["sdd"].property("gimapRole") == "warning"
    assert page.energy_value.text() == "9" and page.sample_rate.text() == ""  # the header's hint; a rate is per folder
    page.set_scan_folder(first)
    _settle(lambda: page.scan_info is not None and Path(page.scan_info.folder) == first)
    assert page.energy_kind.currentData() == "wavelength" and page.energy_value.text() == "0.11"
    assert page.source_labels["xray"].text().startswith("as last used for this folder")
    assert "⚠ header says 9 keV" in page.source_labels["xray"].text()
    assert "differs from the value used" in page.source_labels["xray"].toolTip()
    assert page.sample_rate.text() == "2.5"
    stored = preferences.get("sputter_page")
    assert set(stored) >= {"folders", "last", "last_scan", "client_folders"} and "python" not in stored
    _close(page)


def test_a_saved_preset_fills_a_scan_of_the_same_beamtime(tmp_path, monkeypatch):
    import src.gimap.features.sputter.presentation.preset_dialog as dialog

    preferences = InMemoryUserPreferencesRepository()
    page = _page(tmp_path, preferences=preferences)
    _fill(page, scan_folder(tmp_path))
    page.beam_col.setText("700.5")
    page.beam_row.setText("800.5")
    monkeypatch.setattr(dialog, "ask_preset", lambda *_a, **_k: ("invented beamtime", "2030-01-01", "2030-01-09"))
    page.preset_button.click()
    assert preferences.get("sputter_page")["presets"][0]["name"] == "invented beamtime"
    other = _page(tmp_path, preferences=InMemoryUserPreferencesRepository(
        {"sputter_page": {"presets": preferences.get("sputter_page")["presets"]}}))
    other.set_scan_folder(tmp_path / "zz_invented_c")
    _settle(lambda: other.scan_info is not None)
    assert other.beam_col.text() == "700.5" and other.source_labels["beam"].text() == "from preset invented beamtime"
    _close(page)
    _close(other)


# -- the interface language ----------------------------------------------------------------------------------


def test_details_speaks_chinese_after_a_switch(tmp_path):
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

    import re

    from PyQt5.QtWidgets import QAbstractButton, QLabel

    page = _page(tmp_path)
    page.show()
    page.show_details(True)
    page.options_section.set_expanded(True)
    page.open_client_result(write_result(tmp_path / "out"))
    page.show_details(True)
    widgets = [w for w in page.details_view.findChildren((QLabel, QAbstractButton)) if w.text() and w.isVisible()]
    english = {id(w): w.text() for w in widgets}
    try:
        apply_language("zh", roots=(page,))
        page.refresh_language()
        kept = sorted({english[id(w)] for w in widgets if w.text() == english[id(w)]
                       and re.search(r"[A-Za-z]{3,}", re.sub(r"<[^>]+>|GIMaP|PILATUS|S/N|zz_\w+", "", w.text()))
                       and not re.fullmatch(r"[\w.\- ·/:\\]+", w.text().replace("\u200b", ""))})
        assert kept == []  # every fixed text of Details has its Chinese
        assert page.source_labels["sdd"].text() == "文件中没有"
        assert page.source_labels["roll"].text() == "⚠ 文件中没有：假定为 0——请核对"
        assert page.tabs.tabText(2) == "日志"
    finally:
        apply_language(DEFAULT_LANGUAGE, roots=(page,))
        page.refresh_language()
    assert page.source_labels["sdd"].text() == "not in the files" and page.tabs.tabText(2) == "Logs"
    _close(page)


def _template_calls(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            if name in ("tr", "trf", "text") and node.args:
                yield name, node.args[0], node.lineno


def test_every_text_of_the_page_is_in_the_chinese_table():
    import re

    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.presentation.texts import LABEL_TEMPLATES, TEMPLATES

    problems = []
    for path in sorted((ROOT / "src/gimap/features/sputter").rglob("*.py")):
        for name, arg, line in _template_calls(path):
            if isinstance(arg, ast.JoinedStr):
                problems.append(f"{path.name}:{line} {name}(f\"…\") can never match the table")
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value not in ZH \
                    and re.findall(r"[^\W\d_]{2,}", re.sub(r"<[^>]+>|\{\w+\}", "", arg.value)):
                problems.append(f"{path.name}:{line} {arg.value!r}")
    for template in (*TEMPLATES, *LABEL_TEMPLATES):
        if template not in ZH:
            problems.append(f"template {template!r}")
    assert problems == []


# -- the main window -----------------------------------------------------------------------------------------


def test_the_window_asks_about_a_running_analysis_and_stops_it_on_close(monkeypatch):
    from PyQt5.QtWidgets import QMessageBox

    from main import MainWindow
    from src.gimap.app import AppContext
    from src.gimap.integrations.jobs import LocalProcessJobRunner
    from src.gimap.integrations.state import (InMemoryInstrumentProfileRepository, InMemorySessionRepository,
                                              InMemorySettingsRepository)

    context = AppContext(settings=InMemorySettingsRepository(), session=InMemorySessionRepository(),
                         preferences=InMemoryUserPreferencesRepository(), jobs=LocalProcessJobRunner(),
                         instrument_profiles=InMemoryInstrumentProfileRepository([]))
    window = MainWindow(context)
    window.show()
    _settle(lambda: window._initialization_completed, 40)
    components = window.components
    page = components.sputter_page
    assert page in components._language_pages() and "Sputter analysis" not in components.running_jobs()
    monkeypatch.setattr(page, "running", lambda: True)
    assert components.running_jobs()[-1] == "Sputter analysis"
    asked, disposed = [], []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **_k: asked.append(args[2]) or QMessageBox.Yes)
    monkeypatch.setattr(page, "dispose", lambda: disposed.append(True))
    window.close()
    _settle(seconds=.05)
    assert "Sputter analysis" in asked[0] and disposed == [True]
