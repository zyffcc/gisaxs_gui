"""The Sputter page: choose a folder, press Predict (PyTorch installed on the way), read the result.

Offline, with fakes of the model, PyTorch's installer and the scan header, and invented data only.
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
from tests.sputter_fixtures import FakeInspector, FakeInstaller, FakeRunner, result_folder, scan_folder, write_table

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_pip(monkeypatch):
    monkeypatch.setenv("GIMAP_SPUTTER_NO_INSTALL", "1")


def _settle(condition=lambda: False, seconds: float = 10.0) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()


def _page(tmp_path, *, runner=None, installer=None, inspector=None, preferences=None):
    from src.gimap.features.sputter.bootstrap import create_sputter_service
    from src.gimap.features.sputter.presentation.page import SputterPage

    service = create_sputter_service(runner=runner or FakeRunner(), installer=installer or FakeInstaller(),
                                     inspector=inspector or FakeInspector(),
                                     results_root=lambda: tmp_path / "results")
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


# -- Predict in one go ---------------------------------------------------------------------------------------


def test_predict_installs_pytorch_then_predicts_and_shows_the_physical_path(tmp_path):
    installer, runner = FakeInstaller(), FakeRunner()
    page = _page(tmp_path, runner=runner, installer=installer)
    assert "installed on the first Predict" in page.environment_label.text()
    _fill(page, scan_folder(tmp_path))
    seen = []
    page.eventReceived.connect(lambda event: seen.append(page.job_status.progress_bar.value()))
    page.predict()
    assert page.running() and not page.predict_button.isEnabled()
    _settle(lambda: not page.running())
    assert installer.installs == 1 and runner.calls[0][0] == "analyze"
    assert page.job_status.state_label.property("gimapJobState") == "succeeded"
    assert "1 damaged frames left out" in page.job_status.message_label.text()
    assert page.result.variant == "physical" and page.choice_box.currentData() == "physical"
    assert page.result.folder == runner.calls[0][2] and page.result.folder.parent == tmp_path / "results"
    assert "PyTorch 9.9.9: ready" in page.environment_label.text() and page.predict_button.isEnabled()
    assert "intensity modulation period ≈ 7 s (sample rotation?)" in page.result_label.text() and "film from 2.0 s" in page.result_label.text()
    log = page.log.toPlainText()
    assert "Downloading torch: 100%" in log and "a plain line of the model" in log and "INTAKE_EXCLUDED" in log
    assert max(seen) > 0  # the bar moved while the frames were read
    _close(page)


def test_a_failed_installation_says_why_and_predict_stays_available(tmp_path):
    page = _page(tmp_path, installer=FakeInstaller(failure="network"))
    _fill(page, scan_folder(tmp_path))
    page.predict()
    _settle(lambda: not page.running())
    assert page.job_status.state_label.property("gimapJobState") == "failed"
    message = page.job_status.message_label.text()
    assert "pypi.org" in message and "pip install -r" in message
    assert "invented pip output line" in page.log.toPlainText() and page.predict_button.isEnabled()
    assert page.result is None
    _close(page)


def test_missing_values_and_a_missing_model_are_said_before_anything_runs(tmp_path):
    page = _page(tmp_path, runner=FakeRunner(missing=str(tmp_path / "kit")))
    page.predict()
    assert not page.running() and "scan folder" in page.job_status.message_label.text()
    _fill(page, scan_folder(tmp_path))
    page.energy_value.setText("9")
    page.predict()
    assert not page.running() and "Update GIMaP" in page.job_status.message_label.text()
    assert "Update GIMaP" in page.environment_label.text()
    _close(page)


def test_stop_ends_the_run_and_keeps_the_page_usable(tmp_path):
    runner = FakeRunner(wait_for_stop=True)
    page = _page(tmp_path, runner=runner, installer=FakeInstaller(ready=True))
    _fill(page, scan_folder(tmp_path))
    page.predict()
    _settle(lambda: runner.calls)
    _settle(seconds=.2)
    assert page.running() and page.job_status.cancel_button.isEnabled()
    page.job_status.cancel_button.click()
    assert not page.job_status.cancel_button.isEnabled()
    _settle(lambda: not page.running())
    assert page.job_status.state_label.property("gimapJobState") == "cancelled"
    assert "reused on the next run" in page.job_status.message_label.text() and page.predict_button.isEnabled()
    _close(page)


def test_closing_the_window_stops_even_the_writing_of_the_packages(tmp_path):
    page = _page(tmp_path)
    page._busy = "predict"
    page.dispose()
    assert page._should_stop() == "now"
    page._busy = None
    _close(page)


def test_a_finished_predict_says_so_in_a_toast_and_raises_nothing(tmp_path):
    from src.gimap.app.error_guard import ErrorGuard
    from src.gimap.app.presentation.components.toast import visible_toasts

    caught = []
    guard = ErrorGuard(tmp_path / "logs", notify=lambda summary, _details: caught.append(summary)).install()
    try:
        page = _page(tmp_path, installer=FakeInstaller(ready=True))
        _fill(page, scan_folder(tmp_path))
        page.rate.setText("0.07")
        page.predict()
        _settle(lambda: not page.running())
        _settle(seconds=.1)
    finally:
        guard.uninstall()
    assert caught == []
    toasts = [toast for toast in visible_toasts(page) if "is ready" in toast.text()]
    assert toasts and toasts[-1].action_button is not None
    assert page.job_status.state_label.property("gimapJobState") == "succeeded"
    _close(page)


class _HeldRunner(FakeRunner):
    """Waits in ``analyze`` until the test lets it go (to type while the model runs)."""

    def __init__(self, **kwargs):
        import threading

        super().__init__(**kwargs)
        self.go = threading.Event()

    def analyze(self, request, out_folder, on_line, should_stop, environment=None):
        self.go.wait(20)
        return super().analyze(request, out_folder, on_line, should_stop, environment)


def test_a_rate_typed_while_the_model_runs_is_drawn_and_an_untouched_one_is_not(tmp_path):
    runner = _HeldRunner()
    page = _page(tmp_path, runner=runner, installer=FakeInstaller(ready=True))
    _fill(page, scan_folder(tmp_path))
    page.rate.setText("0.07")  # no t0: the toolkit counts from 0
    page.predict()
    _settle(seconds=.2)
    assert page.running() and not page.choice_box.isEnabled()
    runner.go.set()
    _settle(lambda: not page.running())
    assert [call[0] for call in runner.calls] == ["analyze"] and page.t0.text() == "0"  # nothing drawn again
    runner.go.clear()
    page.predict()
    _settle(seconds=.2)
    page.rate.setText("0.5")
    page.rate.textEdited.emit("0.5")
    runner.go.set()
    _settle(lambda: not page.running() and runner.calls[-1][0] == "physical")
    _settle(lambda: not page.running())
    assert runner.calls[-1][0] == "physical" and runner.calls[-1][2:] == (0.5, 0.0, None)
    assert page.rate.text() == "0.5" and page.result.summary["thickness"]["rate_nm_s"] == 0.5
    _close(page)


def test_a_rate_for_another_scan_is_kept_for_its_predict(tmp_path):
    from src.gimap.features.sputter.application import folder_key

    preferences = InMemoryUserPreferencesRepository()
    runner = FakeRunner()
    page = _page(tmp_path, runner=runner, installer=FakeInstaller(ready=True), preferences=preferences)
    first, second = scan_folder(tmp_path, "zz_invented_a"), scan_folder(tmp_path, "zz_invented_b")
    _fill(page, first)
    page.rate.setText("0.2")
    page.predict()
    _settle(lambda: not page.running())
    shown = page.result.folder
    page.set_scan_folder(second)
    _settle(lambda: page.header_ready())
    assert page.rate.text() == "" and page.result.folder == shown  # the new scan's rate is still to be given
    page.rate.setText("0.07")
    page.rate.editingFinished.emit()  # pressing Predict takes the focus from the field first
    assert not page.running() and "another scan" in page.job_status.message_label.text()
    page.predict_button.click()
    _settle(lambda: not page.running())
    assert [call[0] for call in runner.calls] == ["analyze", "analyze"]
    assert runner.calls[-1][1].scan_folder == second and runner.calls[-1][1].rate_nm_s == 0.07
    assert page.result.folder != shown and page.rate.text() == "0.07"
    folders = preferences.get("sputter_page")["folders"]
    assert folders[folder_key(first)]["rate"] == 0.2 and folders[folder_key(second)]["rate"] == 0.07
    _close(page)


def test_predict_pressed_during_a_new_rate_follows_it(tmp_path):
    runner = FakeRunner()
    page = _page(tmp_path, runner=runner, installer=FakeInstaller(ready=True))
    _fill(page, scan_folder(tmp_path))
    page.rate.setText("0.2")
    page.predict()
    _settle(lambda: not page.running())
    page.rate.setText("0.3")
    page.rate.editingFinished.emit()  # the press on Predict takes the focus: the physical files are drawn again
    assert page.running() and page.predict_button.isEnabled()
    page.predict_button.click()  # the click is not lost
    _settle(lambda: not page.running() and len(runner.calls) >= 3)
    _settle(lambda: not page.running())
    assert [call[0] for call in runner.calls] == ["analyze", "physical", "analyze"]
    assert runner.calls[-1][1].rate_nm_s == 0.3
    _close(page)


class _PerFolderInspector(FakeInspector):
    def __init__(self, energies: dict, problems: dict):
        super().__init__()
        self.energies, self.problems = energies, problems

    def inspect(self, folder):
        from dataclasses import replace

        info = super().inspect(folder)
        name = Path(folder).name
        return replace(info, energy_kev_hint=self.energies.get(name, info.energy_kev_hint),
                       problem=self.problems.get(name, ""))


def test_predict_waits_for_the_header_of_a_folder_typed_in(tmp_path):
    from src.gimap.features.sputter.application import NO_FRAMES

    inspector = _PerFolderInspector({"zz_invented_a": 9.0, "zz_invented_d": 11.0}, {"zz_invented_e": NO_FRAMES})
    runner = FakeRunner()
    page = _page(tmp_path, runner=runner, inspector=inspector, installer=FakeInstaller(ready=True),
                 preferences=InMemoryUserPreferencesRepository())
    _fill(page, scan_folder(tmp_path, "zz_invented_a"))
    page.predict()
    _settle(lambda: not page.running())
    later = scan_folder(tmp_path, "zz_invented_d")
    page.scan_picker.set_path(str(later))
    page.scan_picker.path_edit.editingFinished.emit()  # typed, then Predict pressed at once
    page.predict()
    assert not page.header_ready() and "header" in page.job_status.message_label.text()
    _settle(lambda: len(runner.calls) == 2 and not page.running())
    request = runner.calls[-1][1]
    assert request.scan_folder == later and request.energy_ev == 11000.0 and request.sdd_mm == 1000.0
    assert page.source_labels["xray"].text() == "from the file header"
    page.set_scan_folder(tmp_path / "zz_invented_e")
    _settle(lambda: page.header_ready())
    page.predict()
    assert len(runner.calls) == 2 and page.job_status.state_label.property("gimapJobState") == "failed"
    assert "No CBF frames" in page.job_status.message_label.text()
    _close(page)


def test_the_explanation_shown_follows_a_new_result(tmp_path):
    page = _page(tmp_path, installer=FakeInstaller(ready=True))
    _fill(page, scan_folder(tmp_path))
    page.predict()
    _settle(lambda: not page.running())
    page.choice_box.setCurrentIndex(page.choice_box.findData("window"))
    assert page.result.variant == "window"
    page.predict()
    _settle(lambda: not page.running())
    assert page.result.variant == "physical" and page.choice_box.currentData() == "physical"
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
    assert page.source_labels["alpha"].text() == "not in the files — please enter"
    assert page.source_labels["beam"].text() == "empty: estimated from the frames"
    summary = page.scan_summary.text()
    assert "3 frames" in summary and "PILATUS 1M (S/N 99-0001)" in summary and "2030-01-02 03:04:05" in summary
    _close(page)


def test_a_flat_field_energy_is_shown_beside_an_empty_field(tmp_path):
    page = _page(tmp_path, inspector=FakeInspector(energy_hint_source="flat_field"))
    page.set_scan_folder(tmp_path / "zz_invented_scan")
    _settle(lambda: page.scan_info is not None)
    assert page.energy_value.text() == ""
    label = page.source_labels["xray"].text()
    assert label.startswith("not in the files — please enter") and "calibration energy" in label
    assert "flat field: E = 9 keV" in page.scan_summary.text()
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
    page = _page(tmp_path, preferences=preferences, installer=FakeInstaller(ready=True))
    first, second = scan_folder(tmp_path, "zz_invented_a"), scan_folder(tmp_path, "zz_invented_b")
    _fill(page, first)
    page.energy_kind.setCurrentIndex(page.energy_kind.findData("wavelength"))
    page.energy_value.setText("0.11")
    page.rate.setText("0.25")
    page.predict()
    _settle(lambda: not page.running())
    page.set_scan_folder(second)
    _settle(lambda: page.scan_info is not None and Path(page.scan_info.folder) == second)
    assert page.sdd.text() == "1000" and page.source_labels["sdd"].text() == "last used — please check"
    assert page.energy_value.text() == "9" and page.rate.text() == ""  # the header's hint; a rate is per folder
    page.set_scan_folder(first)
    _settle(lambda: page.scan_info is not None and Path(page.scan_info.folder) == first)
    assert page.energy_kind.currentData() == "wavelength" and page.energy_value.text() == "0.11"
    assert page.source_labels["xray"].text().startswith("as last used for this folder")
    assert "the file header suggests 9 keV" in page.source_labels["xray"].text()
    assert page.rate.text() == "0.25"
    stored = preferences.get("sputter_page")
    assert set(stored) >= {"folders", "last", "last_scan", "last_result"} and "python" not in stored
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


def test_cu_is_offered_and_labelled_as_read_with_the_ag_model(tmp_path):
    page = _page(tmp_path)
    page.show()
    index = page.metal.findData("Cu")
    assert "approximated with the Ag model" in page.metal.itemText(index)
    page.metal.setCurrentIndex(index)
    assert page.cu_note.isVisible()
    _close(page)


# -- results -------------------------------------------------------------------------------------------------


def test_a_results_folder_opens_on_the_physical_path_with_its_thickness(tmp_path):
    page = _page(tmp_path)
    page.open_results(result_folder(tmp_path / "res", physical=True))
    assert page.result.variant == "physical" and page.choice_box.isEnabled()
    headers = [page.table.horizontalHeaderItem(i).text() for i in range(page.table.columnCount())]
    for name in ("Stage (path)", "t_eq (nm)", "t_eq range (nm)", "Nominal thickness (nm)", "R trend (nm)"):
        assert name in headers, name
    row = headers.index("t_eq range (nm)")
    assert page.table.item(1, row).text() == "0.12–0.8"
    labels = [ax.get_ylabel() for ax in page.trend.figure.axes]
    assert labels[-1] == "Metal thickness (nm)" and len(labels) == 4  # R, D, TS length, thickness
    texts = [t.get_text() for t in page.trend.figure.axes[-1].texts]
    assert "film: thickness not determined" in texts
    assert "Physical path of the network" in page.report.toPlainText()
    assert "made before the literature method existed" in page.literature.message.text()
    page.choice_box.setCurrentIndex(page.choice_box.findData("window"))
    assert page.result.variant == "window" and page.table.item(1, 0).text() == "0.6"
    _close(page)


def test_a_new_deposition_rate_only_rewrites_the_physical_files(tmp_path):
    runner = FakeRunner()
    page = _page(tmp_path, runner=runner)
    folder = result_folder(tmp_path / "res", physical=True)
    page.open_results(folder)
    assert (page.rate.text(), page.t0.text()) == ("0.2", "0")  # the fields say what the result was drawn with
    page.rate.editingFinished.emit()
    assert runner.calls == []  # leaving the field unchanged rewrites nothing
    page.rate.setText("0.5")
    page.rate.editingFinished.emit()
    _settle(lambda: not page.running())
    assert runner.calls == [("physical", folder, 0.5, 0.0, None)]
    assert page.result.summary["thickness"]["rate_nm_s"] == 0.5 and "rate 0.5 nm/s" in page.report.toPlainText()
    _close(page)


def test_the_end_of_the_deposition_draws_a_plateau_and_belongs_to_its_scan(tmp_path):
    from src.gimap.features.sputter.application import folder_key

    preferences = InMemoryUserPreferencesRepository()
    runner = FakeRunner(literature=True)
    page = _page(tmp_path, runner=runner, installer=FakeInstaller(ready=True), preferences=preferences)
    scan = scan_folder(tmp_path)
    _fill(page, scan)
    page.rate.setText("0.2")
    page.t_end.setText("1.2")
    page.predict()
    _settle(lambda: not page.running())
    assert runner.calls[0][1].t_end_s == 1.2 and page.t_end.text() == "1.2"
    # The nominal thickness stops growing at the end, which is marked; the δ axis has no tick beyond it.
    thickness = page.trend.figure.axes[-1]
    nominal = next(line for line in thickness.get_lines() if line.get_label() == "nominal: rate × (t − t0)")
    assert list(nominal.get_ydata()) == pytest.approx([0.1, 0.2, 0.24, 0.24])
    assert "deposition ended 1.2 s" in [t.get_text() for t in thickness.get_legend().get_texts()]
    top = page.trend.figure.axes[0].child_axes[0]
    page.trend.canvas.draw()
    ticks = [tick for tick in top.get_xticks() if top.get_xlim()[0] <= tick <= top.get_xlim()[1]]
    assert ticks and max(ticks) <= 0.24 + 1e-9
    assert "until the deposition ended at t_end = 1.2 s, then constant at 0.24 nm" in page.report.toPlainText()
    # A new end rewrites only the physical files and is kept for this scan; one before t0 is refused.
    page.t_end.setText("1.7")
    page.t_end.editingFinished.emit()
    _settle(lambda: not page.running())
    assert runner.calls[-1] == ("physical", page.result.folder, 0.2, 0.0, 1.7)
    assert preferences.get("sputter_page")["folders"][folder_key(scan)]["t_end"] == 1.7
    # The literature check uses the δ of the end after it, also for the TS onset of the network's path.
    assert page.result.variant == "physical" and page.choice_box.findData("literature") == 1
    assert "deposition ended at t_end = 1.7 s: later windows stay at δ = 0.34 nm" in page.literature.message.text()
    labels = [t.get_text() for t in page.literature.figure.axes[0].get_legend().get_texts()]
    assert "TS onset of the network path δ = 0.34 nm" in labels
    calls = len(runner.calls)
    page.t0.setText("2")
    page.t0.editingFinished.emit()
    assert len(runner.calls) == calls and not page.running()
    # Taken away: the deposition lasts to the end of the scan again.
    page.t0.setText("0")
    page.t_end.setText("")
    page.t_end.editingFinished.emit()
    _settle(lambda: not page.running())
    assert runner.calls[-1][2:] == (0.2, 0.0, None) and page.result.summary["thickness"]["t_end_s"] is None
    nominal = next(line for line in page.trend.figure.axes[-1].get_lines()
                   if line.get_label() == "nominal: rate × (t − t0)")
    assert list(nominal.get_ydata()) == pytest.approx([0.1, 0.2, 0.3, 0.4])
    _close(page)


def test_refined_windows_rebuild_the_tables_and_the_physical_files(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QFileDialog

    runner = FakeRunner()
    page = _page(tmp_path, runner=runner)
    folder = result_folder(tmp_path / "res", physical=True)
    page.open_results(folder)
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(tmp_path / "refined"))
    page.merge_button.click()
    _settle(lambda: not page.running())
    assert [call[0] for call in runner.calls] == ["table", "physical"]
    _close(page)


def test_a_cu_result_says_on_its_pictures_that_it_was_read_with_the_ag_model(tmp_path):
    from matplotlib.text import Text

    from src.gimap.features.sputter.presentation.sketch_view import save_storyboard, storyboard_figure
    from tests.sputter_fixtures import PHYSICAL_ROWS, PHYSICAL_SUMMARY

    folder = write_table(tmp_path / "res", "structure_physical", [dict(r, metal="Cu") for r in PHYSICAL_ROWS],
                         dict(PHYSICAL_SUMMARY, metal="Cu"))
    page = _page(tmp_path)
    page.open_results(folder)
    note = "Cu: approximated with the Ag model, for reference only."
    assert any(note in t.get_text() for t in page.sketch.figure.findobj(Text))  # Save Image… saves this figure
    figure, picks = storyboard_figure(page.result)
    texts = [t.get_text() for t in figure.findobj(Text)]
    assert picks and any(note in text for text in texts)
    assert any(text.startswith("t = 0.5 s · pre-nucleation") for text in texts)  # the stage of the path
    save_storyboard(page.result, tmp_path / "board.png")
    assert (tmp_path / "board.png").stat().st_size > 0
    _close(page)


def test_the_sketch_draws_a_film_without_particles(tmp_path):
    page = _page(tmp_path)
    from tests.sputter_fixtures import PHYSICAL_ROWS, PHYSICAL_SUMMARY

    rows = [dict(r) for r in PHYSICAL_ROWS]
    rows[3]["L_TS_nm"] = ""
    page.open_results(write_table(tmp_path / "res", "structure_physical", rows, PHYSICAL_SUMMARY))
    view = page.sketch
    view.slider.setValue(view.slider.maximum())
    assert "continuous film" in view.info.text() and "thickness not determined" in view.info.text()
    _close(page)


def test_the_time_consistent_path_and_the_selector_stay_optional(tmp_path):
    import json
    from tests.sputter_fixtures import ROWS, SUMMARY

    folder = result_folder(tmp_path / "res")
    rows = [dict(r, R_grade="S" if r.get("R_nm") else "", sequence_status="data-supported") for r in ROWS]
    rows[1]["R_grade"] = "T"
    write_table(folder, "structure_sequence", rows, dict(SUMMARY, sequence=dict(changed_from_unconstrained=1,
                                                                               first_film_time_s=1.1)))
    test = dict(R=dict(top=dict(wrong_branch_frac=.28, median_rel_error=.15), sel=dict(wrong_branch_frac=.15,
                median_rel_error=.09), oracle=dict(wrong_branch_frac=.08), **{"pass": False}),
                D=dict(top=dict(wrong_branch_frac=.21, median_rel_error=.09), sel=dict(wrong_branch_frac=.13,
                       median_rel_error=.08), oracle=dict(wrong_branch_frac=.14), **{"pass": True}))
    write_table(folder, "structure_selected", [dict(r, R_grade="L" if r.get("R_nm") else "") for r in ROWS],
                dict(SUMMARY, selector=dict(test=test, low_confidence=0)))
    json.loads((folder / "structure_selected_summary.json").read_text(encoding="utf-8"))
    page = _page(tmp_path)
    page.open_results(folder)
    assert page.result.variant == "window"  # no physical files: the per-window table
    page.choice_box.setCurrentIndex(page.choice_box.findData("sequence"))
    assert page.result.variant == "sequence" and page.result.counts("R")["T"] == 1
    assert "not more accurate" in page.report.toPlainText()
    page.open_results(folder, "selected")
    text = page.report.toPlainText()
    assert "Learned selector" in text and "R: not passed" in text and "D: passed" in text and "28%" in text
    _close(page)


# -- the interface language ----------------------------------------------------------------------------------


def test_the_page_speaks_chinese_after_a_switch(tmp_path):
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

    import re

    from PyQt5.QtWidgets import QAbstractButton, QLabel

    page = _page(tmp_path)
    page.show()
    page.open_results(result_folder(tmp_path / "res", physical=True))
    widgets = [w for w in page.findChildren((QLabel, QAbstractButton)) if w.text()]
    english = {id(w): w.text() for w in widgets}
    try:
        apply_language("zh", roots=(page,))
        page.refresh_language()
        kept = sorted({english[id(w)] for w in widgets if w.text() == english[id(w)]
                       and re.search(r"[A-Za-z]{3,}", re.sub(r"<[^>]+>|PyTorch|Python|GIMaP|PILATUS|S/N", "", w.text()))
                       and not re.fullmatch(r"[\w.\-]+", w.text())})  # names (folder, scan) are data
        assert kept == []  # every fixed text of the page has its Chinese
        assert page.predict_button.text() == "预测" and "物理路径" in page.report.toPlainText()
        assert "强度调制周期" in page.result_label.text() and "可能是样品旋转" in page.result_label.text()
        assert page.source_labels["sdd"].text() == "文件中没有——请填写"
        assert page.tabs.tabText(page.tabs.indexOf(page.literature)) == "文献对照"
        assert page.trend.figure.axes[-1].get_ylabel() == "金属厚度（nm）"
        headers = [page.table.horizontalHeaderItem(i).text() for i in range(page.table.columnCount())]
        stages = [page.table.item(row, headers.index("阶段（路径）")).text() for row in range(page.table.rowCount())]
        assert stages == ["成核前", "岛", "岛", "膜"]
        film = page.table.item(3, headers.index("膜厚"))
        assert film.text() == "未确定" and film.toolTip() == "not determined"
        assert page.table.item(1, headers.index("D 核对")).text() == "一致"
        assert "这些点不是误差棒" in page.report.toPlainText()
    finally:
        apply_language(DEFAULT_LANGUAGE, roots=(page,))
        page.refresh_language()
    assert page.predict_button.text() == "Predict" and "Physical path" in page.report.toPlainText()
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
    from src.gimap.features.sputter.presentation.texts import LABEL_TEMPLATES, STAGE_NAMES, TEMPLATES

    problems = []
    for path in sorted((ROOT / "src/gimap/features/sputter").rglob("*.py")):
        for name, arg, line in _template_calls(path):
            if isinstance(arg, ast.JoinedStr):
                problems.append(f"{path.name}:{line} {name}(f\"…\") can never match the table")
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value not in ZH \
                    and re.findall(r"[^\W\d_]{2,}", re.sub(r"<[^>]+>|\{\w+\}", "", arg.value)):
                problems.append(f"{path.name}:{line} {arg.value!r}")
    from src.gimap.features.sputter.domain.physical import NOMINAL, NOMINAL_ENDED
    from src.gimap.features.sputter.domain.structure import READ_COMMON, READ_ERROR_BARS, READ_POINTS
    from src.gimap.features.sputter.presentation.predict_flow import OTHER_SCAN, WAITING_FOR_HEADER
    from src.gimap.features.sputter.application import GRADES, LITERATURE_TEMPLATES
    from src.gimap.features.sputter.presentation.literature_tab import TEMPLATES as TAB_TEMPLATES
    from src.gimap.features.sputter.presentation.results_panel import CHOICES, D_CHECKS, D_OUTSIDE, LIT_STATUS
    from src.gimap.features.sputter.presentation.sketch_view import CU_NOTE
    from src.gimap.features.sputter.presentation.texts import TOOLKIT_TEMPLATES
    from src.gimap.features.sputter.presentation.trend_plot import TEMPLATES as TREND_TEMPLATES

    held = (READ_COMMON, READ_ERROR_BARS, READ_POINTS, NOMINAL, NOMINAL_ENDED, OTHER_SCAN, WAITING_FOR_HEADER, D_OUTSIDE,
            CU_NOTE,
            *D_CHECKS.values(), *LIT_STATUS.values(), *CHOICES.values(), *GRADES.values(), *LITERATURE_TEMPLATES,
            *TAB_TEMPLATES, *TOOLKIT_TEMPLATES, *TREND_TEMPLATES, "Literature",
            "percolation {time} s (literature method)", "film onset {time} s")  # translated from a name
    for template in (*TEMPLATES, *LABEL_TEMPLATES, *STAGE_NAMES.values(), *held):
        if template not in ZH:
            problems.append(f"template {template!r}")
    assert problems == []


# -- the literature method ---------------------------------------------------------------------------------


def test_with_a_rate_the_physical_path_is_shown_first_and_the_literature_method_checks_it(tmp_path):
    from tests.sputter_fixtures import literature_folder

    page = _page(tmp_path)
    page.open_results(literature_folder(tmp_path / "res"))
    assert page.result.variant == "physical" and "film from 2.0 s" in page.result_label.text()
    assert [page.choice_box.itemText(i) for i in range(page.choice_box.count())] == [
        "Physical path (network)", "Literature method (mass balance)", "Per window (network / refined)"]
    assert len(page.literature.figure.axes) == 4  # the Literature tab checks the network's path
    page.choice_box.setCurrentIndex(page.choice_box.findData("literature"))
    assert page.result.variant == "literature"
    assert "percolation (literature method) at δ = 0.35 nm, t = 1.75 s" in page.result_label.text()
    # Size Trend: the method's R and D, δ on the top axis, only δ in the thickness panel, the percolation marked.
    axes = page.trend.figure.axes
    assert [ax.get_ylabel() for ax in axes] == ["R (nm)", "D (nm)", "Metal thickness (nm)"]
    assert axes[0].child_axes and axes[0].child_axes[0].get_xlabel() == "δ, nominal thickness (nm)"
    legend = [t.get_text() for t in axes[0].get_legend().get_texts()]
    assert "Lit: literature method (mass balance)" in legend and "percolation 1.8 s (literature method)" in legend
    assert "68% interval of the percolation (scatter, rate ±10%, t0 ±2 s)" in legend
    assert any("by construction" in t.get_text() for t in axes[-1].texts)
    headers = [page.table.horizontalHeaderItem(i).text() for i in range(page.table.columnCount())]
    assert "Literature method" in headers and "D literature (nm)" not in headers and "D check" not in headers
    assert page.table.item(1, headers.index("R grade")).text() == "Lit"
    assert page.table.item(3, headers.index("Literature method")).text() == "after percolation"
    # The Literature tab: four panels over δ with the published curves, the checks with their verdicts.
    tab = page.literature
    assert page.tabs.tabText(page.tabs.indexOf(tab)) == "Literature" and len(tab.figure.axes) == 4
    assert all(ax.get_xscale() == "log" for ax in tab.figure.axes)
    labels = [t.get_text() for t in tab.figure.axes[0].get_legend().get_texts()]
    assert "Gen19 Ag/PS RF" in labels and "published percolation, Ag/polymers" in labels
    assert "Ag on PS" in tab.message.text() and "4.7–13.5 nm" in tab.message.text() and "first frame" in tab.message.text()
    report = tab.report.toPlainText()
    for words in ("× inconsistent", "✓ consistent", "Earlier than every published value", "Gen19", "f^(-1/3)",
                  "δ = 0.35 nm at t = 1.75 s", "the rate ±10% and t0 ±2 s"):
        assert words in report, words
    labels = [t.get_text() for t in tab.figure.axes[0].get_legend().get_texts()]
    assert "68% interval (scatter, rate ±10%, t0 ±2 s)" in labels and "2π/q* after the percolation" in labels
    # Back on the physical path; its tab shows the same comparison.
    page.choice_box.setCurrentIndex(page.choice_box.findData("physical"))
    assert page.result.variant == "physical" and len(tab.figure.axes) == 4
    page.choice_box.setCurrentIndex(page.choice_box.findData("window"))
    assert tab.figure.axes == [] and "More options ▸ Explanation" in tab.message.text()
    _close(page)


def test_the_nominal_thickness_axis_follows_the_dark_theme(tmp_path):
    from src.gimap.app.presentation.theme import apply_theme
    from tests.sputter_fixtures import literature_folder

    page = _page(tmp_path)
    page.open_results(literature_folder(tmp_path / "res"))
    try:
        apply_theme("dark", 9.0)
        page._theme_changed()
        bottom = page.trend.figure.axes[-1]
        top = page.trend.figure.axes[0].child_axes[0]
        assert top.xaxis.label.get_color() == bottom.xaxis.label.get_color() != "black"
        assert top.spines["top"].get_edgecolor() == bottom.spines["bottom"].get_edgecolor()
        assert top.xaxis.get_ticklabels()[0].get_color() == bottom.xaxis.get_ticklabels()[0].get_color()
    finally:
        apply_theme("light", 9.0)
        _close(page)


def test_a_percolation_that_is_not_determined_is_shaded_from_its_earliest_time(tmp_path):
    import json

    from tests.sputter_fixtures import literature_folder

    folder = literature_folder(tmp_path / "res")
    summary_file = folder / "structure_physical_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    summary["literature"]["percolation"].update(state="not_determined", delta_nm=None, time_s=None, reached=False,
                                                interval_delta_nm=[0.25, None], interval_time_s=[1.25, None],
                                                last_time_s=1.5)
    summary_file.write_text(json.dumps(summary), encoding="utf-8")
    page = _page(tmp_path)
    page.open_results(folder, "literature")
    assert "percolation (literature method) not determined: from δ = 0.25 nm or later" in page.result_label.text()
    legend = [t.get_text() for t in page.trend.figure.axes[0].get_legend().get_texts()]
    assert "percolation not determined: from 1.2 s or later" in legend
    shade = [patch for patch in page.trend.figure.axes[0].patches if patch.get_hatch() == "//"]
    assert shade and shade[0].get_x() == pytest.approx(1.25)
    assert shade[0].get_x() + shade[0].get_width() == pytest.approx(1.5)  # to the end of the peak phase
    assert "Percolation not determined" in page.literature.message.text()
    _close(page)


def test_without_a_rate_the_literature_tab_says_how_to_give_one_and_a_rate_keeps_the_physical_path(tmp_path):
    from tests.sputter_fixtures import literature_folder

    runner = FakeRunner(literature=True)
    page = _page(tmp_path, runner=runner)
    folder = literature_folder(tmp_path / "res", rate=False)
    page.open_results(folder)
    assert page.result.variant == "physical" and page.choice_box.findData("literature") == -1
    message = page.literature.message.text()
    assert "needs the deposition rate" in message and "More options ▸ Deposition rate (nm/s)" in message
    assert page.literature.figure.axes == [] and "not testable" in page.literature.report.toPlainText()
    # A rate: the physical path stays the explanation shown, the literature method is offered second as its check.
    page.rate.setText("0.2")
    page.rate.editingFinished.emit()
    _settle(lambda: not page.running())
    assert runner.calls == [("physical", folder, 0.2, None, None)] and page.result.variant == "physical"
    assert page.choice_box.findData("literature") == 1 and len(page.literature.figure.axes) == 4
    # A new rate keeps the literature method when it is shown; taking the rate away returns to the physical path.
    page.choice_box.setCurrentIndex(page.choice_box.findData("literature"))
    page.rate.setText("0.3")
    page.rate.editingFinished.emit()
    _settle(lambda: not page.running())
    assert runner.calls[-1][2] == 0.3 and page.result.variant == "literature"
    page.rate.setText("")
    page.rate.editingFinished.emit()
    _settle(lambda: not page.running())
    assert page.result.variant == "physical" and "needs the deposition rate" in page.literature.message.text()
    _close(page)


def test_the_literature_tab_speaks_chinese(tmp_path):
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
    from tests.sputter_fixtures import literature_folder

    page = _page(tmp_path)
    page.show()
    page.open_results(literature_folder(tmp_path / "res"), "literature")
    try:
        apply_language("zh", roots=(page,))
        page.refresh_language()
        report = page.literature.report.toPlainText()
        assert "不一致" in report and "δ = 0.35 nm，t = 1.75 s（68% 区间 0.32-0.38 nm，含噪声、速率 ±10% 和 t0 ±2 s）" in report
        assert "早于该区间内所有已发表数值" in report and "参考文献" in report
        assert "PS 上的 Ag" in page.literature.message.text()
        assert page.choice_box.itemText(0) == "物理路径（网络）" and page.choice_box.itemText(1) == "文献方法（质量守恒）"
        assert "文献方法（质量守恒）" in page.report.toPlainText()
        # The Reliability tab says the toolkit's notes with numbers in Chinese too, with δ.
        reliability = page.report.toPlainText()
        assert "δ = 速率*(t - t0)，速率取给定的 0.2 nm/s" in reliability and "given rate" not in reliability
        labels = [t.get_text() for t in page.literature.figure.axes[0].get_legend().get_texts()]
        assert "D = 2π/q*（文献方法）" in labels
    finally:
        apply_language(DEFAULT_LANGUAGE, roots=(page,))
        page.refresh_language()
    _close(page)


# -- the main window -----------------------------------------------------------------------------------------


def test_the_window_asks_about_a_running_prediction_and_stops_it_on_close(monkeypatch):
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
    assert page in components._language_pages() and "Sputter prediction" not in components.running_jobs()
    monkeypatch.setattr(page, "running", lambda: True)
    assert components.running_jobs()[-1] == "Sputter prediction"
    asked, disposed = [], []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **_k: asked.append(args[2]) or QMessageBox.Yes)
    monkeypatch.setattr(page, "dispose", lambda: disposed.append(True))
    window.close()
    _settle(seconds=.05)
    assert "Sputter prediction" in asked[0] and disposed == [True]
