"""The Sputter page's main screen: Analyze with the client package, one time window of its result, the solution
chosen, the evolution; Details as the second view; a result on a porous substrate, a series of static samples and
Details ▸ Static Frames.

Offline, with a fake of the package's process and invented results (``tests/sputter_client_fixtures.py``).
"""

from __future__ import annotations

import json
import os
import re
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QAbstractButton, QApplication, QFileDialog, QLabel

from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.integrations.state import InMemoryUserPreferencesRepository
from src.gimap.features.sputter.application import STATIC_UNSUPPORTED
from tests.sputter_client_fixtures import (POROUS_WANTED, FakeClientRunner, FakeStaticRunner, entry, package_folder,
                                           porous_windows, window, write_result, write_static)
from tests.sputter_fixtures import FakeInspector, FakeInstaller, FakeRunner, result_folder, scan_folder


@pytest.fixture(autouse=True)
def _no_package_of_this_computer(monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_CLIENT", raising=False)
    monkeypatch.setenv("GIMAP_SPUTTER_NO_INSTALL", "1")


def _settle(condition=lambda: False, seconds: float = 10.0) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end and not condition():
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()


def _page(tmp_path, *, client=None, preferences=None, reader=None, static=None):
    from src.gimap.features.sputter.bootstrap import create_client_service, create_sputter_service
    from src.gimap.features.sputter.infrastructure import ClientFolderReader
    from src.gimap.features.sputter.presentation.page import SputterPage

    reader = reader or ClientFolderReader(lambda: tmp_path / "user")
    service = create_sputter_service(
        runner=FakeRunner(), installer=FakeInstaller(ready=True), inspector=FakeInspector(),
        results_root=lambda: tmp_path / "results",
        client=create_client_service(runner=client or FakeClientRunner(), reader=reader, static_runner=static))
    page = SputterPage(service, task_runner=TaskRunner(), preferences=preferences)
    page.resize(1256, 869)  # the page's size in a 1440 x 900 window
    return page


def _close(page) -> None:
    page.dispose()
    page.tasks.wait(10)
    page.close()
    page.deleteLater()


def _fill(page, folder, rate: str = "0.25") -> None:
    page.set_scan_folder(folder)
    _settle(lambda: page.header_ready())
    page.sdd.setText("1000")
    page.alpha.setText("0.3")
    page.sample_rate.setText(rate)


def _tiles(page) -> dict:
    return {key: page.tile_values[key].text() for key, tile in page.tiles.items() if not tile.isHidden()}


# -- a result on the main screen ---------------------------------------------------------------------------------------


def test_a_result_shows_its_default_solution_on_the_tiles(tmp_path):
    page = _page(tmp_path)
    page.show()
    assert not page.details_shown() and page.result_stack.currentIndex() == 0  # the main screen, nothing yet
    assert page.open_client_result(write_result(tmp_path / "out"))
    assert page.result_stack.currentIndex() == 1 and page.client_result.scan == "zz_invented_scan"
    assert (page.window_slider.minimum(), page.window_slider.maximum(), page.window_slider.value()) == (0, 3, 0)
    assert page.window_label.text() == "t = 2 s · 0.4 nm deposited"
    assert "1.5 s after the start of the deposition" in page.window_label.toolTip()
    assert page.result_name.text() == "zz_invented_scan · Ag/Si · zz_family"  # scan, metal on substrate, model
    # the solution of the smooth evolution is shown first, although another one has the lowest fit deviation
    assert page.client_entry_label() == "candidate 2" and page.entry_control.currentData() == "candidate 2"
    assert [page.entry_control.itemText(i) for i in range(page.entry_control.count())] == \
        ["Candidate 1 · 3.2", "Candidate 2 · 4.4", "Candidate 3 · 9.9", "Refined · 2.1"]
    assert "weight of the network 0.2" in page.entry_control.button(1).toolTip()
    assert _tiles(page) == {"R_nm": "1.5", "H_nm": "2.25", "D_nm": "6", "gap_nm": "3", "width": "0.21",
                            "coverage": "0.3"}
    assert page.deviation_value.text() == "4.4" and not page.default_button.isVisible()
    assert page.patterns.titles == ("Measured image", "Model image: Candidate 2")
    assert page.report_button.isEnabled() and page.result_folder_button.isEnabled()
    _close(page)


def test_another_solution_changes_the_tiles_and_the_model_image_and_is_remembered(tmp_path):
    preferences = InMemoryUserPreferencesRepository()
    page = _page(tmp_path, preferences=preferences)
    page.show()
    out = write_result(tmp_path / "out")
    page.open_client_result(out)
    measured_limits, model_before = page.patterns.limits, page.patterns.model_values().copy()
    radius_before = page.evolution.series["R_nm"][1][0]
    page.entry_control.button(3).click()  # Refined
    assert page.client_entry_label() == "refined" and page.deviation_value.text() == "2.1"
    assert _tiles(page)["R_nm"] == "1.2" and _tiles(page)["D_nm"] == "4.8"
    assert page.patterns.titles[1] == "Model image: Refined"
    assert not np.allclose(page.patterns.model_values(), model_before, equal_nan=True)
    assert page.patterns.limits == measured_limits  # one colour scale, set by the measured image: it does not move
    assert (radius_before, page.evolution.series["R_nm"][1][0]) == (1.5, 1.2)  # the evolution follows the choice
    assert page.default_button.isVisible()
    choices = json.loads((out / "main" / "choices.json").read_text(encoding="utf-8"))
    assert choices["choices"] == {"2": "refined"}
    # another window keeps its own solution
    page.window_next.click()
    assert page.window_label.text() == "t = 4 s · 0.8 nm deposited" and page.client_entry_label() == "refined"
    assert not page.default_button.isVisible()  # this window shows its default
    _close(page)
    # opened again (also at the next start): the choice is back
    again = _page(tmp_path, preferences=preferences)
    again.show()
    assert again.client_result is not None and again.client_result.folder == out  # the result shown last
    assert again.client_entry_label() == "refined" and _tiles(again)["R_nm"] == "1.2"
    again.default_button.click()
    assert again.client_entry_label() == "candidate 2" and _tiles(again)["R_nm"] == "1.5"
    assert json.loads((out / "main" / "choices.json").read_text(encoding="utf-8"))["choices"] == {}
    _close(again)


def test_windows_are_chosen_with_the_slider_the_arrows_and_a_click_on_the_evolution(tmp_path):
    page = _page(tmp_path)
    page.show()
    page.open_client_result(write_result(tmp_path / "out"))
    assert not page.window_previous.isEnabled() and page.window_next.isEnabled()
    page.window_slider.setValue(1)
    # 4 s: no smooth entry, the lowest fit deviation is shown; three entries, so the selector is built again
    assert page.client_window().time_s == 4.0 and page.client_entry_label() == "refined"
    assert page.entry_control.count() == 3 and _tiles(page)["H_nm"] == "3.3"
    page.entry_control.button(1).click()  # a candidate without a height: its tile goes
    assert "H_nm" not in _tiles(page) and _tiles(page)["R_nm"] == "2.6"
    # 6 s: the package had no solution; the page says why and shows no values
    page.window_next.click()
    assert page.client_window().time_s == 6.0 and _tiles(page) == {}
    assert page.patterns.titles == ("", "") and not page.deviation_value.isVisible()
    assert not page.entry_control.isVisible() and page.window_label.text() == "t = 6 s"
    # a click on the evolution selects the nearest window; the mark follows
    assert page.evolution.window_at(0, 7.4) == page.evolution.window_at(1, 7.4, 0.2) == 3
    page.select_window(page.evolution.window_at(0, 7.4))
    assert page.client_window().time_s == 8.0 and page.window_slider.value() == 3
    assert page.evolution._marks[0].get_xdata()[0] == 8.0
    assert _tiles(page) == {"R_nm": "3", "D_nm": "12", "gap_nm": "6", "width": "0.21", "coverage": "0.55"}
    assert not page.window_next.isEnabled() and page.window_previous.isEnabled()
    times, heights = page.evolution.series["H_nm"]
    assert times == [2.0, 4.0, 6.0, 8.0] and np.isnan(heights[1:]).all()  # gaps where a solution has no height
    _close(page)


def test_images_that_are_missing_leave_the_values_and_say_so(tmp_path):
    page = _page(tmp_path)
    page.show()
    out = write_result(tmp_path / "out")
    (out / "main" / "w_2.npz").unlink()
    page.open_client_result(out)
    assert _tiles(page)["R_nm"] == "1.5" and page.patterns.titles == ("", "")
    assert page.patterns.model_values() is None
    _close(page)


def test_the_pictures_show_every_cell_at_its_place_with_its_own_value():
    import math

    from src.gimap.features.sputter.presentation.pattern_view import (cell_edges, colour_limits, decade_ticks,
                                                                      raster_cells, shown)

    edges = cell_edges(np.array([1.0, 2.0, 4.0]))  # halfway between the centres
    assert edges.tolist() == [0.5, 1.5, 3.0, 5.0]
    assert cell_edges(np.array([0.1, 0.5]), positive=True)[0] > 0  # a logarithmic axis starts above zero
    cells = raster_cells(edges, 9)  # cells 1, 1.5 and 2 wide get 2, 3 and 4 of 9 evenly spaced points
    assert np.bincount(cells).tolist() == [2, 3, 4]
    assert raster_cells(np.array([1.0, 10.0, 100.0]), 4, log=True).tolist() == [0, 0, 1, 1]  # evenly in log10
    majors, labels, minors = decade_ticks(-1.3, 0.3)
    assert majors == [-1.0, 0.0] and labels == ["$10^{-1}$", "$10^{0}$"]
    assert minors and all(-1.3 <= u <= 0.3 and u not in majors for u in minors)
    majors, labels, _minors = decade_ticks(math.log10(0.15), math.log10(0.9))  # less than two decades: 1, 2, 5
    assert labels == ["0.2", "0.5"] and majors == pytest.approx([math.log10(0.2), math.log10(0.5)])
    measured = np.array([[np.nan, 0.0, 1.0], [10.0, 100.0, 1000.0]])
    low, high = colour_limits(measured)
    assert 1.0 <= low < high <= 1000.0  # of the positive measured values only
    assert colour_limits(np.full((2, 2), np.nan), np.array([[2.0, 4.0]])) is not None  # nothing measured: the model
    assert colour_limits(np.zeros((2, 2))) is None
    picture = shown(measured, low)
    assert picture.mask.tolist() == [[True, False, False], [False, False, False]]  # no data stays empty
    assert picture[0, 1] == low and picture[1, 2] == 1000.0  # zero counts at the lowest colour, values unchanged


def test_another_window_puts_its_values_into_the_same_pictures(tmp_path):
    from src.gimap.features.sputter.presentation.pattern_view import colour_limits

    page = _page(tmp_path)
    page.show()
    page.open_client_result(write_result(tmp_path / "out"))
    drawn = page.patterns._drawn
    first, limits_before = drawn["images"][0], page.patterns.limits
    assert drawn["images"][0].norm is drawn["images"][1].norm  # one colour scale for both
    page.window_next.click()
    assert page.patterns._drawn is drawn and drawn["images"][0] is first  # updated, not drawn anew
    images = page._patterns(page.client_window())
    limits = colour_limits(images.measured)
    assert page.patterns.limits == limits != limits_before and (first.norm.vmin, first.norm.vmax) == limits
    rows, columns = drawn["rows"], drawn["columns"]
    assert rows.min() == 0 and rows.max() == images.af.size - 1 and columns.max() == images.q.size - 1
    for image, values in zip(drawn["images"], (images.measured, images.model("refined"))):
        expected = np.ma.masked_invalid(np.maximum(values, limits[0]))[np.ix_(rows, columns)]
        assert np.ma.allclose(image.get_array(), expected) and image.get_array().mask[rows == 5].all()
    assert page.patterns.titles == ("Measured image", "Model image: Refined")
    page.window_slider.setValue(2)  # a window without a solution: the message, then the images again
    assert page.patterns._drawn is None
    page.window_slider.setValue(3)
    assert page.patterns._drawn is not None and page.patterns.titles[1] == "Model image: Candidate 1"
    _close(page)


def test_the_main_screen_judges_nothing(tmp_path):
    """Ambiguity is the candidates themselves: no grade, status or reliability wording on the main screen."""
    page = _page(tmp_path)
    page.show()
    page.open_client_result(write_result(tmp_path / "out"))
    shown = [w for w in page.client_view.findChildren((QLabel, QAbstractButton))] + \
        [w for w in page.findChildren((QLabel, QAbstractButton)) if w.parent() is not None
         and not page.details_view.isAncestorOf(w) and not page.client_view.isAncestorOf(w)]
    words = " ".join(f"{w.text()} {w.toolTip()}" for w in shown).lower()
    words += " ".join(page.patterns.titles).lower()
    for forbidden in ("reliab", "trust", "grade", "uncertain", "conditional", "warning", "not validated",
                      "development reference"):
        assert forbidden not in words, forbidden
    _close(page)


@pytest.mark.parametrize("width, height", [(1256, 869), (1096, 769)])  # the page in a 1440 x 900 and a 1280 x 800 window
def test_the_main_screen_fits_the_window_without_expanding_anything(tmp_path, width, height):
    from PyQt5.QtWidgets import QWidget

    host = QWidget()
    host.resize(width, height)
    page = _page(tmp_path)
    page.setParent(host)
    page.setGeometry(0, 0, width, height)  # as the window's page stack gives it
    host.show()
    page.open_client_result(write_result(tmp_path / "out"))
    page.entry_control.button(3).click()  # with Back to Default shown
    _settle(seconds=0.2)
    margins = page.layout().contentsMargins()
    bar = page.layout().itemAt(0).widget()
    room = width - margins.left() - margins.right()
    assert bar.minimumSizeHint().width() <= room and page.client_view.minimumSizeHint().width() <= room
    assert bar.minimumSizeHint().height() + page.layout().spacing() + page.client_view.minimumSizeHint().height() \
        <= height - margins.top() - margins.bottom()
    for widget in (page.analyze_button, page.analyze_status, page.view_switch, page.window_slider, page.window_label,
                   page.patterns.canvas, page.entry_control, page.default_button, page.tiles["R_nm"],
                   page.tiles["coverage"], page.deviation_value, page.evolution.canvas):
        corner = widget.mapTo(page, widget.rect().bottomRight())
        assert widget.isVisible() and corner.x() < width and corner.y() < height, widget.objectName()
        assert widget.width() >= widget.minimumSizeHint().width(), widget.objectName()  # nothing squeezed
    assert page.patterns.canvas.height() >= 220 and page.evolution.canvas.height() >= 170
    _close(page)
    host.deleteLater()


# -- Analyze -----------------------------------------------------------------------------------------------------------


def test_analyze_runs_the_package_and_shows_its_result(tmp_path):
    from src.gimap.app.presentation.components.toast import visible_toasts

    runner = FakeClientRunner()
    preferences = InMemoryUserPreferencesRepository()
    page = _page(tmp_path, client=runner, preferences=preferences)
    page.show()
    scan = scan_folder(tmp_path)
    _fill(page, scan)
    assert "3 frames" in page.scan_summary.text()
    assert page.geometry_summary.full_text() == "9 keV · 1000 mm · αi 0.3° · beam centre from the frames · roll 0°"
    page.sample_metal.setCurrentIndex(page.sample_metal.findData("Au"))
    assert page.metal.currentData() == "Au"  # one metal for both methods
    page.substrate.setCurrentIndex(page.substrate.findData("PS"))
    page.qcm_check.setChecked(True)
    page.film_thickness.setText("35")
    page.mode_box.setCurrentIndex(page.mode_box.findData("quick"))
    page.show_details(True)
    seen = []
    page.clientEvent.connect(lambda _event: seen.append(page.analyze_status.message_label.text()))
    page.analyze_button.click()
    assert page.analyzing() and page.running() and not page.analyze_button.isEnabled()
    _settle(lambda: not page.analyzing())
    request = runner.requests[0]
    assert (request.scan_folder, request.metal, request.substrate, request.mode) == (scan, "Au", "PS", "quick")
    assert (request.rate_nm_s, request.rate_is_qcm, request.film_nm) == (0.25, True, 35.0)
    assert (request.energy_ev, request.sdd_mm, request.alpha_i_deg) == (9000.0, 1000.0, 0.3)
    assert (request.beam_col, request.beam_row, request.sample_roll_deg, request.mask_file) == (None, None, 0.0, None)
    # the output is a subfolder of the scan's results folder
    assert runner.folders[0].name.startswith("iampred-") and runner.folders[0].parent.parent == tmp_path / "results"
    assert runner.folders[0].parent == page.service.results_folder(page._request())
    assert page.analyze_status.state_label.property("gimapJobState") == "succeeded"
    assert page.analyze_status.message_label.text() == "Done: 4 time windows."
    assert "Step 1 of 4: Reading the frames…" in seen and "Step 4 of 4: Drawing the measured and the model images…" in seen
    assert page.client_result.folder == runner.folders[0] and not page.details_shown()  # back on the main screen
    assert "a plain line of the package" in page.client_log.toPlainText()
    assert any("is ready" in toast.text() for toast in visible_toasts(page))
    stored = preferences.get("sputter_page")
    assert stored["last_client_result"] == str(runner.folders[0]) and stored["client_last"]["substrate"] == "PS"
    # the same request again: its result is shown, the package does not run
    page.analyze_button.click()
    _settle(lambda: not page.analyzing())
    assert len(runner.requests) == 1 and "analysed before" in page.analyze_status.message_label.text()
    _close(page)


def test_the_bar_remembers_each_folders_own_values(tmp_path):
    preferences = InMemoryUserPreferencesRepository()
    page = _page(tmp_path, preferences=preferences)
    first, second = scan_folder(tmp_path, "zz_invented_a"), scan_folder(tmp_path, "zz_invented_b")
    _fill(page, first, rate="0.4")
    page.substrate.setCurrentIndex(page.substrate.findData("PEO"))
    page.onset.setText("1.5")
    page.analyze()
    _settle(lambda: not page.analyzing())
    page.set_scan_folder(second)
    _settle(lambda: page.header_ready())
    assert page.sample_rate.text() == "" and page.onset.text() == ""  # a rate and an onset belong to one scan
    assert page.substrate.currentData() == "PEO"  # the substrate used last is offered
    page.sample_rate.setText("0.1")
    page.sample_rate.textEdited.emit("0.1")
    page.set_scan_folder(first)
    _settle(lambda: page.header_ready())
    assert page.sample_rate.text() == "0.4" and page.onset.text() == "1.5"
    _close(page)
    fresh = _page(tmp_path, preferences=preferences)
    assert fresh.substrate.currentData() == "PEO" and fresh.sample_rate.text() == ""
    # a rate typed before the folder is chosen is for that folder: it stays
    fresh.sample_rate.setText("0.7")
    fresh.sample_rate.textEdited.emit("0.7")
    fresh.set_scan_folder(scan_folder(tmp_path, "zz_invented_c"))
    _settle(lambda: fresh.header_ready())
    assert fresh.sample_rate.text() == "0.7"
    fresh.set_scan_folder(first)  # a folder analysed before gets its own values, whatever was typed
    _settle(lambda: fresh.header_ready())
    assert fresh.sample_rate.text() == "0.4" and fresh.onset.text() == "1.5"
    _close(fresh)


def test_analyze_says_what_is_missing_before_anything_runs(tmp_path):
    runner = FakeClientRunner()
    page = _page(tmp_path, client=runner)
    page.show()
    page.analyze()
    assert not page.analyzing() and "scan folder" in page.analyze_status.message_label.text()
    _fill(page, scan_folder(tmp_path), rate="")
    assert page.geometry_summary.property("gimapRole") is None
    page.alpha.setText("")
    assert page.geometry_summary.full_text() == "Geometry to enter: incidence angle"
    assert page.geometry_summary.property("gimapRole") == "warning"
    page.analyze()
    message = page.analyze_status.message_label.text()
    assert "deposition rate" in message and "incidence angle" in message and runner.requests == []
    assert page.analyze_status.state_label.property("gimapJobState") == "failed"
    page.geometry_edit_button.click()  # Edit: Details at the geometry, the cursor in the empty field
    assert page.details_shown() and page.view_switch.currentData() == "details"
    assert QApplication.focusWidget() is page.alpha or page.alpha.hasFocus()
    page.view_switch.button(0).click()
    assert not page.details_shown()
    _close(page)


def _shown(label) -> str:
    """The text of a line without the zero-width spaces that let a folder wrap."""
    return label.text().replace("​", "")


def test_without_the_package_the_page_asks_for_its_folder(tmp_path, monkeypatch):
    from src.gimap.features.sputter.application import CLIENT_MISSING

    runner = FakeClientRunner(missing=CLIENT_MISSING)
    preferences = InMemoryUserPreferencesRepository()
    page = _page(tmp_path, client=runner, preferences=preferences)
    page.show()
    assert page.result_stack.currentIndex() == 0
    assert page.client_empty.title_label.text() == "The analysis package is needed"
    assert "GIMAP_SPUTTER_CLIENT" in page.client_empty.message_label.text()
    assert page.client_empty.action_button.text() == "Choose Folder…" and page.package_label.text() == "not found"
    _fill(page, scan_folder(tmp_path))
    page.analyze()
    assert not page.analyzing() and "was not found" in page.analyze_status.message_label.text()
    chosen = package_folder(tmp_path / "zz_package")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(chosen))
    page.client_empty.action_button.click()
    assert runner.chosen == chosen and preferences.get("sputter_page")["client_package"] == str(chosen)
    assert page.client_empty.title_label.text() == "No result yet"  # now: how to get a result
    assert page.client_empty.action_button.text() == "Open Result…"
    assert _shown(page.package_label) == "zz_invented_package"
    # Open Result… works without the package too
    out = write_result(tmp_path / "out")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(out))
    page.open_result_button.click()
    assert page.client_result is not None and page.result_stack.currentIndex() == 1
    _close(page)


def test_a_real_package_folder_is_checked_and_remembered(tmp_path, monkeypatch):
    """With the real adapter: a folder that is no package is refused with what is missing."""
    from src.gimap.features.sputter.infrastructure import ClientPackageRunner

    runner = ClientPackageRunner(lambda: tmp_path / "user", local=tmp_path / "nowhere")
    preferences = InMemoryUserPreferencesRepository()
    page = _page(tmp_path, client=runner, preferences=preferences)
    assert page.client_empty.title_label.text() == "The analysis package is needed"
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(tmp_path))
    page.package_button.click()
    assert "is not an analysis package" in page.package_label.toolTip()
    assert page.package_label.property("gimapRole") == "warning"
    good = package_folder(tmp_path / "zz_package")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(good))
    page.package_button.click()
    assert _shown(page.package_label) == str(good) and page.package_label.property("gimapRole") is None
    _close(page)
    again = _page(tmp_path, client=ClientPackageRunner(lambda: tmp_path / "user", local=tmp_path / "nowhere"),
                  preferences=preferences)
    assert _shown(again.package_label) == str(good)  # the folder chosen is kept in the preferences
    assert again.client_empty.title_label.text() == "No result yet"
    _close(again)


def test_a_long_package_folder_wraps_inside_the_column_of_details(tmp_path, monkeypatch):
    """A folder has no spaces: unbroken, its line made Details wider than its column and cut off its fields."""
    from src.gimap.features.sputter.infrastructure import ClientPackageRunner

    page = _page(tmp_path, client=ClientPackageRunner(lambda: tmp_path / "user", local=tmp_path / "nowhere"))
    page.resize(1440, 900)
    page.show()
    page.show_details(True)
    page.static_section.set_expanded(True)
    _settle(seconds=0.3)
    inner = page.details_scroll.widget()
    without = inner.minimumSizeHint().width()  # "not found": a short line
    good = package_folder(tmp_path / "zz_a_rather_long_folder_name_for_a_package" / "zz_and_one_more_level" / "zz_package")
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *_a, **_k: str(good))
    page.package_button.click()
    _settle(seconds=0.3)
    assert _shown(page.package_label) == str(good)
    assert page.package_label.minimumSizeHint().width() < page.details_scroll.viewport().width()
    assert inner.minimumSizeHint().width() <= max(without, page.details_scroll.viewport().width())
    _close(page)


def test_cancel_stops_the_package_and_a_failure_keeps_its_last_lines(tmp_path):
    runner = FakeClientRunner(wait_for_stop=True)
    page = _page(tmp_path, client=runner)
    page.show()
    _fill(page, scan_folder(tmp_path))
    page.analyze()
    _settle(lambda: runner.requests)
    _settle(lambda: "Step 1 of 4" in page.analyze_status.message_label.text(), 5)
    assert page.analyzing() and page.analyze_status.cancel_button.isEnabled()
    page.analyze_status.cancel_button.click()
    assert not page.analyze_status.cancel_button.isEnabled() and "Stopping" in page.analyze_status.message_label.text()
    _settle(lambda: not page.analyzing())
    assert page.analyze_status.state_label.property("gimapJobState") == "cancelled"
    assert "reused by the next run" in page.analyze_status.message_label.text() and page.analyze_button.isEnabled()
    _close(page)
    failing = _page(tmp_path / "b", client=FakeClientRunner(status="failed at predict"))
    failing.show()
    _fill(failing, scan_folder(tmp_path / "b"))
    failing.analyze()
    _settle(lambda: not failing.analyzing())
    assert failing.analyze_status.state_label.property("gimapJobState") == "failed"
    assert "(failed at predict)" in failing.analyze_status.message_label.text() and failing.client_result is None
    assert "the invented reason of the failure" in failing.client_log.toPlainText()
    failing.analyze_status.details_button.click()  # Details: the log, in the second view
    assert failing.details_shown() and failing.client_log.isVisible()
    _close(failing)


def test_analyze_pressed_while_the_header_is_read_starts_after_it(tmp_path):
    runner = FakeClientRunner()
    page = _page(tmp_path, client=runner, preferences=InMemoryUserPreferencesRepository())
    _fill(page, scan_folder(tmp_path))
    page.analyze()
    _settle(lambda: len(runner.requests) == 1 and not page.analyzing())
    later = scan_folder(tmp_path, "zz_invented_d")
    page.scan_picker.set_path(str(later))  # typed, then Analyze pressed at once
    page.analyze()
    assert not page.header_ready() and "header" in page.analyze_status.message_label.text()
    _settle(lambda: len(runner.requests) == 2 and not page.analyzing())
    request = runner.requests[1]
    assert request.scan_folder == later and request.rate_nm_s == 0.25  # what the bar said when it was pressed
    assert request.sdd_mm == 1000.0  # the geometry used last, as the form says for a new folder
    another = scan_folder(tmp_path, "zz_invented_e")
    page.scan_picker.set_path(str(another))
    page.analyze()
    page.set_scan_folder(scan_folder(tmp_path, "zz_invented_f"))  # another folder is chosen: the press is forgotten
    _settle(lambda: page.header_ready())
    _settle(seconds=0.2)
    assert len(runner.requests) == 2 and page.analyze_status.state_label.property("gimapJobState") == "idle"
    _close(page)


def test_closing_the_window_stops_analyze(tmp_path):
    runner = FakeClientRunner(wait_for_stop=True)
    page = _page(tmp_path, client=runner)
    _fill(page, scan_folder(tmp_path))
    page.analyze()
    _settle(lambda: runner.requests)
    assert page.running()
    page.dispose()
    assert page._client_should_stop()
    page.tasks.wait(10)
    _close(page)


# -- Details, the earlier method, the project and the language ---------------------------------------------------------


def test_a_folder_of_the_earlier_method_opens_in_details(tmp_path):
    page = _page(tmp_path)
    page.show()
    old = result_folder(tmp_path / "res", physical=True)
    assert page.open_client_result(old)
    assert page.details_shown() and page.result is not None and page.client_result is None
    assert not page.open_client_result(tmp_path / "nothing here", quiet=True)
    _close(page)


def test_the_film_thickness_is_offered_for_a_polymer_film_only(tmp_path):
    page = _page(tmp_path)
    page.show()
    page.show_details(True)
    page.substrate.setCurrentIndex(page.substrate.findData("Si"))
    assert not page.film_thickness.isVisible() and not page.film_label.isVisible()
    page.substrate.setCurrentIndex(page.substrate.findData("PSBMA"))
    assert page.film_thickness.isVisible() and page.film_label.isVisible()
    assert [page.substrate.itemData(i) for i in range(page.substrate.count())] == \
        ["Si", "TiO2", "PS", "PET", "PSBMA", "PNO", "PEO"]
    assert page.sample_metal.itemText(page.sample_metal.findData("Cu")) == "Cu"  # no note of the earlier method here
    _close(page)


def test_a_project_keeps_the_bar_the_options_and_the_result_shown(tmp_path):
    page = _page(tmp_path)
    out = write_result(tmp_path / "out")
    page.open_client_result(out)
    page.substrate.setCurrentIndex(page.substrate.findData("PNO"))
    page.sample_rate.setText("0.33")
    page.qcm_check.setChecked(True)
    page.mode_box.setCurrentIndex(page.mode_box.findData("refine"))
    page.duration.setText("16")
    state = page.project_state()
    assert state["client"]["result_folder"] == str(out) and json.dumps(state)  # plain values only
    other = _page(tmp_path / "other")
    assert other.apply_project_state(state) == []
    assert (other.substrate.currentData(), other.sample_rate.text(), other.qcm_check.isChecked()) == ("PNO", "0.33", True)
    assert other.mode_box.currentData() == "refine" and other.duration.text() == "16"
    assert other.client_result is not None and other.client_result.folder == out
    other.apply_project_state({})  # an older project, or a new one: nothing of Analyze
    assert other.client_result is None and other.sample_rate.text() == "" and other.mode_box.currentData() == "auto"
    assert other.result_stack.currentIndex() == 0 and not other.details_shown()
    _close(page)
    _close(other)


def test_the_main_screen_speaks_chinese_after_a_switch(tmp_path):
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

    page = _page(tmp_path)
    page.show()
    _fill(page, scan_folder(tmp_path))
    page.open_client_result(write_result(tmp_path / "out"))
    widgets = [w for w in page.findChildren((QLabel, QAbstractButton)) if w.text()]
    english = {id(w): w.text() for w in widgets}
    try:
        apply_language("zh", roots=(page,))
        page.refresh_language()
        kept = sorted({english[id(w)] for w in widgets if w.text() == english[id(w)]
                       and re.search(r"[A-Za-z]{3,}", re.sub(r"<[^>]+>|PyTorch|Python|GIMaP|PILATUS|S/N|PSBMA", "", w.text()))
                       and not re.fullmatch(r"[\w.\- ·/]+", w.text())})  # names (scan, sample, family, package) are data
        assert kept == []
        assert page.analyze_button.text() == "分析" and page.view_switch.itemText(0) == "主屏"
        assert page.window_label.text() == "t = 2 s · 已沉积 0.4 nm"
        assert [page.entry_control.itemText(i) for i in range(2)] == ["候选 1 · 3.2", "候选 2 · 4.4"]
        assert page.entry_control.itemText(3) == "精修解 · 2.1" and page.client_entry_label() == "candidate 2"
        assert page.patterns.titles == ("实测图", "模型图：候选 2")
        assert page.geometry_summary.full_text().endswith("光束中心由帧估计 · 滚转角 0°")
        assert page.tiles["gap_nm"].findChild(QLabel).text() == "间隙"
        assert page.deviation_caption.text() == "拟合偏差（越小越好）"
        legend = [t.get_text() for t in page.evolution.figure.axes[0].get_legend().get_texts()]
        assert legend == ["半径 R", "高度 H", "间距 D", "间隙"]
        page.show_client_result(None)
        assert page.client_empty.title_label.text() == "尚无结果"
        page.alpha.setText("")
        assert page.geometry_summary.full_text() == "还需填写的几何：入射角"
    finally:
        apply_language(DEFAULT_LANGUAGE, roots=(page,))
        page.refresh_language()
    assert page.analyze_button.text() == "Analyze" and page.client_empty.title_label.text() == "No result yet"
    assert page.geometry_summary.full_text() == "Geometry to enter: incidence angle"
    _close(page)


def test_every_template_of_the_main_screen_is_in_the_chinese_table():
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.application import CLIENT_TEMPLATES
    from src.gimap.features.sputter.presentation import client_texts, evolution_view, pattern_view, real_space_view
    from src.gimap.features.sputter.presentation.views import sputter_main_view as view
    from src.gimap.features.sputter.presentation.views.sputter_page_view import CU_NOTE, EARLIER_METHOD, EARLIER_METHOD_TIP

    held = [*CLIENT_TEMPLATES, *client_texts.TEMPLATES, *pattern_view.TEMPLATES, *evolution_view.TEMPLATES,
            *real_space_view.TEMPLATES, *(text for label, _key, tip in view.IMAGE_CHOICES for text in (label, tip)),
            *(text for _key, caption, _unit, tip in view.TILES for text in (caption, tip)),
            *(text for label, _key, tip in view.MODE_CHOICES for text in (label, tip)),
            *(label for label, _key in view.VIEW_CHOICES), view.DEVIATION_CAPTION, view.DEVIATION_TIP, view.ENTRY_HINT,
            view.RATE_TIP, view.QCM_TIP, view.ANALYZE_TIP, view.FILM_TIP, view.ONSET_TIP, view.DURATION_TIP,
            view.SCAN_FOLDER_TIP, CU_NOTE, EARLIER_METHOD, EARLIER_METHOD_TIP, ", "]
    missing = [template for template in held if template not in ZH]
    assert missing == []
    for template in held:  # a translated template keeps its fields
        assert sorted(re.findall(r"\{(\w+)\}", template)) == sorted(re.findall(r"\{(\w+)\}", ZH[template])), template


def _picture_texts(page) -> list:
    return [item.get_text() for item in page.real_space.figure.texts]


def test_the_solution_shown_can_be_seen_in_real_space(tmp_path):
    """The switch Patterns | Real Space: the picture follows the window and the solution; patterns again on the switch."""
    page = _page(tmp_path)
    page.show()
    windows = [window(2.0, [entry("candidate 1", 3.0, 2.0, rank=1, base_nm=1.5), entry("candidate 2", 4.0, 3.0, rank=2)],
                      smooth="candidate 1"),
               window(4.0, [entry("candidate 1", 3.5, 4.0, rank=1)], smooth="candidate 1")]
    assert page.open_client_result(write_result(tmp_path / "out", windows=windows))
    assert page.image_switch.currentData() == "patterns" and page.pattern_host.currentIndex() == 0
    assert page.real_space.drawn is None  # nothing is drawn while the patterns are shown
    page.image_switch.button(1).click()
    assert page.pattern_host.currentIndex() == 1
    space = page.real_space.drawn
    assert (space.radius, space.distance, space.base, space.height) == (2.0, 8.0, 1.5, 3.0)
    side, top = page.real_space.figure.axes
    assert side.get_title(loc="left") == "side view (cut)" and top.get_title(loc="left") == "top view"
    assert side.get_xlim() == (0.0, space.side) and top.get_xlim() == (0.0, space.field)
    # the caps of one row, twice: their outlines, then their fills over the inner outlines (touching caps as one piece)
    assert [len(collection.get_paths()) for collection in side.collections] == [6, 6]
    assert any("Candidate 1" in words for words in _picture_texts(page))
    page.entry_control.button(1).click()  # another solution: no base layer, another radius
    assert (page.real_space.drawn.radius, page.real_space.drawn.base) == (3.0, 0.0)
    page.select_window(1)
    assert page.real_space.drawn.radius == 4.0
    page.image_switch.button(0).click()
    assert page.pattern_host.currentIndex() == 0 and page.patterns.titles[1] == "Model image: Candidate 1"
    # a porous result without the geometry of its metal: nothing to draw, and the picture says why
    page.open_client_result(write_result(tmp_path / "porous", windows=porous_windows(), wanted=POROUS_WANTED))
    page.image_switch.button(1).click()
    assert page.real_space.drawn is None and any("no arrangement to draw" in words for words in _picture_texts(page))
    _close(page)


def test_coalescing_grains_and_a_porous_layer_are_drawn_as_they_are(tmp_path):
    from src.gimap.features.sputter.application import PorousSpace

    page = _page(tmp_path)
    page.show()
    windows = [window(2.0, [entry("candidate 1", 3.0, 1.0, rank=1, D_nm=10.0)], smooth="candidate 1"),
               window(4.0, [entry("candidate 1", 3.0, 5.2, rank=1, D_nm=10.0, base_nm=2.0)], smooth="candidate 1")]
    assert page.open_client_result(write_result(tmp_path / "out", windows=windows))
    page.image_switch.button(1).click()
    assert page.real_space.arrangement == "islands" and "Separate islands" in _picture_texts(page)
    page.select_window(1)  # grains of 5.2 nm at 10 nm: a network across the view, on a continuous layer
    assert page.real_space.arrangement == "network"
    assert "Continuous layer of 2 nm; the grains on it: connected across the view" in _picture_texts(page)
    porous = dict(R_nm=0.9, width=0.2, f_islands=0.2, cap_share=0.2, depth_nm=2.0, xi_nm=5.0, film_share=0.05,
                  amount_nm=8.0, particle_nm=75.0, pore_depth_nm=400.0, pore_width_nm=60.0, cap_cover=0.3,
                  cap_thickness_nm=10.0, island_amount_nm=1.5, cap_amount_nm=0.5, pore_amount_nm=2.0,
                  cluster_height_nm=1.8, cluster_distance_nm=2.0)
    record = window(3.0, [dict(label="candidate 1", kind="candidate", rank=1, weight=1.0, fit_deviation=4.0, **porous)],
                    smooth="candidate 1")
    assert page.open_client_result(write_result(tmp_path / "tio2", windows=[record], wanted=POROUS_WANTED))
    space = page.real_space.drawn
    assert isinstance(space, PorousSpace) and (space.pore_depth, space.cap_cover, space.deposited) == (400.0, 0.3, 8.0)
    layer, zoom = page.real_space.figure.axes
    assert layer.get_title(loc="left") == "cross-section of the TiO₂ layer"
    assert zoom.get_title(loc="left") == "top of a particle (enlarged)"
    assert "cap on 30% of the tops, 10 nm" in _picture_texts(page)
    assert any(words.startswith("Of the 8 nm deposited") for words in _picture_texts(page))
    _close(page)


def test_the_real_space_picture_speaks_chinese_after_a_switch(tmp_path):
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

    page = _page(tmp_path)
    page.show()
    assert page.open_client_result(write_result(tmp_path / "out"))
    page.image_switch.button(1).click()
    apply_language("zh", roots=(page,))
    page.refresh_language()
    try:
        side, top = page.real_space.figure.axes
        assert side.get_title(loc="left") == "侧视（截面）" and top.get_title(loc="left") == "俯视"
        assert page.image_switch.itemText(1) == "实空间" and page.image_switch.itemText(0) == "图样"
    finally:
        apply_language(DEFAULT_LANGUAGE, roots=(page,))
        page.refresh_language()
        _close(page)


# -- a porous substrate and a series of static samples -----------------------------------------------------------------


def _legend(axes) -> list:
    return [item.get_text() for item in axes.get_legend().get_texts()]


def test_a_porous_result_shows_its_own_tiles_and_plots(tmp_path):
    page = _page(tmp_path)
    page.show()
    page.open_client_result(write_result(tmp_path / "out", windows=porous_windows(), wanted=POROUS_WANTED))
    # no height, distance, gap or coverage: their tiles are hidden; the island share may be above 1
    assert _tiles(page) == {"R_nm": "1.2", "width": "0.17", "f_islands": "2", "cap_share": "0.25", "depth_nm": "1.62",
                            "xi_nm": "4.8", "film_share": "0.04"}
    assert page.tile_captions["R_nm"].full_text() == "Island radius (relative)"
    assert "relative size" in page.tiles["R_nm"].toolTip() and "pores" in page.tiles["xi_nm"].toolTip()
    assert page.window_label.text() == "t = 3 s · 0.6 nm deposited" and page.entry_control.count() == 2
    sizes, shares = page.evolution.figure.axes[:2]
    assert _legend(sizes) == ["Island radius (relative)", "Vertical correlation", "Lateral correlation"]
    assert _legend(shares) == ["Island share", "Cap share", "Film share"]
    assert (sizes.get_ylabel(), shares.get_ylabel(), shares.get_xlabel()) == ("Size (nm)", "Share", "Time in the scan (s)")
    assert shares.get_ylim()[1] >= 2.0  # an island share above 1 stays inside the plot
    assert page.evolution.series["xi_nm"] == ([3.0, 6.0, 9.0], [4.8, 8.0, 12.0])
    assert np.isnan(page.evolution.series["coverage"][1]).all() and np.isnan(page.evolution.series["H_nm"][1]).all()
    page.entry_control.button(1).click()  # the other candidate: the plots follow as for any result
    assert _tiles(page)["R_nm"] == "1.7" and page.evolution.series["xi_nm"][1][0] == pytest.approx(6.8)
    # a result on a flat substrate afterwards: a radius again, and the coverage alone at the right
    page.open_client_result(write_result(tmp_path / "flat"))
    assert page.tile_captions["R_nm"].full_text() == "Radius R" and "relative" not in page.tiles["R_nm"].toolTip()
    assert set(_tiles(page)) == {"R_nm", "H_nm", "D_nm", "gap_nm", "width", "coverage"}
    sizes, shares = page.evolution.figure.axes[:2]
    assert _legend(sizes) == ["Radius R", "Height H", "Distance D", "Gap"]
    assert shares.get_ylabel() == "Coverage" and shares.get_legend() is None
    _close(page)


@pytest.mark.parametrize("width, height", [(1256, 869), (1096, 769)])
def test_the_seven_tiles_of_a_porous_result_fit_the_window(tmp_path, width, height):
    from PyQt5.QtWidgets import QWidget

    host = QWidget()
    host.resize(width, height)
    page = _page(tmp_path)
    page.setParent(host)
    page.setGeometry(0, 0, width, height)
    host.show()
    page.open_client_result(write_result(tmp_path / "out", windows=porous_windows(), wanted=POROUS_WANTED))
    _settle(seconds=0.2)
    margins = page.layout().contentsMargins()
    assert page.client_view.minimumSizeHint().width() <= width - margins.left() - margins.right()
    shown = [tile for tile in page.tiles.values() if not tile.isHidden()]
    assert len(shown) == 7
    for widget in (*shown, page.deviation_value, page.evolution.canvas):
        corner = widget.mapTo(page, widget.rect().bottomRight())
        assert widget.isVisible() and corner.x() < width and corner.y() < height, widget.objectName()
        assert widget.width() >= widget.minimumSizeHint().width(), widget.objectName()
    assert all(page.tile_values[key].width() >= page.tile_values[key].sizeHint().width() for key in _tiles(page))
    _close(page)
    host.deleteLater()


def test_a_static_series_is_shown_sample_by_sample_and_never_speaks_of_time(tmp_path):
    page = _page(tmp_path)
    page.show()
    assert page.open_client_result(write_static(tmp_path / "zz_series")) and not page.details_shown()
    assert page.window_slider.maximum() == 3 and page.window_label.text() == "zz_a · nominal 4 nm"
    assert page.window_label.toolTip() == "Sample 1 of 4: 4 frames. Substrate: Si."
    assert (page.window_previous.toolTip(), page.window_next.toolTip(), page.window_slider.toolTip()) == \
        ("Previous sample of the series", "Next sample of the series", "Sample shown")
    assert page.result_name.text() == "zz_series · Au · zz_static_family"
    assert _tiles(page) == {"R_nm": "2", "H_nm": "2.4", "D_nm": "6", "gap_nm": "2", "coverage": "0.4"}
    assert page.tiles["R_nm"].toolTip().splitlines() == ["Radius of the grains.", "Range: 1.8 – 2.4 nm"]
    assert page.tiles["coverage"].toolTip().endswith("Range: 0.36 – 0.48") and "Range" not in page.tiles["gap_nm"].toolTip()
    assert page.entry_control.count() == 1 and page.entry_control.itemText(0) == "Model solution · 3.1"
    assert page.patterns.titles == ("Measured image", "Model image: Model solution") and page.report_button.isEnabled()
    sizes, shares = page.evolution.figure.axes[:2]
    assert sizes.get_xlabel() == shares.get_xlabel() == "Nominal thickness (nm)"  # what the series says its axis is
    assert page.evolution.series["D_nm"] == ([4.0, 4.0, 7.0, 10.0], [6.0, 7.5, 10.5, 15.0])
    # two samples at one nominal thickness: a click takes the one whose point is nearer
    assert page.evolution.window_at(0, 4.0, 7.4) == 1 and page.evolution.window_at(0, 4.0, 6.1) == 0
    assert page.evolution.window_at(0, 9.0) == 3 and page.evolution.window_at(1, 6.5, 0.4) == 2
    page.select_window(page.evolution.window_at(0, 4.0, 7.4))
    assert page.client_window().label == "zz_b" and page.window_label.text() == "zz_b · nominal 4 nm"
    assert page.evolution._marks[0].get_xdata()[0] == 4.0 and 7.5 in list(page.evolution._rings[0].get_ydata())
    page.window_slider.setValue(3)
    assert page.window_label.toolTip() == "Sample 4 of 4: 3 frames. Substrate: bare substrate measured."
    assert _tiles(page)["R_nm"] == "5" and not page.window_next.isEnabled()
    words = [f"{widget.text()} {widget.toolTip()}" for widget in page.client_view.findChildren((QLabel, QAbstractButton))
             if widget.isVisible()] + [page.window_slider.toolTip(), *(tile.toolTip() for tile in page.tiles.values())]
    plots = [item.get_text() for axes in (*page.patterns.figure.axes, *page.evolution.figure.axes)
             for item in (axes.title, axes.xaxis.label, axes.yaxis.label)]
    assert [text for text in words + plots if re.search(r"\btime\b|\bt = |deposited|\bscan\b", text, re.I)] == []
    # a result of a scan afterwards has its time words back
    page.open_client_result(write_result(tmp_path / "scan"))
    assert page.window_label.text().startswith("t = 2 s") and page.window_next.toolTip() == "Next time window"
    assert page.evolution.figure.axes[0].get_xlabel() == "Time in the scan (s)"
    _close(page)


# -- Details: Static Frames --------------------------------------------------------------------------------------------


def _static_page(tmp_path, **more):
    runner = FakeStaticRunner(tmp_path / "series", **more)
    page = _page(tmp_path, static=runner, preferences=InMemoryUserPreferencesRepository())
    page.show()
    frames = tmp_path / "zz frames"
    frames.mkdir(parents=True, exist_ok=True)
    page.static_frames.set_path(str(frames))
    return page, runner


def test_the_static_section_starts_with_the_geometry_of_the_package(tmp_path):
    page, runner = _static_page(tmp_path)
    page.show_details(True)
    assert not page.static_section.is_expanded() and page.static_section.toggle_button.text() == "Static Frames (after deposition)"
    page.static_section.set_expanded(True)
    assert page.static_note.text() == ("Au films measured after the deposition at an incidence angle near 0.35°, one "
                                       "sample at a time into a series.")
    # the package's own values (the energy in keV, the beam centre in GIMaP's coordinates), the series' default folder
    assert [getattr(page, name).text() for name in ("static_energy", "static_sdd", "static_beam_col", "static_beam_row",
                                                    "static_roll")] == ["9.5", "4321", "300.5", "700.5", "-0.25"]
    assert page.static_series.path() == "" and page.static_series.path_edit.placeholderText() == str(
        tmp_path / "series" / "series_1")
    assert page.static_mode.currentData() == "quick" and page.static_button.isEnabled()
    page.static_sdd.setText("5000")
    page.static_roll.setText("")
    page.static_mask.set_path(str(tmp_path / "zz_mask.npy"))
    request = page._static_request()
    assert (request.energy_ev, request.sdd_mm, request.beam_col, request.sample_roll_deg) == (9500.0, 5000.0, 300.5, None)
    assert request.series_folder == tmp_path / "series" / "series_1" and request.mask_file == tmp_path / "zz_mask.npy"
    page.static_defaults_button.click()  # Package Values: all of them again, and the package's mask
    assert (page.static_sdd.text(), page.static_roll.text(), page.static_mask.path()) == ("4321", "-0.25", "")
    _close(page)
    # a package that does not know static frames: said in the section, nothing can start
    older, _runner = _static_page(tmp_path / "older", problem=STATIC_UNSUPPORTED)
    assert "no entry for static frames" in older.static_note.text() and not older.static_button.isEnabled()
    assert older.static_energy.text() == "" and older.analyze_button.isEnabled()
    older.analyze_static()
    assert not older.static_analyzing()
    _close(older)
    plain = _page(tmp_path / "plain")  # a service without static frames
    assert not plain.static_button.isEnabled() and plain.static_project_state()["static_sdd"] == ""
    _close(plain)


def test_analyze_sample_adds_to_the_series_and_shows_it(tmp_path):
    from src.gimap.app.presentation.components.toast import visible_toasts

    page, runner = _static_page(tmp_path)
    page.show_details(True)
    page.static_section.set_expanded(True)
    page.static_name.setText("zz 8nm")
    page.static_nominal.setText("8")
    seen = []
    page.staticEvent.connect(lambda _event: seen.append(page.static_status.message_label.text()))
    page.static_button.click()
    assert page.static_analyzing() and page.running()
    assert not page.static_button.isEnabled() and not page.analyze_button.isEnabled()  # one analysis at a time
    _settle(lambda: not page.static_analyzing())
    request = runner.requests[0]
    assert (request.frames, request.name, request.nominal_nm, request.mode) == (str(tmp_path / "zz frames"), "zz 8nm",
                                                                                8.0, "quick")
    assert request.reference_frames == "" and request.series_folder == tmp_path / "series" / "series_1"
    assert page.static_status.state_label.property("gimapJobState") == "succeeded"
    assert page.static_status.message_label.text() == "Done: 1 samples in the series."
    assert "Step 1 of 3: Reading the frames…" in seen and "Step 2 of 3: Finding the solution of the sample…" in seen
    assert "Step 3 of 3: Writing the report and the images of the series…" in seen
    assert not page.details_shown() and page.client_result.static and page.window_label.text() == "zz_8nm · nominal 8 nm"
    assert page.static_button.isEnabled() and page.analyze_button.isEnabled()
    assert "zz: 4 frames, 4 usable" in page.static_log.toPlainText()
    assert any("zz_8nm is in the series" in toast.text() for toast in visible_toasts(page))
    stored = page.preferences.get("sputter_page")
    assert stored["last_client_result"] == str(tmp_path / "series" / "series_1")
    assert stored["static_last"]["static_sdd"] == "4321" and stored["static_last"]["mode"] == "quick"
    # a thinner sample into the same series: it is the one shown, at its place along the nominal thickness
    reference = tmp_path / "zz bare"
    reference.mkdir()
    page.static_name.setText("zz 4nm")
    page.static_nominal.setText("4")
    page.static_reference.set_path(str(reference))
    page.static_mode.setCurrentIndex(page.static_mode.findData("refine"))
    page.analyze_static()
    _settle(lambda: not page.static_analyzing())
    assert (runner.requests[1].reference_frames, runner.requests[1].mode) == (str(reference), "refine")
    assert [window.label for window in page.client_result.windows] == ["zz_4nm", "zz_8nm"]
    assert page.client_window().label == "zz_4nm" and page.static_status.message_label.text().startswith("Done: 2 samples")
    # the same request again: the series as it is
    page.analyze_static()
    _settle(lambda: not page.static_analyzing())
    assert len(runner.requests) == 2 and "analysed before" in page.static_status.message_label.text()
    _close(page)


def test_analyze_sample_says_what_is_missing_and_asks_before_it_replaces(tmp_path, monkeypatch):
    page, runner = _static_page(tmp_path)
    page.static_frames.set_path("")
    page.static_button.click()
    assert not page.static_analyzing() and runner.requests == []
    assert page.static_status.state_label.property("gimapJobState") == "failed"
    message = page.static_status.message_label.text()
    assert "Choose the frames of the sample" in message and "Give the sample a name" in message
    assert "Give the nominal thickness" in message
    page.static_frames.set_path(str(tmp_path / "zz frames"))
    page.static_name.setText("zz_8nm")
    page.static_nominal.setText("8")
    page.analyze_static()
    _settle(lambda: not page.static_analyzing())
    asked = []
    monkeypatch.setattr(page, "_confirm_static_replace", lambda name: asked.append(name) or False)
    page.static_nominal.setText("9")  # the same name with another thickness
    page.analyze_static()
    assert asked == ["zz_8nm"] and not page.static_analyzing() and len(runner.requests) == 1  # not replaced
    monkeypatch.setattr(page, "_confirm_static_replace", lambda name: True)
    page.analyze_static()
    _settle(lambda: not page.static_analyzing())
    assert len(runner.requests) == 2 and page.client_window().nominal_nm == 9.0
    _close(page)


def test_cancel_stops_analyze_sample_and_a_failure_keeps_its_reason(tmp_path):
    page, runner = _static_page(tmp_path, wait_for_stop=True)
    page.static_name.setText("zz_8nm")
    page.static_nominal.setText("8")
    page.analyze_static()
    _settle(lambda: runner.requests)
    _settle(lambda: "Step 1 of 3" in page.static_status.message_label.text(), 5)
    page.analyze()  # Analyze waits for this analysis
    assert page.static_analyzing() and not page.analyzing() and page.static_status.cancel_button.isEnabled()
    page.static_status.cancel_button.click()
    assert "Stopping" in page.static_status.message_label.text()
    _settle(lambda: not page.static_analyzing())
    assert page.static_status.state_label.property("gimapJobState") == "cancelled"
    assert "stay in the series" in page.static_status.message_label.text() and page.static_button.isEnabled()
    _close(page)
    failing, _runner = _static_page(tmp_path / "b", status="failed at predict")
    failing.static_name.setText("zz_8nm")
    failing.static_nominal.setText("8")
    failing.analyze_static()
    _settle(lambda: not failing.static_analyzing())
    assert failing.static_status.state_label.property("gimapJobState") == "failed"
    assert "(failed at predict)" in failing.static_status.message_label.text() and failing.client_result is None
    assert "the last line of the invented log" in failing.static_log.toPlainText()
    failing.static_status.details_button.click()  # Details: the section opens at its log
    assert failing.details_shown() and failing.static_section.is_expanded() and failing.static_log.isVisible()
    closing, runner = _static_page(tmp_path / "c", wait_for_stop=True)
    closing.static_name.setText("zz_8nm")
    closing.static_nominal.setText("8")
    closing.analyze_static()
    _settle(lambda: runner.requests)
    closing.dispose()  # the window closes: the package is told to stop
    assert closing._static_should_stop()
    closing.tasks.wait(10)
    _close(failing)
    _close(closing)


def test_a_zip_archive_asks_which_of_its_folders_holds_the_frames(tmp_path, monkeypatch):
    page, runner = _static_page(tmp_path)
    archive = str(tmp_path / "zz frames.zip")
    asked = []
    monkeypatch.setattr(page, "_ask_archive_folder", lambda folders: asked.append(folders) or folders[1])
    runner.archive = ["zz_8nm/"]
    page.set_static_source(page.static_frames, archive)  # frames in one folder: nothing to ask
    assert page.static_frames.path() == archive and asked == []
    runner.archive = ["", "zz_8nm/", "zz_bare/"]
    page.set_static_source(page.static_reference, archive)
    assert asked == [["", "zz_8nm/", "zz_bare/"]] and page.static_reference.path() == archive + "::zz_8nm/"
    monkeypatch.setattr(page, "_ask_archive_folder", lambda folders: None)  # cancelled: the field stays
    page.set_static_source(page.static_reference, str(tmp_path / "another.zip"))
    assert page.static_reference.path() == archive + "::zz_8nm/"
    runner.archive = []
    page.set_static_source(page.static_frames, str(tmp_path / "empty.zip"))  # no frames in it: said, not taken
    assert page.static_frames.path() == archive
    page.set_static_source(page.static_frames, str(tmp_path / "zz_one.cbf"))  # one frame: as it is
    assert page.static_frames.path() == str(tmp_path / "zz_one.cbf")
    _close(page)


def test_the_masks_of_the_package_are_offered_by_their_file_names(tmp_path):
    client = FakeClientRunner()
    page = _page(tmp_path, client=client, static=FakeStaticRunner(tmp_path / "series"))
    for button, picker in ((page.package_masks_button, page.mask_picker), (page.static_masks_button, page.static_mask)):
        menu = button.menu()
        menu.aboutToShow.emit()
        assert [action.text() for action in menu.actions()] == ["The analysis package brings no masks"]
        assert not menu.actions()[0].isEnabled()
        client.mask_files = [tmp_path / "assets" / "zz_first_mask.npy", tmp_path / "assets" / "zz_second&mask.npy"]
        menu.aboutToShow.emit()
        assert [action.text() for action in menu.actions()] == ["zz_first_mask.npy", "zz_second&&mask.npy"]
        menu.actions()[1].trigger()
        assert picker.path() == str(tmp_path / "assets" / "zz_second&mask.npy")
        client.mask_files = []
    assert page._values()["mask"] == page.mask_picker.path()  # Analyze and the earlier method use the field
    _close(page)


def test_a_project_keeps_the_static_section(tmp_path):
    page, _runner = _static_page(tmp_path)
    page.static_name.setText("zz_8nm")
    page.static_nominal.setText("8")
    page.static_series.set_path(str(tmp_path / "zz_series"))
    page.static_mode.setCurrentIndex(page.static_mode.findData("full"))
    page.static_beam_col.setText("")
    page.static_section.set_expanded(True)
    state = page.project_state()
    assert json.dumps(state) and state["static"]["frames"] == str(tmp_path / "zz frames")
    other, _other = _static_page(tmp_path / "other")
    assert other.apply_project_state(state) == []
    assert (other.static_name.text(), other.static_nominal.text(), other.static_mode.currentData()) == ("zz_8nm", "8", "full")
    assert other.static_frames.path() == str(tmp_path / "zz frames") and other.static_section.is_expanded()
    assert (other.static_beam_col.text(), other.static_beam_row.text()) == ("", "700.5")  # an emptied field stays empty
    assert other.static_series.path() == str(tmp_path / "zz_series")
    other.apply_project_state({})  # an older project: an empty section with the geometry of the package
    assert (other.static_name.text(), other.static_frames.path(), other.static_mode.currentData()) == ("", "", "quick")
    assert other.static_beam_col.text() == "300.5" and not other.static_section.is_expanded()
    _close(page)
    _close(other)


def test_static_frames_speak_chinese_after_a_switch(tmp_path):
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language

    page, _runner = _static_page(tmp_path)
    page.open_client_result(write_static(tmp_path / "zz_series"))
    page.static_button.click()  # nothing to analyse yet: what is missing, in the status
    try:
        apply_language("zh", roots=(page,))
        page.refresh_language()
        assert page.window_label.text() == "zz_a · 名义厚度 4 nm" and page.window_next.toolTip() == "下一个样品"
        assert page.window_label.toolTip() == "第 1 个样品，共 4 个：用了 4 帧。 基底：Si。"
        assert page.entry_control.itemText(0) == "模型解 · 3.1" and page.patterns.titles[1] == "模型图：模型解"
        assert page.evolution.figure.axes[0].get_xlabel() == "名义厚度 (nm)"
        assert page.tiles["R_nm"].toolTip().splitlines()[-1] == "范围：1.8 – 2.4 nm"
        assert page.static_section.toggle_button.text() == "静态帧（沉积之后测量）"
        assert page.static_button.text() == "分析这个样品" and page.static_note.text().startswith("沉积之后在入射角约 0.35°")
        assert "请给样品起一个含字母或数字的名字。" in page.static_status.message_label.text()
        assert page.package_masks_button.text() == page.static_masks_button.text() == "分析包自带的掩膜"
        page.open_client_result(write_result(tmp_path / "porous", windows=porous_windows(), wanted=POROUS_WANTED))
        assert page.tile_captions["R_nm"].full_text() == "岛半径（相对值）"
        assert _legend(page.evolution.figure.axes[1]) == ["岛份额", "帽层份额", "成膜份额"]
        assert page.evolution.figure.axes[1].get_ylabel() == "份额"
    finally:
        apply_language(DEFAULT_LANGUAGE, roots=(page,))
        page.refresh_language()
    assert page.tile_captions["R_nm"].full_text() == "Island radius (relative)"
    assert page.static_button.text() == "Analyze Sample" and "Give the sample a name" in page.static_status.message_label.text()
    _close(page)


def test_every_template_of_static_frames_and_of_a_porous_result_is_in_the_chinese_table():
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.application import STATIC_STEP_TEXT, STATIC_TEMPLATES
    from src.gimap.features.sputter.presentation import static_flow
    from src.gimap.features.sputter.presentation.views import sputter_static_view as view

    held = [*STATIC_TEMPLATES, *STATIC_STEP_TEXT.values(), *static_flow.TEMPLATES, view.STATIC_TITLE, view.STATIC_TIP,
            *(text for label, _key, tip in view.STATIC_MODE_CHOICES for text in (label, tip)), view.FRAMES_TIP,
            view.NAME_TIP, view.NOMINAL_TIP, view.REFERENCE_TIP, view.SERIES_TIP, view.GEOMETRY_NOTE, view.CENTRE_TIP,
            view.MASKS_BUTTON, view.MASKS_TIP, "Analyze Sample", "Package Values", "File…", "Sample name",
            "Bare substrate", "Series folder", "of the package", "of the package (.npy)",
            "folder, CBF file or zip archive", "Choose the Folder of the Frames", "Choose a CBF File or a Zip Archive",
            "Choose the Folder of the Series"]
    missing = [template for template in held if template not in ZH]
    assert missing == []
    for template in held:  # a translated template keeps its fields
        assert sorted(re.findall(r"\{(\w+)\}", template)) == sorted(re.findall(r"\{(\w+)\}", ZH[template])), template
