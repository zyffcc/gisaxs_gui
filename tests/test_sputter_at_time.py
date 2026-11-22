"""The structure at any time of an analysed scan (2026-11-22): a time between the windows, analysed on demand by the
package (``iampred.at_time``: the joint path interpolated there, then refined to that time's frames)."""

from __future__ import annotations

import json

from src.gimap.app.presentation.task_runner import TaskRunner
from src.gimap.features.sputter.infrastructure import ClientFolderReader
from tests.sputter_client_fixtures import FakeClientRunner, write_at_time, write_result
from tests.sputter_fixtures import FakeInspector


def _page(tmp_path, runner=None):
    from src.gimap.features.sputter.bootstrap import create_client_service, create_sputter_service
    from src.gimap.features.sputter.presentation.page import SputterPage

    reader = ClientFolderReader(lambda: tmp_path / "user")
    service = create_sputter_service(inspector=FakeInspector(), results_root=lambda: tmp_path / "results",
                                     client=create_client_service(runner=runner or FakeClientRunner(), reader=reader))
    page = SputterPage(service, task_runner=TaskRunner())
    page.resize(1256, 869)
    return page


def _close(page) -> None:
    page.dispose()
    page.tasks.wait(10)
    page.close()
    page.deleteLater()


def test_a_time_on_demand_joins_the_windows_of_its_result_until_the_scan_is_analysed_again(tmp_path):
    folder = write_result(tmp_path / "out")
    write_at_time(folder, 3.0, 8)
    result = ClientFolderReader().read(folder)
    times = [window.time_s for window in result.windows]
    assert times == sorted(times) and 3.0 in times
    asked = next(window for window in result.windows if window.time_s == 3.0)
    assert asked.on_demand and asked.labels() == ["interpolated", "refined"] and asked.default_label() == "refined"
    assert not any(window.on_demand for window in result.windows if window.time_s != 3.0)
    write_at_time(folder, 2.0, 8)  # a time the analysis has a window for keeps that window
    assert not next(window for window in ClientFolderReader().read(folder).windows if window.time_s == 2.0).on_demand
    main = folder / "main" / "main.json"  # analysed again: another main.json, the times asked for before are dropped
    main.write_text(json.dumps(dict(json.loads(main.read_text(encoding="utf-8")), note="analysed again")),
                    encoding="utf-8")
    assert not any(window.on_demand for window in ClientFolderReader().read(folder).windows)


def test_a_time_is_interpolated_at_once_then_refined_and_shown_as_a_window(tmp_path):
    runner = FakeClientRunner()
    page = _page(tmp_path, runner)
    page.show()
    assert page.open_client_result(write_result(tmp_path / "out"))
    assert page.at_time_row.isVisibleTo(page)
    assert (page.at_time_value.minimum(), page.at_time_value.maximum()) == (2.0, 8.0)  # the windows' times
    page.at_time_value.setValue(5.0)
    page.at_time_button.click()
    assert not page.at_time_button.isEnabled() and "interpolating" in page.at_time_note.full_text()
    page.tasks.wait(10)
    page.tasks.wait(10)
    assert [(time_s, refine) for _folder, time_s, refine in runner.time_requests] == [(5.0, 0), (5.0, 8)]
    window = page.client_window()
    assert window.time_s == 5.0 and window.on_demand and window.chosen_label() == "refined"
    assert "on demand" in page.window_label.text() and page.tile_values["R_nm"].text() == "2.6"
    assert page.at_time_button.isEnabled() and "refined to the frames of this time" in page.at_time_note.full_text()
    page.select_window(0)
    assert not page.client_window().on_demand
    _close(page)


def test_no_row_without_a_package_that_can_analyse_a_time(tmp_path):
    runner = FakeClientRunner()
    runner.at_time = False
    page = _page(tmp_path, runner)
    page.show()
    assert page.open_client_result(write_result(tmp_path / "out"))
    assert not page.at_time_row.isVisibleTo(page)
    _close(page)
