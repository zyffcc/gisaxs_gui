"""The Sputter page and the speed of an analysis: the parallel processes the package runs with (Details ▸ Analyze
options), what the live view says about reading the frames from disk while the import runs, and the time of every step
in Details ▸ Report.

Offline, with a fake of the package's process, invented live events and invented results (no user data).
"""

from __future__ import annotations

import json
import os
import re
import shutil

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.gimap.features.sputter.application import ClientProgress, ClientService, LiveEvent, LiveRun
from src.gimap.integrations.state import InMemoryUserPreferencesRepository
from tests.sputter_client_fixtures import FakeClientRunner, write_result
from tests.sputter_fixtures import scan_folder
from tests.test_sputter_client import _Process, _request, _runner, _sample, _static
from tests.test_sputter_main_screen import _close, _fill, _page, _settle

SLOW_NOTE = ("Reading the frames from this disk is the slow part (a USB or network drive, or an on-access virus scan "
             "of every file). Copying the scan folder to a local SSD first makes the import faster.")


@pytest.fixture(autouse=True)
def _no_package_of_this_computer(monkeypatch):
    monkeypatch.delenv("GIMAP_SPUTTER_CLIENT", raising=False)
    monkeypatch.setattr(ClientService, "cores", staticmethod(lambda: 8))  # the same choices on every computer


class _WorkersRunner(FakeClientRunner):
    """A package runner that says how many processes the computer takes now."""

    def workers(self) -> int:
        return 5


def _frames(done: int, **timing) -> LiveEvent:
    return LiveEvent("frames", "intake", {"chunk": done // 128, "done": done, "total": 1241, **timing})


def _items(box) -> list:
    return [box.itemText(index) for index in range(box.count())]


# -- the parallel processes ------------------------------------------------------------------------------------------


def test_the_parallel_processes_reach_the_request_and_are_kept(tmp_path):
    preferences = InMemoryUserPreferencesRepository()
    runner = _WorkersRunner()
    page = _page(tmp_path, client=runner, preferences=preferences)
    page.show()
    page.show_details(True)
    box = page.workers_box
    assert _items(box) == ["Automatic (5)", "1", "2", "3", "4", "5", "6", "7", "8"] and box.currentIndex() == 0
    _fill(page, scan_folder(tmp_path / "scan"))
    page.analyze_button.click()
    _settle(lambda: not page.analyzing() and bool(runner.requests))
    assert runner.requests[-1].workers == 5  # Automatic: the number worked out when the analysis starts
    box.setCurrentIndex(box.findData(2))
    box.activated.emit(box.currentIndex())
    assert preferences.get("sputter_page")["workers"] == 2
    shutil.rmtree(runner.folders[-1])  # the same request would show the result it has
    page.analyze_button.click()
    _settle(lambda: not page.analyzing() and len(runner.requests) == 2)
    assert runner.requests[-1].workers == 2
    assert runner.requests[0].label() == runner.requests[1].label()  # the processes are not part of the request's name
    _close(page)
    again = _page(tmp_path, client=_WorkersRunner(), preferences=preferences)
    assert again.workers_box.currentData() == 2 and again.run_workers() == 2
    again.workers_box.setCurrentIndex(0)
    again.workers_box.activated.emit(0)
    assert preferences.get("sputter_page")["workers"] == "auto" and again.chosen_workers() is None
    _close(again)
    plain = _page(tmp_path)  # a runner that does not say its number: Automatic alone, the package takes its own
    assert plain.workers_box.itemText(0) == "Automatic" and plain.run_workers() is None
    _close(plain)


def test_the_commands_take_the_processes_of_the_request(tmp_path):
    runner = _runner(tmp_path, _Process())
    data = tmp_path / "data"
    chosen = runner.command(_request(tmp_path / "scan", workers=2), tmp_path / "out", data)
    automatic = runner.command(_request(tmp_path / "scan"), tmp_path / "out", data)
    assert chosen[chosen.index("--workers") + 1] == "2" and automatic[automatic.index("--workers") + 1] == "3"
    static = _static(tmp_path)
    command = static.command(_sample(tmp_path, workers=4), data)
    assert command[command.index("--workers") + 1] == "4"
    assert _sample(tmp_path, workers=4).digest("x") == _sample(tmp_path).digest("x")


# -- the live view while the frames are read -------------------------------------------------------------------------


def test_the_run_adds_up_the_disk_time_of_the_chunks():
    run = LiveRun()
    run.feed(_frames(128))  # an older package: no timing
    assert run.disk_per_frame() is None and run.disk_share() is None and not run.disk_slow()
    run.feed(_frames(256, frames=128, read_s=100.0, wall_s=120.0, mb=800.0))
    run.feed(_frames(384, frames=128, read_s=110.0, wall_s=127.0, mb=800.0))
    assert run.read_frames == 256 and run.read_mb == 1600.0
    assert run.disk_per_frame() == pytest.approx(210.0 / 256) and run.disk_share() == pytest.approx(210.0 / 247)
    assert run.disk_slow()
    fast = LiveRun()
    fast.feed(_frames(128, frames=128, read_s=2.0, wall_s=40.0))
    assert not fast.disk_slow()
    fast.feed(LiveEvent("reading", "intake", {"frames_read": 128, "frames_total": 128, "workers": 6, "seconds": 41.0,
                                              "read_s_per_frame": 0.016, "read_share": 0.05, "mb_read": 400.0,
                                              "slow": True}))
    assert fast.disk_slow() and fast.processes() == 6  # the package's own word counts


def test_the_live_view_says_how_the_frames_are_read_and_when_the_disk_is_slow(tmp_path):
    page = _page(tmp_path)
    page.show()
    tracker = page._client_tracker = ClientProgress()
    tracker.feed("[1/4] intake ...")
    page._live_begin(tmp_path / "out", workers=4)
    page._live_event(_frames(312))  # without the timing fields: the line as before, no note
    page._live_draw()
    assert page.live_steps.detail("intake") == "312 of 1,241 frames read"
    assert not page.live_disk_note.isVisibleTo(page)
    assert page.live_clock.text().endswith(" · Processes: 4")
    page._live_event(_frames(440, frames=128, read_s=105.0, wall_s=123.5, mb=900.0))
    page._live_draw()
    assert page.live_steps.detail("intake") == "Reading 440 of 1,241 frames · disk 0.82 s per frame (85 % of the time)"
    assert page.live_disk_note.isVisibleTo(page) and page.live_disk_note.message_label.text() == SLOW_NOTE
    page._live_end()
    # a fast disk: the seconds per frame are said, nothing else
    page._client_tracker = tracker
    page._live_begin(tmp_path / "out")
    page._live_event(_frames(128, frames=128, read_s=1.28, wall_s=64.0))
    page._live_draw()
    assert page.live_steps.detail("intake") == "Reading 128 of 1,241 frames · disk 0.01 s per frame (2 % of the time)"
    assert not page.live_disk_note.isVisibleTo(page)
    page._live_event(LiveEvent("reading", "intake", {"frames_read": 1241, "frames_total": 1241, "workers": 6,
                                                     "read_s_per_frame": 0.2, "read_share": 0.6, "slow": True}))
    page._live_draw()
    assert page.live_disk_note.isVisibleTo(page)
    page._live_tick()
    assert page.live_clock.text().endswith(" · Processes: 6")
    page._live_end()
    _close(page)


# -- Details ▸ Report: the time per step -----------------------------------------------------------------------------

STEPS = [{"step": "intake", "returncode": 0, "seconds": 161.2, "log": "analyze_intake.log"},
         {"step": "predict", "returncode": 0, "seconds": 302.0, "log": "analyze_predict.log"},
         {"step": "predict (refine)", "returncode": 0, "seconds": 3725.0, "log": "analyze_predict_refine.log"},
         {"step": "report", "returncode": 0, "seconds": 3.4, "log": "analyze_report.log"},
         {"step": "present (main)", "returncode": 0, "seconds": 20.0, "log": "analyze_present_main.log"}]
READING = {"frames_read": 1241, "frames_total": 1241, "workers": 8, "seconds": 160.0, "read_s_per_frame": 0.82,
           "read_share": 0.85, "mb_read": 9930.0, "slow": True}


def _with(out, **keys):
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    analysis.update(keys)
    (out / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    return out


def _plain(html_text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html_text.replace("&nbsp;", " "))).strip()


def test_the_report_shows_the_time_of_every_step_and_how_the_frames_were_read(tmp_path):
    page = _page(tmp_path)
    page.show()
    page.open_client_result(_with(write_result(tmp_path / "new"), steps=STEPS, reading=READING, workers=8))
    page.show_details(True)
    page.tabs.setCurrentIndex(2)  # Report
    words = _plain(page.report_steps.text())
    assert words.startswith("Time per step Step Time (m:ss) Read the frames 2:41 1,241 of 1,241 frames read · disk "
                            "0.82 s per frame (85 % of the time) Find the solutions 5:02")
    assert "Find the solutions (refine) 1:02:05" in words and "Draw the images (main) 0:20" in words
    assert "All steps 1:10:12" in words and words.endswith("Processes: 8")
    assert page.report_disk_note.isVisibleTo(page) and page.report_disk_note.message_label.text() == SLOW_NOTE
    # an older result: the steps and their time, nothing about reading or processes
    page.open_client_result(write_result(tmp_path / "old"))
    words = _plain(page.report_steps.text())
    assert "Read the frames 0:01 Find the solutions 0:01" in words and "All steps 0:04" in words
    assert "frames read" not in words and "Processes" not in words and page.report_disk_note.isHidden()
    # a scan read before: the import row says so; another number of processes for the import is said there
    reused = [{"step": "intake", "returncode": 0, "seconds": 0.0, "log": None, "note": "scan already imported: reused"}]
    page.open_client_result(_with(write_result(tmp_path / "reused"), steps=reused + STEPS[1:2],
                                  reading=dict(READING, slow=False, read_s_per_frame=0.01, read_share=0.1), workers=4))
    words = _plain(page.report_steps.text())
    assert "Read the frames 0:00 Read before for this scan · 1,241 of 1,241 frames read · disk 0.01 s per frame " \
           "(10 % of the time) · Processes: 8" in words and words.endswith("Processes: 4")
    assert page.report_disk_note.isHidden()
    _close(page)


def test_the_new_texts_speak_chinese(tmp_path):
    from src.gimap.app.presentation.i18n import DEFAULT_LANGUAGE, apply_language
    from src.gimap.app.presentation.i18n.zh import ZH
    from src.gimap.features.sputter.presentation import reading_texts, result_tabs, workers_flow
    from src.gimap.features.sputter.presentation.views.sputter_main_view import WORKERS_TIP

    held = [*reading_texts.TEMPLATES, *workers_flow.TEMPLATES, *result_tabs.TEMPLATES, WORKERS_TIP,
            "Parallel processes"]
    assert [template for template in held if template not in ZH] == []
    for template in held:
        assert sorted(set(re.findall(r"{(\w+)}", ZH[template]))) == sorted(set(re.findall(r"{(\w+)}", template)))
    page = _page(tmp_path, client=_WorkersRunner())
    page.show()
    page.open_client_result(_with(write_result(tmp_path / "out"), steps=STEPS[:2], reading=READING, workers=8))
    try:
        apply_language("zh", roots=(page,))
        page.refresh_language()
        words = _plain(page.report_steps.text())
        assert words.startswith("各步用时") and "磁盘每帧 0.82 s（占用时的 85 %）" in words and "进程数：8" in words
        assert page.report_disk_note.message_label.text() == ZH[SLOW_NOTE]
        assert page.workers_box.itemText(0) == "自动（5）"
    finally:
        apply_language(DEFAULT_LANGUAGE, roots=(page,))
        page.refresh_language()
    assert page.workers_box.itemText(0) == "Automatic (5)"
    assert _plain(page.report_steps.text()).startswith("Time per step")
    _close(page)
